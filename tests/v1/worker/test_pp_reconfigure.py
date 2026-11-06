# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect KV/request continuity and transaction recovery at the cheapest level.

The worker owns layer/KV rebinding; its contract is tested on real GPUs by the
acceptance script. These unit tests cover control ordering and failures before
that expensive boundary, plus allocation-aware memory reporting.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.pp_reconfigure.control import (
    ReconfigurationController,
    validate_workers,
)
from vllm.distributed.pp_reconfigure.monitor import memory_metrics, write_event
from vllm.distributed.pp_reconfigure.transport import PPTransport
from vllm.distributed.pp_reconfigure.worker import (
    layer_range,
    overlap_weight_load_and_kv,
)


class Engine:
    def __init__(self, fail=None, mutate=False):
        self.engine_core = self
        self.paused = False
        self.fail = fail
        self.mutate = mutate
        self.calls = []
        self.request: dict[str, Any] = {
            "num_computed_tokens": 17,
            "output_token_ids": [1, 2],
            "block_ids": [[1, 3]],
        }

    async def is_paused(self):
        return self.paused

    async def pause_generation(self, **kwargs):
        assert kwargs == {"mode": "keep", "clear_cache": False}
        self.calls.append("pause")
        self.paused = True

    async def resume_generation(self):
        self.calls.append("resume")
        self.paused = False

    async def call_utility_async(self, method, *args):
        if method == "molink_state":
            return {
                "batch_queue_depth": 0,
                "requests": {"same": deepcopy(self.request)},
            }
        assert method == "molink_migrate"
        action = args[0]
        self.calls.append(action)
        if action == "commit" and self.mutate:
            self.request["num_computed_tokens"] += 1
        if action == "rollback":
            self.request["num_computed_tokens"] = 17
        return [{"stage": i, "ok": action != self.fail} for i in range(2)]


@pytest.mark.asyncio
async def test_pause_preserves_requests_and_phases_are_ordered(tmp_path):
    engine = Engine()
    before = deepcopy(engine.request)
    result = await ReconfigurationController(engine, tmp_path).reconfigure(6)
    assert engine.calls == [
        "describe",
        "pause",
        "prepare",
        "connect",
        "load_and_transfer",
        "commit",
        "finalize",
        "resume",
    ]
    assert engine.request == before
    assert result["status"] == "resumed" and not engine.paused
    assert set(result["timings_ms"]) == set(engine.calls) - {"describe"} | {"validate"}
    for phase in result["timings_ms"]:
        assert (tmp_path / result["run_id"] / f"{phase}.log").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["prepare", "connect", "load_and_transfer", "commit"]
)
async def test_phase_failure_rolls_back_before_resume(tmp_path, failure):
    engine = Engine(fail=failure)
    with pytest.raises(RuntimeError, match="rolled_back_and_resumed"):
        await ReconfigurationController(engine, tmp_path).reconfigure(6)
    assert engine.calls[-2:] == ["rollback", "resume"]
    assert not engine.paused
    assert "finalize" not in engine.calls


@pytest.mark.asyncio
async def test_failure_after_release_keeps_engine_paused(tmp_path):
    engine = Engine(fail="finalize")
    with pytest.raises(RuntimeError):
        await ReconfigurationController(engine, tmp_path).reconfigure(6)
    assert engine.paused and "resume" not in engine.calls
    assert "rollback" not in engine.calls


@pytest.mark.asyncio
async def test_changed_request_state_prevents_release(tmp_path):
    engine = Engine(mutate=True)
    with pytest.raises(RuntimeError, match="request/KV mapping changed"):
        await ReconfigurationController(engine, tmp_path).reconfigure(6)
    assert "finalize" not in engine.calls
    assert engine.request["num_computed_tokens"] == 17


@pytest.mark.parametrize("boundary", [0, 10, -1, 6.0, True, "6"])
def test_reject_invalid_split_before_mutation(boundary):
    with pytest.raises(ValueError):
        layer_range(0, boundary, 10)


def test_layer_movement_is_contiguous_and_nonempty():
    assert layer_range(0, 6, 10) == (0, 6)
    assert layer_range(1, 6, 10) == (6, 10)


def test_worker_mismatch_rejected():
    with pytest.raises(ValueError, match="num_blocks"):
        validate_workers(
            [{"ok": True, "stage": i, "num_blocks": 10 + i} for i in range(2)]
        )


def test_memory_estimate_excludes_unused_preallocated_kv():
    metrics = memory_metrics(900, 1000, 600, 100, 2000)
    assert metrics["active_memory_estimate_bytes"] == 400
    assert metrics["active_memory_estimate_fraction"] == 0.2
    assert metrics["torch_reserved_bytes"] == 1000


def test_lossless_transfer_preserves_strided_kv_blocks():
    source = torch.arange(48, dtype=torch.int16).reshape(4, 3, 4).transpose(1, 2)
    destination = torch.empty_like(source)
    payload = []

    def send(tensor, dst):
        payload.append(tensor.clone())

    def recv(shape, dtype, src):
        return payload.pop(0)

    group = SimpleNamespace(rank_in_group=0, send=send, recv=recv)
    with patch(
        "vllm.distributed.pp_reconfigure.transport.get_pp_group", return_value=group
    ):
        transport = PPTransport()
        assert transport.copy(source, None, 0) == source.numel() * 2
        group.rank_in_group = 1
        transport.copy(None, destination, 0)
    assert torch.equal(source, destination)


def test_phase_logs_are_append_only_jsonl(tmp_path):
    import json

    for event in ("begin", "end"):
        write_event(str(tmp_path), "pause", event=event)
    records = [
        json.loads(line) for line in (tmp_path / "pause.log").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == ["begin", "end"]


def test_weight_load_and_kv_are_concurrent_and_joined(tmp_path):
    """Both tasks must enter before either can complete, without timing guesses."""
    import threading

    entered = threading.Barrier(2)
    completed = []

    def operation(name):
        entered.wait(timeout=5)
        completed.append(name)
        return {"operation": name}

    result = overlap_weight_load_and_kv(
        lambda: operation("weight_load"), lambda: operation("kv"), 1, str(tmp_path)
    )
    assert result["ok"] and set(completed) == {"weight_load", "kv"}
    assert result["overlap_ms"] > 0
    assert (tmp_path / "weight_load.log").exists()
    assert (tmp_path / "kv.log").exists()


@pytest.mark.parametrize("failed", ["weight_load", "kv"])
def test_failure_waits_for_both_parallel_tasks_before_rollback(failed):
    import threading

    entered = threading.Barrier(2)
    finished = []

    def operation(name):
        entered.wait(timeout=5)
        finished.append(name)
        if name == failed:
            raise ValueError("stage failed")
        return {}

    result = overlap_weight_load_and_kv(
        lambda: operation("weight_load"), lambda: operation("kv"), 1
    )
    assert not result["ok"] and "stage failed" in result["error"]
    assert set(finished) == {"weight_load", "kv"}


@pytest.mark.parametrize("missing", [None, "model.layers.6.self_attn.k_proj.weight"])
def test_local_loading_maps_qkv_and_only_reads_requested_layer(tmp_path, missing):
    """The vLLM mapper packs local Q/K/V into the staged layer; missing shards fail."""
    from safetensors.torch import save_file
    from torch import nn

    from vllm.config.load import LoadConfig
    from vllm.distributed.pp_reconfigure import weights

    layer = nn.Module()
    layer.add_module("self_attn", nn.Module())
    layer.self_attn.add_module("qkv_proj", nn.Module())
    param = nn.Parameter(torch.zeros(6, 2), requires_grad=False)

    def weight_loader(param, tensor):
        offset = {"q": 0, "k": 2, "v": 4}[tensor.shard_id]
        param.data[offset : offset + 2].copy_(tensor)

    param.weight_loader = weight_loader
    layer.self_attn.qkv_proj.register_parameter("weight", param)
    tensors = {
        f"model.layers.6.self_attn.{name}_proj.weight": torch.full((2, 2), float(i))
        for i, name in enumerate(("q", "k", "v"), 1)
    }
    tensors["model.layers.0.unrelated.weight"] = torch.zeros(1)
    if missing:
        tensors.pop(missing)
    save_file(tensors, tmp_path / "model.safetensors")
    config = SimpleNamespace(
        load_config=LoadConfig(load_format="safetensors"),
        model_config=SimpleNamespace(model=str(tmp_path), revision=None),
    )
    with patch.object(weights, "process_weights_after_loading"):
        if missing:
            with pytest.raises(ValueError, match="missing layer weights"):
                weights.load_local_layers(config, {6: layer}, torch.device("cpu"))
        else:
            result = weights.load_local_layers(config, {6: layer}, torch.device("cpu"))
            assert result["layers"] == [6]
            assert result["loaded_bytes"] == 48
            assert torch.equal(
                param, torch.tensor([[1.0] * 2] * 2 + [[2.0] * 2] * 2 + [[3.0] * 2] * 2)
            )


def test_resource_frame_roundtrip_and_identity_validation():
    from vllm.distributed.pp_reconfigure.resource_monitor import (
        pack_report,
        unpack_report,
    )

    report = {"stage": 1, "pid": 123, "sequence": 1, "gpu_utilization_percent": 23}
    assert unpack_report(pack_report(report)) == report
    for invalid in ({**report, "stage": 2}, {**report, "sequence": True}):
        with pytest.raises(ValueError):
            unpack_report(pack_report(invalid))
    with pytest.raises(ValueError, match="length"):
        unpack_report(torch.zeros(16384, dtype=torch.uint8))


def test_resource_monitor_records_existing_pp_group_collective(tmp_path):
    """Check framing and receipt logging; the real-GPU test verifies NCCL delivery."""
    import json
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm.distributed.pp_reconfigure.resource_monitor import (
        ResourceMonitor,
        pack_report,
    )

    monitor = ResourceMonitor(
        0, lambda: {"gpu_utilization_percent": 10}, log_dir=str(tmp_path)
    )
    monitor.sample_once()
    peer = {"stage": 1, "pid": 123, "sequence": 1, "gpu_utilization_percent": 25}
    group = SimpleNamespace(world_size=2, rank_in_group=0, unique_name="pp:test")
    group.all_gather = Mock(
        side_effect=lambda frame, dim: torch.cat((frame, pack_report(peer)))
    )
    with patch("torch.cuda.device", return_value=nullcontext()):
        result = monitor.publish(group, torch.device("cpu"))
    group.all_gather.assert_called_once()
    assert result["master_stage"] == 0 and result["transport"] == "nccl"
    assert result["workers"][1]["metrics"] == peer
    assert result["reports_received"][1] == 1
    received = json.loads((tmp_path / "master-received.log").read_text())
    assert received["from_pid"] == 123
    assert received["transport"] == "nccl" and received["group"] == "pp:test"


def test_resource_tick_dispatches_periodically_without_requests():
    from unittest.mock import Mock

    from vllm.v1.engine.core import EngineCore

    core = EngineCore.__new__(EngineCore)
    core._molink_resource_interval = 1.0
    core._molink_resource_next = 0.0
    core.collective_rpc = Mock()
    with patch("vllm.v1.engine.core.time.monotonic", return_value=100):
        core._molink_maybe_publish_resources()
    with patch("vllm.v1.engine.core.time.monotonic", return_value=100.5):
        core._molink_maybe_publish_resources()
    with patch("vllm.v1.engine.core.time.monotonic", return_value=101.1):
        core._molink_maybe_publish_resources()
    assert core.collective_rpc.call_count == 2
    core.collective_rpc.assert_called_with("molink_publish_resources")


def test_resource_tick_wakes_idle_engine_input_poll():
    import queue
    from unittest.mock import Mock

    from vllm.v1.engine.core import EngineCoreProc

    core = EngineCoreProc.__new__(EngineCoreProc)
    core.has_work = lambda: False
    core.is_running = lambda: True
    core._notify_idle_state_callbacks = Mock()
    core.input_queue = Mock()
    core.input_queue.get.side_effect = queue.Empty
    core.aborts_queue = queue.Queue()
    core.process_input_queue_block = True
    core._molink_resource_interval = 1.0
    core._molink_resource_next = 101.0
    with patch("vllm.v1.engine.core.time.monotonic", return_value=100):
        core._process_input_queue()
    core.input_queue.get.assert_called_once_with(block=True, timeout=1.0)


def test_resource_monitor_deduplicates_shared_kv_and_handles_resume():
    from vllm.distributed.pp_reconfigure.monitor import StageMonitor

    monitor = StageMonitor()
    cached = SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set())
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[
            SimpleNamespace(req_id="a", block_ids=([0, 1, 2],)),
            SimpleNamespace(req_id="b", block_ids=([1, 3],)),
        ],
        scheduled_cached_reqs=cached,
    )
    monitor.update_requests(output)
    assert set().union(*monitor.request_blocks.values()) - {0} == {1, 2, 3}
    output.scheduled_new_reqs = []
    output.preempted_req_ids = {"a"}
    monitor.update_requests(output)
    assert set(monitor.request_blocks) == {"b"}
    cached.req_ids, cached.new_block_ids = ["b"], [([4, 5],)]
    cached.resumed_req_ids = {"b"}
    monitor.update_requests(output)
    assert monitor.request_blocks["b"] == {4, 5}
    output.finished_req_ids = {"b"}
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=[], new_block_ids=[], resumed_req_ids=set()
    )
    monitor.update_requests(output)
    assert monitor.request_blocks == {}


def test_idle_status_requests_do_not_starve_periodic_resource_sync():
    import queue
    from unittest.mock import Mock

    from vllm.v1.engine.core import EngineCoreProc

    core = EngineCoreProc.__new__(EngineCoreProc)
    core.has_work = lambda: False
    core.is_running = lambda: True
    core._notify_idle_state_callbacks = Mock()
    core._molink_maybe_publish_resources = Mock()
    core._handle_client_request = Mock()
    core.input_queue = Mock()
    core.input_queue.get.side_effect = [("status", None), queue.Empty]
    core.aborts_queue = queue.Queue()
    core.process_input_queue_block = True
    core._molink_resource_interval = 0.0
    core._process_input_queue()
    assert core._molink_maybe_publish_resources.call_count == 2
    core._handle_client_request.assert_called_once_with("status", None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("query_len", [1, 3])
def test_stacked_addresses_match_dense_attention_after_resize(query_len):
    """Noncontiguous K/V writes and attention survive pointer-only compaction."""
    from vllm.distributed.pp_reconfigure.cache import StackedKVCache
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    device = torch.device("cuda:0")
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(2)]
    pool = StackedKVCache(names, spec, 5, device)
    name = names[1]
    live_address = pool.handles[name][3].item()
    pool.remove_blocks([1, 4])
    pool.add_blocks([1, 4])
    assert pool.handles[name][3].item() == live_address
    torch.manual_seed(42)
    key = torch.randn(32, 2, 64, device=device, dtype=spec.dtype)
    value = torch.randn_like(key)
    slots = torch.cat((torch.arange(32, 48), torch.arange(48, 64))).to(device)
    dense = torch.zeros(5, 2, 16, 128, device=device, dtype=spec.dtype)
    dense_k, dense_v = dense.transpose(1, 2).split(64, dim=-1)
    scale = torch.tensor(1.0, device=device)
    triton_reshape_and_cache_flash(
        key, value, dense_k, dense_v, slots, "auto", scale, scale
    )
    k, v = pool.key_value(name)
    triton_reshape_and_cache_flash(
        key,
        value,
        k,
        v,
        slots,
        "auto",
        scale,
        scale,
        block_addresses=pool.handles[name],
    )
    for block_id in (2, 3):
        torch.testing.assert_close(
            pool.block(name, block_id), dense[block_id], rtol=0, atol=0
        )
    q = torch.randn(query_len, 4, 64, device=device, dtype=spec.dtype)
    block_table = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
    expected, actual = torch.empty_like(q), torch.empty_like(q)
    kwargs = dict(
        q=q,
        cu_seqlens_q=torch.tensor([0, query_len], device=device, dtype=torch.int32),
        seqused_k=torch.tensor([32], device=device, dtype=torch.int32),
        max_seqlen_q=query_len,
        max_seqlen_k=32,
        softmax_scale=0.125,
        causal=True,
        window_size=(-1, -1),
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    unified_attention(
        k=dense_k, v=dense_v, out=expected, block_table=block_table, **kwargs
    )
    unified_attention(
        k=k,
        v=v,
        out=actual,
        block_table=pool.handles[name][block_table.long()],
        block_table_is_address=True,
        **kwargs,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dirty_drain_keeps_later_writes_and_coalesces_overwrites():
    from vllm.distributed.pp_reconfigure.patching import DirtyBitmap

    bitmap = DirtyBitmap(64)
    bitmap.mark([3, 7, 3])
    slots, watermark = bitmap.drain()
    assert slots == [3, 7] and watermark == 3
    bitmap.mark([3, 8])
    assert bitmap.drain() == ([3, 8], 5)
    assert bitmap.drain() == ([], 5)
    with pytest.raises(ValueError):
        bitmap.mark([64])


@pytest.mark.parametrize("copy_block", [None, 2])
def test_inference_marks_optional_block_copies_without_advancing_token_count(
    copy_block,
):
    from vllm.distributed.pp_reconfigure.live import LiveLayerMigration
    from vllm.distributed.pp_reconfigure.patching import DirtyBitmap
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    live = LiveLayerMigration.__new__(LiveLayerMigration)
    live.stage, live.step_count = 0, 0
    live.log_dir = None
    live.bitmap = DirtyBitmap(64)
    handle = torch.empty(0)
    handle._pipelive = (
        SimpleNamespace(
            last_slots=torch.tensor([1, 3, -1]), spec=SimpleNamespace(block_size=16)
        ),
        "layer",
    )
    live.prepared = SimpleNamespace(
        source=0, moving=[0], old_kv={"model.layers.0.self_attn.attn": handle}
    )
    copies = None if copy_block is None else [KVCacheBlockCopy(1, copy_block)]
    output = SimpleNamespace(total_num_scheduled_tokens=2, kv_cache_block_copies=copies)
    with patch("torch.cuda.Event"):
        live.after_step(output)
    expected = [1, 3] + (list(range(32, 48)) if copy_block is not None else [])
    assert live.bitmap.drain() == (expected, 2)


@pytest.mark.parametrize("in_flight", ["host", "cuda"])
def test_receiver_rejects_migration_until_inference_completes(in_flight):
    import threading
    from unittest.mock import Mock

    from vllm.distributed.pp_reconfigure.communication import MigrationChannel

    channel = MigrationChannel.__new__(MigrationChannel)
    channel.mutex = threading.Lock()
    channel.group = SimpleNamespace(pipelive_pending=Mock(query=lambda: False))
    if in_flight == "host":
        channel.mutex.acquire()
    channel.receive_control = lambda: {"op": "ACK"}
    channel.send_control = Mock()
    apply = Mock()
    assert channel.receive(apply)
    channel.send_control.assert_called_once_with({"op": "REJECT"})
    apply.assert_not_called()
    assert channel.mutex.locked() == (in_flight == "host")
    if in_flight == "host":
        channel.mutex.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_patches_preserve_unwritten_slots_and_stack_release():
    from vllm.distributed.pp_reconfigure.cache import StackedKVCache
    from vllm.distributed.pp_reconfigure.patching import copy_slots
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(4)]
    source = StackedKVCache(names, spec, 4, torch.device("cuda:0"))
    target = StackedKVCache(names, spec, 4, torch.device("cuda:0"))
    name = names[2]
    source.block(name, 2).fill_(42)
    source.block(name, 3).fill_(17)
    payload = copy_slots(source.handles[name], [32, 49, 47])
    copy_slots(target.handles[name], [32, 49, 47], payload)
    assert torch.all(target.block(name, 2)[:, [0, 15]] == 42)
    assert torch.all(target.block(name, 2)[:, 1:15] == 0)
    assert torch.all(target.block(name, 3)[:, 1] == 17)
    copy_kv_cache_blocks_inplace(source.handles.values(), 4, [(2, 1)])
    torch.testing.assert_close(source.block(name, 1), source.block(name, 2))
    address = source.handles[name].clone()
    before = source.allocated_bytes
    source.retire_layers(names[:2])
    assert source.allocated_bytes == before // 2
    source.add_blocks([4])
    torch.testing.assert_close(source.handles[name][:4], address)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [None, "prepare_live", "start_live", "progress_live", "sync_live", "commit"],
)
async def test_async_streams_before_pause_and_syncs_before_commit(tmp_path, failure):
    class LiveEngine(Engine):
        async def call_utility_async(self, method, *args):
            workers = await super().call_utility_async(method, *args)
            if method == "molink_state":
                return workers
            action = args[0]
            for worker in workers:
                worker.update(
                    pipelive=True,
                    source=0,
                    max_intermediate_blocks=64,
                    max_final_blocks=64,
                    num_blocks=64,
                )
                if action == "prepared_live":
                    worker["prepared"] = True
                if action == "progress_live":
                    assert not self.paused
                    worker.update(
                        full_done=True, weights_ready=True, scheduled=9, applied=9
                    )
                if action == "sync_live":
                    assert self.paused
                    worker.update(
                        compute_intervals_ms=[(1, 5)],
                        communication_intervals_ms=[(2, 4)],
                        bytes=100,
                        steps=3,
                        scheduled=9,
                        applied=9,
                        dirty_slots=0,
                        kv_ready=True,
                    )
            return workers

    engine = LiveEngine(fail=failure)
    if failure:
        with pytest.raises(RuntimeError, match="rolled_back_and_resumed"):
            await ReconfigurationController(engine, tmp_path).reconfigure(6, "async")
        assert engine.calls[-2:] == ["rollback", "resume"]
        return
    result = await ReconfigurationController(engine, tmp_path).reconfigure(6, "async")
    assert engine.calls == [
        "describe",
        "prepare_live",
        "prepared_live",
        "start_live",
        "progress_live",
        "pause",
        "sync_live",
        "commit",
        "finalize",
        "resume",
    ]
    assert result["metrics"]["overlap_fraction"] == 1
    assert result["metrics"]["inference_steps_during_migration"] == 3


def test_resize_keeps_live_ids_and_updates_free_queue_capacity():
    from vllm.distributed.pp_reconfigure.resize import commit_pool_resize, plan_resize
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(8, enable_caching=True, hash_block_size=16)
    live = pool.get_new_blocks(3)
    ids = [block.block_id for block in live]
    plan = plan_resize(pool, 4)
    assert plan["remove"] == [7, 6, 5, 4]
    commit_pool_resize(pool, plan)
    assert pool.num_gpu_blocks == 4 and pool.get_num_free_blocks() == 0
    assert [block.block_id for block in live] == ids
    with pytest.raises(ValueError, match="active"):
        plan_resize(pool, 3)
    plan = plan_resize(pool, 10)
    commit_pool_resize(pool, plan)
    assert pool.num_gpu_blocks == 10 and pool.get_num_free_blocks() == 6
    assert {b.block_id for b in pool.get_new_blocks(6)} == {4, 5, 6, 7, 8, 9}


@pytest.mark.parametrize("failure", [None, "prepare", "commit"])
def test_resize_commits_scheduler_only_after_all_workers_prepare(failure):
    from unittest.mock import Mock

    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.engine.core import EngineCore

    pool = BlockPool(8, enable_caching=True, hash_block_size=16)
    pool.get_new_blocks(2)
    manager = SimpleNamespace(
        block_pool=pool,
        kv_cache_config=SimpleNamespace(num_blocks=8),
        watermark_blocks=2,
    )
    core = EngineCore.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        kv_cache_manager=manager,
        scheduler_config=SimpleNamespace(watermark=0.25),
        cache_config=SimpleNamespace(num_gpu_blocks=8),
        set_pause_state=Mock(),
    )

    def rpc(method, args):
        return [
            {
                "stage": i,
                "ok": not (failure and args[0] == f"{failure}_resize_live" and i == 1),
            }
            for i in range(2)
        ]

    core.collective_rpc = Mock(side_effect=rpc)
    with patch.dict("os.environ", {"HETVLLM_PIPELIVE": "1"}):
        if failure:
            with pytest.raises(RuntimeError):
                core.molink_resize_kv(4)
            if failure == "prepare":
                assert pool.num_gpu_blocks == 8
                assert (
                    core.collective_rpc.call_args.kwargs["args"][0]
                    == "rollback_resize_live"
                )
            else:
                from vllm.v1.core.sched.interface import PauseState

                assert pool.num_gpu_blocks == 4
                core.scheduler.set_pause_state.assert_called_once_with(
                    PauseState.PAUSED_ALL
                )
        else:
            core.molink_resize_kv(4)
            assert pool.num_gpu_blocks == 4 and manager.watermark_blocks == 1


@pytest.mark.parametrize("error", [ValueError("internal"), RuntimeError("internal")])
def test_internal_worker_errors_are_not_validation_conflicts(error):
    from vllm.distributed.pp_reconfigure.worker import migration_rpc

    migration = SimpleNamespace(describe=lambda boundary: (_ for _ in ()).throw(error))
    worker = SimpleNamespace(_molink_migration=migration)
    with patch(
        "vllm.distributed.pp_reconfigure.worker.get_pp_group",
        return_value=SimpleNamespace(rank_in_group=0),
    ):
        result = migration_rpc(worker, "describe", 12)
    assert result["error_category"] == "internal"
    with pytest.raises(RuntimeError):
        validate_workers([result, {"ok": True, "stage": 1}])


@pytest.mark.parametrize("validation", [True, False])
def test_utility_rpc_preserves_only_explicit_validation_errors(validation):
    from concurrent.futures import Future

    import msgspec

    from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError
    from vllm.v1.engine import UtilityOutput
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.engine.core_client import _process_utility_output

    error_type = ReconfigurationValidationError if validation else ValueError

    def fail():
        raise error_type("rejected")

    outputs = []
    EngineCoreProc._invoke_utility_method(
        "molink_resize_kv", fail, UtilityOutput(7), outputs.append
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(outputs[0]), type=UtilityOutput
    )
    future = Future()
    _process_utility_output(decoded, {7: future})
    assert isinstance(future.exception(), ReconfigurationValidationError) == validation


@pytest.mark.asyncio
async def test_validation_failure_keeps_conflict_type_and_does_not_pause(tmp_path):
    from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError

    class InvalidEngine(Engine):
        async def call_utility_async(self, method, *args):
            return [
                {
                    "stage": i,
                    "ok": False,
                    "error_category": "validation",
                    "error": "unaligned boundary",
                }
                for i in range(2)
            ]

    engine = InvalidEngine()
    with pytest.raises(ReconfigurationValidationError, match="unchanged"):
        await ReconfigurationController(engine, tmp_path).reconfigure(13)
    assert not engine.paused


def test_finalize_collects_old_cyclic_weights_before_return():
    import gc
    import weakref

    from vllm.distributed.pp_reconfigure.live import LiveLayerMigration

    class OldLayer:
        pass

    old = OldLayer()
    old.weight_loader = old  # Real parameters hold bound methods back to the layer.
    reference = weakref.ref(old)
    migration = LiveLayerMigration.__new__(LiveLayerMigration)
    migration.stage = 1
    migration.prepared = SimpleNamespace(
        committed=True,
        source=0,
        old_layers=[old],
        transport=SimpleNamespace(close=lambda: None),
    )
    gc.freeze()
    del old
    try:
        with patch("torch.accelerator.empty_cache"):
            migration.finalize()
        assert reference() is None
    finally:
        gc.unfreeze()


@pytest.mark.parametrize("pending", ["dirty", "unapplied", "not_ready"])
@pytest.mark.asyncio
async def test_pending_kv_updates_prevent_commit(tmp_path, pending):
    class PendingEngine(Engine):
        async def call_utility_async(self, method, *args):
            workers = await super().call_utility_async(method, *args)
            if method == "molink_state":
                return workers
            for w in workers:
                w.update(
                    pipelive=True,
                    source=0,
                    max_intermediate_blocks=64,
                    max_final_blocks=64,
                    num_blocks=64,
                )
                if args[0] == "load_and_transfer":
                    w.update(
                        kv_ready=pending != "not_ready",
                        dirty_slots=0,
                        scheduled=9,
                        applied=9,
                    )
                    if w["stage"] == 0 and pending == "dirty":
                        w["dirty_slots"] = 1
                    if w["stage"] == 1 and pending == "unapplied":
                        w["applied"] = 8
            return workers

    engine = PendingEngine()
    with pytest.raises(RuntimeError, match="required KV updates remain"):
        await ReconfigurationController(engine, tmp_path).reconfigure(12)
    assert "commit" not in engine.calls
    assert engine.calls[-2:] == ["rollback", "resume"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_incoming_sparse_cache_allocates_only_retained_blocks():
    from vllm.distributed.pp_reconfigure.cache import StackedKVCache
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(2)]
    pool = StackedKVCache(names, spec, 0, torch.device("cuda:0"))
    pool.add_blocks([0, 63])
    assert set(pool.stacks) == {0, 63}
    assert pool.allocated_bytes == 2 * pool.allocation_bytes
    assert pool.handles[names[0]].numel() == 64
    assert not pool.handles[names[0]][1:63].any()


@pytest.mark.asyncio
async def test_insufficient_migration_budget_rejected_before_mutation(tmp_path):
    from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError

    class NoCapacityEngine(Engine):
        async def call_utility_async(self, method, *args):
            workers = await super().call_utility_async(method, *args)
            for w in workers:
                w.update(pipelive=True, max_intermediate_blocks=0, max_final_blocks=0)
            return workers

    engine = NoCapacityEngine()
    with pytest.raises(ReconfigurationValidationError, match="insufficient memory"):
        await ReconfigurationController(engine, tmp_path).reconfigure(12)
    assert engine.calls == ["describe"]
    assert not engine.paused


@pytest.mark.parametrize("kind, status", [("validation", 409), ("internal", 500)])
@pytest.mark.parametrize(
    "endpoint, payload",
    [
        ("/molink/reconfigure", {"boundary": 13}),
        ("/molink/resize_kv", {"num_blocks": 1}),
    ],
)
def test_http_conflicts_do_not_hide_internal_errors(kind, status, endpoint, payload):
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError
    from vllm.entrypoints.serve.dev.molink import api_router

    async def fail(*args):
        if kind == "validation":
            raise ReconfigurationValidationError("invalid request")
        raise ValueError("internal bug")

    control = SimpleNamespace(
        lock=asyncio.Lock(),
        reconfigure=fail,
        engine=SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=fail)),
    )
    app = FastAPI()
    app.include_router(api_router.router)
    with (
        patch.object(api_router, "controller", return_value=control),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        assert client.post(endpoint, json=payload).status_code == status
