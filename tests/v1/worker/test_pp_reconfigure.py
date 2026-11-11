# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect KV/request continuity and transaction recovery at the cheapest level.

The worker owns layer/KV rebinding; its contract is tested on real GPUs by the
migration benchmark. These unit tests cover control ordering and failures before
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

    async def pause_scheduler_async(self, **kwargs):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("factor", [1, 2, 4, 8])
@pytest.mark.parametrize("online_reconfiguration", [False, True])
def test_initial_kv_plan_respects_physical_stacked_budget(
    monkeypatch, block_size, factor, online_reconfiguration
):
    """Automatic planning must fit the allocator's real storage, including padding."""
    from vllm.distributed.pp_reconfigure.cache import StackedKVCache
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        KVCacheLayout,
    )

    monkeypatch.setenv(
        "VLLM_PP_RECONFIGURE_ONLINE", "1" if online_reconfiguration else "0"
    )
    monkeypatch.setenv("VLLM_PP_RECONFIGURE_STACK_SIZE", str(factor))
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=32, head_size=128, dtype=torch.float16
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(8)]
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=1,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        )
    )
    budget = 32 << 20
    planned = get_kv_cache_config_from_groups(
        config, [KVCacheGroupSpec(names, spec)], budget
    )
    if not online_reconfiguration:
        assert planned.num_blocks == (16 if block_size == 16 else 4)
        return
    pool = StackedKVCache(
        names, spec, planned.num_blocks, torch.device("cuda:0"), factor
    )
    assert 0 < pool.allocated_bytes <= budget
    assert planned.kv_cache_tensors[0].size == pool.allocated_bytes


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
    handle._pp_reconfigure = (
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
    channel.group = SimpleNamespace(pp_reconfigure_pending=Mock(query=lambda: False))
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


def test_channel_accounts_for_rejection_and_each_sender_wait():
    import threading
    from collections import defaultdict
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm.distributed.pp_reconfigure.communication import MigrationChannel

    channel = MigrationChannel.__new__(MigrationChannel)
    channel.mutex = threading.RLock()
    channel.timings_ms = defaultdict(float)
    channel.group = SimpleNamespace(rank_in_group=0, pp_reconfigure_pending=Mock())
    channel.stream, channel.comm = Mock(), Mock()
    channel.send_control = Mock()
    channel.receive_control = Mock(
        side_effect=[{"op": op} for op in ("REJECT", "ACCEPT", "APPLIED")]
    )
    channel.rejections = 0
    with (
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch("torch.cuda.Event"),
        patch(
            "vllm.distributed.pp_reconfigure.communication.time.perf_counter",
            side_effect=range(12),
        ),
        patch("vllm.distributed.pp_reconfigure.communication.time.sleep"),
    ):
        channel.send(torch.ones(1), {"version": 1})
    assert channel.rejections == 1
    assert dict(channel.timings_ms) == {
        "communication_lock_wait": 2000,
        "inference_nccl_wait": 2000,
        "handshake": 2000,
        "nccl_enqueue_and_wait": 1000,
        "applied_ack_wait": 1000,
        "retry_backoff": 1000,
    }
    channel.comm.send.assert_called_once()


def test_inference_metadata_wait_does_not_reserve_migration_nccl(monkeypatch):
    """Reverse migration may enter while the upstream stage is still computing."""
    import threading
    from unittest.mock import Mock

    from vllm.distributed import pp_link
    from vllm.distributed.parallel_state import GroupCoordinator

    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size = 2
    group.rank_in_group = 1
    group.use_cpu_custom_send_recv = False
    group.device_group = group.cpu_group = None
    group.device = torch.device("cuda:0")
    group.pp_reconfigure_mutex = threading.Lock()
    entered = []

    def metadata(src):
        assert src == 0
        assert group.pp_reconfigure_mutex.acquire(blocking=False)
        group.pp_reconfigure_mutex.release()
        entered.append(True)
        return [("marker", 42)]

    group.recv_object = metadata
    monkeypatch.setattr(pp_link, "is_link_receiver", lambda rank: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    event = object()

    def record_event():
        assert group.pp_reconfigure_mutex.locked()
        return event

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: Mock(record_event=record_event)
    )
    assert group.recv_tensor_dict() == {"marker": 42}
    assert entered == [True] and group.pp_reconfigure_pending is event
    assert not group.pp_reconfigure_mutex.locked()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slots", [[], [1, 17, 31]])
def test_snapshot_batches_layers_and_acknowledges_after_all_are_applied(slots):
    """One handshake carries every layer, preserving slot and watermark semantics."""
    import threading
    from collections import defaultdict

    from vllm.distributed.pp_reconfigure.cache import StackedKVCache
    from vllm.distributed.pp_reconfigure.live import LiveLayerMigration
    from vllm.distributed.pp_reconfigure.patching import PatchLifecycle, copy_slots
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(8)]
    source = StackedKVCache(names, spec, 2, torch.device("cuda:0"), 4)
    target = StackedKVCache(names, spec, 2, torch.device("cuda:0"), 4)
    for index, name in enumerate(names):
        for block in (0, 1):
            source.block(name, block).fill_(index + 1)
    live = LiveLayerMigration.__new__(LiveLayerMigration)
    live.prepared = SimpleNamespace(
        moving=list(range(8)),
        old_kv=source.handles,
        old_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
        ),
    )
    live.cache_lock = threading.RLock()
    live.last_write = torch.cuda.current_stream().record_event()
    live.log_dir = None
    live.communication_intervals, live.patch_intervals = [], []
    live.bytes_sent = live.patch_rounds = live.applied = 0
    live.transferred_slots = set()
    live.lifecycle = PatchLifecycle([0, 1])
    live.send_version = 0
    live.full_done = False
    received = []

    def send(payload, metadata):
        live.channel.stream.synchronize()
        assert live.applied == 0
        assert payload.nbytes <= 64 << 20
        assert metadata["last"] and metadata["watermark"] == 24
        assert metadata["version"] == 1
        for name, layer_payload in zip(metadata["layers"], payload, strict=True):
            copy_slots(target.handles[name], metadata["slots"], layer_payload)
        torch.cuda.current_stream().synchronize()
        received.append(metadata)
        return None, None

    live.channel = SimpleNamespace(
        stream=torch.cuda.Stream(), send=send, timings_ms=defaultdict(float)
    )
    live._send_snapshot(slots, 24)
    assert len(received) == 1 and live.applied == 24
    assert live.transferred_slots == set(slots)
    assert live.patch_rounds == bool(slots)
    for index, name in enumerate(names):
        expected = torch.zeros_like(target.block(name, 0))
        if slots:
            expected[:, 1] = index + 1
        torch.testing.assert_close(target.block(name, 0), expected)
        expected.zero_()
        if slots:
            expected[:, [1, 15]] = index + 1
        torch.testing.assert_close(target.block(name, 1), expected)


def test_history_manifest_evicts_only_unreferenced_prefixes():
    from vllm.distributed.pp_reconfigure.resize import required_migration_blocks
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(7, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(4)
    live, shared, idle, alias = blocks
    shared.ref_cnt += 1
    for block, key in zip(blocks, (b"live", b"shared", b"idle", b"alias")):
        block.set_block_hash(key)
        pool.cached_block_hash_to_block.insert(key, block)
    pool.free_blocks([idle, alias])
    alias_key = b"alias-extra"
    pool.cached_block_hash_to_block.insert(alias_key, alias)
    pool.cached_block_hashes_by_block[alias.block_id] = {alias_key}
    free_before = pool.get_num_free_blocks()
    generations = [b.allocation_generation for b in pool.blocks]
    assert required_migration_blocks(pool) == [live.block_id, shared.block_id]
    assert pool.cached_block_hash_to_block.get_one_block(b"live") is live
    assert pool.cached_block_hash_to_block.get_one_block(b"shared") is shared
    for key in (b"idle", b"alias", alias_key):
        assert pool.cached_block_hash_to_block.get_one_block(key) is None
    assert idle.block_hash is None and alias.block_hash is None
    assert pool.get_num_free_blocks() == free_before
    assert [b.allocation_generation for b in pool.blocks] == generations


def test_cpu_weight_cache_bounds_resident_tensors_and_falls_back_to_disk():
    from unittest.mock import Mock

    from vllm.distributed.pp_reconfigure.weights import CPUWeightCache

    cache = CPUWeightCache(8)
    cache.get("a", lambda: torch.tensor([1.0]))
    cache.get("b", lambda: torch.tensor([2.0]))
    unused = Mock(side_effect=AssertionError("cache hit must not read disk"))
    assert cache.get("a", unused).item() == 1
    cache.get("c", lambda: torch.tensor([3.0]))
    assert list(cache.tensors) == ["a", "c"]
    assert cache.metrics()["resident_bytes"] == 8
    assert cache.metrics()["evicted_bytes"] == 4
    oversized = torch.arange(3, dtype=torch.float32)
    assert cache.get("large", lambda: oversized) is oversized
    assert cache.metrics()["resident_bytes"] == 8 and "large" not in cache.tensors
    disabled = CPUWeightCache(0)
    loader = Mock(return_value=torch.tensor([4.0]))
    disabled.get("a", loader)
    disabled.get("a", loader)
    assert loader.call_count == 2 and disabled.metrics()["resident_bytes"] == 0
    with pytest.raises(ValueError, match="nonnegative"):
        CPUWeightCache(-1)


def test_cpu_weight_prefetch_selects_target_layers_within_budget(tmp_path):
    from safetensors.torch import save_file

    from vllm.distributed.pp_reconfigure.weights import CPUWeightCache

    path = tmp_path / "weights.safetensors"
    tensors = {
        f"model.layers.{i}.weight": torch.tensor([float(i)]) for i in (12, 13, 14)
    }
    save_file(tensors, path)
    cache = CPUWeightCache(8)
    with patch(
        "vllm.distributed.pp_reconfigure.weights.local_weight_files",
        return_value=[str(path)],
    ):
        cache.prefetch(None, [12, 13, 14])
        cache.thread.join(5)
    assert not cache.thread.is_alive() and cache.prefetch_error is None
    assert set(cache.tensors) == {"model.layers.12.weight", "model.layers.13.weight"}
    assert cache.metrics()["resident_bytes"] == 8
    torch.testing.assert_close(
        cache.tensors["model.layers.12.weight"], tensors["model.layers.12.weight"]
    )


def test_prefetch_error_preserves_foreground_disk_error():
    from unittest.mock import Mock

    from vllm.distributed.pp_reconfigure.weights import CPUWeightCache

    cache = CPUWeightCache(8)
    with patch(
        "vllm.distributed.pp_reconfigure.weights.local_weight_files",
        side_effect=OSError("offline"),
    ):
        cache.prefetch(None, [12])
        cache.thread.join(5)
    assert "offline" in cache.prefetch_error
    with pytest.raises(OSError, match="offline"):
        cache.get("missing", Mock(side_effect=OSError("offline")))


def test_block_reallocation_advances_generation_and_patch_replay_is_rejected():
    from vllm.distributed.pp_reconfigure.patching import PatchLifecycle
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(3, enable_caching=True, hash_block_size=16)
    block = pool.get_new_blocks(1)[0]
    assert block.allocation_generation == 1
    pool.free_blocks([block])
    assert block in pool.get_new_blocks(2)
    assert block.allocation_generation == 2
    lifecycle = PatchLifecycle([0, 2, 1])
    with lifecycle.lock:
        assert lifecycle.accept(1, [(1, 1), (2, 1)]) == {2}
        assert lifecycle.accept(2, [(1, 2)]) == {1}
        for version in (1, 2, 4):
            with pytest.raises(ValueError, match="replayed"):
                lifecycle.accept(version, [(1, 2)])


def test_unchanged_generations_do_not_wait_for_inflight_patch():
    import threading

    from vllm.distributed.pp_reconfigure.patching import PatchLifecycle

    lifecycle = PatchLifecycle([0, 1])
    finished = threading.Event()
    observed = []

    def observe():
        observed.append(lifecycle.observe([0, 1]))
        finished.set()

    with lifecycle.lock:
        thread = threading.Thread(target=observe)
        thread.start()
        completed_while_patch_locked = finished.wait(1)
    thread.join(5)
    assert not thread.is_alive()
    assert completed_while_patch_locked
    assert observed == [[]]


def test_reused_generation_waits_for_inflight_patch():
    import threading

    from vllm.distributed.pp_reconfigure.patching import PatchLifecycle

    lifecycle = PatchLifecycle([0, 1])
    started = threading.Event()
    finished = threading.Event()
    observed = []

    def observe():
        started.set()
        observed.append(lifecycle.observe([0, 2]))
        finished.set()

    with lifecycle.lock:
        thread = threading.Thread(target=observe)
        thread.start()
        assert started.wait(5)
        assert not finished.wait(0.05)
        assert lifecycle.generations == [0, 1]
    thread.join(5)
    assert not thread.is_alive()
    assert observed == [[1]]
    assert lifecycle.generations == [0, 2]


def test_history_only_mode_defers_dirty_patches_until_final_sync():
    import threading

    from vllm.distributed.pp_reconfigure.live import LiveLayerMigration
    from vllm.distributed.pp_reconfigure.patching import DirtyBitmap

    live = LiveLayerMigration.__new__(LiveLayerMigration)
    pool = SimpleNamespace(spec=SimpleNamespace(block_size=1))
    live.prepared = SimpleNamespace(
        moving=[0],
        kv_ready=False,
        old_kv={
            "model.layers.0.self_attn.attn": SimpleNamespace(
                _pp_reconfigure=(pool, None)
            )
        },
    )
    live.history_block_ids = [1]
    live.patching = False
    live.finish = threading.Event()
    live.cache_lock = threading.RLock()
    live.bitmap = DirtyBitmap(2)
    live.bitmap.mark([1])
    live.applied = 0
    historical_done = threading.Event()
    snapshots, errors = [], []
    controls: list[dict] = []

    def snapshot(slots, watermark, full=False):
        snapshots.append((slots, watermark, full))
        live.applied = watermark
        if full:
            historical_done.set()

    def run():
        try:
            live._send_loop()
        except BaseException as exc:
            errors.append(exc)

    live._send_snapshot = snapshot
    live.channel = SimpleNamespace(
        send_control=controls.append, receive_control=lambda: {"op": "STOPPED"}
    )
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert historical_done.wait(5)
        assert snapshots == [([1], 0, True)] and thread.is_alive()
    finally:
        live.finish.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert snapshots == [([1], 0, True), ([1], 1, False)]
    assert controls == [{"op": "STOP"}] and live.prepared.kv_ready


def test_reused_block_is_fully_dirtied_without_changing_convergence_watermark():
    from vllm.distributed.pp_reconfigure.live import LiveLayerMigration
    from vllm.distributed.pp_reconfigure.patching import DirtyBitmap, PatchLifecycle

    live = LiveLayerMigration.__new__(LiveLayerMigration)
    live.stage = 0
    live.log_dir = None
    live.lifecycle = PatchLifecycle([0, 1, 1])
    live.bitmap = DirtyBitmap(48)
    live.prepared = SimpleNamespace(
        source=0,
        old_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ]
        ),
    )
    live.before_step(SimpleNamespace(pp_block_generations=[0, 2, 1]))
    assert live.bitmap.drain() == (list(range(16, 32)), 0)
    live.before_step(SimpleNamespace(pp_block_generations=[0, 1, 1]))
    assert live.bitmap.drain() == ([], 0)
    assert live.lifecycle.generations == [0, 2, 1]


@pytest.mark.parametrize("still_running", [False, True])
def test_background_error_rolls_back_only_after_all_tasks_quiesce(still_running):
    import threading
    from unittest.mock import Mock

    from vllm.distributed.pp_reconfigure.live import LiveLayerMigration
    from vllm.distributed.pp_reconfigure.worker import LayerMigration

    live = LiveLayerMigration.__new__(LiveLayerMigration)
    live.preparing = False
    live.active = True
    live.error = "injected weight load failure"
    live.finish = threading.Event()
    live.kv_thread = Mock(is_alive=lambda: still_running)
    live.weights_thread = Mock(is_alive=lambda: False)
    live.worker = SimpleNamespace(device=torch.device("cuda:0"))
    live.gpu_profiler = None
    with (
        patch("torch.accelerator.synchronize") as synchronize,
        patch.object(
            LayerMigration, "rollback", return_value={"rolled_back": True}
        ) as restore,
    ):
        if still_running:
            with pytest.raises(TimeoutError, match="did not quiesce"):
                live.rollback()
            restore.assert_not_called()
            assert live.active
        else:
            assert live.rollback() == {"rolled_back": True}
            restore.assert_called_once()
            live.kv_thread.join.assert_called_once_with(timeout=60)
            live.weights_thread.join.assert_called_once_with(timeout=60)
            synchronize.assert_called_once_with(live.worker.device)
            assert not live.active
        assert live.finish.is_set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_inflight_patch_cannot_overwrite_reallocated_block():
    """Apply a captured old payload after reuse, then apply the new generation."""
    from vllm.distributed.pp_reconfigure.cache import StackedKVCache
    from vllm.distributed.pp_reconfigure.live import LiveLayerMigration
    from vllm.distributed.pp_reconfigure.patching import PatchLifecycle, copy_slots
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(4)]
    source = StackedKVCache(names, spec, 2, torch.device("cuda:0"), 4)
    incoming = StackedKVCache(names, spec, 2, torch.device("cuda:0"), 4)
    slots = list(range(16, 32))
    for name in names:
        source.block(name, 1).fill_(7)
        incoming.block(name, 1).fill_(42)
    old_payload = torch.stack([copy_slots(source.handles[n], slots) for n in names])
    torch.accelerator.synchronize()
    for name in names:
        source.block(name, 1).fill_(9)
    new_payload = torch.stack([copy_slots(source.handles[n], slots) for n in names])
    torch.accelerator.synchronize()
    live = LiveLayerMigration.__new__(LiveLayerMigration)
    live.prepared = SimpleNamespace(
        new_kv=incoming.handles,
        kv_ready=False,
        old_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
        ),
    )
    live.log_dir = None
    live.stage = 1
    live.lifecycle = PatchLifecycle([0, 2])
    live.stale_slots = live.bytes_sent = live.patch_rounds = live.applied = 0
    live.full_done = False
    live.transferred_slots = set()
    events = iter([(old_payload, 1, 1), (new_payload, 2, 2)])

    def receive(apply):
        item = next(events, None)
        if item is None:
            return False
        payload, version, generation = item
        with torch.cuda.stream(live.channel.stream):
            apply(
                payload,
                dict(
                    transaction=None,
                    version=version,
                    generations=[(1, generation)],
                    layers=names,
                    slots=slots,
                    watermark=version,
                    last=True,
                    full=False,
                ),
            )
        for name in names:
            assert torch.all(incoming.block(name, 1) == (42 if version == 1 else 9))
            assert torch.all(source.block(name, 1) == 9)
        return True

    live.channel = SimpleNamespace(stream=torch.cuda.Stream(), receive=receive)
    live._receive_loop()
    assert live.stale_slots == 16 and live.lifecycle.version == 2
    assert live.prepared.kv_ready and live.applied == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "prepare_live",
        "start_live",
        "progress_live",
        "sync_live",
        "commit",
        "start_cleanup",
        "cleanup_progress",
    ],
)
@pytest.mark.parametrize("mode", ["async", "async_no_patching"])
async def test_async_streams_before_pause_and_syncs_before_commit(
    tmp_path, failure, mode
):
    class LiveEngine(Engine):
        async def call_utility_async(self, method, *args):
            workers = await super().call_utility_async(method, *args)
            if method == "molink_state":
                return workers
            action = args[0]
            for worker in workers:
                worker.update(
                    online_reconfiguration=True,
                    source=0,
                    max_intermediate_blocks=64,
                    max_final_blocks=64,
                    num_blocks=64,
                )
                if action == "prepared_live":
                    worker["prepared"] = True
                if action == "finalize":
                    worker["cleanup_pending"] = True
                if action in ("start_cleanup", "cleanup_progress"):
                    assert not self.paused
                    worker["cleanup_done"] = True
                if action == "progress_live":
                    assert not self.paused
                    worker.update(
                        full_done=True,
                        weights_ready=True,
                        scheduled=1000 if mode == "async_no_patching" else 9,
                        applied=0 if mode == "async_no_patching" else 9,
                    )
                if action == "sync_live":
                    assert self.paused
                    worker.update(
                        compute_intervals_ms=[(1, 5)],
                        communication_intervals_ms=[(2, 4)],
                        bytes=100,
                        steps=3,
                        scheduled=1000 if mode == "async_no_patching" else 9,
                        applied=1000 if mode == "async_no_patching" else 9,
                        dirty_slots=0,
                        kv_ready=True,
                    )
            return workers

    start_action = "start_history" if mode == "async_no_patching" else "start_live"
    engine = LiveEngine(fail=start_action if failure == "start_live" else failure)
    if failure in ("start_cleanup", "cleanup_progress"):
        with pytest.raises(RuntimeError, match="committed_and_resumed"):
            await ReconfigurationController(engine, tmp_path).reconfigure(6, mode)
        assert not engine.paused and "rollback" not in engine.calls
        return
    if failure:
        with pytest.raises(RuntimeError, match="rolled_back_and_resumed"):
            await ReconfigurationController(engine, tmp_path).reconfigure(6, mode)
        assert engine.calls[-2:] == ["rollback", "resume"]
        return
    result = await ReconfigurationController(engine, tmp_path).reconfigure(6, mode)
    assert engine.calls == [
        "describe",
        "prepare_live",
        "prepared_live",
        start_action,
        "progress_live",
        "pause",
        "sync_live",
        "commit",
        "finalize",
        "resume",
        "start_cleanup",
        "cleanup_progress",
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
    with patch.dict("os.environ", {"VLLM_PP_RECONFIGURE_ONLINE": "1"}):
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

    outputs: list[UtilityOutput] = []
    EngineCoreProc._invoke_utility_method(
        "molink_resize_kv", fail, UtilityOutput(7), outputs.append
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(outputs[0]), type=UtilityOutput
    )
    future: Future[Any] = Future()
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


@pytest.mark.parametrize("defer", [True, False])
def test_finalize_retains_old_weights_until_post_resume_cleanup(monkeypatch, defer):
    import gc
    import weakref
    from contextlib import nullcontext

    from vllm.distributed.pp_reconfigure.live import LiveLayerMigration

    monkeypatch.setenv("VLLM_PP_RECONFIGURE_DEFER_CLEANUP", "1" if defer else "0")

    class OldLayer:
        weight_loader: "OldLayer"

    old = OldLayer()
    old.weight_loader = old  # Real parameters hold bound methods back to the layer.
    reference = weakref.ref(old)
    migration = LiveLayerMigration.__new__(LiveLayerMigration)
    migration.stage = 1
    migration.worker = SimpleNamespace(device=torch.device("cuda:0"))
    migration.cleanup_thread = None
    migration.cleanup_error = None
    migration.log_dir = None
    migration.prepared = SimpleNamespace(
        committed=True,
        source=0,
        old_layers=[old],
        transport=SimpleNamespace(close=lambda: None),
    )
    gc.freeze()
    del old
    try:
        with (
            patch("torch.accelerator.empty_cache"),
            patch("torch.accelerator.device_index", return_value=nullcontext()),
        ):
            migration.finalize()
            assert migration.prepared is None
            if defer:
                assert reference() is not None
                migration.start_cleanup()
            migration._wait_cleanup()
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
                    online_reconfiguration=True,
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
                w.update(
                    online_reconfiguration=True,
                    max_intermediate_blocks=0,
                    max_final_blocks=0,
                )
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

    async def fail(*args, **kwargs):
        if kind == "validation":
            raise ReconfigurationValidationError("invalid request")
        raise ValueError("internal bug")

    control = SimpleNamespace(
        lock=asyncio.Lock(),
        version=0,
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


def test_resource_monitor_defers_incomplete_cuda_reports_without_waiting(tmp_path):
    from unittest.mock import Mock

    from vllm.distributed.pp_reconfigure.resource_monitor import (
        ResourceMonitor,
        pack_report,
    )

    monitor = ResourceMonitor(0, lambda: {}, log_dir=str(tmp_path))
    local = {"stage": 0, "pid": 1, "sequence": 1}
    peer = {"stage": 1, "pid": 2, "sequence": 1}
    host = torch.stack([pack_report(local), pack_report(peer)])
    start = Mock()
    start.elapsed_time.return_value = 2
    end = Mock()
    end.query.return_value = False
    monitor.pending.append((start, end, host, local, "pp:test"))
    monitor.collect_reports()
    assert not monitor.reports
    start.elapsed_time.assert_not_called()
    end.synchronize.assert_not_called()
    end.query.return_value = True
    monitor.collect_reports()
    assert monitor.snapshot()["reports_received"] == {0: 1, 1: 1}
    assert monitor.publish_seconds == 0.002
