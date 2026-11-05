# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect KV/request continuity and transaction recovery at the cheapest level.

The worker owns layer/KV rebinding; its contract is tested on real GPUs by the
acceptance script. These unit tests cover control ordering and failures before
that expensive boundary, plus allocation-aware memory reporting.
"""

from copy import deepcopy
from types import SimpleNamespace
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
        self.request = {
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
