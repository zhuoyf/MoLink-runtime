# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts: conditioned estimates, feasible decisions, fail-closed security."""

import math
from typing import Any

import pytest

from vllm.distributed.pp_reconfigure.adaptive.model import (
    NodeState,
    PerformanceModel,
    RuntimeState,
)
from vllm.distributed.pp_reconfigure.adaptive.optimizer import (
    OptimizerConfig,
    PartitionOptimizer,
)
from vllm.distributed.pp_reconfigure.adaptive.security import (
    PlacementPlan,
    SecurityCapabilities,
    SecurityCostModel,
    SecurityPolicy,
)


def scenario():
    state = RuntimeState(
        100,
        0,
        8,
        16,
        [NodeState(i, 8, 10**10, 10**8, 10**6, 1024, 100, 10) for i in range(2)],
        4,
        128,
        4,
        "decode",
    )
    model = PerformanceModel()
    for stage, ms in ((0, 80), (1, 10)):
        model.update(model.compute_key(stage, 8, "decode", 4, 128, 4), ms, 99)
    model.update("network/bytes_per_second", 1e9, 99)
    model.update("network/latency_seconds", 0.001, 99)
    for boundary in (4, 12):
        key = PartitionOptimizer.migration_key(8, boundary, 4 * 1024 * 10)
        for metric, value in (("total", 1), ("pause", 0.1), ("interference", 0.2)):
            model.update(key + "/" + metric, value, 99)
    return state, model


def test_ewma_rejects_invalid_and_duplicate_observations(tmp_path):
    model = PerformanceModel()
    assert model.update("x", 10, 1)
    assert model.update("x", 20, 2)
    assert model.predict("x") == 12
    for value, timestamp in ((math.nan, 3), (-1, 3), (30, 2)):
        assert not model.update("x", value, timestamp)
    model.save(tmp_path / "model.json")
    assert PerformanceModel.load(tmp_path / "model.json").predict("x") == 12


def test_compute_does_not_mix_workloads():
    state, model = scenario()
    assert model.compute(0, 4, "decode", 4, 128, 4) == 40
    assert model.compute(0, 4, "prefill", 4, 128, 4) is None
    assert model.compute(0, 4, "decode", 8, 128, 4) is None


def test_optimizer_accounts_for_migration_and_cooldown():
    state, model = scenario()
    optimizer = PartitionOptimizer()
    assert optimizer.decide(state, model, 100)["target"] == 4
    optimizer.last_migration = 99
    assert optimizer.decide(state, model, 100)["reason"] == "cooldown"
    optimizer.last_migration = -math.inf
    key = optimizer.migration_key(8, 4, 4 * 1024 * 10)
    model.estimates[key + "/pause"].value = 100
    assert optimizer.decide(state, model, 100)["target"] == 8


def test_memory_staleness_and_failure_prevent_migration():
    state, model = scenario()
    optimizer = PartitionOptimizer()
    state.nodes[1].memory_budget = 10**8
    assert optimizer.decide(state, model, 100)["target"] == 8
    assert optimizer.decide(state, model, 200)["reason"] == "paused_or_stale"
    optimizer.failed = True
    assert optimizer.decide(state, model, 100)["reason"] == "failure_latched"


def test_security_requires_real_capabilities_and_cost_provider():
    policy = SecurityPolicy("trusted")
    plan = PlacementPlan(4, ("tee", "tee"), "trusted")
    with pytest.raises(ValueError):
        policy.validate(
            plan, [SecurityCapabilities("a"), SecurityCapabilities("b")], 100
        )
    caps = [SecurityCapabilities(n, True, True, True, 101) for n in ("a", "b")]
    policy.validate(plan, caps, 100)
    with pytest.raises(ValueError):
        policy.validate(plan, caps, 102)
    with pytest.raises(ValueError):
        SecurityCostModel().cost_seconds(plan, None)
    SecurityPolicy().validate(PlacementPlan(4), [], 100)


def test_configuration_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        PerformanceModel(math.nan)
    with pytest.raises(ValueError):
        OptimizerConfig(horizon=math.inf)


@pytest.mark.asyncio
async def test_stale_version_is_rejected_before_executor(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from vllm.distributed.pp_reconfigure.control import ReconfigurationController
    from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError

    engine = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=AsyncMock())
    )
    controller = ReconfigurationController(engine, tmp_path)
    with pytest.raises(ReconfigurationValidationError, match="version"):
        await controller.reconfigure(4, "async", expected_version=1)
    engine.engine_core.call_utility_async.assert_not_called()


def test_configured_kv_capacity_caps_the_prediction():
    state, model = scenario()
    for node in state.nodes:
        node.block_cap = 12
    decision = PartitionOptimizer().decide(state, model, 100)
    assert all(c["capacity_blocks"] == 12 for c in decision["candidates"] if c["valid"])


def test_fresh_measurements_propagate_gpu_speed_changes():
    _, model = scenario()
    key = model.compute_key(0, 4, "decode", 4, 128, 4)
    model.freeze_compute_baseline()
    model.update(key, 400, 110)
    assert model.compute(0, 8, "decode", 4, 128, 4) == pytest.approx(224)


def test_nonfinite_node_measurements_are_rejected():
    with pytest.raises(ValueError, match="measurement"):
        NodeState(0, 8, math.nan, 100, 10, 10, 100, 2)


def test_migration_payload_is_separate_from_allocation_granularity():
    state, model = scenario()
    for node in state.nodes:
        node.kv_payload_bytes_per_layer_block = 64
    key = PartitionOptimizer.migration_key(8, 4, 4 * 64 * 10)
    for metric, value in (("total", 2), ("pause", 0.2), ("interference", 0.3)):
        model.update(key + "/" + metric, value, 101)
    candidate = PartitionOptimizer().evaluate(state, model, 4, 101)
    assert candidate["migration_total"] == 2
    assert candidate["peak_bytes"][1] > state.nodes[1].used_blocks * 64


def test_no_gain_before_a_migration_finishes_within_the_horizon():
    state, model = scenario()
    key = PartitionOptimizer.migration_key(8, 4, 4 * 1024 * 10)
    model.estimates[key + "/total"].value = 120
    model.estimates[key + "/pause"].value = 0.1
    model.estimates[key + "/interference"].value = 0
    assert PartitionOptimizer().decide(state, model, 100)["target"] == 8


def test_rerunning_experiment_preserves_raw_data_without_mixing_metrics(tmp_path):
    import json

    from experiments.adaptive_pp.experiment import archive_previous_run

    (tmp_path / "profile.json").write_text("calibration")
    (tmp_path / "online_update_requests.jsonl").write_text("raw requests")
    (tmp_path / "summary.jsonl").write_text(
        json.dumps({"label": "online_update"})
        + "\n"
        + json.dumps({"label": "comparison_0_full"})
        + "\n"
    )
    archive_previous_run(tmp_path, "online_update")
    assert (tmp_path / "profile.json").read_text() == "calibration"
    assert (
        json.loads((tmp_path / "summary.jsonl").read_text())["label"]
        == "comparison_0_full"
    )
    archived = list((tmp_path / "archive").glob("*/online_update_requests.jsonl"))
    assert len(archived) == 1 and archived[0].read_text() == "raw requests"


def test_idle_nodes_do_not_migrate_using_previous_request_timings():
    state, model = scenario()
    state.running = 0
    decision = PartitionOptimizer().decide(state, model, 100)
    assert decision["target"] == state.boundary and decision["reason"] == "idle"


def test_runtime_rejects_stale_future_and_incompatible_compute_samples(
    tmp_path, monkeypatch
):
    from copy import deepcopy

    from vllm.distributed.pp_reconfigure.adaptive.runtime import AdaptiveScheduler

    model = PerformanceModel()
    model.conditions = {
        "activation_bytes_per_token": 256,
        "total_layers": 16,
        "compute_timing_scope": "forward_after_pp_receive",
    }
    path = tmp_path / "profile.json"
    model.save(path)
    runtime = AdaptiveScheduler(None, path, tmp_path)
    monkeypatch.setattr(
        "vllm.distributed.pp_reconfigure.adaptive.runtime.time.time", lambda: 100
    )
    reports: dict[str, Any] = {"scheduler_paused": False, "workers": []}
    for stage in (0, 1):
        sample = {
            "sequence": 1,
            "timestamp": 99,
            "gpu_ms": 10,
            "layers": 8,
            "phase": "decode",
            "batch": 4,
            "context": 128,
            "tokens": 4,
        }
        reports["workers"].append(
            {
                "metrics": {
                    "stage": stage,
                    "sample_time": 100,
                    "gpu_uuid": str(stage),
                    "layer_range": [stage * 8, (stage + 1) * 8],
                    "num_blocks": 100,
                    "kv_capacity_bytes": 819200,
                    "torch_allocated_bytes": 919200,
                    "memory_budget_bytes": 10**10,
                    "weight_bytes_per_layer": 100,
                    "active_kv_blocks": 10,
                    "preemptions": 0,
                    "stage_last": sample,
                    "workload_samples": [sample],
                    "compute_timing_scope": "forward_after_pp_receive",
                }
            }
        )
    state = runtime.ingest(reports, 0)
    assert state.boundary == 8 and state.batch == 4
    key = model.compute_key(0, 8, "decode", 4, 128, 4)
    runtime.ingest(reports, 0)
    assert runtime.model.estimates[key].count == 1
    future = deepcopy(reports)
    for worker in future["workers"]:
        worker["metrics"]["stage_last"].update(timestamp=101, sequence=2)
    with pytest.raises(ValueError, match="fresh workload"):
        runtime.ingest(future, 0)
    assert runtime.model.estimates[key].count == 1
    stale = deepcopy(reports)
    stale["workers"][0]["metrics"]["sample_time"] = 80
    with pytest.raises(ValueError, match="stale"):
        runtime.ingest(stale, 0)
    reports["workers"][0]["metrics"]["compute_timing_scope"] = "runner"
    with pytest.raises(ValueError, match="timing scope"):
        runtime.ingest(reports, 0)


def test_affine_compute_keeps_stage_overhead_when_layers_change():
    model = PerformanceModel()
    for layers, elapsed in ((2, 12), (4, 14), (8, 18)):
        model.update(model.compute_key(0, layers, "decode", 4, 128, 4), elapsed, layers)
    model.freeze_compute_baseline()
    assert model.compute(0, 6, "decode", 4, 128, 4) == pytest.approx(16)
    model.update(model.compute_key(0, 4, "decode", 4, 128, 4), 28, 100)
    assert model.compute(0, 6, "decode", 4, 128, 4) == pytest.approx(19.2)


def test_security_runtime_refuses_trusted_mode_without_a_provider(tmp_path):
    from vllm.distributed.pp_reconfigure.adaptive.runtime import AdaptiveScheduler

    profile = tmp_path / "profile.json"
    PerformanceModel().save(profile)
    with pytest.raises(ValueError, match="trusted execution provider"):
        AdaptiveScheduler(
            None, profile, tmp_path, OptimizerConfig(security_mode="trusted")
        )


def test_hold_uses_allocated_kv_capacity_not_unallocated_free_memory():
    state, model = scenario()
    optimizer = PartitionOptimizer()
    current = optimizer.evaluate(state, model, state.boundary, 100)
    assert current["capacity_blocks"] == 100
    target = optimizer.evaluate(state, model, 4, 100)
    assert target["capacity_blocks"] > 100


@pytest.mark.asyncio
async def test_runtime_discards_state_when_configuration_changes_during_collection(
    tmp_path,
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from vllm.distributed.pp_reconfigure.adaptive.runtime import AdaptiveScheduler

    profile = tmp_path / "profile.json"
    PerformanceModel().save(profile)
    control = SimpleNamespace(version=0, lock=asyncio.Lock(), reconfigure=AsyncMock())

    async def collect(method):
        control.version += 1
        return {}

    control.engine = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=collect)
    )
    runtime = AdaptiveScheduler(control, profile, tmp_path)
    await runtime.tick()
    control.reconfigure.assert_not_called()
    assert not runtime.model.estimates


def test_decode_balances_the_pipeline_bottleneck_for_concurrent_requests():
    state, model = scenario()
    state.boundary = 4
    state.nodes[0].layers, state.nodes[1].layers = 4, 12
    for stage in (0, 1):
        model.estimates[model.compute_key(stage, 8, "decode", 4, 128, 4)].value = 8
        model.update(model.compute_key(stage, 8, "prefill", 4, 128, 4), 8, 99)
    assert PartitionOptimizer().decide(state, model, 100)["target"] == 8
    state.phase = "prefill"
    assert PartitionOptimizer().decide(state, model, 100)["target"] == 4
