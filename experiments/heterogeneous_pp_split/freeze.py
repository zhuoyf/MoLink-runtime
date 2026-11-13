"""Freeze only the measured, validated short-test protocol."""

import hashlib
import json
from pathlib import Path

from run import HERE, OUT, save


def fingerprint():
    paths = sorted(HERE.glob("*.py"))
    paths.append(HERE.parent / "adaptive_pp/experiment.py")
    return {
        str(p.relative_to(HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def main():
    config = json.loads((HERE / "config.json").read_text())
    assert config["splits"] == list(range(8, 25))
    assert (config["warmup_seconds"], config["measure_seconds"]) == (30, 300)
    calibration = json.loads((OUT / "calibration.json").read_text())
    for key in ["contention_duty", "mps_threads_percent", "available_gib"]:
        assert calibration["config"][key] == config[key], f"calibration differs: {key}"
    measured = calibration["contended"]
    compute_ratio = measured[1]["tflops"] / measured[0]["tflops"]
    bandwidth_ratio = measured[0]["gb_s"] / measured[1]["gb_s"]
    assert 0.6 <= compute_ratio <= 0.8, compute_ratio
    assert 0.55 <= bandwidth_ratio <= 0.75, bandwidth_ratio
    evidence = {}
    for name in config["workloads"]:
        result = json.loads((OUT / f"pilot/{name}-k16/result.json").read_text())
        assert result["status"] == "ok", name
        directory = Path(result["directory"])
        used = json.loads((directory / "parameters.json").read_text())["config"]
        for key, value in config.items():
            if key not in ["warmup_seconds", "measure_seconds"]:
                assert used[key] == value, f"pilot config differs: {name}/{key}"
        stages = result["stages"]
        assert [s["visible_sms"] for s in stages] == [[84], [48]], name
        assert result["request_supply_active_fraction"] >= 0.9, name
        prefill = sum(s["phase_gpu_ms"]["prefill"] for s in stages)
        decode = sum(s["phase_gpu_ms"]["decode"] for s in stages)
        evidence[name] = {
            "result": str(directory / "result.json"),
            "prefill_gpu_time_fraction": prefill / (prefill + decode),
            "peak_kv_fraction": max(s["kv_cache_usage"] for s in stages),
            "preemptions": max(s["preemptions"] for s in stages),
            "supply_active_fraction": result["request_supply_active_fraction"],
            "stage_busy_fraction": [s["forward_busy_fraction"] for s in stages],
        }
    assert evidence["A"]["prefill_gpu_time_fraction"] > 0.5, evidence["A"]
    assert evidence["B"]["prefill_gpu_time_fraction"] < 0.1, evidence["B"]
    assert evidence["C"]["peak_kv_fraction"] > 0.85, evidence["C"]
    assert evidence["C"]["preemptions"] > evidence["B"]["preemptions"], evidence
    save(
        OUT / "frozen.json",
        {
            "validated": True,
            "config": config,
            "fingerprint": fingerprint(),
            "calibration": calibration,
            "compute_gpu1_over_gpu0": compute_ratio,
            "bandwidth_gpu0_over_gpu1": bandwidth_ratio,
            "pilot": evidence,
        },
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
