# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate formal launch on measured pilot evidence and immutable inputs."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmark import BASE, ROOT, configuration


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(check):
    path = ROOT / "frozen.json"
    if check:
        config = json.loads(path.read_text())
        for name, digest in config["sources"].items():
            assert sha(Path(name)) == digest, f"Frozen input changed: {name}"
        assert sha(path) == (ROOT / "frozen.sha256").read_text().strip()
        print("Frozen inputs verified", flush=True)
        return
    assert not path.exists(), "Already frozen; never tune formal parameters"
    best = json.loads((ROOT / "pilot/phase_best.json").read_text())
    assert best["A"] != best["B"] and best["B"] != best["C"]
    dynamic = ROOT / "pilot/dynamic"
    summary = json.loads((dynamic / "summary.json").read_text())
    assert summary["complete"] and summary["preemptions"] == 0
    records = [
        json.loads(x) for x in (dynamic / "requests.jsonl").read_text().splitlines()
    ]
    reference = json.loads(
        (ROOT / "pilot/profile/A_16_requests.jsonl").read_text().splitlines()[0]
    )["token_ids"]
    assert all(r["token_ids"] == reference for r in records)
    assert all(
        c["resumed_requests"] == 0 and c["prefill_tokens"] == 32 * len(records)
        for c in summary["counters"]
    )
    outcomes = [
        json.loads(x)
        for x in (dynamic / "decisions/outcome.log").read_text().splitlines()
    ]
    assert all(x["status"] == "resumed" for x in outcomes)
    targets = [x["result"]["boundary"] for x in outcomes]
    assert any(b > 16 for b in targets) and any(b < 16 for b in targets), targets
    for phase in "ABC":
        for boundary in range(4, 32, 4):
            data = json.loads(
                (ROOT / f"pilot/profile/{phase}_{boundary}.json").read_text()
            )
            for w in data["simulation"]["workers"]:
                assert w["num_blocks"] == data["simulation"]["capacity"]
                assert w["kv_capacity_bytes"] <= w["kv_budget_bytes"]
    import platform

    import torch

    for phase, stage in (("B", 1), ("C", 0)):
        base = json.loads((ROOT / "pilot/profile/A_16.json").read_text())
        changed = json.loads((ROOT / f"pilot/profile/{phase}_16.json").read_text())
        baseline_ms = base["state"]["workers"][stage]["stage_last"]["gpu_ms"]
        changed_ms = changed["state"]["workers"][stage]["stage_last"]["gpu_ms"]
        assert changed_ms - baseline_ms > 32, "injected delay not observed"
    config = configuration(120)
    config["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
    }
    source_paths = (
        list(ROOT.glob("*.py"))
        + list(ROOT.glob("*.sh"))
        + [BASE, ROOT / "pilot/phase_best.json"]
    )
    source_paths += list(Path("vllm/distributed/pp_reconfigure/adaptive").glob("*.py"))
    source_paths += [
        Path(p)
        for p in (
            "vllm/distributed/pp_reconfigure/heterogeneity.py",
            "vllm/distributed/pp_reconfigure/monitor.py",
            "vllm/v1/worker/gpu_worker.py",
            "vllm/v1/worker/gpu/model_runner.py",
            "vllm/v1/engine/core.py",
            "vllm/entrypoints/serve/dev/molink/api_router.py",
        )
    ]
    config["sources"] = {str(p): sha(p) for p in source_paths}
    config["pilot_validation"] = {
        "requests": len(records),
        "token_mismatches": 0,
        "extra_prefill_tokens": 0,
        "targets": targets,
    }
    path.write_text(json.dumps(config, indent=2))
    (ROOT / "frozen.sha256").write_text(sha(path) + "\n")
    print("Pilot passed; formal inputs frozen", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
