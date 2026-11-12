"""Check window accounting and real pilot evidence before formal launch."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scan", ROOT / "benchmark.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)
records = [
    dict(times=[9, 10, 15, 20, 21], end=22, ttft_ms=1, tpot_ms=1),
    dict(times=[11, 19, 22], end=23, ttft_ms=6000, tpot_ms=1),
]
r = scan.summarize(records, 10, 20, dict(ttft_ms=5000, tpot_ms=100))
assert r["throughput"] == 0.4 and r["goodput"] == 0.2
assert r["completed_in_window"] == 0


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


if __name__ == "__main__":
    for name in ("full_c4", "threshold_c64", "static4_c64", "static28_c64"):
        path = ROOT / "pilot" / name
        summary = json.loads((path / "summary.json").read_text())
        assert summary["complete"] and summary["duration"] == 120
        phases = rows(path / "phases.jsonl")
        assert [p["phase"] for p in phases] == ["A", "B", "C"]
        assert max(abs(p["requested"] - p["planned"]) for p in phases[1:]) < 1
        samples = rows(path / "samples.jsonl")
        assert all(
            "running_requests" in s
            and "waiting_requests" in s
            and "kv_cache_usage" in s
            for s in samples
        )
        assert all(
            "gpu_utilization_percent" in w for s in samples for w in s["workers"]
        )
        events = rows(path / "load.jsonl")
        assert max(e["active"] for e in events) == summary["concurrency"]
        assert sum(e["event"] == "submit" for e in events) > summary["concurrency"]
        assert all(e["time"] < summary["end"] for e in events if e["event"] == "submit")
        for slot in range(summary["concurrency"]):
            own = [e for e in events if e["slot"] == slot]
            assert all(
                b["time"] - a["time"] < 0.1
                for a, b in zip(own, own[1:])
                if a["event"] == "complete"
            )
    outcomes = rows(ROOT / "pilot/full_c4/decisions/outcome.log")
    assert any(e.get("status") == "resumed" for e in outcomes), (
        "No automatic migration observed"
    )
    (ROOT / "pilot/PASSED").write_text(
        "Window accounting, replenishment, phases, tokens, "
        "telemetry, automatic migration checked.\n"
    )
    print("PASSED")
