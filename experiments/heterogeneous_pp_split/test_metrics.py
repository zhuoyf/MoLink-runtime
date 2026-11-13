"""Fixed-window accounting must exclude drain tokens but include SLO latency."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "split_run", Path(__file__).with_name("run.py")
)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_window_and_slo_use_complete_request():
    rows = [
        {
            "start": 10,
            "end": 13,
            "times": [10.5, 11, 12.5],
            "ttft_ms": 500,
            "tpot_ms": 1000,
        },
        {
            "start": 10,
            "end": 11.5,
            "times": [11, 11.5],
            "ttft_ms": 1000,
            "tpot_ms": 500,
        },
    ]
    result = run.summarize(rows, 10, 12, {"ttft_ms": 800, "tpot_ms": 1100})
    assert result["throughput"] == 2
    assert result["goodput"] == 1
    assert result["requests"] == 2
    assert result["drain_seconds"] == 1


def test_no_baseline_or_failed_trials_are_not_zero_measurements(tmp_path):
    run.OUT = tmp_path
    run.save(
        tmp_path / "search/A-k08/result.json",
        {"workload": "A", "k": 8, "status": "failed", "reason": "OOM"},
    )
    run.report()
    import json

    result = json.loads((tmp_path / "best.json").read_text())
    assert result["best"]["A"] == {}
    assert result["completed_groups"] == 1
    assert (tmp_path / "curves.png").exists()


def test_resume_skips_terminal_outcomes_and_retries_interrupted(tmp_path):
    paths = [tmp_path / str(k) / "result.json" for k in range(51)]
    run.save(paths[0], {"status": "ok"})
    run.save(paths[1], {"status": "failed", "reason": "OOM"})
    paths[2].parent.mkdir()  # Interrupted attempt has no committed result.
    remaining = [p for p in paths if not run.terminal_result(p)]
    assert len(remaining) == 49
    assert paths[2] in remaining


def test_report_compares_best_with_16_and_keeps_ties(tmp_path):
    import json

    run.OUT = tmp_path
    for k, throughput in [(15, 120), (16, 100), (17, 120)]:
        run.save(
            tmp_path / f"search/A-k{k}/result.json",
            {
                "workload": "A",
                "k": k,
                "status": "ok",
                "throughput": throughput,
                "goodput": throughput,
                "p95_ttft_ms": 1,
                "p95_tpot_ms": 2,
                "stages": [{"stage_gpu_ms_mean": 3, "kv_cache_usage": 0.5}] * 2,
            },
        )
    run.report()
    value = json.loads((tmp_path / "best.json").read_text())["best"]["A"]["throughput"]
    assert value["ties"] == [15, 17]
    assert abs(value["gain_over_16_percent"] - 20) < 1e-9
