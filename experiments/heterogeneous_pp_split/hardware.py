"""Real contention, reservation and independent GEMM/copy calibration."""

import argparse
import json
import signal
import time
from pathlib import Path

import torch


def kernels(gpu):
    torch.cuda.set_device(gpu)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    x = torch.ones(64 * 1024 * 1024, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    return (lambda: torch.mm(a, b, out=c)), (lambda: y.copy_(x))


def bench(gpu):
    gemm, copy = kernels(gpu)
    result = {}
    for name, fn, units in [
        ("tflops", gemm, 2 * 4096**3 / 1e12),
        ("gb_s", copy, 2 * 256 * 2**20 / 1e9),
    ]:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        trials = []
        for _ in range(3):
            start = time.perf_counter()
            count = 0
            while time.perf_counter() - start < 1:
                for _ in range(10):
                    fn()
                torch.cuda.synchronize()
                count += 10
            trials.append(units * count / (time.perf_counter() - start))
        result[name] = sorted(trials)[1]
        result[name + "_trials"] = trials
    return result


def hold(gpu, available, duty, ready):
    gemm, copy = kernels(gpu)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    reserve = max(0, free - int(available * 2**30))
    ballast = torch.empty(reserve, device="cuda", dtype=torch.uint8)
    ballast.fill_(1)
    torch.cuda.synchronize()
    Path(ready).write_text(
        json.dumps(
            {
                "gpu": gpu,
                "reserved_bytes": reserve,
                "free_bytes": torch.cuda.mem_get_info()[0],
                "total_bytes": total,
                "duty": duty,
            }
        )
    )
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    fn = copy if gpu == 0 else gemm
    while running:
        if duty == 0:
            time.sleep(0.2)
            continue
        start = time.perf_counter()
        while time.perf_counter() - start < 0.02 * duty:
            fn()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        time.sleep(max(0, elapsed * (1 - duty) / max(duty, 1e-9)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["bench", "hold"])
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--available", type=float, default=12)
    p.add_argument("--duty", type=float, default=0.35)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.mode == "bench":
        Path(a.out).write_text(json.dumps(bench(a.gpu), indent=2))
    else:
        hold(a.gpu, a.available, a.duty, a.out)
