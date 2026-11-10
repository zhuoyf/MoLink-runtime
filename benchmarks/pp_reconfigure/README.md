# Pipeline migration benchmark

Run from the repository root with the existing `.venv`, `uv`, two available
GPUs, and a local Llama checkpoint. The benchmark checks exact output tokens,
active request state, zero preemptions, and inference progress during online
migration. It saves total migration time, pause time, phase timings, and token
timestamps for both directions.

## Small-model continuity check

```bash
uv run --no-project --python .venv/bin/python \
  benchmarks/pp_reconfigure/prepare_tiny_model.py \
  --output /tmp/pp-test-model --tokenizer /path/to/llama-tokenizer

MODEL=/tmp/pp-test-model bash benchmarks/pp_reconfigure/serve_online.sh
```

In another terminal:

```bash
uv run --no-project --python .venv/bin/python \
  benchmarks/pp_reconfigure/compare_migration.py \
  --boundaries 6 8 --iterations 1 --verify-kv
```

## Migration performance

For a 32-layer Llama checkpoint, start the service with a 20/12 partition:

```bash
MODEL=/path/to/llama7b VLLM_PP_LAYER_PARTITION=20,12 \
  bash benchmarks/pp_reconfigure/serve_online.sh \
  --dtype float16 --block-size 64 --gpu-memory-utilization 0.70 \
  --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 256 \
  --num-gpu-blocks-override 256
```

```bash
uv run --no-project --python .venv/bin/python \
  benchmarks/pp_reconfigure/compare_migration.py \
  --boundaries 12 20 --input-tokens 2048 --output-tokens 256 \
  --concurrency 4 --iterations 5 \
  --output benchmark_results/pp_reconfigure/comparison.json

uv run --no-project --python .venv/bin/python \
  benchmarks/pp_reconfigure/analyze_migration.py \
  benchmark_results/pp_reconfigure/comparison.json --target 12 \
  --output-dir benchmark_results/pp_reconfigure/summary
```

Use `--input-tokens 3584` for the longer workload and a distinct output path.
Both boundaries must align to the layer stack size, and input plus output must
fit the model context and available KV blocks. The default modes are `sync` and
`async`; `--modes` also supports `async_no_patching`. Each mode visits the first
boundary and returns to the second boundary on every iteration. `--resize-kv`
additionally exercises 48/64-block resizing with the small workload.

`total_ms` includes preparation and cleanup; `pause_ms` spans drain through
resume. `stream_token_times` supports inference latency analysis. Set
`VLLM_PP_RECONFIGURE_PROFILE=1` on the service for separate CUDA traces, then
pass `--trace /path/to/gpu-stage0.json` to the analyzer to measure kernel
overlap. `--verify-kv` enables digest validation. Profile and digest runs add
overhead, so collect ordinary latency measurements separately.
