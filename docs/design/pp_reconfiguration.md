# Pipeline reconfiguration

The `vllm.distributed.pp_reconfigure` package migrates model layers and KV cache
between two pipeline stages while preserving active requests. It provides
synchronous migration, online migration with incremental KV patches, and online
migration with a final synchronization only.

## Components

| Module | Responsibility |
| --- | --- |
| `control.py` | Transaction coordination, admission checks, pause, commit and recovery |
| `worker.py` | Layer binding, weight manifests and worker transaction state |
| `live.py` | Background history transfer, patching and resource retirement |
| `cache.py` | Address-based, layer-stacked KV allocations |
| `resize.py` | Scheduler block capacity and stable logical block IDs |
| `patching.py` | Dirty slots, allocation generations and patch versions |
| `communication.py` | Ordered control messages and migration NCCL operations |
| `weights.py` | Budgeted CPU weight caching and checkpoint loading |
| `monitor.py`, `resource_monitor.py` | Transaction logs and resource snapshots |

The current implementation supports two local GPUs, TP=1, a local Llama
checkpoint, uniform unquantized full attention, eager execution, and the Triton
attention backend. Stage count stays fixed. Both current and target layer
boundaries must align to the configured stack size; unsupported targets are
rejected before committing a configuration change.

## Configuration

Environment variables are registered in `vllm.envs`. Set them consistently on
all workers before starting the server.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_PP_RECONFIGURE_ENABLED` | `0` | Enable the reconfiguration control interface |
| `VLLM_PP_RECONFIGURE_ONLINE` | `0` | Allocate address-based KV caches for online migration |
| `VLLM_PP_RECONFIGURE_STACK_SIZE` | `2` | Number of layers in a physical KV stack |
| `VLLM_PP_RECONFIGURE_KV_TRANSPORT` | `nixl` | Synchronous transport: `nixl` or `pp`; use `pp` for online mode |
| `VLLM_PP_RECONFIGURE_WEIGHT_CACHE_BYTES` | `0` | Per-worker CPU weight tensor budget; zero loads from checkpoint |
| `VLLM_PP_RECONFIGURE_DEFER_CLEANUP` | `1` | Clean up retired allocations after resuming generation |
| `VLLM_PP_RECONFIGURE_PROFILE` | `0` | Capture migration CUDA activity for diagnostics |
| `VLLM_PP_RECONFIGURE_PROFILE_CPU` | `0` | Include CPU activity when profiling is enabled |
| `VLLM_PP_RECONFIGURE_LOG_DIR` | `pp_reconfigure_logs` | Local transaction output directory |

`VLLM_PP_LAYER_PARTITION` specifies the initial number of layers on each stage.
A stack size of four requires boundaries divisible by four; use a stack size of
two for a two-layer migration.

## Transaction and consistency

1. Validate the partition, checkpoint and common intermediate KV capacity.
2. Prepare incoming weights and detached KV allocations. Keep existing handles
   for retained layers and preserve logical block IDs during shrink/expand.
3. Transfer history for active blocks and reusable shared prefix-cache blocks.
   Unused, disabled and null blocks are excluded from the common manifest.
4. Track writes and block copies in a dirty bitmap. Gather all moving layers
   into immutable payloads bounded by 16 MiB, then transfer them in the background.
5. Pause and drain inference after history finishes and the original token-lag
   convergence condition is met. Apply the final dirty set and synchronize both
   workers before committing the new partition.
6. Resume generation, retire old allocations, and restore feasible KV capacity.
   Further migration and resize operations wait for resource retirement.

Each patch carries a transaction ID, a strictly increasing version and block
allocation generations. Fresh logical ownership advances the generation;
shared prefix references retain it. The destination filters older generations,
and the source marks the entire reused block dirty without advancing the token
watermark. Acknowledgement follows completed device application. Final versions
and watermarks must agree before commit.

An unchanged generation observation does not wait for an in-flight patch;
actual ownership updates still take the lifecycle lock. Incoming shadow KV is
separate from the cache currently used by inference. Source gathers wait for
previous writes, and the next forward waits for gather completion before
reusing the source slots.

CPU metadata waits do not hold the GPU communication lock. NCCL payloads and
sampling broadcasts retain their completion-event ordering. Rollback quiesces
background threads and device work before restoring the old configuration.
Post-commit cleanup errors preserve the committed partition rather than
restoring retired resources.

CPU weight caching uses a bounded LRU with background prefetch. Missing,
disabled or over-budget entries retain the checkpoint loading path. The budget
covers cached tensors, not total process RSS.

## Running the benchmark

Create a deterministic ten-layer checkpoint using a compatible local tokenizer:

```bash
uv run --no-project --python .venv/bin/python \
  benchmarks/pp_reconfigure/prepare_tiny_model.py \
  --output /tmp/pp-test-model --tokenizer /path/to/llama-tokenizer

MODEL=/tmp/pp-test-model VLLM_PP_LAYER_PARTITION=8,2 \
  bash benchmarks/pp_reconfigure/serve_online.sh
```

In another terminal:

```bash
uv run --no-project --python .venv/bin/python \
  benchmarks/pp_reconfigure/compare_migration.py \
  --model pp-reconfigure-test --boundaries 6 8 --iterations 1 --verify-kv \
  --output benchmark_results/pp_reconfigure/continuity.json
```

The existing project service integration exposes `POST /molink/reconfigure`:

```json
{"boundary": 6, "mode": "async", "verify_kv": false}
```

Modes are `sync`, `async`, and `async_no_patching`. The API default is `sync`.
`GET /molink/monitor` returns scheduler and worker state. The controller
serializes transactions, and HTTP client cancellation does not abort a running
migration. Inspect monitor state and transaction logs after an HTTP failure.

`total_ms` ends after cleanup and capacity restoration, before the final state
snapshot RPC. `pause_ms` includes draining and ends when Resume returns.
`timings_ms` records individual host phases. Client token gaps and actual CUDA
kernel overlap are separate measurements; event-window estimates and Commit
time alone do not establish service pause or device overlap. KV verification
and profiling add overhead and should be disabled in latency measurements.

`analyze_migration.py`, `visualize_migration.py`, and `plot_timeline.py` read
benchmark outputs to produce local summaries and plots. Their CLI help describes
required inputs. Generated measurements belong in the ignored local output
directories or outside the repository.

## Tests

```bash
uv run --no-project --python .venv/bin/python -m pytest \
  tests/v1/worker/test_pp_reconfigure.py -q
```

The suite covers cache addressing, block lifetime, generations, patch ordering,
resource budgets, rollback and asynchronous cleanup. Real-GPU continuity runs
verify tokens against a static baseline and compare final KV digests in both
directions. Test artifacts are local and are not part of the source package.
