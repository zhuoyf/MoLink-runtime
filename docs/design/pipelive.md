# PipeLive online KV migration on vLLM 0.29.0

Reference: [PipeLive v2, 23 September 2026](https://arxiv.org/html/2604.12171v2),
§4 Algorithm 1, §5 and §6.1. Base: `feat/molink-sync-pp-reconfigure` at
`0a6ce7e685`. Development: `feat/pipelive-online-kv-migration`.

## Scope and source audit

The base provides two-stage Llama (TP=DP=1), manual contiguous layer-boundary
changes, eager execution, native KV binding, safetensors layer loading, keep
pause/resume, rollback before source release, JSONL logs and continuity checks.
Its `prepare` duplicates destination KV and its controller pauses before loading
or transferring data. The new opt-in path reuses its model binding, loader,
control plane and scheduler, while implementing the paper's live KV mechanism.
There is no partition-selection policy.

Supported configuration: CUDA, MRv2, `LlamaForCausalLM`, PP=2, TP=DP=1, eager
execution, `TRITON_ATTN`, uniform unquantized full-attention KV, local safetensors,
and NCCL pipeline forwarding (`HETVLLM_PP_LINK=0`). The existing synchronous
prototype remains available when `HETVLLM_PIPELIVE` is unset.

## Implementation and paper mapping

| Component | Files and responsibility |
| --- | --- |
| §5 allocation and stacking | `pp_reconfigure/cache.py`: independently allocated stacks, one physical allocation per block index and k layers; address handles and stack retirement |
| §5 address-based attention | `triton_attn.py`, `triton_unified_attention.py`, `triton_reshape_and_cache_flash.py`: resolve block IDs to absolute addresses for KV stores and both prefill/decode attention |
| §4 resizing | `pp_reconfigure/resize.py`, `live.py`, `engine/core.py`: common block budget, native free queue/hash eviction, worker preparation before scheduler commit, rollback of staged expansion |
| §6.1 communication | `communication.py`, `parallel_state.py`, `patching.py`: separate reusable PyNccl communicator, low-priority migration stream, device mutex, TCP ACK/ACCEPT/REJECT, CUDA completion reservations |
| §6.1 dirty patching | `patching.py`, `live.py`, `gpu_worker.py`: physical-slot bitmap, bounded immutable snapshots, source write/gather ordering, receiver application acknowledgements |
| §4 commit | `control.py`, `live.py`: convergence, keep pause and drain, final patch/STOP acknowledgement, weights/KV readiness, all-worker binding, source release and resume |
| Native cache consumers | `worker/utils.py`, `monitor.py`: address-aware block zero/copy and actual stacked allocation accounting |
| Validation | `test_pp_reconfigure.py`, `compare_pipelive.py`, `analyze_pipelive.py`: failure ordering, GPU correctness, multi-request continuity, metrics and optional CUPTI analysis |

### Memory and block lifetime

`HETVLLM_PIPELIVE_STACK` selects k (default 2); boundaries and layer ranges must
align to k. Set k=1 for single-layer granularity. Each stack is rounded to 2 MiB.
For Llama-2-7B FP16, block size 64 and k=2 fill that allocation exactly.

The native scheduler's block IDs remain stable. Shrinking removes only free
blocks, evicts their prefix hashes through the native BlockPool, and compacts the
physical allocation dictionary without copying live KV. Address handles provide
the indirection to retain active high-numbered IDs. Expansion restores disabled
IDs before appending IDs. Scheduler capacity and its admission watermark follow
the new common budget. Worker expansion is staged before changing scheduler
state; a failed prepare releases staged blocks. A failed destructive commit
keeps scheduling paused.

The intermediate budget uses the union of current and target layers, incoming
weight size and actual stacked allocation size. The minimum feasible budget
across workers is applied before preparation; shrinking below active occupancy
is rejected. Post-commit resizing restores the feasible budget, capped by an
explicit `num_gpu_blocks_override` when supplied.

All migration allocations remain owned by the transaction until synchronization
and commit/rollback. Scheduler frees and reuse during inference do not deallocate
physical stacks. Newly written reused slots are patched under their global
block IDs. Source allocations are retired only in complete layer stacks after
all workers have committed. Retained layers keep their original cache handles.

### Streaming and communication ordering

The migration communicator is bootstrapped once with vLLM's existing CPU group;
its NCCL unique ID/communicator is independent of PP forwarding. Migration uses
priority 0; the inference stream uses priority -1. There is one sender/receiver
pair for this two-GPU configuration.

The sender acquires its device mutex and sends ACK over TCP. The receiver tries
its mutex without blocking and rejects if inference owns it or has an incomplete
CUDA communication event. On REJECT, the sender releases and retries after 1 ms.
On ACCEPT, both enter the migration communicator. GPU completion and receiver
application precede APPLIED and mutex release.

MRv2 also uses a sibling NCCL group to return sampled tokens. Its operations and
PP forwarding share the same mutex/completion reservation. A reservation remains
busy until its CUDA event completes; inference can enqueue computation while
migration cannot enter. This preserves asynchronous worker execution while
serializing GPU communication across all groups used by this configuration.

The source snapshots at most 16 MiB per transfer. Under the cache lock, gathering
waits for prior source writes and records a gather-complete event. The next
inference step waits for that event before writing. The lock is released before
network transfer, allowing computation to overlap an immutable staging payload.
The receiver scatters into detached incoming caches on the migration stream.

### Dirty tracking and commit

After a real inference step, physical `slot_mapping` entries mark the bitmap;
padding is ignored. Native copy-on-write block copies mark their entire
destination blocks. Copies do not advance the scheduled-token count. Dirty
extraction/reset is atomic. Every patch round covers all moving layers and is
acknowledged after its final chunk is applied.

Progress uses migration-relative cumulative token watermarks. A watermark is
advanced only after every chunk/layer of its snapshot is acknowledged, so
coalesced writes cannot be mistaken for newly applied tokens. The initial
historical transfer must finish before the 50-token convergence threshold is
used. After keep pause drains pipeline batches, the final dirty set is applied,
STOP is acknowledged, and both background threads are joined. Only then can
weights/KV readiness permit commit.

The controller checks that active request IDs, computed-token counts, output
IDs, preemption counters and block mappings survive the final pause unchanged.
`verify_kv=true` additionally hashes every allocated migrating layer block on
both devices before commit. This expensive check is excluded from performance
runs. Errors before release roll back and resume; errors after release begins
retain the keep pause.

## Explicit integration assumptions

- The paper does not provide a vLLM 0.29.0 integration. Its address-based
  PagedAttention mechanism is implemented as a specialization of the existing
  Triton attention kernels. Standard allocation/binding interfaces are reused.
- PyTorch owns rounded stack allocations; this implementation does not add CUDA
  VMM remapping. Returned memory can remain in PyTorch's reusable allocator
  cache, which is included in feasibility accounting.
- Keeping native logical IDs stable adapts the paper's pointer compaction to
  vLLM's BlockPool ABI. Freed address entries become zero; active entries retain
  their physical pointers.
- Uniform full-attention layers write the same slot set per step, so one bitmap
  per GPU pair represents that set for all migrating layers.
- Literal received-slot counting is ambiguous when dirty bits coalesce writes
  or slots are reused. Snapshot watermarks provide a conservative applied-token
  boundary while preserving the paper's drain/patch/convergence procedure.
- Initial history includes every allocated block, including reusable prefix
  cache and currently free blocks. This conservatively preserves cached history;
  bytes reported include those blocks rather than only active-request KV.
- Weight loading reuses the base loader's local-checkpoint fallback from §6.2.
  Startup CPU weight preloading is outside this KV reproduction. Total migration
  time includes the fallback's I/O and staging.

## Running the comparison

On a host with sufficient shared memory, use the existing environment:

```bash
MOLINK_TEST_MODEL=/workspace/model/llama7b \
MOLINK_TEST_NAME=pipelive-llama7b MOLINK_TEST_PARTITION=20,12 \
bash benchmarks/pp_reconfigure/serve_pipelive.sh \
  --dtype float16 --block-size 64 --gpu-memory-utilization 0.48

.venv/bin/python benchmarks/pp_reconfigure/compare_pipelive.py \
  --model pipelive-llama7b --boundaries 18 20 --iterations 3 \
  --output /workspace/molink-reconfiguration-results/pipelive/llama7b-performance.json
```

The benchmark launcher enables vLLM's existing `VLLM_BATCH_INVARIANT=1` for exact
multi-request token comparisons; see `docs/features/batch_invariance.md`. Normal
batch-dependent floating-point results otherwise drift even across repeated
static runs. This is a benchmark setting, not a change to the migration algorithm.

Run once with `--verify-kv` for full source/destination KV hashes. Enable
`HETVLLM_PIPELIVE_PROFILE=1` on the server for a separate CPU/CUDA trace capture;
`sync_live` writes `gpu-stageN.json` in each transaction directory. Do not mix
profile/hash timings into the unprofiled performance table.

Development endpoints (existing opt-in router):

- `POST /molink/reconfigure`: `{"boundary":18,"mode":"async"}`;
  `mode="sync"` gives the paired stop-the-world baseline on the same cache/transport.
- `POST /molink/reconfigure`: add `"verify_kv":true` for hash validation.
- `POST /molink/resize_kv`: `{"num_blocks":48}` then `64` exercises shrink/expand.
- `GET /molink/monitor`: request state, layer placement and physical KV allocation.

The comparison starts three requests, triggers after all produce tokens, checks
192 tokens per request against the static baseline, verifies unchanged request
state across commit and zero preemptions, and exercises resizing during live
requests. GPU 0→1 is the overlap acceptance direction; the reverse restores the
initial layout and checks correctness. Reverse source-stage compute may have no
overlap because it depends on incoming activations from the earlier stage.

## Validation and experiment artifacts

Hardware: two RTX 3090 Ti, 24 GiB each, driver 535.261.03; vLLM
`0.29.0+molink.cu129`. Llama-2-7B-chat FP16, PP `20/12 → 18/14`, moving zero-based
layers 18–19, 64 blocks of 64 tokens, k=2, TP=DP=1, eager Triton attention.
The original server on port 26010 remained running; experiments use port 26020.

Stages were checked independently. Logs are under
`/workspace/molink-reconfiguration-results/`:

| Stage | Evidence |
| --- | --- |
| Base | `pipelive-baseline-tests.log`: 29 tests |
| Address layout | `pipelive-stage1-tests.log`: 31 tests, exact dense/address prefill and decode after resize |
| Bitmap/transport | `pipelive-stage2-tests.log`: 35 tests, dirty overwrite/drain, scatter, stack release and receiver rejection |
| Resize/control | `pipelive-stage3-tests.log`: 36 tests |
| Native attention regression | `pipelive-stage4-tests.log`: 38 tests including two existing attention reference cases |
| Failure paths | stages 5–7 and `pipelive-final-unit-tests.log`: optional copy metadata, CUDA admission reservation, prepare failures and scheduler resize recovery |
| Final checks | `pipelive-final-tests.log`: 49 tests including the two original attention cases; `pipelive-precommit-final.log`: required hooks, mypy and accelerator API checks |
| Paired performance | `pipelive/llama7b-performance.json`: 12 successful migrations, three forward and three reverse per mode |
| Hash/trace validation | `pipelive/llama7b-profiled-verified.json`: four successful migrations, exact tokens and full KV hashes in both directions |
| Final-code acceptance | `pipelive/final-e2e.json`: three active requests, 192 exact tokens each, live 48→64 block resizing and equal source/destination KV hashes |
| Report | `pipelive/report/report.md`, `summary.json`, `comparison.svg`, `comparison.png` |
| Worker/control data | `pipelive/migrations/<run-id>/`: phase JSONL, source patch records, convergence, before/after states, final transaction and exact token arrays |
| Tooling | `pipelive-precommit.log`; relevant pre-commit hooks and Ruff |

This container has a 64 MiB `/dev/shm`, partly occupied by the original service.
The experiment-only `pipelive/environment/` artifacts contain the exact
`sitecustomize.py` and `nccl_shm_redirect.c/.so` used to run natively: the Python
shim caps CPU ring chunks at 64 KiB (native overflow fallback remains available)
and skips an unused optional TileLang JIT-monitor import that hung locally; the
preload redirects only NCCL's shared-memory files to
`/workspace/molink-runtime/pipelive-shm`. It does not replace NCCL or KV transport.
NCCL used its registered-host SHM transport; these measurements do not represent
the paper's A100/L40S InfiniBand testbed. Full setup/failure logs are retained
alongside the results. No source workaround for this container was added to vLLM.

Reproduce this container setup by compiling the recorded C shim with
`gcc -shared -fPIC -O2 ... -ldl`, then setting `PYTHONPATH` to that environment
directory and `LD_PRELOAD` to its `.so` when invoking the launcher above.

## Metric definitions and current result

The three unprofiled GPU 0→1 trials give:

| Mode | Total mean ms | Pause mean ms | NCCL mean ms | Payload mean MiB | Forward-span overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sync | 465.17 | 322.57 | 35.94 | 128.00 | 0.0% |
| Async | 942.98 | 38.73 | 32.02 | 128.72 | 23.4% |

Pause decreases 88.0%. Background migration takes longer overall on this setup,
while continuing inference; the measured cutover has not reached the paper's
reported sub-10-ms pause.

Total runs from validation through resume, including preparation and cleanup.
Pause runs from requesting keep pause through resume, including pipeline drain.
Communication is source CUDA send-event duration and actual payload bytes; CPU
handshake, gather and receiver scatter are outside that duration. Forward-span
overlap is the sum of intersections of source forward-event and migration-send
intervals, divided by send duration. It includes gaps between inference kernels;
CUPTI kernel analysis is recorded separately and excludes those gaps.

The separate profiled GPU 0→1 run measured 27.996 ms of migration NCCL kernels
and 0.220 ms intersecting actual inference kernels: **0.78% kernel overlap**.
This is substantially below the event-window ratio and must not be presented
as 23.4% simultaneous kernel execution. The raw trace is
`pipelive/migrations/46923a8faca84411b60f47e028036b46/gpu-stage0.json`;
`analyze_pipelive.py` identifies the migration stream from its slot-gather
kernels and intersects its NCCL kernels with the attention/compute stream.
The trace has 7 inference steps and records the current deterministic
batch-invariant kernel configuration. Profiler and hash overhead are excluded
from the performance table.

## Visual validation experiment

`compare_pipelive.py` records client token-arrival timestamps and snapshots of
physical KV allocation before/after online 64→48→64 block resizing. Each trial
keeps three requests active, migrates layers 18–19, compares all 192 output tokens
per request against a static baseline, and checks request/block continuity.
`visualize_pipelive.py` renders token delivery, pause costs, physical KV resizing,
and source/destination token watermarks from the original transaction logs.

After starting the server with the Llama-2-7B command above, run sequentially:

```bash
OUT=/workspace/molink-reconfiguration-results/pipelive/visual-20261106
.venv/bin/python benchmarks/pp_reconfigure/compare_pipelive.py \
  --model pipelive-llama7b --boundaries 18 20 --iterations 3 \
  --output "$OUT/comparison.json"
.venv/bin/python benchmarks/pp_reconfigure/compare_pipelive.py \
  --model pipelive-llama7b --boundaries 18 20 --iterations 1 --verify-kv \
  --output "$OUT/verification.json"
.venv/bin/python benchmarks/pp_reconfigure/visualize_pipelive.py \
  "$OUT/comparison.json" --verification "$OUT/verification.json" \
  --warmup-iterations 1 --output-dir "$OUT/report"
```

The first complete iteration (both modes and directions) warms transfer kernels
and is excluded from timing means. The remaining two iterations provide paired
forward measurements; reverse migrations restore the initial placement and
check correctness. Online resizing occurs in the first iteration and is shown
separately from the measured migration pair. Full KV hash validation is a separate
run and is excluded from performance figures.

The renderer asserts that every request delivers tokens during background
migration, nonempty incremental patches are applied, final token lag is zero,
physical KV allocations shrink and recover, active block mappings survive
resizing, output tokens match the static baseline, and verification hashes agree.
Client timestamps measure streamed token delivery and can include HTTP buffering;
they do not measure simultaneous GPU kernel execution. KV allocation values
measure live stack tensors, rather than PyTorch's reserved allocator memory.

### 6 November 2026 repeat

Artifacts: `/workspace/molink-reconfiguration-results/pipelive/visual-20261106/`.
Twelve two-layer migrations passed token continuity checks; four separate
hash-verified migrations also passed. After excluding one full warmup iteration,
the two forward trials per mode measured 548.11/1494.21 ms total and
375.42/194.23 ms pause for sync/async, respectively: a 48.3% pause reduction.
Online resizing changed GPU 0 KV allocations 1280→960→1280 MiB and GPU 1
768→576→768 MiB while preserving active block mappings and zero preemptions.

In the illustrated async trial, each request delivered three tokens between
background migration start and the final pause request. Its two nonempty patch
rounds were applied during pipeline drain; zero rounds completed before the
pause request. This demonstrates incremental synchronization and final
consistency, but does not establish continuous patch application during normal
inference for this workload. The benchmark reports measured overlap, including
zero, separately from its token/state correctness assertions.

A supplementary eight-layer trial passed its synchronous forward migration to
boundary 12, but reverse migration to boundary 20 failed capacity validation:
GPU 0 reported zero feasible intermediate blocks. That trial is incomplete and
excluded from the performance summary. Its cause remains unresolved; see
`limitations.md` and `patch-demo.log` beside the artifacts. No runtime migration
code was changed for these experiments.
