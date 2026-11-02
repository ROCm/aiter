# Q-A/KV projection + weighted RMSNorm: assembly optimization

Work began October 19, 2026, 20:07 America/Chicago. Requested stopping deadline:20:55 (October20,01:55 UTC). GPU work finished before the deadline; completion metadata is in `STATUS.json`.
Final accepted choice: BF16 WMMA, FP8 storage, BN128, split-K16, four waves, 192 producer CTAs, one fused launch. The optimized two-launch alternative is retained because it can be better for large weight working sets.

## Final results — second measurement round only

Median graph elapsed time per complete region invocation, microseconds. Each row compares variants in the same benchmark pass.

| Weight working set | Original two-launch A | Old fused G | Optimized two-launch | Accepted fused split16 | Split16 latency reduction vs A |
|---|---:|---:|---:|---:|---:|
| Single layer (6 MiB) | 4.676 | 5.247 | 3.343 | 2.733 | 41.6% |
| 16 copies (96 MiB) | 4.446 | 5.238 | 3.221 | 3.154 | 29.1% |
| 128 copies (768 MiB) | 5.420 | 6.785 | 3.683 | 3.730 | 31.2% |
| All 43 actual layer weights | 5.596 | — | 3.935 | 3.841 | 31.4% |

Split32 was also validated but was worse in the final working-set tests (2.858us single-layer, 3.964us at 768 MiB, 4.117us across actual 43-layer weights). Split16 remains the default. The optimized separate path narrowly beats the fused path at 768 MiB; fusion is not a universal win. These are isolated region timings, not model throughput.

## Scope and baseline

Single GPU gfx1250, concurrency 1, Flash layer-0 checkpoint. Q-A: 4096→1024; KV: 4096→512. Input FP8 quantization is outside the timed region. Output is FP32 weighted RMSNorm of the BF16-rounded projection results. No Q-B, final per-head Q norm, RoPE, attention, MoE, or serving benchmark is included.

The baseline is the isolated controlled WMMA implementation from `qkv-atomic-ab`, preserved under `baseline/`. It is **not** a claim about the native AITER/ATOM serving GEMM decomposition. Baseline A: FP8 storage→BF16 operands→WMMA FP32 partials→separate split-K reduction and weighted norm. Baseline G: the previous atomic fused implementation with per-tile and per-group completion, sumsq publication, and in-kernel reset.

## Assembly findings and changes

- Original operand-layout conversions move both operands through LDS every K=128 iteration. The generated loop has four barrier pairs and 32 LDS load instructions before four WMMA instructions. Split-K=8 executes that loop four times. The final output conversion adds another LDS round trip.
- Direct linear equivalents of the WMMA dot-operand layouts remove operand LDS shuffles and K-loop barriers. Direct loads alone were slower with device-scope uncached reads; the load pattern and caching must be tuned together.
- `scaled_upcast` emits gfx1250 `v_cvt_scale_pk8_bf16_fp8`, replacing FP8→FP32 conversion, FP32 scale multiplication, and FP32→BF16 conversion. The accepted math still uses BF16 WMMA with FP32 accumulation.
- Cached `.ca` reads are used for immutable projection operands. Cross-CTA accumulator communication retains device scope. This distinction matters: coherent L2 does not remove the need to finish writes before signaling completion.
- The baseline 32-column/four-wave WMMA tile duplicates work across waves. Two waves for 32 columns, or four waves for 64/128 columns, avoid that duplication. The useful batch dimension remains one row; other matrix rows are computational padding.
- The new fused protocol uses one completion counter per normalization group. Every producer atomically accumulates its FP32 partial outputs, drains memory operations, and signals its group. The last producer normalizes the entire completed group. There are no per-tile completion counters or sumsq mailboxes.
- Reformatting accumulator outputs reduces the number of FP32 atomic instructions. Some LDS traffic remains for this output conversion and normalization; the accepted kernel is not LDS-free.
- The last producer resets its accumulator region and group counter in the same invocation. Reset time is included. Scratch must be initialized before first use, must not be shared by overlapping invocations, and is reused sequentially on one stream.

## Synchronization ownership

Each `(output tile, K split)` CTA exclusively computes a partial. FP32 device-scope atomics accumulate into `Acc[1536]`. After `s_wait_loadcnt 0`, `s_wait_storecnt 0`, and a CTA barrier, the CTA increments its group counter using a relaxed device-scope atomic. Only the last ticket holder loads the completed group, rounds to BF16, computes FP32 sumsq and weighted norm, writes Y, and resets that group's Acc/counter. Q and KV use disjoint output regions and separate counters, padded 128 int32 entries apart. No polling and no full-grid residency requirement.

## Correctness

The expanded validation contains 268 inputs: the saved actual serving input, eight embedding-derived inputs, zero/constant cases, and 256 reproducible random or perturbed inputs at different scales. Independent reference: FP64 projection, BF16 rounding, FP32 normalization. Each candidate was replayed 2,144 times with changing inputs; **every output** was checked. The 0.001 relative-RMS threshold was unchanged, and Q and KV were also checked separately.

Initial accepted BF16 candidate: maximum combined relative RMS 0.000048119 (0.004812%); worst Q relative RMS 0.000172005 (0.017201%); all outputs passed; counters and accumulators returned to zero.

Native FP8 WMMA variants passed the original nine cases but **failed expanded validation**: maximum combined RMS 0.001472367 (0.147237%), 336 failing outputs out of 2,144. Both externally applied block scaling and native hardware scaling are rejected. Their timing results are experimental only.


Final launcher validation (`final-validation/results.json`): both split16 and split32 passed all 268 input cases × 8 changing-input passes, checking every output. Q and KV groups were also checked separately.

Checkpoint-wide check (`layer-validation/results.json`): all 43 layers × 25 inputs × 2 passes × 2 split choices passed (4,300 checked outputs). The saved `layer-fixtures/` contains all tested region weights.

Long stress (`long-stress/results.json`): **100,000 complete split16 invocations checked**, using1,000 distinct layer/input pairs drawn from 43 layers and 268 available inputs. Every invocation output was checked on every graph replay; all passed. Maximum combined relative RMS was 0.000770997; maximum Q 0.000626630; maximum KV 0.000775803. Accumulators and counters returned to zero after each 1,000-invocation graph. Atomic summation order is nondeterministic; this test validates the existing error threshold, not bitwise identity.

## Experiments retained

- `sweep1`: tile width, split-K 1/2/4/8/16, direct operand layouts.
- `sweep2b`: scaled conversion, replicated padding rows, wave count, K-block size.
- `sweep3`: read caching, scalar scale loads, offline tiled weight packing.
- `sweep4`: single group-completion protocol, scratch versus FP32 atomics.
- `sweep5`: K/N tile and split sweep.
- `sweep6`: offline BF16 weight expansion; slower, doubles stored weight bytes.
- `sweep7`: atomic output layout and norm vectorization.
- `sweep8`: padded group counters and native FP8 experiment.
- `sweep9`, `sweep9b`: unrolling and native FP8 scaling; rejected for accuracy later.
- `sweep10`: bulk TDM loads into LDS, padded/unswizzled layouts. Unsupported swizzled TDM layouts are recorded as compile failures.
- `sweep11`: one-wave masked normalization; fewer barriers, no convincing latency win.
- `sweep12`: larger tiles and eight-wave CTAs.
- `sweep13`: parallel 2-D scratch reduction. Generated many barriers; slower.
- `sweep14`: sequential vectorized scratch reduction, occupancy hints and ILP scheduling.
- `sweep15`: caching/vectorization for normalization weights.
- `sweep16`: leader-wave atomic publication, with/without the otherwise required CTA barrier; no win.
- `sweep17`: offline K128-major weight packing; slower. Its experimental kernel is preserved as `packed128_experiment.py`; the measured accepted `fused.py` was restored byte-for-byte.
- `cache-bench`: 1/16/64/128 distinct weight copies (6/96/384/768 MiB working sets). These are working-set sensitivity tests, not a guarantee of fully cold caches or a serving simulation.
- `prefetch-bench`: normal and speculative gfx1250 TDM L2 prefetch. Both slower than no-prefetch in tested small/large working sets.
- `timeline`: instrumented per-CTA timestamps using `MSG_RTN_GET_REALTIME`, whose 100 MHz rate was queried from HIP. This is diagnostic instrumentation, not the source of headline latency.

The profiler's hardware-counter discovery utility could not load its packaged `librocprofv3-list-avail.so`; no hardware-counter attribution is claimed. ISA, CUDA-event graph measurements, torch traces, and device timestamps are the evidence available.

## Measurement rules

Each timing experiment captures many complete region invocations in a CUDA graph, warms up, measures two complete rounds, and reports round 2. Variant order alternates forward/reverse. All raw samples and both rounds are preserved. In-kernel per-invocation reset is counted. Initialization outside timed batches is only first-use/shared-scratch setup, not a substitute for reset between invocations. GPU process checks were performed before benchmark runs. No serving server was started.

## Limits and next steps

This establishes an isolated region optimization, not an ATOM throughput improvement. Before integration: validate on all target layer weights and live decode inputs, preserve the exact projection/rounding/normalization ABI, give each concurrent execution independent scratch, then compare one server startup and two full clients per configuration, reporting client 2. The separate earlier Flash routing/numerical acceptance issues remain unresolved by this region work.

## Saved artifact and software

Full directory: `/home/vigokhal/dsv4-monokernel/qkv-asm-opt/`. Archive: `/home/vigokhal/monokernel-handoff/qkv-asm-opt-20261020.tgz`, with adjacent SHA256 file. Code, all experiment outputs, generated ISA/LLVM, traces, compile cache, weight/input fixtures, and skills are included. The entire model checkpoint and Docker image are not embedded; the saved region fixtures make the recorded benchmarks independent of the model directory.

Test image: `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, imageID `sha256:d1434db7030a8355a332b2a9237598fadb07a8bf7bb6dbbd081567a13d54686a`. Exact Torch/Triton/device metadata: `final-benchmark/manifest.json`. The hashes of `accepted.py`, `fused.py`, `common.py`, `opt.py`, `device_scope_helpers.py`, and `final_bench.py` were verified against the measurement manifest after restoring the final experimental edit. Later documentation and optional validation harness updates are separately covered by the artifact manifest.

Read `ASM_ANALYSIS.md` for instruction-level details and `README.md` for ABI/reproduction. Included skills contain CDNA-specific guidance; do not apply its per-XCD/L2 assumptions blindly to gfx1250. No server, image publication, branch push, or serving integration was performed in this session.
