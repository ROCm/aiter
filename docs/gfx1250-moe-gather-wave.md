# gfx1250 MoE gather route-metadata cache

## Change
Read all top-k row indices and weights in one coalesced load per wave, then
broadcast their values with `readlane` before output-tail divergence.
This removes the shared-memory metadata staging and workgroup barrier on the
selected path. FP32 split and route summation order, per-row 64-bit resource
bases, dropped-route handling, dead-tail output contract and BF16 rounding are
unchanged. No atomics or persistent state are introduced.

The host selects this variant only for gfx1250, M=4..256, D=6144, topk=5,
BF16 output, FP32 route weights and split_k=1. All other contracts retain the
LDS implementation. This is a metadata-distribution change, not a GEMM2 fusion.
A Boolean enters the factory cache key, not the exact token count.
AOT replay respects the selected build target.

## Measured gather-only result
C5 GPU0, gfx1250 / 256 CUs, FlyDSL 0.5.0, PyTorch
2.11.0+rocm10.1.0a20260811-bkc.20260820. The validated M3 baseline was AITER
55cde6a0c49cc04f39f6141e312938a8c48e61fb and ATOM
7b7e7961451a8d9dddac9574d855efea6754ecfe. The gather source at the recipe
baseline was byte-identical to AITER main f98317425ca8710aa6880a88079a1d60819e5c60.

Three independent processes, four disjoint source buffers, production vec
selection, 16 calls per graph, 101 randomized A/B/A control blocks.
Each number below is a median of process medians; the range is over paired
process speedups. A/A is a second independent LDS graph. Source values are
synthetic, not captured model activations.

| M | Vec dwords | LDS (us) | Wave (us) | Paired speedup range | Max A/A delta (us) |
|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 1.7789 | 1.7101 | 1.040–1.043x | 0.0125 |
| 12 | 2 | 1.8090 | 1.7401 | 1.040–1.043x | 0.0026 |
| 16 | 2 | 1.8128 | 1.7401 | 1.036–1.045x | 0.0050 |
| 40 | 2 | 1.8803 | 1.8052 | 1.040–1.042x | 0.0100 |
| 64 | 2 | 1.9392 | 1.8502 | 1.045–1.048x | 0.0151 |
| 128 | 4 | 1.9742 | 1.8603 | 1.061–1.062x | 0.0076 |
| 256 | 4 | 2.2947 | 2.2234 | 1.031–1.041x | 0.0100 |

The final checked-in benchmark also ran on the PR worktree with random
non-dyadic source values and exact before/after equality. One reproduction
(seed 10999, 31 samples) measured M4 1.8728/1.7902 us, M16
1.9154/1.8102 us and M128 2.0330/1.9805 us (first LDS graph / wave).

## Important: not an end-to-end performance claim
A public `fused_moe` A/B on the 11-10 M3 recipe kept route/quant fusion,
GEMM1/GEMM2 and the existing 129-expert/top-5 configuration unchanged.
It used four disjoint weight-buffer sets (3.88 GB), synthetic activations,
top-4 routed experts plus shared expert 128, and changed-input/padded-route
graph checks. All outputs matched exactly. Whole-MoE timing differences
were mostly within noise, and some broad candidate cases regressed. This
is why dispatch is bounded and this PR does NOT claim a whole-MoE win.

No TP4 server, AgentX throughput/TTFT/TPOT or GSM8K comparison was performed
for this patch. The existing recipe's split-K consumer fusion is retained,
not reimplemented. A serial-read wave-cache experiment was slower and was
discarded. Persistent-worker grid trimming was also inconclusive and is
not included. Do not add up standalone percentages into a model speedup.

## Validation
- 89 tests on the PR/main worktree: output and route weight dtype combinations,
  split slices, top-k up to 32, tail columns, dropped routes with NaN-poisoned
  backing rows, graph/live-row changes, concurrent streams, >4 GiB source-row
  addressing, retained LDS path and dispatch boundaries.
- 18 additional random-input public-wrapper A/B cases against the exact old
  kernel, including M257/M512 fallback and changed-input graph replay: byte exact.
- Broader direct old/new harness: 70 cases with exact dyadic CPU-order reference.
- Compile-only AOT: gfx1250 wave path and gfx950 retained LDS path passed.
  gfx950 compilation is not gfx950 hardware execution.
- Recipe route/quant/padding regression control: BOTH baseline and candidate
  154 passed / 1 failed. The same pre-existing M4536 test requires the kernel
  `moe_route_lds_psum_sc`, but its current tuned A-preshuffle path uses other
  routing kernels. Numerical equality passes before that name assertion.
  This known failure was neither suppressed nor declared fixed.
- Black, Ruff and `git diff --check` passed.

## Reproduce
From a matching AITER checkout on gfx1250:

    python -m pytest -q op_tests/flydsl_tests/test_moe_gather_wave.py
    python op_tests/flydsl_tests/bench_moe_gather_wave.py --seed 10201
    python op_tests/flydsl_tests/bench_moe_gather_wave.py --seed 10202
    python op_tests/flydsl_tests/bench_moe_gather_wave.py --seed 10203

The benchmark prints all samples as JSON, checks exact output equality and
includes its own A/A graph. It times only gather, with rotated sources and
fixed route maps. The working set may fit cache for small M; it is not a
simulation of 60 layers or a cold-weight serving benchmark.

The public dispatcher preserves the old path at M>256; explicitly invoking
the benchmark's wave variant outside the measured range does not change
production dispatch.
