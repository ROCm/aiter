# gfx1250 BF16/FP16 unified TDM GEMM

An explicit FlyDSL entrypoint, **not a change to the existing GEMM dispatcher or
tuning tables**. Supports BF16 and FP16 input/output; FP8 PTPC is outside this PR.

~~~python
from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm

# A: [M,K], weight: [N,K]. B is a logical [K,N] transpose view.
y = flydsl_gemm_a16w16_tdm(
    a, weight.t(), block_m=256, block_n=256,
    block_k=128, m_waves=2, n_waves=2, stages=2, group_m=4,
)
# Small output grids / long K: opt into low-precision atomic split-K.
y = flydsl_gemm_a16w16_tdm(a, weight.t(), split_k=4)
~~~

## Unified schedule

One `@flyc.kernel`, one launch. Both compile-time schedules share setup, TDM
operands, C-shuffle and output handling; there is no second HTI kernel.
All waves issue disjoint TDM slices through the same operand/partition helper.

The hot loop explicitly unrolls **two consecutive K tiles**. Double buffering
uses literal LDS ring slots 0/1; odd trip counts and deeper rings have a remainder
and final drain. The NT interleaved schedule follows wave-parity serpentine
quadrants:
- even: Q00 -> Q01 -> Q11 -> Q10;
- odd: Q00 -> Q10 -> Q11 -> Q01.

Independent second-half LDS reads are issued among current-quadrant WMMAs.
The last two quadrants prepare the next K step's first-half fragments.
At tile boundaries, TDM completion and all prior LDS reads precede ring reuse.
Scheduling hints request WMMA/DS interleaving; they do not promise fixed cycles.
`interleave=False` uses a simple schedule in the same kernel, including NN B
and small wave tiles that cannot support quadrant interleaving.

A/B and C-shuffle rows have **16-byte LDS padding**, eight half elements.
Grouped M ordering is applied at cluster granularity, not to split-K rank:
all members address the same output tile. The helper describes grouped cluster
coordinates then row-major M/N ranks; this entrypoint uses M/N cluster 1x1 and
K cluster size 1..9. It does not implement M/N multicast.

## Packed atomic split-K (explicit-call default)

Rank 0 clears its output tile. Agent release, store wait, cluster barrier and
acquire fence precede GEMM. After C-shuffle each rank issues four packed
two-element atomic additions per 128-bit read group, directly into C.
No FP32 partial workspace, persistent counter, or separate zero/reduce kernel.
Optional bias is added once by rank 0 after a second release/barrier/acquire
sequence. Existing C contents are overwritten, not accumulated.

Within a split, WMMA accumulates in FP32. **Each partial is rounded to BF16/FP16
before an atomic addition to BF16/FP16 C.** Arrival order is nondeterministic.
This is NOT the previous FP32-workspace reduction:
- rounding can destroy a small residual under cancellation;
- FP16 partials can overflow even when the final FP32 sum would be finite;
- split-K is rejected when deterministic algorithms are enabled.

Use split_k=1 for deterministic accumulation / FP32 cancellation behavior.
Tests explicitly demonstrate cancellation and partial overflow; these are
documented numerical limitations, not concealed by loose random-input checks.

## Constraints

BF16 or FP16 only. K divisible by split_k*block_k, with at least `stages`
full tiles per split. Stages 2..16, K tile 32/64/128/256/512, at most 32 waves,
at most 320 KiB LDS. Interleaving requires NT and sufficiently large even
quadrants. Leading strides and tensor pointers are 16-byte aligned.
Padded/offset views and M/N edges are supported.

Odd N is supported for split_k=1 or `split_k_policy="workspace"`. Packed atomics cannot safely touch a
missing high half. Tensor footprints/grouped-swizzle arithmetic are bounded.
The wrapper records tensor lifetimes on the requested stream, but callers must
order external producer streams normally. Concurrent writes to the same output
from unrelated streams are not supported.

## Reproduce

    python op_tests/test_gemm_a16w16_tdm.py
    python op_tests/bench_gemm_a16w16_tdm.py --dtype bf16 --shape 8192,8192,8192
    python op_tests/bench_gemm_a16w16_tdm.py --dtype fp16 --shape 256,256,8192

C5 gfx1250, FlyDSL 0.5.0, control-host date 2026-11-12; C5's clock is behind.
95 tests passed in an isolated cold compile cache before adding the final
memory fences, and all 95 passed again afterward. The final suite extends this
to **102 passed**, adding non-square serpentine tiles and a 16-stage small tile. Coverage includes BF16/FP16,
split-K 1..9, odd/even K tile counts, stages 2..5/16, NN, edge and offset buffers,
bias, changed graph inputs, concurrent streams, argument guards, cancellation
and overflow. ISA inspection confirms buffer_atomic_pk_add_bf16/f16 and the
cluster synchronization. Finite coverage is not proof for every configuration.

## Initial revision performance (vector-store epilogue)

Explicit release/acquire fences included. Exclusive GPU, four calls per graph,
31 randomized rounds. Same 256x256x128 tile, 2x2 waves and two buffers:

| dtype | M=N=K | simple us | interleaved us | interleaved TFLOPS |
|---|---:|---:|---:|---:|
| BF16 | 4096 | 101.52 | 72.79 | 1888 |
| FP16 | 4096 | 104.77 | 74.87 | 1836 |
| BF16 | 8192 | 750.29 | 530.03 | 2074 |
| FP16 | 8192 | 773.36 | 554.86 | 1982 |

Separate same-tile split sweep, 128x128x128, eight calls per graph:
BF16 M=N=256,K8192: 37.77 us (split1) -> 14.09 us (split8).
FP16: 37.57 -> 14.25 us. BF16 relative L2 vs FP32 grows from about 0.00166
to 0.00401; FP16 from 0.000207 to 0.000505.

At M=N=1024,K8192, split8 is slightly slower than split1. At 8192 cubed it
is much slower. Split-K is opt-in, not universally faster or numerically
equivalent. Timing before the explicit fences is not the release benchmark.

The existing explicit FlyDSL backend was also measured using its untuned
default arguments, not a best-tuned AITER baseline. Local torch.mm was unusually
slow in this bring-up image; no general rocBLAS speedup is claimed. No LLM
serving / TTFT / TPOT improvements are claimed for this standalone GEMM.


A rerun through the public benchmark command measured:
BF16 8192 cubed: simple 760.66 us, A/A 766.83 us, interleaved 535.26 us
(**2054 TFLOPS**); FP16: simple 786.22 us, A/A 795.58 us, interleaved 548.09 us
(**2006 TFLOPS**). These are local measurements, not peak hardware claims.
Raw samples and the split sweep are in `gfx1250_tdm_gemm_results.json`.


## TDM output-store regression correction

The initial unified rewrite replaced the old no-bias, non-split TDM output
store with a vector-store epilogue. An old/new same-device comparison exposed
a regression that was not visible in the simple/interleaved-only benchmark.
For split_k=1 and no bias, C-shuffle now uses a TDM store through the common
operand helper again. Bias and packed atomic split-K paths are unchanged.

BF16 8192 cubed, 256x256x128, stages2, same C5 GPU and FlyDSL0.5.0:
- an ablation at 2x2 waves/group_m0 measured 556.38 us (vector stores) versus
  441.48 us (TDM stores), with bit-identical outputs;
- separately restoring quadrant fences or scheduler hint counts did not
  reproduce that gain;
- final old/new graph comparison: old 448.66 us, fixed unified 450.65 us at
  2x2/group_m0; old 490.93 us, fixed 479.70 us at 2x4/group_m0.
  This reverses the earlier apparent recommendation of 2x4 over 2x2.

The historical 2.79 PFLOPS was real: old all-wave HTI on f01/FlyDSL0.4.1,
2x2 waves, group_m0, three rotating-input runs, median 394.30 us.
With the same 10-buffer/500-warmup/100-profiler-event procedure on current C5,
three old-source run means were 431.51/441.25/440.41 us; fixed unified means
444.54/446.39/450.63 us (about 2.46 PFLOPS at the median).
Both use split_k=1, so packed atomic reduction is NOT the cause.
Different hardware/runtime/compiler conditions remain a confound; the residual
historical gap has not been attributed to clocks or compiler without evidence.
Do not present the historical 2.79 PFLOPS as a current C5 measurement.

The correction passed the 102 existing tests plus four BF16/FP16, 2x2/2x4
TDM-store graph-replay cases with odd N and output padding guards.

## Offline tunable reduction policy

The kernel now also supports `split_k_policy="workspace"` with FP32 partials
and fixed-order FP32 cluster reduction. `TdmTunedPolicy` loads exact-contract
results from the AITER offline tuner; the tuner defaults to this safer strategy
plus split1, with low-precision atomic candidates only under explicit opt-in.
See [tuning and runtime usage](gfx1250_tdm_split_policy.md).
Earlier performance sections above describe the packed-only revisions; the
policy document and its separate result file describe the added strategy.
