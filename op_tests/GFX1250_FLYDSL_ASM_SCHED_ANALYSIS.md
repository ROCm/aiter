# gfx1250 MX128 FlyDSL tuning and native ASM comparison

## Scope

This note records the current tuning state on
`perf/gfx1250-flydsl-waitcnt-align` for six DSV4 GEMMs with row-major or
A-preshuffled activations:

```text
M=512, N=6144,  K=7168
M=512, N=7168,  K=3072
M=512, N=7168,  K=16384
M=512, N=65536, K=1536
M=512, N=2048,  K=7168
M=512, N=8192,  K=1536
```

FlyDSL uses MX128 E8M0 scales. The native ASM comparison kernel uses MX32
scales, so its timings are useful scheduling targets but are not identical
input semantics.

## Final tuned profiles

| Shape (M x N x K) | Row-major A profile | A-preshuffled profile |
|---|---|---|
| `512x6144x7168` | `t128x256x128_mw2_nw2_nb4_sk2_cm4_cn2_fsk` | `t256x256x128_mw2_nw2_nb4_sk4_cm1_cn2_fsk_apre` |
| `512x7168x3072` | `t128x128x128_mw2_nw2_nb4_sk1_cm4_cn4` | `t128x128x128_mw2_nw2_nb4_sk1_cm4_cn4_apre` |
| `512x7168x16384` | `t128x256x128_mw2_nw2_nb3_sk2_cm4_cn2_fsk` | `t256x256x128_mw2_nw2_nb4_sk4_cm2_cn2_fsk_apre` |
| `512x65536x1536` | `t256x256x128_mw2_nw2_nb4_sk1_cm2_cn4_ps2` | `t256x256x128_mw2_nw2_nb4_sk1_cm2_cn2_apre_ps2` |
| `512x2048x7168` | `t128x128x128_mw2_nw2_nb4_sk4_cm2_cn2_fsk` | `t128x128x128_mw2_nw2_nb4_sk4_cm2_cn2_fsk_apre` |
| `512x8192x1536` | `t128x128x128_mw2_nw2_nb4_sk1_cm4_cn4` | `t128x128x128_mw2_nw2_nb4_sk1_cm4_cn4_apre` |

The full names in the tuned CSVs use the
`flydsl_mxfp8_128_bpreshuffle_compute_wmma_` prefix.

## FlyDSL result versus the previous config

These are GPU-only `rocprofv3` kernel durations. Each candidate had 20 warmup
launches and 100 measured launches; the reported value is the 5%-trimmed mean.
The previous and tuned candidates were measured in the same profiler run.

| Shape (M x N x K) | Previous config (us) | Tuned (us) | Improvement | Speedup |
|---|---:|---:|---:|---:|
| `512x6144x7168` | 13.9124 | 12.7167 | 8.59% | 1.094x |
| `512x7168x16384` | 23.5240 | 19.2532 | 18.16% | 1.222x |
| `512x65536x1536` | 22.0130 | 18.7708 | 14.73% | 1.173x |
| **Three-shape sum** | **59.4494** | **50.7407** | **14.65%** | **1.172x** |

A second run through the public tuned-config dispatch measured 13.0338,
19.7380, and 19.7946 us respectively. This confirms that both requested large
gaps are now in the 19-us class when selected from the checked-in CSV.

## Final production-default validation

The final post-review `rocprofv3` public-dispatch check exercised both A
layouts, including the strided-A-scale path. Constant data and scales passed
exactly on all six shapes. The 5%-trimmed GPU timings were:

| Shape (M x N x K) | Row-major A (us) | A-preshuffled (us) | A-pre reduction |
|---|---:|---:|---:|
| `512x6144x7168` | 14.5584 | 12.7881 | 12.16% |
| `512x7168x3072` | 8.2247 | 7.9607 | 3.21% |
| `512x7168x16384` | 23.7999 | 19.5389 | 17.90% |
| `512x65536x1536` | 21.5881 | 19.2641 | 10.77% |
| `512x2048x7168` | 8.1173 | 8.1512 | -0.42% |
| `512x8192x1536` | 6.1318 | 5.9491 | 2.98% |
| **Six-shape sum** | **82.4202** | **73.6521** | **10.64%** |

Two row-major defaults changed after two same-command baseline/candidate runs:

| Shape | Previous row default (us) | Selected row default (us) | Improvement |
|---|---:|---:|---:|
| `512x7168x3072` | 8.7148 | 8.2790 | 5.00% |
| `512x2048x7168` | 10.0823 | 8.1640 | 19.03% |

The mirrored A-preshuffle candidates for `512x6144x7168`,
`512x7168x16384`, and `512x65536x1536` were 4.76%, 1.05%, and 0.35% slower,
respectively, so their existing row-major defaults remain selected. The
`512x8192x1536` candidate was already the selected row-major profile.

For `512x2048x7168`, both final layouts now use a fused split-K=4 profile and
produce the final output in one dispatch. The old row-major default and the
old A-preshuffle fallback both required a separate reduction.

## Native ASM comparison

The native kernels are:

```text
row-major A:
_ZN5aiter47f8gemm_bf16_mxfp8fp8_BpreShuffle_256x256_4x2_psE

A-preshuffled:
_ZN5aiter48f8gemm_bf16_mxfp8fp8_ABpreShuffle_256x256_4x2_psE
```

The production comparison uses native ASM with split-K=1, so ASM writes the
final result in one dispatch and launches no reduction kernel. FlyDSL uses the
selected production default for each shape. Its `_fsk` profiles perform their
split-K reduction inside the GEMM dispatch and also return a final result in
one dispatch. Thus every number below is a complete GEMM with no external
reducer on either side.

Each value is the mean of four order-balanced profiler runs. Every run used 20
warmups followed by 100 measured dispatches, and each run's value is the
5%-trimmed mean of GPU-only `rocprofv3` durations. A positive gap means FlyDSL
is slower than native ASM; a negative gap means FlyDSL is faster.

| Shape | FlyDSL split-K | ASM apre (us) | FlyDSL apre (us) | FlyDSL vs ASM | ASM row (us) | FlyDSL row (us) | FlyDSL vs ASM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `512x6144x7168` | 4 | 21.4510 | 13.5796 | -36.69% | 24.7291 | 14.6396 | -40.80% |
| `512x7168x3072` | 1 | 11.3876 | 7.9908 | -29.83% | 13.2382 | 7.9341 | -40.07% |
| `512x7168x16384` | 4 | 46.4057 | 19.5583 | -57.85% | 51.6492 | 23.7318 | -54.05% |
| `512x65536x1536` | 1 | 16.1429 | 18.9654 | +17.48% | 18.3593 | 20.9724 | +14.23% |
| `512x2048x7168` | 4 | 22.1127 | 8.0643 | -63.53% | 25.5539 | 7.9750 | -68.79% |
| `512x8192x1536` | 1 | 7.7229 | 6.1920 | -19.82% | 8.7594 | 6.1662 | -29.61% |
| **Six-shape sum** | - | **125.2229** | **74.3504** | **-40.63%** | **142.2892** | **81.4190** | **-42.78%** |

A-preshuffling reduces native ASM time on every shape. It helps the FlyDSL
256x256 profiles, while the three 128x128 profiles are effectively neutral
within run-to-run noise:

| Shape | ASM A-preshuffle time reduction | FlyDSL A-preshuffle time reduction |
|---|---:|---:|
| `512x6144x7168` | 13.26% | 7.24% |
| `512x7168x3072` | 13.98% | -0.71% |
| `512x7168x16384` | 10.15% | 17.59% |
| `512x65536x1536` | 12.07% | 9.57% |
| `512x2048x7168` | 13.47% | -1.12% |
| `512x8192x1536` | 11.83% | -0.42% |
| **Six-shape sum** | **11.99%** | **8.68%** |

These are still scheduling comparisons rather than identical datatype
comparisons: native ASM uses MX32 scales and FlyDSL uses MX128 scales. Constant
unit scales make the mathematical correctness check identical, but production
scale grouping differs. FlyDSL wins five shapes and the aggregate for both A
layouts. The remaining target is `512x65536x1536`, where native ASM split-K=1
is 17.48% faster with A preshuffle and 14.23% faster with row-major A.

The formerly cited 17-us-class native result for `512x7168x16384` was a raw
split-K=4 partial-output kernel. It is not a complete production result without
a reducer. Native split-K=1 takes 46.4057 us with A preshuffle; the selected
one-dispatch FlyDSL result takes 19.5583 us.

## Correctness checks

- Constant data and constant scales produced the exact expected BF16 output
  for native ASM split-K=1 and both final FlyDSL layouts on all six shapes.
- The public dispatch check also passed row-major A, strided A scales, and
  A-preshuffled A for all six constant-data cases.
- On randomized data and automatic E8M0 scales, the two newly promoted
  row-major profiles (`512x7168x3072` and `512x2048x7168`) were bitwise
  identical to their A-preshuffled counterparts.
- Against the FP32-accumulating reference, fused split-K=4 has the expected
  BF16 reduction-order sensitivity: 5.66%-6.18% of elements can fall outside
  the test's 1% elementwise tolerance. The matching row-major and A-preshuffled
  profiles produce the same values, and no NaN, Inf, layout mismatch, or
  external-reduction dependency was observed.
- A direct MX32 compile-and-run smoke test also produced the exact expected
  BF16 output with the normal installed FlyDSL compiler.

## Scheduling changes retained

- The experimental MX32 VGPR-pinning path was removed because it depended on
  a nonstandard LLVM intrinsic that the normal FlyDSL compiler cannot lower.
  MX32 now uses the supported scheduling path, while the measured MX128
  production optimizations remain intact.
- MX128 `t256x256`, four-buffer kernels now use the native startup cadence:
  issue tensor loads 0-2, wait with two loads outstanding, seed the first LDS
  fragments, and then issue tensor load 3 before entering steady state. This
  ordering matches the startup sequence in the native A-preshuffle 4x2 ISA.
- The MX128 `t256x256`, split-K=1 path pins the refill barrier wait after four
  independent WMMAs. Without the scheduling fence LLVM placed the signal and
  wait back-to-back; native ASM separates them with three WMMAs. The extra
  MX128 slot was faster than matching native exactly.
- MX128 split-K=4 keeps the hardware wave-ID parity traversal; replacing it
  with logical parity was slightly slower.
- MX128 `t256x256`, four-buffer, split-K=1 uses one steady-state traversal. It
  removes the duplicate runtime branch for the short-K large-N profile and is
  the source change used by `512x65536x1536`.
- The final result does not include the rejected no-expert-scheduler, nb3,
  iterative-ILP, opposite-traversal, or split-K=4 single-path experiments.

### Native-startup checkpoint

The startup-cadence change was measured against the immediately preceding
branch state in paired `rocprofv3` runs. Each result below is the 5%-trimmed
mean of the target kernel dispatches from its run.

| Shape (M x N x K) | Previous schedule (us) | Native startup (us) | Improvement |
|---|---:|---:|---:|
| `512x6144x7168` | 13.2603 | 13.0983 | 1.22% |
| `512x7168x16384` | 20.1769 | 19.6449 | 2.64% |
| `512x65536x1536` | 19.7311 | 19.5575 | 0.88% |

All three dispatches passed the constant-data correctness check. A separate
warm-cache recheck of `512x65536x1536` measured 19.2723 us and also passed.

The steady-state refill-wait fence was checked in both run orders:

| Pair | Previous schedule (us) | Separated wait (us) | Improvement |
|---|---:|---:|---:|
| Baseline then candidate | 18.5332 | 18.3092 | 1.21% |
| Candidate then baseline | 18.8575 | 18.3662 | 2.61% |
| **Mean** | **18.6954** | **18.3377** | **1.91%** |

Applying the fence to split-K=4 regressed `512x6144x7168` by 1.88% and was
neutral on `512x7168x16384`, so it remains scoped to MX128 split-K=1.

## Reproduction

Run both checked-in tuned dispatches:

```bash
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$PWD/aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv" \
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_ABPRESHUFFLE="$PWD/aiter/configs/model_configs/dsv4_a8w8_blockscale_abpreshuffle_tuned_gemm.csv" \
ENABLE_CK=0 \
python3 op_tests/test_gemm_a8w8_blockscale.py \
  --flydsl --ck_preshuffle True --apre True \
  --data-init constant --scale-init constant \
  -m 512 \
  -nk 6144,7168 7168,3072 7168,16384 65536,1536 2048,7168 8192,1536 \
  --table
```

GPU-only timing:

```bash
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$PWD/aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv" \
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_ABPRESHUFFLE="$PWD/aiter/configs/model_configs/dsv4_a8w8_blockscale_abpreshuffle_tuned_gemm.csv" \
ENABLE_CK=0 \
rocprofv3 --stats --kernel-trace -f csv -o /tmp/gfx1250_dsv4 -- \
python3 op_tests/test_gemm_a8w8_blockscale.py \
  --flydsl --ck_preshuffle True --apre True \
  --data-init constant --scale-init constant \
  -m 512 \
  -nk 6144,7168 7168,3072 7168,16384 65536,1536 2048,7168 8192,1536 \
  --table
```

Use the normal installed FlyDSL compiler. Both the MX32 smoke test and the
MX128 production defaults were validated with that compiler; no patched LLVM
toolchain is required.
