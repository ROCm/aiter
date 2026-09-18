# gfx1250 MX128 FlyDSL tuning and native ASM comparison

## Scope

This note records the current tuning state on
`perf/gfx1250-flydsl-waitcnt-align` for six DSV4 A-preshuffle GEMMs:

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

| Shape (M x N x K) | FlyDSL profile |
|---|---|
| `512x6144x7168` | `t256x256x128_mw2_nw2_nb4_sk4_cm1_cn2_fsk_apre` |
| `512x7168x3072` | `t128x128x128_mw2_nw2_nb4_sk1_cm4_cn4_apre` |
| `512x7168x16384` | `t256x256x128_mw2_nw2_nb4_sk4_cm2_cn2_fsk_apre` |
| `512x65536x1536` | `t256x256x128_mw2_nw2_nb4_sk1_cm2_cn2_apre_ps2` |
| `512x2048x7168` | `t128x128x128_mw2_nw2_nb4_sk4_cm2_cn2_fsk_apre` |
| `512x8192x1536` | `t128x128x128_mw2_nw2_nb4_sk1_cm4_cn4_apre` |

The full names in the tuned CSV use the
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

## Extended six-shape validation

The expanded public-dispatch check passed for all six shapes. Its event-timed
means were:

| Shape (M x N x K) | A-preshuffle (us) |
|---|---:|
| `512x6144x7168` | 12.8518 |
| `512x7168x3072` | 8.2330 |
| `512x7168x16384` | 19.3574 |
| `512x65536x1536` | 19.2513 |
| `512x2048x7168` | 8.2339 |
| `512x8192x1536` | 6.0729 |

`512x2048x7168` previously had no A-preshuffle row and fell back to a generic
split-K kernel plus a separate reduction. Two paired profiler runs measured
that path at 10.2814 and 9.8566 us end-to-end. The selected fused split-K=4
profile measured 7.9306 and 7.8308 us, for an average 21.73% improvement.

## Native ASM comparison

The exact native kernel is:

```text
_ZN5aiter48f8gemm_bf16_mxfp8fp8_ABpreShuffle_256x256_4x2_psE
```

Split-K raw-kernel numbers must not be compared with a fused or reduced result
without accounting for the reduction kernel. The relevant measurements are:

| Shape | Native ASM | FlyDSL | Interpretation |
|---|---:|---:|---|
| `512x6144x7168` | 9.6862 us raw GEMM; 20.5877 us with reduction | 12.7167 us fused | ASM raw is faster; FlyDSL is 38.23% faster than the measured ASM complete path |
| `512x7168x16384` | 16.3938 us raw GEMM | 17.6802 us raw GEMM | Raw-to-raw FlyDSL gap is 7.85% |
| `512x7168x16384` | about 25.7000 us with reduction | 19.2532 us fused | FlyDSL complete path is about 25.08% faster |
| `512x65536x1536` | 16.2844 us, split-K=1 | 18.7708 us, split-K=1 | FlyDSL is 15.27% slower |

For `512x7168x16384`, the native ASM kernel is therefore correctly described
as a 17-us-class raw kernel. Its separate split-K reduction is not included in
that 16.3938-us number.

## Correctness checks

- Constant data and constant scales passed the reference check for all three
  tuned-config dispatches.
- With uniform data and automatic E8M0 scales, A-preshuffle output was bitwise
  identical to the matching row-major-A FlyDSL profile for all three shapes.
- Against the FP32-accumulating reference, the split-K=4 profiles have the
  expected BF16 reduction-order sensitivity: 5.88% and 6.18% of elements fell
  outside the test's 1% elementwise tolerance. The split-K=1 large-N profile
  was within the normal warning range at 0.12%. No NaN/Inf or A-preshuffle
  layout discrepancy was observed.

## Scheduling changes retained

- Native-style explicit VGPR pinning and its wait scheduling remain scoped to
  the MX32 path.
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

Run the checked-in tuned dispatches:

```bash
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

Do not set `FLYDSL_COMPILE_LLVM_DIR=/app/llvm-pin-tools` with the current
installed FlyDSL package: that older external toolchain rejects the generated
buffer-resource MLIR syntax. The normal compiler path is the validated setup.
