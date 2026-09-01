# DSV3 Remaining 11 Tuning Cases

## Summary

Comparison of the original `main`, original `branch`, and tuned MXFP4 FlyDSL results for the 11 cases that were not faster than `main` in the previous 17-case comparison.

- Raw measurements only; no averages.
- Tuned Stage1/Stage2/Total values come from the fresh `--kernel` rerun on GPU1.
- Positive percentage means tuned is slower; negative percentage means tuned is faster.

## Test configuration

| Item | Value |
|---|---|
| Quantization | `q=4` (MXFP4) |
| Activation | SiLU |
| Model dimension | `model_dim=7168` |
| Device | `HIP_VISIBLE_DEVICES=1` |
| Routing | `AITER_MOE_EXPERT_BALANCE=true` |
| Mode | `--no-flydsl-csv --kernel` |

## Total latency comparison (us)

| Case | Original main | Original branch | Tuned | Tuned vs main | Tuned vs branch |
|---|---:|---:|---:|---:|---:|
| Case 08 · T2048/I256/E257/K9 | 296.259 | 332.158 | 298.352 | +0.71% | -10.18% |
| Case 09 · T4096/I256/E257/K9 | 485.943 | 528.566 | 567.671 | +16.82% | +7.40% |
| Case 10 · T8192/I256/E257/K9 | 801.407 | 866.801 | 900.783 | +12.40% | +3.92% |
| Case 22 · T1024/I512/E257/K9 | 318.469 | 341.126 | 344.400 | +8.14% | +0.96% |
| Case 28 · T32768/I2048/E64/K8 | 7585.090 | 7987.440 | 7596.800 | +0.15% | -4.89% |
| Case 40 · T16384/I2048/E32/K8 | 3706.370 | 3945.750 | 3883.990 | +4.79% | -1.57% |
| Case 41 · T32768/I2048/E32/K8 | 7196.410 | 7611.330 | 7577.900 | +5.30% | -0.44% |
| Case 48 · T1024/I256/E256/K8 | 171.043 | 181.483 | 213.356 | +24.74% | +17.56% |
| Case 49 · T2048/I256/E256/K8 | 236.694 | 303.053 | 272.720 | +15.22% | -10.01% |
| Case 77 · T512/I2048/E33/K8 | 205.320 | 222.084 | 215.786 | +5.10% | -2.84% |
| Case 82 · T16384/I2048/E33/K8 | 3662.660 | 3972.640 | 3876.470 | +5.84% | -2.42% |

## Stage latency comparison (us)

| Case | Main S1 | Branch S1 | Tuned S1 | Main S2 | Branch S2 | Tuned S2 |
|---|---:|---:|---:|---:|---:|---:|
| Case 08 · T2048/I256/E257/K9 | 118.345 | 142.813 | 142.432 | 177.913 | 189.345 | 155.920 |
| Case 09 · T4096/I256/E257/K9 | 188.580 | 191.102 | 216.437 | 297.363 | 337.465 | 351.233 |
| Case 10 · T8192/I256/E257/K9 | 271.261 | 267.421 | 325.167 | 530.146 | 599.381 | 575.617 |
| Case 22 · T1024/I512/E257/K9 | 178.588 | 188.007 | 215.664 | 139.880 | 153.118 | 128.735 |
| Case 28 · T32768/I2048/E64/K8 | 3793.490 | 4014.460 | 3975.900 | 3791.600 | 3972.980 | 3620.910 |
| Case 40 · T16384/I2048/E32/K8 | 1898.670 | 2024.900 | 2060.340 | 1807.710 | 1920.850 | 1823.650 |
| Case 41 · T32768/I2048/E32/K8 | 3597.140 | 3838.450 | 3895.390 | 3599.270 | 3772.880 | 3682.510 |
| Case 48 · T1024/I256/E256/K8 | 80.200 | 84.844 | 115.647 | 90.843 | 96.639 | 97.710 |
| Case 49 · T2048/I256/E256/K8 | 93.627 | 124.074 | 123.714 | 143.067 | 178.979 | 149.006 |
| Case 77 · T512/I2048/E33/K8 | 127.057 | 136.889 | 134.310 | 78.263 | 85.196 | 81.476 |
| Case 82 · T16384/I2048/E33/K8 | 2034.650 | 2054.240 | 1992.160 | 1628.010 | 1918.400 | 1884.300 |

## Kernel names

| Case | Original main S1 / S2 | Original branch S1 / S2 | Tuned S1 / S2 |
|---|---|---|---|
| Case 08 · T2048/I256/E257/K9 | `flydsl_moe1_afp4_wfp4_bf16_t128x64x256_w3` / `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_sbm128` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128` |
| Case 09 · T4096/I256/E257/K9 | `moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16` / `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_persist_sbm128` | `flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w4_bnt0` / `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_persist_sbm64` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_mxmoe_g2_a4w4_128x256x256` |
| Case 10 · T8192/I256/E257/K9 | `flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w4_bnt0` / `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_persist` | `flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w4_bnt0` / `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_persist_sbm64` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_mxmoe_g2_a4w4_128x256x256` |
| Case 22 · T1024/I512/E257/K9 | `flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w3` / `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_atomic_persist` | `flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w3` / `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_atomic_persist_nt_sbm64` | `flydsl_mxmoe_g1_a4w4_64x256x256` / `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_persist_sbm64` |
| Case 28 · T32768/I2048/E64/K8 | `moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16` / `flydsl_moe2_afp4_wfp4_bf16_t128x128x256_atomic_persist` | `flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w4_bnt0_fp4` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_atomic_persist_sbm128` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128` |
| Case 40 · T16384/I2048/E32/K8 | `moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16` / `flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_persist_sbm128` | `flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w2_bnt0_fp4` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_atomic_persist_sbm128` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128` |
| Case 41 · T32768/I2048/E32/K8 | `moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16` / `flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_persist_sbm128` | `flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w4_bnt0_fp4` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_atomic_persist_sbm128` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128` |
| Case 48 · T1024/I256/E256/K8 | `flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w4` / `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce` | `flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w4` / `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | `flydsl_mxmoe_g1_a4w4_64x256x256` / `flydsl_mxmoe_g2_a4w4_64x256x256_atomic_nt` |
| Case 49 · T2048/I256/E256/K8 | `flydsl_moe1_afp4_wfp4_bf16_t128x64x256_w3` / `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_sbm128` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` |
| Case 77 · T512/I2048/E33/K8 | `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_fp4` / `flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_persist` | `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2_bnt0_fp4` / `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_persist_sbm64` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128` |
| Case 82 · T16384/I2048/E33/K8 | `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_bnt0_fp4` / `flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_persist_async_w4_cumul3` | `flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w2_bnt0_fp4` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_atomic_persist_sbm128` | `flydsl_mxmoe_g1_a4w4_128x256x256` / `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128` |

## Source

| Source | Path / description |
|---|---|
| Full comparison | `dsv3-fp4-kernel-speed-comparison-full.canvas.tsx` |
| Tuned output | `reports/dsv3_fp4_17_tune/dsv3_fp4_17_tuned_20260831_075210.csv` |
| Stage timing run | `op_tests/test_moe_2stage.py` with `--kernel` on GPU1 |
