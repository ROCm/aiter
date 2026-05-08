# Performance Comparison: MFMA Kernel vs JAX vs TransformerEngine (CK) vs ck_pr_6764

**Config:** bs=2048, nheads=32, hdim=128, bfloat16, causal=False, seqlen_q == seqlen_kv

**MFMA kernel selection:** mfma_4x4 for seq 1–4, mfma_16x16 for seq 5–17

**ck_pr_6764:** measured with `run_mha_performance_comparison.sh` (asm v2 `fwd_v3=0`, `bwd_v3=0`, CK path).

## Forward Pass (mean time in ms)

| seq | JAX (ms) | TE/CK (ms) | MFMA (ms) | kernel | vs JAX | vs TE/CK | ck_pr_6764 (ms) |
|----:|---------:|-----------:|----------:|:-------|-------:|---------:|----------------:|
| 1 | 0.051 | 1.275 | 0.029 | mfma_4x4 | 1.76x | 43.97x | 0.088 |
| 2 | 0.233 | 1.298 | 0.047 | mfma_4x4 | 4.96x | 27.62x | 0.091 |
| 3 | 0.256 | 1.306 | 0.063 | mfma_4x4 | 4.06x | 20.73x | 0.096 |
| 4 | 0.286 | 1.316 | 0.080 | mfma_4x4 | 3.57x | 16.45x | 0.098 |
| 5 | 0.324 | 1.366 | 0.262 | mfma_16x16 | 1.24x | 5.21x | 0.103 |
| 6 | 0.354 | 1.375 | 0.272 | mfma_16x16 | 1.30x | 5.06x | 0.113 |
| 7 | 0.389 | 1.389 | 0.290 | mfma_16x16 | 1.34x | 4.79x | 0.120 |
| 8 | 0.415 | 1.379 | 0.291 | mfma_16x16 | 1.43x | 4.74x | 0.134 |
| 9 | 0.496 | 1.403 | 0.304 | mfma_16x16 | 1.63x | 4.62x | 0.135 |
| 10 | 0.516 | 1.404 | 0.320 | mfma_16x16 | 1.61x | 4.39x | 0.147 |
| 11 | 0.565 | 1.409 | 0.330 | mfma_16x16 | 1.71x | 4.27x | 0.156 |
| 12 | 0.582 | 1.423 | 0.335 | mfma_16x16 | 1.74x | 4.25x | 0.169 |
| 13 | 0.637 | 1.424 | 0.358 | mfma_16x16 | 1.78x | 3.98x | 0.175 |
| 14 | 0.649 | 1.433 | 0.359 | mfma_16x16 | 1.81x | 3.99x | 0.187 |
| 15 | 0.702 | 1.436 | 0.380 | mfma_16x16 | 1.85x | 3.78x | 0.198 |
| 16 | 0.686 | 1.439 | 0.386 | mfma_16x16 | 1.78x | 3.73x | 0.203 |
| 17 | 0.793 | 1.452 | 0.401 | mfma_16x16 | 1.98x | 3.62x | 0.267 |

## Backward Pass (mean time in ms)

| seq | JAX (ms) | TE/CK (ms) | MFMA (ms) | vs JAX | vs TE/CK | ck_pr_6764 (ms) |
|----:|---------:|-----------:|----------:|-------:|---------:|----------------:|
| 1 | 0.087 | 2.317 | 0.264 | 0.33x | 8.78x | 0.202 |
| 2 | 0.350 | 2.377 | 0.308 | 1.14x | 7.72x | 0.211 |
| 3 | 0.415 | 2.472 | 0.386 | 1.08x | 6.40x | 0.254 |
| 4 | 0.461 | 2.548 | 0.463 | 1.00x | 5.50x | 0.270 |
| 5 | 0.509 | 2.606 | 0.503 | 1.01x | 5.18x | 0.285 |
| 6 | 0.519 | 2.669 | 0.523 | 0.99x | 5.10x | 0.301 |
| 7 | 0.604 | 2.705 | 0.561 | 1.08x | 4.82x | 0.323 |
| 8 | 0.835 | 2.737 | 0.570 | 1.46x | 4.80x | 0.346 |
| 9 | 0.790 | 2.851 | 0.606 | 1.30x | 4.70x | 0.358 |
| 10 | 0.731 | 2.877 | 0.634 | 1.15x | 4.54x | 0.382 |
| 11 | 0.894 | 2.899 | 0.657 | 1.36x | 4.41x | 0.403 |
| 12 | 0.922 | 2.957 | 0.677 | 1.36x | 4.37x | 0.421 |
| 13 | 0.987 | 3.000 | 0.734 | 1.34x | 4.09x | 0.435 |
| 14 | 0.924 | 3.017 | 0.753 | 1.23x | 4.01x | 0.472 |
| 15 | 1.084 | 3.083 | 0.792 | 1.37x | 3.89x | 0.483 |
| 16 | 1.185 | 3.096 | 0.813 | 1.46x | 3.81x | 0.500 |
| 17 | 1.246 | 3.441 | 0.998 | 1.25x | 3.45x | 2.183 |

## Summary

### Forward (MFMA reference)
- **vs JAX:** 1.24x -- 4.96x faster (geometric mean ~1.9x)
- **vs TE/CK:** 3.62x -- 43.97x faster (geometric mean ~6.5x)

### Backward (MFMA reference)
- **vs JAX:** 0.33x -- 1.46x (geometric mean ~1.1x)
- **vs TE/CK:** 3.45x -- 8.78x faster (geometric mean ~4.9x)

### Forward (ck_pr_6764, geometric mean vs references)
- **vs JAX:** ~2.9x
- **vs TE/CK:** ~10.0x

### Backward (ck_pr_6764, geometric mean vs references)
- **vs JAX:** ~1.7x
- **vs TE/CK:** ~7.3x

### Notes
- JAX numbers are from TransformerEngine benchmark on the same GPU class
- TE/CK = TransformerEngine using CK (Composable Kernel) backend
- MFMA forward uses mfma_4x4x4 for seq 1–4, mfma_16x16x16 for seq 5–17
- MFMA backward uses mfma_16x16x16 for all sequence lengths
- Forward speedup columns (vs JAX / vs TE/CK) are reference_mean / MFMA time (higher = MFMA faster)
- Backward vs columns use the same ratio convention as the MFMA reference table
- Backward seq=1 is slower than JAX in the MFMA reference because JAX uses a highly optimized scalar path for single-token attention
- **ck_pr_6764** timings are produced by this repo’s MHA v2 benchmark (`mask=0`, batch layout BHSD)

