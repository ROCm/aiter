# MI450-MC vs MI455 — Triton Kernel Performance Comparison

| | MI450-MC | MI455 |
|---|---|---|
| **Host** | f19-12 | heliosr-1b114-d01-3 |
| **GPU** | gfx1250 (0x75c7), 4× GPUs | gfx1250 (0x75c1) |
| **Commit** | 4934499e3 | a7242f1a4 |
| **Triton** | 3.8.0+amd.rocm7.2.0.git7cb7b059 | 332c72a7859 |
| **Date** | 2026-10-08 | 2026-10-08 |

---

## GEMM (Dense)

| Test | Shape (M×N×K) | MI450-MC (TFLOPS) | MI455 (TFLOPS) | Speedup |
|------|---------------|-------------------:|---------------:|--------:|
| a16w16 | 8192×6144×6144 | 572.7 | 1445.75 | **2.52×** |
| a16w16 | 8192×5120×4096 | 610.8 | 1575.60 | **2.58×** |
| a16w16 | 256×5120×4096 | 277.8 | 502.74 | **1.81×** |
| a16w16 (persistent) | 8192×5120×2880 | 638.8 | 1576.39 | **2.47×** |
| a8w8 blockscale (preshuffle) | 4096×2048×7168 | 1309.6 | 2701.23 | **2.06×** |
| a8w8 blockscale (preshuffle) | 16×2048×7168 | 14.0 | 10.59 | 0.76× |
| afp4wfp4 (preshuffle) | 4096×7168×16384 | 684.3 | 632.28 | 0.92× |
| afp8wfp8 (preshuffle) | 4096×7168×16384 | 1961.9 | 4780.42 | **2.44×** |

**Takeaway:** Large-batch dense GEMM is consistently ~2.5× faster on MI455. The small-batch decode case (M=16, a8w8) and mxfp4 GEMM show slight regressions — likely latency-bound or tuning-sensitive. The afp8wfp8 (mxfp8) kernel sees a strong 2.44× lift.

---

## GEMM (Batched)

| Test | Shape (B×M×N×K) | MI450-MC (TFLOPS) | MI455 (TFLOPS) | Speedup |
|------|-----------------|-------------------:|---------------:|--------:|
| batched_gemm_bf16 | 16×2048×1024×4096 | 575.4 | 1346.83 | **2.34×** |
| batched_gemm_a8w8 (MLA BMM) | 128×1024×512×128 | 248.4 | 183.56 | 0.74× |

**Takeaway:** BF16 batched GEMM scales ~2.3×. The MLA absorbed BMM (small K=128) regresses — this is a latency-dominated shape.

---

## MoE

### a8w4 (fp8 × mxfp4, 7168×4096, E256T8, M2048)

| Phase | MI450-MC |  | MI455 |  | Speedup |
|-------|---:|---:|---:|---:|--------:|
| | us | TFLOPS | us | TFLOPS | (time) |
| moe1 | 1064.3 | 904.0 | 480.28 | 2003 | **2.22×** |
| moe2 | 582.6 | 825.7 | 282.17 | 1705 | **2.07×** |
| **total** | **1593.9** | **905.4** | **750.61** | **1923** | **2.12×** |

### a8w4 preshuffle (fp8 × mxfp4, 7168×4096, E256T8, M2048)

| Phase | MI450-MC |  | MI455 |  | Speedup |
|-------|---:|---:|---:|---:|--------:|
| | us | TFLOPS | us | TFLOPS | (time) |
| moe1 | 1004.1 | 958.1 | 501.28 | 1919 | **2.00×** |
| moe2 | 581.2 | 827.7 | 306.97 | 1567 | **1.89×** |
| **total** | **1605.5** | **898.9** | **793.49** | **1819** | **2.02×** |

### a4w4 (mxfp4 × mxfp4, 7168×4096, E256T8, M2048)

| Phase | MI450-MC |  | MI455 |  | Speedup |
|-------|---:|---:|---:|---:|--------:|
| | us | TFLOPS | us | TFLOPS | (time) |
| moe1 | 1522.9 | 631.7 | 671.67 | 1432 | **2.27×** |
| moe2 | 769.1 | 625.4 | 325.05 | 1480 | **2.37×** |
| **total** | **2246.9** | **642.3** | **992.73** | **1454** | **2.26×** |

**Takeaway:** MoE kernels are uniformly ~2.0–2.3× faster on MI455 across all precisions and preshuffle variants.

---

## Attention

| Test | Config | Metric | MI450-MC | MI455 | Speedup |
|------|--------|--------|---:|---:|--------:|
| unified_attention bf16 | b4 hq64 hk8 sq1024 sk8192 d128 | TFLOPS | 47.80 | 88.95 | **1.86×** |
| unified_attention fp8 | b4 hq64 hk8 sq1024 sk8192 d128 | TFLOPS | 39.42 | 73.63 | **1.87×** |
| MLA decode (varlen) | b32 ctx8192 hq16 hkv1 lora512 | TB/s | 8.235 | 9.10 | **1.10×** |
| KDA decode | h24 b64 s1 | TB/s | 4.760 | 8.28 | **1.74×** |
| fp8_mqa_logits | default | TFLOPS | 392.7 | 754.78 | **1.92×** |
| chunk_kda | h24 b1 s4096 | — | FAIL | FAIL | — |

**Takeaway:** Attention kernels see ~1.9× improvement. MLA decode, being memory-bandwidth-bound, gains only 1.10×. KDA decode picks up a solid 1.74×. Both platforms fail `chunk_kda` with the same Gluon `_reinterpret` compilation error.

---

## Norm / Fusion

| Test | Shape (M×N) | Metric | MI450-MC (GB/s) | MI455 (GB/s) | Speedup |
|------|-------------|--------|---:|---:|--------:|
| fused_add_rmsnorm_pad | 8192×7168 | BW | 6066.8 | 11853.64 | **1.95×** |
| fused_rmsnorm_add | 8192×7168 | BW | 6483.3 | 13088.95 | **2.02×** |
| rmsnorm + mxfp4 quant | 4096×7168 | BW | 1719.7 | 3451.51 | **2.01×** |

**Takeaway:** Norm/fusion kernels scale almost exactly 2× — consistent with a ~2× memory bandwidth advantage.

---

## Quantization / Other

| Test | Shape (M×N) | Metric | MI450-MC (GB/s) | MI455 (GB/s) | Speedup |
|------|-------------|--------|---:|---:|--------:|
| fused_clamp_act_mul | 8192×3584 | BW | 3414.6 | 7162.16 | **2.10×** |
| quant_mxfp4 (gluon) | 4096×7168 | BW | 3575.0 | 7182.88 | **2.01×** |
| quant_mxfp8 (gluon) | 4096×7168 | BW | 2754.4 | 7680.76 | **2.79×** |

**Takeaway:** Memory-bound quant/fusion ops are ~2× faster, matching bandwidth ratio. mxfp8 quant sees an outlier 2.79× — suggests MI450-MC was underperforming on this kernel.

---

## Summary

| Category | Typical MI455 Speedup | Notes |
|----------|----------------------:|-------|
| Dense GEMM (large batch) | 2.4–2.6× | Compute-bound, scales with peak FLOPS |
| Dense GEMM (small batch) | 0.8–0.9× | Latency-bound decode shapes regress slightly |
| Batched GEMM (bf16) | 2.3× | Strong scaling |
| MoE (all precisions) | 2.0–2.3× | Consistent across a8w4, a4w4, preshuffle |
| Attention (compute) | 1.9× | Unified attention bf16/fp8, MQA logits |
| Attention (BW-bound) | 1.1–1.7× | MLA decode (1.1×), KDA decode (1.7×) |
| Norm / Fusion | 2.0× | Tracks memory bandwidth ratio |
| Quant / Elementwise | 2.0–2.8× | Memory-bandwidth-bound ops |

**Overall:** MI455 delivers a consistent **~2× improvement** across most kernel categories versus MI450-MC. Compute-bound large-batch GEMM peaks at **2.5×+**. Memory-bandwidth-bound kernels (norm, quant, fusion) track at **~2×**, indicating roughly double the effective memory bandwidth. Small-batch latency-sensitive shapes (decode GEMM, MLA) show modest or negative scaling — these are launch-overhead-dominated and may benefit from further tuning. Both platforms share the same `chunk_kda` compilation failure.
