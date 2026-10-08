# gfx1250 Gluon Kernel Performance Report

**Commits compared:** A = `1666e85e` → B = `17201243`

## Summary

- **27 benchmarks** across GEMM, Batched GEMM, MoE, Attention, Norm/Fusion, and Quant
- **No regressions detected** — all deltas within noise for the vast majority of tests
- **One notable improvement:** MoE mxfp4×mxfp4 `moe1` improved **+35.1%** (876.8 → 649 us)

### Verdict

Commit B is **performance-neutral to positive** relative to A. The single large improvement
in MoE a4w4 moe1 is real (spread 1.2%), and no benchmark regressed outside of noise.

---

## Detailed Results

### GEMM (Gluon warp-pipeline, `gemm_warp_pipeline_cdna5.py`)

| Shape | Dtype | Metric | A | B | B vs A | Spread |
|---|---|---|---:|---:|---:|---:|
| 4096×4096×65536 | bf16 | TFLOPS | 2,382 | 2,380 | -0.1% | 0.2% |
| 4096×4096×65536 | mxfp8 (tile 256³, cluster 2×2) | TFLOPS | 5,392 | 5,392 | -0.0% | 0.4% |
| 4096×4096×65536 | mxfp4 | TFLOPS | 10,579 | 10,536 | -0.4% | 1.3% |
| 4096×4096×65536 | fp8→mxfp4 | TFLOPS | 7,415 | 7,413 | -0.0% | 0.2% |

All within noise (spread ≤ 1.3%).

### GEMM (`bench_gemm_a16w16.py` — `gemm_a16w16.py` kernel)

| Shape | Config | Metric | A | B | B vs A | Spread |
|---|---|---|---:|---:|---:|---:|
| bf16 8192×6144×6144 | tuned compute_bound 256×256×64 | ms | 0.4219 | 0.4213 | +0.1% | 1.0% |
| bf16 8192×5120×4096 | tuned bandwidth_bound 256×256×128 | ms | 0.2171 | 0.2174 | -0.1% | 0.6% |
| bf16 256×5120×4096 | tuned compute_bound 64×64×128 | ms | 0.0202 | 0.02012 | +0.4% | 36.1% |
| bf16 persistent 8192×5120×2880 | tuned | ms | 0.1476 | 0.1446 | +2.1% | 3.2% |

Persistent GEMM shows a modest +2.1% improvement. The 256×5120 shape has high spread (36.1%)
due to the very small absolute runtime (~20 µs) — result is in noise.

### GEMM — FP8 blockscale (`bench_gemm_a8w8_blockscale.py`)

| Shape | Config | Metric | A | B | B vs A | Spread |
|---|---|---|---:|---:|---:|---:|
| 4096×2048×7168 | fp8 preshuffle (DSv4) | ms | 0.04272 | 0.04252 | +0.5% | 0.9% |
| 16×2048×7168 | fp8 preshuffle (decode) | ms | 0.04288 | 0.04189 | +2.4% | 7.4% |

Decode shape (+2.4%) is within its 7.4% spread; still directionally positive.

### GEMM — MX formats (`bench_gemm_afp4wfp4.py`, `bench_gemm_afp8wfp8.py`)

| Shape | Format | Metric | A | B | B vs A | Spread |
|---|---|---|---:|---:|---:|---:|
| 4096×7168×16384 | mxfp4 preshuffle | ms | 1.410 | 1.401 | +0.6% | 4.5% |
| 4096×7168×16384 | mxfp8 preshuffle | ms | 0.1937 | 0.1936 | +0.1% | 0.2% |

Neutral.

### Batched GEMM

| Shape | Config | Metric | A | B | B vs A | Spread |
|---|---|---|---:|---:|---:|---:|
| B=16, 2048×1024×4096 | bf16 tuned | ms | 0.1954 | 0.1968 | -0.7% | 1.8% |
| B=128, 1024×512×128 | bf16×fp8 MLA absorbed (gluon) | ms | 0.09074 | 0.09208 | -1.5% | 3.8% |

Within noise.

### MoE (`bench_moe_gemm_a8w4_cudagraph.py`)

**fp8 × mxfp4, DSv3 7168/4096, 256 experts top-8, 2048 tokens:**

| Metric | A | B | B vs A | Spread |
|---|---:|---:|---:|---:|
| moe1 (us) | 483.4 | 481.1 | +0.5% | 2.9% |
| moe2 (us) | 277.3 | 272.4 | +1.8% | 1.5% |
| total (us) | 764.4 | 761.6 | +0.4% | 3.1% |

**fp8 × mxfp4 preshuffled, DSv3, 2048 tokens:**

| Metric | A | B | B vs A | Spread |
|---|---:|---:|---:|---:|
| moe1 (us) | 503.5 | 506.3 | -0.6% | 1.5% |
| moe2 (us) | 306.4 | 306.5 | -0.0% | 1.0% |
| total (us) | 803.0 | 805.9 | -0.4% | 0.8% |

Neutral.

### MoE (`bench_moe_gemm_a4w4_cudagraph.py`)

**mxfp4 × mxfp4, DSv3 7168/4096, 256 experts top-8, 2048 tokens:**

| Metric | A | B | B vs A | Spread |
|---|---:|---:|---:|---:|
| moe1 (us) | 876.8 | **649.0** | **+35.1%** | 1.2% |
| moe2 (us) | 316.6 | 313.9 | +0.9% | 1.8% |
| total (us) | 1,192 | **962.5** | **+23.8%** | 1.2% |

**Significant improvement.** The moe1 kernel is 35% faster in B, bringing total MoE a4w4
latency down by ~24%. Low spread (1.2%) confirms this is a real gain.

### Attention

| Benchmark | Config | Metric | A | B | B vs A | Spread |
|---|---|---|---:|---:|---:|---:|
| unified_attention | bf16 b4 hq64 hk8 d128, q1024/kv8192, page 256 | TFLOPS | 87.93 | 87.98 | +0.1% | 0.1% |
| unified_attention | fp8 b4 hq64 hk8 d128, q1024/kv8192, page 512 | TFLOPS | 72.64 | 72.61 | -0.0% | 0.1% |
| MLA decode | b32 ctx8192, 16 heads (DSv3 TP8) | ms | 0.0667 | 0.06645 | +0.4% | 5.0% |
| KDA decode | 24 heads (Kimi-K3 TP4), b64 | TB/s | 8.388 | 8.395 | +0.1% | 0.7% |
| chunk KDA prefill | 24 heads, 4096 tokens | us | 141.5 | 141.4 | +0.1% | 1.5% |
| chunk KDA prepare | — | us | 52.44 | 55.44 | -5.4% | 12.1% |
| chunk KDA walk:64,4 | — | us | 108.0 | 110.1 | -1.9% | 21.6% |
| fp8 MQA logits | DSv3.2 indexer, defaults | ms | 0.1774 | 0.1758 | +0.9% | 0.8% |

Attention kernels are stable. chunk KDA sub-components (prepare, walk) show variation
but are within their high spread bands and the total is flat.

### Norm / Fusion

| Benchmark | Shape | Metric | A | B | B vs A | Spread |
|---|---|---|---:|---:|---:|---:|
| fused_add_rmsnorm_pad | 8192×7168, residual | ms | 0.04135 | 0.04115 | +0.5% | 2.1% |
| fused_rmsnorm_add | 8192×7168, residual | ms | 0.03708 | 0.03714 | -0.2% | 2.4% |
| rmsnorm + mxfp4 quant | 4096×7168 | ms | 0.02103 | 0.02127 | -1.1% | 4.6% |

All within noise.

### Quant / Other

| Benchmark | Shape | Metric | A | B | B vs A | Spread |
|---|---|---|---:|---:|---:|---:|
| fused_clamp_act_mul | 8192×3584 | ms | 0.02342 | 0.02343 | -0.0% | 0.6% |
| quant mxfp4 (gluon) | 4096×7168 | ms | 0.01065 | 0.01068 | -0.3% | 4.5% |
| quant mxfp8 (gluon) | 4096×7168 | ms | 0.01200 | 0.01202 | -0.2% | 1.7% |

Neutral.

---

## Key Takeaways

1. **MoE a4w4 moe1 kernel: +35% faster** — the standout result. Total MoE a4w4 latency
   drops ~24% (1,192 → 963 us). Spread is low (1.2%), confirming this is real.

2. **Everything else is neutral** — deltas are within measurement noise across all
   GEMM, attention, norm, and quant benchmarks.

3. **No regressions** — the largest negative delta is chunk KDA prepare at -5.4%,
   but its spread is 12.1%, making this indistinguishable from noise.

4. **High-spread results** (bf16 256×5120×4096 at 36.1%, chunk KDA walk at 21.6%)
   reflect the inherent measurement variability of very short kernels (~20-110 µs)
   and should not be interpreted as regressions.
