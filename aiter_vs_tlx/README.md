# AITER vs TLX Kernel Performance Comparison — MI355

**Task:** Collect perf numbers on MI355 node for GEMM, Paged Attention, Grouped GEMM, and Flash Attention (fwd) kernels in TLX and AITER.

- **TLX repo:** https://github.com/facebookexperimental/triton
- **TLX kernels:** https://github.com/facebookexperimental/triton/tree/main/third_party/tlx/tutorials
- **AITER:** uses Gluon kernel when available, otherwise Triton kernels

---

## 1. GEMM — bf16, Y[M,N] = X[M,K]·Wᵀ (TFLOPS)

Roofline ceiling: **2517 TFLOPS** (MI355X bf16 matrix, dense). All shapes are compute-bound (arithmetic intensity >> 314 FLOP/B crossover).

| M | N | K | AITER Triton tuned | AITER % roof | Config | TLX | TLX % roof | TLX variant | TLX / AITER |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 2048 | **833** | **33.1%** | 128×128×128, 8w | 702 | 27.9% | pipelined | 0.84x |
| 4096 | 4096 | 4096 | 1174 | 46.6% | 256×256×64, 8w, persistent | 1393 | 55.3% | interwave | 1.19x |
| 8192 | 8192 | 8192 | 1230 | 48.9% | 256×256×64, 8w, persistent | 1464 | 58.2% | interwave_streamk | 1.19x |
| 8192 | 10240 | 8192 | 1228 | 48.8% | 256×256×64, 8w, persistent | 1473 | 58.5% | interwave_streamk | 1.20x |
| 8192 | 57344 | 8192 | 1221 | 48.5% | 256×256×64, 8w, persistent | 1467 | 58.3% | interwave_streamk | 1.20x |
| 8192 | 8192 | 28672 | 1220 | 48.5% | 256×256×64, 8w, persistent | 1469 | 58.4% | pingpong | 1.20x |

**Geomean (6 shapes):** AITER 1140 TFLOPS · TLX 1287 TFLOPS · **TLX/AITER 1.13x** (AITER faster in 1/6)
**Roofline (geomean):** AITER **45.3%** of peak · TLX **51.1%** of peak

---

## 2. Grouped GEMM — fp16 (TFLOPS)

Roofline ceiling: **2517 TFLOPS** (MI355X fp16 matrix, dense). Shapes with small M/group have lower arithmetic intensity and hit BW limits earlier.

| M total | K | N | Groups | Rows/group | Group sizes | AITER Triton tuned | AITER % roof | Config | TLX | TLX % roof | TLX variant | TLX / AITER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 65536 | 4096 | 4096 | 16 | 4096 | equal | 1077 | 42.8% | 256×256×64, 8w, work-stealing | 1279 | 50.8% | amd_grouped_gemm | 1.19x |
| 49152 | 1408 | 2048 | 64 | 768 | equal | 766 | 30.4% | 256×256×64, 8w | 866 | 34.4% | amd_grouped_gemm | 1.13x |
| 49152 | 1408 | 2048 | 64 | 768 | uneven | 659 | 26.2% | 128×256×64, 4w, work-stealing | 720 | 28.6% | amd_grouped_gemm | 1.09x |
| 32768 | 6144 | 16384 | 8 | 4096 | equal | 1126 | 44.7% | 256×256×64, 8w, work-stealing | 1373 | 54.6% | amd_grouped_gemm | 1.22x |
| 32768 | 6144 | 16384 | 8 | 4096 | uneven | 1045 | 41.5% | 256×256×64, 8w | 1294 | 51.4% | amd_grouped_gemm | 1.24x |
| 16384 | 7168 | 4096 | 32 | 512 | uneven | 753 | 29.9% | 256×256×64, 8w, work-stealing | 938 | 37.3% | amd_grouped_gemm | 1.24x |

**Geomean (6 shapes):** AITER 885 TFLOPS · TLX 1048 TFLOPS · **TLX/AITER 1.18x** (AITER faster in 0/6)
**Roofline (geomean):** AITER **35.2%** of peak · TLX **41.6%** of peak

---

## 3. Flash Attention Forward — bf16, H_q = H_kv, S_q = S_kv (TFLOPS; causal counts ½ FLOPs)

Roofline ceiling: **2517 TFLOPS** (MI355X bf16 matrix, dense). BW/compute crossover ≈ 314 FLOP/B.
Regime: small-S (S≤1024) shapes are **bandwidth-bound**; large-S/D=128 shapes are **compute-bound**.

Bold = better of Gluon vs Triton for that shape.

| B | H | S | D | Mask | Regime | AITER Gluon tuned | Gluon config | AITER Triton tuned | Triton config | Best AITER % roof | TLX | TLX % roof | TLX variant | TLX / best AITER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 1024 | 64 | full | BW-bound | **643** | 256×128, 8w | 555 | 256×64, 8w, 2st | **25.6%** | 692 | 27.5% | cluster_persistent | 1.08x |
| 1 | 64 | 4096 | 64 | full | compute | 766 | 256×64, 4w | **782** | 256×128, 8w, 2st | **31.1%** | 827 | 32.9% | persistent | 1.06x |
| 1 | 64 | 8192 | 64 | full | compute | **794** | 256×64, 4w | 786 | 128×128, 4w, 2st | **31.5%** | 850 | 33.8% | persistent | 1.07x |
| 1 | 64 | 1024 | 128 | full | BW-bound | **711** | 128×64, 4w | 682 | 256×64, 8w, 2st | **28.2%** | 853 | 33.9% | cluster | 1.20x |
| 1 | 64 | 4096 | 128 | full | compute | 859 | 256×64, 8w | **927** | 256×64, 8w, 3st | **36.8%** | 1103 | 43.8% | cluster | 1.19x |
| 1 | 64 | 8192 | 128 | full | compute | 841 | 128×64, 4w | **937** | 256×64, 8w, 3st | **37.2%** | 1150 | 45.7% | cluster | 1.23x |
| 2 | 64 | 8192 | 128 | full | compute | 822 | 256×64, 8w | **952** | 256×64, 8w, 3st | **37.8%** | 1152 | 45.8% | cluster | 1.21x |
| 1 | 64 | 16384 | 128 | full | compute | 825 | 256×64, 8w | **960** | 256×64, 8w, 3st | **38.1%** | 1168 | 46.4% | cluster | 1.22x |
| 1 | 64 | 1024 | 64 | causal | BW-bound | **378** | 128×64, 4w | 295 | 128×32, 4w, 2st | **15.0%** | 313 | 12.4% | cluster | 0.83x |
| 1 | 64 | 4096 | 64 | causal | compute | 615 | 256×64, 4w | **641** | 128×128, 4w, 2st | **25.5%** | 727 | 28.9% | persistent | 1.13x |
| 1 | 64 | 8192 | 64 | causal | compute | **700** | 256×64, 4w | 689 | 128×128, 4w, 2st | **27.8%** | 808 | 32.1% | persistent | 1.15x |
| 1 | 64 | 1024 | 128 | causal | BW-bound | **429** | 128×64, 4w | 367 | 128×32, 4w, 2st | **17.0%** | 456 | 18.1% | cluster | 1.06x |
| 1 | 64 | 4096 | 128 | causal | compute | 695 | 128×64, 4w | **704** | 256×64, 8w, 3st | **28.0%** | 953 | 37.9% | cluster | 1.35x |
| 1 | 64 | 8192 | 128 | causal | compute | 772 | 256×64, 8w | **800** | 256×64, 8w, 3st | **31.8%** | 1066 | 42.3% | cluster | 1.33x |
| 2 | 64 | 8192 | 128 | causal | compute | 776 | 256×64, 8w | **798** | 256×64, 8w, 3st | **31.7%** | 1060 | 42.1% | cluster | 1.33x |
| 1 | 64 | 16384 | 128 | causal | compute | 722 | 128×64, 4w | **805** | 256×64, 8w, 3st | **32.0%** | 1100 | 43.7% | cluster | 1.37x |

**Geomean (16 shapes):** AITER Gluon 693 · AITER Triton 698 · TLX 847 · **TLX/best AITER 1.17x** (AITER faster in 1/16)
**Roofline (geomean, best AITER):** AITER **27.7%** of peak · TLX **33.6%** of peak

Gluon wins 6/16 shapes · Triton wins 10/16 shapes

---

## 4. Paged Attention Decode — bf16, page size 16 (TB/s of KV cache read)

Roofline ceiling: **8.0 TB/s** (MI355X HBM bandwidth). This kernel is memory-bandwidth-bound.

Bold = better of Gluon vs Triton for that shape (n/a rows have only Gluon available).

| Batch | Context | Q heads | KV heads | D | Query len | AITER Gluon tuned | Gluon config | AITER Triton tuned | Triton config | Best AITER % roof | TLX | TLX % roof | TLX / best AITER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8192 | 64 | 8 | 64 | 1 | **1.96** | ps=True, 32 splits | 1.23 | V2, part 256 | **24.5%** | 2.45 | 30.6% | 1.25x |
| 8 | 8192 | 64 | 8 | 64 | 1 | **4.85** | ps=True, 8 splits | 4.26 | V2, part 256 | **60.6%** | 5.33 | 66.6% | 1.10x |
| 32 | 8192 | 64 | 8 | 64 | 1 | **5.74** | ps=True, 2 splits | 4.56 | V2, part 1024 | **71.8%** | 5.30 | 66.3% | 0.92x |
| 128 | 8192 | 64 | 8 | 64 | 1 | **6.17** | ps=True, 1 split | 4.75 | V2, part 2048 | **77.1%** | 5.85 | 73.1% | 0.95x |
| 1 | 32768 | 64 | 8 | 64 | 1 | **3.64** | ps=True, 32 splits | 2.48 | V2, part 512 | **45.5%** | 4.25 | 53.1% | 1.17x |
| 8 | 32768 | 64 | 8 | 64 | 1 | **5.72** | ps=True, 8 splits | 4.55 | V2, part 1024 | **71.5%** | 5.40 | 67.5% | 0.94x |
| 32 | 32768 | 64 | 8 | 64 | 1 | **6.19** | ps=True, 4 splits | 4.80 | V2, part 2048 | **77.4%** | 5.88 | 73.5% | 0.95x |
| 8 | 131072 | 64 | 8 | 64 | 1 | **6.00** | ps=True, 32 splits | 4.81 | V2, part 2048 | **75.0%** | 6.21 | 77.6% | 1.03x |
| 1 | 8192 | 64 | 8 | 64 | 4 | **1.54** | ps=False | n/a | no qlen>1 path | **19.3%** | 1.50 | 18.8% | 0.97x |
| 8 | 8192 | 64 | 8 | 64 | 4 | **4.36** | ps=False | n/a | no qlen>1 path | **54.5%** | 4.52 | 56.5% | 1.04x |
| 32 | 8192 | 64 | 8 | 64 | 4 | **5.44** | ps=True, 2 splits | n/a | no qlen>1 path | **68.0%** | 4.92 | 61.5% | 0.90x |
| 128 | 8192 | 64 | 8 | 64 | 4 | **5.88** | ps=True, 1 split | n/a | no qlen>1 path | **73.5%** | 5.90 | 73.8% | 1.00x |
| 1 | 32768 | 64 | 8 | 64 | 4 | **1.89** | ps=False | n/a | no qlen>1 path | **23.6%** | 3.43 | 42.9% | 1.82x |
| 8 | 32768 | 64 | 8 | 64 | 4 | **5.46** | ps=True, 8 splits | n/a | no qlen>1 path | **68.3%** | 5.38 | 67.3% | 0.99x |
| 32 | 32768 | 64 | 8 | 64 | 4 | **5.82** | ps=True, 4 splits | n/a | no qlen>1 path | **72.8%** | 5.86 | 73.3% | 1.01x |
| 8 | 131072 | 64 | 8 | 64 | 4 | **5.88** | ps=True, 16 splits | n/a | no qlen>1 path | **73.5%** | 5.91 | 73.9% | 1.00x |
| 1 | 8192 | 64 | 8 | 128 | 1 | **3.05** | ps=False | 2.20 | V2, part 256 | **38.1%** | 3.10 | 38.8% | 1.02x |
| 8 | 8192 | 64 | 8 | 128 | 1 | 4.95 | ps=True, 4 splits | **5.72** | V2, part 512 | **71.5%** | 6.15 | 76.9% | 1.07x |
| 32 | 8192 | 64 | 8 | 128 | 1 | **6.01** | ps=True, 1 split | 5.67 | V2, part 512 | **75.1%** | 6.00 | 75.0% | 1.00x |
| 128 | 8192 | 64 | 8 | 128 | 1 | **6.20** | ps=True, 1 split | 5.72 | V2, part 8192 | **77.5%** | 6.45 | 80.6% | 1.04x |
| 1 | 32768 | 64 | 8 | 128 | 1 | **4.07** | ps=True, 32 splits | 3.88 | V2, part 512 | **50.9%** | 5.26 | 65.8% | 1.29x |
| 8 | 32768 | 64 | 8 | 128 | 1 | **5.75** | ps=True, 4 splits | 5.16 | V2, part 2048 | **71.9%** | 5.93 | 74.1% | 1.03x |
| 32 | 32768 | 64 | 8 | 128 | 1 | **6.25** | ps=True, 1 split | 5.62 | V2, part 512 | **78.1%** | 6.22 | 77.8% | 0.99x |
| 8 | 131072 | 64 | 8 | 128 | 1 | **6.04** | ps=True, 4 splits | 5.78 | V2, part 8192 | **75.5%** | 6.26 | 78.3% | 1.04x |
| 1 | 8192 | 64 | 8 | 128 | 4 | **2.06** | ps=False | n/a | no qlen>1 path | **25.8%** | 1.89 | 23.6% | 0.92x |
| 8 | 8192 | 64 | 8 | 128 | 4 | **5.08** | ps=False | n/a | no qlen>1 path | **63.5%** | 5.16 | 64.5% | 1.02x |
| 32 | 8192 | 64 | 8 | 128 | 4 | **5.72** | ps=True, 1 split | n/a | no qlen>1 path | **71.5%** | 5.55 | 69.4% | 0.97x |
| 128 | 8192 | 64 | 8 | 128 | 4 | **5.77** | ps=True, 1 split | n/a | no qlen>1 path | **72.1%** | 5.84 | 73.0% | 1.01x |
| 1 | 32768 | 64 | 8 | 128 | 4 | **2.47** | ps=False | n/a | no qlen>1 path | **30.9%** | 4.31 | 53.9% | 1.75x |
| 8 | 32768 | 64 | 8 | 128 | 4 | **5.63** | ps=True, 8 splits | n/a | no qlen>1 path | **70.4%** | 5.55 | 69.4% | 0.99x |
| 32 | 32768 | 64 | 8 | 128 | 4 | **5.95** | ps=True, 1 split | n/a | no qlen>1 path | **74.4%** | 5.63 | 70.4% | 0.95x |
| 8 | 131072 | 64 | 8 | 128 | 4 | **6.01** | ps=True, 16 splits | n/a | no qlen>1 path | **75.1%** | 5.68 | 71.0% | 0.94x |

**Geomean (32 shapes):** AITER Gluon 4.61 TB/s · AITER Triton 4.15 TB/s (16 qlen=1 shapes only) · TLX 4.86 TB/s · **TLX/best AITER 1.05x** (AITER faster in 14/32)
**Roofline (geomean, best AITER):** AITER **57.6%** of HBM peak · TLX **60.8%** of HBM peak

Gluon wins 15/16 comparable shapes (qlen=1) · Triton wins 1/16 (B=8, ctx=8192, D=128, qlen=1)

---

## Summary

MI355X hardware ceilings used for roofline: **2517 TFLOPS** bf16/fp16 matrix (dense), **8.0 TB/s** HBM bandwidth. BW/compute crossover ≈ 314 FLOP/B.

| Kernel | Shapes | HW ceiling | AITER % of ceiling (geomean) | TLX % of ceiling (geomean) | TLX / best tuned AITER (geomean) | AITER faster in |
|---|---|---|---|---|---|---|
| GEMM | 6 | 2517 TFLOPS (compute-bound) | 45.3% | 51.1% | 1.13x | 1/6 |
| Grouped GEMM | 6 | 2517 TFLOPS (compute-bound) | 35.2% | 41.6% | 1.18x | 0/6 |
| Flash attention fwd | 16 | 2517 TFLOPS (mixed regime) | 27.7% | 33.6% | 1.17x | 1/16 |
| Paged-attention decode | 32 | 8.0 TB/s HBM (BW-bound) | 57.6% | 60.8% | 1.05x | 14/32 |

### Roofline notes

- **GEMM / Grouped GEMM** are solidly compute-bound on all tested shapes. Neither implementation comes close to the 2517 TFLOPS ceiling (45% and 51% respectively), which is typical for production kernels of this size — the gap reflects instruction scheduling overhead, register pressure, and wave-quantization effects. TLX's streamk/interwave variants extract ~6pp more than AITER's persistent kernels on large shapes.
- **Flash attention** straddles regimes. Small-S shapes (S≤1024) are bandwidth-bound: AITER wins or ties there (Gluon's tiling is better matched). Large-S D=128 shapes are compute-bound, and TLX's cluster scheduling pulls ahead more noticeably (~44% vs ~37% of peak).
- **Paged attention decode** is bandwidth-bound throughout. Both implementations reach 58–77% of the 8 TB/s ceiling on larger batches — strong efficiency for a decode kernel. The worst outliers are `batch=1, ctx=32768, qlen=4` (AITER 24–31% vs TLX 43–54%), where low parallelism and no split-k path limit occupancy. AITER Gluon beats TLX on 18 of 32 shapes once large-batch / long-context shapes are considered.
