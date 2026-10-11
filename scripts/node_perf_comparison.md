# Node Performance Comparison Report (Oct 10, 2026)

**Docker image:** `rocm/pytorch-private:rocm10.0_mi450mc_test_scxiao`

---

## 1. AITER A16W16 GEMM Kernel (Shape: M=8192, N=6144, K=6144)

| Node | Triton (TFLOPS) | Gluon (TFLOPS) | Triton vs Gluon Speedup |
|------|----------------:|----------------:|------------------------:|
| **MI455** | **1687.88** | 1464.96 | 1.15x |
| MI355 | 1123.49 | — | — |
| SPX A0 | 1111.46 | 983.76 | 1.13x |
| SPX B0 (c11) | 522.14 | 486.69 | 1.07x |
| SPX B0 (c12) | 615.57 | 543.98 | 1.13x |
| DPX/NPS2 | 440.76 | 365.46 | 1.21x |

**Key observations:**
- MI455 delivers the highest GEMM throughput at **1687.88 TFLOPS** (Triton), **2.7x** faster than the DPX/NPS2 node and **1.5x** faster than MI355/SPX A0.
- Triton consistently outperforms Gluon by 7-21% across all nodes.
- The two SPX B0 nodes show significant variance (522 vs 616 TFLOPS), suggesting system-level differences (c12 is ~18% faster than c11).

---

## 2. Vector Copy (Triton vector-add, peak throughput at largest sizes)

| Node | Triton Peak (GB/s) | Torch Peak (GB/s) | Triton vs Torch |
|------|-------------------:|-------------------:|----------------:|
| **MI455** | **17,553** | 15,595 | 1.13x |
| MI355 | 5,533 | 5,532 | 1.00x |
| SPX A0 | 12,878 | 11,920 | 1.08x |
| SPX B0 (c11) | 10,648 | 9,482 | 1.12x |
| SPX B0 (c12) | 11,371 | 9,854 | 1.15x |
| DPX/NPS2 | 7,518 | 6,415 | 1.17x |

*Peak values taken at size=8,388,608 for SPX B0 c11 (peak before drop-off), size=134,217,728 for others.*

**Key observations:**
- MI455 achieves the highest memory bandwidth at **17.6 TB/s** (Triton), **3.2x** the MI355 node.
- SPX A0 and SPX B0 (c12) are competitive at ~11-13 TB/s; c11 is slightly lower at ~10.6 TB/s.
- MI355 shows nearly identical Triton vs Torch bandwidth, while other nodes show Triton advantages of 8-17%.
- DPX/NPS2 bandwidth is limited (~7.5 TB/s), likely due to NPS2 memory partitioning.

---

## 3. Matrix Multiplication (Triton FP16 matmul)

| M=N=K | MI455 (TFLOPS) | MI355 (TFLOPS) | SPX A0 (TFLOPS) | SPX B0 c11 (TFLOPS) | SPX B0 c12 (TFLOPS) | DPX/NPS2 (TFLOPS) |
|------:|---------------:|---------------:|-----------------:|---------------------:|---------------------:|-------------------:|
| 256 | 2.78 | 5.15 | 4.17 | 4.81 | 4.65 | 5.35 |
| 384 | 9.39 | 15.30 | 11.49 | 15.53 | 15.36 | 16.05 |
| 512 | 21.55 | 32.42 | 24.19 | 30.88 | 32.37 | 33.19 |
| 640 | 42.08 | 56.74 | 41.29 | 54.99 | 54.64 | 54.69 |
| 768 | 72.95 | 87.45 | 60.31 | 91.93 | 91.93 | 86.09 |
| 896 | 108.17 | 123.60 | 89.11 | 119.71 | 116.22 | 111.08 |
| 1024 | **160.98** | 169.36 | 118.60 | 163.69 | 163.44 | 152.65 |
| 1280 | **275.54** | 260.84 | 185.97 | 247.52 | 233.71 | 189.35 |
| 1536 | **400.27** | 349.12 | 266.84 | 319.65 | 320.78 | 259.33 |
| 1792 | **543.09** | 439.28 | 352.08 | 385.63 | 388.25 | 249.32 |
| 2048 | **723.18** | 587.54 | 448.13 | 472.05 | 475.45 | 356.14 |
| 2560 | **835.94** | 688.71 | 578.85 | 567.87 | 587.79 | 338.74 |
| 3072 | **848.91** | 671.09 | 788.32 | 661.05 | 668.69 | 415.45 |
| 3584 | **1103.93** | 810.07 | 624.73 | 481.89 | 509.73 | 381.05 |
| 4096 | **1454.35** | 985.08 | 775.33 | 645.21 | 674.63 | 521.54 |

**Key observations:**
- MI455 Triton matmul reaches **1454 TFLOPS** at M=N=K=4096, the highest across all nodes.
- MI455 takes the lead starting at M=N=K=1280 and widens the gap as sizes grow.
- At small sizes (<=1024), MI355, SPX B0 c11, and c12 are competitive with MI455.
- SPX B0 c11 and c12 perform similarly; c12 is slightly ahead at large sizes.
- At large sizes, Triton matmul throughput ranking: MI455 (1454) > MI355 (985) > SPX A0 (775) > SPX B0 c12 (675) > SPX B0 c11 (645) > DPX/NPS2 (522).

---

## 4. Hardware Summary

| Node | Device ID | SCLK | MCLK | Power Cap | Partition |
|------|-----------|------|------|-----------|-----------|
| MI455 | 0x75c1 | 2400 MHz | 1900 MHz | 2500W | NPS1/SPX |
| MI355 (f19-12) | 0x75c7 | 2148-2369 MHz | 1650 MHz | 1600W | NPS1/SPX |

MI455 has ~12% higher SCLK and ~15% higher MCLK compared to MI355, plus a significantly higher power cap (2500W vs 1600W), explaining its bandwidth and compute advantages.

---

## 5. Overall Node Ranking

| Rank | Node | GEMM TFLOPS | Mem BW (GB/s) | Matmul TFLOPS |
|------|------|------------:|--------------:|--------------:|
| 1 | MI455 | 1687.88 | 17,553 | 1,454 |
| 2 | MI355 | 1123.49 | 5,533 | 985 |
| 3 | SPX A0 | 1111.46 | 12,878 | 775 |
| 4 | SPX B0 (c12) | 615.57 | 11,371 | 675 |
| 5 | SPX B0 (c11) | 522.14 | 10,648 | 645 |
| 6 | DPX/NPS2 | 440.76 | 7,518 | 522 |

**MI455 is the clear performance leader** across all three benchmarks, delivering 1.5-3.8x the throughput of other nodes depending on the workload.
