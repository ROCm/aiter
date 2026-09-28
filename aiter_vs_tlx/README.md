# AITER vs TLX Kernel Performance Comparison — MI355

**Task:** Collect perf numbers on MI355 node for GEMM, Paged Attention, Grouped GEMM, and Flash Attention (fwd) kernels in TLX and AITER.

- **TLX repo:** https://github.com/facebookexperimental/triton
- **TLX kernels:** https://github.com/facebookexperimental/triton/tree/main/third_party/tlx/tutorials
- **AITER:** uses Gluon kernel when available, otherwise Triton kernels

---

## 1. GEMM — bf16, Y[M,N] = X[M,K]·Wᵀ (TFLOPS)

| M | N | K | AITER Triton tuned | Config | TLX | TLX variant | TLX / AITER |
|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 2048 | **833** | 128×128×128, 8w | 702 | pipelined | 0.84x |
| 4096 | 4096 | 4096 | 1174 | 256×256×64, 8w, persistent | 1393 | interwave | 1.19x |
| 8192 | 8192 | 8192 | 1230 | 256×256×64, 8w, persistent | 1464 | interwave_streamk | 1.19x |
| 8192 | 10240 | 8192 | 1228 | 256×256×64, 8w, persistent | 1473 | interwave_streamk | 1.20x |
| 8192 | 57344 | 8192 | 1221 | 256×256×64, 8w, persistent | 1467 | interwave_streamk | 1.20x |
| 8192 | 8192 | 28672 | 1220 | 256×256×64, 8w, persistent | 1469 | pingpong | 1.20x |

**Geomean (6 shapes):** AITER 1140 TFLOPS · TLX 1287 TFLOPS · **TLX/AITER 1.13x** (AITER faster in 1/6)

---

## 2. Grouped GEMM — fp16 (TFLOPS)

| M total | K | N | Groups | Rows/group | Group sizes | AITER Triton tuned | Config | TLX | TLX variant | TLX / AITER |
|---|---|---|---|---|---|---|---|---|---|---|
| 65536 | 4096 | 4096 | 16 | 4096 | equal | 1077 | 256×256×64, 8w, work-stealing | 1279 | amd_grouped_gemm | 1.19x |
| 49152 | 1408 | 2048 | 64 | 768 | equal | 766 | 256×256×64, 8w | 866 | amd_grouped_gemm | 1.13x |
| 49152 | 1408 | 2048 | 64 | 768 | uneven | 659 | 128×256×64, 4w, work-stealing | 720 | amd_grouped_gemm | 1.09x |
| 32768 | 6144 | 16384 | 8 | 4096 | equal | 1126 | 256×256×64, 8w, work-stealing | 1373 | amd_grouped_gemm | 1.22x |
| 32768 | 6144 | 16384 | 8 | 4096 | uneven | 1045 | 256×256×64, 8w | 1294 | amd_grouped_gemm | 1.24x |
| 16384 | 7168 | 4096 | 32 | 512 | uneven | 753 | 256×256×64, 8w, work-stealing | 938 | amd_grouped_gemm | 1.24x |

**Geomean (6 shapes):** AITER 885 TFLOPS · TLX 1048 TFLOPS · **TLX/AITER 1.18x** (AITER faster in 0/6)

---

## 3. Flash Attention Forward — bf16, H_q = H_kv, S_q = S_kv (TFLOPS; causal counts ½ FLOPs)

Bold = better of Gluon vs Triton for that shape.

| B | H | S | D | Mask | AITER Gluon tuned | Gluon config | AITER Triton tuned | Triton config | TLX | TLX variant | TLX / best AITER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 1024 | 64 | full | **643** | 256×128, 8w | 555 | 256×64, 8w, 2st | 692 | cluster_persistent | 1.08x |
| 1 | 64 | 4096 | 64 | full | 766 | 256×64, 4w | **782** | 256×128, 8w, 2st | 827 | persistent | 1.06x |
| 1 | 64 | 8192 | 64 | full | **794** | 256×64, 4w | 786 | 128×128, 4w, 2st | 850 | persistent | 1.07x |
| 1 | 64 | 1024 | 128 | full | **711** | 128×64, 4w | 682 | 256×64, 8w, 2st | 853 | cluster | 1.20x |
| 1 | 64 | 4096 | 128 | full | 859 | 256×64, 8w | **927** | 256×64, 8w, 3st | 1103 | cluster | 1.19x |
| 1 | 64 | 8192 | 128 | full | 841 | 128×64, 4w | **937** | 256×64, 8w, 3st | 1150 | cluster | 1.23x |
| 2 | 64 | 8192 | 128 | full | 822 | 256×64, 8w | **952** | 256×64, 8w, 3st | 1152 | cluster | 1.21x |
| 1 | 64 | 16384 | 128 | full | 825 | 256×64, 8w | **960** | 256×64, 8w, 3st | 1168 | cluster | 1.22x |
| 1 | 64 | 1024 | 64 | causal | **378** | 128×64, 4w | 295 | 128×32, 4w, 2st | 313 | cluster | 0.83x |
| 1 | 64 | 4096 | 64 | causal | 615 | 256×64, 4w | **641** | 128×128, 4w, 2st | 727 | persistent | 1.13x |
| 1 | 64 | 8192 | 64 | causal | **700** | 256×64, 4w | 689 | 128×128, 4w, 2st | 808 | persistent | 1.15x |
| 1 | 64 | 1024 | 128 | causal | **429** | 128×64, 4w | 367 | 128×32, 4w, 2st | 456 | cluster | 1.06x |
| 1 | 64 | 4096 | 128 | causal | 695 | 128×64, 4w | **704** | 256×64, 8w, 3st | 953 | cluster | 1.35x |
| 1 | 64 | 8192 | 128 | causal | 772 | 256×64, 8w | **800** | 256×64, 8w, 3st | 1066 | cluster | 1.33x |
| 2 | 64 | 8192 | 128 | causal | 776 | 256×64, 8w | **798** | 256×64, 8w, 3st | 1060 | cluster | 1.33x |
| 1 | 64 | 16384 | 128 | causal | 722 | 128×64, 4w | **805** | 256×64, 8w, 3st | 1100 | cluster | 1.37x |

**Geomean (16 shapes):** AITER Gluon 693 · AITER Triton 698 · TLX 847 · **TLX/best AITER 1.17x** (AITER faster in 1/16)

Gluon wins 6/16 shapes · Triton wins 10/16 shapes

---

## 4. Paged Attention Decode — bf16, page size 16 (TB/s of KV cache read)

Bold = better of Gluon vs Triton for that shape (n/a rows have only Gluon available).

| Batch | Context | Q heads | KV heads | D | Query len | AITER Gluon tuned | Gluon config | AITER Triton tuned | Triton config | TLX | TLX / best AITER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8192 | 64 | 8 | 64 | 1 | **1.96** | ps=True, 32 splits | 1.23 | V2, part 256 | 2.45 | 1.25x |
| 8 | 8192 | 64 | 8 | 64 | 1 | **4.85** | ps=True, 8 splits | 4.26 | V2, part 256 | 5.33 | 1.10x |
| 32 | 8192 | 64 | 8 | 64 | 1 | **5.74** | ps=True, 2 splits | 4.56 | V2, part 1024 | 5.30 | 0.92x |
| 128 | 8192 | 64 | 8 | 64 | 1 | **6.17** | ps=True, 1 split | 4.75 | V2, part 2048 | 5.85 | 0.95x |
| 1 | 32768 | 64 | 8 | 64 | 1 | **3.64** | ps=True, 32 splits | 2.48 | V2, part 512 | 4.25 | 1.17x |
| 8 | 32768 | 64 | 8 | 64 | 1 | **5.72** | ps=True, 8 splits | 4.55 | V2, part 1024 | 5.40 | 0.94x |
| 32 | 32768 | 64 | 8 | 64 | 1 | **6.19** | ps=True, 4 splits | 4.80 | V2, part 2048 | 5.88 | 0.95x |
| 8 | 131072 | 64 | 8 | 64 | 1 | **6.00** | ps=True, 32 splits | 4.81 | V2, part 2048 | 6.21 | 1.03x |
| 1 | 8192 | 64 | 8 | 64 | 4 | **1.54** | ps=False | n/a | no qlen>1 path | 1.50 | 0.97x |
| 8 | 8192 | 64 | 8 | 64 | 4 | **4.36** | ps=False | n/a | no qlen>1 path | 4.52 | 1.04x |
| 32 | 8192 | 64 | 8 | 64 | 4 | **5.44** | ps=True, 2 splits | n/a | no qlen>1 path | 4.92 | 0.90x |
| 128 | 8192 | 64 | 8 | 64 | 4 | **5.88** | ps=True, 1 split | n/a | no qlen>1 path | 5.90 | 1.00x |
| 1 | 32768 | 64 | 8 | 64 | 4 | **1.89** | ps=False | n/a | no qlen>1 path | 3.43 | 1.82x |
| 8 | 32768 | 64 | 8 | 64 | 4 | **5.46** | ps=True, 8 splits | n/a | no qlen>1 path | 5.38 | 0.99x |
| 32 | 32768 | 64 | 8 | 64 | 4 | **5.82** | ps=True, 4 splits | n/a | no qlen>1 path | 5.86 | 1.01x |
| 8 | 131072 | 64 | 8 | 64 | 4 | **5.88** | ps=True, 16 splits | n/a | no qlen>1 path | 5.91 | 1.00x |
| 1 | 8192 | 64 | 8 | 128 | 1 | **3.05** | ps=False | 2.20 | V2, part 256 | 3.10 | 1.02x |
| 8 | 8192 | 64 | 8 | 128 | 1 | 4.95 | ps=True, 4 splits | **5.72** | V2, part 512 | 6.15 | 1.07x |
| 32 | 8192 | 64 | 8 | 128 | 1 | **6.01** | ps=True, 1 split | 5.67 | V2, part 512 | 6.00 | 1.00x |
| 128 | 8192 | 64 | 8 | 128 | 1 | **6.20** | ps=True, 1 split | 5.72 | V2, part 8192 | 6.45 | 1.04x |
| 1 | 32768 | 64 | 8 | 128 | 1 | **4.07** | ps=True, 32 splits | 3.88 | V2, part 512 | 5.26 | 1.29x |
| 8 | 32768 | 64 | 8 | 128 | 1 | **5.75** | ps=True, 4 splits | 5.16 | V2, part 2048 | 5.93 | 1.03x |
| 32 | 32768 | 64 | 8 | 128 | 1 | **6.25** | ps=True, 1 split | 5.62 | V2, part 512 | 6.22 | 0.99x |
| 8 | 131072 | 64 | 8 | 128 | 1 | **6.04** | ps=True, 4 splits | 5.78 | V2, part 8192 | 6.26 | 1.04x |
| 1 | 8192 | 64 | 8 | 128 | 4 | **2.06** | ps=False | n/a | no qlen>1 path | 1.89 | 0.92x |
| 8 | 8192 | 64 | 8 | 128 | 4 | **5.08** | ps=False | n/a | no qlen>1 path | 5.16 | 1.02x |
| 32 | 8192 | 64 | 8 | 128 | 4 | **5.72** | ps=True, 1 split | n/a | no qlen>1 path | 5.55 | 0.97x |
| 128 | 8192 | 64 | 8 | 128 | 4 | **5.77** | ps=True, 1 split | n/a | no qlen>1 path | 5.84 | 1.01x |
| 1 | 32768 | 64 | 8 | 128 | 4 | **2.47** | ps=False | n/a | no qlen>1 path | 4.31 | 1.75x |
| 8 | 32768 | 64 | 8 | 128 | 4 | **5.63** | ps=True, 8 splits | n/a | no qlen>1 path | 5.55 | 0.99x |
| 32 | 32768 | 64 | 8 | 128 | 4 | **5.95** | ps=True, 1 split | n/a | no qlen>1 path | 5.63 | 0.95x |
| 8 | 131072 | 64 | 8 | 128 | 4 | **6.01** | ps=True, 16 splits | n/a | no qlen>1 path | 5.68 | 0.94x |

**Geomean (32 shapes):** AITER Gluon 4.61 TB/s · AITER Triton 4.15 TB/s (16 qlen=1 shapes only) · TLX 4.86 TB/s · **TLX/best AITER 1.05x** (AITER faster in 14/32)

Gluon wins 15/16 comparable shapes (qlen=1) · Triton wins 1/16 (B=8, ctx=8192, D=128, qlen=1)

---

## Summary

| Kernel | Shapes | TLX / best tuned AITER (geomean) | AITER faster in |
|---|---|---|---|
| GEMM | 6 | 1.13x | 1/6 |
| Grouped GEMM | 6 | 1.18x | 0/6 |
| Flash attention fwd | 16 | 1.17x | 1/16 |
| Paged-attention decode | 32 | 1.05x | 14/32 |
