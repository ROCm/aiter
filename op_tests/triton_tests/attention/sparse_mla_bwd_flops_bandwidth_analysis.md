# Sparse MLA Backward — FLOPs & Data Movement Analysis

## Config

```
T     = 4096   (total tokens, batch=1 × seq=4096)
H     = 128    (num attention heads)
D     = 576    (D_V + D_R = 512 + 64)
D_V   = 512    (KV lora rank)
D_R   = 64     (QK rope head dim)
TOPK  = 1024   (KV positions per query)
BH    = 64     (heads per program)
TK    = 16     (KV tokens per tile)
```

Source: `op_tests/triton_tests/attention/bench_bwd_configs.py` line 86-89.

---

## 1. FLOPs Calculation

### Per (token, head, KV_position): 8 dot products

Each query head attends to each of its TOPK KV positions. Per (t, h, k) triplet,
the backward pass computes 8 dot products:

| # | Operation | Operand dims | FLOPs | Purpose |
|---|-----------|-------------|-------|---------|
| 1 | Q_lora · K_lora | [1, D_V] · [D_V, 1] | 2 × 512 = 1024 | S (lora part) |
| 2 | Q_rope · K_rope | [1, D_R] · [D_R, 1] | 2 × 64 = 128 | S (rope part) |
| 3 | dO · K_lora | [1, D_V] · [D_V, 1] | 2 × 512 = 1024 | dP |
| 4 | dS × K_lora | [1, 1] × [1, D_V] | 2 × 512 = 1024 | dQ (lora part) |
| 5 | dS × K_rope | [1, 1] × [1, D_R] | 2 × 64 = 128 | dQ (rope part) |
| 6 | dS × Q_lora | [1, 1] × [1, D_V] | 2 × 512 = 1024 | dKV (lora via dS) |
| 7 | P × dO | [1, 1] × [1, D_V] | 2 × 512 = 1024 | dKV (V gradient) |
| 8 | dS × Q_rope | [1, 1] × [1, D_R] | 2 × 64 = 128 | dKV (rope part) |
| | **Total** | | **5504** | |

Breakdown: 5 dot products over D_V=512, 3 dot products over D_R=64.

```
FLOPs_per_thk = 5 × 2 × D_V + 3 × 2 × D_R
             = 5 × 1024 + 3 × 128
             = 5120 + 384
             = 5504
```

### Total FLOPs

```
Total FLOPs = T × H × TOPK × 5504
            = 4096 × 128 × 1024 × 5504
            = 2,955,337,179,136
            ≈ 2.96 TFLOP
```

### Note on benchmark script FLOP formula

The benchmark (`bench_bwd_configs.py` line 75-78) uses:
```python
flops_bwd = total_tokens * num_heads * topk * (2*d_qk + 2*kv_lora_rank + 2*d_qk + 2*d_qk)
          = T * H * K * (2×576 + 2×512 + 2×576 + 2×576)
          = T * H * K * 4480
```
This undercounts: it misses dot 7 (P^T × dO, worth 2×512=1024 FLOPs) and miscounts
some rope terms. The correct per-element FLOP count is 5504, not 4480.

---

## 2. Data Movement — Per Program

One program = one token × BH=64 heads × all TOPK/TK=64 tiles.
Number of programs = T × ceil(H / BH) = 4096 × 2 = 8192.

### 2a. Reads from HBM

**Prologue (loaded once, reused across all 64 tiles):**

| Data | Shape | Element size | Bytes | Notes |
|------|-------|-------------|-------|-------|
| Q_lora | [BH=64, D_V=512] | bf16 | 65,536 | Stays in VGPRs for dots 1,6 |
| Q_rope | [BH=64, D_R=64] | bf16 | 8,192 | Stays in VGPRs for dots 2,8 |
| dO | [BH=64, D_V=512] | bf16 | 65,536 | Stays in VGPRs for dots 3,7 |
| LSE | [BH=64] | fp32 | 256 | Softmax normalizer |
| Delta | [BH=64] | fp32 | 256 | = sum(O * dO) |
| **Subtotal (NEW/tl.trans)** | | | **139,776** | |
| Q_T (OLD only) | [D=576, BH=64] | bf16 | 73,728 | Pre-transposed Q |
| dO_T (OLD only) | [D_V=512, BH=64] | bf16 | 65,536 | Pre-transposed dO |
| **Subtotal (OLD)** | | | **279,040** | |

OLD kernel loads 2× more prologue data (Q_T and dO_T for dots 6-8).

**Per-tile (64 tiles per program):**

| Data | Shape | Element size | Bytes/tile | Bytes total (×64) |
|------|-------|-------------|-----------|-------------------|
| KV tile | [D=576, TK=16] | bf16 | 18,432 | 1,179,648 |
| TopK indices | [TK=16] | int32 | 64 | 4,096 |
| **Subtotal** | | | **18,496** | **1,183,744** |

KV tiles are the dominant per-tile cost. Note: each KV position is identified by
a TopK index pointing to an arbitrary row in the KV cache (gathered access).

**Total reads per program:**

| Kernel variant | Prologue | Tiles | Total |
|----------------|----------|-------|-------|
| OLD (Q_T, dO_T) | 279,040 | 1,183,744 | 1,462,784 (1.39 MiB) |
| NEW (tl.trans) | 139,776 | 1,183,744 | 1,323,520 (1.26 MiB) |

### 2b. Writes to HBM

**Epilogue (once per program):**

| Data | Shape | Element size | Bytes |
|------|-------|-------------|-------|
| dQ | [BH=64, D=576] | bf16 | 73,728 |

**Per-tile atomicAdd (64 tiles per program):**

| Data | Shape | Element size | Bytes/tile | Bytes total (×64) |
|------|-------|-------------|-----------|-------------------|
| dKV_lora | [TK=16, D_V=512] | fp32 | 32,768 | 2,097,152 |
| dKV_rope | [TK=16, D_R=64] | fp32 | 4,096 | 262,144 |
| **Subtotal** | | | **36,864** | **2,359,296** |

AtomicAdd is a read-modify-write (RMW): each operation reads the old value, adds,
and writes the new value. So atomicAdd generates BOTH read and write traffic.

**Total writes per program:**

```
dQ store:          73,728 bytes
dKV atomicAdd:  2,359,296 bytes (write portion)
                2,359,296 bytes (read portion, counted separately)
Total write:    2,433,024 bytes (2.32 MiB)
Total RMW read: 2,359,296 bytes (2.25 MiB)
```

---

## 3. Total Data Movement (T=4096, H=128, TOPK=1024)

8192 programs total. Using the OLD kernel (current best performance):

| Category | Per program | Total | GiB |
|----------|------------|-------|-----|
| Prologue reads (Q, dO, Q_T, dO_T, LSE, Delta) | 279,040 | 2,285,895,680 | 2.13 |
| KV tile reads (64 tiles × 18,496) | 1,183,744 | 9,696,862,208 | 9.03 |
| TopK index reads | 4,096 | 33,554,432 | 0.03 |
| **Total pure reads** | **1,462,784** | **11,982,757,888** | **11.16** |
| dKV atomicAdd reads (RMW) | 2,359,296 | 19,327,352,832 | 18.00 |
| **Grand total reads** | **3,822,080** | **31,310,110,720** | **29.16** |
| | | | |
| dQ writes | 73,728 | 603,979,776 | 0.56 |
| dKV atomicAdd writes | 2,359,296 | 19,327,352,832 | 18.00 |
| **Total writes** | **2,433,024** | **19,931,332,608** | **18.56** |
| | | | |
| **Total HBM traffic** | **6,255,104** | **51,241,443,328** | **47.72** |

### Traffic breakdown by source

```
KV tile reads:        9.03 GiB  (18.9%)  — loaded per tile, gathered
dKV atomicAdd RMW:   36.00 GiB  (75.5%)  — read + write combined
Prologue reads:       2.13 GiB   (4.5%)  — Q, dO, Q_T, dO_T, LSE, Delta
dQ writes:            0.56 GiB   (1.2%)  — epilogue store
                     ─────────
Total:               47.72 GiB  (100%)
```

**dKV atomicAdd dominates at 75.5% of total traffic.**

### KV tile reuse across head groups

Each token has H/BH = 128/64 = 2 programs (2 head groups). Both programs load the
same KV tiles (same TopK indices). Unique KV data per token:

```
Unique KV per token = TOPK × D × 2 = 1024 × 576 × 2 = 1.125 MiB
Total unique KV     = T × 1.125 MiB = 4.5 GiB
Loaded KV           = 9.03 GiB (2× due to 2 head groups)
```

The second head group's KV loads may hit L2 cache (MI300X L2 = 256 MB total,
32 MB per XCD). Whether this happens depends on scheduling — if both head groups
for the same token run on the same XCD close in time, L2 reuse is likely.

---

## 4. Arithmetic Intensity and Bound Analysis

```
Arithmetic Intensity = Total FLOPs / Total HBM traffic
                     = 2.96 TFLOP / 47.72 GiB
                     = 2.96 × 10^12 / 51.24 × 10^9
                     ≈ 57.8 FLOP/byte
```

### MI300X roofline thresholds

```
Peak bf16 MFMA:  1300 TFLOPS (approx, all 304 CUs)
Peak HBM BW:     5.3 TB/s
Compute-bound threshold: 1300 / 5.3 ≈ 245 FLOP/byte
```

At 57.8 FLOP/byte, **the kernel is memory-bound** (4.2× below the compute-bound
threshold).

### If dKV atomicAdd could be eliminated

Without atomicAdd traffic (hypothetical):
```
Traffic = 47.72 - 36.00 = 11.72 GiB
AI = 2.96 TFLOP / 11.72 GiB = 2.96e12 / 12.58e9 ≈ 235 FLOP/byte
```
This would bring the kernel close to compute-bound (~235 vs threshold ~245).
The atomicAdd is the single biggest bottleneck.

### Observed performance

From benchmark (OLD kernel, BH=64 TK=16 w=4 s=2):
```
Time:    58.3 ms
TFLOPS:  41.2 (using script's undercounted 4480 formula)
         50.7 (using correct 5504 formula)

Effective BW = 47.72 GiB / 58.3 ms = 47.72 / 0.0583 = 818 GiB/s ≈ 0.82 TB/s
HBM utilization = 0.82 / 5.3 = 15.5%
```

The low HBM utilization (15.5%) suggests the kernel is not purely bandwidth-limited
either — VGPR spills (105 scratch load/store per spill) and atomicAdd contention
add latency beyond simple bandwidth cost.

---

## 5. Summary

| Metric | Value |
|--------|-------|
| Total FLOPs | 2.96 TFLOP |
| Total HBM traffic | 47.72 GiB |
| Arithmetic intensity | 57.8 FLOP/byte |
| Memory-bound? | Yes (threshold ~245 FLOP/byte) |
| Dominant traffic | dKV atomicAdd: 75.5% of total |
| Observed HBM utilization | ~15.5% |
| Key bottlenecks | (1) dKV atomicAdd contention, (2) VGPR spills |

---

## 6. Ablation Study — Isolating Performance Factors

Source: `op_tests/triton_tests/attention/profile_bwd_ablation.py`

To understand the 6x gap between roofline (9.67 ms) and actual (58.42 ms), we ran
modified kernels with specific features disabled. All use BH=64 TK=16 w=4 s=2 unless
noted. Results from MI300X inside Docker container `yewang_triton_dsa`.

### Ablation variants

| ID | Variant | What's disabled | What remains |
|----|---------|----------------|-------------|
| A | Baseline (BH=64) | Nothing | All 8 dots + atomicAdd (105 spills) |
| B | No-spill (BH=32) | Spills (0 spills) | All 8 dots + atomicAdd, half head reuse |
| C | No atomicAdd | dKV scatter removed | All 8 dots computed, dKV discarded |
| D | dQ-only (BH=64) | Dots 6-8, Q_T/dO_T loads, atomicAdd | Dots 1-5 only |
| D2 | dQ-only (BH=32) | Same as D, + no spills | Dots 1-5, 0 spills |
| F | S-P-dS only (BH=64) | Dots 4-8, all dQ/dKV | Dots 1-3 + softmax |
| F2 | S-P-dS only (BH=32) | Same as F, + no spills | Dots 1-3, 0 spills |

### Results

```
Ablation                                              ms      vs A
──────────────────────────────────────────────   ────────   ───────
A.  Baseline (BH=64, full, 105 spills)            58.42    100.0%
B.  No-spill baseline (BH=32, 0 spills)          112.87    193.2%
C.  No atomicAdd (BH=64, full compute)              4.87      8.3%
D.  dQ-only (BH=64, dots 1-5)                       4.87      8.3%
D2. dQ-only (BH=32, dots 1-5)                       7.87     13.5%
F.  S-P-dS only (BH=64, dots 1-3)                   3.02      5.2%
F2. S-P-dS only (BH=32, dots 1-3)                   4.48      7.7%
```

### ISA validation: DCE check

Before interpreting results, we verified whether Triton's dead code elimination (DCE)
affected ablation C. Source: `check_ablation_isa.py`.

```
Variant              MFMAs   VGPRs  AGPRs  Spills  global_atomic  scratch_load
────────────────   ──────   ─────  ─────  ──────  ─────────────  ────────────
A. Baseline           308     512    256     105              0            64
C. No atomicAdd       172     423    167       0              0             0
D. dQ-only            172     423    167       0              0             0
F. S-P-dS only        136     228      8       0              0             0
```

**C and D are IDENTICAL** (same MFMA count, same register usage, same everything).
Triton DCE'd the entire dKV computation (dots 6-8) in ablation C because the result
was never stored. This means **ablation C is NOT a valid "compute dKV but skip scatter"
test** — it's effectively the same as dQ-only.

**Consequence**: We CANNOT separate "dKV compute cost" from "atomicAdd cost" using
ablation C. The measured A − C = 53.55 ms includes BOTH dKV compute and atomicAdd.

MFMA breakdown across ablations:
- F → D: +36 MFMAs (dots 4-5: dQ accumulation)
- D → A: +136 MFMAs (dots 6-8: dKV compute + supporting dKV ops)

### Key findings

**1. dKV path (compute + atomicAdd + spills) is 91.7% of total runtime.**

```
Cost of dKV path = A - D = 58.42 - 4.87 = 53.55 ms  (91.7%)
```

This includes dots 6-8 compute (136 MFMAs), atomicAdd scatter (4.83B ops), Q_T/dO_T
prologue loads, and VGPR spills (105 spills caused by dKV path register pressure).
We cannot decompose this further from timing alone because Triton DCE'd the
compute-without-scatter variant.

However, the dKV compute cannot be more than a few ms (136 MFMAs at peak throughput ≈
0.3 ms), so the vast majority of the 53.55 ms is atomicAdd + spill latency.

**2. dQ path (dots 1-5) takes only 4.87 ms (8.3%).**

```
dQ-only (D):  4.87 ms  — dots 1-5, no spills (172 MFMAs, 0 spills)
```

All of S, P, dS recomputation plus dQ accumulation fits in 4.87 ms with zero spills.
The dKV path is what causes register pressure → spills → performance collapse.

**3. dQ accumulation (dots 4-5) costs 1.84 ms.**

```
Cost of dQ = D - F = 4.87 - 3.02 = 1.84 ms  (3.2%)
```

Two extra matmuls ([BH,TK]×[TK,D_V] and [BH,TK]×[TK,D_R]) per tile (+36 MFMAs).

**4. S + P + dS baseline is 3.02 ms (5.2%).**

This includes dots 1-3, softmax recomputation, and KV tile loading from global memory.
This is the irreducible compute floor for the attention backward pass (136 MFMAs).

**5. VGPR spills actually HELP at BH=64 vs BH=32.**

```
BH=64 (105 spills):  58.42 ms
BH=32 (0 spills):   112.87 ms  ← 1.93× SLOWER
```

Counter-intuitive: BH=32 eliminates all spills but is 2× slower because:
- Half the head reuse: each KV tile loaded once per 32 heads instead of 64
- 2× more programs: 4096×4=16384 programs vs 4096×2=8192
- 2× more atomicAdd operations to the same dKV positions
- The spill cost at BH=64 (~105 scratch ops) is dwarfed by the atomicAdd cost

**6. Derived cost breakdown for BH=64:**

```
Component                          ms        % of total    Note
─────────────────────────────   ────────    ──────────    ─────────────────
S + P + dS (dots 1-3)              3.02         5.2%     136 MFMAs
dQ accumulation (dots 4-5)          1.84         3.2%     +36 MFMAs
dKV path (dots 6-8 + atomic)       53.55        91.7%    +136 MFMAs + 4.83B atomics
─────────────────────────────   ────────    ──────────
Total                              58.42       100.0%     308 MFMAs
```

Note: dKV compute is estimated at ~0.3 ms (136 MFMAs at peak), so atomicAdd
scatter accounts for ~53 ms of the 53.55 ms dKV path cost.

---

## 7. Hardware Counter Profiling (rocprof)

Source: rocprof with PMC counters on the baseline kernel (BH=64, TK=16, w=4, s=2).

### Kernel resource usage (from rocprof CSV)

```
Grid:        2,097,152 threads  (= 4096 tokens × 2 head_groups × 256 threads/block)
Block:       256 threads (= 4 warps × 64 lanes)
LDS:         65,536 bytes (64 KB per workgroup)
Scratch:     424 bytes per thread
arch_vgpr:   128
accum_vgpr:  384
SGPRs:       64
Wave_size:   64
SQ_WAVES:    32,768 total waves (= 8192 workgroups × 4 warps)
```

Note: arch_vgpr=128 + accum_vgpr=384 means the kernel uses 128 regular VGPRs and
384 accumulator VGPRs (AGPRs). Combined = 512, exceeding 256+256 budget → explains
the 105 spills reported by Triton (424 bytes scratch = 106 fp32 values ≈ 105 spills).

### Memory traffic counters (TCP = L1 cache level)

```
TCP_TOTAL_READ_sum:                14,112,784,384  cache lines
TCP_TOTAL_WRITE_sum:                  281,542,656  cache lines
TCP_TOTAL_ATOMIC_WITHOUT_RET_sum:   4,831,838,208  atomic ops

(TCP_TOTAL_ATOMIC_WITH_RET_sum = 0 → all atomics are fire-and-forget, no return value)
```

### HBM traffic (FETCH_SIZE / WRITE_SIZE in KB)

```
FETCH_SIZE:  ~85,042,000 KB  ≈ 81.1 GiB  (total HBM reads)
WRITE_SIZE:   ~2,951,259 KB  ≈  2.8 GiB  (total HBM writes)
```

### Analysis of measured vs predicted traffic

**Measured HBM reads: 81.1 GiB vs predicted 29.2 GiB (2.8× more)**

The excess comes from:
- Scratch spills: 424 bytes/thread × 32,768 waves × 64 threads = ~895 MiB of scratch
  space, with each spill location accessed multiple times during the kernel
- atomicAdd read-modify-write: each fp32 atomic requires reading the old value, which
  shows up in FETCH_SIZE. With 4.83 billion atomic ops × 4 bytes = 19.3 GiB of atomic
  reads alone
- L2 cache misses: gathered KV access with random TopK indices causes poor cache
  utilization, potentially re-fetching cache lines

**Measured HBM writes: 2.8 GiB vs predicted 18.6 GiB**

The write traffic is much lower than predicted because:
- L2 cache write coalescing: many atomicAdd ops to nearby addresses are coalesced in
  L2 before being flushed to HBM
- The 4.83 billion atomic ops target only T×D = 4096×576 = 2.36M unique fp32 locations
  (9.4 MiB). Many atomics to the same address are absorbed by L2 write-combining

**Atomic operation count: 4.83 billion**

```
Predicted:  T × (H/BH) × TOPK × D = 4096 × 2 × 1024 × 576 = 4,831,838,208
Measured:   TCP_TOTAL_ATOMIC_WITHOUT_RET_sum = 4,831,838,208
```

Exact match. Every program issues exactly TOPK × D = 589,824 atomicAdd operations.

### Instruction counts

```
SQ_INSTS_FLAT:  62,259,200  (flat global memory instructions per SIMD)
SQ_INSTS_SMEM:     425,984  (scalar memory instructions)
SQ_WAVES:           32,768  (total waves launched)
```

Flat instructions per wave: 62,259,200 / 32,768 ≈ 1,900 flat memory ops per wave.
This includes all global loads, stores, and atomics.

---

## 8. Root Cause Summary

The 6x roofline gap is dominated by the **dKV path** (compute + atomicAdd scatter +
spills), which accounts for 91.7% of runtime. Due to Triton DCE, we cannot perfectly
separate dKV compute from atomicAdd, but MFMA analysis strongly suggests atomicAdd
is the dominant factor within that 91.7%.

```
                              Roofline    Actual     Gap
                              ────────   ────────   ─────
Compute time:                  2.27 ms                    (not the bottleneck)
Memory BW time:                9.67 ms                    (if all traffic were streaming)
Actual kernel time:                       58.42 ms   6.0×

dKV path (compute+atomic+spill):          53.55 ms   91.7% of runtime
  ├─ dKV compute (estimated):              ~0.3 ms   (136 MFMAs at peak)
  ├─ VGPR spills (estimated):              ~1-2 ms   (105 spills, 424B scratch/thread)
  └─ atomicAdd (estimated):               ~51-52 ms  (4.83B atomic ops, by subtraction)
dQ path (dots 1-5):                        4.87 ms    8.3% of runtime
```

**Why atomicAdd is so expensive:**

1. **Sheer volume**: 4.83 billion atomicAdd ops (confirmed by rocprof: exact match
   with prediction), each a 4-byte read-modify-write
2. **Contention**: Many programs write to the same dKV positions simultaneously.
   With TOPK=1024, each KV token receives updates from many query tokens across
   all heads. The L2 cache serializes conflicting atomics to the same address.
3. **Random access pattern**: TopK indices are essentially random, so atomic targets
   are scattered across the entire dKV buffer (~9.4 MiB), causing poor L2 locality.

**Implications for optimization:**

- Eliminating VGPR spills (e.g., via v17 HIP kernel) would save only ~1-2 ms
  because spills are hidden behind atomicAdd latency
- The most impactful optimization would reduce atomic contention: e.g., partial dKV
  accumulation in LDS/registers before atomic scatter, or restructuring the algorithm
  to compute dKV with fewer atomic collisions
- BH=64 with spills is still better than BH=32 without spills, because doubling
  head reuse halves atomic traffic — confirming atomicAdd as the binding constraint
- Triton DCE complicates ablation studies: any future ablation that computes a value
  without storing it will be DCE'd. Use `tl.debug_barrier()` or store to a dummy
  buffer to prevent this.

---

## 9. Contention Reduction Experiments (FAILED)

Source: `bench_bwd_per_hg_dkv.py`, `bench_bwd_token_group_dkv.py`

### Hypothesis

The atomicAdd bottleneck is due to address contention — many programs writing to the
same dKV element simultaneously, serializing at L2.

### Experiment: Per-head-group dKV buffers

Allocate `dKV[num_hg, T, D]` instead of `dKV[T, D]`. Each head group writes to its
own slice. Reduces contention from ~2048 to ~1024 atomics per element.

```
Result: 58.38 ms → 58.50 ms  (no improvement)
```

### Experiment: Token-group partitioning

Partition query tokens into G groups, each with its own dKV buffer. Reduces cross-token
contention by G×. Tested G = 1, 2, 4, 8, 16, 32.

```
  G   atomics/elem   kernel ms   reduce ms   total ms   speedup
  1       ~2048         58.52       0.00       58.52      1.00x
  2       ~1024         58.47       0.29       58.76      1.00x
  4        ~512         58.58       0.02       58.60      1.00x
  8        ~256         58.62       0.03       58.65      1.00x
 16        ~128         58.61       0.04       58.65      1.00x
 32         ~64         58.99       0.08       59.07      0.99x
```

### Conclusion

**Contention is NOT the bottleneck.** Reducing atomics per address from 2048 down to
64 (32× reduction) produces zero improvement. The bottleneck is the **total volume**
of 4.83 billion atomic operations saturating L2 atomic throughput, regardless of
whether they collide.

To improve performance, the algorithm must fundamentally **reduce the number of
atomicAdd operations issued**, not just spread them across more addresses.

Possible approaches:
1. Accumulate partial dKV across multiple KV tiles before scattering (fewer, larger atomics)
2. Split dQ and dKV into separate kernels with different tiling strategies
3. Reverse the iteration order for dKV: iterate queries per KV position (gather pattern,
   direct accumulation, zero atomics) instead of KV per query (scatter pattern)

---

## 10. Split dQ + Gather-Based dKV Experiment (FAILED)

Source: `bench_bwd_split_dq_dkv.py`

### Approach

Split the fused backward into three steps:

1. **dQ kernel** (dots 1-5): same as ablation D, but also stores intermediates
   dS[T, H, TOPK] and P[T, H, TOPK] in bf16 to global memory. These are needed
   by the dKV kernel.

2. **Inverted TopK index build** (torch.argsort + bincount): construct a CSR-format
   index mapping each KV position to the list of (query_token, topk_slot) pairs
   that reference it.

3. **dKV gather kernel**: for each KV position k, iterate over all references from
   the inverted index, load Q[t,:,:] and dO[t,:,:] for each referencing token t,
   accumulate dKV[k,:] in registers directly — zero atomicAdd operations.

### Intermediate buffer sizes

```
dS: [4096, 128, 1024] bf16 = 1.000 GiB
P:  [4096, 128, 1024] bf16 = 1.000 GiB
Total: 2.000 GiB
```

### Results

```
Component                 ms       Notes
─────────────────    ─────────    ──────────────────────────────────
dQ kernel               5.63     dots 1-5 + store dS/P to global
Inverted index           1.00     torch argsort + bincount (excl. first call)
dKV gather (A)         311.99     multi-program, head-group atomics
dKV tiled  (B)         324.00     single-program, tiled heads, true zero atomics
─────────────────    ─────────
Split total (A)        318.62     5.5× SLOWER than baseline
Split total (B)        330.63     5.7× SLOWER than baseline
Baseline (fused)        58.44
```

### Why this failed: data movement explosion

The gather approach reverses the iteration order but dramatically increases data
movement. Each KV position has on average 1024 references (confirmed: mean=1024,
min=921, max=1144 with random indices). For each reference, the dKV kernel must
load:

```
Per reference, per head-group (BLOCK_H=64):
  Q[64, 576]:     64 × 576 × 2 = 73,728 bytes  (bf16)
  dO[64, 512]:    64 × 512 × 2 = 65,536 bytes  (bf16)
  dS[64]:         64 × 2 = 128 bytes            (bf16)
  P[64]:          64 × 2 = 128 bytes            (bf16)
  ─────────────────────────────────────────────
  Total per ref per HG: 139,520 bytes

Total data movement:
  4,194,304 refs × 2 head_groups × 139,520 bytes = 1,090 GiB
```

Compare to the baseline's 47.7 GiB — the gather approach loads **22.8× more data**.
Even at the theoretical 5.3 TB/s HBM bandwidth, the minimum time is 221 ms.

The fundamental problem: in the scatter pattern, each program loads Q/dO once (for
its query token) and iterates over TOPK KV positions. In the gather pattern, each
program loads Q/dO for every referencing query token (avg 1024 times). The Q/dO data
cannot be reused across references because each reference points to a different token.

### Correctness

The split approach is numerically correct:
```
dQ:  max_rel_diff = 0.0 (exact match — same kernel)
dKV: max_rel_diff = 2.4e-6 (bf16 intermediate rounding)
```

### Inverted index statistics

With random TopK indices (uniform over T=4096 tokens):
```
Refs per KV position: min=921, max=1144, mean=1024.0
Total references: 4,194,304 (= T × TOPK = 4096 × 1024)
```

The near-uniform distribution means L2 cache cannot help — every KV position
pulls data from nearly all query tokens.

### Conclusion

The gather approach eliminates atomicAdd entirely but replaces it with 23× more
global memory traffic. For this problem shape (D=576, H=128, TOPK=1024), the data
amplification far outweighs the atomic elimination benefit.

**The scatter pattern is inherently better for this workload** because:
1. Q/dO are large (576×128 = 73K bf16 per token) and loaded once per program
2. TOPK=1024 means each KV position is referenced ~1024 times → 1024 Q/dO loads
3. No Q/dO sharing between references (different tokens)

The only viable path to reducing the 58 ms runtime appears to be:
- Reducing the total number of atomic operations (e.g., partial accumulation across
  tiles before scattering)
- Or using a fundamentally different algorithm (e.g., flash-attention-style tiling
  that processes both Q and KV in tiles with controlled data reuse)

---

## 11. Split dQ + dKV with Head-Group Fusion (SUCCESS)

Source: `bench_bwd_dkv_hg_fused.py`

### Approach

Split the fused backward into two kernels, then apply head-group fusion to the dKV
kernel only (where it's trivial because no dQ accumulators are needed):

1. **dQ kernel** (grid T×2): dots 1-5, stores dS and P intermediates to global bf16.
   Same as ablation D plus intermediate storage. Zero spills.

2. **dKV kernel with HG fusion** (grid T×1): for each tile of TOPK positions, loop
   over NUM_HG=2 head groups, accumulate dKV from both HGs, then scatter ONCE.
   Halves atomic count: 2.42B instead of 4.83B. Zero spills.

### Why HG fusion works in the split kernel but not in the fused kernel

In the fused kernel, dQ accumulators ([64, 576] fp32 per HG) must persist across
all tiles. Two HGs of dQ accumulators need 288 AGPRs (> 256 limit). Impossible.

In the split dKV kernel, there are NO dQ accumulators. The only persistent state
is dKV_acc[D, TK] fp32, which is tiny (~36 AGPRs). Both HGs are processed
sequentially with the same register space.

### ISA analysis

```
Kernel                   VGPRs  AGPRs  Spills  MFMAs  scratch_load  scratch_store
─────────────────────   ─────  ─────  ──────  ─────  ────────────  ─────────────
Baseline fused             512    256     105    308            64             36
dQ + intermediates         430    174       0    172             0              0
dKV HG-fused (TK=16)      282     54       0     68             0              0
dKV HG-fused (TK=64)      477    221       0    136             0              0
```

Both split kernels have ZERO spills. The dKV kernel is especially register-light.

### Results

Best dKV kernel config: TK=64, num_warps=4, num_stages=1.

```
Component                ms       Notes
───────────────────   ────────   ────────────────────────────
dQ kernel               5.68    dots 1-5, store dS/P (0 spills)
dKV HG-fused           29.13    dots 6-8, HG-fused scatter (0 spills)
───────────────────   ────────
Split total             34.81
Baseline (fused)        58.38
Speedup                 1.68×
```

### Config sweep for dKV kernel

```
  TK   warps    dKV ms    total ms    speedup
  16     1      77.25      82.93      0.70x
  16     2      39.85      45.53      1.28x
  16     4      35.35      41.03      1.42x
  32     2      32.16      37.84      1.54x
  64     2      35.76      41.44      1.41x
  64     4      29.13      34.81      1.68x  ← best
 128     4      33.45      39.13      1.49x
```

TK=64 with 4 warps is optimal. Larger TK reduces scatter operations per tile (fewer,
larger batches), and 4 warps provide enough parallelism to hide latency.

### Correctness

```
dQ:  max_abs=0.0  max_rel=0.0  (exact — same kernel as baseline dQ path)
dKV: max_abs=1.26e-5  max_rel=2.58e-6  (bf16 intermediate rounding)
```

PASS at both TK=16 and TK=64.

### Atomic analysis

```
Baseline atomics:  4,831,838,208  (4.83B)
Split HG-fused:    2,415,919,104  (2.42B)
Reduction:         2.0×

dKV kernel time:    29.13 ms  (measured)
Predicted atomic:   26.5 ms   (2.42B / 91 Gops/s)
Overhead:           2.6 ms    (Q_T/dO_T L2 reload + compute)
```

### Memory cost

Intermediate storage for dS and P:
```
dS: [4096, 128, 1024] bf16 = 1.0 GiB
P:  [4096, 128, 1024] bf16 = 1.0 GiB
Total: 2.0 GiB extra (fits in MI300X 192 GiB HBM)
```

### Summary

Head-group fusion in a separate dKV kernel achieves **1.68× speedup** (34.8 ms vs
58.4 ms) by halving atomic operations from 4.83B to 2.42B. The split architecture
is key: it separates dQ (register-heavy, no atomics) from dKV (atomic-heavy, no
dQ registers), allowing each kernel to be optimized independently.

Further improvements:
- bf16 packed atomics could give another 2× atomic reduction (if Triton supports it)
- The dQ kernel could potentially be further optimized (currently 5.68 ms)
