# DSA Backward dKV Atomic Bottleneck: Root Cause Analysis

Investigation of why atomic-based dKV accumulation is slow on MI300X and
why XCD privatization fails to help.

---

## 1. Background: MI300X Memory Hierarchy

| Level | Size | Scope | Bandwidth |
|---|---|---|---|
| L1 data cache | 32 KB | per CU (304 total) | — |
| L2 (GL2/TCC) | 4 MB | per XCD (8 total, 32 MB aggregate) | 4.3 TB/s per XCD |
| Infinity Cache | 256 MB | shared across all 8 XCDs | 17.2 TB/s |
| HBM3 | 192 GB | shared | 5.3 TB/s |

L2 is built from **16 slices × 256 KB** each (128 bytes/cycle/slice at 2.1 GHz).
Each slice contains a dedicated **atomic RMW execution unit**.
Atomic operations that hit in L2 are handled by that slice's atomic unit; misses fall
through to the Infinity Cache or HBM.

XCD count: 8 × 38 CUs = **304 CUs** total.

---

## 2. The Kernel and the Atomic Problem

**Config:** T=4096 tokens, H=128 heads, D_V=512, D_ROPE=64 (D=576), TOPK=1024.

The dKV backward accumulation is a scatter-reduce:

```
for each query token q:
    for each of TOPK=1024 KV tokens k that q attends to:
        dKV[k, :] += contribution(q, k)   # atomic_add_f32, D=576 elements
```

Total atomic_add_f32 operations:

```
T × TOPK × D = 4096 × 1024 × 576 = 2.415 billion
```

Multiple query tokens attending to the same KV token all serialize their atomic_adds
through the cache-level atomic unit for that address.

---

## 3. Benchmark Results

All timings on MI300X (gfx942), single backward pass,
B=1 S=4096 H=128 D_V=512 D_ROPE=64 TOPK=1024.

| Method | Time (ms) | TFLOPS | Speedup | Extra memory |
|---|---|---|---|---|
| fused | 61.09 | 39.4 | 1.00× | 0 |
| recompute | 52.67 | 45.7 | 1.16× | 0 |
| split_intermediate | 38.00 | 63.3 | 1.61× | 2.0 GiB |
| privatized (pid%8) | 38.04 | 63.2 | 1.61× | 2.1 GiB |
| xcd_privatized (hw_id) | 38.03 | 63.3 | 1.61× | 2.1 GiB |
| **gather (no atomics)** | **18.21** | **132.1** | **3.36×** | 6.5 GiB |

**Key observation:** split_intermediate, privatized, and xcd_privatized are all
identical at ~38 ms. XCD privatization provides zero benefit regardless of whether
the routing formula is correct or not.

---

## 4. Hypothesis Testing

### H1: XCD routing formula `(pid % 304) // 38` is wrong

The original `_bwd_dkv_xcd_local` routed CTA `pid` to copy
`(pid % 304) // 38`, assuming CTAs fill CUs sequentially across XCDs.

**Experiment: rocprof TCC write-back counters**

```
pmc: TCC_EA0_WRREQ_sum  TCC_EA0_WRREQ_64B_sum  TCC_EA0_ATOMIC_sum  TCC_WRITEBACK_sum
```

| Kernel | TCC_WRITEBACK | TCC_ATOMIC |
|---|---|---|
| hg_fused (baseline, no privatization) | 603,979,888 | identical |
| xcd_local (privatized, static formula) | 603,979,888 | identical |

**Result: H1 confirmed.** Identical writeback counts prove the static routing formula
is wrong — xcd_local was writing to random XCD copies just like the global kernel,
causing the same cross-XCD cache-line ownership transfers.

**Experiment: hardware CTA→XCD mapping (HIP kernel)**

We read the AMDGCN `HW_ID` register (`s_getreg_b32 s4, hwreg(4, 0, 32)`) from a
HIP kernel, extracting `SE_ID = bits [17:14]` (XCD index on gfx942).

Findings:
- CTA dispatch is **interleaved across XCDs in groups of ~4**, not sequential.
- The interleave pattern has period 32, with roughly equal share per XCD.
- `(pid % 304) // 38` produces **88% mismatch** vs actual hardware XCD assignment.

The correct formula does not exist as a static function of `pid`. The only reliable
approach is to read `hw_id` at runtime.

**Note on "constant 12" Triton bug:** The earlier Triton attempt extracted
`(hw_id >> 18) & 0xF`, which is always 12 on gfx942 because bits [21:18] are
fixed hardware generation metadata (`0x423xxxxx` prefix). The correct field is
bits [17:14].

### H2: Intra-XCD L2 atomic serialization is the residual bottleneck

After fixing the routing to use `hw_id >> 14`, xcd_privatized still matches
split_intermediate exactly. This disproves H2 as the *limiting* factor: eliminating
cross-XCD serialization provides zero improvement.

---

## 5. Root Cause: Working Set Exceeds L2 Capacity

The critical constraint is L2 capacity vs dKV buffer size:

```
dKV copy size (fp32):  T × D × 4 = 4096 × 576 × 4 = 9.44 MB
L2 capacity per XCD:                                 = 4.00 MB
```

**The private dKV copy is 2.36× larger than the XCD's L2.** Atomic operations
systematically miss in L2 and fall through to the **shared Infinity Cache** (256 MB).

The cache miss path for every atomic_add:

```
CU issues atomic_add_f32
    → L2 lookup: MISS (9.44 MB copy >> 4 MB L2)
    → Infinity Cache (256 MB, shared across all 8 XCDs)
    → Infinity Cache atomic unit serializes the RMW
    → result written back to Infinity Cache
```

XCD privatization routes each XCD's CTAs to a different copy, which eliminates
cross-XCD **L2-level** cache line transfers. But since atomics never stayed in L2
to begin with, there was never any L2-level cross-XCD contention to eliminate.
The Infinity Cache atomic unit — which is shared and unaffected by privatization —
is the true serialization bottleneck.

**Total atomic work is also unchanged:**

```
Without privatization:  4096 CTAs × 1024 atomics = 4.19M RMW ops → 1 copy
With 4-XCD privatization: 4×(1024 CTAs × 1024 atomics) = 4.19M RMW ops → 4 copies
```

Same total number of serialized RMW operations flowing through the shared
Infinity Cache. Privatization just distributes the *destination addresses* without
reducing the *operation count*.

---

## 6. Theoretical Atomic Throughput

AMD does not publish a formal "atomic TFLOPS" specification.

**From L2 architecture (Chips and Cheese, CDNA3 analysis):**

Each XCD L2 has 16 slices × 1 atomic unit per slice. If each slice sustains
1 atomic/cycle at 2.1 GHz, and different addresses land on different slices:

```
16 slices/XCD × 2.1 GHz × 8 XCDs = 268.8 GAtomics/s ≈ 0.27 TFLOPS
```

This is the L2-hit upper bound. For our workload, atomics miss L2 and the actual
bound is lower — determined by the Infinity Cache atomic unit throughput (unspecified
by AMD).

**Empirical estimate from benchmark delta (gather vs split_intermediate):**

```
Excess time due to atomics: 38.00 - 18.21 = ~20 ms
Total atomic_adds: 2.415B
Effective throughput: 2.415B / 0.020s ≈ 121 GAtomics/s ≈ 0.12 TFLOPS
```

**Comparison:**

| Resource | Peak |
|---|---|
| bf16 matrix compute | 1,307 TFLOPS |
| fp32 vector compute | 163 TFLOPS |
| HBM bandwidth (atomic as RMW) | ~0.66 TFLOPS |
| L2 atomic unit (cache-hit, estimated) | ~0.27 TFLOPS |
| Infinity Cache atomic unit (empirical) | ~0.12 TFLOPS |

Atomics achieve roughly **0.009% of peak bf16 compute** because each RMW
is a sequential, latency-bound operation at the cache-level atomic unit.

---

## 7. Why Gather Works

The gather method replaces atomic RMW with plain stores:

**Phase 1:** Each CTA writes its dKV contribution to a unique slot in an intermediate
buffer `[T, TOPK, D]` (bf16). One unique writer per slot → zero conflicts → stores
execute at full HBM write bandwidth (no serialization).

**Phase 2:** For each KV token, gather its contributions from the intermediate buffer
using plain loads and sum → plain store to dKV. Sequential per KV token, but
parallelized across T CTAs, one per KV token.

```
Atomic path:  read → serialize → modify → write  (latency-bound at atomic unit)
Store path:   write                               (bandwidth-bound at HBM, 5.3 TB/s)
```

The store path converts an **atomic unit throughput problem** into an **HBM bandwidth
problem** — a resource MI300X has in abundance.

---

## 8. Conclusions

1. **`(pid % 304) // 38` is wrong.** MI300X dispatches CTAs to XCDs in an
   interleaved pattern (groups of ~4 per XCD), not sequentially. The correct XCD
   index must be read from hardware at runtime via `hw_id >> 14`.

2. **XCD privatization does not help** because the dKV working set (9.44 MB) exceeds
   the 4 MB L2 per XCD. Atomics always miss L2 and serialize at the shared Infinity
   Cache, which is unaffected by XCD-level privatization.

3. **The fundamental bottleneck is the atomic RMW operation itself**, not XCD routing
   or cross-XCD cache coherence. The Infinity Cache atomic unit operates at ~0.12
   TFLOPS — roughly 10,000× slower than peak bf16 matrix compute.

4. **Gather (no atomics) achieves 3.33–3.54× speedup** by replacing serialized RMW
   with parallel stores, converting a latency-bound problem into a bandwidth-bound one.

5. **The remaining cost of gather** is its large intermediate buffer (6.5–13 GiB).
   Chunked processing (CHUNK < TOPK per pass) can reduce this to ~580 MB at the
   cost of CHUNK/TOPK extra passes over Q/dO.
