# v17 HIP Backward Kernel — Potential Improvements

Source: `csrc/kernels/mla/hk/mi3xx_sparse_mla_bwd_train_v17.cuh`

---

## 1. Vectorized KV Global Loads

**Current**: Each thread loads one bf16 (2 bytes) per iteration in the cooperative KV load loop:
```cpp
for (int idx = threadIdx.x; idx < TK17 * D17; idx += NT17) {
    int ki = idx / D17;
    int di = idx % D17;
    hip_bfloat16 v = p.KV[kvpos[ki] * p.stride_kv_t + di];
    lds_kv_curr[ki * D17 + di] = v;
}
```
64 threads × 2 bytes = 128 bytes per wavefront access (1 cache line). Coalesced but
uses element-wise loads.

**Improvement**: Cast to `uint64_t*` or `float2*` to load 4 bf16 per thread per
iteration. Reduces global load instruction count by 4x. D17=576 is divisible by 4,
so alignment is fine.

**Status**: Not investigated.

---

## 2. Reduce atomicAdd Volume for dKV

**The problem**: dKV atomicAdd is 91.7% of total kernel runtime (53.55 ms out of
58.42 ms). The kernel issues 4.83 billion atomicAdd operations (confirmed by rocprof).

**What we tried (FAILED)**:
- Per-head-group dKV buffers (G=2): 58.50 ms → no improvement
- Token-group partitioning (G=2..32): 58.47-58.99 ms → no improvement at any G

Reducing contention per address does NOT help because the bottleneck is the **total
volume** of atomic operations (4.83B), not address-level contention. The L2 atomic
throughput is saturated regardless of collision rate.

**What might work**: Fundamentally reduce the NUMBER of atomicAdd operations:
- Accumulate partial dKV in registers or LDS across multiple tiles before scattering
- Restructure the algorithm: instead of iterating over KV per query (scatter dKV),
  iterate over queries per KV (gather, direct accumulation, no atomics)
- Split kernel: compute dQ in one kernel (no atomics needed), compute dKV in a
  separate kernel with a different tiling that avoids atomics

**What worked: Split dQ/dKV + head-group fusion (1.68× speedup)**:

Split the fused kernel into separate dQ and dKV kernels. The dKV kernel fuses both
head groups (NUM_HG=2) into a single program, accumulating dKV from both HGs before
scattering — halving atomics from 4.83B to 2.42B.

```
dQ kernel:                5.68 ms  (0 spills, 172 MFMAs)
dKV HG-fused (TK=64 w=4): 29.13 ms  (0 spills, 136 MFMAs)
Split total:              34.81 ms  (1.68× faster than baseline 58.38 ms)
```

Key insight: splitting eliminates the register conflict between dQ accumulators
(128-144 AGPRs) and dKV computation. The dKV kernel has only 36 AGPRs → zero spills.

Cost: 2 GiB intermediate storage (dS + P in bf16).

**Status**: Proven effective. Further improvement possible via bf16 packed atomics
(another 2× reduction) or by applying the same split to the v17 HIP kernel.

Source: `bench_bwd_per_hg_dkv.py`, `bench_bwd_token_group_dkv.py`,
        `bench_bwd_split_dq_dkv.py`, `bench_bwd_dkv_hg_fused.py`,
        `sparse_mla_bwd_flops_bandwidth_analysis.md` (sections 6-11).

---

## 3. Continuous Atomic Pipelining (Future Direction)

**The problem**: In the current dKV kernel (split approach, 29.13 ms), atomics are issued
in **bursts** — all MFMAs for a tile complete first, then all atomics scatter at the end.
The L2 atomic ALUs sit idle during MFMA computation, and MFMAs stall while atomics drain.

```
Current (bursty):
  MFMA units:     ████░░░░████░░░░████░░░░████░░░░
  L2 atomic ALU:  ░░░░████░░░░████░░░░████░░░░████
                  ~50% atomic utilization
```

**Hardware context (MI300X)**:
- L2 atomic peak throughput: ~409 GOPS (estimated from Chips & Cheese MI300A measurement
  of 306.65 GOPS on 6 XCDs, scaled to 8 XCDs). Source: INT32 atomic_add microbenchmark,
  no contention (each thread → unique address).
- Our dKV kernel: ~83 GOPS (2.42B atomics / 29.13 ms) = **~20% of hardware peak**.
- The gap is partially due to bursty issue pattern, random scatter access, and cross-XCD
  coherence overhead for scattered TopK indices.
- AMD does NOT publish official L2 atomic throughput specs. The 409 GOPS is an estimate.
- Atomics on CDNA3 execute at the L2 cache (dedicated atomic ALUs). They cannot bypass
  L2 to go directly to HBM — HBM has no compute units for read-modify-write. Cross-XCD
  atomics are routed through Coherent Masters → Infinity Fabric → Coherent Slaves (with
  probe filter) → remote L2 atomic ALU.

**Proposed optimization**: Overlap atomic scatter of tile N with MFMA compute of tile N+1
using double-buffered dKV accumulators.

```
Pipelined (continuous):
  MFMA units:     ████████████████████████████████
  L2 atomic ALU:  ░░██████████████████████████████
                  ~90%+ atomic utilization
```

**Implementation**:

```
// Double-buffered dKV accumulators: buf_A and buf_B (36 AGPRs each = 72 total, fits in 256)

// Tile 0: compute into buf_A
dKV_A = Q_T @ dS_0 + dO_T @ P_0    // MFMAs

for tile = 1 .. NUM_TILES-1:
    // Fire-and-forget scatter buf_A (no s_waitcnt vmcnt!)
    global_atomic_add_f32(dKV_ptr, buf_A[...])    // async, CU continues

    // Immediately start computing tile N+1 into buf_B
    dKV_B = Q_T @ dS_tile + dO_T @ P_tile         // MFMAs run while atomics drain

    // Swap buffers
    swap(buf_A, buf_B)

// Final scatter
global_atomic_add_f32(dKV_ptr, buf_A[...])
s_waitcnt vmcnt(0)  // wait only at the very end
```

**Potential speedup estimate**:

```
If atomic utilization improves from 20% → 40% of HW peak:
  Effective throughput: 83 → 166 GOPS
  Atomic time: 2.42B / 166G = 14.6 ms
  Compute time: fully overlapped (free)
  dKV kernel: ~15 ms (vs 29.13 ms today, ~2x speedup)
  Total bwd: ~21 ms (vs 34.81 ms today, ~1.7x additional speedup)
```

**Why this requires a HIP kernel (not Triton or TileLang)**:
- Need `s_waitcnt vmcnt(N)` control — Triton/TileLang insert automatic waits
- Need explicit AGPR double-buffering — compilers don't expose this
- Need `__builtin_amdgcn_s_setprio()` for MFMA vs memory priority scheduling
- `global_atomic_add_f32` is fire-and-forget at ISA level, but Triton treats
  `tl.atomic_add` as synchronous from the compiler's perspective
- TileLang's `split_store=2` splits the scatter into 2 chunks for register/smem
  staging, but does NOT overlap compute with atomics — both chunks still execute
  sequentially after all MFMAs complete

**Variant: intra-tile interleaving**: Instead of pipelining across tiles, split
each tile's dKV into chunks and interleave within a single tile iteration:

```
// Within one tile: compute + scatter in chunks
dKV_chunk1 = Q_T[:256] @ dS + dO_T[:256] @ P     // first 256 rows of D=512
atomic_add(dKV_ptr[:256], dKV_chunk1)              // fire-and-forget

dKV_chunk2 = Q_T[256:] @ dS + dO_T[256:] @ P     // next 256 rows (overlaps with chunk1 atomics)
atomic_add(dKV_ptr[256:], dKV_chunk2)              // fire-and-forget

dKV_rope = Q_rope_T @ dS                          // rope part (overlaps with chunk2 atomics)
atomic_add(dKV_rope_ptr, dKV_rope)
```

This produces smaller, more frequent atomic bursts — keeping the L2 pipeline
more continuously fed.

**Status**: Not implemented. Requires HIP kernel (planned in
`csrc/kernels/mla/hk/mi3xx_sparse_mla_bwd_train.cuh`).

**References**:
- Chips & Cheese MI300A atomic benchmark: https://chipsandcheese.com/p/sizing-up-mi300as-gpu
- CDNA3 ISA atomic instructions: AMD Instinct MI300 ISA Reference Guide, sections 9-10
- CDNA3 cross-XCD coherence: https://chipsandcheese.com/p/amds-cdna-3-compute-architecture

---
