# Chunked dKV Backward: Implementation Plan (v2)

Two approaches to eliminate atomics / large buffers by chunking,
with dS/P recomputed on-the-fly (no 2 GiB intermediate storage).

Config: T=4096, H=128, D_V=512, D_ROPE=64 (D=576), TOPK=1024, 8 XCDs, 304 CUs.

---

## Key Insight: Eliminating dS/P Buffers

The 2 GiB dS/P storage in `split_intermediate` exists because dQ and dKV are
computed in separate kernels — dS/P are computed in the dQ kernel and read back
in the dKV kernel.

Both chunked approaches process one slice of TOPK at a time. Within each slice,
dQ partial and dKV partial are computed from the same S/P, so S/P can live
entirely in registers/LDS and be discarded after each slice.

```
split_intermediate:                   chunked (both approaches):

Phase 0: compute dQ                   Per chunk [r_start, r_end):
         store dS[T,H,TOPK] ← 1 GiB    compute S, P, dS in registers
         store P[T,H,TOPK]  ← 1 GiB    accumulate dQ partial  → dQ[T,H,D]
                                         compute dKV contribution
Phase 1: load dS, P                     (approach A: atomic_add to chunk copy)
         compute dKV                     (approach B: store to intermediate)
         atomic_add → dKV               discard S, P, dS

dS/P buffers: 2 GiB                   dS/P buffers: 0
```

dQ accumulation across passes is conflict-free: each query CTA owns
its own `dQ[q, :, :]` rows and accumulates with plain load→add→store.
No atomics required.

---

## Side-by-Side: Flow Diagrams

```
┌────────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
│   APPROACH A: Persistent Chunked           │  │   APPROACH B: Chunked Gather             │
│              XCD Privatized                │  │              (no dS/P storage)           │
│                                            │  │                                          │
│  grid: 304 CTAs  (1 per physical CU)       │  │  grid: T CTAs  (1 per query token)       │
│  → hardware forces all 8 XCDs active       │  │  → regular launch, any XCDs active       │
│  → no routing formula needed               │  │                                          │
├────────────────────────────────────────────┤  ├──────────────────────────────────────────┤
│                                            │  │                                          │
│  kernel start:                             │  │  allocate:                               │
│    xcd = (hw_id >> 14) & 0x7  (once)       │  │    interm  [T, R_CHUNK, D]  bf16  301MB  │
│    my_tokens = [pid*(T/304),               │  │    dkv_acc [T, D]           fp32    9MB  │
│                (pid+1)*(T/304))            │  │    dQ      [T, H, D]        bf16  600MB  │
│                                            │  │                                          │
│  allocate:                                 │  │  ┌──────────────────────────────────┐    │
│    dkv_chunk [8, K_CHUNK, D]  fp32  25MB   │  │  │ for r_start in 0..TOPK..R_CHUNK │    │
│    dQ        [T, H, D]        bf16  600MB  │  │  │                                 │    │
│                                            │  │  │  ── Phase A (grid: T) ──         │    │
│  for k_start in 0..T..K_CHUNK:  (3 passes)│  │  │  for r in [r_start, r_end):      │    │
│                                            │  │  │    k   = topk_idx[q, r]          │    │
│    zero dkv_chunk[8, K_CHUNK, D]           │  │  │    S   = Q[q]·K[k]ᵀ / scale     │    │
│                                            │  │  │    P   = softmax(S)              │    │
│    for q in my_tokens:  (13-14 tokens)     │  │  │    dS  = dO[q]·V[k]ᵀ - δ[q]    │    │
│      for r in 0..TOPK:  (all ranks)        │  │  │    ── accumulate dQ partial ──   │    │
│        k  = topk_idx[q, r]                 │  │  │    dq_tmp = load dQ[q]           │    │
│        S  = Q[q]·K[k]ᵀ / scale            │  │  │    dq_tmp += dS*K[k] + P*V[k]   │    │
│        P  = softmax(S)                     │  │  │    store dQ[q] ← dq_tmp          │    │
│        dS = dO[q]·V[k]ᵀ - δ[q]           │  │  │    ── store intermediate ──       │    │
│        ── accumulate dQ ──                 │  │  │    STORE interm[q, r-r_start, :] │    │
│        dq_reg += dS*K[k] + P*V[k]         │  │  │    (unique writer, no atomics)   │    │
│        ── accumulate dKV ──                │  │  │                                 │    │
│        if k_start ≤ k < k_end:            │  │  │  ── rebuild CSR (host/device) ── │    │
│          ATOMIC_ADD                        │  │  │  inv_ptr[T+1], inv_data[T*R_C]   │    │
│          dkv_chunk[xcd, k-k_start, :]     │  │  │  for topk[:,r_start:r_end] only  │    │
│          ← 3.14 MB/XCD, fits in L2 ✓      │  │  │                                 │    │
│      write dQ[q] ← dq_reg                 │  │  │  ── Phase B (grid: T) ──         │    │
│                                            │  │  │  for each kv_token k:            │    │
│    _bwd_dkv_reduce_copies:                 │  │  │    for entry in CSR[k]:          │    │
│    dkv_chunk[8, K_CHUNK, D]               │  │  │      (q, r_off) = decode(entry)  │    │
│    → dKV[k_start:k_end, :]  (plain store) │  │  │      dkv_acc[k] +=               │    │
│                                            │  │  │        interm[q, r_off, :]       │    │
│  end for k_start  (3 passes)              │  │  │  (last pass: write dkv_acc→dKV)  │    │
│                                            │  │  │                                 │    │
│  Output: dQ[T,H,D] bf16, dKV[T,D] bf16   │  │  │  end for r_start (16 passes)     │    │
│                                            │  │  └──────────────────────────────────┘    │
│                                            │  │                                          │
│                                            │  │  Output: dQ[T,H,D] bf16, dKV[T,D] bf16 │
└────────────────────────────────────────────┘  └──────────────────────────────────────────┘
```

---

## Side-by-Side: First Principle Calculations

### Chunk Size Selection

```
┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
│  Approach A: K_CHUNK (KV token range)   │  │  Approach B: R_CHUNK (TOPK rank range)  │
├─────────────────────────────────────────┤  ├─────────────────────────────────────────┤
│                                         │  │                                         │
│  Goal: dkv_chunk/XCD fits in L2 (4 MB) │  │  Goal: interm fits in ~300 MB           │
│                                         │  │                                         │
│  K_CHUNK × D × 4 ≤ 4 MB                │  │  T × R_CHUNK × D × 2 ≤ 300 MB          │
│  K_CHUNK ≤ 4,194,304 / 2304 = 1820     │  │  R_CHUNK ≤ 300 MB / (4096×576×2) = 63  │
│  → use K_CHUNK = 1365 (3 equal passes) │  │  → use R_CHUNK = 64 (16 equal passes)  │
│                                         │  │                                         │
│  chunk/XCD: 1365×576×4 = 3.14 MB ✓    │  │  interm: 4096×64×576×2 = 301 MB ✓      │
└─────────────────────────────────────────┘  └─────────────────────────────────────────┘
```

### Memory Footprint

```
                         split_interm  full_gather   Approach A   Approach B
─────────────────────────────────────────────────────────────────────────────
dS buffer [T,H,TOPK]bf16   1.0 GiB      1.0 GiB        0            0
P  buffer [T,H,TOPK]bf16   1.0 GiB      1.0 GiB        0            0
interm [T,TOPK,D]  bf16       0          4.83 GiB       0          301 MB *
interm [T,R_CHUNK,D]bf16      0            0            0          301 MB
dkv_copies[8,T,D]  fp32       0            0          75.5 MB        0
dkv_chunk [8,K,D]  fp32       0            0           25 MB **       0
dkv_acc   [T,D]    fp32       0            0             0           9 MB
dQ output [T,H,D]  bf16    600 MB        600 MB       600 MB       600 MB
─────────────────────────────────────────────────────────────────────────────
Extra beyond dQ:           2.0 GiB      5.4 GiB       25 MB        310 MB

*  reused each of 16 passes     **  reused each of 3 passes
```

### Total Arithmetic Work

```
┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
│  Approach A (304 CTAs, persistent)      │  │  Approach B (T CTAs per pass)           │
├─────────────────────────────────────────┤  ├─────────────────────────────────────────┤
│                                         │  │                                         │
│  Recompute S, P, dS per token per rank  │  │  Recompute S, P, dS per token per rank  │
│  (same as "recompute" method)           │  │  (same as "recompute" method)           │
│                                         │  │                                         │
│  dQ: T×H×TOPK×(2D) FLOPs               │  │  dQ: T×H×TOPK×(2D) FLOPs               │
│    = 4096×128×1024×1152 = 619 GFLOP    │  │    = 619 GFLOP                          │
│                                         │  │                                         │
│  S/P/dS recompute:                      │  │  S/P/dS recompute:                      │
│    T×H×TOPK×(2D+softmax) ≈ 1.3 TFLOP  │  │    ≈ 1.3 TFLOP                         │
│                                         │  │                                         │
│  atomic_add_f32: 2.415B ops             │  │  store (Phase A): 4.83 GB written       │
│    → bottleneck: L2 atomic unit        │  │  gather (Phase B): 4.83 GB read         │
│                                         │  │    → bottleneck: HBM bandwidth         │
│  kernel passes: 3                       │  │                                         │
│  CTA count: 304 (persistent)            │  │  kernel passes: 16                      │
│  tokens per CTA: ~13-14                 │  │  CTA count: T=4096 per pass             │
└─────────────────────────────────────────┘  └─────────────────────────────────────────┘
```

### Bottleneck Analysis

```
┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
│  Approach A bottleneck                  │  │  Approach B bottleneck                  │
├─────────────────────────────────────────┤  ├─────────────────────────────────────────┤
│                                         │  │                                         │
│  Bottleneck 1: recompute S/P/dS         │  │  Bottleneck 1: recompute S/P/dS         │
│    recompute method: 52 ms total        │  │    same cost as recompute method        │
│    split_intermediate: 38 ms total      │  │    overhead vs split_intermediate:      │
│    delta ≈ 14 ms recompute overhead     │  │    ~14 ms extra vs split_interm         │
│                                         │  │                                         │
│  Bottleneck 2: L2 atomic unit           │  │  Bottleneck 2: HBM bandwidth (Phase B)  │
│                                         │  │    4.83 GB reads × 2 (read+write) =     │
│  dkv_chunk per XCD: 3.14 MB < 4 MB L2  │  │    9.66 GB total interm traffic         │
│  → atomics hit L2 slice atomic units    │  │    at 5.3 TB/s: 9.66/5300 = 1.8 ms    │
│                                         │  │    (compute-bound in practice)          │
│  L2 atomic throughput (8 XCDs):         │  │                                         │
│    16 slices × 2.1 GHz × 8 XCDs        │  │  Phase B also: 16 CSR rebuilds          │
│    = 268.8 GAtomics/s = 0.27 TFLOPS    │  │    argsort(T×R_CHUNK=262K) ×16          │
│                                         │  │    est. < 1 ms per rebuild (GPU sort)   │
│  vs Infinity Cache (current):           │  │                                         │
│    0.12 TFLOPS empirical                │  │  Phase A: identical to gather Phase A   │
│    → 2.25× improvement on atomic part  │  │    expect similar compute time          │
│                                         │  │                                         │
│  persistent 304 CTAs:                   │  │  Expected total:                        │
│    → 8 XCDs always active               │  │    recompute overhead: ~14 ms           │
│    → no routing guess needed            │  │    + gather phases: ~5 ms               │
│    → no pid→XCD formula required        │  │    → ~19-22 ms                          │
│                                         │  │                                         │
│  Expected total:                        │  │  (similar to full gather ~18 ms,        │
│    recompute overhead: ~14 ms           │  │   marginally slower due to 16 passes    │
│    + atomic work at L2: ~9 ms           │  │   and CSR rebuilds)                     │
│    + chunk overhead: ~3 ms              │  │                                         │
│    → ~25-30 ms                          │  │                                         │
└─────────────────────────────────────────┘  └─────────────────────────────────────────┘
```

### Summary Comparison (all methods)

```
                  fused  recompute  split   gather  chk_priv(A)  chk_gather(B)
──────────────────────────────────────────────────────────────────────────────
Time (ms)          61      52        38      18       25-30         19-22
Extra mem        0        0        2 GiB   5.4 GiB    25 MB        310 MB
dS/P storage     No       No       Yes      Yes        No            No
Atomics          Yes      Yes      Yes      No         Yes           No
Bottleneck     S/P+atm  S/P+atm  atomic  compute   L2 atomic    compute/BW
8 XCDs?        partial  partial  partial  partial    always       partial *
Grid              T        T      T+T       T+T        304         T+T ×16

* XCDs active depend on wave count; 4 active for T=4096 with 4 warps
```

---

## Implementation Plan

### Approach B (Chunked Gather, Recommended First)

**New kernels needed:**

1. `_bwd_chunk_dq_and_intermediate` — fused kernel replacing `_bwd_dq_store_intermediates`
   - Inputs: Q, KV, dO, topk_idx, lse, delta (precomputed), r_start, R_CHUNK
   - Recomputes S, P, dS for ranks [r_start, r_end)
   - Accumulates dQ partial into `dQ[T, H, D]` bf16 (plain load→add→store, no conflict)
   - Stores `interm[T, R_CHUNK, D]` bf16 (plain store, one writer per slot)
   - Replace `dS_buf` and `P_buf` args with `r_start` and `R_CHUNK`

2. `_bwd_dkv_gather` — reuse as-is (reads from interm[T, R_CHUNK, D])

3. Python loop orchestration:
   ```python
   dQ     = torch.zeros(T, H, D, dtype=torch.bfloat16, device=device)
   dkv_acc = torch.zeros(T, D, dtype=torch.float32,   device=device)
   interm  = torch.empty(T, R_CHUNK, D, dtype=torch.bfloat16, device=device)

   for r_start in range(0, TOPK, R_CHUNK):
       _bwd_chunk_dq_and_intermediate[grid](
           ..., dQ, interm, r_start=r_start, R_CHUNK=R_CHUNK)
       inv_ptr, inv_data = _build_inverted_topk(
           topk_indices[:, r_start:r_start+R_CHUNK])
       _bwd_dkv_gather[grid](
           interm, inv_ptr, inv_data, dkv_acc, ..., is_last=(r_start+R_CHUNK>=TOPK))
   dkv = dkv_acc.to(torch.bfloat16).unsqueeze(1)
   ```

**Open question:** `_build_inverted_topk` currently runs on CPU via torch.argsort.
Move to GPU with `torch.argsort(..., stable=True)` to avoid 16× host-device syncs.

---

### Approach A (Persistent Chunked XCD Privatized)

**New kernel needed:**

`_bwd_persistent_xcd_chunked` — single persistent kernel
- Launch: `grid=(304,)`, `num_warps=4`
- At kernel start: read `xcd = (hw_id >> 14) & 0x7`
- Assign tokens: `my_start = pid * (T // 304)`, `my_end = my_start + (T // 304)`
- Outer loop over k_chunks [k_start, k_end):
  - Inner loop over my_tokens:
    - Inner-inner loop over all TOPK ranks:
      - Recompute S, P, dS
      - Accumulate dQ in registers
      - If k_start ≤ topk_idx[q, r] < k_end: atomic_add to dkv_chunk[xcd, k-k_start, :]
    - Write dQ[q] to HBM
  - Sync and reduce dkv_chunk → dKV[k_start:k_end] (separate reduce kernel)

**Complexity concern:** The persistent kernel reorders the loop nest (token outer,
rank inner) and accesses Q/KV randomly within each token's TOPK. Memory access
patterns are harder to coalesce than the T-CTA approach. Careful tiling required.

---

## Open Questions Before Implementation

| Question | Affects |
|---|---|
| Does 3.14 MB dkv_chunk stay in L2 under pressure from Q/KV streaming? | Approach A speedup estimate |
| GPU argsort cost for 262K elements × 16 rebuilds | Approach B overhead |
| dQ bf16 accumulation error over 16 passes acceptable? (vs fp32 acc) | Both: numerical precision |
| Persistent kernel occupancy: 304 CTAs × 4 warps = 1216 waves fills 8 XCDs? | Approach A correctness |

On the last point: 304 CTAs × 4 warps = 1216 waves.
Each XCD: 38 CUs × 4 SIMD × 16 slots = 2432 wave slots.
1216 waves / 8 XCDs = 152 waves/XCD = 4 waves/CU → fits comfortably, all 8 XCDs active ✓
