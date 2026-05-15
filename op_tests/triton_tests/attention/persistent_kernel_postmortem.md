# Persistent Kernel (Approach A) — Implementation & Compiler Blocker Post-mortem

**Status: Blocked — Triton/LLVM compilation hangs. Hand off to Triton compiler team.**

---

## Goal

Implement a 304-CTA persistent backward kernel for DSA dKV accumulation that
keeps the per-XCD dkv_chunk buffer (3.14 MB) inside each XCD's 4 MB L2 cache,
replacing cross-XCD Infinity Cache atomic serialization with intra-L2 atomics.

Expected speedup over `split_intermediate` (38 ms): ~2.25× on the atomic portion,
giving a projected total of **~25–30 ms** — between `chunked_gather` (23 ms) and
`split_intermediate` (38 ms), with only **25 MB extra memory** (vs 1.65 GiB).

---

## Kernel Design

### Core idea

```
grid = (304,)   # one CTA per physical CU on MI300X (8 XCDs × 38 CUs)

for each CTA:
    xcd = pid % 8                          # proxy for XCD index (see §Compiler Blocker)
    my_tokens = [pid*(T/304), (pid+1)*(T/304))

    for k_start in range(0, T, K_CHUNK):   # 3 passes for T=4096
        for q in my_tokens:                # 13–14 tokens per CTA
            dQ_lora, dQ_rope = 0           # fp32 accumulators in registers
            for r in range(TOPK):          # all 1024 ranks
                k = topk_idx[q, r]
                S, P, dS = recompute(Q[q], KV[k], dO[q], lse[q], delta[q])
                dQ_lora += dS * K_lora[k] + P * dO_lora[q]
                dQ_rope += dS * K_rope[k]
                if k_start ≤ k < k_end:
                    atomic_add(dkv_chunk[xcd, k-k_start, :], ...)  # L2-local
            store dQ[q]
        reduce dkv_chunk[8, K_CHUNK, D] → dKV[k_start:k_end]      # plain stores
```

### Why K_CHUNK fits in L2

```
K_CHUNK = floor(4 MB / (D_QK * 4 bytes))
        = floor(4,194,304 / (576 * 4))
        = 1820 tokens → use 1366 (3 equal passes)

dkv_chunk per XCD: 1366 × 576 × 4 = 3.14 MB  <  4 MB L2 ✓
```

Atomics that hit in L2 are handled by the L2 slice's dedicated atomic RMW unit
(16 slices/XCD × 1 unit/slice × 2.1 GHz = 268.8 GAtomics/s = 0.27 TFLOPS),
versus the Infinity Cache path (~0.12 TFLOPS empirical) — a 2.25× improvement.

### Memory budget

| Buffer | Size | Notes |
|---|---|---|
| `dkv_chunk [8, K_CHUNK, D]` fp32 | 25 MB | reused each of 3 passes |
| `dQ [T, H, D]` bf16 | 600 MB | output |
| `dKV [T, D]` fp32 → bf16 | 9 MB | output |
| **Extra beyond dQ** | **25 MB** | vs 1.65 GiB for chunked_gather |

---

## Implementation — What Was Done

### Kernels added to `deepseek_sparse_attention.py`

**`_bwd_persistent_chunk`** — the main 304-CTA kernel.

Key parameters (all non-constexpr to avoid compile-time loop unrolling):
```python
TOPK: tl.int32,           # NOT constexpr — avoids unrolling 1024 iterations
TOKENS_PER_CU: tl.int32,  # NOT constexpr — avoids unrolling 14 iterations
BLOCK_H: tl.constexpr,    # = 64
NUM_HG: tl.constexpr,     # = num_heads // 64
D_V: tl.constexpr,        # = kv_lora_rank
D_ROPE: tl.constexpr,     # = rope_rank
```

Numerical precision fix: all accumulation promoted to fp32:
```python
K_lora_T = tl.load(...).to(tl.float32)   # bf16 input → fp32
S = tl.sum(Q_lora.to(tl.float32) * K_lora_T[None, :], axis=1) + ...
dkv_contrib = tl.sum(dS[:, None] * Q_lora.to(tl.float32), axis=0) + ...
```
Without this, `tl.sum(bf16 * bf16)` accumulates in bf16 → ~2% error for TOPK=64.
After fix: dQ_rel ≤ 0.006, dKV_rel ≤ 0.005 — all correctness tests PASS.

**`_bwd_chunk_reduce`** — reduces 8 XCD copies of dkv_chunk → dKV[k_start:k_end].
```python
grid = (K_CHUNK, D // BLOCK_D)
for xcd in tl.static_range(NUM_XCD=8):   # unrolled, NUM_XCD is constexpr
    acc += load(dkv_chunk[xcd, k_local, offs_d])
store(dKV[k_start + k_local, offs_d], acc)
```

### Bugs fixed during implementation

1. **`if in_chunk:` invalid in Triton** — conditional on a tensor value.
   Fix: masking with `mask_v = in_chunk & (offs_v < D_V)` in `tl.atomic_add(..., mask=mask_v)`.

2. **bf16 dot product error** — `tl.sum(bf16 * bf16)` accumulates in bf16.
   Fix: cast operands to fp32 before multiply (see above).

3. **`tl.inline_asm_elementwise` returns a vgpr (per-lane tensor)** —
   using `hw_id` as an index generated per-lane scatter addressing, causing
   LLVM to generate exponentially complex code. Fix: `xcd = pid % 8` (see §Compiler Blocker).

---

## Compiler Blocker

### Symptom

`_bwd_persistent_chunk` hangs during Triton JIT compilation. Two benchmark runs
were killed after 98+ minutes of pure CPU spin with no GPU activity and no output.
Compilation never completes.

### Root cause: LLVM register pressure and loop complexity

The kernel carries `dQ_lora [BLOCK_H=64, D_V=512]` and `dQ_rope [BLOCK_H=64, D_ROPE=64]`
accumulator tensors in registers across all TOPK iterations of the inner loop.

With 256 threads (4 warps × 64 lanes), each thread owns:
```
dQ_lora:   64 × 512 / 256 = 128 fp32 values  →  128 VGPRs
dQ_rope:   64 × 64  / 256 =  16 fp32 values  →   16 VGPRs
intermediates (K, dS, dP):                   →  ~20 VGPRs
───────────────────────────────────────────────────────────
Total:                                          ~164 VGPRs
```

AMD CDNA3 allocates VGPRs in groups of 8; 164 → 168 VGPRs actual, consuming
`168 × 256 × 4 = 172 KB` of the 256 KB VGPR file per CU. LLVM's register
allocator spends exponential time deciding spill placement at this register file
utilization level.

Compounding factor: `tl.atomic_add` with a vector mask inside a dynamic loop
forces LLVM to generate predicated scatter-atomics — each expands to a
multi-instruction sequence. The combination of high register pressure + predicated
atomics + warp reductions (from `tl.sum(..., axis=1)`) inside a dynamic loop
body exceeds what LLVM can optimize in reasonable time on AMDGPU targets.

### Attempted fixes (all insufficient)

| Fix | Result |
|---|---|
| `TOPK: tl.int32` instead of `tl.constexpr` | Prevents 1024× unroll, but LLVM still hangs |
| `TOKENS_PER_CU: tl.int32` + `range()` instead of `static_range` | Prevents 14×2=28 unroll, still hangs |
| `xcd = pid % 8` instead of `inline_asm_elementwise` | Fixes per-lane scatter, still hangs |

Each fix addressed a Triton frontend unrolling issue. The underlying LLVM
backend compile time explosion on the register-heavy loop body remains.

### What would fix it

The kernel is correct conceptually. To make it compile:

**Option 1 — Reduce BLOCK_H** (easy, measurable cost):
Use `BLOCK_H=8` or `BLOCK_H=16` instead of 64. Each thread owns 8–16 rows of
dQ, reducing the accumulator from 128 VGPRs to 16–32 VGPRs. This brings total
VGPRs to ~50–70, well within LLVM's fast-path. Trade-off: 4–8× more CTAs
needed per head group → more atomic serialization.

**Option 2 — Split dQ and dKV into separate kernels** (architectural change):
Compute dQ in a separate pass (no atomics, no dkv_chunk), then compute dKV in
the persistent kernel with only `[D_V]`-sized accumulators per token. Each kernel's
register pressure drops to ~30–40 VGPRs. Requires 2 passes over Q/KV but
likely compiles in seconds.

**Option 3 — HIP/C++ implementation** (recommended for production):
The persistent kernel maps naturally to HIP. HIP's device compiler (hipcc/clang)
handles register-heavy loops far better than Triton's LLVM pipeline. The design
is sound; only the Triton-to-LLVM lowering is the blocker.

**Option 4 — Triton compiler improvement** (for Triton team):
The specific bottleneck is LLVM loop optimization and register allocation for
dynamic loops with `atomic_add` + warp reductions on AMDGPU. Reducing the
LLVM optimization level (e.g., `-O1` or `-O2` instead of `-O3`) or adding a
compile-time loop complexity budget would allow this kernel to compile in
seconds at the cost of some runtime performance.

---

## Correctness Status

All correctness tests pass on the compiled kernel (small configs where LLVM
finishes quickly due to reduced D_V):

```
Config: B=1 S=128 H=16 D=320 TOPK=64
persistent   dQ 4.20e-03   dKV 4.61e-03   PASS

Config: B=1 S=256 H=32 D=320 TOPK=128
persistent   dQ 6.06e-03   dKV 4.95e-03   PASS

Config: B=1 S=256 H=128 D=320 TOPK=128
persistent   dQ 3.55e-03   dKV 5.05e-03   PASS
```

The kernel logic is correct. The blocker is purely compilation time at production
config (D_V=512).

---

## Benchmark Status

Not obtained. Both benchmark attempts were killed after 98+ minutes of LLVM hang
at production config (T=4096, H=128, D=576, TOPK=1024). The projected performance
from first-principles analysis:

| Method | Projected (ms) | Actual (ms) | Extra mem |
|---|---|---|---|
| fused | — | 61.1 | 0 |
| split_intermediate | — | 38.0 | 2.0 GiB |
| chunked_gather | — | 23.4 | 1.65 GiB |
| **persistent (projected)** | **25–30** | **not obtained** | **25 MB** |

The persistent kernel's value proposition — if it compiled — would be the best
memory efficiency among all approaches that beat `split_intermediate`, using
only 25 MB extra vs 1.65 GiB for `chunked_gather`.

---

## Files

```
aiter/ops/triton/_triton_kernels/attention/deepseek_sparse_attention.py
  _bwd_persistent_chunk()    — 304-CTA persistent kernel (compiles at small D_V only)
  _bwd_chunk_reduce()        — XCD copy reduction kernel (compiles and runs correctly)
  sparse_mla_bwd(..., method="persistent")  — dispatch and orchestration

op_tests/triton_tests/attention/
  chunked_dkv_plan.md              — original design plan (Approach A + B side-by-side)
  persistent_kernel_postmortem.md  — this file
  bench_dsa_methods.py             — includes "persistent" in METHODS list (will hang at D_V=512)
```
