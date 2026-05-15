# Sparse MLA Backward — Triton tl.trans vs v17 HIP (ds_bpermute) Flow Comparison

## Summary of Change

**Triton tl.trans**: Transpose dS[BH,16] and P[BH,16] via `tl.trans()`, which Triton
lowers to LDS write → s_barrier → LDS read. Requires staging VGPRs → 313 spills.

**v17 HIP**: Same algorithm, but transpose via `ds_bpermute_b32` (LDS crossbar
permutation). No LDS allocation, no barrier, no staging VGPRs → 0 spills.
Additionally transposes each Q/dO chunk (16x16) per D-iteration via ds_bpermute.

Sources:
- Triton tl.trans: `op_tests/triton_tests/attention/dump_bwd_trans_isa.py`
- v17 HIP: `csrc/kernels/mla/hk/mi3xx_sparse_mla_bwd_train_v17.cuh`

Note: The Triton tl.trans kernel here is the same as the "NEW" column in
`triton_sparse_mla_bwd_flow_comparison.md`.

---

## Side-by-Side: Python Wrapper

```
Triton tl.trans                             v17 HIP
─────────────────────────────────────────   ─────────────────────────────────────────
# No pre-transpose needed                  # No pre-transpose needed
# (same as Triton — both eliminate Q_T)    # (same as Triton — both eliminate Q_T)

_sparse_mla_bwd_kernel_trans(               launch_sparse_mla_bwd_v17(params)
    Q, KV, dO, TopK, LSE, Delta,            params.Q = Q
    dQ, dKV,                                params.KV = KV
    ...strides...,                          params.dO = dO
    scale, num_heads,                       ...
)                                           grid(total_tokens, cdiv(H, 64))
grid(total_tokens, cdiv(H, BH=64))         block(256), smem(36864)
```

## Side-by-Side: Kernel Parameters

```
Triton tl.trans                             v17 HIP
─────────────────────────────────────────   ─────────────────────────────────────────
Q_ptr, KV_ptr, dO_ptr,                     Q, KV, dO            (same)
TopK_ptr, LSE_ptr, Delta_ptr,              TopK, LSE, Delta     (same)
dQ_ptr, dKV_ptr,                           dQ, dKV              (same)
stride_q_t, stride_q_h,                    stride_q_t, stride_q_h,
stride_kv_t,                               stride_kv_t,
stride_do_t, stride_do_h,                  stride_do_t, stride_do_h,
stride_dq_t, stride_dq_h,                 stride_dq_t, stride_dq_h,
stride_dkv_t,                              stride_dkv_t,
stride_topk_t,                             stride_topk_t,
scale, num_heads,                          scale, num_heads, topk, total_tokens
TOPK, BLOCK_H, TILE_K (constexpr)         (hardcoded: BH=64, TK=16, D=576)
```

## Side-by-Side: Prologue (Register Loading)

```
Triton tl.trans (5 tensors)                 v17 HIP (5 tensors, per warp)
─────────────────────────────────────────   ─────────────────────────────────────────
Q_lora    [BH=64, 512]  ← Q[:D_V]         b_q[0..35]  (36 v4bf16)
Q_rope    [BH=64, 64]   ← Q[D_V:]           = Q[h_start+lr, dc*16+lg*4+j]
                                              per warp: [16, 576] in VGPRs

dO        [BH=64, 512]  ← dO[:,:]         b_do[0..31] (32 v4bf16)
                                              per warp: [16, 512] in VGPRs

lse       [BH=64]                          my_lse    (1 fp32 per lane)
delta     [BH=64]                          my_delta  (1 fp32 per lane)

dQ_lora   [BH, 512]  = 0                  dq_acc[36][4] = 0
dQ_rope   [BH, 64]   = 0                    per warp: [16, 576] fp32 accumulator

Total persistent data:                     Total persistent data (per warp):
  3 × [64,512] + 2 × [64,64] + 2 scalar     b_q(36) + b_do(32) + dq(144fp32)
  = 104K bf16/fp32 elements                  + lse(1) + delta(1)
  (same as Triton — no Q_T/dO_T)            = 68 VGPRs + 144 fp32 accumulators
```

## Side-by-Side: Tile Loop

### Phase 1: Load KV tile

```
Triton tl.trans                             v17 HIP
─────────────────────────────────────────   ─────────────────────────────────────────
K_lora_T [512, 16] ← tl.load(KV[pos,:DV]) Cooperative LDS load (all 256 threads):
K_rope_T [64, 16]  ← tl.load(KV[pos,DV:])   KV[16, 576] bf16 → lds_kv_curr
                                              __syncthreads()
(next tile topk_pos prefetched)            (double-buffered: ping/pong LDS slots)
```

### Phase 2: Recompute S and P (IDENTICAL math)

```
Triton tl.trans                             v17 HIP
─────────────────────────────────────────   ─────────────────────────────────────────
S  = tl.dot(Q_lora, K_lora_T)             for dc in 0..35:
S += tl.dot(Q_rope, K_rope_T)               a_kv = LDS[lr * D + doff + lg*4+j]
S *= scale                                   s_acc = mfma(a_kv, b_q[dc], s_acc)
S = where(valid, S, -inf)                  (S and dP fused in same LDS-read loop)
P = exp(S - lse[:, None])
P = where(valid, P, 0)                    sv = s_acc * scale
                                            pv = exp(sv - my_lse)
# 2 tl.dot → Triton auto-tiles MFMAs      # 36 explicit 16x16x16 MFMAs
```

### Phase 3: Compute dP and dS (IDENTICAL math)

```
Triton tl.trans                             v17 HIP
─────────────────────────────────────────   ─────────────────────────────────────────
dP = tl.dot(dO_val, K_lora_T)             for dc in 0..31:  (fused in Phase 2 loop)
dS = P * (dP - delta[:, None]) * scale      dp_acc = mfma(a_kv, b_do[dc], dp_acc)
dS = where(valid, dS, 0)
                                            ds_val = pv * (dp_acc - my_delta) * scale
# 1 tl.dot                                # 32 MFMAs (reuse same a_kv from S loop)
```

### Phase 4: Accumulate dQ (IDENTICAL math)

```
Triton tl.trans                             v17 HIP
─────────────────────────────────────────   ─────────────────────────────────────────
V_lora = tl.trans(K_lora_T)               b_ds = bf16(ds_val)
dQ_lora += tl.dot(dS, V_lora)             for dc in 0..35:
K_rope  = tl.trans(K_rope_T)                a_kv_t = LDS[(lg*4+j)*D + doff + lr]
dQ_rope += tl.dot(dS, K_rope)               dq_acc[dc] = mfma(a_kv_t, b_ds, dq_acc)

# tl.trans(K_lora_T): [512,16] → [16,512] # No transpose — just read LDS with
# tl.trans(K_rope_T): [64,16]  → [16,64]  # swapped row/col index pattern
# Triton uses LDS for these K transposes   # Same data, different access order

# 2 tl.dot for dQ                         # 36 MFMAs for dQ
```

### Phase 5: Compute dKV and scatter (KEY DIFFERENCE)

```
Triton tl.trans                             v17 HIP (ds_bpermute)
─────────────────────────────────────────   ─────────────────────────────────────────
# Transpose via LDS round-trip:            # Transpose via ds_bpermute_b32:
dS_T = tl.trans(dS)    [BH,16] → [16,BH]  transpose_mfma16x16(ds_val, ds_T)
P_T  = tl.trans(P)     [BH,16] → [16,BH]  transpose_mfma16x16(pv, p_T)
                                            # per warp: [16,16] → [16,16]
# Lowered to amdgcn:                       #
#   ds_write_b16 × 1024                    # Lowered to amdgcn:
#   s_barrier                               #   16 ds_bpermute_b32
#   ds_read_u16  × 1024                    #   12 v_cndmask
#   (× 2 for dS and P)                    #   ~40 instructions per transpose
#                                           #   (× 2 for dS and P)
# Needs staging VGPRs → SPILLS            #
#                                           # No LDS, no barrier, no staging VGPRs

# dKV matmul (Q/dO from prologue regs)    # dKV matmul (Q/dO from prologue regs)
dKV_lora  = tl.dot(dS_T, Q_lora)          for dc in 0..35:
dKV_lora += tl.dot(P_T, dO_val)             # Transpose Q chunk via ds_bpermute:
dKV_rope  = tl.dot(dS_T, Q_rope)            q_T = transpose_mfma16x16(b_q[dc])
                                              dkv = mfma(q_T, b_ds_T, dkv)
                                              if dc < 32:
                                                # Transpose dO chunk:
                                                do_T = transpose_mfma16x16(b_do[dc])
                                                dkv = mfma(do_T, b_p_T, dkv)
                                              atomicAdd(dKV[kvpos[lr], doff+lg*4+j])

# 3 tl.dot, all 64 heads at once          # 68 MFMAs, 16 heads/warp × 4 warps
# 2 transposes (large, 64x16)             # 70 transposes (small, 16x16 each)
# 2 s_barrier                              # 0 barriers

# Atomic scatter                           # Atomic scatter
tl.atomic_add(dKV_lora, mask=valid)        atomicAdd per warp (4× more atomics)
tl.atomic_add(dKV_rope, mask=valid)        # 4 warps → 4 partial sums per element
```

## Side-by-Side: Epilogue

```
Triton tl.trans                             v17 HIP
─────────────────────────────────────────   ─────────────────────────────────────────
tl.store(dQ_lora)  [BH, 512] bf16         if valid_warp:
tl.store(dQ_rope)  [BH, 64]  bf16           dQ[h, d] = bf16(dq_acc[dc][j])
```

## 8 Dot Products: Comparison

| # | Triton tl.trans | v17 HIP (ds_bpermute) | Shape | Purpose |
|---|-----------------|----------------------|-------|---------|
| 1 | Q_lora @ K_lora_T | mfma(a_kv, b_q) ×32 | [BH,512]@[512,16] | S (nope) |
| 2 | Q_rope @ K_rope_T | mfma(a_kv, b_q) ×4 | [BH,64]@[64,16] | S (rope) |
| 3 | dO @ K_lora_T | mfma(a_kv, b_do) ×32 | [BH,512]@[512,16] | dP |
| 4 | dS @ V_lora | mfma(a_kv_t, b_ds) ×32 | [BH,16]@[16,512] | dQ (nope) |
| 5 | dS @ K_rope | mfma(a_kv_t, b_ds) ×4 | [BH,16]@[16,64] | dQ (rope) |
| 6 | **dS_T @ Q_lora** | **mfma(q_T, b_ds_T)** ×32 | [16,BH]@[BH,512] | dKV (nope) |
| 7 | **P_T @ dO** | **mfma(do_T, b_p_T)** ×32 | [16,BH]@[BH,512] | dKV (V grad) |
| 8 | **dS_T @ Q_rope** | **mfma(q_T, b_ds_T)** ×4 | [16,BH]@[BH,64] | dKV (rope) |

Dots 1-5: identical algorithm, same operand layout.
Dots 6-8: identical math, but v17 additionally transposes each Q/dO chunk via ds_bpermute.

## Why Triton tl.trans Spills and v17 Doesn't

```
Triton tl.trans (LDS round-trip):           v17 ds_bpermute (register-only):
─────────────────────────────────────       ─────────────────────────────────────
1. dS[64,16] fp32 live in VGPRs            1. dS[16,16] fp32 in 4 VGPRs
2. Pin source VGPRs for ds_write           2. ds_bpermute reads src VGPR directly
3. ds_write_b16 × 1024 to LDS             3. Result lands in dst VGPR
4. s_barrier (all warps must reach)        4. s_waitcnt lgkmcnt(0) per-warp
5. Allocate dst VGPRs for ds_read          5. v_cndmask selects correct lane
6. ds_read_u16  × 1024 from LDS           6. Done — src VGPRs reusable

Steps 2+5 need staging VGPRs that          No staging VGPRs at any point.
overlap with Q, dO, dQ still live          Same 4 VGPRs reused for src → dst.
→ exceeds 256 VGPRs → 313 spills
```

## Measured Results

| Metric | Triton tl.trans | v17 HIP ds_bpermute |
|--------|----------------:|--------------------:|
| VGPRs | 256 | 256 |
| VGPR Spills | **313** | **0** |
| SGPRs | ~106 | 75 |
| LDS total | ~8KB + K tile | 36864 (KV double-buf) |
| Barriers per tile (Phase 5) | 2 (s_barrier) | 0 |
| Transposes per tile (Phase 5) | 2 large (64×16) | 70 small (16×16) |
| dKV atomicAdd | 1× per element | 4× per element |
| Occupancy | 1 wave/SIMD | 1 wave/SIMD |
| Correctness | Verified | Verified (<1% rel err) |
