# Triton Sparse MLA Backward — Flow Comparison: Old vs New (tl.trans optimization)

## Summary of Change

**Before**: Pre-transpose Q and dO in Python before the kernel, load 3 extra large tensors
(Q_lora_T, Q_rope_T, dO_T) into registers for dKV computation.

**After**: Transpose the small intermediates dS[BH,16] and P[BH,16] inside the kernel
using `tl.trans()`, reuse Q_lora/Q_rope/dO already in registers. No pre-transposed copies needed.

---

## Side-by-Side: Python Wrapper

```
OLD                                         NEW
─────────────────────────────────────────   ─────────────────────────────────────────
# Pre-transpose Q and dO                   # (eliminated — no transpose needed)
Q_T  = Q.transpose(1,2).contiguous()
dO_T = dO.transpose(1,2).contiguous()

# Extra memory: 2 × T×D×H×2 bytes
# For B1_S4096_H128: ~2.3GB traffic
# ~0.4ms on MI300X

_sparse_mla_bwd_kernel(                    _sparse_mla_bwd_kernel(
    Q, KV, dO, TopK, LSE, Delta,               Q, KV, dO, TopK, LSE, Delta,
    Q_T, dO_T,          ← extra args            dQ, dKV,
    dQ, dKV,                                    ...
    stride_qt_t, ...,   ← extra strides
    ...
)                                          )
```

## Side-by-Side: Kernel Parameters

```
OLD                                         NEW
─────────────────────────────────────────   ─────────────────────────────────────────
Q_ptr, KV_ptr, dO_ptr,                     Q_ptr, KV_ptr, dO_ptr,
Q_T_ptr,             ← extra               (removed)
dO_T_ptr,            ← extra               (removed)
TopK_ptr, LSE_ptr, Delta_ptr,              TopK_ptr, LSE_ptr, Delta_ptr,
dQ_ptr, dKV_ptr,                           dQ_ptr, dKV_ptr,
stride_q_t, stride_q_h,                    stride_q_t, stride_q_h,
stride_kv_t,                               stride_kv_t,
stride_do_t, stride_do_h,                  stride_do_t, stride_do_h,
stride_qt_t,         ← extra               (removed)
stride_dot_t,        ← extra               (removed)
...                                        ...
```

## Side-by-Side: Prologue (Register Loading)

```
OLD (8 tensors in registers)                NEW (5 tensors in registers)
─────────────────────────────────────────   ─────────────────────────────────────────
Q_lora    [BH, 512]  ← Q[:D_V]            Q_lora    [BH, 512]  ← Q[:D_V]
Q_rope    [BH, 64]   ← Q[D_V:]            Q_rope    [BH, 64]   ← Q[D_V:]
dO        [BH, 512]  ← dO[:,:]            dO        [BH, 512]  ← dO[:,:]
Q_lora_T  [512, BH]  ← Q_T[:D_V]          (eliminated)
Q_rope_T  [64, BH]   ← Q_T[D_V:]          (eliminated)
dO_T      [512, BH]  ← dO_T[:,:]          (eliminated)
lse       [BH]                             lse       [BH]
delta     [BH]                             delta     [BH]
dQ_lora   [BH, 512]  = 0                  dQ_lora   [BH, 512]  = 0
dQ_rope   [BH, 64]   = 0                  dQ_rope   [BH, 64]   = 0

Total persistent data:                     Total persistent data:
  5 × [BH,512] + 3 × [BH,64] + 2 scalar    3 × [BH,512] + 2 × [BH,64] + 2 scalar
  = 5×32K + 3×4K + 2 = 172K elements        = 3×32K + 2×4K + 2 = 104K elements
  ↓ 39% less register pressure               ↓ SIGNIFICANT reduction
```

## Side-by-Side: Tile Loop

### Phase 1-2: Load K, Recompute S, P, dS (IDENTICAL)

```
OLD                                         NEW
─────────────────────────────────────────   ─────────────────────────────────────────
K_lora_T [D_V, TK] ← load                 K_lora_T [D_V, TK] ← load
K_rope_T [D_R, TK] ← load                 K_rope_T [D_R, TK] ← load

S = Q_lora @ K_lora_T + Q_rope @ K_rope_T S = Q_lora @ K_lora_T + Q_rope @ K_rope_T
S *= scale                                 S *= scale
P = exp(S - lse)                           P = exp(S - lse)

dP = dO @ K_lora_T                        dP = dO @ K_lora_T
dS = P * (dP - delta) * scale             dS = P * (dP - delta) * scale
```

### Phase 3: dQ Accumulation (IDENTICAL)

```
OLD                                         NEW
─────────────────────────────────────────   ─────────────────────────────────────────
V_lora = trans(K_lora_T)                   V_lora = trans(K_lora_T)
dQ_lora += dS @ V_lora                    dQ_lora += dS @ V_lora

K_rope = trans(K_rope_T)                   K_rope = trans(K_rope_T)
dQ_rope += dS @ K_rope                    dQ_rope += dS @ K_rope
```

### Phase 4/5: dKV Computation (KEY DIFFERENCE)

```
OLD (Phase 5)                               NEW (Phase 4)
─────────────────────────────────────────   ─────────────────────────────────────────
# Uses pre-transposed copies in regs       # Transpose small intermediates instead
# Q_lora_T[512,BH], Q_rope_T[64,BH],
# dO_T[512,BH] all loaded in prologue
                                            dS_T = tl.trans(dS)    # [TK, BH]
                                            P_T  = tl.trans(P)     # [TK, BH]

# dKV_lora: [D_V, TK]                     # dKV_lora: [TK, D_V]
dKV_lora = Q_lora_T @ dS                  dKV_lora = dS_T @ Q_lora
dKV_lora += dO_T @ P                      dKV_lora += P_T @ dO

# dKV_rope: [D_R, TK]                     # dKV_rope: [TK, D_R]
dKV_rope = Q_rope_T @ dS                  dKV_rope = dS_T @ Q_rope

# Atomic scatter [D, K] layout            # Atomic scatter [K, D] layout
# dKV_ptr + offs_v[:, None] * stride +     # dKV_ptr + safe_pos[:, None] * stride +
#           safe_pos[None, :] * 1          #           offs_v[None, :]
atomic_add(dKV_lora)                       atomic_add(dKV_lora)
atomic_add(dKV_rope)                       atomic_add(dKV_rope)
```

## 8 Dot Products: Layout Change

| # | OLD Operation | NEW Operation | Shape Change |
|---|---------------|---------------|-------------|
| 1 | Q_lora @ K_lora_T | Q_lora @ K_lora_T | [BH,D]@[D,K] same |
| 2 | Q_rope @ K_rope_T | Q_rope @ K_rope_T | [BH,R]@[R,K] same |
| 3 | dO @ K_lora_T | dO @ K_lora_T | [BH,D]@[D,K] same |
| 4 | dS @ V_lora | dS @ V_lora | [BH,K]@[K,D] same |
| 5 | dS @ K_rope | dS @ K_rope | [BH,K]@[K,R] same |
| 6 | **Q_lora_T @ dS** | **dS_T @ Q_lora** | [D,BH]@[BH,K] → [K,BH]@[BH,D] |
| 7 | **dO_T @ P** | **P_T @ dO** | [D,BH]@[BH,K] → [K,BH]@[BH,D] |
| 8 | **Q_rope_T @ dS** | **dS_T @ Q_rope** | [R,BH]@[BH,K] → [K,BH]@[BH,R] |

Dots 1-5: unchanged. Dots 6-8: swap the transpose from the large operand to the small one.

## Expected Benefits

| Metric | OLD | NEW | Delta |
|--------|-----|-----|-------|
| Pre-transposed tensors in registers | 3 (Q_lora_T, Q_rope_T, dO_T) | 0 | -3 tensors |
| Register elements (persistent, BH=64) | ~172K | ~104K | -39% |
| Python-side transpose | 2 × .transpose().contiguous() | None | Eliminated |
| Extra memory traffic (B1_S4096_H128) | ~2.3GB (~0.4ms) | 0 | Eliminated |
| VGPR spills (predicted) | 105 | TBD (test needed) | Expected improvement |
| tl.trans() on intermediates | 0 calls | 2 calls (dS, P) | +2 tiny transposes |

## Atomic Scatter Layout Change

**OLD**: dKV computed as [D, K], scattered in column-major pattern
```
dKV_ptr + offs_v[:, None] * stride + safe_pos[None, :] * 1
```

**NEW**: dKV computed as [K, D], scattered in row-major pattern
```
dKV_ptr + safe_pos[:, None] * stride + offs_v[None, :]
```

The [K, D] layout may be slightly better for atomic coalescing since tokens in the
same tile scatter to contiguous D-dim elements per token.

## Status

- [x] Code change implemented
- [ ] Correctness test (run test_sparse_mla_bwd_train.py)
- [ ] Performance benchmark (check VGPR spills, timing, TFLOPS)
- [ ] Commit once verified
