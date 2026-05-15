# Triton Sparse MLA Backward Kernel — Flow Overview

Source: `aiter/ops/triton/_triton_kernels/attention/sparse_mla_bwd_train.py`

## Background: MLA Absorbed Formulation

In DeepSeek's MLA, K and V are different projections of the same compressed latent `c_t`:

```
c_t  [512]     <- compressed latent (stored in KV cache)

K = c_t @ W_UK    <- up-project to key space
V = c_t @ W_UV    <- up-project to value space (different projection)
```

In the **absorbed formulation**, projection matrices are folded into Q and output:

```
S = Q @ (c_t @ W_UK)^T  ->  S = (Q @ W_UK^T) @ c_t^T    (W_UK absorbed into Q)
O = P @ (c_t @ W_UV)    ->  O = (P @ c_t) @ W_UV          (W_UV applied after kernel)
```

So the kernel operates on `c_t` directly. `KV[:, :512]` is `c_t` (nope portion),
used for both attention score and value weighting. `KV[:, 512:576]` is the rope
component, only used for attention score.

```
KV[token, 576]  <- single compressed vector per token
  |-- [:512]   = c_t (nope) -- used as both "K" and "V" in the kernel
  `-- [512:]   = rope component -- only used for attention score S
```

## Step 1: Preprocessing (separate kernel)

```
Input:  O[T, H, D_V],  dO[T, H, D_V]
Output: Delta[T, H]

Delta[t, h] = sum_d( O[t,h,d] * dO[t,h,d] )     (row-wise dot product)
```

Grid: `(total_tokens, ceil(num_heads / BLOCK_H))`

## Step 2: Python-side prep (before main kernel)

```python
Q_T  = Q.transpose(1,2).contiguous()    # [T, H, D_QK] -> [T, D_QK, H]
dO_T = dO.transpose(1,2).contiguous()   # [T, H, D_V]  -> [T, D_V, H]
```

These pre-transposed copies exist because `tl.trans()` on computed tensors (P, dS)
doesn't work well on AMD. By pre-transposing Q and dO, we can compute
`dKV = Q_T @ dS` directly as `[D, H] @ [H, K] -> [D, K]` without transposing
the intermediate results P and dS.

Extra cost: ~2.3GB memory traffic for B1_S4096_H128 config (~0.4ms on MI300X).
TileLang avoids this by using `transpose_A=True` in `T.gemm()`.

## Step 3: Main kernel

Grid: `(total_tokens, ceil(num_heads / BLOCK_H))`
Each program: 1 query token x BLOCK_H heads.
Best autotune config: `BLOCK_H=64, TILE_K=16, warps=4, stages=2`

### Prologue: load loop-invariant data into registers

```
Q_lora    [BH, 512]   <- Q[t, h, :D_V]           # for S, dQ
Q_rope    [BH, 64]    <- Q[t, h, D_V:]            # for S, dQ
dO        [BH, 512]   <- dO[t, h, :]              # for dP
Q_lora_T  [512, BH]   <- Q_T[t, :D_V, h]          # for dKV (pre-transposed)
Q_rope_T  [64, BH]    <- Q_T[t, D_V:, h]          # for dKV (pre-transposed)
dO_T      [512, BH]   <- dO_T[t, :, h]            # for dKV (pre-transposed)
lse       [BH]        <- LSE[t, h]
delta     [BH]        <- Delta[t, h]
dQ_lora   [BH, 512]   = 0                         # accumulator
dQ_rope   [BH, 64]    = 0                         # accumulator
```

**Register pressure hotspot**: Q_lora + Q_rope + dO + Q_lora_T + Q_rope_T + dO_T +
dQ_lora + dQ_rope all live simultaneously -> 105 VGPR spills on MI300X.

### Tile loop: `for t in range(ceil(topk / TILE_K))`

Each iteration processes TILE_K=16 KV tokens from the topk set.

```
+-------------------------------------------------------------+
| Phase 1: Load KV tile from global (gathered via topk indices)|
|                                                              |
|   K_lora_T  [512, 16]  <- KV[topk_pos, :D_V]               |
|   K_rope_T  [64, 16]   <- KV[topk_pos, D_V:]               |
|   (next tile's topk_pos prefetched concurrently)             |
+--------------------------------------------------------------+
| Phase 2: Recompute S and P                                   |
|                                                              |
|   S  [BH,16] = Q_lora @ K_lora_T + Q_rope @ K_rope_T       |
|   S *= scale                                                 |
|   P  [BH,16] = exp(S - lse)                                 |
+--------------------------------------------------------------+
| Phase 3: Compute dP and dS                                   |
|                                                              |
|   dP [BH,16] = dO @ K_lora_T     (same data as V in fwd)    |
|   dS [BH,16] = P * (dP - delta) * scale                     |
+--------------------------------------------------------------+
| Phase 4: Accumulate dQ                                       |
|                                                              |
|   V_lora = trans(K_lora_T)        # [16, 512]               |
|   dQ_lora += dS @ V_lora          # [BH,512]                |
|   K_rope  = trans(K_rope_T)       # [16, 64]                |
|   dQ_rope += dS @ K_rope          # [BH,64]                 |
+--------------------------------------------------------------+
| Phase 5: Compute dKV and atomic scatter                      |
|                                                              |
|   dKV_lora_T [512,16] = Q_lora_T @ dS + dO_T @ P           |
|   dKV_rope_T [64,16]  = Q_rope_T @ dS                       |
|                                                              |
|   atomic_add(dKV[topk_pos, :D_V],  dKV_lora_T)              |
|   atomic_add(dKV[topk_pos, D_V:],  dKV_rope_T)              |
+--------------------------------------------------------------+
```

### Epilogue: store dQ

```
dQ[t, h, :D_V]  <- dQ_lora    (cast to bf16)
dQ[t, h, D_V:]  <- dQ_rope    (cast to bf16)
```

## Step 4: Python-side postprocess

```python
dkv_out = dkv.unsqueeze(1).to(kv.dtype)  # [T, D_QK] fp32 -> [T, 1, D_QK] bf16
```

## 8 dot products per tile iteration

| # | Operation | Shape | Purpose |
|---|-----------|-------|---------|
| 1 | Q_lora @ K_lora_T | [BH,512] @ [512,16] -> [BH,16] | S (nope) |
| 2 | Q_rope @ K_rope_T | [BH,64] @ [64,16] -> [BH,16] | S (rope) |
| 3 | dO @ K_lora_T | [BH,512] @ [512,16] -> [BH,16] | dP |
| 4 | dS @ V_lora | [BH,16] @ [16,512] -> [BH,512] | dQ (nope) |
| 5 | dS @ K_rope | [BH,16] @ [16,64] -> [BH,64] | dQ (rope) |
| 6 | Q_lora_T @ dS | [512,BH] @ [BH,16] -> [512,16] | dKV (nope) |
| 7 | dO_T @ P | [512,BH] @ [BH,16] -> [512,16] | dKV (V grad) |
| 8 | Q_rope_T @ dS | [64,BH] @ [BH,16] -> [64,16] | dKV (rope) |

Dots 1-3 use K_lora_T (same c_t data) for both score and value roles.
Dots 6-7 both contribute to dKV_lora: dot 6 is the gradient through the score path,
dot 7 is the gradient through the value path.

## Key bottleneck: register pressure

All 8 dot products share operands that must stay live simultaneously. With BH=64:

| Tensor | Elements | Role |
|--------|----------|------|
| Q_lora | BH x 512 | dots 1, 4 |
| Q_rope | BH x 64 | dots 2, 5 |
| dO | BH x 512 | dot 3 |
| Q_lora_T | 512 x BH | dot 6 |
| Q_rope_T | 64 x BH | dot 8 |
| dO_T | 512 x BH | dot 7 |
| dQ_lora | BH x 512 | accumulator (dot 4) |
| dQ_rope | BH x 64 | accumulator (dot 5) |

This exceeds the 256 VGPR + 256 AGPR budget on MI300X, causing **105 spills** to
scratch memory (L2-backed, ~424 bytes/lane).

## MI300X benchmark results

| Config | ms | TFLOPS | Autotune |
|--------|---:|-------:|----------|
| B1 S4096 H128 topk1024 | 61.1 | 39.4 | BH=64 TK=16 w=4 s=2 |
| B1 S4096 H128 topk2048 | 119.4 | 40.3 | BH=64 TK=16 w=4 s=2 |
| B1 S8192 H128 topk1024 | 121.9 | 39.5 | BH=64 TK=16 w=4 s=2 |
| B1 S8192 H128 topk2048 | 238.2 | 40.4 | BH=64 TK=16 w=4 s=2 |
| B2 S4096 H128 topk1024 | 121.9 | 39.4 | BH=64 TK=16 w=4 s=2 |
| B1 S4096 H32 topk1024 | 14.4 | 22.7 | BH=32 TK=16 w=8 s=3 |
| B1 S4096 H16 topk1024 | 27.7 | 10.8 | BH=16 TK=16 w=8 s=2 |
| B1 S32768 H128 topk2048 | 960.4 | 40.1 | BH=64 TK=16 w=4 s=2 |
| B1 S65536 H128 topk2048 | 1923.3 | 40.0 | BH=64 TK=16 w=4 s=2 |
| B1 S131072 H128 topk2048 | 3904.5 | 39.4 | BH=64 TK=16 w=4 s=2 |

Note: timing includes transpose prep, preprocess kernel, main kernel, and fp32->bf16 cast.
