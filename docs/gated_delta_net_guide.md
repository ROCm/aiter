# AITER Gated Delta Net Operators Guide

This guide documents the Gated Delta Net (GDN) operators in AITER, implementing the gated delta rule recurrent mechanism for linear attention models. These are **forward-only** inference implementations adapted from flash-linear-attention.

---

## Quick Reference

| Use Case | Recommended Operation | When to Use |
|----------|---------------------|-------------|
| **Autoregressive decode** | `fused_recurrent_gated_delta_rule` | Short sequences, low memory |
| **Long-sequence prefill** | `chunk_gated_delta_rule` | Sequences > 1024 tokens |
| **Fused sigmoid gating** | `fused_sigmoid_gating_delta_rule_update` | Models with sigmoid gate parameterization |

---

## 1. Fused Recurrent (Decode / Short Sequences)

Processes sequences step-by-step in a single fused kernel. Optimal for autoregressive generation.

```python
from aiter.ops.triton.gated_delta_net import fused_recurrent_gated_delta_rule

o, final_state = fused_recurrent_gated_delta_rule(
    q,              # [B, T, H, K] — queries
    k,              # [B, T, H, K] — keys
    v,              # [B, T, HV, V] — values (HV >= H for GVA)
    g=g,            # [B, T, HV] — global gate/decay (log space), optional
    gk=gk,          # [B, T, HV, K] — per-key gate (log space), optional
    gv=gv,          # [B, T, HV, V] — per-value gate (log space), optional
    beta=beta,      # [B, T, HV] or [B, T, HV, V] — beta parameter
    scale=None,     # defaults to 1/sqrt(K)
    initial_state=h0,  # [N, HV, K, V] — initial hidden state
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,    # [N+1] for variable-length sequences
)
# o: [B, T, HV, V], final_state: [N, HV, K, V]
```

**Delta rule update (per step):**
```
h = h * exp(g)                          # Apply decay
v_new = beta * (v - sum(h * k, dim=-2)) # Delta rule
h = h + k[:, :, None] * v_new[:, None, :] # Update
o = sum(h * q, dim=-2)                  # Output
```

**Features:**
- Grouped Value Attention (GVA): `HV > H` allows more value heads than query/key heads
- Three gating modes: global (`g`), per-key (`gk`), per-value (`gv`)
- Variable-length sequences via `cu_seqlens`
- Memory: O(B × HV × K × V) for hidden state

---

## 2. Chunk-Based (Prefill / Long Sequences)

Divides sequences into 64-token chunks for parallel processing. Better throughput for long sequences.

```python
from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule

o, final_state = chunk_gated_delta_rule(
    q,              # [B, T, H, K]
    k,              # [B, T, H, K]
    v,              # [B, T, H, V]
    g,              # [B, T, H] — gate/decay (log space, required)
    beta,           # [B, T, H] — beta parameter (required)
    scale=None,
    initial_state=h0,  # [N, H, K, V]
    output_final_state=True,
    cu_seqlens=None,
)
```

**Chunk processing pipeline:**
1. `chunk_local_cumsum` — cumulative gates within chunks
2. `chunk_scaled_dot_kkt_fwd` — WY representation (K @ K^T)
3. `solve_tril` — triangular system solve
4. `recompute_w_u_fwd` — recompute W and U matrices
5. `chunk_gated_delta_rule_fwd_h` — hidden state updates
6. `chunk_fwd_o` — final output

**Constraints:** K ≤ 256 (uses 64-dim blocks, max 4 blocks), chunk size fixed at 64.

---

## 3. Fused Sigmoid Gating

Combines sigmoid gating computation with delta rule update in a single kernel:

```python
from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update
)

o, final_state = fused_sigmoid_gating_delta_rule_update(
    q, k, v,
    A_log,          # [HV] — log space gate parameter
    a,              # [B, T, HV] — gate activation
    dt_bias,        # [HV] — delta time bias
    softplus_beta,  # float — softplus beta
    softplus_threshold, # float — softplus threshold
    b,              # [B, T, HV] — beta logits
    initial_state=h0,
    output_final_state=True,
)
```

**Computes internally:**
- `g = -exp(A_log) * softplus(a + dt_bias, beta, threshold)`
- `beta = sigmoid(b)`

---

## 4. Supported Data Types

| Tensor | Supported dtypes | Notes |
|--------|-----------------|-------|
| q, k | float16, bfloat16, float32 | Internally converted to float32 |
| v, beta, g | float16, bfloat16, float32 | — |
| initial_state / final_state | float32 | Always float32 |
| Accumulation | float32 | All internal computation |

**Test tolerances:** fp16: 0.002, bf16: 0.005

---

## 5. Backend Support

All implementations are **Triton-only**.

**Hardware-specific tuning:**
- NVIDIA Hopper: Reduced warp configs (2, 4 instead of 2, 4, 8, 16)
- AMD ROCm: Limited num_stages (2, 3 instead of 2, 3, 4) due to compiler constraints
- Shared memory checks for block size tuning

**Environment variables:**
| Variable | Description |
|----------|-------------|
| `FLA_USE_FAST_OPS=1` | Enable fast math operations |
| `FLA_CACHE_RESULTS=1` | Enable autotune cache (default: on) |
| `FLA_USE_CUDA_GRAPH=1` | Enable CUDA graphs (NVIDIA only) |
| `FLA_USE_TMA=1` | Enable Tensor Memory Accelerator (Hopper+) |

---

## 6. Choosing Between Variants

```
Need gated delta rule inference?
├── Short sequences (T < 512) or decode?
│   └── fused_recurrent_gated_delta_rule()
├── Long sequences (T > 1024)?
│   └── chunk_gated_delta_rule()
├── Sigmoid gate parameterization?
│   └── fused_sigmoid_gating_delta_rule_update()
└── Need training / backward pass?
    └── Use flash-linear-attention library instead
```

---

## 7. Source Files

| Component | Path |
|---|---|
| Main module | `aiter/ops/triton/gated_delta_net/gated_delta_rule.py` |
| Fused recurrent kernel | `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_recurrent.py` |
| Fused sigmoid gating | `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_sigmoid_gating_recurrent.py` |
| Chunk kernels | `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk.py` |
| Chunk hidden state | `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk_delta_h.py` |
| Chunk output | `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk_o.py` |
| Utility: L2 norm | `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils/l2norm.py` |
| Utility: cumsum | `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils/cumsum.py` |
| Utility: triangular solve | `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils/solve_tril.py` |
| Utility: WY representation | `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils/wy_representation.py` |
| Fused causal conv1d split QKV | See [Causal Conv1D Guide](causal_conv1d_guide.md) |

---

## 8. Test Files

| Test | Path |
|------|------|
| All GDN variants | `op_tests/triton_tests/test_gated_delta_rule.py` |
