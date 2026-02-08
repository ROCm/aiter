# AITER Causal Conv1D Operators Guide

This guide documents the Causal Conv1D operators in AITER, used in Mamba-style state-space models and gated architectures for sequence modeling with causal (left-only) convolution.

---

## Quick Reference

| Use Case | Recommended Operation | Backend | Why |
|----------|---------------------|---------|-----|
| **Prefill (variable-length)** | `causal_conv1d_fn` | Triton | Continuous batching, ragged sequences |
| **Decode (single/multi-token)** | `causal_conv1d_update` | Triton | State caching, speculative decoding |
| **Fused conv + QKV split (prefill)** | `causal_conv1d_fn_split_qkv` | Triton | Avoids intermediate allocation |
| **Fused conv + QKV split (decode)** | `causal_conv1d_update_split_qkv` | Triton | Gluon-optimized variants |

---

## 1. Core Operations

### `causal_conv1d_fn` (Forward / Prefill)

Variable-length forward pass with continuous batching support:

```python
from aiter.ops.triton.causal_conv1d import causal_conv1d_fn

out = causal_conv1d_fn(
    x,                      # (dim, cu_seqlen) — concatenated tokens, channel-last
    weight,                 # (dim, width) — convolution weights
    bias,                   # (dim,) or None
    conv_states=conv_states,        # (num_cache_lines, dim, width-1) — state cache
    query_start_loc=query_start_loc, # (batch+1,) int32 — cumulative seq lengths
    seq_lens_cpu=seq_lens_cpu,       # List[int] — per-sequence lengths (CPU)
    cache_indices=cache_indices,     # (batch,) int32 — maps seq to cache slot
    has_initial_state=has_initial_state, # (batch,) bool — load existing state
    activation="silu",      # "silu", "swish", or None
    pad_slot_id=-1,         # sentinel for padded sequences
)
```

**Features:**
- Continuous batching via `cache_indices` for cache slot indirection
- Variable-length sequences via `query_start_loc` (ragged batching)
- Automatic state save/restore for stateful processing
- Kernel widths: 2, 3, 4, 5

### `causal_conv1d_update` (Decode)

Single or multi-token update for autoregressive generation:

```python
from aiter.ops.triton.causal_conv1d import causal_conv1d_update

out = causal_conv1d_update(
    x,                      # (batch, dim, seqlen) or (batch, dim)
    conv_state,             # (num_cache_lines, dim, state_len)
    weight,                 # (dim, width)
    bias=None,
    activation="silu",
    conv_state_indices=indices,      # (batch,) int32 — cache line mapping
    num_accepted_tokens=accepted,    # (batch,) — for speculative decoding
    intermediate_conv_window=window, # saves intermediate states
    pad_slot_id=-1,
)
```

**Features:**
- Single token decode and multi-token speculative decoding
- `num_accepted_tokens` for token rollback in speculative decoding
- `intermediate_conv_window` for saving per-step window states
- Kernel widths: 2, 3, 4

---

## 2. Fused QKV Split Variants

For Gated Delta Net models, fused operations perform causal conv1d **and** split output into Q/K/V, avoiding intermediate tensor allocation.

### Prefill

```python
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.causal_conv1d_fwd_split_qkv import (
    causal_conv1d_fn_split_qkv
)

q, k, v = causal_conv1d_fn_split_qkv(
    x,              # (dim, cu_seqlen) where dim = 2*k_dim + v_dim
    weight, bias, conv_states,
    query_start_loc, seq_lens_cpu,
    k_dim=128,      # query and key dimension
    v_dim=128,      # value dimension
    activation="silu",
)
# q: (cu_seqlen, k_dim), k: (cu_seqlen, k_dim), v: (cu_seqlen, v_dim)
```

### Decode

```python
from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.causal_conv1d_split_qkv import (
    causal_conv1d_update_split_qkv
)

q, k, v = causal_conv1d_update_split_qkv(
    x,              # (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    conv_state, weight,
    key_dim=128, value_dim=128,
    activation="silu",
    use_gluon=True,     # Gluon kernel (default)
    use_gluon_v2=False, # Optimized Gluon v2
)
# q, k: (batch, key_dim, seqlen), v: (batch, value_dim, seqlen)
```

**Decode kernel variants:**
1. Standard Triton
2. Optimized v2 (multi-CU utilization)
3. Gluon (experimental Triton frontend)
4. Gluon v2 (eliminates tuple operations for better register allocation)

---

## 3. Layout Requirements

All operators require **channel-last** input layout (`x.stride(0) == 1`) for coalesced memory access.

| Tensor | Shape | Notes |
|--------|-------|-------|
| `x` (prefill) | `(dim, cu_seqlen)` | Channel-last, concatenated sequences |
| `x` (decode) | `(batch, dim, seqlen)` | `seqlen=1` for single-token decode |
| `weight` | `(dim, width)` | Shared across all sequences |
| `conv_states` | `(num_cache_lines, dim, width-1)` | Indexed via `cache_indices` |

---

## 4. Supported Data Types

| Input | Tolerance |
|-------|-----------|
| float32 | rtol=3e-4, atol=1e-3 |
| bfloat16 | rtol=1e-2, atol=5e-2 |

---

## 5. Backend Support

All implementations are **Triton-only** — no CUDA/HIP/CK/ASM kernels.

| Operator | Triton | Gluon |
|----------|:------:|:-----:|
| `causal_conv1d_fn` | Yes | — |
| `causal_conv1d_update` | Yes | — |
| `causal_conv1d_fn_split_qkv` | Yes | — |
| `causal_conv1d_update_split_qkv` | Yes | Yes (v1, v2) |

---

## 6. Performance Notes

- **Block sizes:** BLOCK_M=8 (tokens), BLOCK_N=256 (features), 2-stage software pipelining
- **State management:** Uses `.ca` cache modifier for prior token loads (L2 cache hints)
- **Gluon v2:** Eliminates tuple operations in hot loops for better register allocation and multi-CU utilization
- **In-place output:** Update kernel writes directly to input tensor `x`

---

## 7. Source Files

| Component | Path |
|---|---|
| Core Python API | `aiter/ops/triton/causal_conv1d.py` |
| Triton kernels | `aiter/ops/triton/_triton_kernels/causal_conv1d.py` |
| Fused prefill split QKV | `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/causal_conv1d_fwd_split_qkv.py` |
| Fused decode split QKV | `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/causal_conv1d_split_qkv.py` |

---

## 8. Test Files

| Test | Path |
|------|------|
| Core causal conv1d | `op_tests/triton_tests/test_causal_conv1d.py` |
