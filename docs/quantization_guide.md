# AITER Quantization & Precision Guide

This guide documents all quantization strategies available in AITER, their backend support, fused operations, and how to choose the right precision for your use case.

---

## Quick Reference: Which Quantization Should I Use?

| Use Case | Strategy | QuantType | Target Dtype | Why |
|----------|----------|-----------|-------------|-----|
| **Best accuracy (no quant)** | None | `QuantType.No` | BF16 | No quantization loss |
| **FP8 inference (simple)** | Per-tensor | `QuantType.per_Tensor` | FP8 | Single scale, lowest overhead |
| **FP8 inference (balanced)** | Per-token | `QuantType.per_Token` | FP8 | Per-row scales, good accuracy/perf |
| **FP8 inference (high accuracy)** | Block-scale (1x128) | `QuantType.per_1x128` | FP8 | Per-block scales, best FP8 accuracy |
| **MXFP4 (max compression)** | Block-scale (1x32) | `QuantType.per_1x32` | FP4 | 4x compression, E8M0 scales |
| **Outlier-aware** | SmoothQuant | per-token + channel scale | FP8/INT8 | Handles activation outliers |
| **KV cache compression** | Per-token/block cache | N/A | FP8/INT8 | Reduces KV cache memory |
| **Fused norm + quant** | RMSNorm + FP8/FP4 | N/A | FP8/FP4 | Saves kernel launches |

---

## 1. QuantType Enum

All quantization strategies in AITER are identified by the `QuantType` enum:

```python
from aiter import QuantType

QuantType.No            # No quantization (pass-through)
QuantType.per_Tensor    # Single scale for entire tensor
QuantType.per_Token     # One scale per token (row)
QuantType.per_1x32      # Block-scale with 32-element groups (for MXFP4)
QuantType.per_1x128     # Block-scale with 128-element groups (for FP8)
QuantType.per_128x128   # Large 2D block quantization
```

### Supported Data Types

| Type | Format | Bits | Usage |
|------|--------|------|-------|
| **FP8 (E4M3)** | E4M3FNUZ (GFX942) / E4M3FN (GFX950) | 8 | Default quantized dtype |
| **FP4 (MXFP4)** | E2M1 packed as `fp4x2` (2 values per byte) | 4 | Maximum compression |
| **INT8** | Symmetric integer | 8 | Legacy quantization |
| **FP8 E8M0** | Exponent-only float | 8 | Block scale storage for MXFP4 |
| **BF16/FP16** | Standard floats | 16 | Input/output precision |
| **FP32** | Single precision | 32 | Scale computation, accumulation |

---

## 2. Per-Tensor Quantization

Single scale factor for the entire tensor. Simplest strategy with lowest overhead.

```
scale = max(|x|) / dtype_max
x_quant = round(x / scale)
```

### Backend Support

| Backend | Function | Data Types |
|---------|----------|-----------|
| **PyTorch** | `per_tensor_quant(x)` | FP8, INT8 |
| **HIP** | `per_tensor_quant_hip(x)` | FP8, INT8 |
| **Triton** | `static_per_tensor_quant_fp8_i8()` / `dynamic_per_tensor_quant_fp8_i8()` | FP8, INT8 |

### Key API

```python
from aiter.ops.quant import per_tensor_quant

x_quant, scale = per_tensor_quant(
    x,                  # (M, N) input tensor
    scale=None,         # None = dynamic (compute from data)
    scale_dtype=torch.float32,
    quant_dtype=torch.int8,
)
# x_quant: (M, N) quantized tensor
# scale: scalar
```

### When to Use

- Weight quantization (weights don't change at inference time)
- When simplicity matters more than accuracy
- As a baseline for comparing other strategies

---

## 3. Per-Token Quantization

One scale factor per row (token). The standard strategy for activation quantization in LLM inference.

```
scale[i] = max(|x[i, :]|) / dtype_max    # per row
x_quant[i, :] = round(x[i, :] / scale[i])
```

### Backend Support

| Backend | Function | Data Types |
|---------|----------|-----------|
| **PyTorch** | `pertoken_quant(x)` | FP8, INT8 |
| **HIP** | `per_token_quant_hip(x)` | FP8, INT8 |
| **Triton** | `dynamic_per_token_quant_fp8_i8()` | FP8, INT8 |

### Key API

```python
from aiter.ops.quant import pertoken_quant

x_quant, scale = pertoken_quant(
    x,                  # (M, N) input
    scale=None,         # None = dynamic
    x_scale=None,       # optional SmoothQuant channel scale
    scale_dtype=torch.float32,
    quant_dtype=torch.int8,
)
# x_quant: (M, N) quantized
# scale: (M, 1) per-token scales
```

### When to Use

- Activation quantization during inference (most common)
- When different tokens have different magnitude distributions
- Pairs with per-tensor weight quantization: `gemm_a8w8(act_quant, weight_quant, act_scale, weight_scale)`

---

## 4. Block-Scale Quantization

Per-block scales for fine-grained quantization. Two block sizes are supported.

### Per-1x128 (FP8 Block-Scale)

```
For each block of 128 elements along K:
    scale[m, k//128] = max(|x[m, k:k+128]|) / dtype_max
    x_quant[m, k:k+128] = round(x[m, k:k+128] / scale[m, k//128])
```

### Per-1x32 (MXFP4 Block-Scale)

```
For each block of 32 elements along K:
    scale[m, k//32] = max(|x[m, k:k+32]|) → E8M0 format
    x_quant[m, k:k+32] = round(x[m, k:k+32] / scale[m, k//32]) → E2M1 packed
```

### Backend Support

| Feature | PyTorch | HIP | Triton |
|---------|:---:|:---:|:---:|
| **per_1x128 (FP8)** | Yes | Yes | Yes |
| **per_1x32 (MXFP4)** | Yes | Yes | Yes |
| **E8M0 scale format** | Yes | Yes | Yes |
| **E8M0 shuffle** | Yes | Yes | Yes |
| **Dynamic scale** | Yes | Yes | Yes |

### Key API

```python
from aiter.ops.quant import per_1x32_f4_quant

# MXFP4 quantization
x_fp4, scale_e8m0 = per_1x32_f4_quant(
    x,                  # (M, N) input
    scale=None,         # None = dynamic
    quant_dtype=torch.float8_e4m3fnuz,  # fp4x2 packed
    shuffle=False,      # E8M0 shuffle for cache optimization
)
# x_fp4: (M, N//2) packed FP4
# scale_e8m0: (M, N//32) E8M0 block scales
```

```python
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

# Triton MXFP4 (optimized)
x_fp4, blockscale = dynamic_mxfp4_quant(
    x,                  # (M, N) input
    scaling_mode="even",
)
```

### When to Use

- FP8 block-scale (per_1x128): Best accuracy for FP8 GEMM, pairs with `gemm_a8w8_blockscale`
- MXFP4 block-scale (per_1x32): Maximum compression (4x), pairs with `gemm_a4w4` on MI350
- When per-tensor/per-token scales are too coarse for your accuracy requirements

---

## 5. SmoothQuant

Reduces per-channel activation outliers by pre-multiplying with a learned channel-wise scale before quantization. Moves quantization difficulty from activations to weights.

```
x_smooth = x * smooth_scale        # per-channel scaling
x_quant = quantize(x_smooth)       # now easier to quantize
w_quant = quantize(w / smooth_scale)  # absorb inverse into weights
```

### Backend Support

| Backend | Function |
|---------|----------|
| **PyTorch** | `pertoken_quant(x, x_scale=smooth_scale)` |
| **HIP** | `smooth_per_token_scaled_quant(out, input, scales, smooth_scale)` |
| **CK** | `moe_smoothquant_fwd(out, input, x_scale, topk_ids, y_scale)` |
| **Fused Norm** | `layernorm2d_fwd_with_smoothquant(out, input, xscale, yscale, ...)` |

### Key API

```python
from aiter.ops.quant import pertoken_quant

# SmoothQuant via per-token quantization
x_quant, scale = pertoken_quant(
    x,
    x_scale=smooth_scale,  # (1, N) learned channel scales
    quant_dtype=torch.int8,
)
```

### When to Use

- Models with significant activation outliers (e.g., OPT, BLOOM)
- When per-token FP8 alone doesn't meet accuracy requirements
- Requires offline calibration to compute `smooth_scale`

---

## 6. Fused Quantization Operations

AITER provides fused kernels that combine normalization, activation, and quantization to reduce memory round-trips.

### FP8 Fused Operations

| Function | Pipeline | Backend |
|----------|----------|---------|
| `fused_rms_fp8_per_tensor_static_quant` | [Add residual] → RMSNorm → FP8 quant | Triton |
| `fused_rms_fp8_group_quant` | RMSNorm → per-group FP8 quant | Triton |
| `fused_reduce_act_mul_fp8_group_quant` | SplitK reduce → activation → mul → FP8 quant | Triton |
| `fused_reduce_rms_fp8_group_quant` | SplitK reduce → RMSNorm → FP8 group quant | Triton |
| `fused_silu_mul_fp8_per_tensor_static_quant` | SiLU(gate) * up → FP8 quant | Triton |
| `fused_flatten_fp8_group_quant` | 3D→2D flatten → group quant | Triton |

### MXFP4 Fused Operations

| Function | Pipeline | Backend |
|----------|----------|---------|
| `fused_rms_mxfp4_quant` | RMSNorm → MXFP4 quant | Triton |
| `fused_reduce_act_mul_and_mxfp4_quant` | SplitK reduce → activation → MXFP4 quant | Triton |
| `fused_reduce_rms_mxfp4_quant` | SplitK reduce → RMSNorm → MXFP4 quant | Triton |
| `fused_flatten_mxfp4_quant` | 3D→2D flatten → MXFP4 quant | Triton |
| `fused_dynamic_mxfp4_quant_moe_sort` | MXFP4 quant → MOE token sorting | Triton |

### Key API Examples

```python
from aiter.ops.triton.quant.fused_fp8_quant import (
    fused_rms_fp8_group_quant,
)

# Fused RMSNorm + FP8 group quantization
out_fp8, out_scales = fused_rms_fp8_group_quant(
    inp1,               # (M, N) input
    inp1_weight,        # (N,) RMSNorm weight
    inp1_epsilon,       # RMSNorm epsilon
    group_size=128,     # quantization group size
    inp2=None,          # optional second input
    res1=None,          # optional residual
)
# out_fp8: (M, N) FP8 quantized
# out_scales: (M, N//128) per-group scales
```

```python
from aiter.ops.triton.quant.fused_mxfp4_quant import (
    fused_rms_mxfp4_quant,
)

# Fused RMSNorm + MXFP4 quantization
out_fp4, out_scales = fused_rms_mxfp4_quant(
    x1,                 # (M, N) input
    x1_weight,          # (N,) RMSNorm weight
    x1_epsilon,         # RMSNorm epsilon
    shuffle=False,      # E8M0 shuffle optimization
)
# out_fp4: (M, N//2) packed FP4
# out_scales: (M, N//32) E8M0 block scales
```

### When to Use

- Always prefer fused ops over separate norm → quant when available
- Reduces kernel launches and memory traffic
- Critical path in LLM inference: norm → quant → GEMM

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/triton/quant/fused_fp8_quant.py` | All FP8 fused operations |
| `aiter/ops/triton/quant/fused_mxfp4_quant.py` | All MXFP4 fused operations |

---

## 7. Quantized KV Cache

AITER supports writing quantized values directly to the KV cache, reducing memory footprint for long-context inference.

### Cache Quantization Methods

| Function | Strategy | Layout |
|----------|----------|--------|
| `reshape_and_cache` | No quantization (BF16) | Standard |
| `reshape_and_cache_with_pertoken_quant` | Per-token FP8 | Standard |
| `reshape_and_cache_with_block_quant` | Block-wise FP8 | Standard |
| `reshape_and_cache_with_block_quant_for_asm_pa` | Block-wise for ASM PA | `[blocks, heads, D/x, 16, x]` |
| `indexer_k_quant_and_cache` | On-demand K quantization | Indexed |

### Key API

```python
from aiter.ops.cache import (
    reshape_and_cache_with_pertoken_quant,
    reshape_and_cache_with_block_quant,
)

# Per-token FP8 KV cache write
reshape_and_cache_with_pertoken_quant(
    key, value,
    key_cache, value_cache,
    k_dequant_scales, v_dequant_scales,
    slot_mapping,
)

# Block-quantized KV cache write
reshape_and_cache_with_block_quant(
    key, value,
    key_cache, value_cache,
    k_dequant_scales, v_dequant_scales,
    slot_mapping,
)
```

### When to Use

- Long-context inference where KV cache is the memory bottleneck
- Per-token: simpler, ~50% memory savings
- Block-quant: more complex, better accuracy for long sequences

---

## 8. MOE Quantization Integration

Quantization is deeply integrated with MOE (Mixture of Experts) operations.

### MOE SmoothQuant

```python
from aiter.ops.quant import moe_smoothquant_fwd

# Apply SmoothQuant with expert routing
moe_smoothquant_fwd(
    out,                # output buffer
    input,              # (M, N) activations
    x_scale,            # (N,) channel scales
    topk_ids,           # (M, topk) expert indices
    y_scale,            # per-token output scales
)
```

### MXFP4 + MOE Sorting

```python
from aiter.ops.triton.quant.fused_mxfp4_quant import (
    fused_dynamic_mxfp4_quant_moe_sort,
)

# Fused: MXFP4 quantization + MOE token sorting
x_fp4, blockscale = fused_dynamic_mxfp4_quant_moe_sort(
    x,                  # (M, N) activations
    sorted_ids,         # MOE sorted token indices
    num_valid_ids,      # number of valid tokens
    token_num,          # total tokens
    topk,               # top-K experts per token
)
```

### Interaction with MOE GEMM

```
Tokens → [Routing] → [Sort by Expert] → [Quantize] → [Expert GEMM] → [Unsort]
                                            ↑
                          QuantType determines precision:
                          - No: BF16 GEMM
                          - per_Token: FP8 A8W8
                          - per_1x128: FP8 block-scale
                          - per_1x32: MXFP4 A4W4
```

---

## 9. Backend Dispatch

AITER provides factory functions that return the appropriate quantization implementation for each backend:

```python
from aiter.ops.quant import get_torch_quant, get_hip_quant, get_triton_quant

# Get per-token quant function for Triton backend
quant_fn = get_triton_quant(QuantType.per_Token)
x_quant, scale = quant_fn(x)

# Get per-block quant function for HIP backend
quant_fn = get_hip_quant(QuantType.per_1x32)
x_quant, scale = quant_fn(x)
```

### Backend Selection

| QuantType | PyTorch | HIP | Triton |
|-----------|:---:|:---:|:---:|
| `No` | pass-through | pass-through | pass-through |
| `per_Tensor` | Yes | Yes | Yes |
| `per_Token` | Yes | Yes | Yes |
| `per_1x32` | Yes | Yes | Yes |
| `per_1x128` | Yes (wrapped) | Yes (wrapped) | Yes (wrapped) |
| `per_128x128` | Yes (wrapped) | - | - |

---

## 10. Data Type Flow

### Typical Inference Quantization Pipeline

```
Input (BF16/FP16)
    │
    ▼
[Optional: Fused RMSNorm + Add Residual]
    │
    ▼
[Optional: SmoothQuant (x * channel_scale)]
    │
    ▼
Scale Computation (dynamic):
    ├── per_Tensor:  max(|x|) → 1 scale
    ├── per_Token:   max(|x|, dim=-1) → M scales
    ├── per_1x128:   max(|x|, groups of 128) → M × (N/128) scales
    └── per_1x32:    max(|x|, groups of 32) → M × (N/32) E8M0 scales
    │
    ▼
Quantization:
    ├── FP8:  x_quant = round(x / scale * fp8_max)
    ├── INT8: x_quant = round(x / scale * 127)
    └── FP4:  x_quant = nearest_fp4(x / scale) → packed 2 per byte
    │
    ▼
GEMM (with on-the-fly dequantization):
    output = (x_quant @ w_quant^T) * x_scale * w_scale
    │
    ▼
Output (BF16/FP16, accumulated in FP32)
```

### Choosing the Right Precision

```
Need maximum accuracy?
├── Yes → QuantType.No (BF16, no quantization)
└── No
    ├── Need 2x compression?
    │   ├── Best accuracy → per_1x128 (FP8 block-scale)
    │   ├── Good accuracy → per_Token (FP8 per-row)
    │   └── Simplest → per_Tensor (FP8 global scale)
    ├── Need 4x compression?
    │   └── per_1x32 (MXFP4) — MI350 preferred
    └── Have activation outliers?
        └── SmoothQuant + per_Token
```

---

## 11. FP4/MXFP4 Utilities

Low-level utilities for working with MXFP4 format.

```python
from aiter.utility.fp4_utils import (
    f32_to_mxfp4,      # FP32 → MXFP4 (packed uint8)
    mxfp4_to_f32,      # MXFP4 → FP32
    f32_to_e8m0,       # FP32 → E8M0 (exponent-only scale)
    e8m0_to_f32,       # E8M0 → FP32
    e8m0_shuffle,      # Cache-optimized E8M0 layout
)

from aiter.int4_utils import (
    convert_int8_to_uint32_int4,  # Pack 8 INT4 → 1 UINT32
    rearrange_4bit_elements,       # Reorder for memory access
)
```

### E8M0 Shuffle

E8M0 shuffle reorders block scales for optimal cache prefetch alignment:

```python
# Pads to (256, 8) multiples and permutes for sequential access
shuffled_scale = e8m0_shuffle(scale)  # (M, N//32) → (M_padded, N//32_padded)
```

---

## 12. Fused Communication + Quantization

For multi-GPU deployments, AITER fuses all-reduce with normalization and quantization:

```python
from aiter.ops.communication import all_reduce_rmsnorm_quant

# Fused: AllReduce + RMSNorm + SmoothQuant
all_reduce_rmsnorm_quant(
    input, residual_in,
    xscale,             # smooth channel scales
    weight, bias,       # RMSNorm parameters
    epsilon,
)
```

---

## 13. Test Files Reference

| Test File | Covers |
|-----------|--------|
| `op_tests/test_quant.py` | Core quantization: per-tensor, per-token, per-block |
| `op_tests/test_smoothquant.py` | SmoothQuant: Torch vs CK vs HIP |
| `op_tests/test_rmsnorm2dFusedAddQuant.py` | Fused RMSNorm + quantization |
| `op_tests/test_layernorm2dFusedAddQuant.py` | Fused LayerNorm + quantization |
| `op_tests/test_fused_qk_norm_rope_cache_quant.py` | Fused QK norm + RoPE + cache + quant |
| `op_tests/test_fused_qk_norm_mrope_cache_quant.py` | Multi-dim RoPE + quant |
| `op_tests/triton_tests/quant/test_quant.py` | Triton quantization kernels |
| `op_tests/triton_tests/quant/test_fused_fp8_quant.py` | Triton FP8 fused ops |
| `op_tests/triton_tests/quant/test_fused_mxfp4_quant.py` | Triton MXFP4 fused ops |

---

## 14. Source Files Reference

### Core Quantization

| File | Purpose |
|------|---------|
| `aiter/ops/quant.py` | All quantization strategies, backend dispatch |
| `aiter/ops/enum.py` | `QuantType` enum definition |
| `aiter/utility/dtypes.py` | Data type mappings and utilities |
| `aiter/utility/fp4_utils.py` | FP4/E8M0 conversion utilities |
| `aiter/int4_utils.py` | INT4 packing utilities |

### Triton Quantization

| File | Purpose |
|------|---------|
| `aiter/ops/triton/quant/quant.py` | Per-tensor, per-token, MXFP4 Triton implementations |
| `aiter/ops/triton/quant/fused_fp8_quant.py` | FP8 fused operations |
| `aiter/ops/triton/quant/fused_mxfp4_quant.py` | MXFP4 fused operations |

### Fused Norm + Quantization

| File | Purpose |
|------|---------|
| `aiter/ops/norm.py` | LayerNorm/RMSNorm with SmoothQuant fusion |
| `aiter/ops/fused_qk_norm_rope_cache_quant.py` | QK norm + RoPE + cache + quant |
| `aiter/ops/fused_qk_norm_mrope_cache_quant.py` | Multi-dim RoPE variant |

### Cache Quantization

| File | Purpose |
|------|---------|
| `aiter/ops/cache.py` | Quantized KV cache write operations |
