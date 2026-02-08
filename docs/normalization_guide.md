# AITER Normalization Variants & Backend Guide

This guide documents all normalization operators available in AITER (RMSNorm, LayerNorm, GroupNorm), their fused variants, backend support, and how to choose the right one for your use case.

---

## Quick Reference: Which Normalization Should I Use?

| Use Case | Recommended Variant | Backend | Why |
|----------|-------------------|---------|-----|
| **Standard LLM inference** | RMSNorm | ASM | Best latency, used by LLaMA/Mistral/DeepSeek |
| **LLM with residual stream** | Fused Add + RMSNorm | CK or ASM | One kernel for residual add + normalize |
| **Quantized inference (FP8)** | RMSNorm + Dynamic Quant | ASM | Fused normalize + per-token FP8 quantize |
| **SmoothQuant inference** | RMSNorm + SmoothQuant | CK | Fused normalize + two-scale quantize |
| **BERT/GPT-2 style models** | LayerNorm | CK | Full LayerNorm with bias support |
| **Vision models (ViT, UNet)** | GroupNorm | CK | Group-wise normalization |
| **Distributed inference** | RS + RMSNorm + Quant + AG | Triton (Iris) | Fused communication + normalize + quantize |
| **Prototyping / new models** | Triton variants | Triton | Portable, supports training (backward pass) |

---

## 1. RMSNorm (Root Mean Square Normalization)

The primary normalization used in modern LLMs (LLaMA, Mistral, DeepSeek, Qwen). Simpler and faster than LayerNorm since it omits the mean-centering step.

**Formula**: `output = input * weight / sqrt(mean(input^2) + epsilon)`

### Backend Support

| Feature | CK | ASM | Triton |
|---------|:---:|:---:|:---:|
| **Basic RMSNorm** | Yes | Yes | Yes |
| **Fused Add + RMSNorm** | Yes | Yes | Yes |
| **+ SmoothQuant** | Yes | - | Yes |
| **+ Dynamic Quant (per-token)** | Yes | Yes | Yes |
| **+ Dynamic Quant (per-group)** | - | Yes | Yes |
| **+ Pad to multiple** | - | - | Yes |
| **Backward pass (training)** | - | - | Yes |
| **BF16 / FP16 input** | Yes | Yes | Yes |
| **FP32 input** | - | - | Yes |
| **Model-sensitive mode** | Yes | - | - |
| **N > 8192** | Yes | - | Yes |
| **Shuffle layout scales** | - | Yes | - |

### Key API Functions

```python
import aiter

# Basic RMSNorm
out = aiter.rmsnorm2d_fwd(input, weight, epsilon)

# Fused Add + RMSNorm (residual connection)
aiter.rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)

# RMSNorm + SmoothQuant (two-scale quantization)
aiter.rmsnorm2d_fwd_with_smoothquant(out, input, xscale, yscale, weight, epsilon)

# Fused Add + RMSNorm + SmoothQuant
aiter.rmsnorm2d_fwd_with_add_smoothquant(
    out, input, residual_in, residual_out,
    xscale, yscale, weight, epsilon
)

# RMSNorm + Dynamic Quantization (per-token)
aiter.rmsnorm2d_fwd_with_dynamicquant(
    out, input, yscale, weight, epsilon,
    group_size=0, shuffle_scale=False
)

# Fused Add + RMSNorm + Dynamic Quantization
aiter.rmsnorm2d_fwd_with_add_dynamicquant(
    out, input, residual_in, residual_out,
    yscale, weight, epsilon
)
```

### ASM Backend (Direct)

```python
# ASM-optimized basic RMSNorm
aiter.rmsnorm(out, input, weight, epsilon)

# ASM: Add + RMSNorm
aiter.add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon)

# ASM: RMSNorm + Quantize (per-token or per-group)
aiter.rmsnorm_quant(out, input, scale, weight, epsilon,
                    group_size=0, shuffle_scale=False)

# ASM: Add + RMSNorm + Quantize
aiter.add_rmsnorm_quant(out, input, residual_in, residual_out,
                        scale, weight, epsilon,
                        group_size=0, shuffle_scale=False)
```

### Triton Backend

```python
from aiter.ops.triton.normalization.rmsnorm import (
    rms_norm,                           # Basic forward
    rmsnorm2d_fwd_with_add,             # Fused add
    rmsnorm2d_fwd_with_smoothquant,     # SmoothQuant
    rmsnorm2d_fwd_with_dynamicquant,    # Dynamic quant
    rmsnorm2d_fwd_with_add_smoothquant, # Add + SmoothQuant
    rmsnorm2d_fwd_with_add_dynamicquant,# Add + dynamic quant
)
```

### Backend Dispatch Logic

The dispatcher in `rmsnorm2d_fwd` and `rmsnorm2d_fwd_with_dynamicquant` selects the backend automatically:

- **CK backend** when `use_model_sensitive_rmsnorm > 0` or `N > 8192`
- **ASM backend** otherwise (for standard sizes, best latency)
- **Triton** must be called explicitly via `aiter.ops.triton.normalization.rmsnorm`

### Performance Notes

- **Large M, small N** (M > 8192, N <= 2048): Triton has a specialized kernel path `_rmsnorm_kernel_large_m_small_n` with configurable BLOCK_M/BLOCK_N
- **Per-group quantization** (`group_size > 0`): Only available on ASM backend; use group sizes 32 or 128
- **Shuffle scale layout**: Optimizes scale tensor layout for downstream ASM kernels; only available on ASM

---

## 2. LayerNorm (Layer Normalization)

Standard LayerNorm with learnable weight and bias. Used in BERT, GPT-2, and other architectures that require mean-centering.

**Formula**: `output = (input - mean) / sqrt(variance + epsilon) * weight + bias`

### Backend Support

| Feature | CK | ASM | Triton |
|---------|:---:|:---:|:---:|
| **Basic LayerNorm** | Yes | - | Yes |
| **Fused Add + LayerNorm** | Yes | Yes | Yes |
| **+ SmoothQuant** | Yes | Yes | Yes |
| **+ Dynamic Quant** | - | - | Yes |
| **Backward pass (training)** | - | - | Yes |
| **BF16 / FP16 input** | Yes | Yes | Yes |
| **FP32 input** | - | - | Yes |
| **x_bias (pre-norm bias)** | Yes | Yes | Yes |

### Key API Functions

```python
import aiter

# Basic LayerNorm
out = aiter.layer_norm(input, weight, bias, epsilon)
# or
out = aiter.layernorm2d_fwd(input, weight, bias, epsilon)

# Fused Add + LayerNorm
aiter.layernorm2d_fwd_with_add(
    out, input, residual_in, residual_out, weight, bias, epsilon
)

# LayerNorm + SmoothQuant
aiter.layernorm2d_fwd_with_smoothquant(
    out, input, xscale, yscale, weight, bias, epsilon
)

# Fused Add + LayerNorm + SmoothQuant
aiter.layernorm2d_fwd_with_add_smoothquant(
    out, input, residual_in, residual_out,
    xscale, yscale, weight, bias, epsilon
)
```

### ASM Backend (Direct)

```python
# ASM: Add + LayerNorm (residual)
aiter.layernorm2d_with_add_asm(
    out, input, residual_in, residual_out, weight, bias, epsilon
)

# ASM: Add + LayerNorm + SmoothQuant
aiter.layernorm2d_with_add_smoothquant_asm(
    out, input, residual_in, residual_out,
    xscale, yscale, weight, bias, epsilon
)
```

### Triton Backend

```python
from aiter.ops.triton.normalization.norm import (
    layer_norm,                              # Basic forward
    layernorm2d_fwd_with_add,                # Fused add
    layernorm2d_fwd_with_dynamicquant,       # Dynamic quant
    layernorm2d_fwd_with_smoothquant,        # SmoothQuant
    layernorm2d_fwd_with_add_dynamicquant,   # Add + dynamic quant
    layernorm2d_fwd_with_add_smoothquant,    # Add + SmoothQuant
)
```

### x_bias Parameter

LayerNorm supports an optional `x_bias` tensor that is added to the input before normalization. This is used in some model architectures that require a pre-normalization bias.

```python
# LayerNorm with pre-norm bias
out = aiter.layer_norm(input, weight, bias, epsilon, x_bias=pre_bias)
```

---

## 3. GroupNorm

Group normalization divides channels into groups and normalizes within each group. Commonly used in vision models (UNet, ViT) where batch normalization is not suitable.

### Backend Support

| Feature | CK |
|---------|:---:|
| **Basic GroupNorm** | Yes |
| **BF16 / FP16 input** | Yes |
| **Affine (weight + bias)** | Yes |

### Key API Functions

```python
from aiter import GroupNorm

# Module-based API (drop-in replacement for torch.nn.GroupNorm)
gn = GroupNorm(num_groups=32, num_channels=256)
out = gn(input)

# Functional API
from aiter.ops.groupnorm import _groupnorm_run
out = _groupnorm_run(input, num_groups, weight, bias, eps)
```

---

## 4. Fused Add + RMSNorm + Pad

A Triton-only variant that combines residual addition, RMS normalization, and output padding to a specified multiple. Useful for models that require dimension alignment.

```python
from aiter.ops.triton.normalization.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad

# RMSNorm with optional residual and padding
output = fused_add_rmsnorm_pad(
    x, weight, epsilon,
    res=None,               # Optional residual tensor
    x_pad_to_multiple=256   # Pad output N to multiple of this
)
# Returns (output,) or (output, residual_out) if residual provided
```

---

## 5. Fused Q/K Norm + RoPE + Cache + Quantization

Mega-fused kernels that combine head splitting, RMSNorm on Q and K, rotary position embedding, KV cache write, and quantization in a single kernel launch. These are the highest-performance path for LLM inference.

### Standard RoPE Variant

```python
import aiter

# Fused: split QKV → norm Q,K → apply RoPE → write KV cache → quantize
aiter.fused_qk_norm_rope_cache_quant_shuffle(
    qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
    qw, kw,                    # Q and K norm weights
    cos_sin_cache,             # Pre-computed RoPE cos/sin
    is_neox_style,             # True for NeoX rotation
    pos_ids,                   # Position IDs for RoPE
    k_cache, v_cache,          # KV cache tensors
    slot_mapping,              # Cache slot mapping
    kv_cache_dtype,            # "auto", "fp8", "int8"
    k_scale, v_scale           # Quantization scales
)

# Per-token scaling variant
aiter.fused_qk_norm_rope_cache_pts_quant_shuffle(...)

# Dual-stream variant
aiter.fused_qk_norm_rope_2way(...)
```

### Multimodal RoPE (MRoPE) Variant

```python
# 3D MRoPE for vision-language models
aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(...)
```

---

## 6. Distributed: Reduce-Scatter + RMSNorm + Quant + All-Gather

For multi-GPU inference, this fused kernel combines distributed communication with normalization and quantization, minimizing inter-GPU synchronization overhead.

```python
from aiter.ops.triton.comms.fused.reduce_scatter_rmsnorm_quant_all_gather import (
    reduce_scatter_rmsnorm_quant_all_gather,
)
```

Requires the [Iris library](https://github.com/ROCm/iris) for GPU-initiated communication.

---

## 7. Quantization Integration

### SmoothQuant

Two-scale quantization that reduces quantization error by smoothing the activation distribution:

- **xscale** `[N]` (per-column): Applied to input before normalization
- **yscale** `[M, 1]` (per-row): Computed during normalization, output per-token scale

```python
# Pattern: input → (input * xscale) → norm → quantize → (output, yscale)
aiter.rmsnorm2d_fwd_with_smoothquant(out, input, xscale, yscale, weight, epsilon)
```

### Dynamic Quantization

Per-token quantization with scale computed on-the-fly:

```python
# Per-token: one scale per row
aiter.rmsnorm2d_fwd_with_dynamicquant(out, input, yscale, weight, epsilon)

# Per-group: group_size=32 or 128, scales per group
aiter.rmsnorm2d_fwd_with_dynamicquant(
    out, input, yscale, weight, epsilon,
    group_size=128, shuffle_scale=False
)
```

### Output Data Types

| Backend | Output Types |
|---------|-------------|
| CK SmoothQuant | INT8 |
| CK Dynamic Quant | INT8, FP8 |
| ASM Quant | INT8, FP8, FP4x2 (GFX950) |
| Triton Dynamic Quant | INT8, FP8 |

---

## 8. Decision Tree

```
Need normalization?
├── LLM model (no bias needed)?
│   ├── Standard inference → aiter.rmsnorm2d_fwd()
│   ├── With residual connection → aiter.rmsnorm2d_fwd_with_add()
│   ├── Need quantized output?
│   │   ├── SmoothQuant → rmsnorm2d_fwd_with_smoothquant()
│   │   ├── Per-token FP8 → rmsnorm2d_fwd_with_dynamicquant()
│   │   └── Per-group FP8 → rmsnorm_quant(group_size=128)
│   ├── Need training (backward) → Triton rms_norm()
│   └── Multi-GPU → reduce_scatter_rmsnorm_quant_all_gather()
├── Model with bias (BERT, GPT-2)?
│   └── LayerNorm → aiter.layer_norm()
└── Vision model?
    └── GroupNorm → aiter.GroupNorm()
```

---

## 9. Source Files

| Component | Path |
|-----------|------|
| RMSNorm Python API | `aiter/ops/rmsnorm.py` |
| LayerNorm Python API | `aiter/ops/norm.py` |
| GroupNorm Python API | `aiter/ops/groupnorm.py` |
| Triton RMSNorm kernels | `aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py` |
| Triton LayerNorm kernels | `aiter/ops/triton/_triton_kernels/normalization/norm.py` |
| Triton Fused Add RMSNorm Pad | `aiter/ops/triton/_triton_kernels/normalization/fused_add_rmsnorm_pad.py` |
| CK RMSNorm kernels | `csrc/py_itfs_ck/rmsnorm_ck_kernels.cu` |
| CK LayerNorm kernels | `csrc/py_itfs_ck/norm_kernels.cu` |
| ASM RMSNorm kernels | `csrc/kernels/rmsnorm_kernels.cu`, `csrc/kernels/rmsnorm_quant_kernels.cu` |
| ASM LayerNorm kernels | `csrc/py_itfs_cu/asm_layernorm.cu` |
| Fused QK Norm + RoPE | `aiter/ops/fused_qk_norm_rope_cache_quant.py` |
| Fused QK Norm + MRoPE | `aiter/ops/fused_qk_norm_mrope_cache_quant.py` |
| Distributed Fused Norm | `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` |

---

## 10. Test Files

| Test | Path |
|------|------|
| RMSNorm basic | `op_tests/test_rmsnorm2d.py` |
| RMSNorm + quant | `op_tests/test_rmsnorm2dFusedAddQuant.py` |
| LayerNorm basic | `op_tests/test_layernorm2d.py` |
| LayerNorm + quant | `op_tests/test_layernorm2dFusedAddQuant.py` |
| GroupNorm | `op_tests/test_groupnorm.py` |
| Triton RMSNorm | `op_tests/triton_tests/normalization/test_rmsnorm.py` |
| Triton LayerNorm | `op_tests/triton_tests/normalization/test_layernorm.py` |
| Triton Fused Add Pad | `op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py` |
| Fused QK Norm + RoPE | `op_tests/test_fused_qk_norm_rope_cache_quant.py` |
| Fused QK Norm + MRoPE | `op_tests/test_fused_qk_norm_mrope_cache_quant.py` |
| Distributed Norm | `op_tests/multigpu_tests/triton_test/test_fused_rs_rmsnorm_quant_ag.py` |
| Benchmark | `op_tests/op_benchmarks/triton/bench_rmsnorm.py` |
