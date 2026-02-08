# AITER Elementwise & Activation Operators Guide

This guide documents all element-wise arithmetic and activation operators available in AITER, including fused gating variants and activation-quantization fusions.

---

## Quick Reference

| Use Case | Recommended Operation | Backend | Why |
|----------|---------------------|---------|-----|
| **SwiGLU gate (LLM FFN)** | `silu_and_mul` | HIP/CK | Standard gated activation for LLaMA/Mistral |
| **SwiGLU + FP8 quantize** | `act_mul_and_fp8_group_quant` | Triton | Fused activation + quantize for inference |
| **SwiGLU + MXFP4 quantize** | `act_mul_and_mxfp4_quant` | Triton | Fused activation + MXFP4 for GFX950 |
| **GELU gate** | `gelu_and_mul` | HIP/CK | GPT-2/BERT-style gated activation |
| **Scaled SiLU (quantized input)** | `scaled_silu_and_mul` | HIP/CK | SiLU with input scale for quantized models |
| **Element-wise arithmetic** | `aiter.add/sub/mul/div` | HIP/CK | Optimized binary ops with broadcasting |
| **Sigmoid / Tanh** | `aiter.sigmoid/tanh` | HIP/CK | AMD-optimized fast math intrinsics |
| **Fused multiply-add** | `fused_mul_add` | Triton | `a * x + b` in one kernel |

---

## 1. Gated Activation Functions (SwiGLU / GeGLU)

The primary activation pattern in modern LLMs. The input tensor has shape `[M, 2*N]` and is split in half: one half is the gate (activation applied), the other is the value. The output is `activation(gate) * value` with shape `[M, N]`.

### Backend Support

| Activation | HIP/CK | Triton | Fused + FP8 Quant | Fused + MXFP4 Quant |
|-----------|:---:|:---:|:---:|:---:|
| **SiLU (SwiGLU)** | Yes | Yes | Yes | Yes |
| **GELU** | Yes | Yes | Yes | Yes |
| **GELU Tanh** | Yes | Yes | Yes | Yes |
| **Scaled SiLU** | Yes | - | - | - |

### Key API Functions

```python
import aiter

# SiLU-and-Mul (SwiGLU gate) — most common for LLaMA/Mistral/DeepSeek
out = torch.empty(M, N, dtype=dtype, device="cuda")
aiter.silu_and_mul(out, input)  # input shape: [M, 2*N]

# Scaled SiLU-and-Mul (for quantized inference with input scale)
aiter.scaled_silu_and_mul(out, input, scale)

# GELU-and-Mul (GeGLU gate)
aiter.gelu_and_mul(out, input)

# GELU-Tanh-and-Mul (approximate GELU gate)
aiter.gelu_tanh_and_mul(out, input)
```

### Fused Activation + Quantization (Triton)

These fuse the gated activation with quantization in a single kernel, avoiding an extra memory round-trip:

```python
from aiter.ops.triton.activation import (
    act_mul_and_fp8_group_quant,    # Activation + FP8 group quantize
    act_mul_and_mxfp4_quant,        # Activation + MXFP4 block-scale quantize
)

# SiLU gate + FP8 group quantization
out, scales = act_mul_and_fp8_group_quant(
    input,                  # [M, 2*N]
    activation="silu",      # "silu", "gelu", or "gelu_tanh"
    group_size=128,
    dtype_quant=torch.float8_e4m3fnuz,
)

# SiLU gate + MXFP4 block-scale quantization
out, scales = act_mul_and_mxfp4_quant(
    input,
    activation="silu",
    scaling_mode="even",    # Scale computation mode
    shuffle=True,           # Shuffle output layout
)
```

---

## 2. Unary Activations

### Sigmoid

Uses AMD fast math intrinsics (`__builtin_amdgcn_exp2f`, `__builtin_amdgcn_rcpf`) for optimized computation.

```python
import aiter

output = aiter.sigmoid(input)
```

### Tanh

```python
output = aiter.tanh(input)
```

### Supported Data Types

| Data Type | Sigmoid | Tanh |
|-----------|:---:|:---:|
| FP16 | Yes | Yes |
| BF16 | Yes | Yes |
| FP32 | Yes | Yes |

---

## 3. Element-wise Binary Arithmetic

Optimized binary operations with full broadcasting support. Uses JIT-compiled kernels that are specialized for the input dtype combination.

### Operations

```python
import aiter

# Out-of-place (return new tensor)
c = aiter.add(a, b)   # c = a + b
c = aiter.sub(a, b)   # c = a - b
c = aiter.mul(a, b)   # c = a * b
c = aiter.div(a, b)   # c = a / b

# In-place (modify first tensor)
aiter.add_(a, b)      # a += b
aiter.sub_(a, b)      # a -= b
aiter.mul_(a, b)      # a *= b
aiter.div_(a, b)      # a /= b
```

### Broadcasting

All binary ops support NumPy-style broadcasting:

```python
# Scalar broadcast
c = aiter.add(tensor_2d, scalar_tensor)

# Dimension broadcast
a = torch.randn(4, 1, device="cuda")
b = torch.randn(1, 8, device="cuda")
c = aiter.mul(a, b)  # → shape [4, 8]
```

### Auto Type Promotion

Output dtype is automatically promoted via `torch.promote_types()`:

```python
a = torch.randn(4, 4, dtype=torch.float16, device="cuda")
b = torch.randn(4, 4, dtype=torch.float32, device="cuda")
c = aiter.add(a, b)  # c.dtype == torch.float32
```

---

## 4. Fused Multiply-Add

Single-kernel element-wise `out = a * x + b` where `a` and `b` can be scalars or tensors:

```python
from aiter.ops.triton.fusions.fused_mul_add import fused_mul_add

# Tensor * tensor + tensor
out = torch.empty_like(x)
fused_mul_add(x, a_tensor, b_tensor, out)

# Scalar * tensor + scalar
fused_mul_add(x, 2.0, 1.0, out)  # out = 2*x + 1
```

---

## 5. GEMM-Fused Activations

Activations can be fused directly into GEMM (matrix multiply) operations, eliminating the need for a separate activation kernel:

### GEMM + Activation

```python
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

# Matrix multiply with post-GEMM activation
y = gemm_a16w16(x, weight, bias=None, dtype=torch.bfloat16,
                activation="silu")  # Applied after matmul
```

### GEMM + Gated Activation (SwiGLU FFN)

```python
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16_gated

# Gated matmul: splits output, applies activation to gate half
y = gemm_a16w16_gated(x, weight, dtype=torch.bfloat16,
                      activation="silu")
```

### Feed-Forward Blocks

Complete FFN blocks with gating built in:

```python
from aiter.ops.triton.gemm.feed_forward.ff_a16w16 import (
    ff_a16w16_gated,    # SwiGLU: x → up_proj → gate*value → down_proj
    ff_a16w16_nogate,   # Standard: x → up_proj → activation → down_proj
)
```

---

## 6. Triton Activation Kernels

Available activation functions in Triton kernels (used internally by fused operations):

| Function | Formula | Usage |
|----------|---------|-------|
| `_silu(x)` | `x * sigmoid(x)` | SwiGLU gates |
| `_gelu(x)` | `0.5 * x * (1 + erf(x / sqrt(2)))` | Standard GELU |
| `_gelu_tanh(x)` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` | Approximate GELU |
| `_tanh(x)` | `2 * sigmoid(2x) - 1` | Tanh |
| `_relu(x)` | `max(0, x)` | ReLU |

---

## 7. Decision Tree

```
Need activation?
├── Gated FFN (SwiGLU/GeGLU)?
│   ├── Standard inference → aiter.silu_and_mul()
│   ├── With input scaling → aiter.scaled_silu_and_mul()
│   ├── Need FP8 output → act_mul_and_fp8_group_quant()
│   ├── Need MXFP4 output → act_mul_and_mxfp4_quant()
│   └── Fused with GEMM → gemm_a16w16_gated() or ff_a16w16_gated()
├── Standalone activation?
│   ├── Sigmoid → aiter.sigmoid()
│   └── Tanh → aiter.tanh()
├── Element-wise arithmetic?
│   ├── Standard ops → aiter.add/sub/mul/div()
│   ├── In-place → aiter.add_/sub_/mul_/div_()
│   └── Fused a*x+b → fused_mul_add()
└── GELU variant?
    ├── Standard → aiter.gelu_and_mul()
    └── Tanh approx → aiter.gelu_tanh_and_mul()
```

---

## 8. Source Files

| Component | Path |
|-----------|------|
| Gated activation API | `aiter/ops/activation.py` |
| Binary/unary ops API | `aiter/ops/aiter_operator.py` |
| Triton activation wrappers | `aiter/ops/triton/activation.py` |
| Triton activation kernels | `aiter/ops/triton/_triton_kernels/activation.py` |
| Triton fused mul-add | `aiter/ops/triton/fusions/fused_mul_add.py` |
| Triton GEMM + activation | `aiter/ops/triton/gemm/basic/gemm_a16w16.py` |
| Triton gated FFN | `aiter/ops/triton/gemm/feed_forward/ff_a16w16.py` |
| HIP activation kernels | `csrc/kernels/activation_kernels.cu` |
| HIP unary operators | `csrc/kernels/unary_operator.cu` |
| HIP binary operators | `csrc/kernels/binary_operator.cu` |

---

## 9. Test Files

| Test | Path |
|------|------|
| Activation (SiLU, scaled) | `op_tests/test_activation.py` |
| Triton activation + quant | `op_tests/triton_tests/test_activation.py` |
| Add | `op_tests/test_aiter_add.py` |
| Add in-place | `op_tests/test_aiter_addInp.py` |
| Sigmoid | `op_tests/test_aiter_sigmoid.py` |
| Fused mul-add | `op_tests/triton_tests/fusions/test_fused_mul_add.py` |
