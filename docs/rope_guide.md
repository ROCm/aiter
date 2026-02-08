# AITER RoPE (Rotary Position Embedding) Guide

This guide documents all RoPE variants available in AITER, their tensor formats, rotation styles, scaling methods, and backend support.

---

## Quick Reference: Which RoPE Variant Should I Use?

| Use Case | Recommended Variant | Backend | Why |
|----------|-------------------|---------|-----|
| **Standard LLM inference** | `rope_cached_positions_2c_fwd` | HIP (default) | Cached cos/sin, processes Q and K together |
| **LLM training** | `RoPECached` autograd class | Triton | Full forward + backward support |
| **Packed sequences (variable length)** | `rope_thd_fwd` | HIP or Triton | THD format handles cumulative seq lengths |
| **Vision models (ViT)** | `rope_2d_fwd` | HIP | 2D height/width position encoding |
| **Vision-language models** | `rope_fwd_3d` | Triton | 3D temporal + spatial encoding |
| **DeepSeek / Qwen VL** | `MRotaryEmbedding` | HIP | Multimodal section-based splitting |
| **Fused with QK norm + cache** | `fused_qk_norm_rope_cache_*` | HIP | Maximum fusion for inference |
| **Prototyping / new architectures** | Triton variants | Triton | Portable, easy to modify |

---

## 1. Rotation Styles

AITER supports two standard rotation styles:

| Style | Enum Value | Description | Used By |
|-------|-----------|-------------|---------|
| **NeoX** | `RotateStyle.NEOX` (0) | Splits dimension in half, rotates second half | LLaMA, Mistral, Qwen |
| **GPT-J** | `RotateStyle.GPTJ` (1) | Interleaved rotation of odd-indexed elements | GPT-J, GPT-NeoX |

**NeoX style**: `[x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]` where x1 = first half, x2 = second half

**GPT-J style**: `[x0, x1, x2, x3, ...] → rotated pairs at interleaved positions`

---

## 2. Tensor Formats

### SBHD Format (Sequence-Batch-Head-Dim)

The default format for most RoPE operations.

```
Input/Output:  (S, B, H, D)    # seq_len, batch, num_heads, head_dim
Frequencies:   (S, 1, 1, D//2) # if reuse_freqs_front_part=True
               (S, 1, 1, D)    # if reuse_freqs_front_part=False
Cos/Sin cache: (max_seq, D//2) # pre-computed
Positions:     (S, B)          # position indices
```

### THD Format (Token-Head-Dim)

Packed sequence format for variable-length batches (no padding).

```
Input/Output:  (T, H, D)       # T = sum of all sequence lengths
Cu_seqlens:    (B+1,)          # cumulative sequence lengths
Frequencies:   (max_seq, D//2) # or (max_seq, D)
```

### 2D Format (Image Positions)

For vision transformers with 2D spatial positions.

```
Input/Output:  (B, S, H, D)    # S = height * width
Cos_h/Sin_h:  height cos/sin   # height-dimension embeddings
Cos_w/Sin_w:  width cos/sin    # width-dimension embeddings
```

### 3D Format (Multimodal)

For vision-language models with temporal + spatial dimensions.

```
Input/Output:  (B, S, H, D)    # S = time * height * width
```

---

## 3. Backend Support

| Feature | HIP/ASM | Triton | CK |
|---------|:---:|:---:|:---:|
| **SBHD forward** | Yes | Yes | Ref only |
| **SBHD backward** | Yes | Yes | - |
| **THD forward** | Yes | Yes | - |
| **THD backward** | Yes | Yes | - |
| **2D forward** | Yes | Yes | - |
| **2D backward** | Yes | - | - |
| **3D forward** | - | Yes | - |
| **Cached cos/sin** | Yes | Yes | - |
| **Position indices** | Yes | Yes | - |
| **Position offsets** | Yes | Yes | - |
| **Two-channel (Q+K)** | Yes | Yes | - |
| **GQA support** | Yes (via 2c APIs) | Yes | - |
| **In-place** | Yes | Yes | - |
| **BF16** | Yes | Yes | Yes |
| **FP16** | Yes | Yes | Yes |
| **FP32** | Yes | Yes | - |

---

## 4. Core API Functions

### Single-Input Forward

```python
import aiter

# With on-the-fly frequencies (SBHD format)
output = aiter.rope_fwd(input, freqs, rotate_style=0,
                        reuse_freqs_front_part=True, nope_first=False)

# In-place variant
aiter.rope_fwd_inplace(input, freqs, rotate_style=0,
                       reuse_freqs_front_part=True, nope_first=False)

# Backward
input_grads = aiter.rope_bwd(output_grads, freqs, rotate_style=0,
                             reuse_freqs_front_part=True, nope_first=False)
```

### Two-Channel (Q and K Together)

```python
# Process Q and K with same frequencies
out_q, out_k = aiter.rope_2c_fwd(
    input_q, input_k, freqs,
    rotate_style=0, reuse_freqs_front_part=True, nope_first=False
)

# In-place
aiter.rope_2c_fwd_inplace(input_q, input_k, freqs, ...)
```

### With Cached Cos/Sin

```python
# Pre-computed cos/sin (most common for inference)
output = aiter.rope_cached_fwd(input, cos, sin, rotate_style=0, ...)

# Two-channel with cache
out_q, out_k = aiter.rope_cached_2c_fwd(input_q, input_k, cos, sin, ...)
```

### With Position Indices

```python
# Position-indexed (for non-contiguous sequences)
output = aiter.rope_cached_positions_fwd(input, cos, sin, positions, ...)

# Two-channel with positions
out_q, out_k = aiter.rope_cached_positions_2c_fwd(
    input_q, input_k, cos, sin, positions, ...
)
```

### With Position Offsets

```python
# Positions + batch-specific offsets
output = aiter.rope_cached_positions_offsets_fwd(
    input, cos, sin, positions, offsets, ...
)

# Two-channel with offsets
out_q, out_k = aiter.rope_cached_positions_offsets_2c_fwd(
    input_q, input_k, cos, sin, positions, offsets, ...
)
```

### THD Format (Packed Sequences)

```python
# Packed variable-length sequences
output = aiter.rope_thd_fwd(input, cu_seqlens, freqs, rotate_style=0, ...)
aiter.rope_thd_fwd_inplace(input, cu_seqlens, freqs, ...)
grads = aiter.rope_thd_bwd(output_grads, cu_seqlens, freqs, ...)
```

### 2D/3D Formats (Vision)

```python
# 2D for vision models
output = aiter.rope_2d_fwd(input, cos_h, sin_h, cos_w, sin_w,
                           img_height, img_width, ...)
aiter.rope_2d_fwd_inplace(input, cos_h, sin_h, cos_w, sin_w,
                          img_height, img_width, ...)

# 3D for vision-language models (Triton only)
from aiter.ops.triton.rope.rope import rope_fwd_3d
output = rope_fwd_3d(x, grid_sizes, freqs, sp_size, sp_rank)
```

---

## 5. Autograd Classes (Training)

For models that need backward pass support:

```python
from aiter.ops.rope import RoPE, RoPECached, RoPETHD, RoPE2D

# Standard RoPE with autograd
output = RoPE.apply(input, freqs, rotate_style, reuse_freqs_front_part, nope_first)

# Cached RoPE with autograd
output = RoPECached.apply(input, cos, sin, rotate_style,
                          reuse_freqs_front_part, nope_first)

# THD format with autograd
output = RoPETHD.apply(input, cu_seqlens, freqs, rotate_style,
                       reuse_freqs_front_part, nope_first)

# 2D format with autograd
output = RoPE2D.apply(input, cos_h, sin_h, cos_w, sin_w,
                      img_height, img_width, rotate_style,
                      reuse_freqs_front_part)
```

---

## 6. Partial RoPE (nope_first)

Some models only apply rotation to a subset of the head dimension. The `nope_first` parameter controls the layout:

| nope_first | Layout | Used By |
|-----------|--------|---------|
| `False` (default) | `[rope_dims, nope_dims]` | Most models |
| `True` | `[nope_dims, rope_dims]` | Some custom architectures |

The rotate dimension is determined by the frequency tensor shape:
- `rotate_dim = freqs.shape[-1] * 2` if `reuse_freqs_front_part=True`
- `rotate_dim = freqs.shape[-1]` otherwise

Non-rotated dimensions are passed through unchanged.

---

## 7. RoPE Scaling Methods

The high-level `RotaryEmbedding` module (`aiter/rotary_embedding.py`) supports multiple scaling strategies for extending context length:

| Scaling Method | Class | Used By |
|---------------|-------|---------|
| **None (standard)** | `RotaryEmbedding` | Default, up to trained context |
| **Linear** | `LinearScalingRotaryEmbedding` | Simple frequency scaling |
| **Dynamic NTK** | `DynamicNTKScalingRotaryEmbedding` | Adaptive base frequency |
| **YaRN** | `YaRNScalingRotaryEmbedding` | Attention scaling + NTK |
| **Phi-3 Long RoPE** | `Phi3LongRoPEScaledRotaryEmbedding` | Phi-3 models |
| **DeepSeek** | `DeepseekScalingRotaryEmbedding` | DeepSeek YaRN variant |
| **LLaMA 3** | `Llama3RotaryEmbedding` | LLaMA 3 low/high freq scaling |
| **Multimodal (MRoPE)** | `MRotaryEmbedding` | Qwen-VL, vision-language |
| **Dual Chunk** | `DualChunkRotaryEmbedding` | Dual Chunk Attention |

---

## 8. Backend Selection (Environment Variables)

```bash
# Use Triton backend (default: 0, use HIP)
export AITER_ROPE_TRITON_BACKEND=1

# Use native PyTorch (slowest, for debugging)
export AITER_ROPE_NATIVE_BACKEND=1

# Enable fused QK norm + RoPE path
export AITER_ROPE_FUSED_QKNORM=1
```

Priority: Native > Triton > HIP (default)

---

## 9. Fused Operations

### Fused QKV Split + QK RoPE

```python
from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope
# Splits QKV tensor and applies RoPE to Q and K in one kernel
```

### Fused BMM + RoPE + KV Cache (MLA)

```python
from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
    fused_fp8_bmm_rope_cat_and_cache_mla,  # FP8 BMM + RoPE + cache
    fused_fp4_bmm_rope_cat_and_cache_mla,  # FP4 BMM + RoPE + cache
)
```

### Fused QK Norm + RoPE + Cache + Quantize

See [Normalization Guide](normalization_guide.md) Section 5 for the mega-fused kernels that combine QK norm, RoPE, KV cache write, and quantization.

---

## 10. Decision Tree

```
Need RoPE?
├── Standard LLM inference?
│   ├── Fixed positions → rope_cached_fwd() or rope_cached_2c_fwd()
│   ├── Variable positions → rope_cached_positions_2c_fwd()
│   └── With batch offsets → rope_cached_positions_offsets_2c_fwd()
├── Variable-length packed sequences?
│   └── rope_thd_fwd()
├── Vision model?
│   ├── 2D spatial → rope_fwd_2d()
│   └── 3D temporal+spatial → rope_fwd_3d()
├── Need backward pass?
│   └── Use autograd classes: RoPE, RoPECached, RoPETHD, RoPE2D
├── Maximum inference fusion?
│   └── fused_qk_norm_rope_cache_quant_shuffle()
└── Multimodal model?
    └── MRotaryEmbedding class
```

---

## 11. Source Files

| Component | Path |
|-----------|------|
| RoPE Python API | `aiter/ops/rope.py` |
| RoPE Triton wrappers | `aiter/ops/triton/rope/rope.py` |
| RoPE Triton kernels | `aiter/ops/triton/_triton_kernels/rope/rope.py` |
| Rotary Embedding module | `aiter/rotary_embedding.py` |
| Fused QKV Split + RoPE | `aiter/ops/triton/rope/fused_qkv_split_qk_rope.py` |
| Fused BMM + RoPE + Cache | `aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py` |
| Fused QK Norm + RoPE | `aiter/ops/fused_qk_norm_rope_cache_quant.py` |
| Fused QK Norm + MRoPE | `aiter/ops/fused_qk_norm_mrope_cache_quant.py` |
| HIP forward kernel | `csrc/pybind/rope_general_fwd_pybind.cu` |
| HIP backward kernel | `csrc/pybind/rope_general_bwd_pybind.cu` |
| HIP position kernel | `csrc/pybind/rope_pos_fwd_pybind.cu` |

---

## 12. Test Files

| Test | Path |
|------|------|
| RoPE main tests | `op_tests/test_rope.py` |
| Triton RoPE tests | `op_tests/triton_tests/rope/test_rope.py` |
| Fused QKV Split + RoPE | `op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py` |
| Fused QK Norm + RoPE + Cache | `op_tests/test_fused_qk_norm_rope_cache_quant.py` |
| Fused QK Norm + MRoPE | `op_tests/test_fused_qk_norm_mrope_cache_quant.py` |
| Fused BMM + RoPE + Cache | `op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py` |
| MLA Decode + RoPE | `op_tests/triton_tests/attention/test_mla_decode_rope.py` |
| RoPE benchmarks | `op_tests/op_benchmarks/triton/bench_rope.py` |
