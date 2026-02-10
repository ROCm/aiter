---
name: aiter-triton-elementwise-kernels
description: Write Triton elementwise kernels, tests, and benchmarks for the aiter project. Use when creating or modifying activation kernels (SiLU, GELU, ReLU), RoPE (Rotary Position Embedding) kernels, fused operation kernels (KV cache, BMM+RoPE, QK concat), softmax, or top-k kernels.
---

# Triton Elementwise & Fusion Kernels in Aiter

## Project Layout

| Component | Path |
|-----------|------|
| Activations | `aiter/ops/triton/_triton_kernels/activation.py` |
| RoPE | `aiter/ops/triton/_triton_kernels/rope/` |
| Fusions | `aiter/ops/triton/_triton_kernels/fusions/` |
| Softmax | `aiter/ops/triton/_triton_kernels/softmax.py` |
| Top-K | `aiter/ops/triton/_triton_kernels/topk.py` |
| Tests | `op_tests/triton_tests/rope/`, `op_tests/triton_tests/fusions/` |
| Benchmarks | `op_tests/op_benchmarks/triton/bench_rope*.py` |

## Activation Kernels

### Activation Primitives

```python
@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _gelu(x):
    return 0.5 * x * (1 + tl.math.erf(x / 1.4142135623730951))

@triton.jit
def _gelu_tanh(x):
    return 0.5 * x * (1 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

@triton.jit
def _relu(x):
    return tl.maximum(x, 0.0)

@triton.jit
def _apply_activation_from_str(x, activation: tl.constexpr):
    """String-based activation dispatch."""
    if activation == "silu":
        return _silu(x)
    elif activation == "gelu":
        return _gelu(x)
    elif activation == "relu":
        return _relu(x)
```

### Fused Activation + Quantization Pattern

Common in MoE feed-forward layers:

```python
@triton.heuristics({"EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0})
@triton.jit
def _act_mul_and_dynamic_fp8_quant_kernel(
    Gate_ptr, Up_ptr, Out_ptr, Scale_ptr,
    M, N,
    stride_gm, stride_gn, stride_um, stride_un, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N if not EVEN_N else None
        gate = tl.load(Gate_ptr + pid_m * stride_gm + cols, mask=mask).to(tl.float32)
        up = tl.load(Up_ptr + pid_m * stride_um + cols, mask=mask).to(tl.float32)
        out = _apply_activation_from_str(gate, ACTIVATION) * up
        # ... quantize and store ...
```

## RoPE Kernels

### RoPE Pattern (SBHD layout)

```python
@triton.jit
def _get_neox_rotated_x(x, BLOCK_D: tl.constexpr):
    """NeoX-style rotation: split dim in half, negate first half."""
    half_d = BLOCK_D // 2
    x0 = tl.load(x_ptr + tl.arange(0, half_d))
    x1 = tl.load(x_ptr + half_d + tl.arange(0, half_d))
    return tl.join(-x1, x0)  # [-x1, x0]

@triton.jit
def _rope_kernel_sbhd_fwd(
    X_ptr, Freqs_ptr, Y_ptr,
    stride_x_s, stride_x_b, stride_x_h, stride_x_d,
    stride_y_s, stride_y_b, stride_y_h, stride_y_d,
    stride_f_s, stride_f_d,
    S, B, H, D,
    BLOCK_D: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
):
    pid_s = tl.program_id(0)  # sequence position
    pid_bh = tl.program_id(1)  # batch * head
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_d = tl.arange(0, BLOCK_D)
    
    # Load input and frequency
    x = tl.load(X_ptr + pid_s * stride_x_s + pid_b * stride_x_b 
                + pid_h * stride_x_h + offs_d * stride_x_d)
    freqs = tl.load(Freqs_ptr + pid_s * stride_f_s + offs_d * stride_f_d)
    
    cos_val = tl.cos(freqs)
    sin_val = tl.sin(freqs)
    
    if IS_NEOX:
        x_rot = _get_neox_rotated_x(x, BLOCK_D)
    else:  # GPT-J style
        x_rot = _get_gptj_rotated_x(x, BLOCK_D)
    
    y = x * cos_val + x_rot * sin_val
    
    # Handle NOPE dimensions (dims without RoPE)
    if HAVE_NOPE:
        # Pass through NOPE dims unchanged
        pass
    
    tl.store(Y_ptr + pid_s * stride_y_s + pid_b * stride_y_b 
             + pid_h * stride_y_h + offs_d * stride_y_d, y)
```

**Grid:** `(S, B * H)` where S=sequence, B=batch, H=heads

### RoPE Layouts

| Layout | Shape | Use Case |
|--------|-------|----------|
| SBHD | (seq, batch, heads, dim) | Standard training |
| THD | (total_tokens, heads, dim) | Variable-length / packed |
| 2D | (batch, seq, dim) | Simple 2D with positions |

### Fused QKV Split + RoPE Pattern

Combines QKV tensor split with RoPE application:

```python
@triton.jit
def _fused_qkv_split_qk_rope_kernel(
    QKV_ptr,  # Fused QKV tensor (B, S, 3*H*D) or (B, S, (H_Q+2*H_K)*D)
    Q_ptr, K_ptr, V_ptr,  # Separate output tensors
    Freqs_ptr,
    # ... dimensions and strides ...
):
    # Split QKV → Q, K, V
    # Apply RoPE to Q and K only
    # V passes through unchanged
```

## Fusion Kernels

### Fused KV Cache Update

Combines RoPE + KV cache write in one kernel:

```python
@triton.jit
def _fused_kv_cache_kernel(
    K_new_ptr, V_new_ptr,  # New KV from current step
    K_cache_ptr, V_cache_ptr,  # KV cache to update
    cache_seqlens_ptr,  # Current sequence length per batch
    # Strides for new KV and cache
    # Block tables for paged cache (optional)
    BLOCK_D: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # head
    
    seq_pos = tl.load(cache_seqlens_ptr + pid_b)
    
    # Load new K, V
    k = tl.load(K_new_ptr + ...)
    v = tl.load(V_new_ptr + ...)
    
    # Optionally apply RoPE to K
    if HAS_ROPE:
        freqs = tl.load(Freqs_ptr + seq_pos * stride_f_s + ...)
        k = k * tl.cos(freqs) + k_rot * tl.sin(freqs)
    
    # Write to cache at seq_pos
    tl.store(K_cache_ptr + pid_b * stride_cb + pid_h * stride_ch 
             + seq_pos * stride_cs + ..., k)
    tl.store(V_cache_ptr + ..., v)
```

### Fused BMM + RoPE + KV Cache

```python
# Located in fusions/fused_bmm_rope_kv_cache.py
# Combines: batch matmul + RoPE + KV cache update
# Useful for MLA decode where QK projection + RoPE + cache happen together
```

## Writing Tests

### RoPE Test Pattern

```python
def ref_rope_sbhd_fwd(x, freqs, is_neox=True):
    """Reference RoPE in PyTorch."""
    cos = freqs.cos()
    sin = freqs.sin()
    if is_neox:
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
    else:  # GPT-J
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rotated * sin

@pytest.mark.parametrize("S,B,H,D,is_neox", [
    (128, 2, 32, 64, True),
    (128, 2, 32, 128, False),
    (1, 1, 1, 64, True),
])
def test_rope(S, B, H, D, is_neox):
    x = torch.randn(S, B, H, D, dtype=torch.bfloat16, device="cuda")
    freqs = torch.randn(S, 1, 1, D, device="cuda")
    ref = ref_rope_sbhd_fwd(x, freqs, is_neox)
    out = rope_fwd(x, freqs, is_neox=is_neox)
    torch.testing.assert_close(ref, out, atol=1e-1, rtol=1e-1)
```

### Activation Test Pattern

```python
def test_silu():
    x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
    ref = torch.nn.functional.silu(x)
    out = triton_silu(x)
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
```

## Prerequisites

Before writing elementwise/fusion kernels, read these foundational skills:
- [Triton Language Guide](../triton-language-guide/SKILL.md) - Triton API and AMD optimizations
- [GPU Kernel Algorithms](../gpu-kernel-algorithms/SKILL.md) - RoPE algorithm, roofline model

## Key Design Notes

- **Elementwise kernels are bandwidth-bound:** Optimize for memory throughput
- **Fusions eliminate intermediate tensors:** Major benefit for decode latency
- **RoPE frequency caching:** `cos_cache`/`sin_cache` precomputed in `RotaryEmbedding.__init__`
- **Backend selection via env vars:** `AITER_ROPE_TRITON_BACKEND=1` enables Triton path
- **NOPE dimensions:** Some architectures have dimensions that skip RoPE (MLA)
