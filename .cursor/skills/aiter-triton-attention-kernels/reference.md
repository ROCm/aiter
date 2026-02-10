# Triton Attention Kernel Reference

## Paged Attention Decode Pattern

PA decode processes single-query tokens against paged KV cache:

```python
@triton.jit
def _pa_decode_kernel(
    Q, K_cache, V_cache, Out,
    block_tables,      # (batch, max_num_blocks) - maps to physical KV blocks
    context_lens,      # (batch,) - actual sequence length per request
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,  # KV cache strides
    stride_ob, stride_oh, stride_od,
    num_kv_heads, block_size, head_dim,
    sm_scale,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # query head
    
    # Load query for this batch/head
    q = tl.load(Q + pid_b * stride_qb + pid_h * stride_qh + tl.arange(0, BLOCK_DMODEL))
    
    # GQA head mapping
    kv_head = pid_h // (num_q_heads // num_kv_heads)
    
    # Iterate over KV cache blocks
    context_len = tl.load(context_lens + pid_b)
    num_blocks = tl.cdiv(context_len, block_size)
    
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    for block_idx in range(num_blocks):
        physical_block = tl.load(block_tables + pid_b * max_num_blocks + block_idx)
        # Load K, V from physical block
        k = tl.load(K_cache + physical_block * stride_kb + kv_head * stride_kh + ...)
        v = tl.load(V_cache + physical_block * stride_kb + kv_head * stride_kh + ...)
        # Online softmax + accumulation
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        # ... softmax, accumulate ...
    
    # Normalize and store
    out = acc / l_i
    tl.store(Out + pid_b * stride_ob + pid_h * stride_oh + tl.arange(0, BLOCK_DMODEL), out)
```

Grid: `(batch_size, num_query_heads)`

## Paged Attention Prefill Pattern

PA prefill processes multiple query tokens:

```python
# Grid: (triton.cdiv(total_tokens, BLOCK_M), num_heads)
# Uses cu_seqlens for variable-length batching
# Each program handles BLOCK_M query tokens against full KV
```

## Flash Attention AMD Backend

Located in `aiter/ops/triton/_triton_kernels/attention/flash_attn_triton_amd/`:

- `fwd_prefill.py` - Forward prefill kernel
- `fwd_decode.py` - Forward decode kernel (splitKV)
- `bwd_prefill.py` - Backward kernel
- `fwd_ref.py` - Reference implementation
- `interface_fa.py` - User-facing API

Key AMD optimizations:
- `waves_per_eu` control for occupancy
- `matrix_instr_nonkdim` for MFMA instruction selection
- `pre_load_v` flag to overlap V loads with QK computation

## Backward Pass Pattern

MHA backward computes dQ, dK, dV:

```python
@triton.jit
def _attn_bwd(
    Q, K, V, dO, dQ, dK, dV, LSE,
    # ... strides and dimensions ...
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Phase 1: Recompute attention weights from Q, K, LSE
    # Phase 2: dV = P^T @ dO
    # Phase 3: dP = dO @ V^T
    # Phase 4: dQ = dP * softmax_grad @ K
    # Phase 5: dK = dP * softmax_grad @ Q^T
```

Two common implementations:
1. **Fused one-kernel**: Single kernel for all gradients
2. **Split dQ/dKdV**: Separate kernels for better parallelism

## Variable-Length (Varlen) Attention

Uses cumulative sequence lengths (`cu_seqlens`) for packed batching:

```python
def attention_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    # q shape: (total_q_tokens, nheads, head_dim) - packed
    # cu_seqlens_q: (batch+1,) - cumulative lengths [0, len1, len1+len2, ...]
    total_q = q.shape[0]
    grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch * nheads)
    # Inside kernel: use cu_seqlens to compute per-request start/end
```

## MLA (Multi-head Latent Attention)

MLA uses a compressed KV representation with latent vectors:

```python
# KV buffer shape: (batch, max_pages, page_size, kv_lora_rank + qk_rope_head_dim)
# Q gets projected and split into: q_nope (latent) + q_rope (positional)
# Attention: softmax(q_nope @ kv_nope^T + q_rope @ kv_rope^T) @ v_nope
```

Key differences from standard MHA:
- Single KV head (nhead_kv=1 typically)
- Compressed KV via LoRA-style projection
- RoPE applied to a subset of dimensions

## Test Utilities

### Shared Reference
```python
from aiter.test_mha_common import attention_ref, generate_qkv
from aiter.test_mha_common import generate_random_padding_mask
```

### FP8 Custom Assert
```python
def fp8_assert_close(tensor_a, tensor_b, atol=0.3, rtol=0.25, max_diff_percentage=0.5):
    """Allow up to max_diff_percentage of elements to exceed tolerance."""
    diff = (tensor_a - tensor_b).abs()
    threshold = atol + rtol * tensor_b.abs()
    failures = (diff > threshold).float().mean() * 100
    assert failures <= max_diff_percentage
```

## Benchmark Patterns

### PA Decode Benchmark
```yaml
pa_decode:
  input_columns: [model, BS, HQ, HK, SEQ_LEN, HEAD_DIM]
  output_columns: [Time_(ms), TFLOPS, Bandwidth_(GB/s)]
```

### PA Prefill Benchmark  
```yaml
pa_prefill:
  input_columns: [model, BS, HQ, HK, MAX_SEQ_LEN, HEAD_DIM]
  output_columns: [Time_(ms), TFLOPS, Bandwidth_(GB/s)]
```

FLOPS computation for attention:
```python
flops = 2 * batch * nheads * seqlen_q * seqlen_k * head_dim  # QK^T
flops += 2 * batch * nheads * seqlen_q * seqlen_k * head_dim  # PV
```
