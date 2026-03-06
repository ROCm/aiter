import torch
import triton
from aiter.ops.triton._triton_kernels.rope.fused_qkv_split_qk_norm_rope_cache import (
    _fused_qkv_split_qk_norm_rope_cache_kernel,
)

def fused_qkv_split_qk_norm_rope_cache(
    qkv: torch.Tensor,
    q_weight: torch.Tensor, # RMS norm weight for Q
    k_weight: torch.Tensor, # RMS norm weight for K
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    key_cache: torch.Tensor,    # Paged KV Cache [num_blocks, num_heads, block_size, head_dim]
    value_cache: torch.Tensor,  # Paged KV Cache [num_blocks, num_heads, block_size, head_dim]
    slot_mapping: torch.Tensor, # Mapping from token index to physical slot [T]
    qh: int,
    kvh: int,
    head_dim: int,
    is_neox: bool = True,
    offsets: torch.Tensor = None,
    reuse_freqs_front_part: bool = True,
    attn_output_gate: bool = False,
    eps: float = 1e-5
):
    T = qkv.shape[0]
    q_size = qh * head_dim
    kv_size = kvh * head_dim
    
    # Get Paged Attention block size from cache shape (usually 16 or 32)
    # Cache shape: [num_blocks, num_heads, block_size, head_dim]
    block_size = key_cache.shape[2]

    assert qh >= kvh and qh % kvh == 0, "qh must be mutiple of kvh"
    q = torch.empty((T, qh, head_dim), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((T, kvh, head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((T, kvh, head_dim), dtype=qkv.dtype, device=qkv.device)
    
    if attn_output_gate:
        gate = torch.empty((T, qh, head_dim), dtype=qkv.dtype, device=qkv.device)
    else:
        gate = None

    assert qkv.shape[-1] == q_size + 2 * kv_size, "Shape error"
    assert head_dim == triton.next_power_of_2(
        head_dim
    ), "head_dim should be power of 2"

    # Logic for dimension splitting
    BLOCK_D = head_dim
    BLOCK_D_HALF = head_dim // 2

    BLOCK_T = 32
    num_warps = 4
    grid = (triton.cdiv(T, BLOCK_T), qh)

    _fused_qkv_split_qk_norm_rope_cache_kernel[grid](
        qkv_ptr=qkv,
        q_weight_ptr=q_weight,
        k_weight_ptr=k_weight,
        cos_ptr=cos,
        sin_ptr=sin,
        pos_ptr=positions,
        off_ptr=offsets,
        q_ptr=q,
        gate_ptr=gate,
        k_ptr=k,
        v_ptr=v,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        T=T,
        eps=eps,
        stride_qkv_t=qkv.stride(0),
        stride_qkv_d=qkv.stride(1),
        stride_cos_t=cos.stride(0),
        stride_cos_d=cos.stride(1),
        stride_pos_t=positions.stride(0),
        stride_q_t=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_kv_t=k.stride(0),
        stride_kv_h=k.stride(1),
        stride_kv_d=k.stride(2),
        key_cache_stride_t=key_cache.stride(0),
        key_cache_stride_h=key_cache.stride(1),
        key_cache_stride_d=key_cache.stride(3), # head_dim stride
        key_cache_stride_b=key_cache.stride(2), # block_size stride
        value_cache_stride_t=value_cache.stride(0),
        value_cache_stride_h=value_cache.stride(1),
        value_cache_stride_d=value_cache.stride(3),
        value_cache_stride_b=value_cache.stride(2),
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        HAVE_POS=(positions is not None),
        HAVE_OFFS=(offsets is not None),
        ENABLE_GATED_Q=attn_output_gate,
        QH=qh,
        KVH=kvh,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    if attn_output_gate:
        return q, gate, k, v 
    else:
        return q, k, v