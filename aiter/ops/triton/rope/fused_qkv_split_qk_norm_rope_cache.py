import torch
import triton

from aiter.ops.triton._triton_kernels.rope.fused_qkv_split_qk_norm_rope_cache import (
    _fused_qkv_split_qk_norm_rope_cache_kernel,
)


def fused_qkv_split_qk_norm_rope_cache(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,  # RMS norm weight for Q
    k_weight: torch.Tensor,  # RMS norm weight for K
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    key_cache: torch.Tensor,  # Paged KV Cache [num_blocks, num_heads, block_size, head_dim]
    value_cache: torch.Tensor,  # Paged KV Cache [num_blocks, num_heads, block_size, head_dim]
    slot_mapping: torch.Tensor,  # Mapping from token index to physical slot [T]
    qh: int,
    kvh: int,
    head_dim: int,
    is_neox: bool = True,
    offsets: torch.Tensor = None,
    reuse_freqs_front_part: bool = True,
    attn_output_gate: bool = False,
    k_scale: torch.Tensor = None,
    v_scale: torch.Tensor = None,
    eps: float = 1e-5,
    gated_qkv_layout: str = "interleaved",
):
    """Split packed ``qkv``, RMSNorm Q and K, apply RoPE, write K/V into paged caches.

    Shapes follow ``qh`` / ``kvh`` / ``head_dim``. For RoPE, ``rotary_dim_half`` refers
    to the half-width of the rotated subspace. The required ``cos`` / ``sin`` last-dim
    shape depends on ``reuse_freqs_front_part``: when True, ``cos.shape[-1]`` and
    ``sin.shape[-1]`` are the half-width (that is, ``rotary_dim_half``); when False,
    they are the full rotary width (that is, ``2 * rotary_dim_half``). The
    ``reuse_freqs_front_part`` flag must match how ``cos``/``sin`` were built for
    partial rotation. When ``attn_output_gate`` is True, ``gated_qkv_layout`` is
    ``"interleaved"`` (Q then gate per head in the flat Q+gate region) or ``"blocked"``
    (all Q then all gate). Paged layout is inferred from ``key_cache`` /
    ``value_cache`` (block size is ``shape[2]``); ``slot_mapping`` maps each token row
    to a physical slot index.

    Args:
        qkv: ``[T, packed_dim]`` flat tensor (Q [+ gate], K, V).
        q_weight, k_weight: RMSNorm gamma ``(head_dim,)``.
        cos, sin: RoPE tables. If ``reuse_freqs_front_part`` is True, the last dim is
            the rotary half-width; if False, the last dim is the full rotary width.
        positions: Token positions into the RoPE table.
        key_cache, value_cache: ``[num_blocks, heads, block_size, head_dim]``.
        slot_mapping: ``[T]`` int32 (or index dtype), slot per token for cache write.
        qh: Total query heads; ``kvh`` key/value heads; ``head_dim`` per-head size.
        is_neox: NeoX vs GPT-J rotation style.
        offsets: Optional position offsets (same semantics as other rope ops).
        reuse_freqs_front_part: Frequency-table layout selector for partial RoPE.
            True means front-part reuse tables with ``cos/sin.shape[-1] ==
            rotary_dim_half``; False means non-reuse tables with
            ``cos/sin.shape[-1] == 2 * rotary_dim_half``.
        attn_output_gate: Whether Q+gate is packed in ``qkv``; returns ``(q, gate, k, v)``.
        k_scale, v_scale: Optional per-call scalars applied before cache write.
        eps: RMSNorm epsilon.
        rotary_dim_half: Optional half-width of the rotated subspace. When set, it
            corresponds to ``cos.shape[-1]`` only in reuse mode; in non-reuse mode the
            table last dim is the full rotary width, ``2 * rotary_dim_half``.
        gated_qkv_layout: ``"interleaved"`` or ``"blocked"`` when ``attn_output_gate``.
    """
    T = qkv.shape[0]
    q_size = qh * head_dim
    kv_size = kvh * head_dim

    # Get Paged Attention block size from cache shape (usually 16 or 32)
    # Cache shape: [num_blocks, num_heads, block_size, head_dim]
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[2]

    # Validate that all valid (non-negative) slot indices are within the
    # allocated cache; slots < 0 are padding tokens and are intentionally skipped.
    valid_slots = slot_mapping[slot_mapping >= 0]
    if valid_slots.numel() > 0:
        max_slot = int(valid_slots.max().item())
        assert max_slot < num_blocks * block_size, (
            f"slot_mapping contains slot {max_slot} which is out of bounds "
            f"for cache with {num_blocks} blocks of size {block_size} "
            f"(max valid slot: {num_blocks * block_size - 1})"
        )

    assert qh >= kvh and qh % kvh == 0, "qh must be multiple of kvh"
    q = torch.empty((T, qh, head_dim), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((T, kvh, head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((T, kvh, head_dim), dtype=qkv.dtype, device=qkv.device)

    if attn_output_gate:
        gate = torch.empty((T, qh, head_dim), dtype=qkv.dtype, device=qkv.device)
    else:
        gate = None

    if attn_output_gate:
        assert qkv.shape[-1] == 2 * q_size + 2 * kv_size, "Shape error"
        assert gated_qkv_layout in (
            "interleaved",
            "blocked",
        ), 'gated_qkv_layout must be "interleaved" or "blocked"'
    else:
        assert qkv.shape[-1] == q_size + 2 * kv_size, "Shape error"
    assert head_dim == triton.next_power_of_2(head_dim), "head_dim should be power of 2"

    assert cos.shape[-1] == sin.shape[-1], "cos and sin must match in last dim"
    # the effective rotary dim, half or full of the rotary dim depending on reuse_freqs_front_part
    ROTARY_DIM_EFFECTIVE = cos.shape[-1]

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
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        stride_qkv_t=qkv.stride(0),
        stride_qkv_d=qkv.stride(1),
        stride_cos_t=cos.stride(0),
        stride_cos_d=cos.stride(-1),
        stride_pos_t=positions.stride(0),
        stride_q_t=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_kv_t=k.stride(0),
        stride_kv_h=k.stride(1),
        stride_kv_d=k.stride(2),
        key_cache_stride_t=key_cache.stride(0),
        key_cache_stride_h=key_cache.stride(1),
        key_cache_stride_d=key_cache.stride(3),  # head_dim stride
        key_cache_stride_b=key_cache.stride(2),  # block_size stride
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
        ROTARY_DIM_EFFECTIVE=ROTARY_DIM_EFFECTIVE,
        BLOCKED_GATED_LAYOUT=(attn_output_gate and gated_qkv_layout == "blocked"),
        HAVE_K_SCALE=k_scale is not None,
        HAVE_V_SCALE=v_scale is not None,
        NUM_BLOCKS=num_blocks,
        num_warps=num_warps,
    )

    if attn_output_gate:
        return q, gate, k, v
    else:
        return q, k, v
