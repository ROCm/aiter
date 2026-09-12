import torch
import triton

from aiter.ops.triton._triton_kernels.rope.fused_qkv_split_qk_norm_rope_cache import (
    _fused_qkv_split_qk_norm_rope_cache_kernel,
)


def infer_rope_cache_triton_block_t(T: int, device: torch.device) -> int:
    """Pick Triton token tile ``BLOCK_T`` for :func:`fused_qkv_split_qk_norm_rope_cache`.

    Same heuristic as vLLM ``input_quant_fp8.calc_rows_per_block``: scale tile size with
    sequence length and SM/CU count so the launch grid stays in a sensible range — use
    ``ceil(T / (2 * sm_count))`` rounded up to the next power of two, then cap (the
    vLLM quant path caps at 4 rows; this kernel keeps a larger upper bound due to
    per-token work and ``tl.arange(0, BLOCK_T)`` register pressure).
    """
    if device.type != "cuda":
        raise ValueError(
            "fused_qkv_split_qk_norm_rope_cache expects a CUDA/HIP device "
            f"(got {device!r})."
        )
    device_id = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    sm_count = max(
        int(torch.cuda.get_device_properties(device_id).multi_processor_count),
        1,
    )
    block_t = triton.next_power_of_2(triton.cdiv(T, 2 * sm_count))
    return max(1, min(int(block_t), 32))


def fused_qkv_split_qk_norm_rope_cache(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,  # RMS norm weight for Q
    k_weight: torch.Tensor,  # RMS norm weight for K
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
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
    kv_cache_layout: str = "HND",
):
    """Split packed ``qkv``, RMSNorm Q and K, apply RoPE, write K/V into paged caches.

    Shapes follow ``qh`` / ``kvh`` / ``head_dim``. Paged KV layout is selected by
    ``kv_cache_layout``:

    - ``"HND"``: ``[num_blocks, num_kv_heads, block_size, head_dim]`` (default).
    - ``"NHD"``: ``[num_blocks, block_size, num_kv_heads, head_dim]``.

    For RoPE, ``rotary_dim_half`` refers
    to the half-width of the rotated subspace. The required ``cos`` / ``sin`` last-dim
    shape depends on ``reuse_freqs_front_part``: when True, ``cos.shape[-1]`` and
    ``sin.shape[-1]`` are the half-width (that is, ``rotary_dim_half``); when False,
    they are the full rotary width (that is, ``2 * rotary_dim_half``). The
    ``reuse_freqs_front_part`` flag must match how ``cos``/``sin`` were built for
    partial rotation. When ``attn_output_gate`` is True, ``gated_qkv_layout`` is
    ``"interleaved"`` (Q then gate per head in the flat Q+gate region) or ``"blocked"``
    (all Q then all gate). Block size and strides follow ``kv_cache_layout``;
    ``slot_mapping`` maps each token row to a physical slot index.

    Args:
        qkv: ``[T, packed_dim]`` flat tensor (Q [+ gate], K, V).
        q_weight, k_weight: RMSNorm gamma ``(head_dim,)``.
        cos, sin: RoPE tables. If ``reuse_freqs_front_part`` is True, the last dim is
            the rotary half-width; if False, the last dim is the full rotary width.
        positions: Token positions into the RoPE table.
        key_cache, value_cache: Same rank-4 shape; layout per ``kv_cache_layout``.
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
        kv_cache_layout: ``"HND"`` or ``"NHD"`` (case-insensitive).
    """
    T = qkv.shape[0]
    q_size = qh * head_dim
    kv_size = kvh * head_dim

    layout = kv_cache_layout.upper()
    if layout not in ("HND", "NHD"):
        raise ValueError(
            'kv_cache_layout must be "HND" or "NHD" ' f"(got {kv_cache_layout!r})."
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            "key_cache and value_cache must have the same shape "
            f"(got {tuple(key_cache.shape)} vs {tuple(value_cache.shape)})."
        )
    num_blocks = key_cache.shape[0]
    if layout == "HND":
        if key_cache.shape[1] != kvh or key_cache.shape[3] != head_dim:
            raise ValueError(
                "HND key_cache expected "
                f"[num_blocks, {kvh}, block_size, {head_dim}], "
                f"got {tuple(key_cache.shape)}."
            )
        block_size = key_cache.shape[2]
        key_cache_stride_t = key_cache.stride(0)
        key_cache_stride_h = key_cache.stride(1)
        key_cache_stride_b = key_cache.stride(2)
        key_cache_stride_d = key_cache.stride(3)
        value_cache_stride_t = value_cache.stride(0)
        value_cache_stride_h = value_cache.stride(1)
        value_cache_stride_b = value_cache.stride(2)
        value_cache_stride_d = value_cache.stride(3)
    else:
        if key_cache.shape[2] != kvh or key_cache.shape[3] != head_dim:
            raise ValueError(
                "NHD key_cache expected "
                f"[num_blocks, block_size, {kvh}, {head_dim}], "
                f"got {tuple(key_cache.shape)}."
            )
        block_size = key_cache.shape[1]
        key_cache_stride_t = key_cache.stride(0)
        key_cache_stride_b = key_cache.stride(1)
        key_cache_stride_h = key_cache.stride(2)
        key_cache_stride_d = key_cache.stride(3)
        value_cache_stride_t = value_cache.stride(0)
        value_cache_stride_b = value_cache.stride(1)
        value_cache_stride_h = value_cache.stride(2)
        value_cache_stride_d = value_cache.stride(3)

    total_num_kv_cache_tokens = num_blocks * block_size

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

    BLOCK_T = infer_rope_cache_triton_block_t(T, qkv.device)
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
        key_cache_stride_t=key_cache_stride_t,
        key_cache_stride_h=key_cache_stride_h,
        key_cache_stride_d=key_cache_stride_d,
        key_cache_stride_b=key_cache_stride_b,
        value_cache_stride_t=value_cache_stride_t,
        value_cache_stride_h=value_cache_stride_h,
        value_cache_stride_d=value_cache_stride_d,
        value_cache_stride_b=value_cache_stride_b,
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
        total_num_kv_cache_tokens=total_num_kv_cache_tokens,
        num_warps=num_warps,
    )

    if attn_output_gate:
        return q, gate, k, v
    else:
        return q, k, v
