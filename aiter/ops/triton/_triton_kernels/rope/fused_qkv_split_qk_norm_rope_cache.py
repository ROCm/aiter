import triton
import triton.language as tl
from aiter.ops.triton.rope.rope import _get_gptj_rotated_x, _get_neox_rotated_x


# GAMMA rms norm
@triton.jit
def _rms_norm(
    tensor,  # Pre-loaded block: (BLOCK_M, BLOCK_D)
    weight,  # Pre-loaded block: (BLOCK_D,)
    BLOCK_D,  # Constant
    eps,  # Scalar
):
    tensor_f32 = tensor.to(tl.float32)
    tensor_sq = tensor_f32 * tensor_f32
    variance = tl.sum(tensor_sq, axis=1) / BLOCK_D

    inv_rms = tl.rsqrt(variance + eps)[:, None]
    tensor_normed = tensor_f32 * inv_rms

    w_f32 = weight.to(tl.float32)
    tensor_final = tensor_normed * (1.0 + w_f32[None, :])

    return tensor_final.to(tensor.dtype)


@triton.jit
def _rms_norm_returning_inv(
    tensor,
    weight,
    BLOCK_D,
    eps,
):
    tensor_f32 = tensor.to(tl.float32)
    tensor_sq = tensor_f32 * tensor_f32
    variance = tl.sum(tensor_sq, axis=1) / BLOCK_D

    inv_rms = tl.rsqrt(variance + eps)[:, None]
    tensor_normed = tensor_f32 * inv_rms

    w_f32 = weight.to(tl.float32)
    tensor_final = tensor_normed * (1.0 + w_f32[None, :])

    return tensor_final.to(tensor.dtype), inv_rms


@triton.jit
def _partial_neox_rotated(
    inv_rms,
    weight_ptr,
    qkv_ptr,
    t_offs,
    d_offs,
    stride_qkv_t,
    stride_qkv_d,
    head_d_offset,
    x_mask,
    ROTARY_DIM_HALF: tl.constexpr,
):
    """Compute the neox-rotated+normed vector for partial rotation.

    Re-loads from qkv at the swapped dim positions and applies the same
    inv_rms with the weight at the swapped position, then the neox sign.
    For dims beyond rotary_dim this produces 0 (rot_sign = 0).
    """
    ROTARY_DIM: tl.constexpr = ROTARY_DIM_HALF * 2
    swap_d = tl.where(
        d_offs < ROTARY_DIM_HALF,
        d_offs + ROTARY_DIM_HALF,
        tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs),
    )
    rot_sign = tl.where(
        d_offs < ROTARY_DIM_HALF,
        -1.0,
        tl.where(d_offs < ROTARY_DIM, 1.0, 0.0),
    )

    swap_offs = (
        t_offs[:, None] * stride_qkv_t
        + (head_d_offset + swap_d)[None, :] * stride_qkv_d
    )
    swapped = tl.load(qkv_ptr + swap_offs, mask=x_mask)

    swap_w = tl.load(weight_ptr + swap_d).to(tl.float32)
    rotated = swapped.to(tl.float32) * inv_rms * (1.0 + swap_w[None, :])
    rotated = rotated * rot_sign[None, :]
    return rotated


@triton.jit
def _partial_gptj_rotated(
    inv_rms,
    weight_ptr,
    qkv_ptr,
    t_offs,
    d_offs,
    stride_qkv_t,
    stride_qkv_d,
    head_d_offset,
    x_mask,
    ROTARY_SPAN: tl.constexpr,
):
    """Partial GPT-J RoPE partner: pairs (0,1), (2,3), ... within ``[0, ROTARY_SPAN)``; identity beyond."""
    swap_d = tl.where(
        d_offs < ROTARY_SPAN,
        tl.where((d_offs % 2) == 0, d_offs + 1, d_offs - 1),
        d_offs,
    )
    rot_sign = tl.where(
        d_offs < ROTARY_SPAN,
        tl.where((d_offs % 2) == 0, -1.0, 1.0),
        0.0,
    )

    swap_offs = (
        t_offs[:, None] * stride_qkv_t
        + (head_d_offset + swap_d)[None, :] * stride_qkv_d
    )
    swapped = tl.load(qkv_ptr + swap_offs, mask=x_mask)

    swap_w = tl.load(weight_ptr + swap_d).to(tl.float32)
    rotated = swapped.to(tl.float32) * inv_rms * (1.0 + swap_w[None, :])
    rotated = rotated * rot_sign[None, :]
    return rotated


@triton.jit
def _fused_qkv_split_qk_norm_rope_cache_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    q_ptr,
    gate_ptr,
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    T,
    eps,
    stride_qkv_t,
    stride_qkv_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_h,
    stride_kv_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    k_scale_ptr,
    v_scale_ptr,
    total_num_kv_cache_tokens: tl.int64,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    ENABLE_GATED_Q: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # PagedAttention block size
    ROTARY_DIM_EFFECTIVE: tl.constexpr = 0,
    BLOCKED_GATED_LAYOUT: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
):
    tl.assume(stride_qkv_t > 0)
    tl.assume(stride_qkv_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_q_t > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_t > 0)
    tl.assume(stride_kv_h > 0)
    tl.assume(stride_kv_d > 0)

    # # Half-width of cos/sin table for Neox folding; 0 means full-head (legacy).
    # EFFECTIVE_RDH: tl.constexpr = (
    #     ROTARY_DIM_HALF if ROTARY_DIM_HALF > 0 else BLOCK_D_HALF
    # )
    # # ref_rope_sbhd_fwd rotate_dim = cos.shape[-1] * (2 if reuse else 1)
    ROTARY_SPAN: tl.constexpr = (
        ROTARY_DIM_EFFECTIVE * 2 if REUSE_FREQS_FRONT_PART else ROTARY_DIM_EFFECTIVE
    )
    PARTIAL_ROTATION: tl.constexpr = ROTARY_SPAN < BLOCK_D
    # NeoX swap operates on first ROTARY_SPAN dims; pair-half width is ROTARY_SPAN // 2.
    EFFECTIVE_RDH: tl.constexpr = ROTARY_SPAN // 2

    pid_t = tl.program_id(0)
    hq = tl.program_id(1)

    tl.assume(pid_t >= 0)
    tl.assume(hq >= 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < EFFECTIVE_RDH),
                d_cos_offs,
                d_cos_offs - EFFECTIVE_RDH,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < EFFECTIVE_RDH
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < EFFECTIVE_RDH
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < EFFECTIVE_RDH * 2

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    if PARTIAL_ROTATION:
        cos = tl.load(cos_ptr + cos_offs, mask=cos_mask, other=1.0)
        sin = tl.load(sin_ptr + cos_offs, mask=cos_mask, other=0.0)
    else:
        cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
        sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        qk_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        qk_rotated_mask = (d_offs % 2 == 0)[None, :]

    if ENABLE_GATED_Q:
        if BLOCKED_GATED_LAYOUT:
            q_lane_base = hq * BLOCK_D
            gate_lane_base = QH * BLOCK_D + hq * BLOCK_D
        else:
            q_lane_base = hq * (2 * BLOCK_D)
            gate_lane_base = hq * (2 * BLOCK_D) + BLOCK_D
        Q_HEAD_STRIDE: tl.constexpr = 2 * BLOCK_D
    else:
        Q_HEAD_STRIDE: tl.constexpr = BLOCK_D
        q_lane_base = hq * BLOCK_D
        gate_lane_base = 0
    q_in_offs = (
        t_offs[:, None] * stride_qkv_t + (q_lane_base + d_offs)[None, :] * stride_qkv_d
    )
    q = tl.load(qkv_ptr + q_in_offs, mask=x_mask)

    q_weight_offs = d_offs
    q_weight = tl.load(q_weight_ptr + q_weight_offs)
    if PARTIAL_ROTATION:
        q, inv_rms_q = _rms_norm_returning_inv(q, q_weight, BLOCK_D, eps)
    else:
        q = _rms_norm(q, q_weight, BLOCK_D, eps)

    if ENABLE_GATED_Q:
        d_gate_offs = tl.arange(0, BLOCK_D)
        x_gate_mask = t_mask[:, None] & (d_gate_offs < BLOCK_D)[None, :]
        gate_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + (gate_lane_base + d_gate_offs)[None, :] * stride_qkv_d
        )
        gate = tl.load(qkv_ptr + gate_in_offs, mask=x_gate_mask)
        gate_out_offs = (
            t_offs[:, None] * stride_q_t
            + d_gate_offs[None, :] * stride_q_d
            + hq * stride_q_h
        )
        tl.store(gate_ptr + gate_out_offs, gate, mask=x_gate_mask)

    if PARTIAL_ROTATION:
        if IS_NEOX:
            q_rotated = _partial_neox_rotated(
                inv_rms_q,
                q_weight_ptr,
                qkv_ptr,
                t_offs,
                d_offs,
                stride_qkv_t,
                stride_qkv_d,
                q_lane_base,
                x_mask,
                EFFECTIVE_RDH,
            )
        else:
            q_rotated = _partial_gptj_rotated(
                inv_rms_q,
                q_weight_ptr,
                qkv_ptr,
                t_offs,
                d_offs,
                stride_qkv_t,
                stride_qkv_d,
                q_lane_base,
                x_mask,
                ROTARY_SPAN,
            )
    elif IS_NEOX:
        q_rotated = _get_neox_rotated_x(
            q, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )
    else:
        q_rotated = _get_gptj_rotated_x(
            q, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )

    q_out_offs = (
        t_offs[:, None] * stride_q_t + d_offs[None, :] * stride_q_d + hq * stride_q_h
    )
    q = q * cos + q_rotated * sin
    q = q.to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + q_out_offs, q, mask=x_mask)

    if hq < KVH:
        if HAVE_K_SCALE:
            k_scale = tl.load(k_scale_ptr)
        else:
            k_scale = 1
        if HAVE_V_SCALE:
            v_scale = tl.load(v_scale_ptr)
        else:
            v_scale = 1

        Q_SIZE = QH * Q_HEAD_STRIDE
        KV_SIZE = KVH * BLOCK_D
        KV_HEAD_OFFS = hq * BLOCK_D
        k_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + KV_HEAD_OFFS) + d_offs)[None, :] * stride_qkv_d
        )
        v_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + KV_SIZE + KV_HEAD_OFFS) + d_offs)[None, :] * stride_qkv_d
        )
        k = tl.load(qkv_ptr + k_in_offs, mask=x_mask)
        v = tl.load(qkv_ptr + v_in_offs, mask=x_mask)

        k_weight_offs = d_offs
        k_weight = tl.load(k_weight_ptr + k_weight_offs)
        if PARTIAL_ROTATION:
            k, inv_rms_k = _rms_norm_returning_inv(k, k_weight, BLOCK_D, eps)
            K_HEAD_D_OFFSET = Q_SIZE + KV_HEAD_OFFS
            if IS_NEOX:
                k_rotated = _partial_neox_rotated(
                    inv_rms_k,
                    k_weight_ptr,
                    qkv_ptr,
                    t_offs,
                    d_offs,
                    stride_qkv_t,
                    stride_qkv_d,
                    K_HEAD_D_OFFSET,
                    x_mask,
                    EFFECTIVE_RDH,
                )
            else:
                k_rotated = _partial_gptj_rotated(
                    inv_rms_k,
                    k_weight_ptr,
                    qkv_ptr,
                    t_offs,
                    d_offs,
                    stride_qkv_t,
                    stride_qkv_d,
                    K_HEAD_D_OFFSET,
                    x_mask,
                    ROTARY_SPAN,
                )
        else:
            k = _rms_norm(k, k_weight, BLOCK_D, eps)
            if IS_NEOX:
                k_rotated = _get_neox_rotated_x(
                    k, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
                )
            else:
                k_rotated = _get_gptj_rotated_x(
                    k, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
                )

        k = k * cos + k_rotated * sin

        # Store to contiguous K/V buffers
        kv_out_offs = (
            t_offs[:, None] * stride_kv_t
            + d_offs[None, :] * stride_kv_d
            + hq * stride_kv_h
        )
        tl.store(k_ptr + kv_out_offs, k.to(k_ptr.dtype.element_ty), mask=x_mask)
        tl.store(v_ptr + kv_out_offs, v.to(v_ptr.dtype.element_ty), mask=x_mask)

        # KV Caching Logic: skip cache writes for padding (slot < 0) or out-of-range
        # slots. Bounds use total_num_kv_cache_tokens (= num_blocks * block_size)
        # so we avoid Python-side GPU sync (max/assert on slot_mapping).
        slots = tl.load(slot_mapping_ptr + t_offs, mask=t_mask)
        valid_slot = (slots >= 0) & (slots < total_num_kv_cache_tokens)
        safe_slots = tl.where(valid_slot, slots, 0)

        b_idx = safe_slots % BLOCK_SIZE
        t_slot_idx = safe_slots // BLOCK_SIZE

        cache_mask = x_mask & valid_slot[:, None]

        k_scale_rcprl = 1 / k_scale
        k = k * k_scale_rcprl

        v_scale_rcprl = 1 / v_scale
        v = v * v_scale_rcprl

        k_cache_offs = (
            t_slot_idx[:, None] * key_cache_stride_t
            + hq * key_cache_stride_h
            + d_offs[None, :] * key_cache_stride_d
            + b_idx[:, None] * key_cache_stride_b
        )
        tl.store(
            key_cache_ptr + k_cache_offs,
            k.to(key_cache_ptr.dtype.element_ty),
            mask=cache_mask,
        )

        v_cache_offs = (
            t_slot_idx[:, None] * value_cache_stride_t
            + hq * value_cache_stride_h
            + d_offs[None, :] * value_cache_stride_d
            + b_idx[:, None] * value_cache_stride_b
        )
        tl.store(
            value_cache_ptr + v_cache_offs,
            v.to(value_cache_ptr.dtype.element_ty),
            mask=cache_mask,
        )
