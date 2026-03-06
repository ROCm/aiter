import triton
import triton.language as tl
from aiter.ops.triton.rope.rope import _get_gptj_rotated_x, _get_neox_rotated_x

@triton.jit
def _rms_norm(
    tensor,
    weight,
    BLOCK_D,
    eps
):
    tensor_f32 = tensor.to(tl.float32)
    tensor_sq_sum = tl.sum(tensor_f32 * tensor_f32, axis=1)
    tensor_rsqrt = tl.rsqrt(tensor_sq_sum / BLOCK_D + eps)

    tensor_normed = tensor_f32 * tensor_rsqrt[:, None]
    tensor_normed = tensor_normed * (1.0 + weight[None, :])
    tensor = tensor_normed.to(tensor.dtype)
    return tensor

@triton.jit
def _fused_qkv_split_qk_rope_kernel(
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
    BLOCK_SIZE: tl.constexpr, # PagedAttention block size
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
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        qk_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        qk_rotated_mask = (d_offs % 2 == 0)[None, :]

    H_OFFS_SIZE = hq * BLOCK_D
    q_in_offs = (
        t_offs[:, None] * stride_qkv_t
        + (H_OFFS_SIZE + d_offs)[None, :] * stride_qkv_d
    )
    q = tl.load(qkv_ptr + q_in_offs, mask=x_mask)

    q_weight_offs = d_offs
    q_weight = tl.load(q_weight_ptr + q_weight_offs)
    q = _rms_norm(q, q_weight, BLOCK_D, eps)

    if ENABLE_GATED_Q:
        Q_SIZE = QH * BLOCK_D
        d_gate_offs = tl.arange(0, BLOCK_D)
        x_gate_mask = t_mask[:, None] & (d_gate_offs < BLOCK_D)[None, :]
        gate_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + (Q_SIZE + H_OFFS_SIZE + d_gate_offs)[None, :]
            * stride_qkv_d
        )
        gate = tl.load(qkv_ptr + gate_in_offs, mask=x_gate_mask)
        gate_out_offs = (
            t_offs[:, None] * stride_q_t
            + d_gate_offs[None, :] * stride_q_d
            + hq * stride_q_h
        )
        tl.store(gate_ptr + gate_out_offs, gate, mask=x_gate_mask)

    if IS_NEOX:
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
        Q_HEAD_DIM_STRIDE_MULT = 1
        if ENABLE_GATED_Q:
            Q_HEAD_DIM_STRIDE_MULT = 2
        Q_HEAD_SIZE_IN = BLOCK_D * Q_HEAD_DIM_STRIDE_MULT
        Q_SIZE = QH * Q_HEAD_SIZE_IN
        KV_SIZE = KVH * BLOCK_D
        k_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + H_OFFS_SIZE) + d_offs)[None, :]
            * stride_qkv_d
        )
        v_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + KV_SIZE + H_OFFS_SIZE) + d_offs)[None, :]
            * stride_qkv_d
        )
        k = tl.load(qkv_ptr + k_in_offs, mask=x_mask)
        v = tl.load(qkv_ptr + v_in_offs, mask=x_mask)

        k_weight_offs = d_offs
        k_weight = tl.load(k_weight_ptr + k_weight_offs)
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

        # KV Caching Logic
        slots = tl.load(slot_mapping_ptr + t_offs, mask=t_mask)
        
        # Calculate Paged Cache offsets
        b_idx = slots % BLOCK_SIZE
        t_slot_idx = slots // BLOCK_SIZE
        
        # We process a block of T. We need to handle mask for slots >= 0
        cache_mask = x_mask & (slots[:, None] >= 0)

        k_cache_offs = (
            t_slot_idx[:, None] * key_cache_stride_t
            + hq * key_cache_stride_h
            + d_offs[None, :] * key_cache_stride_d
            + b_idx[:, None] * key_cache_stride_b
        )
        tl.store(key_cache_ptr + k_cache_offs, k.to(key_cache_ptr.dtype.element_ty), mask=cache_mask)

        v_cache_offs = (
            t_slot_idx[:, None] * value_cache_stride_t
            + hq * value_cache_stride_h
            + d_offs[None, :] * value_cache_stride_d
            + b_idx[:, None] * value_cache_stride_b
        )
        tl.store(value_cache_ptr + v_cache_offs, v.to(value_cache_ptr.dtype.element_ty), mask=cache_mask)