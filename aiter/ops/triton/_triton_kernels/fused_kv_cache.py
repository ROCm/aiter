import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.rope import (
    _get_gptj_rotated_x_1D,
    _get_neox_rotated_x_1D,
)


@triton.jit
def _unit_rope(
    x_ptrs,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    x = tl.load(x_ptrs).to(tl.float64)

    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x * cos + x_pe_rotated * sin

    return x_pe


@triton.jit
def _fused_qk_rope_cosine_cache_llama_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    offs_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    q_out_ptr,
    T,
    T_slot,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    cos_stride_t,
    cos_stride_d,
    q_out_stride_t,
    q_out_stride_h,
    q_out_stride_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    key_cache_stride_x,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    k_scale_ptr,
    v_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    FLASH_LAYOUT: tl.constexpr,
    HAVE_POS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_pe_offs = tl.arange(0, BLOCK_D_pe)

    if pid < T * QH:
        pid_t = pid // QH
        pid_hq = pid % QH
        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
            else:
                d_cos_offs = d_pe_offs // 2
                d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe

        else:
            d_cos_offs = d_pe_offs

        pos = tl.load(pos_ptr + pid_t)
        if HAVE_POS:
            offset = tl.load(offs_ptr + pid_t)
            pos = pos + offset
        cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs).to(tl.float64)
        sin = tl.load(sin_ptr + cos_offs).to(tl.float64)

        q_ptrs = (
            q_ptr + pid_t * q_stride_t + pid_hq * q_stride_h + d_pe_offs * q_stride_d
        )
        q_pe = _unit_rope(
            q_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
        q_out_ptrs = (
            q_out_ptr
            + pid_t * q_out_stride_t
            + pid_hq * q_out_stride_h
            + d_pe_offs * q_out_stride_d
        )
        tl.store(q_out_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_t
                pid_b = pid_slot
                pid_hk = pid_hq // QH_PER_KH
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )
                k_pe = _unit_rope(
                    k_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                        + d_pe_offs * key_cache_stride_d
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE)
                    x_offs = tl.arange(0, X_SIZE)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )

                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                v_out_ptrs = (
                    value_cache_ptr
                    + pid_t_slot * value_cache_stride_t
                    + pid_hk * value_cache_stride_h
                    + d_pe_offs * value_cache_stride_d
                    + pid_b * value_cache_stride_b
                )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))
    else:
        pid = pid - T * QH + T * KH
        if pid < T_slot * KH:
            pid_t = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_t
                pid_b = pid_slot
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )

                k_pe = tl.load(k_ptrs)

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + d_pe_offs * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE)
                    x_offs = tl.arange(0, X_SIZE)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )
                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                v_out_ptrs = (
                    value_cache_ptr
                    + pid_t_slot * value_cache_stride_t
                    + pid_hk * value_cache_stride_h
                    + d_pe_offs * value_cache_stride_d
                    + pid_b * value_cache_stride_b
                )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))
