import triton
import triton.language as tl
from aiter.ops.triton.rope.rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D
from aiter.ops.triton._triton_kernels.kv_cache import _store_mla_kv_cache
from aiter.ops.triton._triton_kernels.quant.quant import _nvfp4_quant_op


@triton.jit
def _store_kv_cache_kernel(
    key_cache_ptr,
    value_cache_ptr,
    pid_t_slot,
    pid_hk,
    pid_b,
    d_pe_offs,
    k_pe,
    v,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    key_cache_stride_x,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    value_cache_stride_x,
    value_cache_stride_slot_chunk,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    FLASH_LAYOUT: tl.constexpr,
    VALUE_SHUFFLE_LAYOUT: tl.constexpr,
    SCALE_K_WIDTH: tl.constexpr,
):
    if key_cache_ptr.dtype.element_ty == tl.uint8:
        K_WIDTH: tl.constexpr = 16
        NVFP4_QUANT_BLOCK_SIZE: tl.constexpr = 16
        BLOCK_D_pe_STORE: tl.constexpr = BLOCK_D_pe // 2
        BLOCK_D_pe_scales: tl.constexpr = BLOCK_D_pe // NVFP4_QUANT_BLOCK_SIZE

        k_pe, k_pe_scales = _nvfp4_quant_op(k_pe, BLOCK_D_pe, 1, NVFP4_QUANT_BLOCK_SIZE)
        v, v_scales = _nvfp4_quant_op(v, BLOCK_D_pe, 1, NVFP4_QUANT_BLOCK_SIZE)

        d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe_STORE // K_WIDTH).to(tl.int64)
        k_width_shfl = tl.arange(0, K_WIDTH).to(tl.int64)
        k_pe = k_pe.reshape((BLOCK_D_pe_STORE // K_WIDTH, K_WIDTH))
        v = v.reshape((BLOCK_D_pe_STORE // K_WIDTH, K_WIDTH))

        key_cache_ptrs = (
            key_cache_ptr
            + pid_t_slot * key_cache_stride_t
            + pid_hk * key_cache_stride_h
        )
        value_cache_ptrs = (
            value_cache_ptr
            + pid_t_slot * value_cache_stride_t
            + pid_hk * value_cache_stride_h
        )

        key_value_cache_offs = (
            (pid_b // 16) * BLOCK_D_pe_STORE * 16
            + (pid_b % 16) * K_WIDTH
            + d_pe_offs_shfl[:, None] * K_WIDTH * 16
            + k_width_shfl[None, :]
        ) * key_cache_stride_d

        tl.store(
            key_cache_ptrs + key_value_cache_offs,
            k_pe.to(key_cache_ptr.dtype.element_ty),
        )
        tl.store(
            value_cache_ptrs + key_value_cache_offs,
            v.to(value_cache_ptr.dtype.element_ty),
        )

        d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe_scales // SCALE_K_WIDTH).to(tl.int64)
        k_pe_width_shfl = tl.arange(0, SCALE_K_WIDTH).to(tl.int64)
        k_pe_scales = k_pe_scales.reshape(
            (BLOCK_D_pe_scales // SCALE_K_WIDTH, SCALE_K_WIDTH)
        )
        v_scales = v_scales.reshape((BLOCK_D_pe_scales // SCALE_K_WIDTH, SCALE_K_WIDTH))
        pid_sub_blk = pid_b % 128
        key_cache_pe_scales_offs = (
            BLOCK_SIZE * BLOCK_D_pe_STORE
            + (pid_b // 128) * BLOCK_D_pe_scales * 128
            + d_pe_offs_shfl[:, None] * SCALE_K_WIDTH * 128
            + (pid_sub_blk % 32) * 4 * SCALE_K_WIDTH
            + (pid_sub_blk // 32) * SCALE_K_WIDTH
            + k_pe_width_shfl[None, :]
        ) * key_cache_stride_d
        e4m3_dtype = tl.float8e4nv
        tl.store(
            key_cache_ptrs + key_cache_pe_scales_offs,
            k_pe_scales.to(e4m3_dtype).to(key_cache_ptr.dtype.element_ty, bitcast=True),
        )
        tl.store(
            value_cache_ptrs + key_cache_pe_scales_offs,
            v_scales.to(e4m3_dtype).to(value_cache_ptr.dtype.element_ty, bitcast=True),
        )
    else:

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
            dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
            x_offs = tl.arange(0, X_SIZE).to(tl.int64)
            k_out_ptrs = (
                key_cache_ptr
                + pid_t_slot * key_cache_stride_t
                + pid_hk * key_cache_stride_h
                + dx_offs[:, None] * key_cache_stride_d
                + pid_b * key_cache_stride_b
                + x_offs[None, :] * key_cache_stride_x
            )

        if VALUE_SHUFFLE_LAYOUT:
            slot_chunk = pid_b // X_SIZE
            x_off = pid_b % X_SIZE
            v_out_ptrs = (
                value_cache_ptr
                + pid_t_slot * value_cache_stride_t
                + pid_hk * value_cache_stride_h
                + slot_chunk * value_cache_stride_slot_chunk
                + d_pe_offs * value_cache_stride_d
                + x_off * value_cache_stride_x
            )
        else:
            v_out_ptrs = (
                value_cache_ptr
                + pid_t_slot * value_cache_stride_t
                + pid_hk * value_cache_stride_h
                + d_pe_offs * value_cache_stride_d
                + pid_b * value_cache_stride_b
            )

        tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))
        tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))


@triton.jit
def _unit_cat(
    x1_ptr,
    x2_ptr,
    x_out_ptr,
    b_in,
    b_out,
    h,
    d1_offs,
    d2_offs,
    x1_stride_b,
    x1_stride_h,
    x1_stride_d,
    x2_stride_b,
    x2_stride_h,
    x2_stride_d,
    x_out_stride_b,
    x_out_stride_h,
    x_out_stride_d,
    k_scale,
    BLOCK_D1: tl.constexpr,
):
    x1_offs = b_in * x1_stride_b + h * x1_stride_h + d1_offs * x1_stride_d
    x2_offs = b_in * x2_stride_b + h * x2_stride_h + d2_offs * x2_stride_d
    x_out_offs = b_out * x_out_stride_b + h * x_out_stride_h

    x1 = tl.load(x1_ptr + x1_offs)
    x2 = tl.load(x2_ptr + x2_offs)

    x1 = (x1 / k_scale).to(x_out_ptr.dtype.element_ty)
    x2 = (x2 / k_scale).to(x_out_ptr.dtype.element_ty)
    tl.store(x_out_ptr + x_out_offs + d1_offs * x_out_stride_d, x1)
    tl.store(x_out_ptr + x_out_offs + (d2_offs + BLOCK_D1) * x_out_stride_d, x2)


@triton.jit
def _unit_rope_cat(
    x_nope_ptr,
    x_pe_ptr,
    cos,
    sin,
    x_out_ptr,
    b_in,
    b_out,
    h,
    d_nope_offs,
    d_pe_offs,
    x_nope_stride_b,
    x_nope_stride_h,
    x_nope_stride_d,
    x_pe_stride_b,
    x_pe_stride_h,
    x_pe_stride_d,
    x_out_stride_b,
    x_out_stride_h,
    x_out_stride_d,
    k_scale,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    x_nope_offs = (
        b_in * x_nope_stride_b + h * x_nope_stride_h + d_nope_offs * x_nope_stride_d
    )
    x_pe_offs = b_in * x_pe_stride_b + h * x_pe_stride_h + d_pe_offs * x_pe_stride_d
    x_out_offs = b_out * x_out_stride_b + h * x_out_stride_h

    x_nope = tl.load(x_nope_ptr + x_nope_offs)
    x_pe = tl.load(x_pe_ptr + x_pe_offs)

    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x_pe * cos + x_pe_rotated * sin
    x_pe = x_pe / k_scale
    x_nope = x_nope / k_scale
    x_nope = x_nope.to(x_out_ptr.dtype.element_ty)
    x_pe = x_pe.to(x_out_ptr.dtype.element_ty)

    tl.store(x_out_ptr + x_out_offs + d_nope_offs * x_out_stride_d, x_nope)
    tl.store(x_out_ptr + x_out_offs + (d_pe_offs + BLOCK_D_nope) * x_out_stride_d, x_pe)


@triton.jit
def _fused_qk_rope_cat_and_cache_mla_kernel(
    q_nope_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    decode_q_pe_out_ptr,
    k_pe_out_ptr,
    q_nope_zeros_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    B,
    B_slot,
    num_decode_toks_for_zeros,
    q_nope_stride_b,
    q_nope_stride_h,
    q_nope_stride_d,
    q_pe_stride_b,
    q_pe_stride_h,
    q_pe_stride_d,
    k_nope_stride_b,
    k_nope_stride_h,
    k_nope_stride_d,
    k_pe_stride_b,
    k_pe_stride_h,
    k_pe_stride_d,
    pos_stride_b,
    cos_stride_b,
    cos_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_d,
    decode_q_pe_out_stride_b,
    decode_q_pe_out_stride_h,
    decode_q_pe_out_stride_d,
    k_pe_out_stride_b,
    k_pe_out_stride_h,
    k_pe_out_stride_d,
    q_nope_zeros_out_stride_b,
    q_nope_zeros_out_stride_h,
    q_nope_zeros_out_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    k_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1,
    SHUFFLED_KV_CACHE: tl.constexpr = False,
    SCALE_K_WIDTH_NOPE: tl.constexpr = 4,
    SCALE_K_WIDTH_ROPE: tl.constexpr = 4,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_nope_offs = tl.arange(0, BLOCK_D_nope).to(tl.int64)
    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    if pid < B * QH:
        pid_b = pid // QH
        pid_hq = pid % QH
        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
                # d_cos_mask = d_cos_offs < BLOCK_D_pe
            else:
                d_cos_offs = d_pe_offs // 2
                # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe
        else:
            d_cos_offs = d_pe_offs
            # d_cos_mask = d_cos_offs < BLOCK_D_pe

        pos = tl.load(pos_ptr + pid_b * pos_stride_b)
        cos_offs = pos * cos_stride_b + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)

        q_nope_ptrs = (
            q_nope_ptr
            + pid_b * q_nope_stride_b
            + pid_hq * q_nope_stride_h
            + d_nope_offs * q_nope_stride_d
        )
        q_pe_ptrs = (
            q_pe_ptr
            + pid_b * q_pe_stride_b
            + pid_hq * q_pe_stride_h
            + d_pe_offs * q_pe_stride_d
        )
        q_out_ptrs = q_out_ptr + pid_b * q_out_stride_b + pid_hq * q_out_stride_h
        q_nope = tl.load(q_nope_ptrs)
        q_pe = _unit_rope(
            q_pe_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
        tl.store(
            q_out_ptrs + d_nope_offs * q_out_stride_d,
            q_nope.to(q_out_ptr.dtype.element_ty),
        )
        tl.store(
            q_out_ptrs + (d_pe_offs + BLOCK_D_nope) * q_out_stride_d,
            q_pe.to(q_out_ptr.dtype.element_ty),
        )

        if pid < num_decode_toks_for_zeros * QH:
            decode_q_pe_out_ptrs = (
                decode_q_pe_out_ptr
                + pid_b * decode_q_pe_out_stride_b
                + pid_hq * decode_q_pe_out_stride_h
            )
            tl.store(
                decode_q_pe_out_ptrs + d_pe_offs * decode_q_pe_out_stride_d,
                q_pe.to(decode_q_pe_out_ptr.dtype.element_ty),
            )

        if OUTPUT_Q_NOPE_ZEROS:
            if pid < num_decode_toks_for_zeros * QH:
                z = tl.zeros(
                    (BLOCK_D_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty
                )
                tl.store(
                    q_nope_zeros_out_ptr
                    + pid_b * q_nope_zeros_out_stride_b
                    + pid_hq * q_nope_zeros_out_stride_h
                    + d_nope_offs * q_nope_zeros_out_stride_d,
                    z,
                )

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if BLOCK_SIZE > 1:
                    pid_t_slot = pid_slot // BLOCK_SIZE
                    pid_blk = pid_slot % BLOCK_SIZE
                else:
                    pid_t_slot = pid_slot
                    pid_blk = 0
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                pid_hk = pid_hq // QH_PER_KH
                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + d_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)
                k_pe = _unit_rope(
                    k_pe_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = k_nope.to(tl.float32) * k_scale_rcprl
                k_pe = k_pe.to(tl.float32) * k_scale_rcprl

                _store_mla_kv_cache(
                    kv_cache_ptr,
                    pid_t_slot,
                    pid_hk,
                    pid_blk,
                    d_nope_offs,
                    d_pe_offs,
                    kv_cache_stride_b,
                    kv_cache_stride_h,
                    kv_cache_stride_d,
                    k_nope,
                    k_pe,
                    BLOCK_D_nope,
                    BLOCK_D_pe,
                    BLOCK_SIZE,
                    SHUFFLED_KV_CACHE,
                    SCALE_K_WIDTH_NOPE,
                    SCALE_K_WIDTH_ROPE,
                )
    else:
        pid = pid - B * QH + B * KH
        if pid < B_slot * KH:
            pid_b = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if BLOCK_SIZE > 1:
                    pid_t_slot = pid_slot // BLOCK_SIZE
                    pid_blk = pid_slot % BLOCK_SIZE
                else:
                    pid_t_slot = pid_slot
                    pid_blk = 0
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + d_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)
                k_pe = tl.load(k_pe_ptrs)
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = k_nope.to(tl.float32) * k_scale_rcprl
                k_pe = k_pe.to(tl.float32) * k_scale_rcprl

                _store_mla_kv_cache(
                    kv_cache_ptr,
                    pid_t_slot,
                    pid_hk,
                    pid_blk,
                    d_nope_offs,
                    d_pe_offs,
                    kv_cache_stride_b,
                    kv_cache_stride_h,
                    kv_cache_stride_d,
                    k_nope,
                    k_pe,
                    BLOCK_D_nope,
                    BLOCK_D_pe,
                    BLOCK_SIZE,
                    SHUFFLED_KV_CACHE,
                    SCALE_K_WIDTH_NOPE,
                    SCALE_K_WIDTH_ROPE,
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
    x_pe = tl.load(x_ptrs)

    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x_pe * cos + x_pe_rotated * sin

    return x_pe


@triton.jit
def _fused_qk_rope_reshape_and_cache_kernel(
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
    k_out_ptr,
    zeros_out_ptr,
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
    k_out_stride_t,
    k_out_stride_h,
    k_out_stride_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    key_cache_stride_x,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    value_cache_stride_slot_chunk,
    value_cache_stride_x,
    zeros_out_stride_t,
    zeros_out_stride_h,
    zeros_out_stride_d,
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
    SCALE_K_WIDTH: tl.constexpr,
    FLASH_LAYOUT: tl.constexpr,
    VALUE_SHUFFLE_LAYOUT: tl.constexpr = False,
    HAVE_POS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
    HAVE_ZEROS: tl.constexpr = False,
):

    tl.assume(q_stride_t >= 0)
    tl.assume(q_stride_h >= 0)
    tl.assume(q_stride_d >= 0)
    tl.assume(k_stride_t >= 0)
    tl.assume(k_stride_h >= 0)
    tl.assume(k_stride_d >= 0)
    tl.assume(v_stride_t >= 0)
    tl.assume(v_stride_h >= 0)
    tl.assume(v_stride_d >= 0)
    tl.assume(cos_stride_t >= 0)
    tl.assume(cos_stride_d >= 0)
    tl.assume(q_out_stride_t >= 0)
    tl.assume(q_out_stride_h >= 0)
    tl.assume(q_out_stride_d >= 0)
    tl.assume(k_out_stride_t >= 0)
    tl.assume(k_out_stride_h >= 0)
    tl.assume(k_out_stride_d >= 0)
    tl.assume(key_cache_stride_t >= 0)
    tl.assume(key_cache_stride_h >= 0)
    tl.assume(key_cache_stride_d >= 0)
    tl.assume(key_cache_stride_b >= 0)
    tl.assume(key_cache_stride_x >= 0)
    tl.assume(value_cache_stride_t >= 0)
    tl.assume(value_cache_stride_h >= 0)
    tl.assume(value_cache_stride_d >= 0)
    tl.assume(value_cache_stride_b >= 0)
    tl.assume(value_cache_stride_slot_chunk >= 0)
    tl.assume(value_cache_stride_x >= 0)
    tl.assume(zeros_out_stride_t >= 0)
    tl.assume(zeros_out_stride_h >= 0)
    tl.assume(zeros_out_stride_d >= 0)

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

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
                # d_cos_mask = d_cos_offs < BLOCK_D_pe
            else:
                d_cos_offs = d_pe_offs // 2
                # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe
        else:
            d_cos_offs = d_pe_offs
            # d_cos_mask = d_cos_offs < BLOCK_D_pe

        pos = tl.load(pos_ptr + pid_t)
        if HAVE_POS:
            offset = tl.load(offs_ptr + pid_t)
            pos = pos + offset
        cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)

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

        if HAVE_ZEROS:
            z = tl.zeros((BLOCK_D_pe,), dtype=zeros_out_ptr.dtype.element_ty)
            zeros_out_ptrs = (
                zeros_out_ptr
                + pid_t * zeros_out_stride_t
                + pid_hq * zeros_out_stride_h
                + d_pe_offs * zeros_out_stride_d
            )
            tl.store(zeros_out_ptrs, z)

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
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

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

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

                _store_kv_cache_kernel(
                    key_cache_ptr,
                    value_cache_ptr,
                    pid_t_slot,
                    pid_hk,
                    pid_b,
                    d_pe_offs,
                    k_pe,
                    v,
                    key_cache_stride_t,
                    key_cache_stride_h,
                    key_cache_stride_d,
                    key_cache_stride_b,
                    key_cache_stride_x,
                    value_cache_stride_t,
                    value_cache_stride_h,
                    value_cache_stride_d,
                    value_cache_stride_b,
                    value_cache_stride_x,
                    value_cache_stride_slot_chunk,
                    BLOCK_D_pe,
                    BLOCK_SIZE,
                    X_SIZE,
                    FLASH_LAYOUT,
                    VALUE_SHUFFLE_LAYOUT,
                    SCALE_K_WIDTH,
                )
    else:
        pid = pid - T * QH + T * KH
        if pid < T_slot * KH:
            pid_t = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
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

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

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

                _store_kv_cache_kernel(
                    key_cache_ptr,
                    value_cache_ptr,
                    pid_t_slot,
                    pid_hk,
                    pid_b,
                    d_pe_offs,
                    k_pe,
                    v,
                    key_cache_stride_t,
                    key_cache_stride_h,
                    key_cache_stride_d,
                    key_cache_stride_b,
                    key_cache_stride_x,
                    value_cache_stride_t,
                    value_cache_stride_h,
                    value_cache_stride_d,
                    value_cache_stride_b,
                    value_cache_stride_x,
                    value_cache_stride_slot_chunk,
                    BLOCK_D_pe,
                    BLOCK_SIZE,
                    X_SIZE,
                    FLASH_LAYOUT,
                    VALUE_SHUFFLE_LAYOUT,
                    SCALE_K_WIDTH,
                )


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

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

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
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
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
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
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
