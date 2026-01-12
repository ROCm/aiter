import triton
import triton.language as tl
from .rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D


@triton.jit
def _unit_cat(
    x1_ptr,
    x2_ptr,
    x_out_ptr,
    b,
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
    BLOCK_D1: tl.constexpr,
):
    x1_offs = b * x1_stride_b + h * x1_stride_h + d1_offs * x1_stride_d
    x2_offs = b * x2_stride_b + h * x2_stride_h + d2_offs * x2_stride_d
    x_out_offs = b * x_out_stride_b + h * x_out_stride_h

    x1 = tl.load(x1_ptr + x1_offs)
    x2 = tl.load(x2_ptr + x2_offs)

    tl.store(x_out_ptr + x_out_offs + d1_offs * x_out_stride_d, x1)
    tl.store(x_out_ptr + x_out_offs + (d2_offs + BLOCK_D1) * x_out_stride_d, x2)


@triton.jit
def _qk_cat_kernel(
    q1_ptr,
    q2_ptr,
    k1_ptr,
    k2_ptr,
    q_out_ptr,
    k_out_ptr,
    q1_stride_b,
    q1_stride_h,
    q1_stride_d,
    q2_stride_b,
    q2_stride_h,
    q2_stride_d,
    k1_stride_b,
    k1_stride_h,
    k1_stride_d,
    k2_stride_b,
    k2_stride_h,
    k2_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_d,
    k_out_stride_b,
    k_out_stride_h,
    k_out_stride_d,
    QH_PER_KH: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)

    d1_offs = tl.arange(0, BLOCK_D1)
    d2_offs = tl.arange(0, BLOCK_D2)

    _unit_cat(
        q1_ptr,
        q2_ptr,
        q_out_ptr,
        pid_b,
        pid_hq,
        d1_offs,
        d2_offs,
        q1_stride_b,
        q1_stride_h,
        q1_stride_d,
        q2_stride_b,
        q2_stride_h,
        q2_stride_d,
        q_out_stride_b,
        q_out_stride_h,
        q_out_stride_d,
        BLOCK_D1,
    )

    if pid_hq % QH_PER_KH == 0:
        _unit_cat(
            k1_ptr,
            k2_ptr,
            k_out_ptr,
            pid_b,
            pid_hq // QH_PER_KH,
            d1_offs,
            d2_offs,
            k1_stride_b,
            k1_stride_h,
            k1_stride_d,
            k2_stride_b,
            k2_stride_h,
            k2_stride_d,
            k_out_stride_b,
            k_out_stride_h,
            k_out_stride_d,
            BLOCK_D1,
        )


@triton.jit
def _unit_rope_cat(
    x_nope_ptr,
    x_pe_ptr,
    cos,
    sin,
    x_out_ptr,
    b,
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
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    x_nope_offs = (
        b * x_nope_stride_b + h * x_nope_stride_h + d_nope_offs * x_nope_stride_d
    )
    x_pe_offs = b * x_pe_stride_b + h * x_pe_stride_h + d_pe_offs * x_pe_stride_d
    x_out_offs = b * x_out_stride_b + h * x_out_stride_h

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
    x_pe = x_pe.to(x_pe_ptr.dtype.element_ty)

    tl.store(x_out_ptr + x_out_offs + d_nope_offs * x_out_stride_d, x_nope)
    tl.store(x_out_ptr + x_out_offs + (d_pe_offs + BLOCK_D_nope) * x_out_stride_d, x_pe)


@triton.jit
def _qk_rope_cat_kernel(
    q_nope_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
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
    k_out_stride_b,
    k_out_stride_h,
    k_out_stride_d,
    QH_PER_KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)

    d_nope_offs = tl.arange(0, BLOCK_D_nope)
    d_pe_offs = tl.arange(0, BLOCK_D_pe)

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

    _unit_rope_cat(
        q_nope_ptr,
        q_pe_ptr,
        cos,
        sin,
        q_out_ptr,
        pid_b,
        pid_hq,
        d_nope_offs,
        d_pe_offs,
        q_nope_stride_b,
        q_nope_stride_h,
        q_nope_stride_d,
        q_pe_stride_b,
        q_pe_stride_h,
        q_pe_stride_d,
        q_out_stride_b,
        q_out_stride_h,
        q_out_stride_d,
        IS_NEOX,
        BLOCK_D_nope,
        BLOCK_D_pe,
        BLOCK_D_HALF_pe,
    )

    if pid_hq % QH_PER_KH == 0:
        _unit_rope_cat(
            k_nope_ptr,
            k_pe_ptr,
            cos,
            sin,
            k_out_ptr,
            pid_b,
            pid_hq // QH_PER_KH,
            d_nope_offs,
            d_pe_offs,
            k_nope_stride_b,
            k_nope_stride_h,
            k_nope_stride_d,
            k_pe_stride_b,
            k_pe_stride_h,
            k_pe_stride_d,
            k_out_stride_b,
            k_out_stride_h,
            k_out_stride_d,
            IS_NEOX,
            BLOCK_D_nope,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
