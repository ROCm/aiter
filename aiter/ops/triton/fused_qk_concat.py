import torch
import triton
import triton.language as tl
from aiter.ops.triton.rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


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
    BLOCK_D1: tl.constexpr,
    K_SCALE: tl.constexpr = 1,
):
    x1_offs = b_in * x1_stride_b + h * x1_stride_h + d1_offs * x1_stride_d
    x2_offs = b_in * x2_stride_b + h * x2_stride_h + d2_offs * x2_stride_d
    x_out_offs = b_out * x_out_stride_b + h * x_out_stride_h

    x1 = tl.load(x1_ptr + x1_offs)
    x2 = tl.load(x2_ptr + x2_offs)

    if K_SCALE != 1:
        x1 = x1 * K_SCALE
        x2 = x2 * K_SCALE
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


def fused_qk_cat(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
):
    """
    Concat q1 with q2 and k1 with k2 along the last dimension

    Key parameters:
    - q1: Matrix X with shape (B, QH, D1).
    - q2: Matrix W with shape (B, QH, D2).
    - k1: Matrix X with shape (B, KH, D1).
    - k2: Matrix W with shape (B, KH, D2).

    QH must be multiple of KH

    Returns:
    - q_out: The output matrix with shape (B, QH, D1+D2).
    - k_out: The output matrix with shape (B, KH, D1+D2).
    """
    _LOGGER.info(
        f"FUSED_QK_CAT: q1={tuple(q1.shape)} q2={tuple(q2.shape)} k1={tuple(k1.shape)} k2={tuple(k2.shape)} "
    )
    b, qh, d1 = q1.shape
    b2, qh2, d2 = q2.shape
    bk, kh, dk1 = k1.shape
    bk2, kh2, dk2 = k2.shape
    assert (
        b == b2 == bk == bk2
    ), "q1 batch dimension should be identical across all inputs"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2, "K head should be identical"
    assert d1 == dk1, "D dimension of q1 and k1 should be identical"
    assert d2 == dk2, "D dimension of q2 and k2 should be identical"
    assert qh % kh == 0, "Number of Q heads must be multiple of number H heads"

    q_out = torch.empty((b, qh, d1 + d2), dtype=q1.dtype, device=q1.device)
    k_out = torch.empty((b, kh, d1 + d2), dtype=q1.dtype, device=q1.device)

    grid = (b, qh, 1)

    _qk_cat_kernel[grid](
        q1,
        q2,
        k1,
        k2,
        q_out,
        k_out,
        *q1.stride(),
        *q2.stride(),
        *k1.stride(),
        *k2.stride(),
        *q_out.stride(),
        *k_out.stride(),
        QH_PER_KH=qh // kh,
        BLOCK_D1=d1,
        BLOCK_D2=d2,
    )

    return q_out, k_out


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
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    K_SCALE: tl.constexpr = 1,
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
    if K_SCALE != 1:
        x_pe = x_pe * K_SCALE
        x_nope = x_nope * K_SCALE
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


def fused_qk_rope_cat(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool,
):
    """
    Perform RoPE on q_pe and k_pe and concat q_nope with q_pe and k_nope with k_pe along the last dimension

    Key parameters:
    - q_nope: Matrix X with shape (B, QH, D1).
    - q_pe: Matrix W with shape (B, QH, D2).
    - k_nope: Matrix X with shape (B, KH, D1).
    - k_pe: Matrix W with shape (B, KH, D2).

    QH must be multiple of KH

    Returns:
    - q_out: The output matrix with shape (B, QH, D1+D2).
    - k_out: The output matrix with shape (B, KH, D1+D2).
    """
    _LOGGER.info(
        f"FUSED_QK_ROPE_CAT: q_nope={tuple(q_nope.shape)} q_pe={tuple(q_pe.shape)} k_nope={tuple(k_nope.shape)} k_pe={tuple(k_pe.shape)} "
        + f"pos={tuple(pos.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
    )
    b, qh, d_nope = q_nope.shape
    b2, qh2, d_pe = q_pe.shape
    bk, kh, dk1 = k_nope.shape
    bk2, kh2, dk2 = k_pe.shape

    assert (
        b == b2 == bk == bk2
    ), "q1 batch dimension should be identical across all inputs"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2, "K head should be identical"
    assert d_nope == dk1, "D dimension of q_nope and k_nope should be identical"
    assert d_pe == dk2, "D dimension of q_pe and k_pe should be identical"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d_pe // 2) or (
        d_freq == d_pe
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d_pe // 2

    q_out = torch.empty(
        (b, qh, d_nope + d_pe), dtype=q_nope.dtype, device=q_nope.device
    )
    k_out = torch.empty(
        (b, kh, d_nope + d_pe), dtype=q_nope.dtype, device=q_nope.device
    )

    grid = (b, qh, 1)

    _qk_rope_cat_kernel[grid](
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        pos,
        cos,
        sin,
        q_out,
        k_out,
        *q_nope.stride(),
        *q_pe.stride(),
        *k_nope.stride(),
        *k_pe.stride(),
        pos.stride(0),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *k_out.stride(),
        QH_PER_KH=qh // kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_nope=d_nope,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
    )

    return q_out, k_out


@triton.jit
def _qk_rope_cat_and_cache_mla_kernel(
    q_nope_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    q_nope_zeros_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    B,
    B_slot,
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
    q_nope_zeros_out_stride_b,
    q_nope_zeros_out_stride_h,
    q_nope_zeros_out_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr = False,
    K_SCALE: tl.constexpr = 1,
):
    pid = tl.program_id(0)

    d_nope_offs = tl.arange(0, BLOCK_D_nope)
    d_pe_offs = tl.arange(0, BLOCK_D_pe)

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

        _unit_rope_cat(
            q_nope_ptr,
            q_pe_ptr,
            cos,
            sin,
            q_out_ptr,
            pid_b,
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

        if OUTPUT_Q_NOPE_ZEROS:
            z = tl.zeros((BLOCK_D_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty)
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
                _unit_rope_cat(
                    k_nope_ptr,
                    k_pe_ptr,
                    cos,
                    sin,
                    kv_cache_ptr,
                    pid_b,
                    pid_slot,
                    pid_hq // QH_PER_KH,
                    d_nope_offs,
                    d_pe_offs,
                    k_nope_stride_b,
                    k_nope_stride_h,
                    k_nope_stride_d,
                    k_pe_stride_b,
                    k_pe_stride_h,
                    k_pe_stride_d,
                    kv_cache_stride_b,
                    kv_cache_stride_h,
                    kv_cache_stride_d,
                    IS_NEOX,
                    BLOCK_D_nope,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                    K_SCALE,
                )
    else:
        pid = pid - B * QH + B * KH
        if pid < B_slot * KH:
            pid_b = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                _unit_cat(
                    k_nope_ptr,
                    k_pe_ptr,
                    kv_cache_ptr,
                    pid,
                    pid_slot,
                    pid_hk,
                    d_nope_offs,
                    d_pe_offs,
                    k_nope_stride_b,
                    k_nope_stride_h,
                    k_nope_stride_d,
                    k_pe_stride_b,
                    k_pe_stride_h,
                    k_pe_stride_d,
                    kv_cache_stride_b,
                    kv_cache_stride_h,
                    kv_cache_stride_d,
                    BLOCK_D_nope,
                    K_SCALE,
                )


def fused_qk_rope_cat_and_cache_mla(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    is_neox: bool,
    output_q_nope_zeros: bool = False,
):
    """
    Perform RoPE on q_pe and k_pe and concat q_nope with q_pe and k_nope with k_pe along the last dimension
    the concatentaed k_nope and k_pe are copied to kv_cache inplace

    Key parameters:
    - q_nope: Matrix X with shape (B, QH, D1).
    - q_pe: Matrix W with shape (B, QH, D2).
    - k_nope: Matrix X with shape (B_slot, KH, D1).
    - k_pe: Matrix W with shape (B_slot, KH, D2).
    - kv_cache: Matrix W with shape (B_cache, KH, D1 + D2).
    - slot_mapping: Matrix W with shape (B_slot, ).

    B is the number of decode tokens, B_slot is the number of prefill + decode tokens, B_cahce is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: The output matrix with shape (B, QH, D1+D2).
    - kv_cache: The output matrix with shape (B_max, KH, D1 + D2) (inplace).
    """
    _LOGGER.info(
        f"FUSED_QK_ROPE_CAT_AND_CACHE_MLA: q_nope={tuple(q_nope.shape)} q_pe={tuple(q_pe.shape)} k_nope={tuple(k_nope.shape)} k_pe={tuple(k_pe.shape)} "
        + f"pos={tuple(pos.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} kv_cache={tuple(kv_cache.shape)} slot_mapping={tuple(slot_mapping.shape)}"
    )

    b, qh, d_nope = q_nope.shape
    b2, qh2, d_pe = q_pe.shape
    bk, kh, dk1 = k_nope.shape
    bk2, kh2, dk2 = k_pe.shape
    b_cache, h_cache, d_cache = kv_cache.shape
    (b_slot,) = slot_mapping.shape

    assert b == b2, "batch dimension should be identical for q_nope and q_pe"
    assert (
        bk == bk2 == b_slot
    ), "batch dimension should be identical for k_nope, k_pe, and slot_mapping"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2 == h_cache, "K head should be identical"
    assert d_nope == dk1, "D dimension of q_nope and k_nope should be identical"
    assert d_pe == dk2, "D dimension of q_pe and k_pe should be identical"
    assert (
        dk1 + dk2 == d_cache
    ), "D dimension of k_nope and k_pe should be summed up to be the D dimension of kv_cache"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d_pe // 2) or (
        d_freq == d_pe
    ), "cos/sin last dim should be the same or half of the qk last dim"
    assert k_scale.numel() == 1, "k_scale should be a single-element torch.Tensor"
    reuse_freqs_front_part = d_freq == d_pe // 2

    q_out = torch.empty(
        (b, qh, d_nope + d_pe), dtype=q_nope.dtype, device=q_nope.device
    )
    q_nope_zeros_out = None
    if output_q_nope_zeros:
        q_nope_zeros_out = torch.empty(
            (b, qh, d_nope), dtype=q_nope.dtype, device=q_nope.device
        )

    n_pid = b * qh + (b_slot - b) * kh
    grid = (n_pid, 1, 1)
    _qk_rope_cat_and_cache_mla_kernel[grid](
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        pos,
        cos,
        sin,
        q_out,
        q_nope_zeros_out,
        kv_cache,
        slot_mapping,
        b,
        b_slot,
        *q_nope.stride(),
        *q_pe.stride(),
        *k_nope.stride(),
        *k_pe.stride(),
        pos.stride(0),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *q_nope_zeros_out.stride(),
        *kv_cache.stride(),
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_nope=d_nope,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
        OUTPUT_Q_NOPE_ZEROS=output_q_nope_zeros,
        K_SCALE=k_scale.item(),
    )

    if output_q_nope_zeros:
        return q_out, q_nope_zeros_out
    return q_out


@triton.jit
def _unit_rope_cat_per_token_fp8_quant(
    x_nope_ptr,
    x_pe_ptr,
    cos,
    sin,
    x_out_ptr,
    xs_out_ptr,
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
    xs_out_stride_b,
    xs_out_stride_h,
    xs_out_stride_d,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
):
    x_nope_offs = (
        b_in * x_nope_stride_b + h * x_nope_stride_h + d_nope_offs * x_nope_stride_d
    )
    x_pe_offs = b_in * x_pe_stride_b + h * x_pe_stride_h + d_pe_offs * x_pe_stride_d
    x_out_offs = b_out * x_out_stride_b + h * x_out_stride_h
    xs_out_offs = b_out * xs_out_stride_b + h * xs_out_stride_h + 0 * xs_out_stride_d

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

    m = tl.maximum(
        tl.maximum(tl.max(tl.abs(x_nope), axis=-1), tl.max(tl.abs(x_pe), axis=-1)),
        1e-10,
    )
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1 / scale_out[:, None]

    x_nope = tl.clamp(x_nope * scale_recip, DTYPE_MIN, DTYPE_MAX).to(
        x_out_ptr.dtype.element_ty
    )
    x_pe = tl.clamp(x_pe * scale_recip, DTYPE_MIN, DTYPE_MAX).to(
        x_out_ptr.dtype.element_ty
    )

    tl.store(x_out_ptr + x_out_offs + d_nope_offs * x_out_stride_d, x_nope)
    tl.store(x_out_ptr + x_out_offs + (d_pe_offs + BLOCK_D_nope) * x_out_stride_d, x_pe)
    tl.store(xs_out_ptr + xs_out_offs, scale_out)


@triton.jit
def _qk_rope_cat_and_cache_mla_q_per_token_fp8_quant_kernel(
    q_nope_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    qs_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    B,
    B_slot,
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
    qs_out_stride_b,
    qs_out_stride_h,
    qs_out_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    K_SCALE: tl.constexpr = 1,
):
    pid = tl.program_id(0)

    d_nope_offs = tl.arange(0, BLOCK_D_nope)
    d_pe_offs = tl.arange(0, BLOCK_D_pe)

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

        _unit_rope_cat_per_token_fp8_quant(
            q_nope_ptr,
            q_pe_ptr,
            cos,
            sin,
            q_out_ptr,
            qs_out_ptr,
            pid_b,
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
            qs_out_stride_b,
            qs_out_stride_h,
            qs_out_stride_d,
            IS_NEOX,
            BLOCK_D_nope,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
            DTYPE_MAX,
            DTYPE_MIN,
        )

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                _unit_rope_cat(
                    k_nope_ptr,
                    k_pe_ptr,
                    cos,
                    sin,
                    kv_cache_ptr,
                    pid_b,
                    pid_slot,
                    pid_hq // QH_PER_KH,
                    d_nope_offs,
                    d_pe_offs,
                    k_nope_stride_b,
                    k_nope_stride_h,
                    k_nope_stride_d,
                    k_pe_stride_b,
                    k_pe_stride_h,
                    k_pe_stride_d,
                    kv_cache_stride_b,
                    kv_cache_stride_h,
                    kv_cache_stride_d,
                    IS_NEOX,
                    BLOCK_D_nope,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                    K_SCALE,
                )
    else:
        pid = pid - B * QH + B * KH
        if pid < B_slot * KH:
            pid_b = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                _unit_cat(
                    k_nope_ptr,
                    k_pe_ptr,
                    kv_cache_ptr,
                    pid,
                    pid_slot,
                    pid_hk,
                    d_nope_offs,
                    d_pe_offs,
                    k_nope_stride_b,
                    k_nope_stride_h,
                    k_nope_stride_d,
                    k_pe_stride_b,
                    k_pe_stride_h,
                    k_pe_stride_d,
                    kv_cache_stride_b,
                    kv_cache_stride_h,
                    kv_cache_stride_d,
                    BLOCK_D_nope,
                    K_SCALE,
                )


def fused_qk_rope_cat_and_cache_mla_q_per_token_fp8_quant(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    is_neox: bool,
    dtype_quant=torch.int8,
):
    """
    Perform RoPE on q_pe and k_pe and concat q_nope with q_pe and k_nope with k_pe along the last dimension
    the concatentaed k_nope and k_pe are copied to kv_cache inplace

    Key parameters:
    - q_nope: Matrix X with shape (B, QH, D1).
    - q_pe: Matrix W with shape (B, QH, D2).
    - k_nope: Matrix X with shape (B_slot, KH, D1).
    - k_pe: Matrix W with shape (B_slot, KH, D2).
    - kv_cache: Matrix W with shape (B_cache, KH, D1 + D2).
    - slot_mapping: Matrix W with shape (B_slot, ).

    B is the number of decode tokens, B_slot is the number of prefill + decode tokens, B_cahce is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: The output matrix with shape (B, QH, D1+D2).
    - qs_out: The output matrix with shape (B, QH, 1).
    - kv_cache: The output matrix with shape (B_max, KH, D1 + D2) (inplace).
    """
    _LOGGER.info(
        f"FUSED_QK_ROPE_CAT_AND_CACHE_MLA_Q_PER_TOKEN_FP8_QUANT: q_nope={tuple(q_nope.shape)} q_pe={tuple(q_pe.shape)} k_nope={tuple(k_nope.shape)} k_pe={tuple(k_pe.shape)} "
        + f"pos={tuple(pos.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} kv_cache={tuple(kv_cache.shape)} slot_mapping={tuple(slot_mapping.shape)} dtype_quant={dtype_quant}"
    )

    b, qh, d_nope = q_nope.shape
    b2, qh2, d_pe = q_pe.shape
    bk, kh, dk1 = k_nope.shape
    bk2, kh2, dk2 = k_pe.shape
    b_cache, h_cache, d_cache = kv_cache.shape
    (b_slot,) = slot_mapping.shape

    assert b == b2, "batch dimension should be identical for q_nope and q_pe"
    assert (
        bk == bk2 == b_slot
    ), "batch dimension should be identical for k_nope, k_pe, and slot_mapping"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2 == h_cache, "K head should be identical"
    assert d_nope == dk1, "D dimension of q_nope and k_nope should be identical"
    assert d_pe == dk2, "D dimension of q_pe and k_pe should be identical"
    assert (
        dk1 + dk2 == d_cache
    ), "D dimension of k_nope and k_pe should be summed up to be the D dimension of kv_cache"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d_pe // 2) or (
        d_freq == d_pe
    ), "cos/sin last dim should be the same or half of the qk last dim"
    assert k_scale.numel() == 1, "k_scale should be a single-element torch.Tensor"
    reuse_freqs_front_part = d_freq == d_pe // 2

    q_out = torch.empty((b, qh, d_nope + d_pe), dtype=dtype_quant, device=q_nope.device)
    qs_out = torch.empty((b, qh, 1), dtype=torch.float32, device=q_nope.device)

    DTYPE_MAX = (
        torch.finfo(q_out.dtype).max
        if torch.is_floating_point(q_out)
        else torch.iinfo(q_out.dtype).max
    )
    n_pid = b * qh + (b_slot - b) * kh
    grid = (n_pid, 1, 1)
    _qk_rope_cat_and_cache_mla_q_per_token_fp8_quant_kernel[grid](
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        pos,
        cos,
        sin,
        q_out,
        qs_out,
        kv_cache,
        slot_mapping,
        b,
        b_slot,
        *q_nope.stride(),
        *q_pe.stride(),
        *k_nope.stride(),
        *k_pe.stride(),
        pos.stride(0),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *qs_out.stride(),
        *kv_cache.stride(),
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_nope=d_nope,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        K_SCALE=k_scale.item(),
    )

    return q_out, qs_out
