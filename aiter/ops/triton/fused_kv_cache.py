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
    BLOCK_DK_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_nope_offs = tl.arange(0, BLOCK_D_nope)
    dk_nope_offs = tl.arange(0, BLOCK_DK_nope)
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

        q_nope_ptrs = (
            q_nope_ptr 
            + pid_b * q_nope_stride_b 
            + pid_hq * q_nope_stride_h 
            + d_nope_offs * q_nope_stride_d
        )
        q_pe_ptrs = q_pe_ptr + pid_b * q_pe_stride_b + pid_hq * q_pe_stride_h + d_pe_offs * q_pe_stride_d
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
        tl.store(q_out_ptrs + d_nope_offs * q_out_stride_d, q_nope.to(q_out_ptr.dtype.element_ty))
        tl.store(q_out_ptrs + (d_pe_offs + BLOCK_D_nope) * q_out_stride_d, q_pe.to(q_out_ptr.dtype.element_ty))
        
        if pid < num_decode_toks_for_zeros * QH:
            decode_q_pe_out_ptrs = decode_q_pe_out_ptr + pid_b * decode_q_pe_out_stride_b + pid_hq * decode_q_pe_out_stride_h
            tl.store(decode_q_pe_out_ptrs + d_pe_offs * decode_q_pe_out_stride_d, q_pe.to(decode_q_pe_out_ptr.dtype.element_ty))

        if OUTPUT_Q_NOPE_ZEROS and pid < num_decode_toks_for_zeros * QH:
            z = tl.zeros((BLOCK_DK_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty)
            tl.store(
                q_nope_zeros_out_ptr
                + pid_b * q_nope_zeros_out_stride_b
                + pid_hq * q_nope_zeros_out_stride_h
                + dk_nope_offs * q_nope_zeros_out_stride_d,
                z,
            )

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                    
                pid_hk = pid_hq // QH_PER_KH
                k_nope_ptrs = (
                    k_nope_ptr 
                    + pid_b * k_nope_stride_b 
                    + pid_hk * k_nope_stride_h 
                    + dk_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = k_pe_ptr + pid_b * k_pe_stride_b + pid_hk * k_pe_stride_h + d_pe_offs * k_pe_stride_d
                k_pe_out_ptrs = k_pe_out_ptr + pid_b * k_pe_out_stride_b + pid_hk * k_pe_out_stride_h + d_pe_offs * k_pe_out_stride_d
                kv_cache_ptrs = kv_cache_ptr + pid_slot * kv_cache_stride_b + pid_hk * kv_cache_stride_h
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
                k_nope = (k_nope.to(tl.float32) * k_scale_rcprl).to(kv_cache_ptr.dtype.element_ty)
                k_pe = (k_pe.to(tl.float32) * k_scale_rcprl).to(kv_cache_ptr.dtype.element_ty)
                tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope)
                tl.store(kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d, k_pe)
    else:
        pid = pid - B * QH + B * KH
        if pid < B_slot * KH:
            pid_b = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                
                k_nope_ptrs = (
                    k_nope_ptr 
                    + pid_b * k_nope_stride_b 
                    + pid_hk * k_nope_stride_h 
                    + dk_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = k_pe_ptr + pid_b * k_pe_stride_b + pid_hk * k_pe_stride_h + d_pe_offs * k_pe_stride_d
                k_pe_out_ptrs = k_pe_out_ptr + pid_b * k_pe_out_stride_b + pid_hk * k_pe_out_stride_h + d_pe_offs * k_pe_out_stride_d
                kv_cache_ptrs = kv_cache_ptr + pid_slot * kv_cache_stride_b + pid_hk * kv_cache_stride_h
                k_nope = tl.load(k_nope_ptrs)
                k_pe = tl.load(k_pe_ptrs)
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = (k_nope.to(tl.float32) * k_scale_rcprl).to(kv_cache_ptr.dtype.element_ty)
                k_pe = (k_pe.to(tl.float32) * k_scale_rcprl).to(kv_cache_ptr.dtype.element_ty)
                tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope)
                tl.store(kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d, k_pe)


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
    num_decode_toks_for_zeros: int = 0,
    apply_scale: bool = True,
    q_out: torch.Tensor = None,
    decode_q_pe_out: torch.Tensor = None,
    k_pe_out: torch.Tensor = None,
    q_out_dtype=None,
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
    bk, kh, dk_nope = k_nope.shape
    bk2, kh2, dk2 = k_pe.shape
    b_cache, h_cache, d_cache = kv_cache.shape
    (b_slot, ) = slot_mapping.shape

    assert (
        b_slot <= b and b == b2 == bk == bk2
    ), "batch dimension should be identical for q_nope, q_pe, k_nope, and k_pe, and the batch dimeion of slot_mapping should be no more than that of q_nope, q_pe, k_nope, and k_pe"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2 == h_cache, "K head should be identical"
    assert d_pe == dk2, "D dimension of q_pe and k_pe should be identical"
    assert (
        dk_nope + dk2 == d_cache
    ), "D dimension of k_nope and k_pe should be summed up to be the D dimension of kv_cache"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d_pe // 2) or (
        d_freq == d_pe
    ), "cos/sin last dim should be the same or half of the qk last dim"
    if isinstance(k_scale, torch.Tensor):
        assert k_scale.numel() == 1, "k_scale should be a single-element torch.Tensor"
    reuse_freqs_front_part = d_freq == d_pe // 2

    if q_out is None:
        q_out = torch.empty(
            (b, qh, d_nope + d_pe),
            dtype=q_out_dtype if q_out_dtype is not None else q_nope.dtype,
            device=q_nope.device,
        )
    else:
        b_q_out, qh_q_out, d_q_out = q_out.shape
        assert b == b_q_out and qh == qh_q_out and d_nope + d_pe == d_q_out, "q_out shape mismatch"
    
    if decode_q_pe_out is None:
        decode_q_pe_out = torch.empty(
            (num_decode_toks_for_zeros, qh, d_pe),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
    else:
        b_decode_q_pe_out, qh_decode_q_pe_out, d_decode_q_pe_out = decode_q_pe_out.shape
        assert num_decode_toks_for_zeros == b_decode_q_pe_out and qh == qh_decode_q_pe_out and d_pe == d_decode_q_pe_out, "decode_q_pe_out shape mismatch"

    if k_pe_out is None:
        k_pe_out = torch.empty(
            (b, kh, d_pe),
            dtype=k_pe.dtype,
            device=k_pe.device
        )
    else:
        b_k_pe_out, hk_k_pe_out, d_k_pe_out = k_pe_out.shape
        assert b == b_k_pe_out and kh == hk_k_pe_out and d_pe == d_k_pe_out, "k_pe_out shape mismatch"

    q_nope_zeros_out = None
    if num_decode_toks_for_zeros > 0:
        q_nope_zeros_out = torch.empty(
            (num_decode_toks_for_zeros, qh, dk_nope), dtype=q_nope.dtype, device=q_nope.device
        )

    n_pid = b * qh + (b_slot - b) * kh
    grid = (n_pid, 1, 1)
    _fused_qk_rope_cat_and_cache_mla_kernel[grid](
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        pos,
        cos,
        sin,
        q_out,
        decode_q_pe_out,
        k_pe_out,
        q_nope_zeros_out,
        kv_cache,
        slot_mapping,
        b,
        b_slot,
        num_decode_toks_for_zeros,
        *q_nope.stride(),
        *q_pe.stride(),
        *k_nope.stride(),
        *k_pe.stride(),
        pos.stride(0),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *decode_q_pe_out.stride(),
        *k_pe_out.stride(),
        q_nope_zeros_out.stride(0) if q_nope_zeros_out is not None else 0,
        q_nope_zeros_out.stride(1) if q_nope_zeros_out is not None else 0,
        q_nope_zeros_out.stride(2) if q_nope_zeros_out is not None else 0,
        *kv_cache.stride(),
        k_scale_ptr=k_scale,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_nope=d_nope,
        BLOCK_DK_nope=dk_nope,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
        OUTPUT_Q_NOPE_ZEROS=(q_nope_zeros_out is not None),
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        num_warps=1,
    )

    if num_decode_toks_for_zeros > 0:
        return q_out, decode_q_pe_out, k_pe_out, kv_cache, q_nope_zeros_out
    return q_out, decode_q_pe_out, k_pe_out, kv_cache


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
    FLASH_LAYOUT: tl.constexpr,
    HAVE_POS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
    HAVE_ZEROS: tl.constexpr = False,
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


def fused_qk_rope_reshape_and_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
    k_out: torch.Tensor = None,
    output_zeros: bool = True,
    zeros_out: torch.Tensor = None,
):
    """
    Perform RoPE on q and k and along the last dimension and copy k and v in to key_cache and value_cache inplace

    Key parameters:
    - q: shape (T, QH, D).
    - k: shape (T_slot, KH, D).
    - v: shape (T_slot, KH, D).
    - if flash_layout:
    -     key_cache: shape (T_cache, block_size, KH, D).
    -     value_cache: shape (T_cache, block_size, KH, D).
    - else:
    -     key_cache: shape (T_cache, KH, D // x, block_size, x).
    -     value_cache: shape (T_cache, KH, D, block_size).
    - slot_mapping: shape (T_slot, ).

    T is the number of decode tokens, T_cahce * block_size is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: same shape as input q.
    - k_out: same shape as input k.
    - key_cache: same shape as input key_cache (inplace).
    - value_cache: same shape as input value_cache (inplace).
    - zeros_out: same shape as input q.
    """
    _LOGGER.info(
        f"FUSED_QK_ROPE_RESHAPE_AND_CACHE: q={tuple(q.shape)} k={tuple(k.shape)} "
        + f"pos={tuple(pos.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} key_cache={tuple(key_cache.shape)} value_cache={tuple(value_cache.shape)} slot_mapping={tuple(slot_mapping.shape)}"
    )

    t, qh, d = q.shape
    tk, kh, dk = k.shape
    tv, vh, dv = v.shape
    if flash_layout:
        t_cache, block_size, kh_cache, dk_cache = key_cache.shape
        t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape
    else:
        t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape
        t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape
    (t_slot,) = slot_mapping.shape

    assert (
        t == tk == tv and t_slot <= tk
    ), f"Number of tokens should be identical for q, kand v. The number of tokens of slot_mapping should no more than that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"
    assert (
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"
    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"
    assert (
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"
    if flash_layout:
        assert (
            d == dk == dv == dk_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
    else:
        assert (
            d == dk == dv == dkx_cache * x_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
        assert x_cache == triton.next_power_of_2(x_cache), "x_size should be power of 2"

    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"
    assert block_size == triton.next_power_of_2(
        block_size
    ), "block_size should be power of 2"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d // 2) or (
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d // 2

    if q_out is None:
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)

    if k_out is None:
        k_out = torch.empty((tk, kh, dk), dtype=k.dtype, device=q.device)

    if zeros_out is not None:
        tz, qhz, dz = zeros_out.shape
        assert (
            t == tz and qh == qhz and d == dz
        ), f"q and zeros shape mismatch {q.shape=} {zeros_out.shape=}"
        output_zeros = True
    elif output_zeros:
        zeros_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)
    else:
        zeros_out = None

    n_pid = t * qh + (t_slot - t) * kh
    grid = (n_pid, 1, 1)
    _fused_qk_rope_reshape_and_cache_kernel[grid](
        q,
        k,
        v,
        pos,
        cos,
        sin,
        offs,
        key_cache,
        value_cache,
        slot_mapping,
        q_out,
        k_out,
        zeros_out,
        t,
        t_slot,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *k_out.stride(),
        key_cache.stride(0) if not flash_layout else key_cache.stride(0),
        key_cache.stride(1) if not flash_layout else key_cache.stride(2),
        key_cache.stride(2) if not flash_layout else key_cache.stride(3),
        key_cache.stride(3) if not flash_layout else key_cache.stride(1),
        key_cache.stride(4) if not flash_layout else 0,
        value_cache.stride(0) if not flash_layout else value_cache.stride(0),
        value_cache.stride(1) if not flash_layout else value_cache.stride(2),
        value_cache.stride(2) if not flash_layout else value_cache.stride(3),
        value_cache.stride(3) if not flash_layout else value_cache.stride(1),
        zeros_out.stride(0) if zeros_out is not None else 0,
        zeros_out.stride(1) if zeros_out is not None else 0,
        zeros_out.stride(2) if zeros_out is not None else 0,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_pe=d,
        BLOCK_D_HALF_pe=d // 2,
        BLOCK_SIZE=block_size,
        X_SIZE=x_cache if not flash_layout else 0,
        FLASH_LAYOUT=flash_layout,
        HAVE_POS=(offs is not None),
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        HAVE_V_SCALE=(v_scale is not None and apply_scale),
        HAVE_ZEROS=output_zeros,
        num_warps=1,
    )

    if zeros_out is not None:
        return q_out, k_out, key_cache, value_cache, zeros_out
    return q_out, k_out, key_cache, value_cache
