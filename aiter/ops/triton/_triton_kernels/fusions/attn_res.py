# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


# num_warps / num_stages / BL come from the static per-token-count table in the
# wrapper (see _pick_attn_res_config), not from @triton.autotune: a single config
# per shape keeps compile cost bounded and does not break CUDAGraph capture.
@triton.jit(do_not_specialize=["L"])
def attnres_fwd_kernel(
    q,
    res,
    w,
    ow,
    o,
    o_pre,
    rstd,
    logit,
    lse,
    res_packed,
    prefix,
    add_hidden,
    prefix_out,
    N,
    L,
    stride_res_n,
    stride_res_l,
    L2: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    SAVE_OPRE: tl.constexpr,
    SAVE_STATS: tl.constexpr,
    IS_PACKED: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
    DO_ADD: tl.constexpr,
    WRITE_PREF: tl.constexpr,
    HAS_W: tl.constexpr,
):
    """AttnRes forward, ported from fla 0.5.2 ``attnres_fwd_kernel``.

    Per token: RMS-normalize each of the L residual candidates, score each with a
    dot product against ``q * w``, softmax over the candidate axis, and return the
    weighted sum (optionally output-RMSNorm'd). One program per row; the whole D
    stays in registers (``BD = next_pow2(D)``) and the candidate axis is streamed
    in ``BL`` tiles with an online softmax, so each candidate tile is read once.

    Two residual layouts share the same math via ``IS_PACKED``:

    * ``IS_PACKED=False`` (sequence): ``res`` is a length-``L2`` tuple of the
      per-source ``[N, D]`` base pointers. fla gathers row ``l`` with a
      tensor-of-pointers ``tl.where`` chain, but that fails to compile on the AMD
      backend (CanonicalizePointers), so we scan the padded slots with a scalar
      base pointer per slot and keep the matching row via a mask -- same result.
    * ``IS_PACKED=True`` (packed): ``res_packed`` is one contiguous ``[N, L, D]``
      tensor read with row strides. ``HAS_PREFIX`` makes the LAST candidate come
      from a separate ``prefix`` row (so ``res_packed`` holds only the first L-1),
      ``DO_ADD`` folds the caller's ``prefix += add_hidden`` on-load, and
      ``WRITE_PREF`` stores the summed prefix back for downstream reuse. ``HAS_W``
      off means ``q`` already carries the folded ``query * rms_weight`` product.

    ``o_pre`` / ``rstd`` / ``logit`` / ``lse`` are the fla backward checkpoint; this
    is a forward-only port, so ``SAVE_OPRE`` / ``SAVE_STATS`` are off and those
    pointers are passed as ``None``. The parameters are kept so a backward kernel
    can be reintroduced by flipping the flags without changing the launch surface.
    """
    i_n = tl.program_id(0).to(tl.int64)

    # [BD]
    o_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    m_d = o_d < D
    # [BD] q * w, reused across all candidate tiles. HAS_W off => q is pre-folded.
    b_qw = tl.load(q + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_W:
        b_qw *= tl.load(w + o_d, mask=m_d, other=0.0).to(tl.float32)

    # Packed prefix candidate is a single [D] row; load (and fold the add) once.
    if IS_PACKED and HAS_PREFIX:
        n_res = L - 1  # res_packed holds the first L-1 rows; prefix is candidate L-1
        ps = tl.load(prefix + i_n * D + o_d, mask=m_d, other=0.0).to(tl.float32)
        if DO_ADD:
            ps += tl.load(add_hidden + i_n * D + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        if WRITE_PREF:
            tl.store(
                prefix_out + i_n * D + o_d,
                ps.to(prefix_out.dtype.element_ty),
                mask=m_d,
            )
    else:
        n_res = L

    # online softmax over L; b_o accumulates in registers so each v tile is read once
    b_m = tl.full([], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([], dtype=tl.float32)
    b_o = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        # [BL]
        o_l = i_l * BL + tl.arange(0, BL)
        m_l = o_l < L

        # [BL, BD] candidate tile
        if IS_PACKED:
            if HAS_PREFIX:
                m_res = o_l < n_res
                l_safe = tl.minimum(o_l, tl.maximum(n_res - 1, 0))
            else:
                m_res = m_l
                # Clamp padded lanes (o_l >= L when L is not a power of 2) to the
                # last valid row so their address stays in-bounds; they are masked
                # out anyway. Avoids the buffer-ops OOB sentinel (0x80000000) path
                # that faults on gfx1250 when the drop is not honored.
                l_safe = tl.minimum(o_l, L - 1)
            b_v = tl.load(
                res_packed
                + i_n * stride_res_n
                + l_safe[:, None] * stride_res_l
                + o_d[None, :],
                mask=m_res[:, None] & m_d[None, :],
                other=0.0,
            ).to(tl.float32)
            if HAS_PREFIX:
                # broadcast the prefix row into the last candidate while in-reg
                b_v = tl.where((o_l == n_res)[:, None], ps[None, :], b_v)
        else:
            # AMD-safe gather: scan the padded slots, mask in the matching row.
            b_v = tl.zeros([BL, BD], dtype=tl.float32)
            for i in tl.static_range(0, L2):
                b_v += tl.load(
                    tl.multiple_of(res[i] + (i_n * D + o_d[None, :]), (1, 16)),
                    mask=(o_l == i)[:, None] & m_l[:, None] & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)

        # [BL] per-candidate RMSNorm + logit
        b_rstd = tl.rsqrt(tl.sum(b_v * b_v, axis=1) / D + eps)
        b_logit = tl.sum(b_v * b_qw[None, :], axis=1) * b_rstd
        b_s = tl.where(m_l, b_logit * scale, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=0)), b_m
        b_r = tl.exp(b_mp - b_m)
        # [BL]
        b_p = tl.exp(b_s - b_m)
        b_acc = b_acc * b_r + tl.sum(b_p, axis=0)
        # [BD]
        b_o = b_o * b_r + tl.sum(b_p[:, None] * b_v, axis=0)

        # rstd and logit are the fla bwd_dv checkpoint; off in this forward-only port
        if SAVE_STATS:
            p_rstd = rstd + i_n + o_l * N
            p_logit = logit + i_n + o_l * N
            tl.store(p_rstd, b_rstd.to(rstd.dtype.element_ty), mask=m_l)
            tl.store(p_logit, b_logit.to(logit.dtype.element_ty), mask=m_l)

    if SAVE_STATS:
        tl.store(lse + i_n, b_m + tl.log(b_acc))

    # [BD] pre-norm mixed residual sum_l p_l * v_l
    b_o = b_o / b_acc
    if SAVE_OPRE:
        tl.store(o_pre + i_n * D + o_d, b_o.to(o_pre.dtype.element_ty), mask=m_d)
    # fold the optional output RMSNorm into the returned output o
    if HAS_ONORM:
        b_o_rstd = tl.rsqrt(tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0) / D + eps)
        b_ow = tl.load(ow + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_o = b_o * b_o_rstd * b_ow
    tl.store(o + i_n * D + o_d, b_o.to(o.dtype.element_ty), mask=m_d)
