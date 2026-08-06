# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BH": BH}, num_warps=nw, num_stages=1)
        for BH in [512, 1024, 2048]
        for nw in [2, 4, 8]
    ],
    key=["L2", "D", "HAS_ONORM"],
)
@triton.jit(do_not_specialize=["L"])
def _attn_res_fwd_sequence_2pass_kernel(
    q,
    res,
    w,
    ow,
    o,
    o_pre,
    N,
    L,
    L2: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BH: tl.constexpr,
    NS: tl.constexpr,
    HAS_ONORM: tl.constexpr,
):
    """Sequence layout (L independent [N, D] tensors), two-pass over D.

    Only the per-source scalars stay resident (~100 VGPR), so occupancy is high
    and the kernel saturates HBM at large N. The residual is read twice: once
    for the reductions and once for the weighted sum.

    ``o_pre`` is an internal fp32 scratch used only when HAS_ONORM (pass 2 writes
    it, pass 3 reads it back); it is not produced for the caller.
    """
    i_n = tl.program_id(0).to(tl.int64)
    inv_d = 1.0 / D
    b_idx = tl.arange(0, L2)
    b_valid = b_idx < L

    # ---- PASS 1: per-source reductions over D ----
    acc_sq = tl.zeros([L2], dtype=tl.float32)
    acc_dot = tl.zeros([L2], dtype=tl.float32)
    for h0 in tl.range(0, D, BH, num_stages=NS):
        cols = h0 + tl.arange(0, BH)
        h_mask = cols < D
        qw = tl.load(q + cols, mask=h_mask, other=0.0).to(tl.float32) * tl.load(
            w + cols, mask=h_mask, other=0.0
        ).to(tl.float32)
        v = tl.zeros([L2, BH], dtype=tl.float32)
        # A tensor-of-pointers + select chain fails to compile on the AMD
        # backend (CanonicalizePointers), so scan the padded slots with a
        # scalar base pointer per slot and keep the matching row via a mask.
        for i in tl.static_range(0, L2):
            v += tl.load(
                tl.multiple_of(res[i] + (i_n * D + cols[None, :]), (1, 16)),
                mask=(b_idx == i)[:, None] & b_valid[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        acc_sq += tl.sum(v * v, axis=1)
        acc_dot += tl.sum(v * qw[None, :], axis=1)

    b_rstd = tl.rsqrt(acc_sq * inv_d + eps)
    b_logit = acc_dot * b_rstd
    b_s = tl.where(b_valid, b_logit * scale, float("-inf"))
    b_m = tl.max(b_s, axis=0)
    b_p = tl.exp(b_s - b_m)
    b_acc = tl.sum(b_p, axis=0)
    probs = b_p / b_acc

    # ---- PASS 2: o_pre = sum_l p_l v_l, tiled over D ----
    acc_o_sq = tl.zeros([], dtype=tl.float32)
    for h0 in tl.range(0, D, BH, num_stages=NS):
        cols = h0 + tl.arange(0, BH)
        h_mask = cols < D
        v = tl.zeros([L2, BH], dtype=tl.float32)
        for i in tl.static_range(0, L2):
            v += tl.load(
                tl.multiple_of(res[i] + (i_n * D + cols[None, :]), (1, 16)),
                mask=(b_idx == i)[:, None] & b_valid[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        o_blk = tl.sum(probs[:, None] * v, axis=0)
        if HAS_ONORM:
            tl.store(o_pre + i_n * D + cols, o_blk, mask=h_mask)  # fp32 scratch
            acc_o_sq += tl.sum(tl.where(h_mask, o_blk * o_blk, 0.0), axis=0)
        else:
            tl.store(o + i_n * D + cols, o_blk.to(o.dtype.element_ty), mask=h_mask)

    # ---- PASS 3 (onorm only): reload fp32 o_pre, apply output RMSNorm ----
    if HAS_ONORM:
        o_rstd = tl.rsqrt(acc_o_sq * inv_d + eps)
        for h0 in tl.range(0, D, BH, num_stages=NS):
            cols = h0 + tl.arange(0, BH)
            h_mask = cols < D
            opre = tl.load(o_pre + i_n * D + cols, mask=h_mask, other=0.0)
            owv = tl.load(ow + cols, mask=h_mask, other=0.0).to(tl.float32)
            tl.store(
                o + i_n * D + cols,
                (opre * o_rstd * owv).to(o.dtype.element_ty),
                mask=h_mask,
            )


@triton.autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=1) for nw in [8, 16]],
    key=[
        "L2",
        "D",
        "HAS_ONORM",
        "HAS_PREFIX",
        "DO_ADD",
        "HAS_W",
    ],
)
@triton.jit(do_not_specialize=["L"])
def _attn_res_fwd_packed_1pass_kernel(
    q,
    res,
    w,
    ow,
    o,
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
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
    DO_ADD: tl.constexpr,
    WRITE_PREF: tl.constexpr,
    HAS_W: tl.constexpr,
):
    """Packed layout (one contiguous [N, L, D] tensor), whole-row single pass.

    The entire v[L, D] tile is loaded ONCE into registers and reused for both
    the reduction and the weighted output, so the residual is read from HBM
    exactly once. Register pressure is high (~330 VGPR), but the coalesced
    packed load plus the single read win at short sequence lengths where the
    grid is program-starved and occupancy does not matter.

    HAS_PREFIX makes the LAST of the L candidates come from a separate [N, D]
    ``prefix`` tensor instead of from ``res`` (which then holds L-1 rows), so
    the caller does not have to materialize a concatenation. DO_ADD folds the
    caller's ``prefix = prefix + add_hidden`` into that on-load, and WRITE_PREF
    stores the summed prefix back for downstream reuse.
    """
    i_n = tl.program_id(0).to(tl.int64)
    inv_d = 1.0 / D
    b_idx = tl.arange(0, L2)
    b_valid = b_idx < L
    # With a prefix candidate the packed tensor only holds the first L-1 rows.
    if HAS_PREFIX:
        n_res = L - 1
    else:
        n_res = L
    m_res = b_idx < n_res
    b_safe = tl.minimum(b_idx, tl.maximum(n_res - 1, 0))

    cols = tl.arange(0, BD)
    m_d = cols < D
    # HAS_W=False means q already carries the folded query * rms_weight product.
    qw = tl.load(q + cols, mask=m_d, other=0.0).to(tl.float32)
    if HAS_W:
        qw *= tl.load(w + cols, mask=m_d, other=0.0).to(tl.float32)

    # whole-row load of the entire v[L, D] tile ONCE (reused for reduction + output)
    res_base = i_n * stride_res_n + b_safe * stride_res_l
    v = tl.load(
        res + res_base[:, None] + cols[None, :],
        mask=m_res[:, None] & m_d[None, :],
        other=0.0,
    ).to(tl.float32)

    if HAS_PREFIX:
        ps = tl.load(prefix + i_n * D + cols, mask=m_d, other=0.0).to(tl.float32)
        if DO_ADD:
            ps += tl.load(add_hidden + i_n * D + cols, mask=m_d, other=0.0).to(
                tl.float32
            )
        if WRITE_PREF:
            tl.store(
                prefix_out + i_n * D + cols,
                ps.to(prefix_out.dtype.element_ty),
                mask=m_d,
            )
        # ps broadcasts into the last candidate row while still in registers.
        v = tl.where((b_idx == n_res)[:, None], ps[None, :], v)

    acc_sq = tl.sum(v * v, axis=1)
    acc_dot = tl.sum(v * qw[None, :], axis=1)
    b_rstd = tl.rsqrt(acc_sq * inv_d + eps)
    b_logit = acc_dot * b_rstd
    b_s = tl.where(b_valid, b_logit * scale, float("-inf"))
    b_m = tl.max(b_s, axis=0)
    b_p = tl.exp(b_s - b_m)
    b_acc = tl.sum(b_p, axis=0)
    probs = b_p / b_acc

    b_o = tl.sum(probs[:, None] * v, axis=0)  # [BD] pre-norm mix
    if HAS_ONORM:
        o_rstd = tl.rsqrt(tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0) * inv_d + eps)
        owv = tl.load(ow + cols, mask=m_d, other=0.0).to(tl.float32)
        b_o = b_o * o_rstd * owv
    tl.store(o + i_n * D + cols, b_o.to(o.dtype.element_ty), mask=m_d)
