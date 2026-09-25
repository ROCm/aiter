# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels for the delayed ("shifted") mHC seam of DeepSeek-V4.1.

For T tokens and hc_mult = 4 residual streams (h = source stream, j = destination
stream, y = the sub-layer output just produced), one seam computes

    R'_j   = post_j * y + sum_h comb[h][j] * R_h            -> bf16, new residual
    mixes  = flatten(R') @ fn^T, rstd = rsqrt(mean(R'^2) + rms_eps)
    pre'   = sigmoid(mixes[0:4]  * rstd * s0 + b) + hc_pre_eps
    post'  = sigmoid(mixes[4:8]  * rstd * s1 + b) * hc_post_mult
    comb'  = Sinkhorn(mixes[8:24] * rstd * s2 + b)
    x1     = sum_j pre_j * R'_j                             -> bf16 (stream collapse)
    out    = x1 * rsqrt(mean(x1^2) + norm_eps) * w          -> bf16 (the next block's input)

post_j, comb[h][j] and pre_j were produced by the previous seam; pre', post' and
comb' are consumed by the next one.

Two launches. The main kernel is a split-K GEMM over the hidden dimension: a program
owns 16 tokens and a slice of columns of all four streams, post-mixes them in fp32,
stores the bf16 residual, writes the collapse un-normalised while the tile is in
registers, and accumulates its slice of the projection (fp32 MFMA) and of the two
square sums into one partial row per (token, slice). The reduce kernel finishes one
token per program: it sums the partial rows, computes the gates (Sinkhorn with the
hardware reciprocal) and applies the RMSNorm to the staged collapse in place.
"""

import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _split4(v):
    """The four slices of a (..., 4) tensor along its last axis (register renames, no data movement)."""
    a, b = tl.split(tl.reshape(v, v.shape[:-1] + [2, 2]))
    v0, v2 = tl.split(a)
    v1, v3 = tl.split(b)
    return v0, v1, v2, v3


_mhc_fused_post_pre_delayed_rmsnorm_main_kernel_repr = make_kernel_repr(
    "_mhc_fused_post_pre_delayed_rmsnorm_main_kernel",
    ["H", "BLOCK_M", "TILE_K", "NUM_KSPLIT", "HAS_POST"],
)


@triton.jit(repr=_mhc_fused_post_pre_delayed_rmsnorm_main_kernel_repr)
def _mhc_fused_post_pre_delayed_rmsnorm_main_kernel(
    residual_ptr,  # (T, 4, H) bf16 residual entering the seam
    x_ptr,  # (T, H) bf16 sub-layer output                   [HAS_POST]
    post_mix_ptr,  # (T, 4) fp32 post gate from the previous seam  [HAS_POST]
    comb_mix_ptr,  # (T, 4, 4) fp32 comb gate [src h, dst j]       [HAS_POST]
    pre_mix_ptr,  # (T, 4) fp32 pre gate from the previous seam
    fn_ptr,  # (24, 4*H) fp32 gate projection
    residual_out_ptr,  # (T, 4, H) bf16 new residual             [HAS_POST]
    layer_input_ptr,  # (T, H) bf16 collapse; the reduce kernel normalises it in place
    partial_ptr,  # (T, NUM_KSPLIT, 32) fp32: [0:24] projection, [24] sum R'^2, [25] sum x1^2
    T,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    HAS_POST: tl.constexpr,
):
    """Grid (cdiv(T, BLOCK_M), NUM_KSPLIT): program (pid_m, pid_k) owns BLOCK_M tokens and
    H / NUM_KSPLIT hidden columns of all four streams, walked in k-steps of TILE_K columns.

    Each k-step loads the four stream tiles and the x tile, post-mixes them in fp32,
    stores the bf16 residual and the collapse, and feeds one fp32 MFMA dot with the
    streams concatenated along K (k = 4 * column + stream).
    """
    tl.static_assert(BLOCK_M % 16 == 0, "BLOCK_M is the M dimension of the MFMA dot")
    tl.static_assert(H % (NUM_KSPLIT * TILE_K) == 0, "the k-loop has no column mask")
    K_LOOP: tl.constexpr = H // (NUM_KSPLIT * TILE_K)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    m_mask = rm < T
    m2 = m_mask[:, None]
    rj = tl.arange(0, 4)  # stream index
    rn = tl.arange(0, 32)  # projection output (24 real columns)
    n_mask = rn < 24
    res_row = rm * (4 * H)

    # gates of this program's tokens as (BLOCK_M,) vectors, one value per token
    # (pre: q_j, post: p_j, comb: c_hj with h = source stream, j = destination stream)
    q0, q1, q2, q3 = _split4(
        tl.load(pre_mix_ptr + rm[:, None] * 4 + rj[None, :], mask=m2, other=0.0)
    )
    if HAS_POST:
        p0, p1, p2, p3 = _split4(
            tl.load(post_mix_ptr + rm[:, None] * 4 + rj[None, :], mask=m2, other=0.0)
        )
        comb = tl.load(
            comb_mix_ptr + rm[:, None] * 16 + tl.arange(0, 16)[None, :],
            mask=m2,
            other=0.0,
        )
        # cj_j[t, h] = comb[t, h, j]
        cj0, cj1, cj2, cj3 = _split4(tl.reshape(comb, (BLOCK_M, 4, 4)))
        c00, c10, c20, c30 = _split4(cj0)
        c01, c11, c21, c31 = _split4(cj1)
        c02, c12, c22, c32 = _split4(cj2)
        c03, c13, c23, c33 = _split4(cj3)

    acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    sq_r = tl.zeros((BLOCK_M, TILE_K), dtype=tl.float32)
    sq_x = tl.zeros((BLOCK_M, TILE_K), dtype=tl.float32)
    for k in range(K_LOOP):
        rc = pid_k * (K_LOOP * TILE_K) + k * TILE_K + tl.arange(0, TILE_K)
        r0 = tl.load(
            residual_ptr + res_row[:, None] + 0 * H + rc[None, :], mask=m2, other=0.0
        )
        r1 = tl.load(
            residual_ptr + res_row[:, None] + 1 * H + rc[None, :], mask=m2, other=0.0
        )
        r2 = tl.load(
            residual_ptr + res_row[:, None] + 2 * H + rc[None, :], mask=m2, other=0.0
        )
        r3 = tl.load(
            residual_ptr + res_row[:, None] + 3 * H + rc[None, :], mask=m2, other=0.0
        )
        if HAS_POST:
            # post-mix in fp32, rounded to bf16 once; the rounded value feeds everything below
            xt = tl.load(x_ptr + rm[:, None] * H + rc[None, :], mask=m2, other=0.0).to(
                tl.float32
            )
            f0 = r0.to(tl.float32)
            f1 = r1.to(tl.float32)
            f2 = r2.to(tl.float32)
            f3 = r3.to(tl.float32)
            r0 = (
                p0[:, None] * xt
                + c00[:, None] * f0
                + c10[:, None] * f1
                + c20[:, None] * f2
                + c30[:, None] * f3
            ).to(tl.bfloat16)
            r1 = (
                p1[:, None] * xt
                + c01[:, None] * f0
                + c11[:, None] * f1
                + c21[:, None] * f2
                + c31[:, None] * f3
            ).to(tl.bfloat16)
            r2 = (
                p2[:, None] * xt
                + c02[:, None] * f0
                + c12[:, None] * f1
                + c22[:, None] * f2
                + c32[:, None] * f3
            ).to(tl.bfloat16)
            r3 = (
                p3[:, None] * xt
                + c03[:, None] * f0
                + c13[:, None] * f1
                + c23[:, None] * f2
                + c33[:, None] * f3
            ).to(tl.bfloat16)
            tl.store(
                residual_out_ptr + res_row[:, None] + 0 * H + rc[None, :], r0, mask=m2
            )
            tl.store(
                residual_out_ptr + res_row[:, None] + 1 * H + rc[None, :], r1, mask=m2
            )
            tl.store(
                residual_out_ptr + res_row[:, None] + 2 * H + rc[None, :], r2, mask=m2
            )
            tl.store(
                residual_out_ptr + res_row[:, None] + 3 * H + rc[None, :], r3, mask=m2
            )
        g0 = r0.to(tl.float32)
        g1 = r1.to(tl.float32)
        g2 = r2.to(tl.float32)
        g3 = r3.to(tl.float32)
        sq_r += g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3
        # collapse with the carried pre gate, staged un-normalised for the reduce kernel
        x1 = (
            q0[:, None] * g0 + q1[:, None] * g1 + q2[:, None] * g2 + q3[:, None] * g3
        ).to(tl.bfloat16)
        tl.store(layer_input_ptr + rm[:, None] * H + rc[None, :], x1, mask=m2)
        x1f = x1.to(tl.float32)
        sq_x += x1f * x1f
        # projection: A = the four stream tiles interleaved along K (k = 4*c + stream),
        # B = the matching fn columns; Triton loads fn straight into the MFMA B layout
        a = tl.reshape(tl.join(tl.join(g0, g2), tl.join(g1, g3)), (BLOCK_M, 4 * TILE_K))
        fn_t = tl.load(
            fn_ptr
            + rn[:, None, None] * (4 * H)
            + (rj * H)[None, :, None]
            + rc[None, None, :],
            mask=n_mask[:, None, None],
            other=0.0,
        )  # (32, 4, TILE_K) fp32
        b = tl.reshape(tl.permute(fn_t, (0, 2, 1)), (32, 4 * TILE_K))
        acc = tl.dot(a, tl.trans(b), acc=acc, input_precision="ieee")

    vals = tl.where(
        rn[None, :] == 24,
        tl.sum(sq_r, 1)[:, None],
        tl.where(rn[None, :] == 25, tl.sum(sq_x, 1)[:, None], acc),
    )
    tl.store(
        partial_ptr + (rm * NUM_KSPLIT + pid_k)[:, None] * 32 + rn[None, :],
        vals,
        mask=m2,
    )


_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel_repr = make_kernel_repr(
    "_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel",
    ["H", "NUM_KSPLIT", "NUM_SINKHORN_ITERS", "BLOCK_C"],
)


@triton.jit(repr=_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel_repr)
def _mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel(
    partial_ptr,  # (T, NUM_KSPLIT, 32) fp32
    scale_ptr,  # (3,) fp32
    base_ptr,  # (24,) fp32
    next_pre_ptr,  # (T, 4) fp32
    post_out_ptr,  # (T, 4) fp32
    comb_out_ptr,  # (T, 4, 4) fp32
    layer_input_ptr,  # (T, H) bf16, rescaled in place
    norm_w_ptr,  # (H,) norm weight
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult,
    norm_eps,
    H: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Grid (T,): one program per token."""
    tl.static_assert(H % BLOCK_C == 0, "the norm loop has no column mask")
    KSPLIT_POW2: tl.constexpr = triton.next_power_of_2(NUM_KSPLIT)
    t = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, KSPLIT_POW2)
    ks_mask = ks < NUM_KSPLIT
    row = (t * NUM_KSPLIT + ks) * 32  # this token's partial rows are contiguous
    r4 = tl.arange(0, 4)
    r16 = tl.arange(0, 16)

    pre_p = tl.sum(
        tl.load(
            partial_ptr + row[:, None] + r4[None, :], mask=ks_mask[:, None], other=0.0
        ),
        0,
    )
    post_p = tl.sum(
        tl.load(
            partial_ptr + row[:, None] + 4 + r4[None, :],
            mask=ks_mask[:, None],
            other=0.0,
        ),
        0,
    )
    comb_p = tl.sum(
        tl.load(
            partial_ptr + row[:, None] + 8 + r16[None, :],
            mask=ks_mask[:, None],
            other=0.0,
        ),
        0,
    )
    sq_r = tl.sum(tl.load(partial_ptr + row + 24, mask=ks_mask, other=0.0), 0)
    sq_x = tl.sum(tl.load(partial_ptr + row + 25, mask=ks_mask, other=0.0), 0)

    rstd = tl.math.rsqrt(sq_r * (1.0 / (4 * H)) + rms_eps)
    s0 = tl.load(scale_ptr)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    pre = tl.sigmoid(pre_p * rstd * s0 + tl.load(base_ptr + r4)) + hc_pre_eps
    tl.store(next_pre_ptr + t * 4 + r4, pre)
    post = tl.sigmoid(post_p * rstd * s1 + tl.load(base_ptr + 4 + r4)) * hc_post_mult
    tl.store(post_out_ptr + t * 4 + r4, post)

    # Sinkhorn in the reference's order: row softmax, + eps, / (colsum + eps), then
    # NUM_SINKHORN_ITERS - 1 rounds of / (rowsum + eps), / (colsum + eps); fast_dividef is the
    # hardware reciprocal (v_rcp_f32, 1 ulp) instead of the exact division sequence
    A = tl.reshape(comb_p * rstd * s2 + tl.load(base_ptr + 8 + r16), (4, 4))
    P = tl.exp(A - tl.max(A, axis=1)[:, None])
    P = libdevice.fast_dividef(P, tl.sum(P, axis=1)[:, None]) + hc_sinkhorn_eps
    P = libdevice.fast_dividef(P, tl.sum(P, axis=0)[None, :] + hc_sinkhorn_eps)
    for _ in range(NUM_SINKHORN_ITERS - 1):
        P = libdevice.fast_dividef(P, tl.sum(P, axis=1)[:, None] + hc_sinkhorn_eps)
        P = libdevice.fast_dividef(P, tl.sum(P, axis=0)[None, :] + hc_sinkhorn_eps)
    tl.store(comb_out_ptr + t * 16 + r16, tl.reshape(P, (16,)))

    rstd_n = tl.math.rsqrt(sq_x * (1.0 / H) + norm_eps)
    for c0 in range(0, H, BLOCK_C):
        rc = c0 + tl.arange(0, BLOCK_C)
        xp = layer_input_ptr + t * H + rc
        xv = tl.load(xp).to(tl.float32)
        w = tl.load(norm_w_ptr + rc).to(tl.float32)
        tl.store(xp, ((xv * rstd_n) * w).to(tl.bfloat16))
