# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernel: the gates of a delayed mHC seam from split-K partials.

The input is the split-K output of the mHC pre projection of the new residual R'
(T tokens, 4 streams, K = 4 * H):

    gemm   [S, T, >= 24] fp32, last dim contiguous: partial sums of flatten(R') @ fn^T
    sqrsum [S, T] fp32 (any strides): partial sums of R'^2

``mhc_pre_gemm_sqrsum`` writes this layout, and so does the gfx942 seam kernel
(``mhc_fused_post_pre_delayed``), which stores its sum of squares in column 24 of the
same rows. The gates are those of ``mhc_pre`` / ``mhc_pre_big_fuse``:

    rstd   = rsqrt(sum(sqrsum) / K + rms_eps)
    pre'   = sigmoid(mixes[0:4]  * rstd * s0 + b) + hc_pre_eps
    post'  = sigmoid(mixes[4:8]  * rstd * s1 + b) * hc_post_mult
    comb'  = Sinkhorn(mixes[8:24] * rstd * s2 + b): row softmax + eps, / (colsum + eps),
             then sinkhorn_repeat - 1 rounds of / (rowsum + eps), / (colsum + eps)

It replaces ``mhc_pre_big_fuse`` on the delayed path, whose collapse the delayed seam
discards, and the separate pre gate. Reciprocals are v_rcp_f32 (1 ulp), as in big_fuse.

A program owns BLOCK_T tokens x SPLIT_LANES split lanes: each lane accumulates every
SPLIT_LANES-th split and the lanes are reduced once (a serial chain over ~160 splits per
token costs ~90 us at small T). After that one lane holds one token, and the 24 mixes are
24 [BLOCK_T] vectors, so softmax and the Sinkhorn row / column sums are elementwise.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _rcp(x):
    # v_rcp_f32 (1 ulp), as big_fuse's Sinkhorn loop; IEEE division is a ~10 instruction
    # dependent sequence and the gates are one long dependent chain per token.
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _sigmoid(x):
    return _rcp(1.0 + tl.exp(-x))


@triton.jit
def _norm4(a, b, c, d, eps):
    r = _rcp(a + b + c + d + eps)
    return a * r, b * r, c * r, d * r


@triton.jit
def _softmax4(a, b, c, d, eps):
    m = tl.maximum(tl.maximum(a, b), tl.maximum(c, d))
    a, b, c, d = tl.exp(a - m), tl.exp(b - m), tl.exp(c - m), tl.exp(d - m)
    r = _rcp(a + b + c + d)
    return a * r + eps, b * r + eps, c * r + eps, d * r + eps


_mhc_delayed_gates_kernel_repr = make_kernel_repr(
    "_mhc_delayed_gates_kernel", ["REPEAT", "BLOCK_T", "SPLIT_LANES"]
)


@triton.jit(repr=_mhc_delayed_gates_kernel_repr)
def _mhc_delayed_gates_kernel(
    gemm_ptr,  # (S, T, >= 24) fp32, last dim contiguous
    sqrsum_ptr,  # (S, T) fp32
    scale_ptr,  # (3,) fp32
    base_ptr,  # (24,) fp32
    post_ptr,  # (T, 4) fp32
    comb_ptr,  # (T, 4, 4) fp32
    pre_ptr,  # (T, 4) fp32
    T,
    S,
    stride_gs,
    stride_gt,
    stride_ss,
    stride_st,
    inv_k,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult,
    REPEAT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SPLIT_LANES: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    tm = t < T
    t2 = t.to(tl.int64)[:, None]
    lane = tl.arange(0, SPLIT_LANES)[None, :]
    z = tl.zeros((BLOCK_T, SPLIT_LANES), dtype=tl.float32)
    m0, m1, m2, m3, m4, m5, m6, m7 = z, z, z, z, z, z, z, z
    c00, c01, c02, c03, c10, c11, c12, c13 = z, z, z, z, z, z, z, z
    c20, c21, c22, c23, c30, c31, c32, c33 = z, z, z, z, z, z, z, z
    sq = z
    for s0 in range(0, S, SPLIT_LANES):
        s = s0 + lane
        mk = tm[:, None] & (s < S)
        p = gemm_ptr + s * stride_gs + t2 * stride_gt
        m0 += tl.load(p + 0, mask=mk, other=0.0)
        m1 += tl.load(p + 1, mask=mk, other=0.0)
        m2 += tl.load(p + 2, mask=mk, other=0.0)
        m3 += tl.load(p + 3, mask=mk, other=0.0)
        m4 += tl.load(p + 4, mask=mk, other=0.0)
        m5 += tl.load(p + 5, mask=mk, other=0.0)
        m6 += tl.load(p + 6, mask=mk, other=0.0)
        m7 += tl.load(p + 7, mask=mk, other=0.0)
        c00 += tl.load(p + 8, mask=mk, other=0.0)
        c01 += tl.load(p + 9, mask=mk, other=0.0)
        c02 += tl.load(p + 10, mask=mk, other=0.0)
        c03 += tl.load(p + 11, mask=mk, other=0.0)
        c10 += tl.load(p + 12, mask=mk, other=0.0)
        c11 += tl.load(p + 13, mask=mk, other=0.0)
        c12 += tl.load(p + 14, mask=mk, other=0.0)
        c13 += tl.load(p + 15, mask=mk, other=0.0)
        c20 += tl.load(p + 16, mask=mk, other=0.0)
        c21 += tl.load(p + 17, mask=mk, other=0.0)
        c22 += tl.load(p + 18, mask=mk, other=0.0)
        c23 += tl.load(p + 19, mask=mk, other=0.0)
        c30 += tl.load(p + 20, mask=mk, other=0.0)
        c31 += tl.load(p + 21, mask=mk, other=0.0)
        c32 += tl.load(p + 22, mask=mk, other=0.0)
        c33 += tl.load(p + 23, mask=mk, other=0.0)
        sq += tl.load(sqrsum_ptr + s * stride_ss + t2 * stride_st, mask=mk, other=0.0)
    m0, m1, m2, m3 = tl.sum(m0, 1), tl.sum(m1, 1), tl.sum(m2, 1), tl.sum(m3, 1)
    m4, m5, m6, m7 = tl.sum(m4, 1), tl.sum(m5, 1), tl.sum(m6, 1), tl.sum(m7, 1)
    c00, c01, c02, c03 = tl.sum(c00, 1), tl.sum(c01, 1), tl.sum(c02, 1), tl.sum(c03, 1)
    c10, c11, c12, c13 = tl.sum(c10, 1), tl.sum(c11, 1), tl.sum(c12, 1), tl.sum(c13, 1)
    c20, c21, c22, c23 = tl.sum(c20, 1), tl.sum(c21, 1), tl.sum(c22, 1), tl.sum(c23, 1)
    c30, c31, c32, c33 = tl.sum(c30, 1), tl.sum(c31, 1), tl.sum(c32, 1), tl.sum(c33, 1)
    rstd = tl.rsqrt(tl.sum(sq, 1) * inv_k + rms_eps)
    s0 = tl.load(scale_ptr) * 1.0
    s1 = tl.load(scale_ptr + 1) * 1.0
    s2 = tl.load(scale_ptr + 2) * 1.0

    po = pre_ptr + t * 4
    tl.store(
        po + 0, _sigmoid(m0 * rstd * s0 + tl.load(base_ptr + 0)) + hc_pre_eps, mask=tm
    )
    tl.store(
        po + 1, _sigmoid(m1 * rstd * s0 + tl.load(base_ptr + 1)) + hc_pre_eps, mask=tm
    )
    tl.store(
        po + 2, _sigmoid(m2 * rstd * s0 + tl.load(base_ptr + 2)) + hc_pre_eps, mask=tm
    )
    tl.store(
        po + 3, _sigmoid(m3 * rstd * s0 + tl.load(base_ptr + 3)) + hc_pre_eps, mask=tm
    )
    po = post_ptr + t * 4
    tl.store(
        po + 0, _sigmoid(m4 * rstd * s1 + tl.load(base_ptr + 4)) * hc_post_mult, mask=tm
    )
    tl.store(
        po + 1, _sigmoid(m5 * rstd * s1 + tl.load(base_ptr + 5)) * hc_post_mult, mask=tm
    )
    tl.store(
        po + 2, _sigmoid(m6 * rstd * s1 + tl.load(base_ptr + 6)) * hc_post_mult, mask=tm
    )
    tl.store(
        po + 3, _sigmoid(m7 * rstd * s1 + tl.load(base_ptr + 7)) * hc_post_mult, mask=tm
    )

    b = base_ptr + 8
    k = rstd * s2
    eps = hc_sinkhorn_eps
    c00, c01, c02, c03 = _softmax4(
        c00 * k + tl.load(b + 0),
        c01 * k + tl.load(b + 1),
        c02 * k + tl.load(b + 2),
        c03 * k + tl.load(b + 3),
        eps,
    )
    c10, c11, c12, c13 = _softmax4(
        c10 * k + tl.load(b + 4),
        c11 * k + tl.load(b + 5),
        c12 * k + tl.load(b + 6),
        c13 * k + tl.load(b + 7),
        eps,
    )
    c20, c21, c22, c23 = _softmax4(
        c20 * k + tl.load(b + 8),
        c21 * k + tl.load(b + 9),
        c22 * k + tl.load(b + 10),
        c23 * k + tl.load(b + 11),
        eps,
    )
    c30, c31, c32, c33 = _softmax4(
        c30 * k + tl.load(b + 12),
        c31 * k + tl.load(b + 13),
        c32 * k + tl.load(b + 14),
        c33 * k + tl.load(b + 15),
        eps,
    )
    c00, c10, c20, c30 = _norm4(c00, c10, c20, c30, eps)
    c01, c11, c21, c31 = _norm4(c01, c11, c21, c31, eps)
    c02, c12, c22, c32 = _norm4(c02, c12, c22, c32, eps)
    c03, c13, c23, c33 = _norm4(c03, c13, c23, c33, eps)
    for _ in range(REPEAT - 1):
        c00, c01, c02, c03 = _norm4(c00, c01, c02, c03, eps)
        c10, c11, c12, c13 = _norm4(c10, c11, c12, c13, eps)
        c20, c21, c22, c23 = _norm4(c20, c21, c22, c23, eps)
        c30, c31, c32, c33 = _norm4(c30, c31, c32, c33, eps)
        c00, c10, c20, c30 = _norm4(c00, c10, c20, c30, eps)
        c01, c11, c21, c31 = _norm4(c01, c11, c21, c31, eps)
        c02, c12, c22, c32 = _norm4(c02, c12, c22, c32, eps)
        c03, c13, c23, c33 = _norm4(c03, c13, c23, c33, eps)
    po = comb_ptr + t * 16
    tl.store(po + 0, c00, mask=tm)
    tl.store(po + 1, c01, mask=tm)
    tl.store(po + 2, c02, mask=tm)
    tl.store(po + 3, c03, mask=tm)
    tl.store(po + 4, c10, mask=tm)
    tl.store(po + 5, c11, mask=tm)
    tl.store(po + 6, c12, mask=tm)
    tl.store(po + 7, c13, mask=tm)
    tl.store(po + 8, c20, mask=tm)
    tl.store(po + 9, c21, mask=tm)
    tl.store(po + 10, c22, mask=tm)
    tl.store(po + 11, c23, mask=tm)
    tl.store(po + 12, c30, mask=tm)
    tl.store(po + 13, c31, mask=tm)
    tl.store(po + 14, c32, mask=tm)
    tl.store(po + 15, c33, mask=tm)
