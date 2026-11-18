# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _fused_dual_rmsnorm_mxfp8_quant_tile(
    q_ptr,
    k_ptr,
    gq,
    gk,
    yq_ptr,
    sq_ptr,
    yk_ptr,
    M,
    KQ,
    KK,
    stride_qm,
    stride_qn,
    stride_km,
    stride_kn,
    stride_yqm,
    stride_yqn,
    stride_sqm,
    stride_sqn,
    stride_ykm,
    stride_ykn,
    eps_q,
    eps_k,
    BLOCK_SIZE_KQ: gl.constexpr,
    BLOCK_SIZE_KK: gl.constexpr,
    QUANT_BLOCK_SIZE: gl.constexpr,
    first,
    AMAX_FLOOR: gl.constexpr = 0.0,
    ROWS_PER_ITER: gl.constexpr = 4,
):
    # Keep each row's native lane and wave mapping for exact RMS sums.
    q_layout: gl.constexpr = gl.BlockedLayout(
        [1, max(1, BLOCK_SIZE_KQ // 128)], [1, 32], [1, 4], [1, 0]
    )
    k_layout: gl.constexpr = gl.BlockedLayout(
        [1, max(1, BLOCK_SIZE_KK // 128)], [1, 32], [1, 4], [1, 0]
    )
    q_cols = gl.arange(0, BLOCK_SIZE_KQ, layout=gl.SliceLayout(0, q_layout))
    k_cols = gl.arange(0, BLOCK_SIZE_KK, layout=gl.SliceLayout(0, k_layout))
    q_rows = gl.arange(0, ROWS_PER_ITER, layout=gl.SliceLayout(1, q_layout))
    k_rows = gl.arange(0, ROWS_PER_ITER, layout=gl.SliceLayout(1, k_layout))
    qr = first.to(gl.int64) + q_rows
    kr = first.to(gl.int64) + k_rows
    qm = (qr[:, None] < M) & (q_cols[None, :] < KQ)
    km = (kr[:, None] < M) & (k_cols[None, :] < KK)
    xq = gl.load(
        q_ptr + qr[:, None] * stride_qm + q_cols[None, :] * stride_qn, qm, other=0
    ).to(gl.float32)
    xk = gl.load(
        k_ptr + kr[:, None] * stride_km + k_cols[None, :] * stride_kn, km, other=0
    ).to(gl.float32)
    nq = gl.rsqrt(gl.sum(xq * xq, 1) / KQ + eps_q)
    nk = gl.rsqrt(gl.sum(xk * xk, 1) / KK + eps_k)
    yq = xq * nq[:, None] * gq[None, :]
    yk = xk * nk[:, None] * gk[None, :]
    # Use A03's rounded-amax recipe, not A09's ceil-scale recipe.
    groups: gl.constexpr = BLOCK_SIZE_KQ // QUANT_BLOCK_SIZE
    grouped = gl.reshape(yq, (ROWS_PER_ITER * groups, QUANT_BLOCK_SIZE))
    amax = gl.max(gl.abs(grouped), 1, keep_dims=True)
    if AMAX_FLOOR > 0:
        amax = gl.maximum(amax, AMAX_FLOOR)
    bits = (amax.to(gl.int32, bitcast=True) + 0x200000).to(
        gl.uint32, bitcast=True
    ) & 0xFF800000
    exponent = gl.floor(gl.log2(bits.to(gl.float32, bitcast=True))) - 8
    exponent = gl.minimum(gl.maximum(exponent, -127), 127)
    scales = (exponent.to(gl.int32) + 127).to(gl.uint8)
    quantized = grouped * gl.exp2(-exponent)
    quantized = gl.reshape(quantized, (ROWS_PER_ITER, BLOCK_SIZE_KQ))
    gl.store(
        yq_ptr + qr[:, None] * stride_yqm + q_cols[None, :] * stride_yqn,
        quantized.to(yq_ptr.dtype.element_ty),
        qm,
    )
    scales = gl.reshape(scales, (ROWS_PER_ITER, groups))
    scale_layout: gl.constexpr = scales.type.layout
    sr = first.to(gl.int64) + gl.arange(
        0, ROWS_PER_ITER, layout=gl.SliceLayout(1, scale_layout)
    )
    sg = gl.arange(0, groups, layout=gl.SliceLayout(0, scale_layout))
    gl.store(
        sq_ptr + sr[:, None] * stride_sqm + sg[None, :] * stride_sqn,
        scales,
        (sr[:, None] < M) & (sg[None, :] < KQ // QUANT_BLOCK_SIZE),
    )
    gl.store(
        yk_ptr + kr[:, None] * stride_ykm + k_cols[None, :] * stride_ykn,
        yk.to(yk_ptr.dtype.element_ty),
        km,
    )


@gluon.jit(
    repr=make_kernel_repr(
        "_fused_dual_rmsnorm_mxfp8_quant_ps_kernel",
        ["BLOCK_SIZE_KQ", "BLOCK_SIZE_KK", "ROWS_PER_ITER"],
    )
)
def _fused_dual_rmsnorm_mxfp8_quant_ps_kernel(
    q_ptr,
    k_ptr,
    gq_ptr,
    gk_ptr,
    yq_ptr,
    sq_ptr,
    yk_ptr,
    M,
    KQ,
    KK,
    stride_qm,
    stride_qn,
    stride_km,
    stride_kn,
    stride_yqm,
    stride_yqn,
    stride_sqm,
    stride_sqn,
    stride_ykm,
    stride_ykn,
    eps_q,
    eps_k,
    BLOCK_SIZE_KQ: gl.constexpr,
    BLOCK_SIZE_KK: gl.constexpr,
    QUANT_BLOCK_SIZE: gl.constexpr,
    NUM_PRGMS: gl.constexpr,
    AMAX_FLOOR: gl.constexpr = 0.0,
    ROWS_PER_ITER: gl.constexpr = 4,
):
    # Keep each row's native lane and wave mapping for exact RMS sums.
    q_layout: gl.constexpr = gl.BlockedLayout(
        [1, max(1, BLOCK_SIZE_KQ // 128)], [1, 32], [1, 4], [1, 0]
    )
    k_layout: gl.constexpr = gl.BlockedLayout(
        [1, max(1, BLOCK_SIZE_KK // 128)], [1, 32], [1, 4], [1, 0]
    )
    q_cols = gl.arange(0, BLOCK_SIZE_KQ, layout=gl.SliceLayout(0, q_layout))
    k_cols = gl.arange(0, BLOCK_SIZE_KK, layout=gl.SliceLayout(0, k_layout))
    gq = gl.load(gq_ptr + q_cols, q_cols < KQ, other=0).to(gl.float32)
    gk = gl.load(gk_ptr + k_cols, k_cols < KK, other=0).to(gl.float32)
    full_rows = M - M % ROWS_PER_ITER
    for first in range(
        gl.program_id(0) * ROWS_PER_ITER, full_rows, NUM_PRGMS * ROWS_PER_ITER
    ):
        _fused_dual_rmsnorm_mxfp8_quant_tile(
            q_ptr,
            k_ptr,
            gq,
            gk,
            yq_ptr,
            sq_ptr,
            yk_ptr,
            M,
            KQ,
            KK,
            stride_qm,
            stride_qn,
            stride_km,
            stride_kn,
            stride_yqm,
            stride_yqn,
            stride_sqm,
            stride_sqn,
            stride_ykm,
            stride_ykn,
            eps_q,
            eps_k,
            BLOCK_SIZE_KQ,
            BLOCK_SIZE_KK,
            QUANT_BLOCK_SIZE,
            first,
            AMAX_FLOOR,
            ROWS_PER_ITER,
        )
    # Only the remaining rows take the single-row reduction path.
    if gl.program_id(0) < M % ROWS_PER_ITER:
        tail = full_rows + gl.program_id(0)
        _fused_dual_rmsnorm_mxfp8_quant_tile(
            q_ptr,
            k_ptr,
            gq,
            gk,
            yq_ptr,
            sq_ptr,
            yk_ptr,
            M,
            KQ,
            KK,
            stride_qm,
            stride_qn,
            stride_km,
            stride_kn,
            stride_yqm,
            stride_yqn,
            stride_sqm,
            stride_sqn,
            stride_ykm,
            stride_ykn,
            eps_q,
            eps_k,
            BLOCK_SIZE_KQ,
            BLOCK_SIZE_KK,
            QUANT_BLOCK_SIZE,
            tail,
            AMAX_FLOOR,
            1,
        )
