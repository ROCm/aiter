# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.quant.mono_memory import (
    load_dev,
    store_dev_ht,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit(
    repr=make_kernel_repr(
        "_fused_dual_rmsnorm_mxfp8_quant_ps_kernel",
        ["BLOCK_SIZE_KQ", "BLOCK_SIZE_KK", "ROWS_PER_ITER", "USE_BUFFER"],
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
    USE_BUFFER: gl.constexpr = True,
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
    tg = gl.program_id(1)
    for first in range(tg, M, NUM_PRGMS * ROWS_PER_ITER):
        gq = load_dev(gq_ptr, q_cols, q_cols < KQ).to(gl.float32)
        gk = load_dev(gk_ptr, k_cols, k_cols < KK).to(gl.float32)
        qr = first.to(gl.int64) + q_rows * NUM_PRGMS
        kr = first.to(gl.int64) + k_rows * NUM_PRGMS
        qm = (qr[:, None] < M) & (q_cols[None, :] < KQ)
        km = (kr[:, None] < M) & (k_cols[None, :] < KK)
        xq = gl.full((ROWS_PER_ITER, BLOCK_SIZE_KQ), 0, gl.float32, q_layout)
        xk = gl.full((ROWS_PER_ITER, BLOCK_SIZE_KK), 0, gl.float32, k_layout)
        # Rebase each row in 64 bits; buffer offsets contain only the column.
        for slot in gl.static_range(ROWS_PER_ITER):
            row = first.to(gl.int64) + slot * NUM_PRGMS
            qv = load_dev(
                q_ptr + row * stride_qm,
                q_cols.to(gl.int64) * stride_qn,
                (row < M) & (q_cols < KQ),
                USE_BUFFER,
            ).to(gl.float32)
            kv = load_dev(
                k_ptr + row * stride_km,
                k_cols.to(gl.int64) * stride_kn,
                (row < M) & (k_cols < KK),
                USE_BUFFER,
            ).to(gl.float32)
            xq = gl.where(q_rows[:, None] == slot, qv[None, :], xq)
            xk = gl.where(k_rows[:, None] == slot, kv[None, :], xk)
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
        scales = gl.reshape(scales, (ROWS_PER_ITER, groups))
        scale_layout: gl.constexpr = scales.type.layout
        sr = gl.arange(0, ROWS_PER_ITER, layout=gl.SliceLayout(1, scale_layout))
        sg = gl.arange(0, groups, layout=gl.SliceLayout(0, scale_layout))
        for slot in gl.static_range(ROWS_PER_ITER):
            row = first.to(gl.int64) + slot * NUM_PRGMS
            store_dev_ht(
                yq_ptr + row * stride_yqm,
                q_rows[:, None].to(gl.int64) * 0
                + q_cols[None, :].to(gl.int64) * stride_yqn,
                quantized.to(yq_ptr.dtype.element_ty),
                qm & (q_rows[:, None] == slot),
                USE_BUFFER,
            )
            store_dev_ht(
                sq_ptr + row * stride_sqm,
                sr[:, None].to(gl.int64) * 0 + sg[None, :].to(gl.int64) * stride_sqn,
                scales,
                (row < M)
                & (sr[:, None] == slot)
                & (sg[None, :] < KQ // QUANT_BLOCK_SIZE),
                USE_BUFFER,
            )
            store_dev_ht(
                yk_ptr + row * stride_ykm,
                k_rows[:, None].to(gl.int64) * 0
                + k_cols[None, :].to(gl.int64) * stride_ykn,
                yk.to(yk_ptr.dtype.element_ty),
                km & (k_rows[:, None] == slot),
                USE_BUFFER,
            )
