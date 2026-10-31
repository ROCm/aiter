# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# RMSNorm + optional residual add.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gluon_fused_rms_kernel_repr = make_kernel_repr(
    "_gluon_fused_rms_kernel",
    ["N", "BLOCK_M", "BLOCK_N", "WARPS_N", "HAS_RES", "GEMMA"],
)


@gluon.jit(repr=_gluon_fused_rms_kernel_repr)
def _gluon_fused_rms_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    out_res_ptr,
    M,
    stride_xm,
    stride_resm,
    stride_outm,
    stride_out_resm,
    eps,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WARPS_N: gl.constexpr,
    HAS_RES: gl.constexpr,
    GEMMA: gl.constexpr,
):
    # BLOCK_M rows a program and WARPS_N waves a row.
    # A lane loads 8 contiguous elements (16 bytes) of its row.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [BLOCK_M, WARPS_N], [1, 0])
    pid_m = gl.program_id(0)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
    # No row mask: a row past M reads and writes row M - 1 again, with the same values.
    rows = gl.minimum(offs_m, M - 1)
    mask_n = offs_n < N
    mask = mask_n[None, :]

    x = gl.load(
        x_ptr + rows[:, None] * stride_xm + offs_n[None, :], mask=mask, other=0.0
    ).to(gl.float32)
    if HAS_RES:
        res = gl.load(
            res_ptr + rows[:, None] * stride_resm + offs_n[None, :],
            mask=mask,
            other=0.0,
        )
    w = gl.load(w_ptr + offs_n, mask=mask_n, other=0.0).to(gl.float32)

    # The residual sum is the next layer's input: store it before the reduction.
    if HAS_RES:
        x = x + res.to(gl.float32)
        out_res_ptrs = out_res_ptr + rows[:, None] * stride_out_resm + offs_n[None, :]
        gl.store(out_res_ptrs, x.to(out_res_ptr.dtype.element_ty), mask=mask)

    # The sum of squares: inside the wave, then across the WARPS_N waves of the row.
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / N + eps)
    if GEMMA:
        w = w + 1.0
    out = x * rstd[:, None] * w[None, :]
    gl.store(
        out_ptr + rows[:, None] * stride_outm + offs_n[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@gluon.jit
def _rmsnorm_op(row, weights, n_cols, epsilon):
    row_norm = row * row
    row_norm = gl.sum(row_norm, axis=-1, keep_dims=True)
    norm_factor = gl.rsqrt((row_norm / n_cols) + epsilon)
    return row * norm_factor * weights


_gluon_fused_rms_tdm_kernel_repr = make_kernel_repr(
    "_gluon_fused_rms_tdm_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "WARPS_N", "FIRST_INPUT_RES", "GEMMA"],
)


# The rows go through LDS with TDM copies. The config selects it with USE_TDM.
@gluon.jit(repr=_gluon_fused_rms_tdm_kernel_repr)
def _gluon_fused_rms_tdm_kernel(
    x1_ptr,
    w1_ptr,
    res1_ptr,
    out1_ptr,
    out_res1_ptr,
    eps1,
    M,
    N,
    x1_stride_m,
    res1_stride_m,
    out1_stride_m,
    out_res1_stride_m,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    WARPS_N: gl.constexpr,
    FIRST_INPUT_RES: gl.constexpr,
    GEMMA: gl.constexpr,
):
    start_pid = gl.program_id(0)

    gLayout2D: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 32], [BLOCK_SIZE_M, WARPS_N], [1, 0]
    )
    gLayoutN: gl.constexpr = gl.SliceLayout(0, gLayout2D)
    sharedLayout2D: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sharedLayoutN: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    # descriptors + smem for first input and its weight
    x1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x1_ptr, [M, N], [x1_stride_m, 1], [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
    )
    w1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        w1_ptr, [N], [1], [BLOCK_SIZE_N], sharedLayoutN
    )
    smemX1 = gl.allocate_shared_memory(
        x1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
    )
    smemW1 = gl.allocate_shared_memory(
        w1_ptr.dtype.element_ty, [BLOCK_SIZE_N], sharedLayoutN
    )

    # x1 load issued first for early latency hiding
    gl.amd.gfx1250.tdm.async_load(x1_desc, [start_pid * BLOCK_SIZE_M, 0], smemX1)

    # optional residual input
    if FIRST_INPUT_RES:
        res1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            res1_ptr,
            [M, N],
            [res1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
        smemRes1 = gl.allocate_shared_memory(
            res1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
        )
        gl.amd.gfx1250.tdm.async_load(
            res1_desc, [start_pid * BLOCK_SIZE_M, 0], smemRes1
        )
        out_res1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            out_res1_ptr,
            [M, N],
            [out_res1_stride_m, 1],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            sharedLayout2D,
        )
        smemOutRes1 = gl.allocate_shared_memory(
            out_res1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
        )

    gl.amd.gfx1250.tdm.async_load(w1_desc, [0], smemW1)

    # output descriptor + smem (static alloc; placement before wait is free)
    out1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        out1_ptr,
        [M, N],
        [out1_stride_m, 1],
        [BLOCK_SIZE_M, BLOCK_SIZE_N],
        sharedLayout2D,
    )
    smemOut1 = gl.allocate_shared_memory(
        out1_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], sharedLayout2D
    )

    gl.amd.gfx1250.tdm.async_wait(1)

    x1 = smemX1.load(gLayout2D).to(gl.float32)

    if FIRST_INPUT_RES:
        res1_loaded = smemRes1.load(gLayout2D).to(gl.float32)
        x1 = x1 + res1_loaded
        smemOutRes1.store(x1.to(out_res1_ptr.dtype.element_ty))
        gl.amd.gfx1250.tdm.async_store(
            out_res1_desc, [start_pid * BLOCK_SIZE_M, 0], smemOutRes1
        )
        gl.amd.gfx1250.tdm.async_wait(1)
    else:
        gl.amd.gfx1250.tdm.async_wait(0)

    w1 = smemW1.load(gLayoutN).to(gl.float32)
    if GEMMA:
        w1 = w1 + 1.0
    w1 = w1.reshape(1, BLOCK_SIZE_N)
    w1 = gl.convert_layout(w1, gLayout2D)
    norm1 = _rmsnorm_op(x1, w1, N, eps1)

    smemOut1.store(norm1.to(out1_ptr.dtype.element_ty))
    gl.amd.gfx1250.tdm.async_store(out1_desc, [start_pid * BLOCK_SIZE_M, 0], smemOut1)

    gl.amd.gfx1250.tdm.async_wait(0)
