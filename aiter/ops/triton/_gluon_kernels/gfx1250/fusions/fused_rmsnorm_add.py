# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# RMSNorm of each row with an optional residual add: s = x + res (HAS_RES), out = s / rms(s) * w (1 + w with GEMMA).
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_REPR_KEYS = ["N", "BLOCK_M", "BLOCK_N", "WARPS_N", "HAS_RES", "GEMMA"]
_gluon_fused_rms_kernel_repr = make_kernel_repr("_gluon_fused_rms_kernel", _REPR_KEYS)
_gluon_fused_rms_tdm_kernel_repr = make_kernel_repr(
    "_gluon_fused_rms_tdm_kernel", _REPR_KEYS
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
    # BLOCK_M rows a program and WARPS_N waves a row. A lane loads 8 contiguous elements (16 bytes) of its row.
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

    # The sum of squares: in the wave, then across the WARPS_N waves of the row.
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
def _tdm_tile(
    ptr, M, stride_m, N: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr
):
    # A [BLOCK_M, BLOCK_N] TDM tile of an [M, N] tensor, and its LDS buffer.
    layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        ptr, [M, N], [stride_m, 1], [BLOCK_M, BLOCK_N], layout
    )
    smem = gl.allocate_shared_memory(ptr.dtype.element_ty, [BLOCK_M, BLOCK_N], layout)
    return desc, smem


# The same op with the tiles copied between HBM and LDS by TDM. The config selects it with USE_TDM.
@gluon.jit(repr=_gluon_fused_rms_tdm_kernel_repr)
def _gluon_fused_rms_tdm_kernel(
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
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [BLOCK_M, WARPS_N], [1, 0])
    w_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    row0 = gl.program_id(0) * BLOCK_M

    # The loads first: x, res, then the weight.
    x_desc, x_smem = _tdm_tile(x_ptr, M, stride_xm, N, BLOCK_M, BLOCK_N)
    gl.amd.gfx1250.tdm.async_load(x_desc, [row0, 0], x_smem)
    if HAS_RES:
        res_desc, res_smem = _tdm_tile(res_ptr, M, stride_resm, N, BLOCK_M, BLOCK_N)
        gl.amd.gfx1250.tdm.async_load(res_desc, [row0, 0], res_smem)
        out_res_desc, out_res_smem = _tdm_tile(
            out_res_ptr, M, stride_out_resm, N, BLOCK_M, BLOCK_N
        )
    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        w_ptr, [N], [1], [BLOCK_N], w_layout
    )
    w_smem = gl.allocate_shared_memory(w_ptr.dtype.element_ty, [BLOCK_N], w_layout)
    gl.amd.gfx1250.tdm.async_load(w_desc, [0], w_smem)
    out_desc, out_smem = _tdm_tile(out_ptr, M, stride_outm, N, BLOCK_M, BLOCK_N)

    # The TDM copies end in order: wait(1) leaves only the newest copy (the weight) in flight.
    gl.amd.gfx1250.tdm.async_wait(1)
    x = x_smem.load(layout).to(gl.float32)
    if HAS_RES:
        x = x + res_smem.load(layout).to(gl.float32)
        out_res_smem.store(x.to(out_res_ptr.dtype.element_ty))
        gl.amd.gfx1250.tdm.async_store(out_res_desc, [row0, 0], out_res_smem)
        # The weight is in LDS: only the residual-sum store can be in flight.
        gl.amd.gfx1250.tdm.async_wait(1)
    else:
        gl.amd.gfx1250.tdm.async_wait(0)
    w = w_smem.load(gl.SliceLayout(0, layout)).to(gl.float32)
    if GEMMA:
        w = w + 1.0

    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / N + eps)
    out = x * rstd[:, None] * w[None, :]
    out_smem.store(out.to(out_ptr.dtype.element_ty))
    gl.amd.gfx1250.tdm.async_store(out_desc, [row0, 0], out_smem)
    gl.amd.gfx1250.tdm.async_wait(0)
