# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import mx_e8m0_scale
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.constexpr_function
def _quant_layout(BLOCK_SIZE_M, num_warps):
    # A warp tile is 8 rows x 64 cols. Put all warps on M if it fits, else on N.
    warps_m = num_warps if (BLOCK_SIZE_M // 8) >= num_warps else 1
    return gl.BlockedLayout(
        size_per_thread=[1, 8],  # N is contiguous; 8 elems give dwordx4 loads
        threads_per_warp=[8, 8],
        warps_per_cta=[warps_m, num_warps // warps_m],
        order=[1, 0],
    )


@gluon.jit
def _store_quant_tile(
    out_tensor,
    bs_e8m0,
    out_ptr,
    bs_ptr,
    stride_out_m_in,
    stride_out_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    pid_m,
    pid_n,
    M,
    n_out,
    n_scales,
    BLOCK_SIZE_M: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    NUM_QUANT_BLOCKS: gl.constexpr,
    EVEN_M_N: gl.constexpr,
):
    # Store one quantized tile and its e8m0 scales; n_out and n_scales are row extents.
    stride_out_m = gl.cast(stride_out_m_in, gl.int64)
    stride_out_n = gl.cast(stride_out_n_in, gl.int64)
    stride_bs_m = gl.cast(stride_bs_m_in, gl.int64)
    stride_bs_n = gl.cast(stride_bs_n_in, gl.int64)

    out_m_local = gl.arange(0, BLOCK_SIZE_M)
    out_n_local = gl.arange(0, OUT_BLOCK_N)
    out_block_ptr = (
        out_ptr
        + (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_out_m
        + (pid_n * OUT_BLOCK_N).to(gl.int64) * stride_out_n
    )
    out_offs = (
        out_m_local[:, None] * stride_out_m_in + out_n_local[None, :] * stride_out_n_in
    )
    if EVEN_M_N:
        gl.amd.cdna4.buffer_store(out_tensor, out_block_ptr, out_offs)
    else:
        out_mask = (pid_m * BLOCK_SIZE_M + out_m_local < M)[:, None] & (
            pid_n * OUT_BLOCK_N + out_n_local < n_out
        )[None, :]
        gl.amd.cdna4.buffer_store(out_tensor, out_block_ptr, out_offs, mask=out_mask)

    bs_m_local = gl.arange(0, BLOCK_SIZE_M)
    bs_n_local = gl.arange(0, NUM_QUANT_BLOCKS)
    bs_block_ptr = (
        bs_ptr
        + (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_bs_m
        + (pid_n * NUM_QUANT_BLOCKS).to(gl.int64) * stride_bs_n
    )
    bs_offs = (
        bs_m_local[:, None] * stride_bs_m_in + bs_n_local[None, :] * stride_bs_n_in
    )
    if EVEN_M_N:
        gl.amd.cdna4.buffer_store(bs_e8m0, bs_block_ptr, bs_offs)
    else:
        bs_mask = (pid_m * BLOCK_SIZE_M + bs_m_local < M)[:, None] & (
            pid_n * NUM_QUANT_BLOCKS + bs_n_local < n_scales
        )[None, :]
        gl.amd.cdna4.buffer_store(bs_e8m0, bs_block_ptr, bs_offs, mask=bs_mask)


@gluon.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
):
    """Quantize a bf16 tile to packed mxfp4 and e8m0 scales.
    Reduce over the even/odd split so amax keeps the split layout.
    """
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_grouped = x.reshape(
        BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2
    )
    evens, odds = gl.split(x_grouped)
    amax = gl.maximum(gl.abs(evens), gl.abs(odds)).to(gl.float32)
    amax = gl.max(amax, axis=-1, keep_dims=True)
    bs_e8m0 = mx_e8m0_scale(amax, 2)
    bs_e8m0 = bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

    x_fp4 = gl.amd.cdna4.scaled_downcast(x, bs_e8m0, "e2m1", axis=1)

    return x_fp4, bs_e8m0


_gluon_dynamic_mxfp4_quant_kernel_gfx950_repr = make_kernel_repr(
    "gluon_dynamic_mxfp4_quant_kernel_gfx950",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "NUM_ITER", "NUM_STAGES", "num_warps", "EVEN_M_N"],
)


@gluon.jit(repr=_gluon_dynamic_mxfp4_quant_kernel_gfx950_repr)
def gluon_dynamic_mxfp4_quant_kernel_gfx950(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    NUM_ITER: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    num_warps: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    EVEN_M_N: gl.constexpr,
):
    pid_m = gl.program_id(0)
    start_n = gl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = gl.cast(stride_x_m_in, gl.int64)
    stride_x_n = gl.cast(stride_x_n_in, gl.int64)

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    layout: gl.constexpr = _quant_layout(BLOCK_SIZE_M, num_warps)

    end_n = min(start_n + NUM_ITER, N)

    # Independent of pid_n: computed once so the load offset is loop-invariant.
    local_m = gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, layout))
    local_n = gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, layout))
    x_offs = local_m[:, None] * stride_x_m_in + local_n[None, :] * stride_x_n_in
    # Per-iteration pointer step along N.
    x_block_stride = BLOCK_SIZE_N * stride_x_n
    if not EVEN_M_N:
        x_offs_m = pid_m * BLOCK_SIZE_M + local_m

    # NUM_STAGES==1: plain loop. ==2: double-buffered, warp-pipelined loop.
    if NUM_STAGES == 1:
        for pid_n in range(start_n, end_n):
            x_block_ptr = (
                x_ptr
                + (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_x_m
                + (pid_n * BLOCK_SIZE_N).to(gl.int64) * stride_x_n
            )
            if EVEN_M_N:
                x = gl.amd.cdna4.buffer_load(x_block_ptr, x_offs, cache=".cg")
            else:
                x_offs_n = pid_n * BLOCK_SIZE_N + local_n
                x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
                x = gl.amd.cdna4.buffer_load(
                    x_block_ptr, x_offs, mask=x_mask, cache=".cg"
                )

            out_tensor, bs_e8m0 = _mxfp4_quant_op(
                x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
            )

            _store_quant_tile(
                out_tensor,
                bs_e8m0,
                x_fp4_ptr,
                bs_ptr,
                stride_x_fp4_m_in,
                stride_x_fp4_n_in,
                stride_bs_m_in,
                stride_bs_n_in,
                pid_m,
                pid_n,
                M,
                N // 2,
                (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N // 2,
                NUM_QUANT_BLOCKS,
                EVEN_M_N,
            )
    else:
        # Prologue: first iteration is always valid (grid guarantees start_n < end_n).
        pid_n = start_n
        x_block_ptr = (
            x_ptr
            + (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_x_m
            + (pid_n * BLOCK_SIZE_N).to(gl.int64) * stride_x_n
        )
        if EVEN_M_N:
            x = gl.amd.cdna4.buffer_load(x_block_ptr, x_offs, cache=".cg")
        else:
            x_offs_n = pid_n * BLOCK_SIZE_N + local_n
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = gl.amd.cdna4.buffer_load(x_block_ptr, x_offs, mask=x_mask, cache=".cg")

        for pid_n in range(start_n, end_n):
            # Separate load cluster so the backend can overlap it with compute+store.
            with gl.amd.warp_pipeline_stage("load", priority=1):
                # Scalar (warp-uniform) check, so no divergence.
                has_next = pid_n + 1 < end_n
                x_block_ptr_next = x_block_ptr + x_block_stride
                if EVEN_M_N:
                    if has_next:
                        x_next = gl.amd.cdna4.buffer_load(
                            x_block_ptr_next, x_offs, cache=".cg"
                        )
                    else:
                        x_next = x
                else:
                    x_offs_n_next = (pid_n + 1) * BLOCK_SIZE_N + local_n
                    x_mask_next = (
                        (x_offs_m < M)[:, None]
                        & (x_offs_n_next < N)[None, :]
                        & has_next
                    )
                    x_next = gl.amd.cdna4.buffer_load(
                        x_block_ptr_next, x_offs, mask=x_mask_next, cache=".cg"
                    )

            with gl.amd.warp_pipeline_stage("compute_store", priority=0):
                out_tensor, bs_e8m0 = _mxfp4_quant_op(
                    x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
                )
                _store_quant_tile(
                    out_tensor,
                    bs_e8m0,
                    x_fp4_ptr,
                    bs_ptr,
                    stride_x_fp4_m_in,
                    stride_x_fp4_n_in,
                    stride_bs_m_in,
                    stride_bs_n_in,
                    pid_m,
                    pid_n,
                    M,
                    N // 2,
                    (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N // 2,
                    NUM_QUANT_BLOCKS,
                    EVEN_M_N,
                )

            x = x_next
            x_block_ptr = x_block_ptr_next


@gluon.jit
def _mxfp8_quant_op(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP8_QUANT_BLOCK_SIZE: gl.constexpr,
):
    """Quantize a bf16 tile to fp8 e4m3 and e8m0 scales.
    The output is not packed, so a plain reshape and max works.
    """
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP8_QUANT_BLOCK_SIZE
    x_grouped = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP8_QUANT_BLOCK_SIZE)
    amax = gl.max(gl.abs(x_grouped), axis=-1, keep_dims=True).to(gl.float32)
    bs_e8m0 = mx_e8m0_scale(amax, 8)
    bs_e8m0 = bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

    x_fp8 = gl.amd.cdna4.scaled_downcast(x, bs_e8m0, "e4m3", axis=1)

    return x_fp8, bs_e8m0


_gluon_dynamic_mxfp8_quant_kernel_gfx950_repr = make_kernel_repr(
    "gluon_dynamic_mxfp8_quant_kernel_gfx950",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "NUM_ITER", "num_warps", "EVEN_M_N"],
)


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@gluon.jit(repr=_gluon_dynamic_mxfp8_quant_kernel_gfx950_repr)
def gluon_dynamic_mxfp8_quant_kernel_gfx950(
    x_ptr,
    x_fp8_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp8_m_in,
    stride_x_fp8_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    NUM_ITER: gl.constexpr,
    num_warps: gl.constexpr,
    MXFP8_QUANT_BLOCK_SIZE: gl.constexpr,
    EVEN_M_N: gl.constexpr,
):
    pid_m = gl.program_id(0)
    start_n = gl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = gl.cast(stride_x_m_in, gl.int64)
    stride_x_n = gl.cast(stride_x_n_in, gl.int64)

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP8_QUANT_BLOCK_SIZE
    layout: gl.constexpr = _quant_layout(BLOCK_SIZE_M, num_warps)

    end_n = min(start_n + NUM_ITER, N)

    local_m = gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, layout))
    local_n = gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, layout))
    x_offs = local_m[:, None] * stride_x_m_in + local_n[None, :] * stride_x_n_in
    if not EVEN_M_N:
        x_offs_m = pid_m * BLOCK_SIZE_M + local_m

    for pid_n in range(start_n, end_n):
        x_block_ptr = (
            x_ptr
            + (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_x_m
            + (pid_n * BLOCK_SIZE_N).to(gl.int64) * stride_x_n
        )
        if EVEN_M_N:
            x = gl.amd.cdna4.buffer_load(x_block_ptr, x_offs, cache=".cg")
        else:
            x_offs_n = pid_n * BLOCK_SIZE_N + local_n
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = gl.amd.cdna4.buffer_load(x_block_ptr, x_offs, mask=x_mask, cache=".cg")

        out_tensor, bs_e8m0 = _mxfp8_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP8_QUANT_BLOCK_SIZE
        )
        _store_quant_tile(
            out_tensor,
            bs_e8m0,
            x_fp8_ptr,
            bs_ptr,
            stride_x_fp8_m_in,
            stride_x_fp8_n_in,
            stride_bs_m_in,
            stride_bs_n_in,
            pid_m,
            pid_n,
            M,
            N,
            (N + MXFP8_QUANT_BLOCK_SIZE - 1) // MXFP8_QUANT_BLOCK_SIZE,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            NUM_QUANT_BLOCKS,
            EVEN_M_N,
        )
