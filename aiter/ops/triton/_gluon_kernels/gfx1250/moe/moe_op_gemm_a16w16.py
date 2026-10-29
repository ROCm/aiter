# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon MoE GEMM for gfx1250 with 16-bit activations and 16-bit weights.

Derived from ``moe_op_gemm_a16w4``. The expert decode, gather addressing, bias,
SwiGLU and gammas epilogue are identical. The weight path is simpler: there is
nothing to dequantize, so W loads straight into the WMMA B-operand layout and
each K step issues two TDM loads (X, W) instead of three.

W is ``(E, K, N)`` row-major (N-contiguous), the natural bf16 weight layout, so
the B tile needs no transpose.
"""

import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental import gluon

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_moe_gemm_a16w16_repr = make_kernel_repr(
    "_moe_gemm_a16w16",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "NUM_BUFFERS", "XCD_SWIZZLE"],
)


@gluon.jit(repr=_moe_gemm_a16w16_repr)
def _moe_gemm_a16w16(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    B,
    stride_b_e,  # Bias
    Gammas,
    num_tokens,
    N,
    K,  # shapes
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: gl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    ADD_RESIDUAL: gl.constexpr,
    # MoE config
    N_EXPTS_ACT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    EVEN_K: gl.constexpr,
    num_warps: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):
    NUM_TDM_OPS: gl.constexpr = 2  # X, W
    gl.static_assert(num_warps == 4 or num_warps == 8, "num_warps must be 4 or 8")

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N
    CLAMP_BOUNDS: gl.constexpr = not EVEN_K

    pid = gl.program_id(0)

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    if XCD_SWIZZLE != 1:
        padding_m = grid_m - gl.load(ExptOffsSum)
        unpadded_m = grid_m - padding_m
        total_actual_tiles = unpadded_m * grid_n
        if padding_m > 0 and pid >= total_actual_tiles:
            return
        pid = remap_xcd(pid, total_actual_tiles, XCD_SWIZZLE)
    else:
        unpadded_m = grid_m

    pid_m, pid_n = pid_grid(pid, unpadded_m, grid_n, 1)

    # unpack expert data
    expt_data = gl.load(ExptData + pid_m)
    if XCD_SWIZZLE == 1 and expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    start_m = gl.load(ExptOffs + expt_id)
    expt_id = expt_id.to(index_type)

    # X / gather offsets
    offs_x_m_scalar = BLOCK_M * block_id
    if GatherIndx is None:
        X += start_m * stride_x_m
        offs_x_m = offs_x_m_scalar  # unused in non-gather path
    else:
        if GatherIndx.dtype.element_ty == gl.uint16:
            IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
                0, gl.BlockedLayout([1, 16], [32, 1], [1, num_warps], [0, 1])
            )
            oob_idx = (num_tokens).to(gl.uint16)
        else:
            gl.static_assert(
                GatherIndx.dtype.element_ty == gl.int32,
                "Gather index datatype should be uint16 or int32",
            )
            IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
                0, gl.BlockedLayout([1, 8], [32, 1], [1, num_warps], [0, 1])
            )
            oob_idx = num_tokens

        offs_x_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=IDX_LAYOUT)
        mask_idx = offs_x_m < M
        offs_x_m = offs_x_m % M
        GatherIndx += start_m
        offs_x_m = gl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
        offs_x_m = gl.where(mask_idx, offs_x_m, oob_idx)

    off_w_n = pid_n * BLOCK_N

    W += expt_id * stride_w_e

    # WMMA layout for plain bf16 x bf16: instr_shape [16, 16, 32], k_width=8.
    if num_warps == 4:
        WARP_BASES: gl.constexpr = [[0, 1], [0, 2]]
    else:
        if BLOCK_M == 16:
            WARP_BASES: gl.constexpr = [[0, 1], [0, 2], [0, 4]]
        else:
            WARP_BASES: gl.constexpr = [[0, 1], [0, 2], [1, 0]]

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES,
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    DOT_LAYOUT_X: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    DOT_LAYOUT_W: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )

    # TDM encodes the pad interval in dwords in a 3-bit log2 field, so it cannot
    # exceed 256 dwords, i.e. 512 16-bit elements.
    gl.static_assert(BLOCK_K <= 512, "BLOCK_K exceeds the TDM pad interval")
    gl.static_assert(BLOCK_N <= 512, "BLOCK_N exceeds the TDM pad interval")
    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    # W is (K, N): the B operand orientation WMMA wants, so no transpose.
    SHARED_LAYOUT_W: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
    )
    SHARED_LAYOUT_Y: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[OUT_BLOCK_N, 8]], [BLOCK_M, OUT_BLOCK_N], [1, 0]
    )

    if GatherIndx is None:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(M, K),
            strides=(stride_x_m, stride_x_k),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_X,
        )
    else:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(num_tokens, K),
            strides=(stride_x_m, stride_x_k),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_X,
        )

    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=W,
        shape=(K, N),
        strides=(stride_w_k, stride_w_n),
        block_shape=(BLOCK_K, BLOCK_N),
        layout=SHARED_LAYOUT_W,
    )

    x_buffer = gl.allocate_shared_memory(
        x_desc.dtype, shape=[NUM_BUFFERS] + x_desc.block_shape, layout=x_desc.layout
    )
    w_buffer = gl.allocate_shared_memory(
        w_desc.dtype, shape=[NUM_BUFFERS] + w_desc.block_shape, layout=w_desc.layout
    )

    write_idx = 0
    read_idx = 0

    # Prologue: prime NUM_BUFFERS - 1 tile loads (X, W).
    for _ in gl.static_range(NUM_BUFFERS - 1):
        w_idx = write_idx % NUM_BUFFERS
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(
                x_desc, [offs_x_m_scalar, 0], x_buffer.index(w_idx)
            )
        else:
            gl.amd.gfx1250.tdm.async_gather(x_desc, offs_x_m, x_buffer.index(w_idx))
        gl.amd.gfx1250.tdm.async_load(w_desc, [0, off_w_n], w_buffer.index(w_idx))

        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_K], clamp_bounds=CLAMP_BOUNDS
        )
        w_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            w_desc, add_offsets=[BLOCK_K, 0], clamp_bounds=CLAMP_BOUNDS
        )

        write_idx += 1

    num_k_iter = tl.cdiv(K, BLOCK_K)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Steady state: each iteration issues 1 tile and consumes 1 tile.
    # NUM_BUFFERS - 1 tiles stay in flight.
    for _ in range(num_k_iter - (NUM_BUFFERS - 1)):
        # issue next tile
        w_idx = write_idx % NUM_BUFFERS
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(
                x_desc, [offs_x_m_scalar, 0], x_buffer.index(w_idx)
            )
        else:
            gl.amd.gfx1250.tdm.async_gather(x_desc, offs_x_m, x_buffer.index(w_idx))
        gl.amd.gfx1250.tdm.async_load(w_desc, [0, off_w_n], w_buffer.index(w_idx))

        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_K], clamp_bounds=CLAMP_BOUNDS
        )
        w_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            w_desc, add_offsets=[BLOCK_K, 0], clamp_bounds=CLAMP_BOUNDS
        )

        write_idx += 1

        # wait for the oldest in-flight tile, then consume it
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * NUM_TDM_OPS)
        r_idx = read_idx % NUM_BUFFERS
        x_tile = x_buffer.index(r_idx).load(layout=DOT_LAYOUT_X)
        w_tile = w_buffer.index(r_idx).load(layout=DOT_LAYOUT_W)
        acc = gl.amd.gfx1250.wmma(x_tile, w_tile, acc)
        read_idx += 1

    if B is not None:
        BPtrs = B + expt_id * stride_b_e
        SHARED_LAYOUT_BIAS: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        bias_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=BPtrs,
            shape=(1, N),
            strides=(N, 1),
            block_shape=(1, BLOCK_N),
            layout=SHARED_LAYOUT_BIAS,
        )
        bias_buffer = gl.allocate_shared_memory(
            bias_desc.dtype, shape=[1, BLOCK_N], layout=bias_desc.layout
        )
        gl.amd.gfx1250.tdm.async_load(
            bias_desc,
            [0, pid_n * BLOCK_N],
            bias_buffer,
        )
        TDM_BIAS_WAIT: gl.constexpr = 1
    else:
        TDM_BIAS_WAIT: gl.constexpr = 0

    # Epilogue: drain remaining NUM_BUFFERS - 1 tiles with a counting-down wait threshold.
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait(
            (NUM_BUFFERS - 2 - i) * NUM_TDM_OPS + TDM_BIAS_WAIT
        )
        r_idx = read_idx % NUM_BUFFERS
        x_tile = x_buffer.index(r_idx).load(layout=DOT_LAYOUT_X)
        w_tile = w_buffer.index(r_idx).load(layout=DOT_LAYOUT_W)
        acc = gl.amd.gfx1250.wmma(x_tile, w_tile, acc)
        read_idx += 1

    # bias / activation / write-back
    if B is not None:
        gl.amd.gfx1250.tdm.async_wait(0)
        bias = bias_buffer.reshape((BLOCK_N,)).load(
            layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        acc = acc + bias[None, :]

    if APPLY_SWIGLU:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
    else:
        tl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc

    if Gammas is not None:
        offs_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        gammas = gl.amd.cdna3.buffer_load(
            Gammas, start_m + offs_m, mask=mask_m, other=0.0
        )
        out *= gammas[:, None]

    out = out.to(gl.bfloat16)

    # TDM Store: accumulator -> shared memory -> global memory
    Y += start_m * stride_y_m
    y_buffer = gl.allocate_shared_memory(
        Y.type.element_ty,
        shape=[BLOCK_M, OUT_BLOCK_N],
        layout=SHARED_LAYOUT_Y,
    )
    y_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=Y,
        shape=(M, yN),
        strides=(stride_y_m, stride_y_n),
        block_shape=(BLOCK_M, OUT_BLOCK_N),
        layout=SHARED_LAYOUT_Y,
    )
    y_buffer.store(out)
    gl.amd.gfx1250.tdm.async_store(
        y_desc, [block_id * BLOCK_M, pid_n * OUT_BLOCK_N], y_buffer
    )
    gl.amd.gfx1250.tdm.async_wait(0)
