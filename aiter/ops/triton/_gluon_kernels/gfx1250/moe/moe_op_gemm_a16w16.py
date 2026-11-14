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

import functools

import triton.experimental.gluon.language as gl
import triton.language as tl
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.common.utils import (
    cluster_shape,
    consume_k_block,
)
from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


def get_moe_a16w16_layouts(
    num_warps, BLOCK_M, BLOCK_N, BLOCK_K, OUT_BLOCK_N, num_ctas, idx_width, split
):
    """Layouts for one kernel config, built on the host because make_cga_layout
    cannot be called from a jit kernel. Returns a copy; callers may mutate it."""
    return dict(
        _build_moe_a16w16_layouts(
            num_warps,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            OUT_BLOCK_N,
            num_ctas,
            idx_width,
            split,
        )
    )


@functools.cache
def _build_moe_a16w16_layouts(
    num_warps, BLOCK_M, BLOCK_N, BLOCK_K, OUT_BLOCK_N, num_ctas, idx_width, split
):
    assert BLOCK_K <= 512, "BLOCK_K exceeds the TDM pad interval"
    assert BLOCK_N <= 512, "BLOCK_N exceeds the TDM pad interval"
    if num_warps == 4:
        warp_bases = [[0, 1], [0, 2]]
    elif BLOCK_M == 16:
        warp_bases = [[0, 1], [0, 2], [0, 4]]
    else:
        warp_bases = [[0, 1], [0, 2], [1, 0]]

    # With num_ctas > 1 the BLOCK_M x BLOCK_N tile is a cluster tile and each
    # CTA owns BLOCK_M // ctas_m x BLOCK_N // ctas_n of it. Carrying the CGA
    # layout into the shared layouts is what lets TDM multicast one global read
    # to every CTA of the cluster.
    ctas_m, ctas_n = cluster_shape(num_ctas, BLOCK_M, BLOCK_N, split)
    cga = make_cga_layout([ctas_m, ctas_n], [ctas_m, ctas_n], [0, 1])

    wmma = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=[],
        instr_shape=[16, 16, 32],
        cga_layout=cga,
    )
    dot_x = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_w = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    # No transpose: W is staged as (K, N), already the dot-operand orientation.
    return {
        "WMMA_LAYOUT": wmma,
        "DOT_LAYOUT_X": dot_x,
        "DOT_LAYOUT_W": dot_w,
        "SHARED_LAYOUT_X": gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0], dot_x.cga_layout
        ),
        "SHARED_LAYOUT_W": gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N // ctas_n, 16]], [BLOCK_K, BLOCK_N], [1, 0], dot_w.cga_layout
        ),
        # The TDM store needs the pad interval to equal this CTA's innermost
        # extent, which is OUT_BLOCK_N divided over ctas_n.
        "SHARED_LAYOUT_Y": gl.PaddedSharedLayout.with_identity_for(
            [[OUT_BLOCK_N // ctas_n, 8]], [BLOCK_M, OUT_BLOCK_N], [1, 0], cga
        ),
        # async_gather needs the index and the X shared encoding to agree on
        # the row block basis. The index is 1-D over M and slicing dim 0 leaves
        # dim 1, so X's M bases move into that slot.
        "SHARED_LAYOUT_BIAS": gl.SwizzledSharedLayout(
            1,
            1,
            1,
            [1, 0],
            # All ctas_m CTAs of a column share one bias row, so the CGA
            # covers every CTA of the cluster but splits only along N.
            cga_layout=make_cga_layout([ctas_m, ctas_n], [1, ctas_n], [0, 1]),
        ),
        "IDX_LAYOUT": gl.SliceLayout(
            0,
            gl.BlockedLayout(
                [1, idx_width],
                [32, 1],
                [1, num_warps],
                [0, 1],
                cga_layout=[[0, m] for m, _ in dot_x.cga_layout],
            ),
        ),
    }


_moe_gemm_a16w16_repr = make_kernel_repr(
    "_moe_gemm_a16w16",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_M",
        "NUM_BUFFERS",
        "NUM_SUBTILES",
        "XCD_SWIZZLE",
        "APPLY_SWIGLU",
        "num_warps",
        "num_ctas",
    ],
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
    NUM_SUBTILES: gl.constexpr,
    EVEN_K: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    DOT_LAYOUT_X: gl.constexpr,
    DOT_LAYOUT_W: gl.constexpr,
    SHARED_LAYOUT_X: gl.constexpr,
    SHARED_LAYOUT_W: gl.constexpr,
    SHARED_LAYOUT_Y: gl.constexpr,
    SHARED_LAYOUT_BIAS: gl.constexpr,
    IDX_LAYOUT: gl.constexpr,
    num_warps: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
    num_ctas: gl.constexpr = 1,
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

    # Group-ordered tiles so neighbouring programs share a W tile in L2.
    pid_m, pid_n = pid_grid(pid, unpadded_m, grid_n, GROUP_M)

    # unpack expert data
    expt_data = gl.load(ExptData + pid_m)
    if XCD_SWIZZLE == 1 and expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    # Upcast before the X and Y stride products, which overflow int32 at large M.
    start_m = gl.load(ExptOffs + expt_id).to(index_type)
    expt_id = expt_id.to(index_type)

    # X / gather offsets
    offs_x_m_scalar = BLOCK_M * block_id
    if GatherIndx is None:
        X += start_m * stride_x_m
        offs_x_m = offs_x_m_scalar  # unused in non-gather path
    else:
        if GatherIndx.dtype.element_ty == gl.uint16:
            oob_idx = (num_tokens).to(gl.uint16)
        else:
            gl.static_assert(
                GatherIndx.dtype.element_ty == gl.int32,
                "Gather index datatype should be uint16 or int32",
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

    if num_ctas > 1:
        gl.amd.gfx1250.cluster.arrive()

    num_k_iter = tl.cdiv(K, BLOCK_K)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Steady state: each iteration issues 1 tile and consumes 1 tile.
    # NUM_BUFFERS - 1 tiles stay in flight.
    for _ in range(num_k_iter - (NUM_BUFFERS - 1)):
        if num_ctas > 1:
            # Every CTA has finished reading the slot we overwrite now.
            gl.amd.gfx1250.cluster.wait()
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
        acc = consume_k_block(
            acc,
            x_buffer.index(r_idx),
            w_buffer.index(r_idx),
            DOT_LAYOUT_X,
            DOT_LAYOUT_W,
            BLOCK_K,
            NUM_SUBTILES,
        )
        read_idx += 1
        if num_ctas > 1:
            gl.amd.gfx1250.cluster.arrive()

    if B is not None:
        BPtrs = B + expt_id * stride_b_e
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
        acc = consume_k_block(
            acc,
            x_buffer.index(r_idx),
            w_buffer.index(r_idx),
            DOT_LAYOUT_X,
            DOT_LAYOUT_W,
            BLOCK_K,
            NUM_SUBTILES,
        )
        read_idx += 1

    if num_ctas > 1:
        gl.amd.gfx1250.cluster.wait()

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

    out = out.to(Y.type.element_ty)

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
