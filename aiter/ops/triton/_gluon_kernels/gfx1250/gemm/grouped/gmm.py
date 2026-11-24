# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (  # noqa: F401
    create_shared_layouts,
    create_wmma_layouts,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gmm_gfx1250_kernel_repr = make_kernel_repr(
    "_gmm_gfx1250_kernel",
    [
        "G",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "NUM_BUFFERS",
        "LAYOUT",
        "ADD_BIAS",
        "num_warps",
        "waves_per_eu",
    ],
)


@gluon.jit
def _step_k(
    a_desc,
    b_desc,
    BLOCK_K: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    # lhs is always row-major (M, K).
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, BLOCK_K]
    )
    if LAYOUT[1] == "T":
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[BLOCK_K, 0]
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )
    return a_desc, b_desc


@gluon.jit
def _read_operands(
    a_buffer,
    b_buffer,
    idx,
    NUM_BUFFERS: gl.constexpr,
    LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
):
    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index(idx % NUM_BUFFERS), OPERAND_LAYOUT_A
    )
    if LAYOUT[1] == "T":
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(idx % NUM_BUFFERS), OPERAND_LAYOUT_B
        )
    else:
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B
        )
    return cur_a, cur_b


@gluon.jit(repr=_gmm_gfx1250_kernel_repr)
def _gmm_gfx1250_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    group_sizes_ptr,
    N,
    K,
    stride_bk,
    stride_bn,
    G: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    LAYOUT: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    pid = gl.program_id(axis=0).to(gl.int64)
    num_n_tiles = gl.cdiv(N, BLOCK_N)

    # Map the flat program id to (group, tile inside the group). Programs past
    # the last real tile exit below.
    last_tile = pid * 0
    last_m = pid * 0
    sel_g = pid * 0
    sel_m = pid * 0
    sel_last_m = pid * 0
    sel_tile = pid * 0 - 1
    for g in gl.static_range(G):
        m = gl.load(group_sizes_ptr + g).to(gl.int64)
        num_tiles = gl.cdiv(m, BLOCK_M) * num_n_tiles
        hit = (pid >= last_tile) & (pid < last_tile + num_tiles)
        sel_g = gl.where(hit, g, sel_g)
        sel_m = gl.where(hit, m, sel_m)
        sel_last_m = gl.where(hit, last_m, sel_last_m)
        sel_tile = gl.where(hit, pid - last_tile, sel_tile)
        last_tile += num_tiles
        last_m += m

    if sel_tile >= 0:
        _gmm_tile(
            a_ptr,
            b_ptr,
            c_ptr,
            bias_ptr,
            N,
            K,
            stride_bk,
            stride_bn,
            sel_g,
            sel_m,
            sel_last_m,
            sel_tile,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_BUFFERS,
            LAYOUT,
            SHARED_LAYOUT_A,
            SHARED_LAYOUT_B,
            WMMA_LAYOUT,
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
            ADD_BIAS,
        )


@gluon.jit
def _gmm_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    N,
    K,
    stride_bk,
    stride_bn,
    sel_g,
    sel_m,
    sel_last_m,
    sel_tile,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    LAYOUT: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    num_m_tiles = gl.cdiv(sel_m, BLOCK_M)
    tile_m = sel_tile % num_m_tiles
    tile_n = sel_tile // num_m_tiles

    # Rows of this tile that belong to the group; TDM zero-fills past it.
    rows_left = (sel_m - tile_m * BLOCK_M).to(gl.int32)
    cols_left = (N - tile_n * BLOCK_N).to(gl.int32)
    row0 = sel_last_m + tile_m * BLOCK_M

    a_base = a_ptr + row0 * K
    b_base = b_ptr + sel_g * K * N + tile_n * BLOCK_N * stride_bn

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_base,
        shape=(rows_left, K),
        strides=(K, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )

    if LAYOUT[1] == "T":
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_base,
            shape=(K, cols_left),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_base,
            shape=(cols_left, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B,
        )

    a_buffer = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
        layout=SHARED_LAYOUT_A,
    )
    if LAYOUT[1] == "T":
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B,
        )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Fill the pipeline.
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        a_desc, b_desc = _step_k(a_desc, b_desc, BLOCK_K, LAYOUT)
        load_idx += 1

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    for _ in range(num_k_tiles - (NUM_BUFFERS - 1) - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        a_desc, b_desc = _step_k(a_desc, b_desc, BLOCK_K, LAYOUT)
        load_idx += 1

        cur_a, cur_b = _read_operands(
            a_buffer,
            b_buffer,
            compute_idx,
            NUM_BUFFERS,
            LAYOUT,
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # Peeled last K tile: add_offsets leaves the OOB bound alone, so reset it to
    # the true remaining K extent to zero-fill a partial tail.
    k_main = (num_k_tiles - 1) * BLOCK_K
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, set_bounds=[rows_left, K - k_main]
    )
    if LAYOUT[1] == "T":
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, set_bounds=[K - k_main, cols_left]
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, set_bounds=[cols_left, K - k_main]
        )

    gl.amd.gfx1250.tdm.async_load(
        a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_load(
        b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
    load_idx += 1

    cur_a, cur_b = _read_operands(
        a_buffer,
        b_buffer,
        compute_idx,
        NUM_BUFFERS,
        LAYOUT,
        OPERAND_LAYOUT_A,
        OPERAND_LAYOUT_B,
    )
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
    compute_idx += 1

    # Drain.
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        cur_a, cur_b = _read_operands(
            a_buffer,
            b_buffer,
            compute_idx,
            NUM_BUFFERS,
            LAYOUT,
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    offs_cm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))

    if ADD_BIAS:
        offs_bias = tile_n * BLOCK_N + offs_cn
        bias_vals = gl.load(
            bias_ptr + sel_g * N + offs_bias, mask=offs_bias < N, other=0.0
        )
        accumulator = accumulator + bias_vals.to(gl.float32)[None, :]

    # buffer_store offsets are 32-bit, so rebase C to this tile in int64.
    c_base = c_ptr + row0 * N + tile_n * BLOCK_N
    offs_c = N * offs_cm[:, None] + offs_cn[None, :]
    mask_c = (offs_cm[:, None] < rows_left) & (offs_cn[None, :] < cols_left)

    gl.amd.gfx1250.buffer_store(
        accumulator.to(c_ptr.type.element_ty), c_base, offs_c, mask=mask_c
    )
