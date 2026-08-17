# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon port of the persistent Triton GEMM.
"""

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd

_GLUON_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_SIZE_M",
    "NUM_BUFFERS",
    "USE_ACTIVATION",
    "ADD_BIAS",
    "NUM_WGS",
    "num_warps",
]

_gemm_a16w16_persistent_repr = make_kernel_repr(
    "gemm_a16w16_persistent_gfx1250_kernel_", _GLUON_REPR_KEYS
)


@gluon.jit(repr=_gemm_a16w16_persistent_repr)
def gemm_a16w16_persistent_kernel_(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    NUM_WGS: gl.constexpr,
    num_warps: gl.constexpr,
):
    """C = A x B, one BLOCK_M x BLOCK_N tile of C per persistent-loop iteration."""
    gl.static_assert(NUM_BUFFERS >= 2, "persistent gemm requires NUM_BUFFERS >= 2")

    warp_bases: gl.constexpr = [
        [0, 1] if i == 0 else [1 << (i - 1), 0]
        for i in [0, 1, 2, 3]
        if (1 << i) < num_warps
    ]
    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]
    )
    OPERAND_LAYOUT_A: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    OPERAND_LAYOUT_B: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )

    start_pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    a_buffer = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
        layout=SHARED_LAYOUT_A,
    )
    b_buffer = gl.allocate_shared_memory(
        b_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
        layout=SHARED_LAYOUT_B,
    )

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        block_shape=(BLOCK_K, BLOCK_N),
        layout=SHARED_LAYOUT_B,
    )

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # Persistent loop
    for tile_id in range(start_pid, num_tiles, NUM_WGS):
        # remap tile index
        t = remap_xcd(tile_id, num_tiles, NUM_XCDS=8)
        pid_m, pid_n = pid_grid(t, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

        m_off = pid_m * BLOCK_M
        n_off = pid_n * BLOCK_N

        load_idx = 0
        compute_idx = 0

        accumulator = gl.zeros(
            (BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT
        )
        if ADD_BIAS:
            offs_bias = n_off + gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
            )
            bias_vals = gl.load(
                bias_ptr + offs_bias, mask=offs_bias < N, other=0.0
            )
            accumulator = accumulator + bias_vals[None, :]

        # fill buffers with tiles
        for _ in gl.static_range(NUM_BUFFERS - 1):
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [m_off, load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS),
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [load_idx * BLOCK_K, n_off],
                b_buffer.index(load_idx % NUM_BUFFERS),
            )
            load_idx += 1

        # produce and consume k tiles
        for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [m_off, load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS),
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [load_idx * BLOCK_K, n_off],
                b_buffer.index(load_idx % NUM_BUFFERS),
            )
            # leave the NUM_BUFFERS-1 most recent pairs in flight
            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

            load_idx += 1

            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
            )
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
            )
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            compute_idx += 1

        # drain remaining loads
        for i in gl.static_range(NUM_BUFFERS - 1):
            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
            )
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
            )
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            compute_idx += 1

        if USE_ACTIVATION:
            accumulator = activation(accumulator)

        offs_cm = m_off + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
        )
        offs_cn = n_off + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        gl.amd.gfx1250.buffer_store(
            accumulator.to(c_ptr.type.element_ty),
            c_ptr,
            offs_c,
            mask=mask_c,
        )

        gl.barrier()
