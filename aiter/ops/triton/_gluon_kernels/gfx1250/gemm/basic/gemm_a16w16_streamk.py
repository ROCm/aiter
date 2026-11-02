# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""WG cluster StreamK GEMM for gfx1250, one output tile per workgroup cluster (CGA).

A cluster of CTAS_M x CTAS_N CTAs multicasts its TDM loads and computes one BLOCK_M x BLOCK_N tile
together, so BLOCK_M and BLOCK_N are cluster-wide. The epilogue aliases the dead B ring as the C
stage, which costs no extra LDS.
"""

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import remap_xcd_aid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

from ....common.utils import swizzled_tile_coords
from ...tdm_pipeline import compute_tile, tokens_per_step

_GLUON_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_SIZE_M",
    "NUM_BUFFERS",
    "CTAS_N",
    "TRANSPOSE_B",
    "KTAIL",
    "NUM_SMS",
    "num_warps",
    "num_ctas",
    "waves_per_eu",
]

_gemm_a16w16_streamk_repr = make_kernel_repr(
    "gemm_a16w16_streamk_gfx1250_kernel_", _GLUON_REPR_KEYS
)


@gluon.jit(repr=_gemm_a16w16_streamk_repr)
def gemm_a16w16_streamk_kernel_(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    num_tiles,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    TRANSPOSE_B: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    NUM_SMS: gl.constexpr,
    CTAS_N: gl.constexpr = 1,
    GROUP_SIZE_M: gl.constexpr = 2,
    KTAIL: gl.constexpr = 0,
):
    """BLOCK_M and BLOCK_N are cluster-wide; each CTA owns BLOCK_M / CTAS_M by BLOCK_N / CTAS_N."""
    gl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    num_ctas: gl.constexpr = gl.num_ctas()
    CTAS_M: gl.constexpr = num_ctas // CTAS_N
    # A single CTA fuses its two TDM loads, so one K step leaves one token, not two.
    TOKENS: gl.constexpr = tokens_per_step(num_ctas)

    # NUM_SMS is the grid size in clusters, and program_id is the cluster rank. The AID remap gives
    # each shared L2 a contiguous super-block of the tile space.
    pid = remap_xcd_aid(gl.program_id(0), NUM_SMS)

    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    if TRANSPOSE_B:
        b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B,
        )

    a_buffer = gl.allocate_shared_memory(
        a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout
    )
    b_buffer = gl.allocate_shared_memory(
        b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout
    )

    # Trip-count hint: the host guarantees cdiv(K, BLOCK_K) >= NUM_BUFFERS, so the steady loop runs
    # at least once and the backend can drop the loop guard.
    gl.assume(gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1) > 0)

    # wmma gives an fp32 result, so ANY 2-byte C accumulates in fp32 and narrows at store time.
    # Testing only bfloat16 here leaves fp16 with an fp16 accumulator, which fails to compile.
    ACC_DTYPE: gl.constexpr = (
        gl.float32
        if c_ptr.type.element_ty.primitive_bitwidth == 16
        else c_ptr.type.element_ty
    )

    # Stage C over the dead B ring: one reinterpret, one full-tile store, no extra LDS. Only the B
    # ring works, because the store needs an N-oriented padding interval and the A ring pads on K.
    _C_ELEM: gl.constexpr = c_ptr.type.element_ty.primitive_bitwidth // 8
    _C_PAD_N: gl.constexpr = BLOCK_N // CTAS_N
    _B_RING_PC: gl.constexpr = NUM_BUFFERS * BLOCK_K * _C_PAD_N * 2
    _C_TILE_PC: gl.constexpr = (BLOCK_M // CTAS_M) * _C_PAD_N * _C_ELEM
    C_ALIAS_B: gl.constexpr = (
        (not TRANSPOSE_B)
        and _C_ELEM == 2
        and _C_PAD_N <= 256
        and _C_TILE_PC <= _B_RING_PC
    )
    # Built in every case so the tile helper always has the names. It is used only when
    # C_ALIAS_B is on; otherwise the helper takes the masked store path.
    SHARED_LAYOUT_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[_C_PAD_N, 16]], [BLOCK_M, BLOCK_N], [1, 0], WMMA_LAYOUT.cga_layout
    )
    c_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=SHARED_LAYOUT_C,
    )

    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    for tile_idx in range(pid, num_tiles, NUM_SMS):
        pid_m, pid_n = swizzled_tile_coords(
            tile_idx, num_pid_m, num_pid_n, GROUP_SIZE_M
        )
        compute_tile(
            a_desc,
            b_desc,
            c_desc,
            a_buffer,
            b_buffer,
            c_ptr,
            pid_m * BLOCK_M,
            pid_n * BLOCK_N,
            M,
            N,
            K,
            stride_cm,
            stride_cn,
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
            WMMA_LAYOUT,
            SHARED_LAYOUT_C,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_BUFFERS,
            TRANSPOSE_B,
            ACC_DTYPE,
            C_ALIAS_B,
            KTAIL,
            TOKENS,
        )
