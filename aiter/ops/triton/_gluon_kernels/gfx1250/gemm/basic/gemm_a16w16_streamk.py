# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""WG cluster StreamK GEMM for gfx1250, one output tile per workgroup cluster (CGA).

A cluster of CTAS_M x CTAS_N CTAs multicasts its TDM loads and computes one BLOCK_M x BLOCK_N tile
together, so BLOCK_M and BLOCK_N are cluster-wide. The epilogue aliases the dead B ring as the C
stage, which costs no extra LDS.

The tiles that do not fill a whole wave are the StreamK remainder. Phase 2 splits each of them
into SPLIT_M x SPLIT_N sub-tiles and spreads the sub-tiles over the clusters, so the last wave is
full. Each sub-tile is computed over the whole K and written straight to C, so there is no scratch
and no reduction.
"""

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.common.utils import swizzled_tile_coords
from aiter.ops.triton._gluon_kernels.gfx1250.tdm_pipeline import (
    compute_tile,
    tokens_per_step,
)
from aiter.ops.triton._triton_kernels.common.pid_preprocessing import remap_xcd_aid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_GLUON_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_SIZE_M",
    "NUM_BUFFERS",
    "CTAS_N",
    "TRANSPOSE_A",
    "TRANSPOSE_B",
    "KTAIL",
    "ONE_TILE",
    "STEADY_NONEMPTY",
    "C_OFFSET_64",
    "STREAMK_TILES",
    "SPLIT_M",
    "SPLIT_N",
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
    TRANSPOSE_A: gl.constexpr,
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
    ONE_TILE: gl.constexpr = 0,
    STEADY_NONEMPTY: gl.constexpr = 1,
    C_OFFSET_64: gl.constexpr = 0,
    STREAMK_TILES: gl.constexpr = 0,
    SPLIT_M: gl.constexpr = 1,
    SPLIT_N: gl.constexpr = 1,
    SHARED_REM_A: gl.constexpr = None,
    SHARED_REM_B: gl.constexpr = None,
    OPERAND_REM_A: gl.constexpr = None,
    OPERAND_REM_B: gl.constexpr = None,
    WMMA_REM: gl.constexpr = None,
):
    """BLOCK_M and BLOCK_N are cluster-wide; each CTA owns BLOCK_M / CTAS_M by BLOCK_N / CTAS_N."""
    gl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    num_ctas: gl.constexpr = gl.num_ctas()
    # A single CTA fuses its two TDM loads, so one K step leaves one token, not two.
    TOKENS: gl.constexpr = tokens_per_step(num_ctas)

    # NUM_SMS is the grid size in clusters, and program_id is the cluster rank. The remap gives
    # each XCD one contiguous run of tiles, in the order the hardware queue dispatches the XCDs,
    # so the XCDs that share an AID's L2 hold one contiguous range.
    pid = remap_xcd_aid(gl.program_id(0), NUM_SMS)

    # A transposed means an M-contiguous x, so the descriptor is (K, M) with the strides
    # swapped, exactly as native B is (K, N). The kernel only sees the orientation flag.
    if TRANSPOSE_A:
        a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(K, M),
            strides=(stride_ak, stride_am),
            block_shape=(BLOCK_K, BLOCK_M),
            layout=SHARED_LAYOUT_A,
        )
    else:
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

    # Trip-count hint, so the backend can drop the guard of the steady loop. The host sets the
    # gate: K can give only one K tile, and NUM_BUFFERS has a floor of 2, so the steady loop is
    # then empty. Without the gate this states 0 > 0, and that UB miscompiles the whole kernel:
    # 64x64x256 returned NaN and 64x64x32 returned garbage.
    if STEADY_NONEMPTY:
        gl.assume(gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1) > 0)

    # wmma gives an fp32 result, so ANY 2-byte C accumulates in fp32 and narrows at store time.
    # Testing only bfloat16 here leaves fp16 with an fp16 accumulator, which fails to compile.
    ACC_DTYPE: gl.constexpr = (
        gl.float32
        if c_ptr.type.element_ty.primitive_bitwidth == 16
        else c_ptr.type.element_ty
    )

    # Stage C through LDS and write it with one TDM store. compute_tile allocates the A and B
    # rings and this stage itself, so all three live for one tile only and the shared-memory
    # allocator puts the C stage on top of the dead rings. It therefore costs no extra LDS:
    # at per-CTA 256x256 with BLOCK_K 128 the kernel reserves 278496 bytes either way, while
    # A + B + C unoverlapped would need 393216 against a 327680 cap.
    #
    # The padding interval must equal the innermost block extent of the descriptor, which is
    # the per-CTA N. A dedicated stage can always meet that, so the store no longer depends on
    # the B layout or on BLOCK_K.
    _C_ELEM: gl.constexpr = c_ptr.type.element_ty.primitive_bitwidth // 8
    _C_PAD_N: gl.constexpr = BLOCK_N // CTAS_N
    C_ALIAS_B: gl.constexpr = _C_ELEM == 2 and _C_PAD_N <= 256
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

    # Phase 1: one tile for each cluster. ONE_TILE is a host constexpr saying phase 1
    # gives every worker exactly one tile, so the loop becomes straight-line code.
    if ONE_TILE:
        pid_m, pid_n = swizzled_tile_coords(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)
        compute_tile(
            a_desc,
            b_desc,
            c_desc,
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
            C_OFFSET_64,
            KTAIL,
            TOKENS,
            TRANSPOSE_A,
        )

    else:
        for tile_idx in range(pid, num_tiles - STREAMK_TILES, NUM_SMS):
            pid_m, pid_n = swizzled_tile_coords(
                tile_idx, num_pid_m, num_pid_n, GROUP_SIZE_M
            )
            compute_tile(
                a_desc,
                b_desc,
                c_desc,
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
                C_OFFSET_64,
                KTAIL,
                TOKENS,
                TRANSPOSE_A,
            )

    # Phase 2: the StreamK remainder. Each leftover tile becomes SPLIT_M x SPLIT_N sub-tiles, so
    # more clusters share the last wave. A sub-tile runs the same pipeline on its own layouts.
    if STREAMK_TILES:
        SUB_M: gl.constexpr = BLOCK_M // SPLIT_M
        SUB_N: gl.constexpr = BLOCK_N // SPLIT_N
        SPLITS: gl.constexpr = SPLIT_M * SPLIT_N
        _CR_PAD_N: gl.constexpr = SUB_N // CTAS_N
        # The remainder stages C the same way, on its own sub-tile layout.
        C_ALIAS_R: gl.constexpr = _C_ELEM == 2 and _CR_PAD_N <= 256
        SHARED_REM_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[_CR_PAD_N, 16]], [SUB_M, SUB_N], [1, 0], WMMA_REM.cga_layout
        )
        if TRANSPOSE_A:
            ar_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
                base=a_ptr,
                shape=(K, M),
                strides=(stride_ak, stride_am),
                block_shape=(BLOCK_K, SUB_M),
                layout=SHARED_REM_A,
            )
        else:
            ar_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
                base=a_ptr,
                shape=(M, K),
                strides=(stride_am, stride_ak),
                block_shape=(SUB_M, BLOCK_K),
                layout=SHARED_REM_A,
            )
        if TRANSPOSE_B:
            br_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
                base=b_ptr,
                shape=(N, K),
                strides=(stride_bn, stride_bk),
                block_shape=(SUB_N, BLOCK_K),
                layout=SHARED_REM_B,
            )
        else:
            br_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
                base=b_ptr,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                block_shape=(BLOCK_K, SUB_N),
                layout=SHARED_REM_B,
            )
        cr_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            block_shape=(SUB_M, SUB_N),
            layout=SHARED_REM_C,
        )
        for sub_idx in range(pid, STREAMK_TILES * SPLITS, NUM_SMS):
            tile_idx = num_tiles - STREAMK_TILES + sub_idx // SPLITS
            sub = sub_idx % SPLITS
            pid_m, pid_n = swizzled_tile_coords(
                tile_idx, num_pid_m, num_pid_n, GROUP_SIZE_M
            )
            compute_tile(
                ar_desc,
                br_desc,
                cr_desc,
                c_ptr,
                pid_m * BLOCK_M + (sub // SPLIT_N) * SUB_M,
                pid_n * BLOCK_N + (sub % SPLIT_N) * SUB_N,
                M,
                N,
                K,
                stride_cm,
                stride_cn,
                OPERAND_REM_A,
                OPERAND_REM_B,
                WMMA_REM,
                SHARED_REM_C,
                SUB_M,
                SUB_N,
                BLOCK_K,
                NUM_BUFFERS,
                TRANSPOSE_B,
                ACC_DTYPE,
                C_ALIAS_R,
                C_OFFSET_64,
                KTAIL,
                TOKENS,
                TRANSPOSE_A,
            )
