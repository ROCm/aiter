# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon BF16/FP16 GEMM for gfx950, compute-bound variant.

The kernel is the a16w16 v9 kernel of ROCm/gfx950-gluon-tutorials
(kernels/gemm/intra_wave/a16w16). It splits each tile into four quadrants,
keeps a 2-deep LDS pipeline and pins the accumulators to AGPRs. The wrapper
launches it with the MFMA scheduler of Triton (``schedule_hint``).
"""

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4 as ttgl_cdna4
from triton.runtime.jit import constexpr_function

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# The BLOCK_M and BLOCK_N values that the derived layouts support.
TILE_SIDES = (64, 128, 256)

_gemm_a16w16_compute_bound_repr = make_kernel_repr(
    "_gemm_a16w16_gfx950_compute_bound_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "ADD_BIAS",
        "EVEN_M",
        "EVEN_N",
        "EVEN_K",
    ],
)


@constexpr_function
def compute_gload_layout(shared_layout, load_contig, num_warps, threads_per_warp=64):
    """The coalesced global-load layout for a PaddedSharedLayout. Each warp
    writes contiguous LDS offsets, as CoalesceAsyncCopy does."""
    bases = [list(b) for b in shared_layout.offset_bases]
    rank = len(shared_layout.shape)
    n_reg = int(math.log2(load_contig))
    n_lane = int(math.log2(threads_per_warp))
    n_warp = int(math.log2(num_warps))
    reg = bases[:n_reg]
    lane = bases[n_reg : n_reg + n_lane]
    warp = bases[n_reg + n_lane : n_reg + n_lane + n_warp]
    while len(warp) < n_warp:  # zero-pad (broadcast) if we ran out of bases
        warp.append([0] * rank)
    reg += bases[n_reg + n_lane + n_warp :]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=list(shared_layout.shape),
    )


@gluon.jit(repr=_gemm_a16w16_compute_bound_repr)
def _gemm_a16w16_compute_bound_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
):
    """C = A @ B (+ bias) for one BLOCK_M x BLOCK_N tile, in four quadrant
    accumulators. EVEN_M, EVEN_N and EVEN_K select the paths for ragged shapes."""

    pid = remap_xcd(gl.program_id(axis=0), GRID_MN, NUM_XCDS=NUM_XCDS)
    pid_m, pid_n = pid_grid(
        pid, gl.cdiv(M, BLOCK_M), gl.cdiv(N, BLOCK_N), GROUP_SIZE_M=GROUP_SIZE_M
    )

    # The shared layouts come from the dot-operand layouts, and the global-load
    # layouts come from the shared layouts.
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    sharedLayoutA: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutA, [BLOCK_M // 2, BLOCK_K], a_ptr.dtype.element_ty
    )
    sharedLayoutB: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutB, [BLOCK_K, BLOCK_N // 2], b_ptr.dtype.element_ty
    )
    gLoadLayoutA: gl.constexpr = compute_gload_layout(sharedLayoutA, 8, gl.num_warps())
    gLoadLayoutB: gl.constexpr = compute_gload_layout(sharedLayoutB, 8, gl.num_warps())

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    # The tile bases are 64-bit because a tile can start after element 2^31.
    # The offsets in a tile stay 32-bit.
    a_base = a_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_am
    b_base = b_ptr + (pid_n * BLOCK_N).to(gl.int64) * stride_bn

    # On a ragged last tile, clamp the rows and columns to the matrix edge.
    # The masked stores drop the duplicate rows and columns.
    rows_top = offs_am
    rows_bot = offs_am + BLOCK_M // 2
    if not EVEN_M:
        rows_top = gl.minimum(rows_top, M - 1 - pid_m * BLOCK_M)
        rows_bot = gl.minimum(rows_bot, M - 1 - pid_m * BLOCK_M)
    cols_left = offs_bn
    cols_right = offs_bn + BLOCK_N // 2
    if not EVEN_N:
        cols_left = gl.minimum(cols_left, N - 1 - pid_n * BLOCK_N)
        cols_right = gl.minimum(cols_right, N - 1 - pid_n * BLOCK_N)

    # The wrapper checks that the row strides are multiples of 8. The hints keep
    # the 16-byte loads, because Triton records only strides that are multiples of 16.
    a_rows_top = gl.multiple_of(rows_top * stride_am, 8)
    a_rows_bot = gl.multiple_of(rows_bot * stride_am, 8)
    b_cols_left = gl.multiple_of(cols_left * stride_bn, 8)
    b_cols_right = gl.multiple_of(cols_right * stride_bn, 8)

    # Two sets of offsets: base (even K-steps) and _next (odd K-steps).
    a_top_offsets = a_rows_top[:, None] + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_rows_bot[:, None] + offs_ak[None, :] * stride_ak
    b_left_offsets = offs_bk[:, None] * stride_bk + b_cols_left[None, :]
    b_right_offsets = offs_bk[:, None] * stride_bk + b_cols_right[None, :]

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

    # A ragged K: the K % (2 * BLOCK_K) tail goes first, through buffer 0 with
    # masked loads. The pipelined loop then runs over K_MAIN as for an even K.
    if EVEN_K:
        K_MAIN: gl.constexpr = K
    else:
        K_MAIN: gl.constexpr = K - K % (2 * BLOCK_K)
        K_TAIL: gl.constexpr = K - K_MAIN
        a_tail = a_base + K_MAIN * stride_ak
        b_tail = b_base + K_MAIN * stride_bk
        k_of_a = gl.zeros_like(a_top_offsets) + offs_ak[None, :]
        k_of_b = gl.zeros_like(b_left_offsets) + offs_bk[:, None]
        for t in gl.static_range(gl.cdiv(K_TAIL, BLOCK_K)):
            mask_a = k_of_a + t * BLOCK_K < K_TAIL
            mask_b = k_of_b + t * BLOCK_K < K_TAIL
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemB_left.index(0),
                b_tail,
                b_left_offsets + t * BLOCK_K * stride_bk,
                mask=mask_b,
            )
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemA_top.index(0),
                a_tail,
                a_top_offsets + t * BLOCK_K * stride_ak,
                mask=mask_a,
            )
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemA_bot.index(0),
                a_tail,
                a_bot_offsets + t * BLOCK_K * stride_ak,
                mask=mask_a,
            )
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemB_right.index(0),
                b_tail,
                b_right_offsets + t * BLOCK_K * stride_bk,
                mask=mask_b,
            )
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.wait_group(0)
            b_left = smemB_left.index(0).load(dotOpLayoutB)
            a_top = smemA_top.index(0).load(dotOpLayoutA)
            a_bot = smemA_bot.index(0).load(dotOpLayoutA)
            b_right = smemB_right.index(0).load(dotOpLayoutB)
            acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
            acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
            acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
            acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    iterMax = gl.cdiv(K_MAIN, BLOCK_K)

    # Prologue: load the first two K steps.
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base, a_bot_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base, b_right_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base, a_bot_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base, b_right_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.assume(iterMax > 3)

    # Main loop: two K steps in each iteration.
    for k in range(0, iterMax - 2, 2):

        # Sub-iteration 0: consume buffer 0, prefetch into buffer 0.

        # Region 0: C_tl = DOT(a_top, b_left)
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(0), b_base, b_left_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Region 1: C_bl = DOT(a_bot, b_left)
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(0), a_base, a_top_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Region 2: C_tr = DOT(a_top, b_right)
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(0), a_base, a_bot_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Region 3: C_br = DOT(a_bot, b_right)
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(0), b_base, b_right_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Sub-iteration 1: consume buffer 1, prefetch into buffer 1 with the
        # _next offsets (odd K-step).

        # Region 0: C_tl = DOT(a_top, b_left)
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_base, b_left_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Region 1: C_bl = DOT(a_bot, b_left)
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_base, a_top_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Region 2: C_tr = DOT(a_top, b_right)
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_base, a_bot_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Region 3: C_br = DOT(a_bot, b_right)
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_base, b_right_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    # Epilogue: store each quadrant after its last MFMA.

    # Each thread stores 8 contiguous elements (16 bytes), and one warp spans
    # the BLOCK_N // 2 columns of a quadrant.
    STORE_CONTIG: gl.constexpr = 8
    STORE_LANES_N: gl.constexpr = (BLOCK_N // 2) // STORE_CONTIG
    STORE_LANES_M: gl.constexpr = 64 // STORE_LANES_N
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout(
        [1, STORE_CONTIG], [STORE_LANES_M, STORE_LANES_N], [4, 1], [1, 0]
    )

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = (
        c_ptr
        + (pid_m * BLOCK_M).to(gl.int64) * stride_cm
        + (pid_n * BLOCK_N).to(gl.int64) * stride_cn
    )
    c_rows = gl.multiple_of(offs_cm * stride_cm, 8)  # same alignment hint as the loads
    c_tl_offsets = c_rows[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
    c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
    c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2
    if not EVEN_N:
        # The columns left in C are a multiple of 8. The hint keeps one 16-byte
        # store per thread; it cannot go on N, because Triton drops argument hints.
        n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)
    if not (EVEN_M and EVEN_N):
        # The ragged last tile stores only the rows and columns inside C.
        m_left = M - pid_m * BLOCK_M
        in_top = offs_cm[:, None] < m_left
        in_bot = offs_cm[:, None] + BLOCK_M // 2 < m_left
        if EVEN_N:
            in_left = (
                offs_cn[None, :] < BLOCK_N
            )  # no column limit; gives the mask its shape
            in_right = in_left
        else:
            in_left = offs_cn[None, :] < n_left
            in_right = offs_cn[None, :] + BLOCK_N // 2 < n_left
        mask_tl = in_top & in_left
        mask_tr = in_top & in_right
        mask_bl = in_bot & in_left
        mask_br = in_bot & in_right

    if ADD_BIAS:
        # The kernel adds bias[N] in fp32 before the downcast. It loads the bias
        # here, so that the load latency overlaps the last MFMAs.
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, mfmaLayout))
        if EVEN_N:
            bias_l = gl.load(bias_ptr + pid_n * BLOCK_N + offs_bias).to(gl.float32)
            bias_r = gl.load(bias_ptr + pid_n * BLOCK_N + BLOCK_N // 2 + offs_bias).to(
                gl.float32
            )
        else:
            bias_l = gl.load(
                bias_ptr + pid_n * BLOCK_N + offs_bias, mask=offs_bias < n_left, other=0
            ).to(gl.float32)
            bias_r = gl.load(
                bias_ptr + pid_n * BLOCK_N + BLOCK_N // 2 + offs_bias,
                mask=offs_bias + BLOCK_N // 2 < n_left,
                other=0,
            ).to(gl.float32)

    # Iteration iterMax - 2: the 4-region pattern of the main loop, without loads.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)

    # Iteration iterMax - 1: each quadrant is stored one MFMA after its last MFMA.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    if ADD_BIAS:
        acc_tl = acc_tl + bias_l[None, :]
    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    if EVEN_M and EVEN_N:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tl_offsets, stored_value=c_tl)
    else:
        gl.amd.cdna3.buffer_store(
            ptr=c_base, offsets=c_tl_offsets, stored_value=c_tl, mask=mask_tl
        )

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    if ADD_BIAS:
        acc_bl = acc_bl + bias_l[None, :]
    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    if EVEN_M and EVEN_N:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_bl_offsets, stored_value=c_bl)
    else:
        gl.amd.cdna3.buffer_store(
            ptr=c_base, offsets=c_bl_offsets, stored_value=c_bl, mask=mask_bl
        )

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    if ADD_BIAS:
        acc_tr = acc_tr + bias_r[None, :]
    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    if EVEN_M and EVEN_N:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tr_offsets, stored_value=c_tr)
    else:
        gl.amd.cdna3.buffer_store(
            ptr=c_base, offsets=c_tr_offsets, stored_value=c_tr, mask=mask_tr
        )

    if ADD_BIAS:
        acc_br = acc_br + bias_r[None, :]
    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    if EVEN_M and EVEN_N:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_br_offsets, stored_value=c_br)
    else:
        gl.amd.cdna3.buffer_store(
            ptr=c_base, offsets=c_br_offsets, stored_value=c_br, mask=mask_br
        )


_KERNEL_MAP = {"compute_bound": _gemm_a16w16_compute_bound_kernel}
