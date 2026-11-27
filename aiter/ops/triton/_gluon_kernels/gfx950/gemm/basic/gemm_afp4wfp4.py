# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon MXFP4 GEMM kernels for gfx950.

_KERNEL_MAP maps the kernel_type of a config to a kernel for unshuffled inputs:
"bandwidth_bound" is the default kernel and "compute_bound" is the 8-wave
ping-pong kernel. _PRESHUFFLE_KERNEL_MAP maps it for preshuffled inputs:
"compute_bound" is the 8-wave ping-pong kernel with 16x16x128 MFMAs.
"""

import math

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

try:
    from triton.experimental.gluon.language.amd import warp_pipeline_stage
except ImportError:
    # Triton 3.6 does not have it. Only the ping-pong kernels use it, and the
    # wrapper runs them only on the validated Triton builds.
    warp_pipeline_stage = None


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
    }
)
@gluon.jit
def _gemm_afp4wfp4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    waves_per_eu: gl.constexpr,
    matrix_instr_nonkdim: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    GRID_MN = gl.cdiv(M, BLOCK_SIZE_M) * gl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = gl.program_id(axis=0)
    # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=NUM_XCDS)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: gl.constexpr = 32

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

    blocked_scales: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

    linear_as: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 2], [0, 4], [64, 0], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
        warp_bases=[[0, 0], [0, 0], [32, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
    )

    linear_bs: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 2], [0, 4], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
        warp_bases=[[32, 0], [64, 0], [0, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
    )

    linear_mn: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [0, 128], [64, 0], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]],
        warp_bases=[[0, 32], [0, 64], [32, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )

    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )

    shared_scales: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0, 1]
    )

    MFMA_INSTR_SHAPE: gl.constexpr = [32, 32, 64]
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[2, num_warps // 2],
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = gl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_ak = gl.arange(0, BLOCK_SIZE_K // 2, layout=gl.SliceLayout(0, blocked_mk))
        offs_ak_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_ak
        offs_bk = gl.arange(0, BLOCK_SIZE_K // 2, layout=gl.SliceLayout(1, blocked_kn))
        offs_bk_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_bk
        offs_am = (
            pid_m * BLOCK_SIZE_M
            + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk))
        ) % M
        offs_bn = (
            pid_n * BLOCK_SIZE_N
            + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn))
        ) % N
        offs_a = offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak
        offs_b = offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        # Create pointers for the first block of A and B scales
        offs_ks = gl.arange(
            0,
            BLOCK_SIZE_K // SCALE_GROUP_SIZE,
            layout=gl.SliceLayout(0, blocked_scales),
        )
        offs_ks_split = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + offs_ks
        offs_asm = (
            pid_m * BLOCK_SIZE_M
            + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_scales))
        ) % M
        offs_bsn = (
            pid_n * BLOCK_SIZE_N
            + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(1, blocked_scales))
        ) % N

        offs_as = offs_asm[:, None] * stride_asm + offs_ks_split[None, :] * stride_ask
        # B scales are N x K even though B operand is K x N.
        offs_bs = offs_bsn[:, None] * stride_bsn + offs_ks_split[None, :] * stride_bsk

        # Create shared memories
        smem_a = gl.allocate_shared_memory(
            a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K // 2], layout=shared_a
        )
        smem_b = gl.allocate_shared_memory(
            b_ptr.type.element_ty, [BLOCK_SIZE_K // 2, BLOCK_SIZE_N], layout=shared_b
        )

        smem_as = gl.allocate_shared_memory(
            a_scales_ptr.type.element_ty,
            [BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
            layout=shared_scales,
        )
        smem_bs = gl.allocate_shared_memory(
            b_scales_ptr.type.element_ty,
            [BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
            layout=shared_scales,
        )

        if EVEN_K:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
            )
            a_scales = gl.amd.cdna4.buffer_load(
                ptr=a_scales_ptr,
                offsets=offs_as,
            )
        else:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=offs_ak[None, :] < K - pid_k * SPLITK_BLOCK_SIZE // 2,
            )
            a_scales = gl.amd.cdna4.buffer_load(
                ptr=a_scales_ptr,
                offsets=offs_as,
                mask=offs_ks[None, :]
                < (2 * K // SCALE_GROUP_SIZE)
                - pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE),
            )

        if EVEN_K:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                cache=cache_modifier,
            )
            b_scales = gl.amd.cdna4.buffer_load(
                ptr=b_scales_ptr, offsets=offs_bs, cache=cache_modifier
            )
        else:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=offs_bk[:, None] < K - pid_k * SPLITK_BLOCK_SIZE // 2,
                cache=cache_modifier,
            )
            b_scales = gl.amd.cdna4.buffer_load(
                ptr=b_scales_ptr,
                offsets=offs_bs,
                mask=offs_ks[None, :]
                < (2 * K // SCALE_GROUP_SIZE)
                - pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE),
                cache=cache_modifier,
            )

        smem_as.store(a_scales)
        smem_a.store(a)

        accumulator = gl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout
        )

        # num_stages:2
        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter - 1):

            # advance pointers
            a_ptr += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptr += (BLOCK_SIZE_K // 2) * stride_bk
            a_scales_ptr += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            b_scales_ptr += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

            if EVEN_K:
                a = gl.amd.cdna4.buffer_load(
                    ptr=a_ptr,
                    offsets=offs_a,
                )
            else:
                a = gl.amd.cdna4.buffer_load(
                    ptr=a_ptr,
                    offsets=offs_a,
                    mask=offs_ak[None, :] < K - (k + 1) * (BLOCK_SIZE_K // 2),
                )
            smem_b.store(b)
            smem_bs.store(b_scales)
            curr_a = smem_a.load(layout=dot_a_layout)
            curr_a_scales = smem_as.load(layout=linear_as)

            if EVEN_K:
                a_scales = gl.amd.cdna4.buffer_load(
                    ptr=a_scales_ptr,
                    offsets=offs_as,
                )
            else:
                a_scales = gl.amd.cdna4.buffer_load(
                    ptr=a_scales_ptr,
                    offsets=offs_as,
                    mask=offs_ks[None, :]
                    < (2 * K // SCALE_GROUP_SIZE)
                    - (k + 1) * (BLOCK_SIZE_K // SCALE_GROUP_SIZE),
                )

            curr_b_scales = smem_bs.load(layout=linear_bs)
            if EVEN_K:
                b = gl.amd.cdna4.buffer_load(
                    ptr=b_ptr,
                    offsets=offs_b,
                    cache=cache_modifier,
                )
                b_scales = gl.amd.cdna4.buffer_load(
                    ptr=b_scales_ptr, offsets=offs_bs, cache=cache_modifier
                )
            else:
                b = gl.amd.cdna4.buffer_load(
                    ptr=b_ptr,
                    offsets=offs_b,
                    mask=offs_bk[:, None] < K - (k + 1) * (BLOCK_SIZE_K // 2),
                    cache=cache_modifier,
                )
                b_scales = gl.amd.cdna4.buffer_load(
                    ptr=b_scales_ptr,
                    offsets=offs_bs,
                    mask=offs_ks[None, :]
                    < (2 * K // SCALE_GROUP_SIZE)
                    - (k + 1) * (BLOCK_SIZE_K // SCALE_GROUP_SIZE),
                    cache=cache_modifier,
                )
            curr_b = smem_b.load(layout=dot_b_layout)

            accumulator = gl.amd.cdna4.mfma_scaled(
                a=curr_a,
                a_scale=curr_a_scales,
                a_format="e2m1",
                b=curr_b,
                b_scale=curr_b_scales,
                b_format="e2m1",
                acc=accumulator,
            )

            smem_a.store(a)
            smem_as.store(a_scales)

        # ======= Epilogue ========
        smem_b.store(b)
        smem_bs.store(b_scales)
        curr_a = smem_a.load(layout=dot_a_layout)
        curr_b = smem_b.load(layout=dot_b_layout)
        curr_a_scales = smem_as.load(layout=linear_as)
        curr_b_scales = smem_bs.load(layout=linear_bs)

        accumulator = gl.amd.cdna4.mfma_scaled(
            a=curr_a,
            a_scale=curr_a_scales,
            a_format="e2m1",
            b=curr_b,
            b_scale=curr_b_scales,
            b_format="e2m1",
            acc=accumulator,
        )

        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, linear_mn)
        )
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, linear_mn)
        )
        offs_c = (
            stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        c = gl.convert_layout(c, layout=linear_mn, assert_trivial=False)
        gl.amd.cdna4.buffer_store(c, c_ptr, offs_c, c_mask)


_gemm_afp4wfp4_pingpong_repr = make_kernel_repr(
    "_gemm_afp4wfp4_pingpong_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_WARPS"],
)


@gluon.constexpr_function
def _copy_layout(shared_layout, num_warps):
    """Direct-to-LDS copy layout of a padded LDS tile, as Triton's
    CoalesceAsyncCopy makes it: 16 B for each lane, one LDS run for each wave."""
    bases = [list(b) for b in shared_layout.offset_bases]
    n_reg = 4  # log2(16 B)
    n_lane = 6  # log2(64 lanes)
    n_warp = int(math.log2(num_warps))
    warp = bases[n_reg + n_lane : n_reg + n_lane + n_warp]
    warp += [[0] * len(shared_layout.shape) for _ in range(n_warp - len(warp))]
    return gl.DistributedLinearLayout(
        reg_bases=bases[:n_reg] + bases[n_reg + n_lane + n_warp :],
        lane_bases=bases[n_reg : n_reg + n_lane],
        warp_bases=warp,
        block_bases=[],
        shape=list(shared_layout.shape),
    )


@gluon.jit
def _copy(smem, stage, ptr, offsets):
    async_copy.buffer_load_to_shared(smem.index(stage), ptr, offsets)


@gluon.jit
def _split_scales(scales, layout: gl.constexpr):
    """Split the scales of a tile into the top and bottom halves of the rows. In
    the MFMA scale layout, the half is one register bit, so this only selects
    registers."""
    rows: gl.constexpr = scales.shape[0] // 2
    halves = scales.reshape([2, rows, scales.shape[1]])
    top, bot = gl.split(gl.permute(halves, [1, 2, 0]))
    return gl.convert_layout(top, layout), gl.convert_layout(bot, layout)


@gluon.jit
def _mfma(a, a_scale, b, b_scale, acc):
    return gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_scale,
        a_format="e2m1",
        b=b,
        b_scale=b_scale,
        b_format="e2m1",
        acc=acc,
    )


@gluon.jit
def _store_quadrant(acc, c_ptr, row0, col0, M, N, stride_cm, stride_cn):
    c = acc.to(c_ptr.type.element_ty)
    layout: gl.constexpr = acc.type.layout
    offs_cm = row0 + gl.arange(0, acc.shape[0], layout=gl.SliceLayout(1, layout))
    offs_cn = col0 + gl.arange(0, acc.shape[1], layout=gl.SliceLayout(0, layout))
    offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna4.buffer_store(c, c_ptr, offs_c, mask=c_mask)


@gluon.jit(repr=_gemm_afp4wfp4_pingpong_repr)
def _gemm_afp4wfp4_pingpong_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)

    K is the packed size in bytes. The wrapper checks the limits: a 256x256x256
    tile with 8 warps, no split-K and no K tail.
    """
    gl.static_assert(BLOCK_SIZE_M == 256)
    gl.static_assert(BLOCK_SIZE_N == 256)
    gl.static_assert(BLOCK_SIZE_K == 256)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(stride_ak == 1)
    gl.static_assert(stride_bk == 1)
    gl.static_assert(stride_asm == 1)
    gl.static_assert(stride_bsn == 1)
    gl.assume(stride_am > 0)
    gl.assume(stride_bn > 0)
    gl.assume(stride_cm > 0)
    gl.assume(stride_ask > 0)
    gl.assume(stride_bsk > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    pid = gl.program_id(axis=0)
    # Remap so that each XCD gets a contiguous chunk of pids.
    pid = remap_xcd(pid, num_pid_m * num_pid_n, NUM_XCDS=NUM_XCDS)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: gl.constexpr = 32
    HALF_M: gl.constexpr = BLOCK_SIZE_M // 2
    HALF_N: gl.constexpr = BLOCK_SIZE_N // 2
    # Packed bytes and scales of one K step.
    BLOCK_K_BYTES: gl.constexpr = BLOCK_SIZE_K // 2
    BLOCK_K_SCALES: gl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_SIZE

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    # LDS holds A as top and bottom halves, B as left and right halves.
    shared_a: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_a_layout, [HALF_M, BLOCK_K_BYTES], gl.uint8
    )
    shared_b: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_b_layout, [BLOCK_K_BYTES, HALF_N], gl.uint8
    )
    shared_scales: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0, 1]
    )
    copy_a: gl.constexpr = _copy_layout(shared_a, NUM_WARPS)
    copy_b: gl.constexpr = _copy_layout(shared_b, NUM_WARPS)
    blocked_scales: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    # Scales of one K step, and of one half of the tile.
    linear_as: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_SIZE_M, BLOCK_K_SCALES]
    )
    linear_bs: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_SIZE_N, BLOCK_K_SCALES]
    )
    linear_as_half: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [HALF_M, BLOCK_K_SCALES]
    )
    linear_bs_half: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [HALF_N, BLOCK_K_SCALES]
    )

    # Loop-invariant copy offsets. Rows past M and columns past N wrap to valid
    # rows and columns, and the store masks their results.
    rows_a = gl.arange(0, HALF_M, layout=gl.SliceLayout(1, copy_a))
    offs_ak = gl.arange(0, BLOCK_K_BYTES, layout=gl.SliceLayout(0, copy_a))
    offs_am_top = (pid_m * BLOCK_SIZE_M + rows_a) % M
    offs_a_top = offs_am_top[:, None] * stride_am + offs_ak[None, :]
    offs_am_bot = (pid_m * BLOCK_SIZE_M + HALF_M + rows_a) % M
    offs_a_bot = offs_am_bot[:, None] * stride_am + offs_ak[None, :]
    offs_bk = gl.arange(0, BLOCK_K_BYTES, layout=gl.SliceLayout(1, copy_b))
    cols_b = gl.arange(0, HALF_N, layout=gl.SliceLayout(0, copy_b))
    offs_bn_left = (pid_n * BLOCK_SIZE_N + cols_b) % N
    offs_b_left = offs_bk[:, None] + offs_bn_left[None, :] * stride_bn
    offs_bn_right = (pid_n * BLOCK_SIZE_N + HALF_N + cols_b) % N
    offs_b_right = offs_bk[:, None] + offs_bn_right[None, :] * stride_bn
    offs_asm = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_scales))
    ) % M
    offs_bsn = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(1, blocked_scales))
    ) % N
    offs_ks = gl.arange(0, BLOCK_K_SCALES, layout=gl.SliceLayout(0, blocked_scales))
    offs_as = offs_asm[:, None] + offs_ks[None, :] * stride_ask
    # B scales are N x K even though B operand is K x N.
    offs_bs = offs_bsn[:, None] + offs_ks[None, :] * stride_bsk

    # Create shared memories: two LDS stages of each half and of the scales.
    smem_a_top = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, HALF_M, BLOCK_K_BYTES], layout=shared_a
    )
    smem_a_bot = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, HALF_M, BLOCK_K_BYTES], layout=shared_a
    )
    smem_b_left = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_K_BYTES, HALF_N], layout=shared_b
    )
    smem_b_right = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_K_BYTES, HALF_N], layout=shared_b
    )
    smem_as = gl.allocate_shared_memory(
        a_scales_ptr.type.element_ty,
        [2, BLOCK_SIZE_M, BLOCK_K_SCALES],
        layout=shared_scales,
    )
    smem_bs = gl.allocate_shared_memory(
        b_scales_ptr.type.element_ty,
        [2, BLOCK_SIZE_N, BLOCK_K_SCALES],
        layout=shared_scales,
    )

    num_k_iter = K // BLOCK_K_BYTES
    k0 = num_k_iter % 2  # peel K step 0 when the number of K steps is odd
    acc_tl = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=mfma_layout)
    acc_bl = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=mfma_layout)
    acc_tr = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=mfma_layout)
    acc_br = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=mfma_layout)

    if k0 == 1:
        # Peeled K step 0 in stage 0, lockstep.
        _copy(smem_as, 0, a_scales_ptr, offs_as)
        _copy(smem_bs, 0, b_scales_ptr, offs_bs)
        _copy(smem_a_top, 0, a_ptr, offs_a_top)
        _copy(smem_a_bot, 0, a_ptr, offs_a_bot)
        _copy(smem_b_left, 0, b_ptr, offs_b_left)
        _copy(smem_b_right, 0, b_ptr, offs_b_right)
        async_copy.commit_group()
        async_copy.wait_group(0)
        gl.barrier()
        a_scales = smem_as.index(0).load(linear_as)
        b_scales = smem_bs.index(0).load(linear_bs)
        a_top = smem_a_top.index(0).load(dot_a_layout)
        a_bot = smem_a_bot.index(0).load(dot_a_layout)
        b_left = smem_b_left.index(0).load(dot_b_layout)
        b_right = smem_b_right.index(0).load(dot_b_layout)
        as_top, as_bot = _split_scales(a_scales, linear_as_half)
        bs_left, bs_right = _split_scales(b_scales, linear_bs_half)
        acc_tl = _mfma(a_top, as_top, b_left, bs_left, acc_tl)
        acc_bl = _mfma(a_bot, as_bot, b_left, bs_left, acc_bl)
        acc_tr = _mfma(a_top, as_top, b_right, bs_right, acc_tr)
        acc_br = _mfma(a_bot, as_bot, b_right, bs_right, acc_br)
        gl.barrier()  # WAR: the prologue below refills stage 0

    if num_k_iter - k0 >= 2:
        # Prologue: K steps k0 and k0 + 1 into stages 0 and 1, in the commit
        # groups of the loop: {B_left, B scales}, {A_top, A scales}, {A_bot},
        # {B_right}.
        for s in gl.static_range(2):
            _copy(smem_b_left, s, b_ptr + (k0 + s) * BLOCK_K_BYTES, offs_b_left)
            _copy(
                smem_bs,
                s,
                b_scales_ptr + (k0 + s) * (BLOCK_K_SCALES * stride_bsk),
                offs_bs,
            )
            async_copy.commit_group()
            _copy(smem_a_top, s, a_ptr + (k0 + s) * BLOCK_K_BYTES, offs_a_top)
            _copy(
                smem_as,
                s,
                a_scales_ptr + (k0 + s) * (BLOCK_K_SCALES * stride_ask),
                offs_as,
            )
            async_copy.commit_group()
            _copy(smem_a_bot, s, a_ptr + (k0 + s) * BLOCK_K_BYTES, offs_a_bot)
            async_copy.commit_group()
            _copy(smem_b_right, s, b_ptr + (k0 + s) * BLOCK_K_BYTES, offs_b_right)
            async_copy.commit_group()
        async_copy.wait_group(6)  # the first two groups of stage 0 are in LDS
        gl.barrier()
        b_left = smem_b_left.index(0).load(dot_b_layout)
        bs_left, bs_right = _split_scales(
            smem_bs.index(0).load(linear_bs), linear_bs_half
        )
        a_top = smem_a_top.index(0).load(dot_a_layout)
        as_top, as_bot = _split_scales(smem_as.index(0).load(linear_as), linear_as_half)

        kr = k0 + 2  # K step that refills stage 0; stage 1 gets kr + 1
        for _ in range((num_k_iter - k0 - 2) // 2):
            for stage in gl.static_range(2):
                # K step kr - 2 + stage from LDS stage `stage`. The memory
                # stages refill `stage` with K step kr + stage and read the
                # operands of the next K step from the other stage.
                async_copy.wait_group(5)
                with warp_pipeline_stage("mfma", priority=0):
                    acc_tl = _mfma(a_top, as_top, b_left, bs_left, acc_tl)
                with warp_pipeline_stage("mem", priority=1):
                    a_bot = smem_a_bot.index(stage).load(dot_a_layout)
                    kt = kr + stage
                    _copy(smem_b_left, stage, b_ptr + kt * BLOCK_K_BYTES, offs_b_left)
                    _copy(
                        smem_bs,
                        stage,
                        b_scales_ptr + kt * (BLOCK_K_SCALES * stride_bsk),
                        offs_bs,
                    )
                    async_copy.commit_group()
                async_copy.wait_group(5)
                with warp_pipeline_stage("mfma", priority=0):
                    acc_bl = _mfma(a_bot, as_bot, b_left, bs_left, acc_bl)
                with warp_pipeline_stage("mem", priority=1):
                    b_right = smem_b_right.index(stage).load(dot_b_layout)
                    _copy(smem_a_top, stage, a_ptr + kt * BLOCK_K_BYTES, offs_a_top)
                    _copy(
                        smem_as,
                        stage,
                        a_scales_ptr + kt * (BLOCK_K_SCALES * stride_ask),
                        offs_as,
                    )
                    async_copy.commit_group()
                async_copy.wait_group(5)
                with warp_pipeline_stage("mfma", priority=0):
                    acc_tr = _mfma(a_top, as_top, b_right, bs_right, acc_tr)
                with warp_pipeline_stage("mem", priority=1):
                    b_left = smem_b_left.index(1 - stage).load(dot_b_layout)
                    bs_left_next, bs_right_next = _split_scales(
                        smem_bs.index(1 - stage).load(linear_bs), linear_bs_half
                    )
                    _copy(smem_a_bot, stage, a_ptr + kt * BLOCK_K_BYTES, offs_a_bot)
                    async_copy.commit_group()
                async_copy.wait_group(5)
                with warp_pipeline_stage("mfma", priority=0):
                    acc_br = _mfma(a_bot, as_bot, b_right, bs_right, acc_br)
                with warp_pipeline_stage("mem", priority=1):
                    a_top = smem_a_top.index(1 - stage).load(dot_a_layout)
                    as_top, as_bot = _split_scales(
                        smem_as.index(1 - stage).load(linear_as), linear_as_half
                    )
                    _copy(smem_b_right, stage, b_ptr + kt * BLOCK_K_BYTES, offs_b_right)
                    async_copy.commit_group()
                    if stage == 1:
                        kr += 2
                bs_left, bs_right = bs_left_next, bs_right_next

        # Drain: K steps num_k_iter - 2 (stage 0) and num_k_iter - 1
        # (stage 1), lockstep.
        async_copy.wait_group(0)
        gl.barrier()
        a_bot = smem_a_bot.index(0).load(dot_a_layout)
        b_right = smem_b_right.index(0).load(dot_b_layout)
        acc_tl = _mfma(a_top, as_top, b_left, bs_left, acc_tl)
        acc_bl = _mfma(a_bot, as_bot, b_left, bs_left, acc_bl)
        acc_tr = _mfma(a_top, as_top, b_right, bs_right, acc_tr)
        acc_br = _mfma(a_bot, as_bot, b_right, bs_right, acc_br)
        a_scales = smem_as.index(1).load(linear_as)
        b_scales = smem_bs.index(1).load(linear_bs)
        a_top = smem_a_top.index(1).load(dot_a_layout)
        a_bot = smem_a_bot.index(1).load(dot_a_layout)
        b_left = smem_b_left.index(1).load(dot_b_layout)
        b_right = smem_b_right.index(1).load(dot_b_layout)
        as_top, as_bot = _split_scales(a_scales, linear_as_half)
        bs_left, bs_right = _split_scales(b_scales, linear_bs_half)
        acc_tl = _mfma(a_top, as_top, b_left, bs_left, acc_tl)
        acc_bl = _mfma(a_bot, as_bot, b_left, bs_left, acc_bl)
        acc_tr = _mfma(a_top, as_top, b_right, bs_right, acc_tr)
        acc_br = _mfma(a_bot, as_bot, b_right, bs_right, acc_br)

    row0 = pid_m * BLOCK_SIZE_M
    col0 = pid_n * BLOCK_SIZE_N
    _store_quadrant(acc_tl, c_ptr, row0, col0, M, N, stride_cm, stride_cn)
    _store_quadrant(acc_bl, c_ptr, row0 + HALF_M, col0, M, N, stride_cm, stride_cn)
    _store_quadrant(acc_tr, c_ptr, row0, col0 + HALF_N, M, N, stride_cm, stride_cn)
    _store_quadrant(
        acc_br, c_ptr, row0 + HALF_M, col0 + HALF_N, M, N, stride_cm, stride_cn
    )


# Triton main renamed shared_memory_descriptor._reinterpret to reinterpret.
_REINTERPRET_IS_PUBLIC = gl.constexpr(
    hasattr(gl.shared_memory_descriptor, "reinterpret")
)

_gemm_afp4wfp4_preshuffle_pingpong_repr = make_kernel_repr(
    "_gemm_afp4wfp4_preshuffle_pingpong_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_WARPS"],
)


@gluon.jit
def _load_preshuffled_b(smem, stage, layout: gl.constexpr):
    """Load one LDS stage of a B half as (K bytes, N). The stage holds 16-row
    blocks of shuffle_weight(layout=(16, 16)), and the view undoes the shuffle."""
    tile = smem.index(stage)
    n16: gl.constexpr = tile.shape[0]
    k16: gl.constexpr = tile.shape[1] // 256
    tile = (
        tile.reshape((n16, k16, 16, 16))
        .permute((0, 2, 1, 3))
        .reshape((n16 * 16, k16 * 16))
        .permute((1, 0))
    )
    return tile.load(layout)


@gluon.constexpr_function
def _preshuffled_scale_word_layout(layout):
    """Layout of the int32 words of a preshuffled scale stage (stripes, 64) for
    the 16x16x128 MFMA scale layout `layout` of (rows, K groups): lane l reads
    word l of a stripe, and registers and warps select stripes."""
    lanes = [list(b) for b in layout.lane_bases]
    assert lanes == [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], layout
    assert [list(b) for b in layout.reg_bases[:2]] == [[0, 4], [16, 0]], layout
    stripes = []
    for bases in (layout.reg_bases[2:], layout.warp_bases):
        out = []
        for row, group in bases:
            assert group == 0 and row % 32 == 0, layout
            out.append([row // 32, 0])
        stripes.append(out)
    return gl.DistributedLinearLayout(
        reg_bases=stripes[0],
        lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
        warp_bases=stripes[1],
        block_bases=[],
        shape=[layout.shape[0] // 32, 64],
    )


@gluon.jit
def _load_preshuffled_scales(smem, stage, layout: gl.constexpr):
    """Load the scales of one preshuffled LDS stage in the 16x16x128 MFMA scale
    layout. Byte 2 * (K group bit 2) + (row bit 4) of word l of a stripe is the
    scale of row l % 16 and K group l // 16 (+ 16 rows, + 4 groups)."""
    tile = smem.index(stage)
    gl.static_assert(tile.shape[1] == 256)
    stripes: gl.constexpr = tile.shape[0]
    words_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    if _REINTERPRET_IS_PUBLIC:
        words = tile.reinterpret(gl.int32, [stripes, 64], words_layout)
    else:
        words = tile._reinterpret(gl.int32, [stripes, 64], words_layout)
    w = words.load(_preshuffled_scale_word_layout(layout))
    # (stripe, word, row bit 4, K group bit 2). The MFMA lowering packs the
    # bytes in the order 0, 2, 1, 3 with one v_perm.
    b = gl.join(
        gl.join(w.to(gl.uint8), (w >> 8).to(gl.uint8)),
        gl.join((w >> 16).to(gl.uint8), (w >> 24).to(gl.uint8)),
    )
    b = b.reshape((stripes, 4, 16, 2, 2)).permute((0, 3, 2, 4, 1))
    return gl.convert_layout(b.reshape((stripes * 32, 8)), layout, assert_trivial=True)


@gluon.jit(repr=_gemm_afp4wfp4_preshuffle_pingpong_repr)
def _gemm_afp4wfp4_preshuffle_pingpong_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B with preshuffled B and scales.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)

    K is the packed size in bytes. The wrapper checks the limits: a 256x256x256
    tile with 8 warps, no split-K and no K tail.

    B is the (N // 16, 16 * K) view of shuffle_weight(layout=(16, 16)), and
    stride_bn is its row stride. The scales are the (rows // 32, 2 * K) views of
    shuffle_scale, and stride_asm and stride_bsn are their row strides.
    """
    gl.static_assert(BLOCK_SIZE_M == 256)
    gl.static_assert(BLOCK_SIZE_N == 256)
    gl.static_assert(BLOCK_SIZE_K == 256)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(stride_ak == 1)
    gl.static_assert(stride_bk == 1)
    gl.static_assert(stride_ask == 1)
    gl.static_assert(stride_bsk == 1)
    gl.assume(stride_am > 0)
    gl.assume(stride_bn > 0)
    gl.assume(stride_cm > 0)
    gl.assume(stride_asm > 0)
    gl.assume(stride_bsn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    pid = gl.program_id(axis=0)
    # Remap so that each XCD gets a contiguous chunk of pids.
    pid = remap_xcd(pid, num_pid_m * num_pid_n, NUM_XCDS=NUM_XCDS)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: gl.constexpr = 32
    HALF_M: gl.constexpr = BLOCK_SIZE_M // 2
    HALF_N: gl.constexpr = BLOCK_SIZE_N // 2
    # Packed bytes and scales of one K step.
    BLOCK_K_BYTES: gl.constexpr = BLOCK_SIZE_K // 2
    BLOCK_K_SCALES: gl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_SIZE

    # 16x16x128 with two tiles of each warp in M and N: one 32-row stripe of
    # shuffle_scale holds the scales of both tiles.
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
        tiles_per_warp=[2, 2],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    # LDS holds A as top and bottom halves, B as left and right halves.
    shared_a: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_a_layout, [HALF_M, BLOCK_K_BYTES], gl.uint8
    )
    copy_a: gl.constexpr = _copy_layout(shared_a, NUM_WARPS)
    # LDS keeps the preshuffled bytes: a B half as 16-row blocks of N and the
    # scales as 32-row stripes. Each lane copies 16 B of B and 4 B of scales.
    B_ROWS: gl.constexpr = HALF_N // 16
    B_COLS: gl.constexpr = BLOCK_K_BYTES * 16
    SCALE_ROWS: gl.constexpr = BLOCK_SIZE_M // 32
    SCALE_COLS: gl.constexpr = BLOCK_K_SCALES * 32
    B_K_STEP: gl.constexpr = B_COLS
    SCALE_K_STEP: gl.constexpr = SCALE_COLS
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    shared_scales: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    copy_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[1, 64],
        warps_per_cta=[NUM_WARPS // 2, 2],
        order=[1, 0],
    )
    blocked_scales: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[1, 64],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    # Scales of one K step, and of one half of the tile.
    linear_as: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_SIZE_M, BLOCK_K_SCALES]
    )
    linear_bs: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_SIZE_N, BLOCK_K_SCALES]
    )
    linear_as_half: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [HALF_M, BLOCK_K_SCALES]
    )
    linear_bs_half: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [HALF_N, BLOCK_K_SCALES]
    )

    # Loop-invariant copy offsets. Rows past M and columns past N wrap to valid
    # rows and columns, and the store masks their results.
    rows_a = gl.arange(0, HALF_M, layout=gl.SliceLayout(1, copy_a))
    offs_ak = gl.arange(0, BLOCK_K_BYTES, layout=gl.SliceLayout(0, copy_a))
    offs_am_top = (pid_m * BLOCK_SIZE_M + rows_a) % M
    offs_a_top = offs_am_top[:, None] * stride_am + offs_ak[None, :]
    offs_am_bot = (pid_m * BLOCK_SIZE_M + HALF_M + rows_a) % M
    offs_a_bot = offs_am_bot[:, None] * stride_am + offs_ak[None, :]
    # A row of B is a block of 16 columns, and a row of scales is a stripe of
    # 32 rows. Blocks and stripes wrap like rows and columns.
    blocks_b = gl.arange(0, B_ROWS, layout=gl.SliceLayout(1, copy_b))
    offs_bk = gl.arange(0, B_COLS, layout=gl.SliceLayout(0, copy_b))
    offs_bn_left = (pid_n * (BLOCK_SIZE_N // 16) + blocks_b) % (N // 16)
    offs_b_left = offs_bn_left[:, None] * stride_bn + offs_bk[None, :]
    offs_bn_right = (pid_n * (BLOCK_SIZE_N // 16) + B_ROWS + blocks_b) % (N // 16)
    offs_b_right = offs_bn_right[:, None] * stride_bn + offs_bk[None, :]
    stripes = gl.arange(0, SCALE_ROWS, layout=gl.SliceLayout(1, blocked_scales))
    offs_ks = gl.arange(0, SCALE_COLS, layout=gl.SliceLayout(0, blocked_scales))
    offs_asm = (pid_m * SCALE_ROWS + stripes) % gl.cdiv(M, 32)
    offs_as = offs_asm[:, None] * stride_asm + offs_ks[None, :]
    offs_bsn = (pid_n * SCALE_ROWS + stripes) % gl.cdiv(N, 32)
    offs_bs = offs_bsn[:, None] * stride_bsn + offs_ks[None, :]

    # Create shared memories: two LDS stages of each half and of the scales.
    smem_a_top = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, HALF_M, BLOCK_K_BYTES], layout=shared_a
    )
    smem_a_bot = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, HALF_M, BLOCK_K_BYTES], layout=shared_a
    )
    smem_b_left = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, B_ROWS, B_COLS], layout=shared_b
    )
    smem_b_right = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, B_ROWS, B_COLS], layout=shared_b
    )
    smem_as = gl.allocate_shared_memory(
        a_scales_ptr.type.element_ty,
        [2, SCALE_ROWS, SCALE_COLS],
        layout=shared_scales,
    )
    smem_bs = gl.allocate_shared_memory(
        b_scales_ptr.type.element_ty,
        [2, SCALE_ROWS, SCALE_COLS],
        layout=shared_scales,
    )

    num_k_iter = K // BLOCK_K_BYTES
    k0 = num_k_iter % 2  # peel K step 0 when the number of K steps is odd
    acc_tl = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=mfma_layout)
    acc_bl = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=mfma_layout)
    acc_tr = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=mfma_layout)
    acc_br = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=mfma_layout)

    if k0 == 1:
        # Peeled K step 0 in stage 0, lockstep.
        _copy(smem_as, 0, a_scales_ptr, offs_as)
        _copy(smem_bs, 0, b_scales_ptr, offs_bs)
        _copy(smem_a_top, 0, a_ptr, offs_a_top)
        _copy(smem_a_bot, 0, a_ptr, offs_a_bot)
        _copy(smem_b_left, 0, b_ptr, offs_b_left)
        _copy(smem_b_right, 0, b_ptr, offs_b_right)
        async_copy.commit_group()
        async_copy.wait_group(0)
        gl.barrier()
        a_scales = _load_preshuffled_scales(smem_as, 0, linear_as)
        b_scales = _load_preshuffled_scales(smem_bs, 0, linear_bs)
        a_top = smem_a_top.index(0).load(dot_a_layout)
        a_bot = smem_a_bot.index(0).load(dot_a_layout)
        b_left = _load_preshuffled_b(smem_b_left, 0, dot_b_layout)
        b_right = _load_preshuffled_b(smem_b_right, 0, dot_b_layout)
        as_top, as_bot = _split_scales(a_scales, linear_as_half)
        bs_left, bs_right = _split_scales(b_scales, linear_bs_half)
        acc_tl = _mfma(a_top, as_top, b_left, bs_left, acc_tl)
        acc_bl = _mfma(a_bot, as_bot, b_left, bs_left, acc_bl)
        acc_tr = _mfma(a_top, as_top, b_right, bs_right, acc_tr)
        acc_br = _mfma(a_bot, as_bot, b_right, bs_right, acc_br)
        gl.barrier()  # WAR: the prologue below refills stage 0

    if num_k_iter - k0 >= 2:
        # Prologue: K steps k0 and k0 + 1 into stages 0 and 1, in the commit
        # groups of the loop: {B_left, B scales}, {A_top, A scales}, {A_bot},
        # {B_right}.
        for s in gl.static_range(2):
            _copy(smem_b_left, s, b_ptr + (k0 + s) * B_K_STEP, offs_b_left)
            _copy(
                smem_bs,
                s,
                b_scales_ptr + (k0 + s) * (SCALE_K_STEP * stride_bsk),
                offs_bs,
            )
            async_copy.commit_group()
            _copy(smem_a_top, s, a_ptr + (k0 + s) * BLOCK_K_BYTES, offs_a_top)
            _copy(
                smem_as,
                s,
                a_scales_ptr + (k0 + s) * (SCALE_K_STEP * stride_ask),
                offs_as,
            )
            async_copy.commit_group()
            _copy(smem_a_bot, s, a_ptr + (k0 + s) * BLOCK_K_BYTES, offs_a_bot)
            async_copy.commit_group()
            _copy(smem_b_right, s, b_ptr + (k0 + s) * B_K_STEP, offs_b_right)
            async_copy.commit_group()
        async_copy.wait_group(6)  # the first two groups of stage 0 are in LDS
        gl.barrier()
        b_left = _load_preshuffled_b(smem_b_left, 0, dot_b_layout)
        bs_left, bs_right = _split_scales(
            _load_preshuffled_scales(smem_bs, 0, linear_bs), linear_bs_half
        )
        a_top = smem_a_top.index(0).load(dot_a_layout)
        as_top, as_bot = _split_scales(
            _load_preshuffled_scales(smem_as, 0, linear_as), linear_as_half
        )

        kr = k0 + 2  # K step that refills stage 0; stage 1 gets kr + 1
        for _ in range((num_k_iter - k0 - 2) // 2):
            for stage in gl.static_range(2):
                # K step kr - 2 + stage from LDS stage `stage`. The memory
                # stages refill `stage` with K step kr + stage and read the
                # operands of the next K step from the other stage.
                async_copy.wait_group(5)
                with warp_pipeline_stage("mfma", priority=0):
                    acc_tl = _mfma(a_top, as_top, b_left, bs_left, acc_tl)
                with warp_pipeline_stage("mem", priority=1):
                    a_bot = smem_a_bot.index(stage).load(dot_a_layout)
                    kt = kr + stage
                    _copy(smem_b_left, stage, b_ptr + kt * B_K_STEP, offs_b_left)
                    _copy(
                        smem_bs,
                        stage,
                        b_scales_ptr + kt * (SCALE_K_STEP * stride_bsk),
                        offs_bs,
                    )
                    async_copy.commit_group()
                async_copy.wait_group(5)
                with warp_pipeline_stage("mfma", priority=0):
                    acc_bl = _mfma(a_bot, as_bot, b_left, bs_left, acc_bl)
                with warp_pipeline_stage("mem", priority=1):
                    b_right = _load_preshuffled_b(smem_b_right, stage, dot_b_layout)
                    _copy(smem_a_top, stage, a_ptr + kt * BLOCK_K_BYTES, offs_a_top)
                    _copy(
                        smem_as,
                        stage,
                        a_scales_ptr + kt * (SCALE_K_STEP * stride_ask),
                        offs_as,
                    )
                    async_copy.commit_group()
                async_copy.wait_group(5)
                with warp_pipeline_stage("mfma", priority=0):
                    acc_tr = _mfma(a_top, as_top, b_right, bs_right, acc_tr)
                with warp_pipeline_stage("mem", priority=1):
                    b_left = _load_preshuffled_b(smem_b_left, 1 - stage, dot_b_layout)
                    bs_left_next, bs_right_next = _split_scales(
                        _load_preshuffled_scales(smem_bs, 1 - stage, linear_bs),
                        linear_bs_half,
                    )
                    _copy(smem_a_bot, stage, a_ptr + kt * BLOCK_K_BYTES, offs_a_bot)
                    async_copy.commit_group()
                async_copy.wait_group(5)
                with warp_pipeline_stage("mfma", priority=0):
                    acc_br = _mfma(a_bot, as_bot, b_right, bs_right, acc_br)
                with warp_pipeline_stage("mem", priority=1):
                    a_top = smem_a_top.index(1 - stage).load(dot_a_layout)
                    as_top, as_bot = _split_scales(
                        _load_preshuffled_scales(smem_as, 1 - stage, linear_as),
                        linear_as_half,
                    )
                    _copy(smem_b_right, stage, b_ptr + kt * B_K_STEP, offs_b_right)
                    async_copy.commit_group()
                    if stage == 1:
                        kr += 2
                bs_left, bs_right = bs_left_next, bs_right_next

        # Drain: K steps num_k_iter - 2 (stage 0) and num_k_iter - 1
        # (stage 1), lockstep.
        async_copy.wait_group(0)
        gl.barrier()
        a_bot = smem_a_bot.index(0).load(dot_a_layout)
        b_right = _load_preshuffled_b(smem_b_right, 0, dot_b_layout)
        acc_tl = _mfma(a_top, as_top, b_left, bs_left, acc_tl)
        acc_bl = _mfma(a_bot, as_bot, b_left, bs_left, acc_bl)
        acc_tr = _mfma(a_top, as_top, b_right, bs_right, acc_tr)
        acc_br = _mfma(a_bot, as_bot, b_right, bs_right, acc_br)
        a_scales = _load_preshuffled_scales(smem_as, 1, linear_as)
        b_scales = _load_preshuffled_scales(smem_bs, 1, linear_bs)
        a_top = smem_a_top.index(1).load(dot_a_layout)
        a_bot = smem_a_bot.index(1).load(dot_a_layout)
        b_left = _load_preshuffled_b(smem_b_left, 1, dot_b_layout)
        b_right = _load_preshuffled_b(smem_b_right, 1, dot_b_layout)
        as_top, as_bot = _split_scales(a_scales, linear_as_half)
        bs_left, bs_right = _split_scales(b_scales, linear_bs_half)
        acc_tl = _mfma(a_top, as_top, b_left, bs_left, acc_tl)
        acc_bl = _mfma(a_bot, as_bot, b_left, bs_left, acc_bl)
        acc_tr = _mfma(a_top, as_top, b_right, bs_right, acc_tr)
        acc_br = _mfma(a_bot, as_bot, b_right, bs_right, acc_br)

    row0 = pid_m * BLOCK_SIZE_M
    col0 = pid_n * BLOCK_SIZE_N
    _store_quadrant(acc_tl, c_ptr, row0, col0, M, N, stride_cm, stride_cn)
    _store_quadrant(acc_bl, c_ptr, row0 + HALF_M, col0, M, N, stride_cm, stride_cn)
    _store_quadrant(acc_tr, c_ptr, row0, col0 + HALF_N, M, N, stride_cm, stride_cn)
    _store_quadrant(
        acc_br, c_ptr, row0 + HALF_M, col0 + HALF_N, M, N, stride_cm, stride_cn
    )


_KERNEL_MAP = {
    "bandwidth_bound": _gemm_afp4wfp4_kernel,
    "compute_bound": _gemm_afp4wfp4_pingpong_kernel,
}

_PRESHUFFLE_KERNEL_MAP = {
    "compute_bound": _gemm_afp4wfp4_preshuffle_pingpong_kernel,
}
