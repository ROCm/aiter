# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon MXFP4 GEMM kernel for gfx950."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd


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
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

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
