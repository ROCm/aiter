# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from triton import language as tl

_LOGGER = AiterTritonLogger()
from triton.experimental import gluon
from triton.experimental.gluon import language as gl



@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["SPLITK_BLOCK_SIZE"]) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
#def _gemm_a16_w16_kernel_gluon(
def _gemm_a16_w16_kernel(
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
    stride_ck,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    cache_modifier: gl.constexpr,
    activation: gl.constexpr,
    use_activation: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    SKIP_REDUCE: gl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    gl.assume(stride_am > 0)
    gl.assume(stride_ak > 0)
    gl.assume(stride_bk > 0)
    gl.assume(stride_bn > 0)
    gl.assume(stride_ck > 0)
    gl.assume(stride_cm > 0)
    gl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = gl.program_id(axis=0)
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

    gl.assume(pid_m >= 0)
    gl.assume(pid_n >= 0)
    gl.assume(pid_k >= 0)
    
    blocked_a: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,1),(0,2), (0,4),(8,0), (128, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (16, 0), (32, 0), (64, 0)),
        warp_bases=((1, 0), (2, 0), (4, 0)),
        block_bases=[],
        shape=[256, 64],
    )

    blocked_b: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1,0),(2,0),(4,0),(0,8), (0, 128)),
        lane_bases=((8, 0), (16, 0), (32, 0), (0, 16), (0, 32), (0, 64)),
        warp_bases=((0, 1), (0, 2), (0, 4)),
        block_bases=[],
        shape=[64, 256],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 4],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    shared_a: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs = [[512,16]],
        offset_bases = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16,0], [32,0], [64,0], [1,0], [2,0], [4,0], [8,0], [128,0]],
        block_bases = [],
        shape = [256, 64]
    )
    shared_b: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs = [[512,16]],
        offset_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 128]],
        block_bases = [],
        shape = [64, 256]
    )

    # a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    a_bufs = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    b_bufs = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )


    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        # Create pointers for first block of A and B input matrices
        offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))
        offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_b))
        offs_ak_split = split_k_start + offs_ak
        offs_bk_split = split_k_start + offs_bk
        offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N

        a_offs = offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak
        b_offs = offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
        if ADD_BIAS:
            if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                accumulator = gl.load(bias_ptr + offs_bn).to(dtype=acc_dtype)
                accumulator = gl.broadcast_to(
                    accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                )
            else:
                accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout)
        else:
            accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout)

        split_k_end = gl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = gl.cdiv(k_span, BLOCK_SIZE_K)

        # prologue
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_bufs.index(0), a_ptr, a_offs)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_bufs.index(0), b_ptr, b_offs)
        gl.amd.cdna4.async_copy.commit_group()
        a_offs += BLOCK_SIZE_K * stride_ak
        b_offs += BLOCK_SIZE_K * stride_bk

        gl.amd.cdna4.async_copy.wait_group(0)
        gl.amd.cdna3.s_barrier()

        if gl.amd.cdna3.warp_id() >= 4:
            gl.amd.cdna3.s_barrier()

        gl.assume(num_k_iter > 2)
        # mainloop
        buf_idx = 0
        # mainloop
        for k in range(num_k_iter-1):
            async_idx = (buf_idx + 1) % 2
            gl.amd.cdna4.async_copy.buffer_load_to_shared(a_bufs.index(async_idx), a_ptr, a_offs)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(b_bufs.index(async_idx), b_ptr, b_offs)
            gl.amd.cdna4.async_copy.commit_group()

            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_bufs.index(buf_idx), dot_a_layout)
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_bufs.index(buf_idx), dot_b_layout)
            #cur_a = a_bufs.index(buf_idx).load(layout=dot_a_layout)
            #cur_b = b_bufs.index(buf_idx).load(layout=dot_b_layout)

            gl.amd.cdna3.sched_barrier(0)
            gl.amd.cdna4.async_copy.wait_group(0) ## + lgkmcnt(0)
            gl.amd.cdna3.s_barrier()
            gl.amd.cdna3.sched_barrier(0)

            accumulator = gl.amd.cdna4.mfma(cur_a, cur_b, accumulator)

            a_offs += BLOCK_SIZE_K * stride_ak
            b_offs += BLOCK_SIZE_K * stride_bk
            buf_idx = (buf_idx + 1) % 2
            gl.amd.cdna3.sched_barrier(0)
            gl.amd.cdna3.s_barrier()

        # epilogue
        if gl.amd.cdna3.warp_id() < 4:
            gl.amd.cdna3.s_barrier()
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.amd.cdna3.s_barrier()
        cur_a = a_bufs.index(buf_idx).load(layout=dot_a_layout)
        cur_b = b_bufs.index(buf_idx).load(layout=dot_b_layout)
        accumulator = gl.amd.cdna4.mfma(cur_a, cur_b, accumulator)

        if use_activation and NUM_KSPLIT == 1:
            accumulator = activation(accumulator)

        # Write back the block of the output matrix C with masks.
        c = accumulator.to(c_ptr.type.element_ty)
        #offs_cm = pid_m.to(gl.int64) * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
        #offs_cn = pid_n.to(gl.int64) * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
        c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_k * stride_ck

        # c_ptrs = c_ptr + c_offs
        # c_ptrs = (
        #     c_ptr
        #     + stride_cm * offs_cm[:, None]
        #     + stride_cn * offs_cn[None, :]
        #     + pid_k * stride_ck
        # )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        gl.amd.cdna4.buffer_store(
            stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask
        )

