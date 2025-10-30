# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, List
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
import triton.experimental.gluon.language.amd.cdna4.async_copy as acp


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _batched_gemm_a8w8_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_ascaleb,
    stride_bscaleb,
    stride_biasb,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    BUFFER_SIZE: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_a8w8 function
    below

    Computes the matmul C[i] = A[i] x B[i] and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K).
    - B: Batch tensor B with shape (B, K, N).
    - C: Batch tensor C with shape (B, M, N).
    - A_scale: First scale batch tensor with shape (B, M, 1).
    - B_scale: Second scale batch tensor with shape (B, 1, N).
    - Bias: Bias batch tensor with shape (B, 1, N).
    """

    # -----------------------------------------------------------
    # Get batch program id
    batch_id = gl.program_id(axis=0)
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = gl.program_id(axis=1)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    remap_xcd(pid, GRID_MN)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)

    gl.assume(pid_m >= 0)
    gl.assume(pid_n >= 0)

    # Cast batch id and batch dimension strides to int64 to avoid int32 overflow during offset calculation
    # Note: If you're attempting to cast strides to int64 to prevent integer overflow, use `tl.cast` instead of `.to()`.
    # See https://github.com/ROCm/aiter/pull/597 for rationale
    batch_id = tl.cast(batch_id, gl.int64)
    stride_ab = tl.cast(stride_ab, gl.int64)
    stride_bb = tl.cast(stride_bb, gl.int64)
    stride_cb = tl.cast(stride_cb, gl.int64)

    # Create layouts for contiguous access
    elems_per_thread_mk: gl.constexpr = triton.cdiv(
        BLOCK_SIZE_M * BLOCK_SIZE_K // (gl.num_warps() * 64), BUFFER_SIZE
    )
    elems_per_thread_kn: gl.constexpr = triton.cdiv(
        BLOCK_SIZE_K * BLOCK_SIZE_N // (gl.num_warps() * 64), BUFFER_SIZE
    )
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[elems_per_thread_mk, 16],
        threads_per_warp=[16, 4],
        warps_per_cta=[gl.num_warps(), 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, elems_per_thread_kn],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, gl.num_warps()],
        order=[0, 1],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 64],
        transposed=True,
        warps_per_cta=[2, gl.num_warps() // 2],
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=4, max_phase=4, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=4, max_phase=4, order=[0, 1]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    # Create offsets for first block of A and B input matrices
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))
    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
    )
    offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
    )
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Load first block of A
    if EVEN_K:
        a = gl.amd.cdna4.buffer_load(
            ptr=(a_ptr + batch_id * stride_ab),
            offsets=offs_a,
            mask=(offs_am[:, None] < M),
            other=0.0,
        )
    else:
        a = gl.amd.cdna4.buffer_load(
            ptr=(a_ptr + batch_id * stride_ab),
            offsets=offs_a,
            mask=((offs_ak[None, :] < K) & (offs_am[:, None] < M)),
            other=0.0,
        )

    # Load first block of B
    if EVEN_K:
        b = gl.amd.cdna4.buffer_load(
            ptr=(b_ptr + batch_id * stride_bb),
            offsets=offs_b,
            mask=(offs_bn[None, :] < N),
            other=0.0,
        )
    else:
        b = gl.amd.cdna4.buffer_load(
            ptr=(b_ptr + batch_id * stride_bb),
            offsets=offs_b,
            mask=((offs_bk[:, None] < K) & (offs_bn[None, :] < N)),
            other=0.0,
        )

    # Load the scales
    offs_a_scale = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_b_scale = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=(a_scale_ptr + batch_id * stride_ascaleb),
        offsets=offs_a_scale,
        mask=(offs_a_scale < M),
        other=1.0,
    )
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=(b_scale_ptr + batch_id * stride_bscaleb),
        offsets=offs_b_scale,
        mask=(offs_b_scale < N),
        other=1.0,
    )

    # Allocate shared memory input matrices and scaling factors
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    # Store first block of A and B in shared memory
    if num_k_iter > 1:
        smem_a.store(a)
        smem_b.store(b)

    # Run the main loop
    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    accumulator = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
    )
    zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.int32, layout=mfma_layout)
    for k in range(0, num_k_iter - 1):
        # Advance the ptrs to the next K block.
        offs_a += BLOCK_SIZE_K * stride_ak
        offs_b += BLOCK_SIZE_K * stride_bk

        # Load the next block of A
        if EVEN_K:
            a = gl.amd.cdna4.buffer_load(
                ptr=(a_ptr + batch_id * stride_ab),
                offsets=offs_a,
                mask=(offs_am[:, None] < M),
                other=0.0,
            )
        else:
            a = gl.amd.cdna4.buffer_load(
                ptr=(a_ptr + batch_id * stride_ab),
                offsets=offs_a,
                mask=(
                    (offs_ak[None, :] < K - (k + 1) * BLOCK_SIZE_K)
                    & (offs_am[:, None] < M)
                ),
                other=0.0,
            )

        # Grab the current block of A from shared memory
        cur_a = smem_a.load(layout=dot_a_layout)
        cur_b = smem_b.load(layout=dot_b_layout)

        # Load the next block of B
        if EVEN_K:
            b = gl.amd.cdna4.buffer_load(
                ptr=(b_ptr + batch_id * stride_bb),
                offsets=offs_b,
                mask=(offs_bn[None, :] < N),
                other=0.0,
            )
        else:
            b = gl.amd.cdna4.buffer_load(
                ptr=(b_ptr + batch_id * stride_bb),
                offsets=offs_b,
                mask=(
                    (offs_bk[:, None] < K - (k + 1) * BLOCK_SIZE_K)
                    & (offs_bn[None, :] < N)
                ),
                other=0.0,
            )

        # Perform and store the MFMA operation
        accumulator += gl.amd.cdna4.mfma(cur_a, cur_b, zeros)

        # Store next A and B in shared memory, unless this is the last block
        if k < num_k_iter - 2:
            smem_a.store(a)
            smem_b.store(b)

    # Epilogue: use remaining A and B for last MFMA
    cur_a = gl.convert_layout(a, dot_a_layout)
    cur_b = gl.convert_layout(b, dot_b_layout)
    accumulator += gl.amd.cdna4.mfma(cur_a, cur_b, zeros)

    # Apply scales
    accumulator *= a_scale[:, None] * b_scale[None, :]

    # Load and apply bias
    if HAS_BIAS:
        offs_bias = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        bias = gl.amd.cdna4.buffer_load(
            ptr=(bias_ptr + batch_id * stride_biasb),
            offsets=offs_bias,
            mask=(offs_bias < N),
            other=0.0,
        )
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    # Store result in memory
    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    c_offs = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    gl.amd.cdna4.buffer_store(
        stored_value=c,
        ptr=(c_ptr + batch_id * stride_cb),
        offsets=c_offs,
        mask=((offs_cm[:, None] < M) & (offs_cn[None, :] < N)),
    )


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _batched_gemm_a8w8_kernel_async_copy(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_ascaleb,
    stride_bscaleb,
    stride_biasb,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    BUFFER_SIZE: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_a8w8 function
    below

    Computes the matmul C[i] = A[i] x B[i] and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K).
    - B: Batch tensor B with shape (B, K, N).
    - C: Batch tensor C with shape (B, M, N).
    - A_scale: First scale batch tensor with shape (B, M, 1).
    - B_scale: Second scale batch tensor with shape (B, 1, N).
    - Bias: Bias batch tensor with shape (B, 1, N).
    """

    # -----------------------------------------------------------
    # Get batch program id
    batch_id = gl.program_id(axis=0)
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = gl.program_id(axis=1)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    remap_xcd(pid, GRID_MN)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)

    gl.assume(pid_m >= 0)
    gl.assume(pid_n >= 0)

    # Cast batch id and batch dimension strides to int64 to avoid int32 overflow during offset calculation
    # Note: If you're attempting to cast strides to int64 to prevent integer overflow, use `tl.cast` instead of `.to()`.
    # See https://github.com/ROCm/aiter/pull/597 for rationale
    batch_id = tl.cast(batch_id, gl.int64)
    stride_ab = tl.cast(stride_ab, gl.int64)
    stride_bb = tl.cast(stride_bb, gl.int64)
    stride_cb = tl.cast(stride_cb, gl.int64)

    # Create layouts for contiguous access
    # NOTE: Different from original Gluon kernel, size_per_thread must have size 16, as both
    # buffer_load_to_shared and global_load_to_shared require that size_of_thread * 8 == 128 bits
    # (the kernel is loading 8-bit elements from the tensors)
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[16, 4],
        warps_per_cta=[gl.num_warps(), 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, gl.num_warps()],
        order=[0, 1],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 64],
        transposed=True,
        warps_per_cta=[2, gl.num_warps() // 2],
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=4, max_phase=4, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=4, max_phase=4, order=[0, 1]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    # Create offsets for first block of A and B input matrices
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))
    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
    )
    offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
    )
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Load the scales
    offs_a_scale = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_b_scale = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=(a_scale_ptr + batch_id * stride_ascaleb),
        offsets=offs_a_scale,
        mask=(offs_a_scale < M),
        other=1.0,
    )
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=(b_scale_ptr + batch_id * stride_bscaleb),
        offsets=offs_b_scale,
        mask=(offs_b_scale < N),
        other=1.0,
    )

    # Allocate shared memory input matrices and scaling factors
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    # Load first block of A
    if EVEN_K:
        acp.buffer_load_to_shared(
            dest=smem_a.index(0),
            ptr=(a_ptr + batch_id * stride_ab),
            offsets=offs_a,
            mask=(offs_am[:, None] < M),
            other=0.0,
            cache_modifier=".cg",
        )
    else:
        acp.buffer_load_to_shared(
            dest=smem_a.index(0),
            ptr=(a_ptr + batch_id * stride_ab),
            offsets=offs_a,
            mask=((offs_ak[None, :] < K) & (offs_am[:, None] < M)),
            other=0.0,
            cache_modifier=".cg",
        )

    # Load first block of B
    if EVEN_K:
        acp.buffer_load_to_shared(
            dest=smem_b.index(0),
            ptr=(b_ptr + batch_id * stride_bb),
            offsets=offs_b,
            mask=(offs_bn[None, :] < N),
            other=0.0,
            cache_modifier=".cg",
        )
    else:
        acp.buffer_load_to_shared(
            dest=smem_b.index(0),
            ptr=(b_ptr + batch_id * stride_bb),
            offsets=offs_b,
            mask=((offs_bk[:, None] < K) & (offs_bn[None, :] < N)),
            other=0.0,
            cache_modifier=".cg",
        )

    # Run the main loop
    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    accumulator = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
    )
    zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.int32, layout=mfma_layout)
    for k in range(0, num_k_iter - 1):
        # Wait for loads to finish before using shared memory
        acp.async_wait(num_outstanding=0)

        # Advance the ptrs to the next K block.
        offs_a += BLOCK_SIZE_K * stride_ak
        offs_b += BLOCK_SIZE_K * stride_bk

        # Load the next block of A
        if EVEN_K:
            acp.buffer_load_to_shared(
                dest=smem_a.index((k + 1) % 2),
                ptr=(a_ptr + batch_id * stride_ab),
                offsets=offs_a,
                mask=(offs_am[:, None] < M),
                other=0.0,
                cache_modifier=".cg",
            )
        else:
            acp.buffer_load_to_shared(
                dest=smem_a.index((k + 1) % 2),
                ptr=(a_ptr + batch_id * stride_ab),
                offsets=offs_a,
                mask=(
                    (offs_ak[None, :] < K - (k + 1) * BLOCK_SIZE_K)
                    & (offs_am[:, None] < M)
                ),
                other=0.0,
                cache_modifier=".cg",
            )

        # Load the next block of B
        if EVEN_K:
            acp.buffer_load_to_shared(
                dest=smem_b.index((k + 1) % 2),
                ptr=(b_ptr + batch_id * stride_bb),
                offsets=offs_b,
                mask=(offs_bn[None, :] < N),
                other=0.0,
                cache_modifier=".cg",
            )
        else:
            acp.buffer_load_to_shared(
                dest=smem_b.index((k + 1) % 2),
                ptr=(b_ptr + batch_id * stride_bb),
                offsets=offs_b,
                mask=(
                    (offs_bk[:, None] < K - (k + 1) * BLOCK_SIZE_K)
                    & (offs_bn[None, :] < N)
                ),
                other=0.0,
                cache_modifier=".cg",
            )

        # Grab the current block of A from shared memory
        cur_a = smem_a.index(k % 2).load(layout=dot_a_layout)

        # Grab the current block of B from shared memory
        cur_b = smem_b.index(k % 2).load(layout=dot_b_layout)

        # Perform and store the MFMA operation
        accumulator += gl.amd.cdna4.mfma(cur_a, cur_b, zeros)

    # Epilogue: use remaining A and B for last MFMA
    cur_a = smem_a.index((num_k_iter - 1) % 2).load(layout=dot_a_layout)
    cur_b = smem_b.index((num_k_iter - 1) % 2).load(layout=dot_b_layout)
    accumulator += gl.amd.cdna4.mfma(cur_a, cur_b, zeros)

    # Apply scales
    accumulator *= a_scale[:, None] * b_scale[None, :]

    # Load and apply bias
    if HAS_BIAS:
        offs_bias = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        bias = gl.amd.cdna4.buffer_load(
            ptr=(bias_ptr + batch_id * stride_biasb),
            offsets=offs_bias,
            mask=(offs_bias < N),
            other=0.0,
        )
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    # Store result in memory
    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    c_offs = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    gl.amd.cdna4.buffer_store(
        stored_value=c,
        ptr=(c_ptr + batch_id * stride_cb),
        offsets=c_offs,
        mask=((offs_cm[:, None] < M) & (offs_cn[None, :] < N)),
    )


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-BATCHED_GEMM-A8W8.json"
        print(f"fpath={fpath}")
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    if M + N >= 4096:
        return _get_config._config_dict["large"]
    else:
        return _get_config._config_dict["small"]


def batched_gemm_a8w8(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    splitK: Optional[int] = None,
    YQ: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    use_async_copy: bool = False,
):
    """
    Computes the matmul YQ[i] = XQ[i] x WQ[i]T and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - XQ: Batch tensor XQ with shape (B, M, K).
    - WQ: Batch tensor WQ with shape (B, N, K).
    - X_scale: First scale batch tensor with shape (B, M, 1).
    - W_scale: Second scale batch tensor with shape (B, 1, N).
    - Bias: Bias batch tensor with shape (B, 1, N).
    - YQ: Output Matrix Y with shape (B, M, N). If this is none, then it's created by this API and returned as output

    Returns:
    - YQ: The output batch tensor with shape (B, M, N).
    """
    _LOGGER.info(
        f"BATCHED_GEMM_A8W8: x={tuple(XQ.shape)} w={tuple(WQ.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    # Make sure XQ and WQ are contiguous in memory
    XQ = XQ.contiguous()
    WQ = WQ.contiguous()

    # Check constraints.
    assert XQ.shape[0] == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert XQ.shape[2] == WQ.shape[2], "Incompatible K dimensions!!!"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    # Transpose N and K dimensions of WQ: (B, N, K) -> (B, K, N)
    WQ = WQ.transpose(1, 2)

    B = XQ.shape[0]
    M = XQ.shape[1]
    K = XQ.shape[2]
    N = WQ.shape[2]

    has_bias = bias is not None
    if YQ is None:
        YQ = torch.empty((B, M, N), dtype=dtype, device=XQ.device)

    if config is None:
        config = _get_config(M, N, K)

    grid = lambda META: (  # noqa: E731
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    assert XQ.element_size() == WQ.element_size()
    buffer_size = 16 // XQ.element_size()
    assert buffer_size == 16

    impl = (
        _batched_gemm_a8w8_kernel_async_copy
        if use_async_copy
        else _batched_gemm_a8w8_kernel
    )

    impl[grid](
        XQ,
        WQ,
        YQ,
        x_scale,
        w_scale,
        bias,
        M,
        N,
        K,
        XQ.stride(0),
        XQ.stride(1),
        XQ.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0),
        YQ.stride(1),
        YQ.stride(2),
        x_scale.stride(0),
        w_scale.stride(0),
        bias.stride(0) if has_bias else 0,
        has_bias,
        BUFFER_SIZE=buffer_size,
        **config,
    )

    return YQ
