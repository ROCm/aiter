# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import math
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
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _gemm_a8w8_blockscale_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    warp_bases: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach, with block shape assumed to be the same as BLOCK_SIZE_N/K. 

    Split-K not supported due to design decision.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N
    """

    # TODO: make parameters
    NUM_BUFFERS: gl.constexpr = 3
    L2_PREFETCH_DISTANCE: gl.constexpr = 1

    # Setup
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[4, 8], # 32
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 4], # 32
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 128])


    # TDM Shared Layouts
    tdm_shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_K],
                                                                                [1, 0])
    tdm_shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_SIZE_N, 16]], [BLOCK_SIZE_K, BLOCK_SIZE_N],
                                                                                    [0, 1])
    
    shared_a_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    shared_b_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )


    k_tiles_count = gl.cdiv(K, BLOCK_SIZE_K)

    # Create pointers for first block of A and B input matrices
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))

    smem_scale_a = gl.allocate_shared_memory(
        gl.float32, [BLOCK_SIZE_M], layout=shared_a_scale
    )

    smem_scale_b = gl.allocate_shared_memory(
        gl.float32, [BLOCK_SIZE_N], layout=shared_b_scale
    )

    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
    )
    offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
    )

    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    # Create pointers for the scales
    offs_a_scale = offs_am * stride_ascale_m
    
    
    # TDM tensor descriptors and shared mem
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, 
                                                        shape=(M, K),
                                                        strides=(stride_am, stride_ak), 
                                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                                        layout=tdm_shared_a)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr,
                                                        shape=(K, N),
                                                            strides=(stride_bk, stride_bn),
                                                            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), 
                                                            layout=tdm_shared_b)
    tdm_smem_a = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    tdm_smem_b = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    # edit these if you need
    num_loads = 0 
    num_computes = 0 

    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout
    )
    zeros = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout
    )


    # ------------ Prologue ---------------

    # Load A scale from global
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        cache=cache_modifier,
    )

    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    offs_b_scale_n = offs_bn // GROUP_N
    offs_b_scale = offs_b_scale_n * stride_bscale_n
    # Load b scale from global
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=b_scale_ptr,
        offsets=offs_b_scale,
        cache=cache_modifier,
    )
    # prefetch lds
    if L2_PREFETCH_DISTANCE > NUM_BUFFERS:
        for i in gl.static_range(NUM_BUFFERS - L2_PREFETCH_DISTANCE):
            prefetch_iteration = num_loads + L2_PREFETCH_DISTANCE + NUM_BUFFERS + i
            gl.amd.gfx1250.tdm.prefetch(a_desc, [0, prefetch_iteration * BLOCK_SIZE_K], pred=True)
            gl.amd.gfx1250.tdm.prefetch(b_desc, [prefetch_iteration * BLOCK_SIZE_K, 0], pred=True)
            

    off_am_tdm: gl.int32 = pid_m * BLOCK_SIZE_M
    off_bm_tdm: gl.int32 = pid_n * BLOCK_SIZE_N
    # Loading initial batch of a and b to be in the queue. now 2 things in queue (with num buffers 2)


    # TDM prologue
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(a_desc, [off_am_tdm, num_loads * BLOCK_SIZE_K], tdm_smem_a.index(num_loads % NUM_BUFFERS))
        gl.amd.gfx1250.tdm.async_load(b_desc, [num_loads * BLOCK_SIZE_K, off_bm_tdm], tdm_smem_b.index(num_loads % NUM_BUFFERS))
        num_loads += 1
    
    # Store scales in WMMA slice order so load(SliceLayout(1/0, wmma_layout)) aligns scale[i] with row/col i
    smem_scale_a.store(a_scale)
    smem_scale_b.store(b_scale) 

    # wait for the buffers to finish
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
    # load shared relaxed for a and b -- edit these
    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            tdm_smem_a.index(num_computes % NUM_BUFFERS), dot_a_layout
        )
    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            tdm_smem_b.index(num_computes % NUM_BUFFERS), dot_b_layout
        )

    #scales
    # cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
    # cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    # a_scale = gl.amd.cdna4.buffer_load(
    #     ptr=a_scale_ptr,
    #     offsets=offs_a_scale,
    #     cache=cache_modifier,
    # )
    # # Loading b scale and curr b scale
    # b_scale = gl.amd.cdna4.buffer_load(
    #     ptr=b_scale_ptr,
    #     offsets=offs_b_scale,
    #     cache=cache_modifier,
    # )

    # ----- Main Loop --------

    for k in range(k_tiles_count - 1):
        # Loading a scale and curr A scale
        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
        cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))


        # wmma 
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]
        # async loads
        gl.amd.gfx1250.tdm.async_load(a_desc, [off_am_tdm, num_loads * BLOCK_SIZE_K], tdm_smem_a.index(num_loads % NUM_BUFFERS),
                                pred=1)
        gl.amd.gfx1250.tdm.async_load(b_desc, [num_loads * BLOCK_SIZE_K, off_bm_tdm], tdm_smem_b.index(num_loads % NUM_BUFFERS),
                                    pred=1)
        
        # async wait and load index increase
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        num_loads += 1
        # prefetches
        if L2_PREFETCH_DISTANCE - 1 != 0:
            prefetch_iteration = num_loads + L2_PREFETCH_DISTANCE - 1
            gl.amd.gfx1250.tdm.prefetch(a_desc, [0, prefetch_iteration * BLOCK_SIZE_K], pred=True)
            gl.amd.gfx1250.tdm.prefetch(b_desc, [prefetch_iteration * BLOCK_SIZE_K, 0], pred=True)
        # load shared relaxed for next a and b
        next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS), dot_a_layout
            )
        next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS), dot_b_layout
            )
        # then:
        cur_a = next_a
        cur_b = next_b
        num_computes += 1

        # scales

        
        # Advance the ptrs to the next K block.
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )
        # Loading b scale and curr b scale
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr,
            offsets=offs_b_scale,
            cache=cache_modifier,
        )

        


        # Store in WMMA slice order so next iteration's load aligns scale[i] with row/col i
        smem_scale_a.store(a_scale)
        smem_scale_b.store(b_scale) 

        
    # ======= Epilogue ========

    # load a from the last load
    cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
    cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))


    for i in gl.static_range(NUM_BUFFERS - 2):
  

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)

        # load shared relaxed instead

        next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS), dot_a_layout
            )
        next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS), dot_b_layout
            )
        # wmma
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        num_computes += 1
        acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]
        #acc += zeros * cur_a_scale[:, None] * cur_b_scale[None, :]
        zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)
        cur_a = next_a
        cur_b = next_b
        num_computes += 1

        
    # final scale application
    a_scale_ptr += stride_ascale_k
    b_scale_ptr += stride_bscale_k
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        cache=cache_modifier,
    )
    # Loading b scale and curr b scale
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=b_scale_ptr,
        offsets=offs_b_scale,
        cache=cache_modifier,
    )
    smem_scale_a.store(a_scale)
    smem_scale_b.store(b_scale) 
     # load a from the last load
    cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
    cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
    num_computes += 1
    acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]

    if NUM_BUFFERS > 2:
        gl.amd.sched_barrier(0)


    # Store 
    tdm_shared_c: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )
    tdm_smem_c = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=tdm_shared_c,
    )
    tdm_smem_c.store(acc.to(c_ptr.type.element_ty))

    # Ensure all wavefronts have finished writing to LDS before TDM reads it.
    gl.barrier()

    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=tdm_shared_c,
    )
    gl.amd.gfx1250.tdm.async_store(
        c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], tdm_smem_c
    )
    gl.amd.gfx1250.tdm.async_wait(0)