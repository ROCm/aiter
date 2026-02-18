# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# adapted from triton_kernels package

from typing import Optional
import json
import os
import math
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton._triton_kernels.moe.quant_moe import _compute_static_fp8_quant
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp
from triton.language.core import _aggregate as aggregate


@aggregate
class AsyncCopyDescriptor:
    """
    Workaround descriptor for async copy when gather is used.
    Pre-computes irregular offsets and uses async_copy instead of TDM.
    TODO: Replace with TDM Gather once AMD hardware support is available.
    """
    ptr: ttgl.tensor
    offs: ttgl.tensor  # Pre-computed offsets for each row [BLOCK_M, BLOCK_K]
    offs_k: ttgl.tensor  # K dimension offsets [BLOCK_K]
    step_k: ttgl.tensor  # Step size for K dimension (BLOCK_K * stride_x_k)
    BLOCK_M: ttgl.constexpr
    BLOCK_K: ttgl.constexpr
    
    @gluon.constexpr_function
    def __init__(self, ptr, offs, offs_k, step_k, BLOCK_M, BLOCK_K):
        self.ptr = ptr
        self.offs = offs
        self.offs_k = offs_k
        self.step_k = step_k
        self.BLOCK_M = ttgl.constexpr(BLOCK_M)
        self.BLOCK_K = ttgl.constexpr(BLOCK_K)
    
    @gluon.jit
    def initialize(ptr, gathered_m, off_k, stride_m, stride_k, BLOCK_M: ttgl.constexpr, BLOCK_K: ttgl.constexpr, BLOCKED_MK: ttgl.constexpr):
        """
        Initialize descriptor with gathered row indices.
        
        Args:
            ptr: Base pointer to X tensor
            gathered_m: Gathered row indices [BLOCK_M] (irregular)
            off_k: Starting K offset
            stride_m: Stride for M dimension
            stride_k: Stride for K dimension
        """
        offs_k = off_k + ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, BLOCKED_MK))
        step_k = BLOCK_K * stride_k
        offs = gathered_m.to(ttgl.int32)[:, None] * stride_m + offs_k .to(ttgl.int32)[None, :]* stride_k
        return AsyncCopyDescriptor(ptr, offs, offs_k, step_k, BLOCK_M, BLOCK_K)
    
    @gluon.jit
    def issue_async_load(self, k_idx: int, buffer, mask_k=None):
        """
        Issue async load for K iteration k_idx.
        
        Args:
            k_idx: K iteration index (0, 1, 2, ...)
            buffer: Shared memory buffer to load into
            mask_k: Optional mask for K dimension (for EVEN_K=False case)
        """
        offs = self.offs + k_idx * self.step_k
        if mask_k is not None:
            cp.global_to_shared(buffer, self.ptr + offs, mask=mask_k, other=0.0)
        else:
            cp.global_to_shared(buffer, self.ptr + offs)
        cp.commit_group()


@gluon.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: ttgl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@gluon.jit
def clip(x, limit, clip_lower: ttgl.constexpr):
    res = ttgl.minimum(x, limit)
    if clip_lower:
        res = ttgl.maximum(-limit, res)
    return res


@gluon.jit
def _swiglu(input, alpha, limit):
    gelu, linear = ttgl.split(ttgl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(ttgl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(ttgl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + ttgl.exp2(-1.44269504089 * alpha * gelu))
    return ttgl.fma(s, linear, s)  # (s * (linear + 1))


@gluon.jit
def _reduce_grouped(
    X,
    stride_xb: ttgl.uint64,
    stride_xm: ttgl.uint64,
    stride_xn,  #
    Out,
    stride_om: ttgl.uint64,
    stride_on,  # output tensor
    InIndx,
    B,
    N,  #
    # fused activation function
    APPLY_SWIGLU: ttgl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: ttgl.constexpr,
    K: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    EVEN_N: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
):
    pid_t = ttgl.program_id(1)
    pid_n = ttgl.program_id(0)

    BLOCK_N_OUT: ttgl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K
    
    threads_per_elem_n: ttgl.constexpr = triton.cdiv(
        BLOCK_N // (NUM_WARPS * 32), 16
    )
    threads_per_elem_n_out: ttgl.constexpr = triton.cdiv(
        BLOCK_N_OUT // (NUM_WARPS * 32), 16
    )
    
    blocked_n: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=[threads_per_elem_n, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    
    blocked_n_out: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=[threads_per_elem_n_out, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t,)
    else:
        indxs = ()
        for i in ttgl.static_range(0, K):
            indxs = indxs + (ttgl.load(InIndx + start + i),)
    
    # Setup offsets
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked_n))
    offs_n_out = pid_n * BLOCK_N_OUT + ttgl.arange(0, BLOCK_N_OUT, layout=ttgl.SliceLayout(0, blocked_n_out))
    
    acc = ttgl.zeros([BLOCK_N_OUT], dtype=ttgl.float32, layout=ttgl.SliceLayout(0, blocked_n_out))
    x_n_mask = offs_n < N
    XPtrs = X + offs_n * stride_xn

    for i in ttgl.static_range(0, K):
        # Initialize curr with the same layout as vals will have
        curr = ttgl.zeros([BLOCK_N], dtype=ttgl.float32, layout=ttgl.SliceLayout(0, blocked_n))
        
        # Iterate over split_k partial values
        for b in range(0, B):
            x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
            
            if EVEN_N:
                vals = ttgl.load(x_row_ptr)
            else:   
                vals = ttgl.load(x_row_ptr, mask=x_n_mask, other=0.0)
            vals = vals.to(ttgl.float32)
            curr += vals
    
        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(curr[None, :], alpha, limit)
        curr = ttgl.reshape(curr, [curr.shape[-1]])
        # Convert curr to match acc's layout before adding
        curr = ttgl.convert_layout(curr, ttgl.SliceLayout(0, blocked_n_out))
        # update final accumulator
        acc += curr
    # Compute per-32-col MXFP scales for this tile if requested
    Nrem = N // ACTIVATION_REDUCTION_N
    
    # Write-back
    offs_out = pid_t * stride_om + offs_n_out * stride_on
    if EVEN_N:
        ttgl.store(Out + offs_out, acc)
    else:
        out_n_mask = offs_n_out < Nrem
        ttgl.store(Out + offs_out, acc, mask=out_n_mask)


@gluon.jit
def issue_tile_loads(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    GatherIndx,
    pid_m,
    pid_n,
    k_offset_start,
    producer,
    a_buffer,
    b_buffer,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    K: ttgl.constexpr,
    W_TRANSPOSE: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    EVEN_K: ttgl.constexpr,
):
    """
    Load A, B, and scale tiles asynchronously into shared memory buffers.
    
    Args:
        a_desc: Tensor descriptor or AsyncCopyDescriptor for A
        b_desc: Tensor descriptor for B
        GatherIndx: Gather indices (None for regular access)
        pid_m, pid_n: Block IDs for M and N dimensions
        k_offset_start: Starting K offset for this split-K block
        producer: Producer index (which buffer slot to use)
        a_buffer, b_buffer: Shared memory buffers
        BLOCK_M, BLOCK_N, BLOCK_K: Block sizes
        W_TRANSPOSE: Whether W is transposed
        NUM_BUFFERS: Number of pipeline buffers
    """
    buffer_idx = producer % NUM_BUFFERS
    
    if not EVEN_K:
        mask = k_offset_start + producer * BLOCK_K < K
    else:
        mask = None
    # Load A tile
    if GatherIndx is None:
        ttgl.amd.gfx1250.tdm.async_load(
            a_desc,
            [pid_m * BLOCK_M, k_offset_start + producer * BLOCK_K],
            a_buffer.index(buffer_idx)
        )
    else:
        a_desc.issue_async_load(producer, a_buffer.index(buffer_idx), mask)
    
    # Load B tile
    if W_TRANSPOSE:
        # W is stored as (N, K)
        ttgl.amd.gfx1250.tdm.async_load(
            b_desc,
            [pid_n * BLOCK_N, k_offset_start + producer * BLOCK_K],
            b_buffer.index(buffer_idx)
        )
    else:
        # W is stored as (K, N)
        ttgl.amd.gfx1250.tdm.async_load(
            b_desc,
            [k_offset_start + producer * BLOCK_K, pid_n * BLOCK_N],
            b_buffer.index(buffer_idx)
        )


@gluon.jit
def create_descriptor(
    X,
    stride_x_m,
    stride_x_k,
    XBlockScale,
    stride_x_bs_m,
    stride_x_bs_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WBlockScale,
    stride_w_bs_e,
    stride_w_bs_k,
    stride_w_bs_n,
    GatherIndx,
    start_m,
    block_id,
    pid_k,
    pid_n,
    expt_id,
    M,
    N,
    K,
    grid_m,
    # Constexprs
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    BLOCKSCALE_M: ttgl.constexpr,
    BLOCKSCALE_N: ttgl.constexpr,
    BLOCKSCALE_K: ttgl.constexpr,
    SPLIT_K: ttgl.constexpr,
    N_EXPTS_ACT: ttgl.constexpr,
    BLOCKED_MK: ttgl.constexpr,
    BLOCKED_KN: ttgl.constexpr,
    SHARED_A: ttgl.constexpr,
    SHARED_B: ttgl.constexpr,
    SHARED_A_SCALE: ttgl.constexpr,
    SHARED_B_SCALE: ttgl.constexpr,
    PER_ROW_X_SCALE: ttgl.constexpr,
    W_TRANSPOSE: ttgl.constexpr,
    is_x_blockscale: ttgl.constexpr,
    is_w_blockscale: ttgl.constexpr,
):
    """
    Create tensor descriptors for X, W, and their scales.

    Returns:
        a_desc: Tensor descriptor or AsyncCopyDescriptor for X
        b_desc: Tensor descriptor for W
        a_scale_desc: Tensor descriptor for X scales (or None)
        b_scale_desc: Tensor descriptor for W scales (or None)
        offs_x_m: M dimension offsets for X (used for scales)
        offs_w_n: N dimension offsets for W (used for scales)
    """
    splitk_block_size = ttgl.cdiv(K, SPLIT_K)

    offs_x_m = BLOCK_M * block_id + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_MK))
    offs_x_m = ttgl.max_contiguous(ttgl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)

    # A descriptor
    if GatherIndx is None:
        offs_x_m = start_m + offs_x_m
        # Regular access: Use TDM tensor descriptor
        in_m = grid_m * BLOCK_M 
        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(in_m, K),
            strides=(stride_x_m, stride_x_k),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_A
        )
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = ttgl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
        # Initialize AsyncCopyDescriptor
        # TODO: Replace with TDM Gather once AMD hardware support is available.
        off_k_start = pid_k * splitk_block_size
        a_desc = AsyncCopyDescriptor.initialize(
            X, offs_x_m, off_k_start, stride_x_m, stride_x_k, BLOCK_M, BLOCK_K, BLOCKED_MK
        )

    # B descriptor
    offs_w_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_KN))
    offs_w_n = ttgl.max_contiguous(
        ttgl.multiple_of(offs_w_n % N, BLOCK_N),
        BLOCK_N,
    )
    W += expt_id * stride_w_e

    if W_TRANSPOSE:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=W,
            shape=(N, K),
            strides=(stride_w_n, stride_w_k),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_B
        )
    else:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=W,
            shape=(K, N),
            strides=(stride_w_k, stride_w_n),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_B
        )
    
    # A scale descriptor
    a_scale_desc = None
    if is_x_blockscale:
        if PER_ROW_X_SCALE:
            # XScale: [M, K_blocks]
            if GatherIndx is None:
                a_scale_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=XBlockScale,
                    shape=(M, ttgl.cdiv(K, BLOCKSCALE_K)),
                    strides=(stride_x_bs_m, stride_x_bs_k),
                    block_shape=(BLOCK_M, 1),
                    layout=SHARED_A_SCALE
                )
            else:
                off_k_scale_start = (pid_k * splitk_block_size) // BLOCKSCALE_K
                a_scale_desc = AsyncCopyDescriptor.initialize(
                    XBlockScale, offs_x_m, off_k_scale_start, stride_x_bs_m, stride_x_bs_k, 
                    BLOCK_M, BLOCK_K, BLOCKED_MK
                )
        else:
            # XScale: [M_blocks, K_blocks]
            if GatherIndx is None:
                a_scale_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=XBlockScale,
                    shape=(ttgl.cdiv(M, BLOCKSCALE_M), ttgl.cdiv(K, BLOCKSCALE_K)),
                    strides=(stride_x_bs_m, stride_x_bs_k),
                    block_shape=(1, 1),
                    layout=SHARED_A_SCALE
                )
            else:
                off_k_scale_start = (pid_k * splitk_block_size) // BLOCKSCALE_K
                # Compute M block indices from gathered row indices
                offs_x_scale_m = offs_x_m // BLOCKSCALE_M
                a_scale_desc = AsyncCopyDescriptor.initialize(
                    XBlockScale, offs_x_scale_m, off_k_scale_start, stride_x_bs_m, stride_x_bs_k,
                    BLOCK_M, BLOCK_K, BLOCKED_MK
                )
    
    # B scale descriptor
    b_scale_desc = None
    if is_w_blockscale:
        # WScale: [K_blocks, N_blocks]
        scale_k_blocks = ttgl.cdiv(K, BLOCKSCALE_K)
        scale_n_blocks = ttgl.cdiv(N, BLOCKSCALE_N)
        b_scale_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=WBlockScale,
            shape=(scale_k_blocks, scale_n_blocks),
            strides=(stride_w_bs_k, stride_w_bs_n),
            block_shape=(1, 1),
            layout=SHARED_B_SCALE
        )
    
    return a_desc, b_desc, a_scale_desc, b_scale_desc


@gluon.jit
def _moe_gemm_a8w8_blockscale_gfx1250(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XBlockScale,  # [M, K_blocks] or [M_blocks, K_blocks]
    stride_x_bs_m,
    stride_x_bs_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WBlockScale,  # [K_blocks, N_blocks]
    stride_w_bs_e,
    stride_w_bs_k,
    stride_w_bs_n,
    X_static_scale,
    W_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,  # Bias
    Gammas,
    N,
    K,  # shapes
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: ttgl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: ttgl.constexpr,
    # MoE config
    N_EXPTS_ACT: ttgl.constexpr,
    # optimization config
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    GROUP_M: ttgl.constexpr,
    BLOCKSCALE_M: ttgl.constexpr,
    BLOCKSCALE_N: ttgl.constexpr,
    BLOCKSCALE_K: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    XCD_SWIZZLE: ttgl.constexpr,
    EVEN_K: ttgl.constexpr,
    MASK_K_LIMIT: ttgl.constexpr,
    SPLIT_K: ttgl.constexpr,
    W_CACHE_MODIFIER: ttgl.constexpr,
    # Layouts
    BLOCKED_MK: ttgl.constexpr,
    BLOCKED_KN: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,
    SHARED_A: ttgl.constexpr,
    SHARED_B: ttgl.constexpr,
    SHARED_A_SCALE: ttgl.constexpr,
    SHARED_B_SCALE: ttgl.constexpr,
    DOT_A_LAYOUT: ttgl.constexpr,
    DOT_B_LAYOUT: ttgl.constexpr,
    UPCAST_INDICES: ttgl.constexpr = False,
    # Use per-row or 2D blockscale on X
    PER_ROW_X_SCALE: ttgl.constexpr = False,
    # W transpose: If True, W is stored as (N, K) instead of (K, N)
    W_TRANSPOSE: ttgl.constexpr = False,
    # Number of buffers to use for async_load
    NUM_BUFFERS: ttgl.constexpr = 2,
):
    """
    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - E: Matrix E with shape (E, K, N).
    - Y: Matrix C with shape (E, M, N).
    - x_scale: Scale tensor for A with shape (M // blockscale_m, K // blockscale_k) or (M, K // blockscale_k)
    - w_scale: Scale tensor for B with shape (K // blockscale_k, N // blockscale_n)
    - PER_ROW_X_SCALE: Determines whether we use per-row or 2D blockscale on X

    For this kernel implementation, BLOCKSCALE_K must equal BLOCK_K.
    """
    is_x_blockscale: ttgl.constexpr = XBlockScale is not None
    is_w_blockscale: ttgl.constexpr = WBlockScale is not None

    OUT_BLOCK_N: ttgl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = ttgl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - ttgl.load(ExptOffsSum)
    else:
        padding_m: ttgl.constexpr = 0

    index_type: ttgl.constexpr = ttgl.int64 if UPCAST_INDICES else ttgl.int32

    unpadded_m = grid_m - padding_m
    ttgl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    # pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    # unpack expert data
    expt_data = ttgl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = ttgl.load(ExptHist + expt_id)
    start_m = ttgl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)
    
    # Allocate shared memory buffer
    a_buffer = ttgl.allocate_shared_memory(
        X.type.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layout=SHARED_A
    )
    if W_TRANSPOSE:
        # W is stored as (N, K) in shared memory
        b_buffer = ttgl.allocate_shared_memory(
            W.type.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], layout=SHARED_B
        )
    else:
        # W is stored as (K, N) in shared memory
        b_buffer = ttgl.allocate_shared_memory(
            W.type.element_ty, [NUM_BUFFERS, BLOCK_K, BLOCK_N], layout=SHARED_B
        )
    a_scale_buffer = ttgl.allocate_shared_memory(
        XBlockScale.type.element_ty if XBlockScale is not None else ttgl.float32,
        [NUM_BUFFERS, BLOCK_M],
        layout=SHARED_A_SCALE
    )
    b_scale_buffer = ttgl.allocate_shared_memory(
        WBlockScale.type.element_ty if WBlockScale is not None else ttgl.float32,
        [NUM_BUFFERS, BLOCK_N],
        layout=SHARED_B_SCALE
    )
    
    # Create tensor descriptors
    a_desc, b_desc, a_scale_desc, b_scale_desc = create_descriptor(
        X, stride_x_m, stride_x_k,
        XBlockScale, stride_x_bs_m, stride_x_bs_k,
        W, stride_w_e, stride_w_k, stride_w_n,
        WBlockScale, stride_w_bs_e, stride_w_bs_k, stride_w_bs_n,
        GatherIndx, start_m, block_id, pid_k, pid_n, expt_id,
        M, N, K, grid_m,
        BLOCK_M, BLOCK_N, BLOCK_K,
        BLOCKSCALE_M, BLOCKSCALE_N, BLOCKSCALE_K,
        SPLIT_K, N_EXPTS_ACT,
        BLOCKED_MK, BLOCKED_KN,
        SHARED_A, SHARED_B, SHARED_A_SCALE, SHARED_B_SCALE,
        PER_ROW_X_SCALE, W_TRANSPOSE,
        is_x_blockscale, is_w_blockscale,
    )
    
    producer = 0
    consumer = 0
    acc = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=WMMA_LAYOUT)
    splitk_block_size = ttgl.cdiv(K, SPLIT_K)
    num_k_tiles = ttgl.cdiv(splitk_block_size, BLOCK_K)
    k_offset_start = pid_k * splitk_block_size
    
    # prologue
    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        issue_tile_loads(
            a_desc, b_desc, a_scale_desc, b_scale_desc, GatherIndx,
            pid_m, pid_n, k_offset_start, producer, 
            a_buffer, b_buffer,
            BLOCK_M, BLOCK_N, K, BLOCK_K, W_TRANSPOSE, NUM_BUFFERS, EVEN_K
        )
        producer += 1
    
    # Main loop
    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        # Load next A and B tiles
        issue_tile_loads(
            a_desc, b_desc, a_scale_desc, b_scale_desc, GatherIndx,
                pid_m, pid_n, k_offset_start, producer, 
            a_buffer, b_buffer,
            BLOCK_M, BLOCK_N, K, BLOCK_K, W_TRANSPOSE, NUM_BUFFERS, EVEN_K
        )
        producer += 1
        
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        cur_a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=DOT_A_LAYOUT)
        
        if W_TRANSPOSE:
            # B is stored as (N, K) but WMMA needs (K, N), so we permute
            cur_b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=DOT_B_LAYOUT)
        else:
            cur_b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=DOT_B_LAYOUT)
        
        acc = ttgl.amd.gfx1250.wmma(cur_a, cur_b, acc)
        consumer += 1
    
    # Epilogue
    for i in ttgl.static_range(NUM_BUFFERS - 1):
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        cur_a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=DOT_A_LAYOUT)
        
        if W_TRANSPOSE:
            # B is stored as (N, K) but WMMA needs (K, N), so we permute
            cur_b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=DOT_B_LAYOUT)
        else:
            cur_b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=DOT_B_LAYOUT)
        
        acc = ttgl.amd.gfx1250.wmma(cur_a, cur_b, acc)
        consumer += 1
    
    offs_m = block_id * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_cm = start_m + offs_m
    offs_cn = offs_n
    mask_m = offs_m < M
    mask_n = offs_cn < N

    # scalar fp8 scale
    if X_static_scale is not None:
        acc = acc * ttgl.load(X_static_scale)
    if W_static_scale is not None:
        acc = acc * ttgl.load(W_static_scale)

    # bias
    if B is not None:
        offs_bias = expt_id * stride_b_e + offs_cn
        if pid_k == 0:
            bias = ttgl.load(B + offs_bias, mask=mask_n, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = ttgl.full([BLOCK_N], 0, dtype=ttgl.float32, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        acc = acc + bias[None, :]

    if APPLY_SWIGLU and SPLIT_K == 1:
        acc = _swiglu(acc, alpha, limit)
        ttgl.static_assert(
            acc.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({acc.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + ttgl.arange(0, OUT_BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        mask_n = offs_y_n < yN
    else:
        ttgl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )

    if Gammas is not None:
        offs_gammas = start_m + offs_m
        gammas = ttgl.load(Gammas + offs_gammas, mask=mask_m)
        out *= gammas[:, None]
    # quant
    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, ttgl.load(Quant_static_scale))
    # write-back
    offs_c = stride_y_m * offs_cm[:, None] + stride_y_n * offs_cn[None, :]
    mask_c = mask_m[:, None] & mask_n[None, :]
    ttgl.amd.gfx1250.buffer_store(
        acc.to(Y.type.element_ty),
        Y,
        offs_c,
        mask=mask_c
    )


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def allocate_output(
    x,
    w,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
):
    # ---- output ------
    N = w.shape[-1]
    # by default - M is number of rows in the activations
    M = x.shape[-2]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.shape[0]
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
        )  # compressed number of rows
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=x.device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        # Initialize to zeros to avoid uninitialized memory issues
        final_output = torch.zeros(final_shape, device=x.device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    blockscale_m = 128
    blockscale_k = 128
    blockscale_n = 128
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 2

    split_k = 1
    if block_m == 16:
        block_n = 256
        block_k = 128
        num_warps = 4

        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = gluon.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n >= 64 and grid < 256:
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = gluon.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
    else:
        # for scale preshuffling
        block_n = 256
        block_k = 128
        num_warps = 8

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "blockscale_m": blockscale_m,
        "blockscale_n": blockscale_n,
        "blockscale_k": blockscale_k,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }
    return ret


def reduce_grouped(
    x: torch.Tensor,
    indx: torch.Tensor,
    out: torch.Tensor,
    apply_swiglu=False,
    alpha=1.0,
    limit=None,
    reduction_n=1,
    out_dtype: bool = None,
):
    """
    In-place grouped row reduction.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]` and overwrites the row corresponding to the first
    valid (non-negative) index with the per-group sum. Accumulation is performed
    in float32 for numerical stability, and the result is written back in the
    dtype of `x`.

    Behavior and edge cases
    - Invalid (-1) entries are skipped during accumulation and do not generate
      memory traffic. If a group has no valid entries, nothing is written for
      that group.
    - Reduction is performed tile-by-tile along the N dimension within a single
      kernel launch (persistent along N) to minimize launch overhead.

    Performance notes
    - Memory traffic per group is approximately (valid_rows_read + 1) * N * sizeof(x),
      plus index reads. With no invalid entries, this becomes (K + 1) reads/writes
      of length N per group.

    Returns
    - The input tensor `x` (modified in place).
    """
    if indx is None and x.shape[0] == 1:
        return x.squeeze(0)
    if indx is not None:
        num_groups = indx.shape[0]
    else:
        num_groups = x.shape[-2]
    K = 1 if indx is None else indx.shape[1]
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % reduction_n == 0
    BLOCK_N = 512
    num_blocks = triton.cdiv(x.shape[-1], BLOCK_N)
    
    _reduce_grouped[(num_blocks, num_groups)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),  #
        out,
        out.stride(0),
        out.stride(1),  #
        indx,  #
        x.shape[0],
        x.shape[-1],  #
        apply_swiglu,
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,  #
        NUM_WARPS=2,
        num_warps=2,  #
    )
    return out


def moe_gemm_a8w8_blockscale_gfx1250(
    x,
    w,
    x_block_scales=None,
    w_block_scales=None,
    x_static_scale=None,
    w_static_scale=None,
    quant_static_scale=None,
    bias=None,
    routing_data: RoutingData | None = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=None,
    unpadded_N=None,
    unpadded_K=None,
    per_row_x_scale=False,
    preshuffled=False,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])

    If preshuffled is True, then W is stored as (E, N, K) instead of (E, K, N)
    """
    x_has_blockscale = x_block_scales is not None
    w_has_blockscale = w_block_scales is not None

    assert x_has_blockscale ^ (x_static_scale is not None)
    assert w_has_blockscale ^ (w_static_scale is not None)

    if x_has_blockscale:
        stride_x_bs_m = x_block_scales.stride(0)
        stride_x_bs_k = x_block_scales.stride(1)
    else:
        stride_x_bs_m = 0
        stride_x_bs_k = 0

    if w_has_blockscale:
        stride_w_bs_e = w_block_scales.stride(0)
        stride_w_bs_k = w_block_scales.stride(1)
        stride_w_bs_n = w_block_scales.stride(2)
    else:
        stride_w_bs_e = 0
        stride_w_bs_k = 0
        stride_w_bs_n = 0

    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K = x.shape[-1]
    if preshuffled:
        N, K_w = w.shape[-1], w.shape[-2]
    else:
        K_w, N = w.shape[-1], w.shape[-2]
    assert K == K_w, f"K dimension mismatch: x has K={K}, w has K={K_w}"
    
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K
    # compute optimization flags
    config = get_kernel_config(M, N, K, routing_data)
    if apply_swiglu and config["split_k"] > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = True
        reduction_n_reduction = 2
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    # allocate output memory
    y, y_final = allocate_output(
        x,
        w,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
    )
    stride_bias = None if bias is None else bias.stride(0)
    # moe metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map
    # spmd grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]
    
    num_warps = config["num_warps"]
    BLOCK_M = config["block_m"]
    BLOCK_N = config["block_n"]
    BLOCK_K = config["block_k"]
    
    threads_per_elem_mk = triton.cdiv(BLOCK_M * BLOCK_K // (num_warps * 32), 16)
    threads_per_elem_kn = triton.cdiv(BLOCK_K * BLOCK_N // (num_warps * 32), 16)
    
    blocked_mk = ttgl.BlockedLayout(
        size_per_thread=[threads_per_elem_mk, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    blocked_kn = ttgl.BlockedLayout(
        size_per_thread=[16, threads_per_elem_kn],
        threads_per_warp=[4, 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )
    
    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)
    
    wmma_layout = ttgl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        instr_shape=[16, 16, 64]
    )
    
    # Initialize shared memory layouts
    shared_a = ttgl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]],
        [BLOCK_M, BLOCK_K],
        [1, 0]
    )
    if preshuffled:
        shared_b = ttgl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 16]],
            [BLOCK_N, BLOCK_K],
            [0, 1]
        )
    else:
        shared_b = ttgl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]],
            [BLOCK_K, BLOCK_N],
            [1, 0]
        )
    
    # TODO: figure out the interval and padding
    shared_a_scale = ttgl.PaddedSharedLayout.with_identity_for(
        [[256, 16]],
        [BLOCK_M],
        [0]
    )
    shared_b_scale = ttgl.PaddedSharedLayout.with_identity_for(
        [[256, 16]],
        [BLOCK_N],
        [0]
    )
    
    dot_a_layout = ttgl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=16
    )
    dot_b_layout = ttgl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=16
    )
    
    # launch kernel
    _moe_gemm_a8w8_blockscale_gfx1250[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        x_block_scales,
        stride_x_bs_m,
        stride_x_bs_k,
        w,
        w.stride(0),
        w.stride(1) if not preshuffled else w.stride(2),
        w.stride(2) if not preshuffled else w.stride(1),
        w_block_scales,
        stride_w_bs_e,
        stride_w_bs_k,
        stride_w_bs_n,
        x_static_scale,
        w_static_scale,
        quant_static_scale,
        bias,
        stride_bias,
        gammas,
        N,
        K,
        gather_indx,
        expt_hist,
        expt_token_offs_raw,
        expt_hist_sum,
        expt_block_pid_map,
        grid_m,
        grid_n,
        apply_swiglu_matmul,
        alpha,
        limit,
        reduction_n_matmul,
        routing_data.n_expts_act,
        BLOCK_M=config["block_m"],
        BLOCK_N=config["block_n"],
        BLOCK_K=config["block_k"],
        GROUP_M=config["group_m"],
        BLOCKSCALE_M=config["blockscale_m"],
        BLOCKSCALE_N=config["blockscale_n"],
        BLOCKSCALE_K=config["blockscale_k"],
        NUM_WARPS=config["num_warps"],
        XCD_SWIZZLE=config["xcd_swizzle"],
        SPLIT_K=config["split_k"],
        EVEN_K=K % config["block_k"] == 0,
        MASK_K_LIMIT=K % config["block_k"],
        W_CACHE_MODIFIER=config["w_cache_modifier"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        UPCAST_INDICES=should_upcast_indices(x, w, y),
        PER_ROW_X_SCALE=per_row_x_scale,
        W_TRANSPOSE=preshuffled,
        waves_per_eu=config["waves_per_eu"],
        matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
        kpack=config["kpack"],
        # Layouts (computed in wrapper)
        BLOCKED_MK=blocked_mk,
        BLOCKED_KN=blocked_kn,
        WMMA_LAYOUT=wmma_layout,
        SHARED_A=shared_a,
        SHARED_B=shared_b,
        SHARED_A_SCALE=shared_a_scale,
        SHARED_B_SCALE=shared_b_scale,
        DOT_A_LAYOUT=dot_a_layout,
        DOT_B_LAYOUT=dot_b_layout,
    )
    # Build grouped reduction inputs in a uniform way
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu_reduction,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
    )
    return y_final


if __name__ == "__main__":
    device = torch.device("cuda")

    # Import initialization functions from test file
    from op_tests.triton_tests.moe.test_moe_gemm_a8w8_blockscale import (
        init_routing_data,
        init_compute_data,
        group_shape,
    )

    # Test parameters
    m = 200
    n = 200
    k = 200
    n_expts_tot = 8
    n_expts_act = 2
    do_gather = True
    do_scatter = False
    per_row_x_scale = False
    is_x_blockscale = True
    is_w_blockscale = True
    has_y_gammas = False
    preshuffled = False

    print(f"Configuration:")
    print(f"  Tokens: {m}")
    print(f"  Total Experts: {n_expts_tot}")
    print(f"  Active Experts: {n_expts_act}")
    print(f"  K: {k}, N: {n}")
    print(f"  Group shape: {group_shape}")
    print(f"  Per-row x scale: {per_row_x_scale}")
    print(f"  X blockscale: {is_x_blockscale}, W blockscale: {is_w_blockscale}")
    print()
    
    # Initialize routing data using test file function
    print("Performing routing...")
    m, routing_data, gather_indx, scatter_indx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )
    print(f"  Routing block_m: {routing_data.block_m}")
    print(f"  Expert histogram: {routing_data.expt_hist.tolist()}")
    print()
    
    # Initialize compute data using test file function
    print("Creating input tensors...")
    (
        x_quant,
        x_block_scales,
        x_static_scale,
        w_quant,
        w_block_scales,
        w_static_scale,
        bias,
        gammas,
    ) = init_compute_data(
        m,
        n,
        k,
        gather_indx,
        scatter_indx,
        n_expts_tot,
        n_expts_act,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        has_y_gammas,
        device=device,
        per_row_x_scale=per_row_x_scale,
        is_x_blockscale=is_x_blockscale,
        is_w_blockscale=is_w_blockscale,
    )
    
    print(f"  X shape: {x_quant.shape}, dtype: {x_quant.dtype}")
    print(f"  W shape: {w_quant.shape}, dtype: {w_quant.dtype}")
    if x_block_scales is not None:
        print(f"  X scales shape: {x_block_scales.shape} (per_row_x_scale={per_row_x_scale})")
    else:
        print(f"  X static scale: {x_static_scale}")
    if w_block_scales is not None:
        print(f"  W scales shape: {w_block_scales.shape}")
    else:
        print(f"  W static scale: {w_static_scale}")
    print(f"  Bias shape: {bias.shape}")
    print()
    
    # Run the Gluon kernel
    print("Running MOE GEMM with Gluon kernel...")
    try:
        y_gluon = moe_gemm_a8w8_blockscale_gfx1250(
            x=x_quant,
            w=w_quant,
            x_block_scales=x_block_scales,
            w_block_scales=w_block_scales,
            x_static_scale=x_static_scale,
            w_static_scale=w_static_scale,
            quant_static_scale=None,
            bias=bias,
            routing_data=routing_data,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            gammas=gammas,
            out_dtype=torch.bfloat16,
            apply_swiglu=False,
            per_row_x_scale=per_row_x_scale,
            preshuffled=preshuffled,
        )
        print("========= y_gluon =========")
        print(y_gluon)
        
    except Exception as e:
        print(f"ERROR running Gluon kernel: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import moe_gemm_a8w8_blockscale as ref_moe_gemm_a8w8_blockscale
    
    y_ref = ref_moe_gemm_a8w8_blockscale(
        x_quant,
        w_quant,
        x_block_scales,
        w_block_scales,
        x_static_scale,
        w_static_scale,
        None,  # quant_static_scale
        bias,
        routing_data,
        gather_indx,
        scatter_indx,
        gammas,
        torch.bfloat16,  # out_dtype
        False,  # apply_swiglu
        per_row_x_scale=per_row_x_scale,
    )
    print("========= y_ref =========")
    print(y_ref)
    
    # Compute error
    abs_diff = torch.abs(y_gluon - y_ref)
    rel_diff = abs_diff / (torch.abs(y_ref) + 1e-8)
    
    total_elements = y_gluon.numel()
    maxtol = 6e-1
    rmstol = 6e-2
    
    from op_tests.triton_tests.moe.test_moe_gemm_a8w8_blockscale import assert_close
    assert_close(y_gluon, y_ref, maxtol=maxtol, rmstol=rmstol)
    
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
