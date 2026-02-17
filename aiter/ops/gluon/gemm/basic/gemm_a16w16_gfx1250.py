# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import math
from typing import Optional, Dict
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

from aiter.ops.gluon.activations import _get_activation_from_str
from aiter.ops.gluon.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _get_config(M: int, N: int, K: int):
    config, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K)
    return config, is_tuned


def create_shared_layouts(BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, 
                          BLOCK_K: ttgl.constexpr, PHYSICAL_MK: ttgl.constexpr,
                          PHYSICAL_NK: ttgl.constexpr):
    """
    For A/B matrix: 
      - If PHYSICAL_**: stored as (M, K) - standard layout
      - If not PHYSICAL_**: stored as (K, M) - transposed layout
    """
    if PHYSICAL_MK:
        SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
        )
    else:
        SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_M, 8]], [BLOCK_K, BLOCK_M], [1, 0]
        )
    
    if PHYSICAL_NK:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
        )
    else:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
        )
    
    return (SHARED_LAYOUT_A, SHARED_LAYOUT_B)


@gluon.jit
def issue_l2_prefetches(distance, producer, a_desc, b_desc, off_am, off_bn,
                         BLOCK_K: ttgl.constexpr, PHYSICAL_MK: ttgl.constexpr,
                         PRESHUFFLED: ttgl.constexpr, pred=True):
    """
    Creates L2 prefetch for iteration `producer + distance`.
    """
    if distance == 0:
        return

    prefetch_iteration = producer + distance

    if PHYSICAL_MK:
        ttgl.amd.gfx1250.tdm.prefetch(a_desc, [off_am, prefetch_iteration * BLOCK_K], pred=pred)
    else:
        ttgl.amd.gfx1250.tdm.prefetch(a_desc, [prefetch_iteration * BLOCK_K, off_am], pred=pred)

    if PRESHUFFLED:
        ttgl.amd.gfx1250.tdm.prefetch(b_desc, [off_bn, prefetch_iteration * BLOCK_K], pred=pred)
    else:
        ttgl.amd.gfx1250.tdm.prefetch(b_desc, [prefetch_iteration * BLOCK_K, off_bn], pred=pred)


@gluon.jit
def issue_l2_prefetches_prologue(distance, producer, a_desc, b_desc, off_am, off_bn,
                                  BLOCK_K: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr,
                                  PHYSICAL_MK: ttgl.constexpr, PRESHUFFLED: ttgl.constexpr, pred=True):
    """
    Creates prefetches for iterations [NUM_BUFFERS, distance - NUM_BUFFERS) or no prefetches if distance <= NUM_BUFFERS.
    This skips iterations which are preloaded in the prologue because prefetching them does not make sense for GEMMs.
    """
    if distance <= NUM_BUFFERS:
        return

    for i in ttgl.static_range(distance - NUM_BUFFERS):
        issue_l2_prefetches(NUM_BUFFERS + i, producer, a_desc, b_desc, off_am, off_bn,
                             BLOCK_K, PHYSICAL_MK, PRESHUFFLED, pred)


@gluon.jit
def _gemm_a16w16_gfx1250_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn, 
    BLOCK_M: ttgl.constexpr,     
    BLOCK_N: ttgl.constexpr,      
    BLOCK_K: ttgl.constexpr,      
    NUM_BUFFERS: ttgl.constexpr,  
    PHYSICAL_MK: ttgl.constexpr,  
    PRESHUFFLED: ttgl.constexpr,  
    SHARED_LAYOUT_A: ttgl.constexpr,
    SHARED_LAYOUT_B: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,      
    OPERAND_LAYOUT_A: ttgl.constexpr, 
    OPERAND_LAYOUT_B: ttgl.constexpr, 
    activation: ttgl.constexpr,       
    USE_ACTIVATION: ttgl.constexpr,   
    ADD_BIAS: ttgl.constexpr,         
    L2_PREFETCH_DISTANCE: ttgl.constexpr,
):
    USE_L2_PREFETCH: ttgl.constexpr = L2_PREFETCH_DISTANCE > 0

    pid = ttgl.program_id(axis=0) 
    num_pid_m = ttgl.cdiv(M, BLOCK_M) 
    pid_m = pid % num_pid_m            
    pid_n = pid // num_pid_m           
    
    # Tensor descriptors to tell the TDM hardware how to load the tensors
    if PHYSICAL_MK:
        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A
        )
    else:
        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(K, M),
            strides=(stride_ak, stride_am),  
            block_shape=(BLOCK_K, BLOCK_M),  
            layout=SHARED_LAYOUT_A
        )
    
    if PRESHUFFLED:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),  
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B
        )
    else:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B
        )
    
    # Allocate shared memory buffers for tiles of A and B.
    # These buffers will be used for pipelined loads from global to shared to registers.
    if PHYSICAL_MK:
        a_buffer = ttgl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A
        )
    else:
        a_buffer = ttgl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A
        )
    
    if PRESHUFFLED:
        b_buffer = ttgl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B
        )
    else:
        b_buffer = ttgl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B
        )
    
    # Initialize pipeline state
    # producer: Index of next tile to start loading 
    # consumer: Index of next tile to compute
    # With double buffering: producer is always 1 ahead of consumer
    producer = 0
    consumer = 0
    
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=WMMA_LAYOUT)
    
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    if USE_L2_PREFETCH:
        issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K, NUM_BUFFERS, PHYSICAL_MK, PRESHUFFLED)

    # Fill the pipeline
    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        # Load A tile
        if PHYSICAL_MK:
            ttgl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, producer * BLOCK_K],
                a_buffer.index(producer % NUM_BUFFERS)
            )
        else:
            ttgl.amd.gfx1250.tdm.async_load(
                a_desc,
                [producer * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(producer % NUM_BUFFERS)
            )
        
        # Load B tile
        if PRESHUFFLED:
            ttgl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, producer * BLOCK_K],
                b_buffer.index(producer % NUM_BUFFERS)
            )
        else:
            ttgl.amd.gfx1250.tdm.async_load(
                b_desc,
                [producer * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(producer % NUM_BUFFERS)
            )
        
        producer += 1
    
    # Main pipeline loop
    num_k_tiles = ttgl.cdiv(K, BLOCK_K)
    
    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        if PHYSICAL_MK:
            ttgl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, producer * BLOCK_K],
                a_buffer.index(producer % NUM_BUFFERS)
            )
        else:
            ttgl.amd.gfx1250.tdm.async_load(
                a_desc,
                [producer * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(producer % NUM_BUFFERS)
            )
        
        if PRESHUFFLED:
            ttgl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, producer * BLOCK_K],
                b_buffer.index(producer % NUM_BUFFERS)
            )
        else:
            ttgl.amd.gfx1250.tdm.async_load(
                b_desc,
                [producer * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(producer % NUM_BUFFERS)
            )
        
        producer += 1

        if USE_L2_PREFETCH:
            issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K, PHYSICAL_MK, PRESHUFFLED)
        
        # Wait for consumer to be ready
        # We want (NUM_BUFFERS - 1) * 2 to ensure the consumer's tile is ready while allowing next loads to proceed
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        
        if PHYSICAL_MK:
            cur_a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_A)
        else:
            # A is stored as (K, M) but WMMA needs (M, K), so we permute
            cur_a = a_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
        
        if PRESHUFFLED:
            # B is stored as (N, K) but WMMA needs (K, N), so we permute
            cur_b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
        else:
            cur_b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_B)
        
        accumulator = ttgl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        
        consumer += 1
    
    # No more loads
    for i in ttgl.static_range(NUM_BUFFERS - 1):
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        
        if PHYSICAL_MK:
            cur_a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_A)
        else:
            # A is stored as (K, M) but WMMA needs (M, K), so we permute
            cur_a = a_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
        
        if PRESHUFFLED:
            cur_b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
        else:
            cur_b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_B)
        
        accumulator = ttgl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        consumer += 1
    
    #Bias
    if ADD_BIAS:
        offs_bias = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        bias_vals = ttgl.load(bias_ptr + offs_bias)
        accumulator = accumulator + bias_vals[None, :]
    
    #Activation
    if USE_ACTIVATION:
        accumulator = activation(accumulator)
    
    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    #Store
    ttgl.amd.gfx1250.buffer_store(
        accumulator.to(c_ptr.type.element_ty),
        c_ptr,
        offs_c,
        mask=mask_c
    )


def gemm_a16w16_gfx1250(
    x: torch.Tensor,                      
    w: torch.Tensor,                      
    bias: Optional[torch.Tensor] = None,  
    dtype: torch.dtype = torch.bfloat16,
    y: Optional[torch.Tensor] = None,     
    preshuffled: bool = False,            
    config: Optional[Dict] = None,        
    activation: Optional[str] = None,     
):
    """
    Compute 16 bit gemm y = x @ w + bias
    
    Args:
        x: Input tensor of shape (M, K)
        w: Weight tensor of shape (K, N) or (N, K) if preshuffled
        bias: Optional bias tensor of shape (N,)
        dtype: Output data type
        y: Optional pre-allocated output tensor
        preshuffled: If True, w is stored as (N, K)
        config: Kernel tuning parameters:
            - BLOCK_M: Tile size in M dimension (default: 128)
            - BLOCK_N: Tile size in N dimension (default: 128)
            - BLOCK_K: Tile size in K dimension (default: 32)
            - NUM_BUFFERS: Pipeline stages (default: 2)
            - num_warps: Warps per block (default: 8)
        activation: Activation function ("gelu", "gelu_tanh", "silu", "silu_exp2", "relu")

    Pipeline Structure: will be optimized furhter later (2 buffer for now)        
        Prologue: Pre-load tile into LDS buffers
        Main Loop: For each tile: load next tile while computing current tile  
        Epilogue: Compute remaining tiles
    
    Returns:
        Output tensor of shape (M, N)
    """

    _LOGGER.info(f"GEMM_A16W16: x={tuple(x.shape)} w={tuple(w.shape)}")

    assert x.dtype in (torch.float16, torch.bfloat16), \
        f"Activations (x) must be fp16 or bf16, got {x.dtype}"
    assert w.dtype in (torch.float16, torch.bfloat16), f"Weights (w) must be fp16 or bf16, got {w.dtype}"
    
    M, K = x.shape
    if preshuffled:
        N, K_w = w.shape  
    else:
        K_w, N = w.shape
    
    # Shape check
    assert K == K_w, f"K dimension mismatch: x has K={K}, w has K={K_w}"
    
    #Determine the layout of the tensors for TDM
    #Helps prevent memory misalignment from transposed views
    # - physical_mk: True if x is physically (M, K), False if physically (K, M)
    # - physical_nk: True if w is physically (N, K), False if physically (K, N)
    if x.stride(1) == 1:
        physical_mk = True
    elif x.stride(0) == 1:
        physical_mk = False
    
    if preshuffled: #(N, K)
        if w.stride(1) == 1:
            physical_nk = True
        elif w.stride(0) == 1:
            physical_nk = False 
    else:
        if w.stride(1) == 1:
            physical_nk = False
        elif w.stride(0) == 1:
            physical_nk = True

    if y is None:
        y = torch.empty((M, N), device=x.device, dtype=dtype)
    
    if config is None:
        config, _ = _get_config(M, N, K)
    
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    NUM_BUFFERS = config["NUM_BUFFERS"]
    num_warps = config["num_warps"]
    L2_PREFETCH_DISTANCE = config.get("L2_PREFETCH_DISTANCE", 0)
        
    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)
    
    wmma_layout = ttgl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        instr_shape=[16, 16, 32]
    )
    
    operand_a = ttgl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    operand_b = ttgl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)
    
    # Use helper function to define the shared memory layouts
    shared_layouts = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, physical_mk, physical_nk)
    shared_a, shared_b = shared_layouts[0], shared_layouts[1]
    
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    grid = (num_tiles_m * num_tiles_n, 1)
    
    if preshuffled:
        stride_bk, stride_bn = w.stride(1), w.stride(0)
    else:
        stride_bk, stride_bn = w.stride(0), w.stride(1)
    
    _gemm_a16w16_gfx1250_kernel[grid](
        x, w, y, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        stride_bk, stride_bn,
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        PHYSICAL_MK=physical_mk,  
        PRESHUFFLED=physical_nk,  
        SHARED_LAYOUT_A=shared_a,
        SHARED_LAYOUT_B=shared_b,
        WMMA_LAYOUT=wmma_layout,
        OPERAND_LAYOUT_A=operand_a,
        OPERAND_LAYOUT_B=operand_b,
        activation=_get_activation_from_str(activation) if activation else None,
        USE_ACTIVATION=activation is not None,
        ADD_BIAS=(bias is not None),
        L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE,
        num_warps=num_warps,
    )
    
    return y
