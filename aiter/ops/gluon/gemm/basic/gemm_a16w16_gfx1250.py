# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import math
from typing import Optional, Dict
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.gluon.activations import _get_activation_from_str
from aiter.ops.gluon.utils.gemm_config_utils import get_gemm_config
from aiter.ops.gluon.utils.prefetch import issue_l2_prefetches, issue_l2_prefetches_prologue
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _get_config(M: int, N: int, K: int):
    config, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K)
    return config, is_tuned


def create_shared_layouts(BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, 
                          BLOCK_K: gl.constexpr, PHYSICAL_MK: gl.constexpr,
                          PHYSICAL_KN: gl.constexpr):
    if PHYSICAL_MK:
        SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
        )
    else:
        SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_M, 8]], [BLOCK_K, BLOCK_M], [1, 0]
        )
    
    if PHYSICAL_KN:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
        )
    else:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
        )
    
    return (SHARED_LAYOUT_A, SHARED_LAYOUT_B)


@gluon.jit
def _gemm_a16w16_gfx1250_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn, 
    BLOCK_M: gl.constexpr,     
    BLOCK_N: gl.constexpr,      
    BLOCK_K: gl.constexpr,      
    NUM_BUFFERS: gl.constexpr,  
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,      
    OPERAND_LAYOUT_A: gl.constexpr, 
    OPERAND_LAYOUT_B: gl.constexpr, 
    activation: gl.constexpr,       
    USE_ACTIVATION: gl.constexpr,   
    ADD_BIAS: gl.constexpr,         
    L2_PREFETCH_DISTANCE: gl.constexpr,
):
    USE_L2_PREFETCH: gl.constexpr = L2_PREFETCH_DISTANCE > 0

    pid = gl.program_id(axis=0) 
    num_pid_m = gl.cdiv(M, BLOCK_M) 
    pid_m = pid % num_pid_m            
    pid_n = pid // num_pid_m           
    
    # Tensor descriptors to tell the TDM hardware how to load the tensors
    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(K, M),
            strides=(stride_ak, stride_am),  
            block_shape=(BLOCK_K, BLOCK_M),  
            layout=SHARED_LAYOUT_A
        )
    
    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B
        )
    
    # Allocate shared memory buffers for tiles of A and B.
    # These buffers will be used for pipelined loads from global to shared to registers.
    if PHYSICAL_MK:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A
        )
    else:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A
        )
    
    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B
        )
    
    load_idx = 0
    compute_idx = 0
    
    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    if USE_L2_PREFETCH:
        issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, load_idx, a_desc, b_desc, off_am, off_bn,
                                      BLOCK_K, NUM_BUFFERS, not PHYSICAL_MK, not PHYSICAL_KN)

    # Fill the pipeline
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [load_idx * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )
        
        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [load_idx * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, load_idx * BLOCK_K],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )
        
        load_idx += 1
    
    # Main pipeline loop
    num_k_tiles = gl.cdiv(K, BLOCK_K)
    
    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [load_idx * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )
        
        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [load_idx * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, load_idx * BLOCK_K],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )
        
        load_idx += 1

        if USE_L2_PREFETCH:
            issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, load_idx, a_desc, b_desc,
                                  off_am, off_bn, BLOCK_K, not PHYSICAL_MK, not PHYSICAL_KN)
        
        # Wait for the next tile to be ready for compute
        # (NUM_BUFFERS - 1) * 2 ensures the current compute tile is loaded while allowing next loads to proceed
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        
        if PHYSICAL_MK:
            cur_a = a_buffer.index(compute_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_A)
        else:
            cur_a = a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
        
        if PHYSICAL_KN:
            cur_b = b_buffer.index(compute_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_B)
        else:
            cur_b = b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
        
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        
        compute_idx += 1
    
    # No more loads
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        
        if PHYSICAL_MK:
            cur_a = a_buffer.index(compute_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_A)
        else:
            cur_a = a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
        
        if PHYSICAL_KN:
            cur_b = b_buffer.index(compute_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_B)
        else:
            cur_b = b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
        
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1
    
    #Bias
    if ADD_BIAS:
        offs_bias = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
        bias_vals = gl.load(bias_ptr + offs_bias)
        accumulator = accumulator + bias_vals[None, :]
    
    #Activation
    if USE_ACTIVATION:
        accumulator = activation(accumulator)
    
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
    
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    #Store
    gl.amd.gfx1250.buffer_store(
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
    config: Optional[Dict] = None,        
    activation: Optional[str] = None,     
):
    """
    Compute 16 bit gemm y = x @ w^T + bias

    Args:
        x: Input tensor of shape (M, K)
        w: Weight tensor of shape (N, K), internally transposed
        bias: Optional bias tensor of shape (N,)
        dtype: Output data type
        y: Optional pre-allocated output tensor
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
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, K = w.shape
    w = w.T

    if x.stride(1) == 1:
        physical_mk = True
    elif x.stride(0) == 1:
        physical_mk = False

    if w.stride(1) == 1:
        physical_kn = True
    elif w.stride(0) == 1:
        physical_kn = False

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
    
    wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        instr_shape=[16, 16, 32]
    )
    
    operand_a = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    operand_b = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)
    
    shared_layouts = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, physical_mk, physical_kn)
    shared_a, shared_b = shared_layouts[0], shared_layouts[1]
    
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    grid = (num_tiles_m * num_tiles_n, 1)
    
    _gemm_a16w16_gfx1250_kernel[grid](
        x, w, y, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        PHYSICAL_MK=physical_mk,
        PHYSICAL_KN=physical_kn,
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
