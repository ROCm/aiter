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

    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout
    )
    zeros = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout
    )

    # ----- Main Loop --------

    for k in range(k_tiles_count - 1):
        # Advance the ptrs to the next K block.
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k

        # Loading a scale and curr A scale
        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )
        # Loading b scale and curr b scale
        cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr,
            offsets=offs_b_scale,
            cache=cache_modifier,
        )

        
        # Load A for tiling. now loads both A and B
        gl.amd.gfx1250.tdm.async_load(a_desc, [off_am_tdm, num_loads * BLOCK_SIZE_K], tdm_smem_a.index(num_loads % NUM_BUFFERS),
                                pred=1)
        gl.amd.gfx1250.tdm.async_load(b_desc, [num_loads * BLOCK_SIZE_K, off_bm_tdm], tdm_smem_b.index(num_loads % NUM_BUFFERS),
                                    pred=1)
        num_loads += 1
        
        # prefetching
        if L2_PREFETCH_DISTANCE - 1 != 0:
            prefetch_iteration = num_loads + L2_PREFETCH_DISTANCE - 1
            gl.amd.gfx1250.tdm.prefetch(a_desc, [0, prefetch_iteration * BLOCK_SIZE_K], pred=True)
            gl.amd.gfx1250.tdm.prefetch(b_desc, [prefetch_iteration * BLOCK_SIZE_K, 0], pred=True)
        

        # WMMA 
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        a = tdm_smem_a.index(num_computes % NUM_BUFFERS).load(layout=dot_a_layout)
        b = tdm_smem_b.index(num_computes % NUM_BUFFERS).load(layout=dot_b_layout)
        res = gl.amd.gfx1250.wmma(a, b, zeros)
        num_computes += 1
        acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]

        # Store in WMMA slice order so next iteration's load aligns scale[i] with row/col i
        smem_scale_a.store(a_scale)
        smem_scale_b.store(b_scale) 

    # ======= Epilogue ========

    # load a from the last load
    cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
    cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    for i in gl.static_range(NUM_BUFFERS - 1):
        # wmma
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        a = tdm_smem_a.index(num_computes % NUM_BUFFERS).load(layout=dot_a_layout)
        b = tdm_smem_b.index(num_computes % NUM_BUFFERS).load(layout=dot_b_layout)
        res = gl.amd.gfx1250.wmma(a, b, zeros)
        num_computes += 1
        acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]
        acc += zeros * cur_a_scale[:, None] * cur_b_scale[None, :]
        zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)


    c = acc.to(c_ptr.type.element_ty)

    # # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout)
    )
    c_offs = (
        stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.cdna4.buffer_store(
        stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask
    )


@gluon.jit
def _gemm_a8w8_blockscale_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: gl.constexpr,  # Note: Can be distinct from GEMM block size
    BLOCK_SIZE_N: gl.constexpr,
    ACTUAL_KSPLIT: gl.constexpr,
    MAX_KSPLIT: gl.constexpr,
):

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    blocked_read: gl.constexpr = gl.BlockedLayout(  # (MAX_KSPLIT, BLOCK_M, BLOCK_N)
        size_per_thread=[1, 1, 4],
        threads_per_warp=[1, 8, 4],
        warps_per_cta=[1, 4, 1],
        order=[2, 1, 0],
    )

    # blocked_write: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[1, 4], # (BLOCK_M, BLOCK_N)
    #     threads_per_warp=[4, 8],
    #     warps_per_cta=[4, 1],
    #     order=[1, 0],
    # )

    offs_m = pid_m * BLOCK_SIZE_M + gl.arange(
        0,
        BLOCK_SIZE_M,  # keep dim 1
        gl.SliceLayout(0, gl.SliceLayout(2, blocked_read)),
    )
    offs_n = pid_n * BLOCK_SIZE_N + gl.arange(
        0,
        BLOCK_SIZE_N,  # keep dim 2
        gl.SliceLayout(0, gl.SliceLayout(1, blocked_read)),
    )
    offs_k = gl.arange(
        0, MAX_KSPLIT, gl.SliceLayout(1, gl.SliceLayout(2, blocked_read))  # keep dim 0
    )
    c_in_offs = (
        (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )
    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c_in_mask = (offs_m[None, :, None] < M) & (offs_n[None, None, :] < N)
        c = gl.amd.cdna4.buffer_load(c_in_ptr, c_in_offs, mask=c_in_mask, cache=".ca")
    else:
        c_in_mask = (
            (offs_m[None, :, None] < M)
            & (offs_n[None, None, :] < N)
            & (offs_k[:, None, None] < ACTUAL_KSPLIT)
        )
        c = gl.amd.cdna4.buffer_load(
            c_in_ptr, c_in_offs, mask=c_in_mask, cache=".ca"
        )  # , other=0.0)
    c = tl.sum(c, 0)

    c = c.to(c_out_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, gl.SliceLayout(1, gl.SliceLayout(0, blocked_read))
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, gl.SliceLayout(0, gl.SliceLayout(0, blocked_read))
    )
    c_out_offs = (offs_cm[:, None] * stride_c_out_m) + (
        offs_cn[None, :] * stride_c_out_n
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.cdna4.buffer_store(
        stored_value=c, ptr=c_out_ptr, offsets=c_out_offs, mask=c_mask
    )


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_arch()
        if int(dev.split("gfx")[1]) < 950:
            raise ValueError(
                "Gluon implementation is not supported on this device (requires CDNA4)."
            )
        _get_config._config_dict = {}
        fpath = (
            f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/gfx1250-GEMM-A8W8_BLOCKSCALE.json"
        )
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_arch()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    # Config keys should be named M_LEQ_<bound> or "any"
    bounds = []
    for setting in _get_config._config_dict[key].keys():
        potential_block_m = setting.replace("M_LEQ_", "")
        if potential_block_m.isnumeric():
            bounds.append(int(potential_block_m))

    for bound in bounds:
        if M <= bound and f"M_LEQ_{bound}" in _get_config._config_dict[key]:
            config = _get_config._config_dict[key][f"M_LEQ_{bound}"]
            break
        else:
            config = _get_config._config_dict[key]["any"]

    config = (
        config.copy()
    )  # avoid later inplace modification from interacting with cached config

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    return config


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 8 bit matmul Y = X x WT using the block-scale quantization approach.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - X_scale: Scale tensor for X with shape (M, *scale_k).
    - W_scale: Scale tensor for W with shape (**scale_n, *scale_k).

    Returns:
    - Y: The output matrix with shape (M, N).

    *scale_k = (K + scale_block_size_k - 1) // scale_block_size_k
    **scale_n = (N + scale_block_size_n - 1) // scale_block_size_n
    """
    _LOGGER.info(
        f"GEMM_A8W8_BLOCKSCALE: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = x.shape
    N, K = w.shape

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    # Transpose w and w_scale
    w = w.T
    w_scale = w_scale.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1]))

    if config["NUM_KSPLIT"] == 1:
        assert (
            config["GROUP_K"] == config["BLOCK_SIZE_K"]
        ), f"GROUP_K: {config['GROUP_K']} must equal BLOCK_SIZE_K: {config['BLOCK_SIZE_K']} when not using KSPLIT"

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
        )
    else:
        y_pp = None

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    NUM_WARPS=config["num_warps"]
    warp_bases = [(0, 1)]
    for i in range(int(math.log2(NUM_WARPS // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)
    #print(x)
    # print(w)
    # print(x_scale)
    # print(w_scale)
    # print(y)
    _gemm_a8w8_blockscale_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        NUM_WARPS=config["num_warps"],
        warp_bases=warp_bases,
        **config,
    )
    print("complete")
    print(y_pp)

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )

        _gemm_a8w8_blockscale_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return y
