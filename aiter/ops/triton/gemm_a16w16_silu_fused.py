# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from .activation import _silu_exp2

_LOGGER = AiterTritonLogger()

@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _gemm_a16_w16_kernel_silu_fused_swizzle_n(
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
    stride_cm,
    stride_cn,
    next_M_2,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SWIZZLE_N: tl.constexpr = False,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid = remap_xcd(pid, num_pid_m * num_pid_n)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # swizzle elements in N dimension for B ptrs
    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2
    offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
    # (i % 2): [0, 1, 0, 1, ...] (alternating)
    # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
    # So offs_w1n now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
    offs_bn = (offs_half + (i % 2) * (N // 2)) % N
    offs_am = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    if ADD_BIAS:
        accumulator = tl.load(bias_ptr + offs_bn).to(dtype=acc_dtype)
        accumulator = tl.broadcast_to(
            accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
        )
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
                cache_modifier=cache_modifier,
            )

        accumulator += tl.dot(a, b, input_precision="ieee")

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    
    
    silu_acc, mul_acc = accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
    silu_acc = silu_acc / (1.0 + tl.exp2(-(silu_acc * 1.44269504089)))
    c = (silu_acc * mul_acc).to(c_ptr.type.element_ty)


    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N//2)
    tl.store(c_ptrs, c, mask=c_mask)




@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _gemm_a16_w16_kernel_silu_fused(
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
    stride_cm,
    stride_cn,
    next_M_2,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid = remap_xcd(pid, num_pid_m * num_pid_n)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    offs_i0 = tl.arange(0, BLOCK_SIZE_HALF).to(tl.int64)
    offs_i1 = (tl.arange(0, BLOCK_SIZE_HALF) + N // 2).to(tl.int64)
    # offset for silu_acc
    i0 = (pid_n * BLOCK_SIZE_HALF + offs_i0) % (N // 2)
    # offset for mul_acc
    i1 = (pid_n * BLOCK_SIZE_HALF + offs_i1) % N
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs_i0 = (
        b_ptr
        + (offs_k[:, None] * stride_bk + i0[None, :] * stride_bn)
    )
    b_ptrs_i1 = (
        b_ptr
        + (offs_k[:, None] * stride_bk + i1[None, :] * stride_bn)
    )

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    
    if ADD_BIAS:
        silu_acc = tl.load(bias_ptr + i0).to(dtype=acc_dtype)
        silu_acc = tl.broadcast_to(
            silu_acc[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_HALF)
        )
        mul_acc = tl.load(bias_ptr + i1).to(dtype=acc_dtype)
        mul_acc = tl.broadcast_to(
            mul_acc[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_HALF)
        )
    else:
        silu_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_HALF), dtype=acc_dtype)
        mul_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_HALF), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        
        if EVEN_K:
            a = tl.load(a_ptrs)
            b_i0 = tl.load(b_ptrs_i0, cache_modifier=cache_modifier)
        else:
            a = tl.load(
                a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
            )
            b_i0 = tl.load(
                b_ptrs_i0, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )

        silu_acc += tl.dot(a, b_i0, input_precision="ieee")

        if EVEN_K:
            b_i1 = tl.load(b_ptrs_i1, cache_modifier=cache_modifier)
        else:
            b_i1 = tl.load(
                b_ptrs_i1, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )

        # Perform dot operation and apply scale
        mul_acc += tl.dot(a, b_i1, input_precision="ieee")

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs_i0 += BLOCK_SIZE_K * stride_bk
        b_ptrs_i1 += BLOCK_SIZE_K * stride_bk

    # gated activation
    silu_acc = _silu_exp2(silu_acc)
    accumulator = silu_acc * mul_acc
    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N//2)
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config
            return _get_config._config_dict["default"]["any"]

    bounds = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for bound in bounds:
        if M <= bound and f"M_LEQ_{bound}" in _get_config._config_dict[key]:
            return _get_config._config_dict[key][f"M_LEQ_{bound}"]
    return _get_config._config_dict[key]["any"]


def gemm_a16w16_silu_fused(
    x,
    w,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 16 bit matmul Y = X x W

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - bias: Vector with shape (N).
    - dtype: Optional parameter to specifcy bf16 or fp16 datatype. Default is bf16
    - Y: Output Matrix Y with shape (M, N///2). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N//2). //2 because of the gated SiLU activation.
    """

    _LOGGER.info(f"GEMM_A16W16: x={tuple(x.shape)} w={tuple(w.shape)}")
    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, K = w.shape
    w = w.T
    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    # print("config", config)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # print("grid size", triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]))
    
    _gemm_a16_w16_kernel_silu_fused[grid](
        x,
        w,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        ADD_BIAS=(bias is not None),
        next_M_2=triton.next_power_of_2(M),
        **config,
    )

    return y
