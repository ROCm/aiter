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


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 2,
                "waves_per_eu": 4,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 2,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # M  N  K  Triton (us)  config
        # 16  5120  2880    21.480  (config = 16 128 512 1 8 2 2 16 0)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 2,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": ".cg",
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 16  2880  4096    26.920  (config = 16 128 512 1 8 2 1 16 0)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": ".cg",
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 16  128  2880     7.320  (config = 16 16 512 1 8 2 8 16 0)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 8,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": ".cg",
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 32  5120  2880    17.640  (config = 16 128 256 4 8 2 1 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 32  2880  4096    20.800  (config = 16 128 256 1 8 2 1 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 32  128  2880     7.240  (config = 16 16 512 1 8 2 8 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 8,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 64  5120  2880    20.480  (config = 16 128 512 1 8 2 1 16 0)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": ".cg",
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 64  2880  4096    21.441  (config = 16 128 256 1 8 2 4 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 4,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 64  128  2880     7.240  (config = 16 16 512 1 8 2 1 16 0)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": ".cg",
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 128  5120  2880    23.160  (config = 16 128 128 4 4 2 1 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=4,
            num_stages=2,
        ),
        # 128  2880  4096    29.600  (config = 16 128 256 4 8 2 4 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 4,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 128  128  2880     7.160  (config = 16 16 512 1 8 2 8 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 8,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        # 8192  5120  2880   307.442  (config = 128 128 128 1 4 2 2 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 2,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=4,
            num_stages=2,
        ),
        # 8192  2880  4096   240.442  (config = 128 128 128 4 4 2 2 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 2,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=4,
            num_stages=2,
        ),
        # 8192  128  2880    21.280  (config = 128 32 256 4 8 2 2 16 1)
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 2,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": None,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["next_M_2", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
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

    # remap_xcd(pid, GRID_MN)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    if ADD_BIAS:
        accumulator = tl.load(bias_ptr + offs_bn, cache_modifier=".cg").to(
            dtype=acc_dtype
        )
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

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
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

    if M < 128 and "small" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small"]
    else:
        return _get_config._config_dict[key]["any"]


def gemm_a16w16(
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
    - Y: Output Matrix Y with shape (M, N). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    M, K = x.shape
    N, K = w.shape
    w = w.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # if config is None:
    #    config = _get_config(M, N, K)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a16_w16_kernel[grid](
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
    )

    return y
