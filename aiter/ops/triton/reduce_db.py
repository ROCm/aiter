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

_LOGGER = AiterTritonLogger()


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % (args["NUM_MSPLIT"] * args["BLOCK_SIZE_M"]) == 0
    }
)
@triton.jit
def _reduce_db_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_MSPLIT: tl.constexpr,
    num_stages: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_xm > 0)
    tl.assume(stride_xn > 0)

    pid_unified = tl.program_id(axis=0)
    pid_m = pid_unified % NUM_MSPLIT
    pid_n = pid_unified // NUM_MSPLIT

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    num_m_iter = tl.cdiv(M, NUM_MSPLIT * BLOCK_SIZE_M)

    # Create pointers for first block of X
    offs_xm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_xn[None, :] * stride_xn)

    acc_dtype = tl.float32 if y_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=acc_dtype)

    for m in tl.range(0, num_m_iter, num_stages=num_stages):
        # Load the next block of X, generate a mask by checking the M dimension.
        # If M is out of bounds, set it to 0.
        if EVEN_M:
            x = tl.load(x_ptrs, cache_modifier=".cg")
        else:
            x = tl.load(
                x_ptrs,
                mask=offs_xm[:, None] < M - m * (NUM_MSPLIT * BLOCK_SIZE_M),
                other=0.0,
                cache_modifier=".cg",
            )

        accumulator += tl.sum(x, 0, dtype=tl.float32)

        # Advance the ptrs to the next block.
        x_ptrs += (NUM_MSPLIT * BLOCK_SIZE_M) * stride_xm

    y = accumulator.to(y_ptr.type.element_ty)

    # Write back the block of the output vector Y with masks.
    offs_yn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + offs_yn
    y_mask = offs_yn < N
    if NUM_MSPLIT > 1:
        tl.atomic_add(y_ptrs, y, mask=y_mask, sem="relaxed")
    else:
        tl.store(y_ptrs, y, mask=y_mask)
 

@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-REDUCE-DB.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = "default"

    return _get_config._config_dict[key]["any"]


def reduce_db(
    x,
    dtype: Optional[float] = torch.float32,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the bias reduction across the batch dimension

    Key parameters:
    - X: Matrix X with shape (M, N).
    - dtype: Optional parameter to specify datatype. Default is fp32
    - Y: Output Matrix Y with shape (N,). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    _LOGGER.info(f"REDUCE DB: x={tuple(x.shape)}")

    M, N = x.shape

    if y is None:
        y = torch.zeros((N,), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(N, META["BLOCK_SIZE_N"]) * META["NUM_MSPLIT"],
    )
    kernel = _reduce_db_kernel[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        **config,
    )
    #print(kernel.asm['amdgcn'])

    return y


