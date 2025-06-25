# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _gemm_a8w8_kernel(
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
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8 function
    below

    Computes the 8 bit matmul C = A x B, applies a conversion scale and optionally adds a bias to
    the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: First scale tensor with shape (M, 1).
    - B_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    """

    NUM_XCDS: tl.constexpr = 8

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

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Create pointers for the scale tensors and load them
    offs_a_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
    offs_b_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
    a_scale = tl.load(a_scale_ptr + offs_a_scale)
    b_scale = tl.load(b_scale_ptr + offs_b_scale)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator += tl.dot(a, b, input_precision="ieee")

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale
    accumulator *= a_scale[:, None] * b_scale[None, :]

    # Add bias
    if HAS_BIAS:
        offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
        bias = tl.load(bias_ptr + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    return _get_config._config_dict["any"]


def gemm_a8w8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 8 bit matmul Y = X x WT, applies a conversion scale and optionally adds a bias
    to the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - X_scale: First scale tensor with shape (M, 1).
    - W_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    - Y: Output Matrix Y with shape (M, K). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    # Transpose w
    w = w.T

    M, K = x.shape
    K, N = w.shape

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gemm_a8w8_kernel[grid](
        x,
        w,
        y,
        x_scale,
        w_scale,
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        bias is not None,
        **config,
    )

    return y
