# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
import os
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"]) == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _gemm_a16_w16_atomic_kernel(
    a_ptr,
    b_ptr,
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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    DISABLE_BUFFER_LOADS: tl.constexpr,
    cache_modifier: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    if not DISABLE_BUFFER_LOADS:
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid = remap_xcd(pid, GRID_MN)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    num_k_iter = tl.cdiv(K, NUM_KSPLIT * BLOCK_SIZE_K)

    # Create pointers for first block of A and B input matrices
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    ).to(tl.int64)
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    ).to(tl.int64)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(num_k_iter - 1):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=cache_modifier)

        accumulator += tl.dot(a, b, input_precision="ieee")

        # Advance the ptrs to the next K block.
        a_ptrs += (NUM_KSPLIT * BLOCK_SIZE_K) * stride_ak
        b_ptrs += (NUM_KSPLIT * BLOCK_SIZE_K) * stride_bk

    a = tl.load(
        a_ptrs, mask=offs_k[None, :] < K - (num_k_iter - 1) * (NUM_KSPLIT * BLOCK_SIZE_K), other=0.0
    )
    b = tl.load(
        b_ptrs, mask=offs_k[:, None] < K - (num_k_iter - 1) * (NUM_KSPLIT * BLOCK_SIZE_K), other=0.0
    )
    accumulator += tl.dot(a, b, input_precision="ieee")

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if NUM_KSPLIT == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask, sem="relaxed")


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-ATOMIC.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-ATOMIC-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    if key == "default":
        key = f"{M}_{N}"
        if key not in _get_config._config_dict.keys():
            dev = arch_info.get_device()
            fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-ATOMIC-M={M}-N={N}.json"
            if os.path.exists(fpath):
                with open(fpath, "r") as file:
                    config = json.load(file)
                    _get_config._config_dict[key] = config
            else:
                key = "default"  # fall back to default config

    if M < 32 and "small" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small"]
    elif M >= 32 and M <= 128 and ("medium_M32" in _get_config._config_dict[key] or "medium_M64" in _get_config._config_dict[key] or "medium_M128" in _get_config._config_dict[key]):
        BLK_M = triton.next_power_of_2(M)
        if BLK_M == 32 and "medium_M32" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M32"]
        elif BLK_M == 64 and "medium_M64" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M64"]
        elif BLK_M == 128 and "medium_M128" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M128"]
    elif M > 128 and M <= 256 and "large" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["large"]
    elif M > 256 and "xlarge" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["xlarge"]
    else:
        return _get_config._config_dict[key]["any"]


def gemm_a16w16_atomic(
    x,
    w,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 16 bit matmul Y = X x W
    NOTE: If dtype is set to bf16, aggregation in bf16 with atomic_add will lead to slight precision loss.
    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - dtype: Optional parameter to specifcy bf16 or fp16 datatype. Default is bf16
    - Y: Output Matrix Y with shape (M, N). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    _LOGGER.info(
        f"GEMM_A16W16_ATOMIC: x.shape={tuple(x.shape)}, w.shape={tuple(w.shape)} "
    )

    w = w.T

    M, K = x.shape
    K, N = w.shape

    if config is None:
        config = _get_config(M, N, K)
    # For compatability reasons, these keys may not exist in the config
    # TODO: This needs to be embedded in the configs later
    if "NUM_KSPLIT" not in config:
        config["NUM_KSPLIT"] = 1
    if "cache_modifier" not in config:
        config["cache_modifier"] = ""
    if "DISABLE_BUFFER_LOADS" not in config:
        config["DISABLE_BUFFER_LOADS"] = False

    if y is None:
        # atomic add requires 0 tensor
        if config["NUM_KSPLIT"] == 1:
            y = torch.empty((M, N), dtype=dtype, device=x.device)
        else:
            y = torch.zeros((M, N), dtype=dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"])
        * META["NUM_KSPLIT"],
    )
    kernel = _gemm_a16_w16_atomic_kernel[grid](
        x,
        w,
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
        **config,
    )

    return y
