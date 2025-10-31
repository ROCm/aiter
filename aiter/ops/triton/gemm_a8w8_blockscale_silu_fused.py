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


@triton.jit
def repeat_interleave_triton(
    x,
    xM: tl.constexpr,
    xN: tl.constexpr,
    group_size: tl.constexpr,
    broadcast_dim: tl.constexpr,
):
    """
    Broadcasts the input tensor `x` along the specified dimension `broadcast_dim`
    in groups of size `group_size`.

    Parameters:
    - x: Input tensor to be broadcasted.
    - group_size: Size of each group for broadcasting.
    - broadcast_dim: Dimension along which to perform the broadcasting.

    Returns:
    - A tensor with the same shape as `x`, but with values broadcasted
      in groups along the specified dimension.
    """
    if broadcast_dim == 0:
        assert xM > 0, "broadcast_dim must be specified"
        if xM > 1:
            x = x.reshape(xM, 1, xN)
            x = tl.broadcast_to(x, (xM, group_size, xN))
            x = x.reshape(xM * group_size, xN)
        # else: singleton dimension, no need to broadcast
    else:
        assert xN > 0, "broadcast_dim must be specified"
        if xN > 1:
            x = x.reshape(xM, xN, 1)
            x = tl.broadcast_to(x, (xM, xN, group_size))
            x = x.reshape(xM, xN * group_size)

    return x


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
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
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid = remap_xcd(pid, GRID_MN)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2

    num_scales_along_n: tl.constexpr = (BLOCK_SIZE_HALF + GROUP_N - 1) // GROUP_N
    num_scales_along_k: tl.constexpr = (BLOCK_SIZE_K + GROUP_K - 1) // GROUP_K
    tl.static_assert(num_scales_along_k == 1, "BLOCK_SIZE_K must be <= group_k")
    
    
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    offs_i0 = tl.arange(0, BLOCK_SIZE_HALF).to(tl.int64)
    offs_i1 = (tl.arange(0, BLOCK_SIZE_HALF) + N // 2).to(tl.int64)
    # offset for silu_acc
    i0 = (pid_n * BLOCK_SIZE_HALF + offs_i0) % N
    # offset for mul_acc
    i1 = (pid_n * BLOCK_SIZE_HALF + offs_i1) % N
    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
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

    # Create pointers for the scales
    a_scale_ptrs = (
        a_scale_ptr + offs_am * stride_ascale_m
    )

    i0s = pid_n * (BLOCK_SIZE_HALF // GROUP_N) + tl.arange(0, num_scales_along_n)
    i1s = i0s + ((N // 2) // GROUP_N)
    b_i0_scale_ptrs = (
        b_scale_ptr + i0s[None, :] * stride_bn
    )
    b_i1_scale_ptrs = (
        b_scale_ptr + i1s[None, :] * stride_bn
    )

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    silu_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_HALF), dtype=acc_dtype)
    mul_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_HALF), dtype=acc_dtype)

    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b_i0 = tl.load(b_ptrs_i0, cache_modifier=cache_modifier)
            b_i1 = tl.load(b_ptrs_i1, cache_modifier=cache_modifier)
        else:
            a = tl.load(
                a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
            )
            b_i0 = tl.load(
                b_ptrs_i0, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )
            b_i1 = tl.load(
                b_ptrs_i1, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )

        # scale loads
        start_k = (k * BLOCK_SIZE_K) // GROUP_K

        a_scale = tl.load(
            a_scale_ptrs + start_k * stride_ascale_k, other=0.0
        )

        b_i0_scale = tl.load(
            b_i0_scale_ptrs + start_k * stride_bk,
        )
        b_i1_scale = tl.load(
            b_i1_scale_ptrs + start_k * stride_bk,
        )

        b_i0_scale = repeat_interleave_triton(
            b_i0_scale, 1, num_scales_along_n, GROUP_N, 1
        )
        b_i1_scale = repeat_interleave_triton(
            b_i1_scale, 1, num_scales_along_n, GROUP_N, 1
        )

        # Perform dot operation and apply scale
        silu_acc += tl.dot(a, b_i0, out_dtype=tl.float32) * a_scale * b_i0_scale
        mul_acc += tl.dot(a, b_i1, out_dtype=tl.float32) * a_scale * b_i1_scale

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs_i0 += BLOCK_SIZE_K * stride_bk
        b_ptrs_i1 += BLOCK_SIZE_K * stride_bk

    # gated activation
    silu_acc = _silu_exp2(silu_acc)
    accumulator = silu_acc * mul_acc
    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF).to(tl.int64)
    c_ptrs = (
        c_ptr
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
        + pid_k * stride_ck
    )
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
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8_BLOCKSCALE.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    if M < 32 and "small" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small"]
    elif M <= 128:
        BLK_M = triton.next_power_of_2(M)
        if BLK_M == 32 and "medium_M32" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M32"]
        elif BLK_M == 64 and "medium_M64" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M64"]
        elif BLK_M == 128 and "medium_M128" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M128"]
    elif M <= 256 and "large" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["large"]
    else:
        BLK_M = triton.next_power_of_2(M)
        if f"xlarge_M{BLK_M}" in _get_config._config_dict[key]:
            return _get_config._config_dict[key][f"xlarge_M{BLK_M}"]
        elif "xlarge" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["xlarge"]

    return _get_config._config_dict[key]["any"]


def gemm_a8w8_blockscale_silu_fused(
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

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (  # noqa: E731
        (
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a8w8_blockscale_kernel[grid](
        x,
        w,
        y,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )

    return y
