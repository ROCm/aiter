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
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN_FP8": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_fp8"], args["BLOCK_SIZE_N"]),
        "GRID_MN_BF16": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_bf16"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _fused_gemm_a8w8_blockscale_a16w16_kernel(
    # Pointers to matrices
    a_fp8_ptr,
    b_fp8_ptr,
    bias_fp8_ptr,
    a_fp8_scale_ptr,
    b_fp8_scale_ptr,
    c_fp8_ptr,
    a_bf16_ptr,
    b_bf16_ptr,
    bias_bf16_ptr,
    c_bf16_ptr,
    # Matrix dimensions
    M,
    N_fp8,
    N_bf16,
    K,
    stride_a_fp8_m,
    stride_a_fp8_k,
    stride_b_fp8_k,
    stride_b_fp8_n,
    stride_a_fp8_scale_m,
    stride_a_fp8_scale_k,
    stride_b_fp8_scale_k,
    stride_b_fp8_scale_n,
    stride_c_fp8_k,
    stride_c_fp8_m,
    stride_c_fp8_n,
    stride_a_bf16_m,
    stride_a_bf16_k,
    stride_b_bf16_k,
    stride_b_bf16_n,
    stride_c_bf16_k,
    stride_c_bf16_m,
    stride_c_bf16_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    ADD_BIAS_FP8: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN_FP8: tl.constexpr,
    GRID_MN_BF16: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
    cache_modifier: tl.constexpr,
):

    tl.assume(stride_a_fp8_m > 0)
    tl.assume(stride_a_fp8_k > 0)
    tl.assume(stride_b_fp8_k > 0)
    tl.assume(stride_b_fp8_n > 0)
    tl.assume(stride_c_fp8_k > 0)
    tl.assume(stride_c_fp8_m > 0)
    tl.assume(stride_c_fp8_n > 0)
    tl.assume(stride_a_fp8_scale_m > 0)
    tl.assume(stride_a_fp8_scale_k > 0)
    tl.assume(stride_b_fp8_scale_k > 0)
    tl.assume(stride_b_fp8_scale_n > 0)

    tl.assume(stride_a_bf16_m > 0)
    tl.assume(stride_a_bf16_k > 0)
    tl.assume(stride_b_bf16_k > 0)
    tl.assume(stride_b_bf16_n > 0)
    tl.assume(stride_c_bf16_m > 0)
    tl.assume(stride_c_bf16_n > 0)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_fp8 = tl.cdiv(N_fp8, BLOCK_SIZE_N)
    num_pid_n_bf16 = tl.cdiv(N_bf16, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_fp8 + num_pid_n_bf16

    if NUM_KSPLIT == 1:
        GRID_MN: tl.constexpr = GRID_MN_FP8 + GRID_MN_BF16
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        acc_dtype = tl.float32 if c_fp8_ptr.type.element_ty != tl.int8 else tl.int32

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        if pid_n < num_pid_n_fp8:
            offs_b_fp8_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp8
            a_fp8_ptrs = a_fp8_ptr + (
                offs_am[:, None] * stride_a_fp8_m
                + offs_k_split[None, :] * stride_a_fp8_k
            )
            b_fp8_ptrs = b_fp8_ptr + (
                offs_k_split[:, None] * stride_b_fp8_k
                + offs_b_fp8_n[None, :] * stride_b_fp8_n
            )

            offs_ks = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
            a_scale_ptrs = (
                a_fp8_scale_ptr
                + offs_am * stride_a_fp8_scale_m
                + offs_ks * stride_a_fp8_scale_k
            )
            offs_bsn = offs_b_fp8_n // GROUP_N
            b_scale_ptrs = (
                b_fp8_scale_ptr
                + offs_ks * stride_b_fp8_scale_k
                + offs_bsn * stride_b_fp8_scale_n
            )

            if ADD_BIAS_FP8:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_fp8 = tl.load(bias_fp8_ptr + offs_b_fp8_n).to(
                        dtype=acc_dtype
                    )
                    accumulator_fp8 = tl.broadcast_to(
                        accumulator_fp8[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_fp8 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                    )
            else:
                accumulator_fp8 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_fp8_ptrs)
                    b = tl.load(b_fp8_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_fp8_ptrs,
                        mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_fp8_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                a_scale = tl.load(a_scale_ptrs)
                b_scale = tl.load(b_scale_ptrs)

                accumulator_fp8 += (
                    tl.dot(a, b, input_precision="ieee")
                    * a_scale[:, None]
                    * b_scale[None, :]
                )

                a_fp8_ptrs += BLOCK_SIZE_K * stride_a_fp8_k
                b_fp8_ptrs += BLOCK_SIZE_K * stride_b_fp8_k
                a_scale_ptrs += offs_ks_step * stride_a_fp8_scale_k
                b_scale_ptrs += offs_ks_step * stride_b_fp8_scale_k

            c_fp8 = accumulator_fp8.to(c_fp8_ptr.type.element_ty)

            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_fp8_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_fp8_ptrs = (
                c_fp8_ptr
                + stride_c_fp8_m * offs_cm[:, None]
                + stride_c_fp8_n * offs_c_fp8_n[None, :]
                + pid_k * stride_c_fp8_k
            )
            c_fp8_mask = (offs_cm[:, None] < M) & (offs_c_fp8_n[None, :] < N_fp8)
            tl.store(c_fp8_ptrs, c_fp8, mask=c_fp8_mask)
        else:
            pid_n -= num_pid_n_fp8

            offs_b_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
            a_ptrs = a_bf16_ptr + (
                offs_am[:, None] * stride_a_bf16_m
                + offs_k_split[None, :] * stride_a_bf16_k
            )
            b_ptrs = b_bf16_ptr + (
                offs_k_split[:, None] * stride_b_bf16_k
                + offs_b_bf16_n[None, :] * stride_b_bf16_n
            )

            if ADD_BIAS_BF16:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_bf16 = tl.load(bias_bf16_ptr + offs_b_bf16_n).to(
                        dtype=acc_dtype
                    )
                    accumulator_bf16 = tl.broadcast_to(
                        accumulator_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_bf16 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                    )
            else:
                accumulator_bf16 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                accumulator_bf16 += tl.dot(a, b, input_precision="ieee")

                a_ptrs += BLOCK_SIZE_K * stride_a_bf16_k
                b_ptrs += BLOCK_SIZE_K * stride_b_bf16_k

            c_bf16 = accumulator_bf16.to(c_bf16_ptr.type.element_ty)

            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_bf16_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_bf16_ptrs = (
                c_bf16_ptr
                + stride_c_bf16_m * offs_cm[:, None]
                + stride_c_bf16_n * offs_c_bf16_n[None, :]
                + pid_k * stride_c_bf16_k
            )
            c_bf16_mask = (offs_cm[:, None] < M) & (offs_c_bf16_n[None, :] < N_bf16)
            tl.store(c_bf16_ptrs, c_bf16, mask=c_bf16_mask)


@triton.jit
def _fused_gemm_a8w8_blockscale_a16w16_reduce_kernel(
    bias_fp8_ptr,
    c_fp8_in_ptr,
    c_fp8_out_ptr,
    bias_bf16_ptr,
    c_bf16_in_ptr,
    c_bf16_out_ptr,
    M,
    N_fp8,
    N_bf16,
    stride_c_fp8_in_k,
    stride_c_fp8_in_m,
    stride_c_fp8_in_n,
    stride_c_fp8_out_m,
    stride_c_fp8_out_n,
    stride_c_bf16_in_k,
    stride_c_bf16_in_m,
    stride_c_bf16_in_n,
    stride_c_bf16_out_m,
    stride_c_bf16_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS_FP8: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
):

    tl.assume(stride_c_fp8_in_k > 0)
    tl.assume(stride_c_fp8_in_m > 0)
    tl.assume(stride_c_fp8_in_n > 0)
    tl.assume(stride_c_fp8_out_m > 0)
    tl.assume(stride_c_fp8_out_n > 0)

    tl.assume(stride_c_bf16_in_k > 0)
    tl.assume(stride_c_bf16_in_m > 0)
    tl.assume(stride_c_bf16_in_n > 0)
    tl.assume(stride_c_bf16_out_m > 0)
    tl.assume(stride_c_bf16_out_n > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    num_pid_n_fp8 = tl.cdiv(N_fp8, BLOCK_SIZE_N)
    offs_k = tl.arange(0, MAX_KSPLIT)
    acc_dtype = tl.float32 if c_fp8_in_ptr.type.element_ty != tl.int8 else tl.int32

    if pid_n < num_pid_n_fp8:
        offs_fp8_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp8
        c_fp8_in_ptrs = (
            c_fp8_in_ptr
            + (offs_k[:, None, None] * stride_c_fp8_in_k)
            + (offs_m[None, :, None] * stride_c_fp8_in_m)
            + (offs_fp8_n[None, None, :] * stride_c_fp8_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_fp8 = tl.load(c_fp8_in_ptrs)
        else:
            c_fp8 = tl.load(
                c_fp8_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_fp8 = tl.sum(c_fp8, axis=0)
        if ADD_BIAS_FP8:
            bias_fp8 = tl.load(bias_fp8_ptr + offs_fp8_n).to(dtype=acc_dtype)
            bias_fp8 = tl.broadcast_to(bias_fp8[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
            c_fp8 += bias_fp8

        c_fp8 = c_fp8.to(c_fp8_out_ptr.type.element_ty)

        c_fp8_out_ptrs = (
            c_fp8_out_ptr
            + (offs_m[:, None] * stride_c_fp8_out_m)
            + (offs_fp8_n[None, :] * stride_c_fp8_out_n)
        )

        tl.store(c_fp8_out_ptrs, c_fp8)
    else:
        pid_n -= num_pid_n_fp8

        offs_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
        c_bf16_in_ptrs = (
            c_bf16_in_ptr
            + (offs_k[:, None, None] * stride_c_bf16_in_k)
            + (offs_m[None, :, None] * stride_c_bf16_in_m)
            + (offs_bf16_n[None, None, :] * stride_c_bf16_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_bf16 = tl.load(c_bf16_in_ptrs)
        else:
            c_bf16 = tl.load(
                c_bf16_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_bf16 = tl.sum(c_bf16, axis=0)
        if ADD_BIAS_BF16:
            bias_bf16 = tl.load(bias_bf16_ptr + offs_bf16_n).to(dtype=acc_dtype)
            bias_bf16 = tl.broadcast_to(
                bias_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
            )
            c_bf16 += bias_bf16

        c_bf16 = c_bf16.to(c_bf16_out_ptr.type.element_ty)

        c_bf16_out_ptrs = (
            c_bf16_out_ptr
            + (offs_m[:, None] * stride_c_bf16_out_m)
            + (offs_bf16_n[None, :] * stride_c_bf16_out_n)
        )
        c_bf16_mask = (offs_m[:, None] < M) & (offs_bf16_n[None, :] < N_bf16)
        tl.store(c_bf16_out_ptrs, c_bf16, mask=c_bf16_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N_fp8: int,
    N_bf16: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N_fp8}_{N_bf16}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8={N_fp8}-N16={N_bf16}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    if M < 16 and "small" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small"]
    elif M < 32 and "small_M16" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small_M16"]
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


def fused_gemm_a8w8_blockscale_a16w16(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_fp8_scale: torch.Tensor,
    w_fp8_scale: torch.Tensor,
    x_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
    bias_fp8: Optional[torch.Tensor] = None,
    bias_bf16: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y_fp8: Optional[torch.Tensor] = None,
    y_bf16: Optional[torch.Tensor] = None,
    skip_reduce: Optional[bool] = False,
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
        f"FUSED_GEMM_A8W8_BLOCKSCALE_A16W16: x_fp8={tuple(x_fp8.shape)} w_fp8={tuple(w_fp8.shape)} x_fp8_scale={tuple(x_fp8_scale.shape)} w_scale={tuple(w_fp8_scale.shape)} x_bf16={tuple(x_bf16.shape)} w_bf16={tuple(w_bf16.shape)}"
    )

    M, K = x_fp8.shape
    N_fp8, K = w_fp8.shape
    M, K = x_bf16.shape
    N_bf16, K = w_bf16.shape

    # Check constraints.
    assert (
        x_fp8.shape[0] == x_bf16.shape[0]
    ), "M-dim should be identical for x_fp8 and x_bf16"
    assert (
        x_fp8.shape[1] == x_bf16.shape[1]
    ), "K-dim should be identical for x_fp8 and x_bf16"
    assert x_fp8.shape[1] == w_fp8.shape[1], "Incompatible dimensions!!!"
    assert w_bf16.shape[1] == w_bf16.shape[1], "Incompatible dimensions!!!"

    # Transpose w and w_scale
    w_fp8 = w_fp8.T
    w_bf16 = w_bf16.T
    w_fp8_scale = w_fp8_scale.T

    if config is None:
        config = _get_config(M, N_fp8, N_bf16, K)

    if y_fp8 is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y_fp8 = torch.empty((M, N_fp8), dtype=dtype, device=x_fp8.device)

    if y_bf16 is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y_bf16 = torch.empty((M, N_bf16), dtype=dtype, device=x_bf16.device)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["NUM_KSPLIT"] > 1:
        y_fp8_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_fp8),
            dtype=torch.float32,
            device=y_fp8.device if y_fp8 is not None else x_fp8.device,
        )
        y_bf16_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_bf16),
            dtype=torch.float32,
            device=y_bf16.device if y_bf16 is not None else x_bf16.device,
        )
    else:
        y_fp8_pp = None
        y_bf16_pp = None

    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_fp8_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N_fp8, w_fp8_scale.shape[1]))

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * (
                triton.cdiv(N_fp8, META["BLOCK_SIZE_N"])
                + triton.cdiv(N_bf16, META["BLOCK_SIZE_N"])
            )
        ),
    )
    _fused_gemm_a8w8_blockscale_a16w16_kernel[grid](
        x_fp8,
        w_fp8,
        bias_fp8,
        x_fp8_scale,
        w_fp8_scale,
        y_fp8 if config["NUM_KSPLIT"] == 1 else y_fp8_pp,
        x_bf16,
        w_bf16,
        bias_bf16,
        y_bf16 if config["NUM_KSPLIT"] == 1 else y_bf16_pp,
        M,
        N_fp8,
        N_bf16,
        K,
        x_fp8.stride(0),
        x_fp8.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        x_fp8_scale.stride(0),
        x_fp8_scale.stride(1),
        w_fp8_scale.stride(0),
        w_fp8_scale.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(0),
        y_fp8.stride(0) if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(1),
        y_fp8.stride(1) if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(2),
        x_bf16.stride(0),
        x_bf16.stride(1),
        w_bf16.stride(0),
        w_bf16.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(0),
        y_bf16.stride(0) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(1),
        y_bf16.stride(1) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(2),
        ADD_BIAS_FP8=(bias_fp8 is not None),
        ADD_BIAS_BF16=(bias_bf16 is not None),
        SKIP_REDUCE=skip_reduce,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_fp8_pp, y_bf16_pp
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N_fp8, REDUCE_BLOCK_SIZE_N)
            + triton.cdiv(N_bf16, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_a8w8_blockscale_a16w16_reduce_kernel[grid_reduce](
            bias_fp8,
            y_fp8_pp,
            y_fp8,
            bias_bf16,
            y_bf16_pp,
            y_bf16,
            M,
            N_fp8,
            N_bf16,
            y_fp8_pp.stride(0),
            y_fp8_pp.stride(1),
            y_fp8_pp.stride(2),
            y_fp8.stride(0),
            y_fp8.stride(1),
            y_bf16_pp.stride(0),
            y_bf16_pp.stride(1),
            y_bf16_pp.stride(2),
            y_bf16.stride(0),
            y_bf16.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS_FP8=(bias_fp8 is not None),
            ADD_BIAS_BF16=(bias_bf16 is not None),
        )

    return y_fp8, y_bf16
