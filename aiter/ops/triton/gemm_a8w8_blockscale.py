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
from aiter.ops.triton._gluon_kernels.gemm_a8w8_blockscale import *

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
            f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/gfx950-GEMM-A8W8_BLOCKSCALE.json"
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
