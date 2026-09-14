# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared tuned dispatch for native MXFP8 and gfx950 E8M0 blockscale GEMM."""

import functools
import os

import pandas as pd
import torch

from aiter import logger
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx

MXFP8_KEYS = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "outdtype",
    "bias",
    "scale_block",
    "bpreshuffle",
    "scale_a_transposed",
]


def is_mxfp8_scale(scale):
    return scale is not None and scale.dtype in (torch.uint8, torch.float8_e8m0fnu)


@functools.lru_cache(maxsize=8)
def _load_mxfp8_configs(path):
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.set_index(MXFP8_KEYS).to_dict("index")


@functools.lru_cache(maxsize=4096)
def get_mxfp8_config(
    m,
    n,
    k,
    out_dtype,
    has_bias=False,
    scale_block=32,
    bpreshuffle=False,
    scale_a_transposed=False,
):
    from .flydsl.gemm_mxfp8 import (
        CONFIG_KEYS,
        DEFAULT_CONFIG,
        get_flydsl_mxfp8_configs,
        get_flydsl_mxfp8_kernel_params,
        mxfp8_kernel_config,
    )
    from .flydsl.kernels.scaled_gemm_gfx950 import make_scaled_gemm_param_and_validate

    gfx = get_gfx()
    if gfx != "gfx950":
        raise RuntimeError("FlyDSL MXFP8 GEMM requires gfx950")
    key = (
        gfx,
        get_cu_num(),
        m,
        n,
        k,
        str(out_dtype),
        has_bias,
        scale_block,
        bpreshuffle,
        scale_a_transposed,
    )
    row = _load_mxfp8_configs(AITER_CONFIGS.AITER_CONFIG_GEMM_MXFP8_FILE).get(key)
    if row is not None:
        params = get_flydsl_mxfp8_kernel_params(row["kernelName"], row.get("splitK", 1))
        if row["libtype"] == "flydsl" and params is not None:
            expected = {
                "out_dtype": "bf16" if out_dtype == torch.bfloat16 else "fp32",
                "has_bias": has_bias,
                "scale_block": scale_block,
                "bpreshuffle": bpreshuffle,
                "scale_a_transposed": scale_a_transposed,
                "target_gfx": gfx,
            }
            c = {key: params[key] for key in CONFIG_KEYS}
            if all(params[key] == value for key, value in expected.items()):
                p = mxfp8_kernel_config(
                    c, out_dtype, has_bias, scale_block, bpreshuffle, scale_a_transposed
                )
                if make_scaled_gemm_param_and_validate(m, n, k, p) is not None:
                    return c
        logger.warning("Invalid MXFP8 tuned entry %s; using validated default", row)
    p = mxfp8_kernel_config(
        DEFAULT_CONFIG,
        out_dtype,
        has_bias,
        scale_block,
        bpreshuffle,
        scale_a_transposed,
    )
    if make_scaled_gemm_param_and_validate(m, n, k, p) is not None:
        return dict(DEFAULT_CONFIG)
    configs = get_flydsl_mxfp8_configs(
        m, n, k, out_dtype, has_bias, scale_block, bpreshuffle, scale_a_transposed
    )
    if not configs:
        raise ValueError(f"No supported MXFP8 config for M={m}, N={n}, K={k}")
    return dict(configs[0])


def gemm_mxfp8(
    a,
    b,
    scale_a,
    scale_b,
    *,
    out=None,
    bias=None,
    dtype=torch.bfloat16,
    scale_block=32,
    bpreshuffle=False,
    scale_a_transposed=False,
):
    from .flydsl.gemm_mxfp8 import flydsl_mxfp8_gemm

    config = get_mxfp8_config(
        a.shape[0],
        b.shape[0],
        a.shape[1],
        dtype,
        bias is not None,
        scale_block,
        bpreshuffle,
        scale_a_transposed,
    )
    return flydsl_mxfp8_gemm(
        a,
        b,
        scale_a,
        scale_b,
        out=out,
        bias=bias,
        config=config,
        out_dtype=dtype,
        scale_block=scale_block,
        bpreshuffle=bpreshuffle,
        scale_a_transposed=scale_a_transposed,
    )
