# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatcher for the FlyDSL a8w8 blockscale bpreshuffle batched GEMM (BMM)."""

from __future__ import annotations

import functools

import pandas as pd
import torch
from torch import Tensor

from ..jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx
from ..jit.utils.torch_guard import torch_compile_guard
from ..utility import dtypes
from aiter import logger


@functools.lru_cache(maxsize=1024)
def get_bmm_a8w8_blockscale_bpreshuffle_config(
    B: int,
    M: int,
    N: int,
    K: int,
):
    """Look up the tuned config row for shape ``(B, M, N, K)`` (cached).

    Returns the CSV row as a dict (with ``kernelName``), or ``None`` when the
    shape is untuned.
    """
    fn = get_bmm_a8w8_blockscale_bpreshuffle_config
    tuned_file = AITER_CONFIGS.AITER_CONFIG_A8W8_BLOCKSCALE_BPRESHUFFLE_BMM_FILE
    if not hasattr(fn, "bmm_dict"):
        print("Loading FlyDSL blockscale bpreshuffle BMM config from:", tuned_file)
        bmm_df = pd.read_csv(tuned_file).drop_duplicates()
        # Use (gfx, cu_num, B, M, N, K) key when the CSV has a gfx column (new
        # schema). Fall back to (cu_num, B, M, N, K) for old CSVs that pre-date
        # the gfx column.
        if "gfx" in bmm_df.columns:
            fn.bmm_dict = bmm_df.set_index(
                ["gfx", "cu_num", "B", "M", "N", "K"]
            ).to_dict("index")
            fn.has_gfx = True
        else:
            logger.warning(
                f"{tuned_file} has no 'gfx' column — falling back to cu_num-only "
                "key. Re-run the tuner or migrate the CSV."
            )
            fn.bmm_dict = bmm_df.set_index(
                ["cu_num", "B", "M", "N", "K"]
            ).to_dict("index")
            fn.has_gfx = False

    gfx = get_gfx()
    cu_num = get_cu_num()
    key = (gfx, cu_num, B, M, N, K) if fn.has_gfx else (cu_num, B, M, N, K)
    config = fn.bmm_dict.get(key, None)
    if config is not None:
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"shape is B:{B}, M:{M}, N:{N}, K:{K}, is tuned on cu_num = {cu_num} "
                f"in {tuned_file}, kernel name is {config['kernelName']}, "
                f"splitK is {config.get('splitK')}!"
            )
    else:
        logger.info(
            f"shape is B:{B}, M:{M}, N:{N}, K:{K}, not found tuned config in "
            "blockscale bpreshuffle BMM, will use default config!"
        )
    return config


@functools.lru_cache(maxsize=1024)
def _default_bmm_kernel_name(M: int, N: int, K: int) -> str:
    """Pick a default kernelName for an untuned shape.

    Falls back to the first candidate in the bmm_common list that fits the
    runtime ``(M, N, K)`` (its ``.name`` encodes the tile config the gfx1250
    backend decodes). Cached per shape.
    """
    from .flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_bmm_common import (
        kernel_fits_shape,
        kernels_list,
    )

    for ki in kernels_list.values():
        if kernel_fits_shape(ki, M, N, K):
            return ki.name
    raise RuntimeError(
        f"[bmm_a8w8_blockscale_bpreshuffle] no candidate kernel fits shape "
        f"M={M}, N={N}, K={K}"
    )


def _bmm_out_shape(XQ: Tensor, WQ: Tensor, layout: str):
    """(B, M, N) output shape, ordered per ``layout``."""
    if layout == "bmn":
        B, M, _ = XQ.shape
        return (B, M, WQ.shape[1])
    if layout == "mbn":
        M, B, _ = XQ.shape
        return (M, B, WQ.shape[1])
    raise ValueError(f"layout must be 'bmn' or 'mbn', got {layout!r}")


def bmm_a8w8_blockscale_bpreshuffle_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    layout: str = "bmn",
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    return torch.empty(
        _bmm_out_shape(XQ, WQ, layout), dtype=dtype, device=XQ.device
    )


@torch_compile_guard(gen_fake=bmm_a8w8_blockscale_bpreshuffle_fake)
def bmm_a8w8_blockscale_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    layout: str = "bmn",
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    """Run the tuned FlyDSL a8w8 blockscale bpreshuffle BMM.

    ``XQ`` / ``WQ`` are fp8_e4m3fn (1-byte); ``x_scale`` / ``w_scale`` are uint8
    E8M0 block scales. Shapes follow ``layout`` ("bmn" -> A=[B, M, K], C=[B, M, N];
    "mbn" -> A=[M, B, K], C=[M, B, N]); B is always [B, N, K].
    """
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in bmm_a8w8_blockscale_bpreshuffle"

    if layout == "bmn":
        B, M, K = XQ.shape
    elif layout == "mbn":
        M, B, K = XQ.shape
    else:
        raise ValueError(f"layout must be 'bmn' or 'mbn', got {layout!r}")
    N = WQ.shape[1]

    config = get_bmm_a8w8_blockscale_bpreshuffle_config(B, M, N, K)
    if config is not None:
        kernel_name = str(config["kernelName"])
    else:
        # Untuned shape: fall back to the first fitting bmm_common candidate.
        kernel_name = _default_bmm_kernel_name(M, N, K)

    Y = torch.empty(
        _bmm_out_shape(XQ, WQ, layout), dtype=dtype, device=XQ.device
    )

    if get_gfx() != "gfx1250":
        raise RuntimeError(
            "[bmm_a8w8_blockscale_bpreshuffle] only gfx1250 is supported, "
            f"got {get_gfx()}"
        )
    from .flydsl.bmm_w8a8_gfx1250 import run_bmm_a8w8_blockscale_bmm_gfx1250

    return run_bmm_a8w8_blockscale_bmm_gfx1250(
        XQ.contiguous(), WQ, x_scale, w_scale, Y, layout, kernel_name
    )
