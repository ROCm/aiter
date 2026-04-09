# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
import functools
import pandas as pd
from ..jit.core import (
    compile_ops,
    AITER_ROOT_DIR,
    AITER_CONFIGS,
    AITER_LOG_TUNED_CONFIG,
)
from ..utility import dtypes
from ..jit.utils.chip_info import get_cu_num
from aiter import logger


def gen_batched_gemm_a8w8_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    splitK: int = 0,
) -> Tensor:
    return out


@compile_ops(
    "module_batched_gemm_a8w8",
    fc_name="batched_gemm_a8w8",
    gen_fake=gen_batched_gemm_a8w8_fake_tensors,
)
def batched_gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    splitK: int = 0,
) -> Tensor: ...


@functools.lru_cache(maxsize=1024)
def compute_batched_gemm_SplitK(
    M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int
):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    return splitK


@functools.lru_cache(maxsize=1024)
def get_CKBatchedGEMM_config(
    B: int,
    M: int,
    N: int,
    K: int,
):
    if not hasattr(get_CKBatchedGEMM_config, "ck_batched_gemm_dict"):
        print(
            "Loading CKBatchedGEMM config from:",
            AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE,
        )
        ck_batched_gemm_dict = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE
        ).drop_duplicates()

        get_CKBatchedGEMM_config.ck_batched_gemm_dict = ck_batched_gemm_dict.set_index(
            ["cu_num", "B", "M", "N", "K"]
        ).to_dict("index")
    cu_num = get_cu_num()
    config = get_CKBatchedGEMM_config.ck_batched_gemm_dict.get(
        (cu_num, B, M, N, K), None
    )
    if config is not None:
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"shape is B:{B}, M:{M}, N:{N}, K:{K}, is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE}, kernel name is {config['kernelName']}, splitK is {config['splitK']}!"
            )
        mnk = config["kernelName"].split("_")[3].split("x")[1:]
        config["tile_m"] = int(mnk[0])
        config["tile_n"] = int(mnk[1])
        config["tile_k"] = int(mnk[2])
    else:
        logger.info(
            f"shape is B:{B}, M:{M}, N:{N}, K:{K}, not found tuned config in CKGEMM, will use default config!"
        )
    return config


def batched_gemm_a8w8_CK(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype=dtypes.bf16,
    splitK: Optional[int] = None,
):
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"

    b = XQ.shape[0]
    m = XQ.shape[1]
    n = WQ.shape[1]
    k = XQ.shape[2]
    ck_config = get_CKBatchedGEMM_config(b, m, n, k)
    if splitK is None:
        if ck_config is not None:
            splitK = ck_config["splitK"]
        else:
            splitK = 0
    Y = torch.empty(b, m, n, dtype=dtype, device=XQ.device)
    return batched_gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias, splitK)


@compile_ops(
    "module_mxfp8_batch_gemm_asm",
    fc_name="mxfp8_batch_gemm_asm",
    ffi_type="ctypes",
)
def _mxfp8_batch_gemm_asm(
    A: Tensor,              # A:[B, M, K] fp8 (preshuffled)
    B: Tensor,              # B:[B, N, K] fp8 (preshuffled)
    ScaleA: Tensor,         # ScaleA:[B, M, K/32] uint8 e8m0 (shuffled)
    ScaleB: Tensor,         # ScaleB:[B, N, K/32] uint8 e8m0 (shuffled)
    Out: Tensor,            # Out:[B, M, N] bf16
    kernelName: Optional[str] = None,
) -> None: ...


def batched_gemm_a8w8_ASM(
    A: Tensor,
    B: Tensor,
    ScaleA: Tensor,
    ScaleB: Tensor,
    dtype=dtypes.bf16,
    kernelName: str = "",
):
    """MXFP8 batched GEMM via ASM kernel (gfx1250).

    Args:
        A: [B, M, K] fp8 input (preshuffled: m/2, k/128, 2, 128)
        B: [B, N, K] fp8 weight (preshuffled: n/16, k/16, 16, 16)
        ScaleA: [B, M, K/32] uint8 e8m0 block-wise scale (shuffled: m/32, k/4, 32, 4)
        ScaleB: [B, N, K/32] uint8 e8m0 block-wise scale (shuffled: n/32, k/4, 32, 4)
        dtype: output dtype, only bf16 supported
        kernelName: optional kernel name to force a specific kernel

    Returns:
        Out: [B, M, N] bf16 result
    """
    assert dtype in [
        dtypes.bf16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8_ASM"

    b = A.shape[0]
    m = A.shape[1]
    n = B.shape[1]

    Y = torch.empty(b, m, n, dtype=dtype, device=A.device)
    _mxfp8_batch_gemm_asm(
        A, B, ScaleA, ScaleB, Y,
        kernelName if kernelName else None,
    )
    return Y    


def gen_batched_gemm_a8w8_tune_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor:
    return out


@compile_ops(
    "module_batched_gemm_a8w8_tune",
    fc_name="batched_gemm_a8w8_tune",
    gen_fake=gen_batched_gemm_a8w8_tune_fake_tensors,
)
def batched_gemm_a8w8_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor: ...
