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
)
from ..utility import dtypes
from ..jit.utils.chip_info import get_cu_num


@compile_ops("module_batched_gemm_bf16", fc_name="batched_gemm_bf16")
def batched_gemm_bf16(
    XQ: Tensor, WQ: Tensor, out: Tensor, bias: Optional[Tensor] = None, splitK=0
): ...


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
        ck_batched_gemm_dict = pd.read_csv(
            f"{AITER_ROOT_DIR}/aiter/configs/bf16_tuned_batched_gemm.csv"
        ).drop_duplicates()
        get_CKBatchedGEMM_config.ck_batched_gemm_dict = ck_batched_gemm_dict.set_index(
            ["B", "M", "N", "K"]
        ).to_dict("index")
    config = get_CKBatchedGEMM_config.ck_batched_gemm_dict.get((B, M, N, K), None)
    if config != None:
        mnk = config["kernelName"].split("_")[2].split("x")[1:]
        config["tile_m"] = int(mnk[0])
        config["tile_n"] = int(mnk[1])
        config["tile_k"] = int(mnk[2])
    return config


def batched_gemm_bf16_CK(
    XQ: Tensor,
    WQ: Tensor,
    bias: Optional[Tensor] = None,
    dtype=dtypes.bf16,
    splitK: Optional[int] = None,
):
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_bf16"

    b = XQ.shape[0]
    m = XQ.shape[1]
    n = WQ.shape[1]
    k = XQ.shape[2]
    ck_config = get_CKBatchedGEMM_config(b, m, n, k)
    if splitK == None:
        if ck_config != None:
            splitK = ck_config["splitK"]
        else:
            splitK = 0
    Y = torch.empty(b, m, n, dtype=dtype, device=XQ.device)
    return batched_gemm_bf16(XQ, WQ, Y, bias, splitK)


@compile_ops("module_batched_gemm_bf16_tune", fc_name="batched_gemm_bf16_tune")
def batched_gemm_bf16_tune(
    XQ: Tensor, WQ: Tensor, out: Tensor, kernelId: int, splitK=0
): ...
