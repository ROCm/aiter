# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import List, Optional
import functools
import pandas as pd
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_ROOT_DIR, AITER_CORE_DIR


@compile_ops("module_batched_gemm_a8w8", fc_name="batched_gemm_a8w8")
def batched_gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    splitK = 0
): ...


@functools.lru_cache(maxsize=1024)
def compute_batched_gemm_SplitK(
        M: int,
        N: int,
        K: int,
        tile_m: int,
        tile_n: int,
        tile_k: int):

    device_properties = torch.cuda.get_device_properties(0)
    cu_num = device_properties.multi_processor_count
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while( cusPerTile >= pow(2, splitK+1) and (pow(2, splitK+1) * tile_k) < 2 * K):
        splitK += 1
    return splitK


@functools.lru_cache(maxsize=1024)
def get_CKBatchedGEMM_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(get_CKBatchedGEMM_config, "ck_batched_gemm_dict"):
        ck_batched_gemm_dict = pd.read_csv(f"{AITER_CORE_DIR}/aiter/configs/a8w8_tuned_gemm.csv").drop_duplicates()
        get_CKBatchedGEMM_config.ck_batched_gemm_dict = ck_batched_gemm_dict.set_index(['M','N','K']).to_dict('index')
    config = get_CKBatchedGEMM_config.ck_batched_gemm_dict.get((M,N,K), None)
    if config != None:
        mnk = config['kernelName'].split('_')[2].split('x')[1:]
        config["tile_m"] = int(mnk[0])
        config["tile_n"] = int(mnk[1])
        config["tile_k"] = int(mnk[2])
    return config

def batched_gemm_a8w8_CK(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype=torch.bfloat16,
    splitK: Optional[int] = None
):
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    ck_config = get_CKBatchedGEMM_config(m, n, k)
    if splitK == None:
        if ck_config != None:
            splitK = ck_config['splitK']
        else:
            splitK = 0
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return batched_gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias, splitK)

@compile_ops("module_batched_gemm_a8w8_tune",fc_name="batched_gemm_a8w8_tune")
def batched_gemm_a8w8_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK = 0
): ...
