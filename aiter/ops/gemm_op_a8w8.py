# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
import functools
import pandas as pd
from ..jit.core import (
    compile_ops,
    AITER_CORE_DIR,
)
from ..utility import dtypes
from ..jit.utils.chip_info import get_cu_num


@compile_ops("module_gemm_a8w8", fc_name="gemm_a8w8")
def gemm_a8w8(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops("module_gemm_a8w8_asm", fc_name="gemm_a8w8_asm")
def gemm_a8w8_asm(
    XQ: Tensor,  # A:[M, K] i8
    WQ: Tensor,  # B:[N, K] i8 -> shuffle layout(32,16)
    x_scale: Tensor,  # A_scale:[M, 1] f32
    w_scale: Tensor,  # B_scale:[1, N] f32
    Out: Tensor,  # Out:[M, N] bf16
    bias: Tensor,  # bias:[1, N] f32
    sub_m: Optional[int] = 128,
    sub_n: Optional[int] = 128,
    pad_a: Optional[int] = 0,
    pad_b: Optional[int] = 0,
    pad_c: Optional[int] = 0,
    splitK: Optional[int] = 0,
) -> torch.Tensor: ...


@compile_ops("module_gemm_a8w8_blockscale", fc_name="gemm_a8w8_blockscale")
def gemm_a8w8_blockscale(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
): ...


@compile_ops("module_gemm_a8w8_blockscale_asm", fc_name="flatmm_a8w8_blockscale_asm")
def flatmm_a8w8_blockscale_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
): ...


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    return splitK


@functools.lru_cache(maxsize=1024)
def get_CKGEMM_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(get_CKGEMM_config, "ckgemm_dict"):
        ckgemm_dict = pd.read_csv(
            f"{AITER_CORE_DIR}/aiter/configs/a8w8_tuned_gemm.csv"
        ).drop_duplicates()
        get_CKGEMM_config.ckgemm_dict = ckgemm_dict.set_index(["M", "N", "K"]).to_dict(
            "index"
        )
    config = get_CKGEMM_config.ckgemm_dict.get((M, N, K), None)
    if config != None:
        mnk = config["kernelName"].split("_")[2].split("x")[1:]
        config["tile_m"] = int(mnk[0])
        config["tile_n"] = int(mnk[1])
        config["tile_k"] = int(mnk[2])
    return config


@functools.lru_cache(maxsize=1024)
def get_ASMGEMM_config(M: int, N: int, K: int, bias: bool, dtype: torch.dtype):
    if not hasattr(get_ASMGEMM_config, "asmgemm_dict"):
        asmGemmDictDf = pd.read_csv(
            f"{AITER_CORE_DIR}/aiter/configs/asm_a8w8_gemm.csv"
        ).drop_duplicates()
        asmGemmDictDf.bias = asmGemmDictDf.bias.apply(
            lambda s: True if s in ["True", 1, "true"] else False
        )
        get_ASMGEMM_config.asmgemm_dict = asmGemmDictDf.set_index(
            ["M", "N", "K", "bias", "outdtype"]
        ).to_dict("index")
    return get_ASMGEMM_config.asmgemm_dict.get((M, N, K, bias, str(dtype)), None)


def gemm_a8w8_ASM(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor,
    dtype=dtypes.bf16,
    check=False,
):
    """
    Notes for use gemm_a8w8_ASM:
    1. WQ(weight) must be shuffle, you can use \
        'weightshuffle = shuffle_weight(weight,layout=(32,16))'
    2. Use asm gemm must give bias, if not have bias, please give  \
        'bias=torch.zeros(n,dtype=dtypes.fp32,device='cuda')'
    """
    if check:
        assert dtype in [
            dtypes.bf16,
        ], f"Output {dtype=} is currently not supported in gemm_a8w8_ASM"
        assert (
            x_scale.dtype == dtypes.fp32 and w_scale.dtype == dtypes.fp32
        ), f"{x_scale.dtype=} or {w_scale.dtype=} must be dtypes.fp32"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    if (
        x_scale.dtype == dtypes.fp32
        and w_scale.dtype == dtypes.fp32
        and (asm_config := get_ASMGEMM_config(m, n, k, bias != None, dtype)) != None
    ):
        assert (
            bias != None
        ), "Use asm gemm must give bias, please give a \
            bias=torch.zeros(n,dtype=dtypes.fp32,device='cuda')"
        splitK = asm_config["splitK"]
        Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
        return gemm_a8w8_asm(XQ, WQ, x_scale, w_scale, Y, bias, splitK=splitK)
    return None


def gemm_a8w8_CK(
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
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    ck_config = get_CKGEMM_config(m, n, k)
    if splitK == None:
        if ck_config != None:
            splitK = ck_config["splitK"]
        else:
            splitK = 0
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias, splitK)


def gemm_a8w8_blockscale_CK(
    XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor, dtype=dtypes.bf16
):
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return gemm_a8w8_blockscale(XQ, WQ, x_scale, w_scale, Y)


def flatmm_a8w8_blockscale_ASM(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype=dtypes.fp16,
):
    assert dtype in [
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return flatmm_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, Y)


@compile_ops("module_gemm_a8w8_tune", fc_name="gemm_a8w8_tune")
def gemm_a8w8_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK=0,
): ...


@compile_ops("module_gemm_a8w8_blockscale_tune", fc_name="gemm_a8w8_blockscale_tune")
def gemm_a8w8_blockscale_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK=0,
): ...
