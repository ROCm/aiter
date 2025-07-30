# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from aiter import logger
from ..jit.core import (
    compile_ops,
    AITER_ROOT_DIR,
)
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx
import functools
import pandas as pd
from ..ops.gemm_op_common import get_padded_m


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    ## to make sure the precision is not lost, max is 4
    # return min(splitK, 4)
    return 3


@functools.lru_cache(maxsize=1024)
def get_GEMM_config(M: int, N: int, K: int):
    if not hasattr(get_GEMM_config, "gemm_dict"):
        gemm_dict = pd.read_csv(
            f"{AITER_ROOT_DIR}/aiter/configs/a4w4_blockscale_tuned_gemm.csv"
        ).drop_duplicates()
        get_GEMM_config.gemm_dict = gemm_dict.set_index(
            ["cu_num", "M", "N", "K"]
        ).to_dict("index")
    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = get_GEMM_config.gemm_dict.get((cu_num, padded_M, N, K), None)
        if config is not None:
            logger.info(
                f"shape is M:{M}, N:{N}, K:{K}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in CKGEMM or asmGEMM, kernel name is {config['kernelName']}!"
            )
            break

    return config


def gemm_a4w4(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/32] e8m0 paded
    B_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
    bias: Optional[Tensor] = None,  # bias:[1, N] f32
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    bpreshuffle: Optional[bool] = True,
) -> torch.Tensor:
    """
    A4W4 GEMM kernel for AMD GPUs.
    This function is a wrapper for the A4W4 GEMM kernel.
    It is used to perform matrix multiplication with 4-bit quantization.
    """
    # Load the A4W4 GEMM kernel
    m = A.shape[0]
    n = B.shape[0]
    k = A.shape[-1] * 2
    gfx_arch = get_gfx()
    if gfx_arch in ["gfx942"]:
        raise RuntimeError(
            f"A4W4 GEMM kernel is not supported on gfx942, but got {gfx_arch}!"
        )
    ck_config = get_GEMM_config(m, n, k)
    splitK = 0
    kernelName = ""
    if ck_config is not None:
        splitK = ck_config["splitK"]
        kernelName = ck_config["kernelName"]
    if (
        m < 256
        or (ck_config is not None and kernelName.find("_ZN") == -1)
        # or bias is None
    ):
        return gemm_a4w4_blockscale(A, B, A_scale, B_scale, out, splitK=splitK)
    return gemm_a4w4_asm(
        A,
        B,
        A_scale,
        B_scale,
        out,
        kernelName,
        bias,
        alpha,
        beta,
        bpreshuffle,
        log2_k_split=0,
    )


@compile_ops("module_gemm_a4w4_asm")
def gemm_a4w4_asm(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/32] e8m0 paded
    B_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
    kernelName: str,
    bias: Optional[Tensor] = None,  # bias:[1, N] f32
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    bpreshuffle: Optional[bool] = True,
    log2_k_split: Optional[int] = None,
) -> torch.Tensor: ...


@compile_ops("module_gemm_a4w4_blockscale")
def gemm_a4w4_blockscale(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops("module_gemm_a4w4_blockscale_tune", fc_name="gemm_a4w4_blockscale_tune")
def gemm_a4w4_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int,
    splitK: int = 0,
) -> torch.Tensor: ...
