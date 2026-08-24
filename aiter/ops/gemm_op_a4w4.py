# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import pandas as pd
import torch
from torch import Tensor

from aiter import logger
from aiter.jit.utils.torch_guard import torch_compile_guard

from ..jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG, compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..ops.gemm_op_common import get_padded_m
from ..utility import dtypes


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
    tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
    if not hasattr(get_GEMM_config, "gemm_dict"):
        gemm_dict = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
        ).drop_duplicates()
        # Use (gfx, cu_num, M, N, K) key when the CSV has a gfx column (new schema).
        # Fall back to (cu_num, M, N, K) for old CSVs that pre-date the gfx column.
        if "gfx" in gemm_dict.columns:
            get_GEMM_config.gemm_dict = gemm_dict.set_index(
                ["gfx", "cu_num", "M", "N", "K"]
            ).to_dict("index")
            get_GEMM_config.has_gfx = True
        else:
            logger.warning(
                f"{AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE} has no 'gfx' column -- "
                "falling back to cu_num-only key. Re-run the tuner or migrate the CSV."
            )
            get_GEMM_config.gemm_dict = gemm_dict.set_index(
                ["cu_num", "M", "N", "K"]
            ).to_dict("index")
            get_GEMM_config.has_gfx = False
    gfx = get_gfx()
    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        key = (
            (gfx, cu_num, padded_M, N, K)
            if get_GEMM_config.has_gfx
            else (cu_num, padded_M, N, K)
        )
        config = get_GEMM_config.gemm_dict.get(key, None)
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE}, kernel name is {config['kernelName']}, splitK is {config['splitK']}!"
                )
            break
    else:
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K}, not found tuned config in {tuned_file}, will use default config!"
        )
    return config


def gemm_a4w4_fake(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/block_size] MXFP4: block_size=32 e8m0 padded, NVFP4: block_size=16 e4m3 padded
    B_scale: Tensor,  # B_scale:[N, K/block_size] MXFP4: block_size=32 e8m0 padded, NVFP4: block_size=16 e4m3 padded
    bias: Tensor | None = None,  # bias:[1, N] f32
    dtype: torch.dtype = dtypes.bf16,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
) -> torch.Tensor:
    if dtype in (dtypes.fp4x2, dtypes.fp8):
        raise NotImplementedError(
            "gemm_a4w4 only supports plain-dtype output (e.g. bf16/fp16/fp32); "
            "low-precision fp4/mxfp8 output is not available through this API"
        )
    n = B.shape[0]
    return torch.empty((*A.shape[:-1], n), dtype=dtype, device=A.device)


@torch_compile_guard(gen_fake=gemm_a4w4_fake)
def gemm_a4w4(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/block_size] MXFP4: block_size=32 e8m0 padded, NVFP4: block_size=16 e4m3 padded
    B_scale: Tensor,  # B_scale:[N, K/block_size] MXFP4: block_size=32 e8m0 padded, NVFP4: block_size=16 e4m3 padded
    bias: Tensor | None = None,  # bias:[1, N] f32
    dtype: torch.dtype = dtypes.bf16,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
) -> torch.Tensor:
    """A4W4 GEMM (4-bit quantized matmul) returning one plain-dtype tensor."""
    # Low-precision output has a different return arity/shape (fp4 is [M,N//2],
    # mxfp8 is a (data, scale) tuple), so it lives on separate fixed-arity ops.
    if dtype in (dtypes.fp4x2, dtypes.fp8):
        raise NotImplementedError(
            "gemm_a4w4 only supports plain-dtype output (e.g. bf16/fp16/fp32); "
            "low-precision fp4/mxfp8 output is not available through this API"
        )
    m = A.numel() // A.shape[-1]
    n = B.shape[0]
    k = A.shape[-1] * 2
    gfx_arch = get_gfx()
    out = torch.empty(((m + 31) // 32 * 32, n), dtype=dtype, device=A.device)
    if gfx_arch in ["gfx942"]:
        raise RuntimeError(
            f"A4W4 GEMM kernel is not supported on gfx942, but got {gfx_arch}!"
        )
    ck_config = get_GEMM_config(m, n, k)
    # splitK = None
    splitK = 0
    kernelName = ""
    if ck_config is not None:
        splitK = ck_config.get("splitK", None)
        kernelName = ck_config["kernelName"]
    if (
        ck_config is not None
        and kernelName.find("_ZN") == -1
        # or bias is None
    ):
        splitK = 0 if splitK is None else splitK
        return gemm_a4w4_blockscale(
            A.view(m, k // 2),
            B,
            A_scale,
            B_scale,
            out,
            splitK=splitK,
            kernelName=kernelName,
        )[:m]
    assert (
        out.shape[0] % 32 == 0
    ), "Dim0 of gemm_a4w4_asm output needs to be padded to multiples of 32!"
    gemm_a4w4_asm(
        A.view(m, k // 2),
        B,
        A_scale,
        B_scale,
        out,
        kernelName,
        bias,
        alpha,
        beta,
        bpreshuffle,
        log2_k_split=splitK,
    )
    return out[:m].view(*A.shape[:-1], n)


@compile_ops(
    "module_gemm_a4w4_asm",
    fc_name="gemm_a4w4_asm",
    ffi_type="ctypes",
)
def _gemm_a4w4_asm(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/32] e8m0 paded
    B_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
    kernelName: str | None = None,
    bias: Tensor | None = None,  # bias:[1, N] f32
    alpha: float = 1.0,
    beta: float = 0.0,
    bpreshuffle: int = 1,
    log2_k_split: int = 0,
) -> None: ...


def gemm_a4w4_asm(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/32] e8m0 paded
    B_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
    kernelName: str = "",
    bias: Tensor | None = None,  # bias:[1, N] f32
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
    log2_k_split: int | None = None,
) -> Tensor:
    _gemm_a4w4_asm(
        A,
        B,
        A_scale,
        B_scale,
        out,
        kernelName if kernelName else None,
        bias,
        alpha if alpha is not None else 1.0,
        beta if beta is not None else 0.0,
        int(bpreshuffle) if bpreshuffle is not None else 1,
        log2_k_split if log2_k_split is not None else 0,
    )
    return out


def gen_gemm_a4w4_blockscale_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a4w4_blockscale", gen_fake=gen_gemm_a4w4_blockscale_fake_tensors
)
def gemm_a4w4_blockscale(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
    kernelName: str = "",
) -> Tensor: ...


@compile_ops(
    "module_gemm_a4w4_blockscale_tune",
    fc_name="gemm_a4w4_blockscale_tune",
    gen_fake=gen_gemm_a4w4_blockscale_fake_tensors,
)
def gemm_a4w4_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor: ...
