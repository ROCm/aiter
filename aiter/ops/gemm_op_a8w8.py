# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
import functools
import pandas as pd
from aiter import logger
from ..jit.core import (
    compile_ops,
    AITER_ROOT_DIR,
    AITER_CONFIGS,
    AITER_LOG_TUNED_CONFIG,
)
from ..jit.utils.torch_guard import torch_compile_guard
from ..utility import dtypes
from ..jit.utils.chip_info import get_cu_num
from torch.library import Library
from ..ops.gemm_op_common import get_padded_m

aiter_lib = Library("aiter", "FRAGMENT")


def gen_gemm_a8w8_ck_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8", fc_name="gemm_a8w8", gen_fake=gen_gemm_a8w8_ck_fake_tensors
)
def gemm_a8w8_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    splitK: int = 0,
) -> torch.Tensor: ...


def gen_gemm_a8w8_bpreshuffle_ck_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_bpreshuffle",
    fc_name="gemm_a8w8_bpreshuffle",
    gen_fake=gen_gemm_a8w8_bpreshuffle_ck_fake_tensors,
)
def gemm_a8w8_bpreshuffle_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor: ...


def gen_gemm_a8w8_bpreshuffle_cktile_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_bpreshuffle_cktile",
    fc_name="gemm_a8w8_bpreshuffle_cktile",
    gen_fake=gen_gemm_a8w8_bpreshuffle_cktile_fake_tensors,
)
def gemm_a8w8_bpreshuffle_cktile(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> Tensor: ...


def gen_gemm_a8w8_asm_fake_tensors(
    XQ: Tensor,  # A:[M, K] i8
    WQ: Tensor,  # B:[N, K] i8 -> shuffle layout(32,16)
    x_scale: Tensor,  # A_scale:[M, 1] f32
    w_scale: Tensor,  # B_scale:[1, N] f32
    Out: Tensor,  # Out:[M, N] bf16
    kernelName: str,
    bias: Optional[Tensor],  # bias:[1, N] f32
    bpreshuffle: Optional[bool] = True,
    splitK: Optional[int] = None,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_asm",
    fc_name="gemm_a8w8_asm",
    gen_fake=gen_gemm_a8w8_asm_fake_tensors,
)
def gemm_a8w8_asm(
    XQ: Tensor,  # A:[M, K] i8
    WQ: Tensor,  # B:[N, K] i8 -> shuffle layout(32,16)
    x_scale: Tensor,  # A_scale:[M, 1] f32
    w_scale: Tensor,  # B_scale:[1, N] f32
    Out: Tensor,  # Out:[M, N] bf16
    kernelName: str,
    bias: Optional[Tensor],  # bias:[1, N] f32
    bpreshuffle: Optional[bool] = True,
    splitK: Optional[int] = None,
) -> torch.Tensor: ...


def gen_gemm_a8w8_blockscale_ck_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_blockscale",
    fc_name="gemm_a8w8_blockscale",
    gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
)
def gemm_a8w8_blockscale_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_cktile",
    fc_name="gemm_a8w8_blockscale_cktile",
    gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
)
def gemm_a8w8_blockscale_cktile(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle",
    fc_name="gemm_a8w8_blockscale_bpreshuffle",
    gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor: ...


def gen_flatmm_a8w8_blockscale_asm_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> Tensor:
    return out


@compile_ops(
    "module_gemm_a8w8_blockscale_asm",
    fc_name="flatmm_a8w8_blockscale_asm",
    gen_fake=gen_flatmm_a8w8_blockscale_asm_fake_tensors,
)
def flatmm_a8w8_blockscale_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> Tensor: ...


def gen_gemm_a8w8_blockscale_bpreshuffle_asm_fake_tensors(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
    bpreshuffle: Optional[bool] = True,
) -> Tensor:
    return out


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle_asm",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_asm",
    gen_fake=gen_gemm_a8w8_blockscale_bpreshuffle_asm_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
    bpreshuffle: Optional[bool] = True,
) -> Tensor: ...


def gen_gfx950_a8w8_blockscale_asm_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> Tensor:
    return out


@compile_ops(
    "module_gemm_gfx950_a8w8_blockscale_asm",
    fc_name="gfx950_a8w8_blockscale_asm",
    gen_fake=gen_gfx950_a8w8_blockscale_asm_fake_tensors,
)
def gfx950_a8w8_blockscale_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> Tensor: ...


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    return splitK


_CKGEMM_CONFIG_CACHE = None


@functools.lru_cache(maxsize=1024)
def get_CKGEMM_config(M: int, N: int, K: int, tuned_file="a8w8_tuned_gemm.csv"):
    if tuned_file is None:
        tuned_file = "a8w8_tuned_gemm.csv"
    global _CKGEMM_CONFIG_CACHE

    if _CKGEMM_CONFIG_CACHE is None:
        _CKGEMM_CONFIG_CACHE = {}
    if tuned_file not in _CKGEMM_CONFIG_CACHE:
        ckgemm_dict = pd.read_csv(f"{tuned_file}").drop_duplicates()
        _CKGEMM_CONFIG_CACHE[tuned_file] = ckgemm_dict.set_index(
            ["cu_num", "M", "N", "K"]
        ).to_dict("index")

    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = _CKGEMM_CONFIG_CACHE[tuned_file].get((cu_num, padded_M, N, K), None)
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {tuned_file} , kernel name is {config['kernelName']}!"
                )
            break
    if config is None:
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K}, not found tuned config in {tuned_file}, will use default config!"
        )
    return config


@functools.lru_cache(maxsize=1024)
def get_GEMM_config_with_quant_type(
    M: int,
    N: int,
    K: int,
    q_dtype_w: torch.dtype,
    tuned_file=f"{AITER_ROOT_DIR}/aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv",
):
    # Use dict to cache configs for different files
    if not hasattr(get_GEMM_config_with_quant_type, "file_cache"):
        get_GEMM_config_with_quant_type.file_cache = {}

    # Load file if not cached
    if tuned_file not in get_GEMM_config_with_quant_type.file_cache:
        asmGemmDictDf = pd.read_csv(tuned_file).drop_duplicates()
        get_GEMM_config_with_quant_type.file_cache[tuned_file] = (
            asmGemmDictDf.set_index(["cu_num", "M", "N", "K", "q_dtype_w"]).to_dict(
                "index"
            )
        )

    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = get_GEMM_config_with_quant_type.file_cache[tuned_file].get(
            (cu_num, padded_M, N, K, str(q_dtype_w)), None
        )
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                msg = f"shape M:{M}, N:{N}, K:{K} q_dtype_w:{q_dtype_w}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned, in {tuned_file}!"
                if "libtype" in config:
                    msg += f" libtype is {config['libtype']}!"
                logger.info(msg)
            break
    if config is None:
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K}, q_dtype_w:{q_dtype_w}, not found tuned config in {tuned_file}, will use default config!"
        )
    return config


def gemm_a8w8_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: Optional[int] = None,
) -> Tensor:
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_fake)
def gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: Optional[int] = None,
) -> Tensor:
    # assert dtype in [
    #     dtypes.bf16,
    #     dtypes.fp16,
    # ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    return gemm_a8w8_CK(XQ, WQ, x_scale, w_scale, bias, dtype, splitK)


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
    kernelName = ""
    if (
        x_scale.dtype == dtypes.fp32
        and w_scale.dtype == dtypes.fp32
        and (
            asm_config := get_GEMM_config_with_quant_type(
                m,
                n,
                k,
                dtypes.i8,
                AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
            )
        )
        is not None
    ):
        assert (
            bias is not None
        ), "Use asm gemm must give bias, please give a \
            bias=torch.zeros(n,dtype=dtypes.fp32,device='cuda')"
        splitK = asm_config["splitK"]
        kernelName = asm_config["kernelName"]
        Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
        return gemm_a8w8_asm(
            XQ, WQ, x_scale, w_scale, Y, kernelName, bias, splitK=splitK
        )
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return gemm_a8w8_asm(XQ, WQ, x_scale, w_scale, Y, kernelName, bias, splitK=1)


def gemm_a8w8_CK(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: Optional[int] = None,
) -> Tensor:
    # assert dtype in [
    #     dtypes.bf16,
    #     dtypes.fp16,
    # ], f"Output {dtype=} is currently not supported in gemm_a8w8 CK"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]

    q_dtype_w = WQ.dtype if WQ.dtype in [dtypes.fp8, dtypes.i8] else dtypes.i8
    ck_config = get_GEMM_config_with_quant_type(
        m, n, k, q_dtype_w, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE
    )
    if splitK is None:
        if ck_config is not None:
            splitK = ck_config["splitK"]
        else:
            splitK = 0
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Y, bias, splitK)


def gemm_a8w8_bpreshuffle_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    check: bool = False,
) -> Tensor:
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_bpreshuffle_fake)
def gemm_a8w8_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    check: bool = False,
) -> Tensor:
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]

    # if (
    #     ck_config is None
    #     and dtype == dtypes.bf16
    #     and bias is not None
    #     and WQ.dtype != dtypes.i8
    # ):
    #     res = gemm_a8w8_ASM(XQ, WQ, x_scale, w_scale, bias, dtype=dtype, check=check)
    #     if res is not None:
    #         return res
    assert WQ.dtype == dtypes.fp8, "gemm_a8w8_bpreshuffle only support fp8 now"
    assert bias is None, "gemm_a8w8_bpreshuffle does not support bias now"
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)

    # CKTile only supports bf16 dtype
    config = get_GEMM_config_with_quant_type(
        m,
        n,
        k,
        dtypes.fp8,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
    )
    if config is not None:
        libtype = config["libtype"]
        if libtype == "ck":
            return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y)
        elif libtype == "cktile":
            return gemm_a8w8_bpreshuffle_cktile(XQ, WQ, x_scale, w_scale, Y)
    else:
        return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y)


def gemm_a8w8_blockscale_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    isBpreshuffled=False,
) -> torch.Tensor:
    m = XQ.shape[0]
    n = WQ.shape[0]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return Y


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_fake)
def gemm_a8w8_blockscale(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    isBpreshuffled: bool = False,
) -> torch.Tensor:
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    from aiter.jit.utils.chip_info import get_gfx

    if isBpreshuffled:
        if get_gfx() in ["gfx950"] and m >= 16 and k >= 512 and dtype == dtypes.bf16:
            return gfx950_a8w8_blockscale_ASM(XQ, WQ, x_scale, w_scale, Y)
        else:
            assert 0, "asm kernel only support B preshuffle and m >= 16"
    else:
        config = get_CKGEMM_config(
            m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
        )
        if config is not None:
            libtype = config["libtype"]
            if libtype == "ck":
                return gemm_a8w8_blockscale_ck(XQ, WQ, x_scale, w_scale, Y)
            elif libtype == "cktile":
                return gemm_a8w8_blockscale_cktile(XQ, WQ, x_scale, w_scale, Y)
            else:
                assert 0, f"Unsupported libtype {libtype} for gemm_a8w8_blockscale"
        return gemm_a8w8_blockscale_ck(XQ, WQ, x_scale, w_scale, Y)


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
    # k = XQ.shape[-1]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return flatmm_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, Y)


def gemm_a8w8_blockscale_bpreshuffle_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_bpreshuffle_fake)
def gemm_a8w8_blockscale_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    get_CKGEMM_config(
        m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    )
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return gemm_a8w8_blockscale_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y)


def gfx950_a8w8_blockscale_ASM(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    dtype=dtypes.bf16,
):
    assert dtype in [
        dtypes.bf16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    return gfx950_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, Y)


def gen_gemm_a8w8_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_tune",
    fc_name="gemm_a8w8_tune",
    gen_fake=gen_gemm_a8w8_tune_fake_tensors,
)
def gemm_a8w8_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


def gen_gemm_a8w8_blockscale_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_blockscale_tune",
    fc_name="gemm_a8w8_blockscale_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_cktile_tune",
    fc_name="gemm_a8w8_blockscale_cktile_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_blockscale_cktile_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_bpreshuffle_tune",
    fc_name="gemm_a8w8_bpreshuffle_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_bpreshuffle_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle_tune",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_bpreshuffle_cktile_tune",
    fc_name="gemm_a8w8_bpreshuffle_cktile_tune",
)
def gemm_a8w8_bpreshuffle_cktile_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor: ...
