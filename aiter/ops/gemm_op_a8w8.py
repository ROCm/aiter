# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from typing import Optional

import pandas as pd
import torch
from aiter import logger
from torch import Tensor
from torch.library import Library

from ..jit.core import (
    AITER_CONFIGS,
    AITER_LOG_TUNED_CONFIG,
    compile_ops,
)
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx
from ..jit.utils.torch_guard import torch_compile_guard
from ..ops.gemm_op_common import get_padded_m
from ..utility import dtypes
from ..ops.flydsl.utils import is_flydsl_available

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
    splitK: int = 0,
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
    splitK: int = 0,
) -> torch.Tensor: ...


def gen_gemm_a8w8_bpreshuffle_cktile_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
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
    splitK: int = 0,
) -> Tensor: ...


def _parse_flydsl_kernel_name(kernel_name: str):
    """Parse tile config from flydsl kernelName.
    Returns ``(tile_m, tile_n, tile_k, lds_stage, cshuffle, async_copy,
    waves_per_eu, xcd_swizzle)`` or None on parse failure.
    """
    import re

    m = re.match(
        r"flydsl_bpreshuflle_(\d+)x(\d+)x(\d+)_\w+_\w+_\w+_"
        r"(\d+)x(\d+)x(\d+)x(\d+)x(\d+)",
        kernel_name,
    )
    if m is None:
        return None
    return tuple(int(m.group(i)) for i in range(1, 9))


def gemm_a8w8_bpreshuffle_flydsl(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    config: dict,
) -> Tensor:
    from .flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    kernel_name = config.get("kernelName", "")
    parsed = _parse_flydsl_kernel_name(str(kernel_name))
    if parsed is None:
        return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Out)
    tm, tn, tk, lds, csh, acp, wpe, xcd = parsed

    flydsl_preshuffle_gemm_a8(
        XQ.contiguous(),
        WQ.contiguous(),
        x_scale,
        w_scale,
        Out,
        tm,
        tn,
        tk,
        lds,
        csh,
        acp,
        wpe,
        xcd,
    )
    return Out


@compile_ops(
    "module_gemm_a8w8_asm",
    fc_name="gemm_a8w8_asm",
    ffi_type="ctypes",
)
def _gemm_a8w8_asm(
    XQ: Tensor,  # A:[M, K] i8
    WQ: Tensor,  # B:[N, K] i8 -> shuffle layout(32,16)
    x_scale: Tensor,  # A_scale:[M, 1] f32
    w_scale: Tensor,  # B_scale:[1, N] f32
    Out: Tensor,  # Out:[M, N] bf16
    kernelName: Optional[str] = None,
    bias: Optional[Tensor] = None,  # bias:[1, N] f32
    bpreshuffle: bool = True,
    splitK: int = -1,
) -> None: ...


def gemm_a8w8_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernelName: str = "",
    bias: Optional[Tensor] = None,
    bpreshuffle: Optional[bool] = True,
    splitK: Optional[int] = None,
) -> Tensor:
    _gemm_a8w8_asm(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        kernelName if kernelName else None,
        bias,
        bool(bpreshuffle) if bpreshuffle is not None else True,
        splitK if splitK is not None else -1,
    )
    return Out


def gen_gemm_a8w8_blockscale_ck_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    *args,
    **kwargs,
) -> Tensor:
    # Accept the trailing optional args (isBpreshuffled, splitK, y_is_zeroed)
    # added by the cktile / bpreshuffle_cktile entry points so a single shared
    # fake works for all blockscale FP8 GEMM JIT stubs. The kernel writes into
    # `Out` regardless of those flags.
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
    splitK: int = 0,
    kernelName: str = "",
    y_is_zeroed: bool = False,
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
    isBpreshuffled: bool = False,
    splitK: int = 0,
    kernelName: str = "",
    y_is_zeroed: bool = False,
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
    kernelName: str = "",
    y_is_zeroed: bool = False,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_cktile",
    gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_cktile(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    isBpreshuffled: bool = True,
    splitK: int = 0,
    kernelName: str = "",
    y_is_zeroed: bool = False,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_asm",
    fc_name="flatmm_a8w8_blockscale_asm",
    ffi_type="ctypes",
)
def _flatmm_a8w8_blockscale_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> None: ...
def flatmm_a8w8_blockscale_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> Tensor:
    _flatmm_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, out)
    return out


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle_asm",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_asm",
    ffi_type="ctypes",
)
def _gemm_a8w8_blockscale_bpreshuffle_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Optional[Tensor] = None,
    splitK: int = -1,
    kernelName: Optional[str] = None,
    bpreshuffle: int = 1,
    zero_bias_buf: Optional[Tensor] = None,
    y_is_zeroed: int = 0,
) -> None: ...


# Ref on https://github.com/ROCm/aiter/blob/1be4ee9f70a7a7de5e9f57de2c0ecb9d13ed5983/aiter/ops/gemm_op_a16w16.py#L37-L57
@functools.lru_cache(maxsize=1024)
def get_zero_bias_buf_keyed(
    device: torch.device, stream_id: int, out_shape: int
) -> Tensor:
    return torch.zeros(1, out_shape, dtype=torch.float32, device=device)


def get_zero_bias_buf(B: Tensor) -> Tensor:
    stream = torch.cuda.current_stream(B.device)
    return get_zero_bias_buf_keyed(B.device, stream.cuda_stream, B.shape[0])


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
    zero_bias_buf: Optional[Tensor] = None,
    y_is_zeroed: bool = False,
) -> Tensor:
    if bias is None and zero_bias_buf is None:
        zero_bias_buf = get_zero_bias_buf(B)
    _gemm_a8w8_blockscale_bpreshuffle_asm(
        A,
        B,
        out,
        A_scale,
        B_scale,
        bias,
        splitK if splitK is not None else -1,
        kernelName,
        int(bpreshuffle) if bpreshuffle is not None else 1,
        zero_bias_buf,
        int(y_is_zeroed),
    )
    return out


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    return splitK


_CKGEMM_CONFIG_CACHE: dict = {}
_CKGEMM_HAS_GFX: dict = {}


@functools.lru_cache(maxsize=1024)
def get_CKGEMM_config(M: int, N: int, K: int, tuned_file=None):
    if tuned_file is None:
        tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE
    if tuned_file not in _CKGEMM_CONFIG_CACHE:
        ckgemm_dict = pd.read_csv(f"{tuned_file}").drop_duplicates()
        # Use (gfx, cu_num, M, N, K) key when the CSV has a gfx column (new schema).
        # Fall back to (cu_num, M, N, K) for old CSVs that pre-date the gfx column.
        if "gfx" in ckgemm_dict.columns:
            _CKGEMM_CONFIG_CACHE[tuned_file] = ckgemm_dict.set_index(
                ["gfx", "cu_num", "M", "N", "K"]
            ).to_dict("index")
            _CKGEMM_HAS_GFX[tuned_file] = True
        else:
            logger.warning(
                f"{tuned_file} has no 'gfx' column — falling back to cu_num-only key. "
                "Re-run the tuner or migrate the CSV to add a gfx column."
            )
            _CKGEMM_CONFIG_CACHE[tuned_file] = ckgemm_dict.set_index(
                ["cu_num", "M", "N", "K"]
            ).to_dict("index")
            _CKGEMM_HAS_GFX[tuned_file] = False

    gfx = get_gfx()
    cu_num = get_cu_num()
    has_gfx = _CKGEMM_HAS_GFX[tuned_file]
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        key = (gfx, cu_num, padded_M, N, K) if has_gfx else (cu_num, padded_M, N, K)
        config = _CKGEMM_CONFIG_CACHE[tuned_file].get(key, None)
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


_GEMM_QUANT_TYPE_CACHE: dict = {}
_GEMM_QUANT_TYPE_HAS_GFX: dict = {}


@functools.lru_cache(maxsize=1024)
def get_GEMM_config_with_quant_type(
    M: int,
    N: int,
    K: int,
    q_dtype_w: torch.dtype,
    tuned_file=None,
):
    if tuned_file is None:
        tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    # Load file if not cached
    if tuned_file not in _GEMM_QUANT_TYPE_CACHE:
        asmGemmDictDf = pd.read_csv(tuned_file).drop_duplicates()
        # Use (gfx, cu_num, M, N, K, q_dtype_w) key when the CSV has a gfx column (new schema).
        # Fall back to (cu_num, M, N, K, q_dtype_w) for old CSVs that pre-date the gfx column.
        if "gfx" in asmGemmDictDf.columns:
            _GEMM_QUANT_TYPE_CACHE[tuned_file] = asmGemmDictDf.set_index(
                ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"]
            ).to_dict("index")
            _GEMM_QUANT_TYPE_HAS_GFX[tuned_file] = True
        else:
            logger.warning(
                f"{tuned_file} has no 'gfx' column — falling back to cu_num-only key. "
                "Re-run the tuner or migrate the CSV to add a gfx column."
            )
            _GEMM_QUANT_TYPE_CACHE[tuned_file] = asmGemmDictDf.set_index(
                ["cu_num", "M", "N", "K", "q_dtype_w"]
            ).to_dict("index")
            _GEMM_QUANT_TYPE_HAS_GFX[tuned_file] = False

    gfx = get_gfx()
    cu_num = get_cu_num()
    has_gfx = _GEMM_QUANT_TYPE_HAS_GFX[tuned_file]
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        key = (
            (gfx, cu_num, padded_M, N, K, str(q_dtype_w))
            if has_gfx
            else (cu_num, padded_M, N, K, str(q_dtype_w))
        )
        config = _GEMM_QUANT_TYPE_CACHE[tuned_file].get(key, None)
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
        ), "Use asm gemm must give bias, please give a bias=torch.zeros(n,dtype=dtypes.fp32,device='cuda')"
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
    try:
        return gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Y, bias, splitK)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_CK failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, {splitK=}, config={ck_config}: {e}"
        ) from e


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
        splitK = int(config["splitK"])
        if libtype == "ck":
            return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y, splitK)
        elif libtype == "cktile":
            return gemm_a8w8_bpreshuffle_cktile(XQ, WQ, x_scale, w_scale, Y, splitK)
        elif libtype == "flydsl" and is_flydsl_available():
            return gemm_a8w8_bpreshuffle_flydsl(XQ, WQ, x_scale, w_scale, Y, config)
    try:
        return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y, 0)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_bpreshuffle failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, config={config}: {e}"
        ) from e


def gemm_a8w8_blockscale_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    isBpreshuffled=False,
    out: Optional[Tensor] = None,
    y_is_zeroed: bool = False,
) -> torch.Tensor:
    if out is not None:
        return out
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
    out: Optional[Tensor] = None,
    y_is_zeroed: bool = False,
) -> torch.Tensor:
    """FP8 a8w8 blockscale GEMM (non-bpreshuffle).

    Optional kwargs (used by the zero-init SplitK fusion):
      - out:          caller-provided output buffer. If None, allocated here.
      - y_is_zeroed:  when True, caller guarantees ``out`` is pre-zeroed and
                      the kernel/wrapper skips its zero-init step before the
                      SplitK atomic-add. Honored by both cktile and legacy
                      CK branches: cktile's QuantGemmHostArgs passes the
                      flag down to the kernel; legacy CK gates aiter's
                      ``Y.zero_()`` while telling the invoker to skip its
                      own ``hipMemsetAsync`` via ``skip_zero_init=true``.
                      Has no effect when the chosen kernel runs with
                      KBatch == 1 (no SplitK atomic-add => no zero-init
                      required).

    SplitK and kernel-name selection are *always* driven by the tuned CSV
    (``AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE``).  Callers cannot override
    them: the C++ cktile dispatch is keyed by ``kernelName``, and an empty
    name falls back to a non-tuned default heuristic kernel.  Bypassing the
    CSV would therefore silently lose the tuned kernel selection, so this
    wrapper always resolves splitK and ``kernelName`` from the CSV rather
    than accepting them as arguments.
    """
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    if out is None:
        Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    else:
        assert out.shape == (m, n), f"out shape {tuple(out.shape)} != ({m}, {n})"
        assert out.dtype == dtype, f"out dtype {out.dtype} != {dtype}"
        Y = out
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
            splitK = int(config.get("splitK", 0))
            kernelName = str(config.get("kernelName", ""))
            if libtype == "ck":
                return gemm_a8w8_blockscale_ck(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    Y,
                    splitK=splitK,
                    kernelName=kernelName,
                    y_is_zeroed=y_is_zeroed,
                )
            elif libtype == "cktile":
                return gemm_a8w8_blockscale_cktile(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    Y,
                    splitK=splitK,
                    kernelName=kernelName,
                    y_is_zeroed=y_is_zeroed,
                )
            else:
                assert 0, f"Unsupported libtype {libtype} for gemm_a8w8_blockscale"
        try:
            return gemm_a8w8_blockscale_ck(
                XQ, WQ, x_scale, w_scale, Y, y_is_zeroed=y_is_zeroed
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"gemm_a8w8_blockscale failed for shape M={m}, N={n}, K={k}, "
                f"{dtype=}, config={config}: {e}"
            ) from e


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
    out: Optional[Tensor] = None,
    y_is_zeroed: bool = False,
    tuned_file: Optional[str] = None,
) -> Tensor:
    if out is not None:
        return out
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_bpreshuffle_fake)
def gemm_a8w8_blockscale_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    out: Optional[Tensor] = None,
    y_is_zeroed: bool = False,
    tuned_file: Optional[str] = None,
) -> Tensor:
    """FP8 a8w8 blockscale GEMM with bpreshuffled weight layout.

    Args:
        XQ: FP8 activations, shape (M, K).
        WQ: FP8 weights (preshuffled), shape (N, K) logically.
        x_scale: per_1x128 activation scales.
        w_scale: per_128x128 weight scales.
        dtype: output dtype.
        out: optional pre-allocated output tensor (shape (M, N), dtype `dtype`).
            When provided, the GEMM writes into this tensor instead of allocating a new one.
        y_is_zeroed: when True, the caller has already zeroed `out`, so the
            kernel will skip its internal Y.zero_() / hipMemsetAsync before
            the SplitK atomic_add.  Honored by all libtypes (cktile, ck,
            asm).  Has no effect on the libtype=='ck' branch, which does not
            expose splitK from the CSV (KBatch stays at 1, so the zero-init
            is a no-op regardless of this flag); the plumbing is in place for
            when splitK is exposed there.
        tuned_file: optional path to a tuned CSV to consult for kernel/splitK
            selection.  When None, uses the default
            ``AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE``.  Useful
            when a caller wants to switch between e.g. a no-SplitK and a
            with-SplitK tune at runtime.
    """
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    if tuned_file is None:
        tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    config = get_CKGEMM_config(m, n, k, tuned_file)
    if out is None:
        Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    else:
        assert out.shape == (m, n), f"out shape {tuple(out.shape)} != ({m}, {n})"
        assert out.dtype == dtype, f"out dtype {out.dtype} != {dtype}"
        Y = out
    if config is not None:
        libtype = config["libtype"]
        kernelName = str(config.get("kernelName", ""))
        splitK = int(config.get("splitK", 0))
        if libtype == "cktile":
            return gemm_a8w8_blockscale_bpreshuffle_cktile(
                XQ,
                WQ,
                x_scale,
                w_scale,
                Y,
                splitK=splitK,
                kernelName=kernelName,
                y_is_zeroed=y_is_zeroed,
            )
        elif libtype == "ck":
            return gemm_a8w8_blockscale_bpreshuffle_ck(
                XQ,
                WQ,
                x_scale,
                w_scale,
                Y,
                kernelName=kernelName,
                y_is_zeroed=y_is_zeroed,
            )
        elif libtype == "asm":
            return gemm_a8w8_blockscale_bpreshuffle_asm(
                XQ,
                WQ,
                Y,
                x_scale,
                w_scale,
                splitK=splitK,
                kernelName=kernelName,
                y_is_zeroed=y_is_zeroed,
            )
    try:
        return gemm_a8w8_blockscale_bpreshuffle_ck(
            XQ, WQ, x_scale, w_scale, Y, y_is_zeroed=y_is_zeroed
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_blockscale_bpreshuffle failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, config={config}: {e}"
        ) from e


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
    *args,
    **kwargs,
) -> torch.Tensor:
    # Accept the trailing optional flags some tune entrypoints carry
    # (``preshuffleB``, ``y_is_zeroed``) so a single shared fake works across
    # all blockscale tune JIT stubs. The kernel writes into ``Out`` regardless.
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
    preshuffleB: bool = False,
    y_is_zeroed: bool = False,
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
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_cktile_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
    preshuffleB: bool = True,
    y_is_zeroed: bool = False,
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
