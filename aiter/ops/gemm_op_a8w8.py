# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
import functools
import os
import sys
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

# ---------------------------------
# Optional FlyDSL preshuffle GEMM
# ---------------------------------
# Enable with:
#   export AITER_USE_FLYDSL_GEMM=1
#   export DSL2_ROOT=/path/to/FlyDSL
_flydsl_preshuffle_available: bool = False
_flydsl_preshuffle_failed_once: bool = False
_flydsl_compile_preshuffle_gemm_a8 = None
_flydsl_debug_seen: set[str] = set()


def _flydsl_debug_enabled() -> bool:
    return os.environ.get("AITER_FLYDSL_DEBUG", "0") in ("1", "true", "True", "YES", "yes")


def _flydsl_log_once(key: str, msg: str) -> None:
    if not _flydsl_debug_enabled():
        return
    if key in _flydsl_debug_seen:
        return
    _flydsl_debug_seen.add(key)
    logger.info(msg)


def _use_flydsl_gemm() -> bool:
    return os.environ.get("AITER_USE_FLYDSL_GEMM", "0") in ("1", "true", "True", "YES", "yes")


def _init_flydsl_preshuffle_backend() -> None:
    global _flydsl_preshuffle_available, _flydsl_preshuffle_failed_once, _flydsl_compile_preshuffle_gemm_a8
    if _flydsl_preshuffle_failed_once or _flydsl_preshuffle_available:
        return
    if not _use_flydsl_gemm():
        return

    dsl2_root = os.environ.get("DSL2_ROOT", None)
    if not dsl2_root:
        logger.warning("AITER_USE_FLYDSL_GEMM=1 but DSL2_ROOT is not set; falling back to existing backends.")
        _flydsl_preshuffle_failed_once = True
        return

    # Prefer the FlyDSL python sources from this checkout.
    flydsl_src = os.path.join(dsl2_root, "flydsl", "src")
    for p in (flydsl_src, dsl2_root):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8 as _compile  # type: ignore

        _flydsl_compile_preshuffle_gemm_a8 = _compile
        _flydsl_preshuffle_available = True
    except Exception as e:
        logger.warning(
            f"FlyDSL preshuffle GEMM backend not available ({type(e).__name__}: {e}); falling back to existing backends."
        )
        _flydsl_preshuffle_failed_once = True


def _flydsl_choose_tile_n(n: int) -> Optional[int]:
    for cand in (64, 128, 256):
        if n % cand == 0:
            return cand
    return None


def _flydsl_choose_tile_k(k: int, in_dtype: str) -> int:
    # tile_k is in element units; kernel requires tile_k_bytes % 64 == 0.
    if in_dtype in ("fp16", "bf16"):
        for cand in (128, 64, 32):
            if k % cand == 0:
                return cand
        return 32
    for cand in (256, 128, 64):
        if k % cand == 0:
            return cand
    return 64


@functools.lru_cache(maxsize=256)
def _flydsl_get_exe(n: int, k: int, in_dtype: str, tile_m: int, tile_n: int, tile_k: int, lds_stage: int):
    _init_flydsl_preshuffle_backend()
    if not _flydsl_preshuffle_available or _flydsl_compile_preshuffle_gemm_a8 is None:
        return None
    # NOTE: compile signature requires M,N,K though M is dynamic at runtime; pass M=1.
    return _flydsl_compile_preshuffle_gemm_a8(
        M=1,
        N=int(n),
        K=int(k),
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        in_dtype=in_dtype,
        lds_stage=int(lds_stage),
        use_cshuffle_epilog=False,
    )


def _flydsl_preshuffle_gemm_a8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out_dtype: torch.dtype,
) -> Optional[Tensor]:
    """Best-effort FlyDSL preshuffle GEMM for A8/W8 style kernels.

    ABI matches FlyDSL test: exe(C, A, B_shuf, scaleA, scaleB, M, N, K).
    Returns None if constraints are not met.
    """
    global _flydsl_preshuffle_failed_once, _flydsl_preshuffle_available

    _init_flydsl_preshuffle_backend()
    if not _flydsl_preshuffle_available:
        _flydsl_log_once("flydsl_a8w8_unavailable", "[flydsl][a8w8_bpreshuffle] backend unavailable -> fallback")
        return None
    if not _use_flydsl_gemm():
        _flydsl_log_once("flydsl_a8w8_disabled", "[flydsl][a8w8_bpreshuffle] disabled via AITER_USE_FLYDSL_GEMM")
        return None

    # FlyDSL path is not CUDA-graph-capture safe today (may allocate / cast).
    # During capture, always fall back to existing aiter kernels.
    if torch.cuda.is_current_stream_capturing():
        _flydsl_log_once(
            "flydsl_a8w8_skip_cudagraph",
            "[flydsl][a8w8_bpreshuffle] skip: stream is capturing (cudagraph)",
        )
        return None

    if XQ.dim() != 2 or WQ.dim() != 2:
        _flydsl_log_once(
            f"flydsl_a8w8_skip_rank_{XQ.dim()}_{WQ.dim()}",
            f"[flydsl][a8w8_bpreshuffle] skip: expect 2D tensors, got A.dim={XQ.dim()} B.dim={WQ.dim()}",
        )
        return None
    m, k = XQ.shape
    n = WQ.shape[0]
    if WQ.shape[1] != k:
        _flydsl_log_once(
            f"flydsl_a8w8_skip_shape_{m}_{n}_{k}_{WQ.shape[1]}",
            f"[flydsl][a8w8_bpreshuffle] skip: shape mismatch A[K]={k} B[K]={WQ.shape[1]}",
        )
        return None

    # FlyDSL kernel outputs fp16; we cast below (outside cudagraph capture).
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        _flydsl_log_once(
            f"flydsl_a8w8_skip_outdtype_{str(out_dtype)}",
            f"[flydsl][a8w8_bpreshuffle] skip: unsupported out dtype {out_dtype}",
        )
        return None

    # This wrapper only wires the fp8/int8/int4 codepaths if needed later.
    # In current aiter op, WQ is fp8. Require A also fp8.
    if XQ.dtype != dtypes.fp8 or WQ.dtype != dtypes.fp8:
        _flydsl_log_once(
            f"flydsl_a8w8_skip_dtype_{str(XQ.dtype)}_{str(WQ.dtype)}",
            f"[flydsl][a8w8_bpreshuffle] skip: dtype mismatch A={XQ.dtype} B={WQ.dtype} (need fp8/fp8)",
        )
        return None
    in_dtype = "fp8"

    # FlyDSL kernel has no N tail.
    tile_n = _flydsl_choose_tile_n(n)
    if tile_n is None:
        _flydsl_log_once(
            f"flydsl_a8w8_skip_ntail_{n}",
            f"[flydsl][a8w8_bpreshuffle] skip: N={n} not divisible by supported tile_n (64/128/256)",
        )
        return None
    tile_k = _flydsl_choose_tile_k(k, in_dtype)
    if k % tile_k != 0:
        _flydsl_log_once(
            f"flydsl_a8w8_skip_kalign_{k}_{tile_k}",
            f"[flydsl][a8w8_bpreshuffle] skip: K={k} not divisible by tile_k={tile_k}",
        )
        return None

    # Additional kernel constraints from FlyDSL preshuffle_gemm implementation:
    # bytes_per_thread_a = tile_m*tile_k*elem_bytes/256 must be divisible by 16.
    elem_bytes = 1  # fp8
    tile_m = None
    for cand_m in (64, 32, 16):
        if m % cand_m != 0:
            continue
        bytes_per_thread_a = (cand_m * tile_k * elem_bytes) // 256
        if (cand_m * tile_k * elem_bytes) % 256 != 0:
            continue
        if bytes_per_thread_a % 16 == 0:
            tile_m = cand_m
            break
    if tile_m is None:
        _flydsl_log_once(
            f"flydsl_a8w8_skip_tilem_{m}_{k}_{tile_k}",
            f"[flydsl][a8w8_bpreshuffle] skip: cannot find tile_m for M={m} K={k} tile_k={tile_k} (constraint bytes_per_thread_a%16==0)",
        )
        return None

    # Scales: expect per-row/per-token scales. Accept common (M,1)/(N,1) shapes.
    sa = x_scale.to(torch.float32).contiguous().view(-1)
    sb = w_scale.to(torch.float32).contiguous().view(-1)
    if sa.numel() != m or sb.numel() != n:
        _flydsl_log_once(
            f"flydsl_a8w8_skip_scales_{m}_{n}_{sa.numel()}_{sb.numel()}",
            f"[flydsl][a8w8_bpreshuffle] skip: scales size mismatch sa={sa.numel()} (need {m}) sb={sb.numel()} (need {n})",
        )
        return None

    exe = _flydsl_get_exe(n=n, k=k, in_dtype=in_dtype, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, lds_stage=2)
    if exe is None:
        _flydsl_log_once(
            f"flydsl_a8w8_noexe_{m}_{n}_{k}_{tile_m}_{tile_n}_{tile_k}",
            f"[flydsl][a8w8_bpreshuffle] skip: exe compile returned None for M={m} N={n} K={k} tile(m,n,k)=({tile_m},{tile_n},{tile_k})",
        )
        return None

    out_f16 = torch.empty((m, n), dtype=torch.float16, device=XQ.device)
    try:
        _flydsl_log_once(
            f"flydsl_a8w8_hit_{m}_{n}_{k}_{tile_m}_{tile_n}_{tile_k}",
            f"[flydsl][a8w8_bpreshuffle] HIT M={m} N={n} K={k} in=fp8 out=fp16 tile(m,n,k)=({tile_m},{tile_n},{tile_k}) lds_stage=2",
        )
        exe(out_f16, XQ, WQ, sa, sb, m, n, k)
    except Exception as e:
        _flydsl_log_once(
            f"flydsl_a8w8_fail_{m}_{n}_{k}_{tile_m}_{tile_n}_{tile_k}",
            f"[flydsl][a8w8_bpreshuffle] FAIL M={m} N={n} K={k} tile(m,n,k)=({tile_m},{tile_n},{tile_k}) err={type(e).__name__}: {e}",
        )
        _flydsl_preshuffle_failed_once = True
        _flydsl_preshuffle_available = False
        return None

    return out_f16 if out_dtype == torch.float16 else out_f16.to(out_dtype)


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
def get_bpreshuffle_GEMM_config(
    M: int,
    N: int,
    K: int,
    q_dtype_w: torch.dtype,
    tuned_file=f"{AITER_ROOT_DIR}/aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv",
):
    # Use dict to cache configs for different files
    if not hasattr(get_bpreshuffle_GEMM_config, "file_cache"):
        get_bpreshuffle_GEMM_config.file_cache = {}

    # Load file if not cached
    if tuned_file not in get_bpreshuffle_GEMM_config.file_cache:
        asmGemmDictDf = pd.read_csv(tuned_file).drop_duplicates()
        get_bpreshuffle_GEMM_config.file_cache[tuned_file] = asmGemmDictDf.set_index(
            ["cu_num", "M", "N", "K", "q_dtype_w"]
        ).to_dict("index")

    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = get_bpreshuffle_GEMM_config.file_cache[tuned_file].get(
            (cu_num, padded_M, N, K, str(q_dtype_w)), None
        )
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"shape M:{M}, N:{N}, K:{K} q_dtype_w:{q_dtype_w}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned, in {tuned_file}, libtype is {config['libtype']}!"
                )
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
            asm_config := get_bpreshuffle_GEMM_config(
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
    ck_config = get_CKGEMM_config(m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE)
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
    # Optional FlyDSL preshuffle GEMM path (best-effort; falls back automatically).
    y_flydsl = _flydsl_preshuffle_gemm_a8(XQ.contiguous(), WQ.contiguous(), x_scale, w_scale, out_dtype=dtype)
    if y_flydsl is not None:
        return y_flydsl

    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)

    # CKTile only supports bf16 dtype
    config = get_bpreshuffle_GEMM_config(
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
        get_CKGEMM_config(m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
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
