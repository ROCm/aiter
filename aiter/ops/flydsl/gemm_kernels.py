# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL preshuffle GEMM kernel management: compilation and high-level API."""

import torch
from torch import Tensor

from aiter import logger
from aiter.utility import dtypes

from .utils import is_flydsl_available

_flydsl_compile_fn = None
_flydsl_import_done = False
_flydsl_kernel_cache: dict = {}


def _get_compile_fn():
    """Lazy-import compile_preshuffle_gemm_a8 so the module loads even without FlyDSL."""
    global _flydsl_compile_fn, _flydsl_import_done
    if _flydsl_import_done:
        return _flydsl_compile_fn
    _flydsl_import_done = True
    if not is_flydsl_available():
        logger.info("[FlyDSL] not available, will fall back to CK/CKTile")
        return None
    try:
        from .kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

        _flydsl_compile_fn = compile_preshuffle_gemm_a8
        logger.info("[FlyDSL] loaded preshuffle GEMM compiler")
    except Exception as e:
        logger.info(
            f"[FlyDSL] preshuffle GEMM not available, will fall back to CK/CKTile: {e}"
        )
    return _flydsl_compile_fn


def flydsl_select_tiles(M: int, N: int, K: int):
    """Pick (tile_m, tile_n, tile_k) satisfying kernel divisibility constraints.

    Returns a tuple or *None* when no valid tiling exists.
    """
    for tile_n in (256, 128, 64):
        if N % tile_n == 0:
            break
    else:
        return None
    for tile_k in (512, 256, 128):
        if K % tile_k == 0:
            break
    else:
        return None
    if M <= 16:
        tile_m = 16
    elif M <= 64:
        tile_m = 32
    elif M <= 256:
        tile_m = 64
    else:
        tile_m = 128
    return tile_m, tile_n, tile_k


def flydsl_parse_gemm_kernel_name(kernel_name: str):
    """Extract tuning params from a FlyDSL kernel name.

    Format: flydsl_bpreshuflle_{TM}x{TN}x{TK}_{qa}_{qw}_{dt}_{lds}x{csh}x{acp}x{wpe}_{sched}

    Returns (tile_m, tile_n, tile_k, lds_stage, use_cshuffle_epilog, use_async_copy, waves_per_eu).
    """
    parts = kernel_name.split("_")
    tm, tn, tk = (int(v) for v in parts[2].split("x"))
    cfg = [int(v) for v in parts[6].split("x")]
    return tm, tn, tk, cfg[0], cfg[1], cfg[2], (cfg[3] if len(cfg) > 3 else 0)


def flydsl_preshuffle_gemm_a8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    lds_stage: int = 2,
    use_cshuffle_epilog: int = 0,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
) -> Tensor:
    """Compile (cached) and run a FlyDSL preshuffle GEMM kernel."""
    compile_fn = _get_compile_fn()
    if compile_fn is None:
        raise RuntimeError("[FlyDSL] compile function not available")

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]

    if m % tile_m != 0:
        raise RuntimeError(
            f"[FlyDSL] M ({m}) is not a multiple of tile_m ({tile_m}). "
            f"Arguments not supported! Skipping gemm!"
        )
    if n % tile_n != 0:
        raise RuntimeError(
            f"[FlyDSL] N ({n}) is not a multiple of tile_n ({tile_n}). "
            f"Arguments not supported! Skipping gemm!"
        )
    if k % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL] K ({k}) is not a multiple of tile_k ({tile_k}). "
            f"Arguments not supported! Skipping gemm!"
        )

    if XQ.dtype == dtypes.fp8:
        in_dtype = "fp8"
    elif XQ.dtype == torch.int8:
        in_dtype = "int8"
    else:
        raise ValueError(f"[FlyDSL] unsupported input dtype {XQ.dtype}")

    wpe = None if waves_per_eu <= 0 else waves_per_eu

    if Out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    else:
        out_dtype = "fp16"

    cache_key = (
        m, n, k, in_dtype, out_dtype,
        tile_m, tile_n, tile_k,
        lds_stage, use_cshuffle_epilog, use_async_copy, wpe,
    )
    if cache_key not in _flydsl_kernel_cache:
        try:
            exe = compile_fn(
                M=m, N=n, K=k,
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
                lds_stage=lds_stage,
                use_cshuffle_epilog=bool(use_cshuffle_epilog),
                use_async_copy=bool(use_async_copy),
                waves_per_eu=wpe,
            )
            _flydsl_kernel_cache[cache_key] = exe
            logger.info(
                f"[FlyDSL] compiled preshuffle GEMM ({m},{n},{k} {in_dtype} "
                f"tile={tile_m}x{tile_n}x{tile_k} lds={lds_stage} csh={use_cshuffle_epilog} "
                f"acp={use_async_copy} wpe={waves_per_eu})"
            )
        except Exception as e:
            logger.warning(f"[FlyDSL] compile failed ({m},{n},{k} {in_dtype}): {e}")
            _flydsl_kernel_cache[cache_key] = None

    exe = _flydsl_kernel_cache[cache_key]
    if exe is None:
        raise RuntimeError(f"[FlyDSL] kernel compile returned None for ({m},{n},{k})")

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    exe(
        Out.contiguous().view(-1),
        _as_i8(XQ.contiguous()).view(-1),
        _as_i8(WQ.contiguous()).view(-1),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        m, n,
        torch.cuda.current_stream(),
    )

    return Out
