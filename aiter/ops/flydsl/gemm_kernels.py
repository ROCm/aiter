# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HGEMM APIs."""

from __future__ import annotations

import re
import functools
from itertools import product
from typing import Dict, Optional

import torch
from torch import Tensor

import flydsl.expr as fx
from aiter import logger
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

from aiter.jit.utils.chip_info import get_gfx

from .kernels.hgemm_wmma_gfx950 import hgemm, hgemm_get_configs, infer_has_k_tail

# from .kernels.small_m_hgemm import iter_small_m_registry_configs
from .kernels.tensor_shim import _run_compiled
from .utils import get_shared_memory_per_block, is_flydsl_available

__all__ = [
    "flydsl_hgemm",
]


def _get_dtypes():
    from aiter.utility import dtypes

    return dtypes


_HGEMM_KERNELS: Dict[str, Dict] = {}


def flydsl_kernel_name(
    dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    block_k_warps: int,
    has_bias: bool,
    has_k_tail: bool,
    group_m: int,
    policy: str,
) -> str:
    name = f"flydsl_hgemm_a{dtype}_w{dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}x{stages}_ks{split_k}"
    name += f"_w{block_m_warps}x{block_n_warps}x{block_k_warps}_bias{int(has_bias)}_ktail{int(has_k_tail)}"
    name += f"_gm{group_m}_p{policy}"
    name += f"_{get_gfx()}"
    return name


def get_flydsl_hgemm_kernel_params(name: str) -> Optional[Dict]:
    config = _HGEMM_KERNELS.get(name)
    if config is not None:
        return dict(config)
    return None


def get_flydsl_hgemm_kernels(
    dtype: str,
    out_dtype: str,
    has_bias: bool,
    m: Optional[int] = None,
    n: Optional[int] = None,
    k: Optional[int] = None,
    has_k_tail: Optional[bool] = None,
) -> Dict[str, Dict]:
    kernels = {}
    if any(dim is None for dim in (m, n, k)) and any(
        dim is not None for dim in (m, n, k)
    ):
        raise ValueError(
            "m, n, k must be provided together when requesting shape-aware kernels"
        )
    configs = hgemm_get_configs(dtype, m, n, k)
    for config in configs:
        tile_m = config["TILE_M"]
        tile_n = config["TILE_N"]
        tile_k = config["TILE_K"]
        stages = config["STAGES"]
        split_k = config["SPLIT_K"]
        block_m_warps = config["BLOCK_M_WARPS"]
        block_n_warps = config["BLOCK_N_WARPS"]
        block_k_warps = config["BLOCK_K_WARPS"]
        policy = "ht" if config["USE_HALF_TILE_INTERLEAVED"] else "ft"
        group_m = config["GROUP_M"]
        if has_k_tail is None:
            has_k_tail = infer_has_k_tail(
                k=k,
                split_k=config["SPLIT_K"],
                tile_k=config["TILE_K"],
                is_ht=config["USE_HALF_TILE_INTERLEAVED"],
            )
        name = flydsl_kernel_name(
            dtype=dtype,
            out_dtype=out_dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            stages=stages,
            split_k=split_k,
            block_m_warps=block_m_warps,
            block_n_warps=block_n_warps,
            block_k_warps=block_k_warps,
            has_bias=has_bias,
            has_k_tail=has_k_tail,
            group_m=group_m,
            policy=policy,
        )
        config["HAS_BIAS"] = has_bias
        config["HAS_K_TAIL"] = has_k_tail
        config["dtype"] = dtype
        config["out_dtype"] = out_dtype
        kernels[name] = config
    return kernels


def _register_all_configs():
    for dtype in ("bf16", "fp16"):
        for out_dtype in ("fp16", "bf16"):
            for has_bias in (True, False):
                for has_k_tail in (True, False):
                    _HGEMM_KERNELS.update(
                        get_flydsl_hgemm_kernels(
                            dtype, out_dtype, has_bias, has_k_tail=has_k_tail
                        )
                    )


_register_all_configs()


def flydsl_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    stages: int = 2,
    split_k: int = 1,
    block_m_warps: int = 2,
    block_n_warps: int = 2,
    block_k_warps: int = 1,
    group_m: int = 0,
    policy: str = "ft",
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run FlyDSL HGEMM."""

    user_kwargs = {
        "TILE_M": tile_m,
        "TILE_N": tile_n,
        "TILE_K": tile_k,
        "STAGES": stages,
        "SPLIT_K": split_k,
        "BLOCK_M_WARPS": block_m_warps,
        "BLOCK_N_WARPS": block_n_warps,
        "BLOCK_K_WARPS": block_k_warps,
        "USE_HALF_TILE_INTERLEAVED": False if policy == "ft" else True,
        "GROUP_M": group_m,
    }

    out = hgemm(a, b, out, bias, user_kwargs, stream=stream)
    return out


# ---------------------------------------------------------------------------
# FlyDSL preshuffle GEMM kernel management
# ---------------------------------------------------------------------------

_flydsl_compile_fn = None
_flydsl_import_done = False


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
    xcd_swizzle: int = 0,
) -> Tensor:
    """Compile (cached via lru_cache) and run a FlyDSL preshuffle GEMM kernel."""
    compile_fn = _get_compile_fn()
    if compile_fn is None:
        raise RuntimeError("[FlyDSL] compile function not available")
    dtypes = _get_dtypes()

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]

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
    elif Out.dtype == torch.float16:
        out_dtype = "fp16"
    else:
        raise ValueError(
            f"[FlyDSL] unsupported output dtype {Out.dtype}; expected torch.bfloat16 or torch.float16"
        )

    exe = compile_fn(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        lds_stage=lds_stage,
        use_cshuffle_epilog=bool(use_cshuffle_epilog),
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
        xcd_swizzle=int(xcd_swizzle),
    )

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    out_contig = Out.contiguous()
    # FlyDSL's preshuffle kernel requires an arg_bias slot (used only when
    # epilogue != "none"). Pass an empty tensor as a placeholder for the
    # default epilogue="none" path.
    _dummy_bias = torch.empty(0, dtype=Out.dtype, device=Out.device)
    _run_compiled(
        exe,
        ptr_arg(out_contig.view(-1)),
        ptr_arg(_as_i8(XQ.contiguous()).view(-1)),
        ptr_arg(_as_i8(WQ.contiguous()).view(-1)),
        ptr_arg(x_scale.contiguous().view(-1)),
        ptr_arg(w_scale.contiguous().view(-1)),
        ptr_arg(_dummy_bias),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not Out:
        Out.copy_(out_contig)

    return Out
