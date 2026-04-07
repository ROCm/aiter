# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HGEMM APIs."""

from __future__ import annotations

from itertools import product
from typing import Dict, Optional

import torch

from ..shuffle import shuffle_weight
from .kernels.splitk_hgemm import compile_hgemm_kernel

from aiter.jit.utils.chip_info import get_gfx

__all__ = [
    "flydsl_hgemm",
]

SPLIT_K_COUNTER_MAX_LEN = 128
SPLIT_K_SIGNAL_STATE_COUNT = 3
SPLIT_K_GLOBAL_SEMAPHORE: dict[torch.device, torch.Tensor] = {}
SPLIT_K_GLOBAL_SEMAPHORE_STATE: dict[torch.device, int] = {}


_SPLITK_HGEMM_KERNELS: Dict[str, Dict] = {}


def flydsl_kernel_name(
    stage: int,
    dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_m_warp: int,
    block_n_warp: int,
    async_copy: bool,
    b_to_lds: bool,
    b_preshuffle: bool,
    c_to_lds: bool,
) -> str:
    """Construct kernel name: flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}]."""
    name = (
        f"flydsl_gemm{stage}_a{dtype}_w{dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    )
    name += f"_split_k{split_k}_block_m_warp{block_m_warp}_block_n_warp{block_n_warp}"
    name += f"_async_copy{async_copy}_b_to_lds{b_to_lds}_b_preshuffle{b_preshuffle}_c_to_lds{c_to_lds}"
    name += f"_{get_gfx()}"
    return name


def get_flydsl_splitk_hgemm_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name (O(1))."""
    return _SPLITK_HGEMM_KERNELS.get(name)


def get_flydsl_splitk_hgemm_kernels(dtype: str, out_dtype: str) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported configs."""
    kernels = {}
    tile_ns = [64, 128, 256]
    tile_ks = [64, 128]
    tile_ms = [16, 32, 48, 64, 96, 128]
    split_ks = [1, 2, 4, 8]
    stages = [1, 2]
    block_m_warps = [1]
    block_n_warps = [4]
    async_copy = [True, False]
    b_to_lds = [True, False]
    b_preshuffle = [True, False]
    c_to_lds = [True, False]

    for (
        tile_m,
        tile_n,
        tile_k,
        split_k,
        stage,
        block_m_warp,
        block_n_warp,
        use_async_copy,
        use_b_to_lds,
        use_b_preshuffle,
        use_c_to_lds,
    ) in product(
        tile_ms,
        tile_ns,
        tile_ks,
        split_ks,
        stages,
        block_m_warps,
        block_n_warps,
        async_copy,
        b_to_lds,
        b_preshuffle,
        c_to_lds,
    ):
        params = {
            "stage": stage,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "split_k": split_k,
            "block_m_warps": block_m_warp,
            "block_n_warps": block_n_warp,
            "async_copy": use_async_copy,
            "b_to_lds": use_b_to_lds,
            "b_preshuffle": use_b_preshuffle,
            "c_to_lds": use_c_to_lds,
        }
        name = flydsl_kernel_name(
            stage,
            dtype,
            out_dtype,
            tile_m,
            tile_n,
            tile_k,
            split_k,
            block_m_warp,
            block_n_warp,
            use_async_copy,
            use_b_to_lds,
            use_b_preshuffle,
            use_c_to_lds,
        )
        kernels[name] = params
    return kernels


def _register_all_configs():
    """Pre-populate _KERNEL_PARAMS with all supported configs at import time."""
    for dtype in ("bf16", "f16"):
        for out_dtype in ("f16", "bf16"):
            _SPLITK_HGEMM_KERNELS.update(
                get_flydsl_splitk_hgemm_kernels(dtype, out_dtype)
            )


_register_all_configs()


def _to_kernel_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"Only fp16/bf16 are supported, got {dtype!r}")


def _get_flydsl_shuffle_layout(pack_n: int) -> tuple[int, int]:
    return (16 * pack_n, 16)


def _validate_hgemm_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor],
) -> tuple[int, int, int]:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"`flydsl_hgemm` expects 2D inputs, got a.dim={a.dim()} b.dim={b.dim()}"
        )
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("`flydsl_hgemm` only supports CUDA/ROCm tensors")
    if a.device != b.device:
        raise ValueError(
            f"`a` and `b` must be on the same device, got {a.device=} {b.device=}"
        )
    if a.dtype != b.dtype:
        raise ValueError(
            f"`a` and `b` must have the same dtype, got {a.dtype=} {b.dtype=}"
        )

    m, k = a.shape
    n, bk = b.shape
    if k != bk:
        raise ValueError(
            f"Incompatible GEMM shapes: a={tuple(a.shape)} b={tuple(b.shape)}"
        )

    if out is not None:
        if out.shape != (m, n):
            raise ValueError(f"`out` must have shape {(m, n)}, got {tuple(out.shape)}")
        if out.dtype != a.dtype:
            raise ValueError(
                f"`out` dtype must match input dtype, got {out.dtype=} {a.dtype=}"
            )
        if out.device != a.device:
            raise ValueError(f"`out` must be on {a.device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError("`out` must be contiguous")

    return m, n, k


def _validate_hgemm_tiling(
    m: int,
    n: int,
    k: int,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    pack_n: int,
    split_k: int,
    stages: int,
    block_m_warps: int,
    block_n_warps: int,
) -> None:
    del m

    if tile_k < 32:
        raise ValueError(
            f"Invalid tile_k={tile_k}; latest kernel requires tile_k >= 32"
        )
    if split_k < 1:
        raise ValueError(f"Invalid split_k={split_k}; split_k must be >= 1")
    if stages not in (1, 2):
        raise ValueError(
            f"Invalid stages={stages}; latest kernel only supports stages in {{1, 2}}"
        )
    if pack_n != 1:
        raise ValueError(
            "Latest `hgemm.py` kernel only supports `pack_n=1`; " f"got pack_n={pack_n}"
        )
    if block_m_warps * block_n_warps != 4:
        raise ValueError(
            "Latest `hgemm.py` kernel requires block_m_warps * block_n_warps == 4; "
            f"got {block_m_warps} * {block_n_warps}"
        )

    warp_atom_m = 16
    warp_atom_n = 16

    if tile_m % (block_m_warps * warp_atom_m) != 0:
        raise ValueError(
            f"Invalid tiling: tile_m={tile_m} must be divisible by "
            f"block_m_warps * 16 = {block_m_warps * warp_atom_m}"
        )
    if tile_n % (block_n_warps * warp_atom_n) != 0:
        raise ValueError(
            f"Invalid tiling: tile_n={tile_n} must be divisible by "
            f"block_n_warps * 16 = {block_n_warps * warp_atom_n}"
        )

    block_n = tile_n
    if n < block_n or n % block_n != 0:
        raise ValueError(
            f"Invalid N for this kernel: N={n} must satisfy N >= {block_n} and N % {block_n} == 0"
        )

    if k % split_k != 0:
        raise ValueError(
            f"Invalid split-K: K={k} must be divisible by split_k={split_k}"
        )

    ks = k // split_k
    if ks < tile_k or ks % tile_k != 0:
        raise ValueError(
            f"Invalid K for this kernel: K/split_k={ks} must satisfy "
            f">= tile_k={tile_k} and % tile_k == 0"
        )

    block_threads = block_m_warps * block_n_warps * 64
    ldg_vec_size = 8
    ldg_reg_a_count = (tile_m * tile_k) // ldg_vec_size // block_threads
    ldg_reg_b_count = (tile_n * tile_k) // ldg_vec_size // block_threads
    ldg_reg_c_count = (tile_m * tile_n) // ldg_vec_size // block_threads
    if ldg_reg_a_count < 1 or ldg_reg_b_count < 1:
        raise ValueError(
            "Invalid tile combination: requires at least one vectorized global load per thread "
            f"(got ldg_reg_a_count={ldg_reg_a_count}, ldg_reg_b_count={ldg_reg_b_count})"
        )
    if split_k > 1 and ldg_reg_c_count < 1:
        raise ValueError(
            "Invalid split-K tile combination: requires at least one vectorized C load/store per thread "
            f"(got ldg_reg_c_count={ldg_reg_c_count})"
        )


def _get_split_k_global_semaphore(device: torch.device) -> torch.Tensor:
    semaphore = SPLIT_K_GLOBAL_SEMAPHORE.get(device)
    if semaphore is None:
        semaphore = torch.zeros(
            (SPLIT_K_SIGNAL_STATE_COUNT * SPLIT_K_COUNTER_MAX_LEN,),
            dtype=torch.int32,
            device=device,
        )
        SPLIT_K_GLOBAL_SEMAPHORE[device] = semaphore
        SPLIT_K_GLOBAL_SEMAPHORE_STATE[device] = int(0)
    return semaphore


def _get_split_k_signal_state(device: torch.device) -> int:
    return SPLIT_K_GLOBAL_SEMAPHORE_STATE[device]


def _advance_split_k_signal_state(device: torch.device) -> None:
    SPLIT_K_GLOBAL_SEMAPHORE_STATE[device] = (
        _get_split_k_signal_state(device) + 1
    ) % SPLIT_K_SIGNAL_STATE_COUNT


def _check_split_k_counter_capacity(
    m: int, n: int, tile_m: int, tile_n: int, split_k: int
) -> None:
    if split_k <= 1:
        return
    bm = (m + tile_m - 1) // tile_m
    bn = n // tile_n
    required = bm * bn
    if required > SPLIT_K_COUNTER_MAX_LEN:
        raise ValueError(
            "Split-K counter capacity exceeded: "
            f"requires {required} counters, max supported is {SPLIT_K_COUNTER_MAX_LEN}"
        )


def _compile_flydsl_hgemm(
    dtype: str,
    m: int,
    n: int,
    k: int,
    *,
    tile_k: int = 64,
    block_m_warps: int = 1,
    block_n_warps: int = 4,
    tile_m: int = 128,
    tile_n: int = 128,
    pack_n: int = 1,
    stages: int = 2,
    async_copy: bool = False,
    b_to_lds: bool = False,
    b_preshuffle: bool = True,
    split_k: int = 1,
    c_to_lds: bool = False,
    signal_state: int = 0,
):
    """Compile and cache a FlyDSL HGEMM kernel launcher."""

    if dtype not in {"f16", "bf16"}:
        raise ValueError(f"`dtype` must be 'f16' or 'bf16', got {dtype!r}")
    if b_preshuffle and b_to_lds:
        raise ValueError(
            "Latest `hgemm.py` requires b_to_lds=False when b_preshuffle=True"
        )

    _validate_hgemm_tiling(
        m,
        n,
        k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        pack_n=pack_n,
        split_k=split_k,
        stages=stages,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
    )

    kernel = compile_hgemm_kernel(
        dtype,
        signal_state,
        n,
        k,
        TILE_K=tile_k,
        BLOCK_M_WARPS=block_m_warps,
        BLOCK_N_WARPS=block_n_warps,
        TILE_M=tile_m,
        TILE_N=tile_n,
        STAGES=stages,
        ASYNC_COPY=async_copy,
        B_TO_LDS=b_to_lds,
        B_PRE_SHUFFLE=b_preshuffle,
        SPLIT_K=split_k,
        C_TO_LDS=c_to_lds,
    )

    def launcher(
        out: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        stream=None,
    ):
        runtime_m = int(a.shape[0])
        _check_split_k_counter_capacity(runtime_m, n, tile_m, tile_n, split_k)
        semaphore = _get_split_k_global_semaphore(a.device)
        launch_stream = (
            torch.cuda.current_stream(device=a.device) if stream is None else stream
        )
        return kernel(out, a, b, runtime_m, semaphore, stream=launch_stream)

    return launcher


def flydsl_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    pack_n: int = 1,
    split_k: int = 1,
    block_m_warps: int = 1,
    block_n_warps: int = 4,
    stages: int = 2,
    async_copy: bool = False,
    b_to_lds: bool = False,
    b_preshuffle: bool = True,
    auto_shuffle_b: bool = False,
    c_to_lds: bool = False,
) -> torch.Tensor:
    """Run FlyDSL HGEMM.
    `a` is `(M, K)`.
    `b` is `(N, K)`, optionally pre-shuffled via `shuffle_weight()`.
    Returns `(M, N)`.
    """

    m, n, k = _validate_hgemm_inputs(a, b, out)
    kernel_dtype = _to_kernel_dtype(a.dtype)

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    if b_preshuffle and not getattr(b, "is_shuffled", False):
        if auto_shuffle_b:
            b = shuffle_weight(b, layout=_get_flydsl_shuffle_layout(pack_n))
        else:
            raise ValueError(
                "`b_preshuffle=True` expects `b` to be pre-shuffled. "
                f"Use `shuffle_weight(b, layout={_get_flydsl_shuffle_layout(pack_n)})` "
                "first or pass `auto_shuffle_b=True`."
            )

    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    _get_split_k_global_semaphore(a.device)
    signal_state = _get_split_k_signal_state(a.device)

    launcher = _compile_flydsl_hgemm(
        kernel_dtype,
        m,
        n,
        k,
        tile_k=tile_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        tile_m=tile_m,
        tile_n=tile_n,
        pack_n=pack_n,
        stages=stages,
        async_copy=async_copy,
        b_to_lds=b_to_lds,
        b_preshuffle=b_preshuffle,
        split_k=split_k,
        c_to_lds=c_to_lds,
        signal_state=signal_state,
    )

    launcher(out, a, b)
    if split_k > 1:
        _advance_split_k_signal_state(a.device)
    return out


# ---------------------------------------------------------------------------
# FlyDSL preshuffle GEMM kernel management
# ---------------------------------------------------------------------------

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

    cache_key = (
        m,
        n,
        k,
        in_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        lds_stage,
        use_cshuffle_epilog,
        use_async_copy,
        wpe,
    )
    if cache_key not in _flydsl_kernel_cache:
        try:
            exe = compile_fn(
                M=m,
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
        m,
        n,
        torch.cuda.current_stream(),
    )

    return Out
