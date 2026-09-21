# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 A4W6 GEMM."""

import functools
import os

import torch
from torch import Tensor

from aiter import logger

from ..jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG, compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from .gemm_op_a6w6 import _K_TILE, _TILE, _ceil
from .gemm_op_mixed_mxfp import (
    _find_mixed_mxfp_config,
    _get_device_gfx_cu,
    _load_mixed_mxfp_configs,
)

_MFMA32_SMALL_KERNEL = "_ZN5aiter40f4f6gemm_bf16_per1x32Fp4Fp6_m32_s0_a5_ntE"
_MFMA32_SWZ0_KERNEL = "_ZN5aiter39f4f6gemm_bf16_per1x32Fp4Fp6_m32_s0_a4_tE"
_MFMA32_GROUPED_KERNEL = "_ZN5aiter39f4f6gemm_bf16_per1x32Fp4Fp6_m32_s3_a5_tE"
_MFMA32_LONG_K_KERNEL = "_ZN5aiter39f4f6gemm_bf16_per1x32Fp4Fp6_m32_s3_a6_tE"
_MAX_BUFFER_BYTES = 1 << 31
_MAX_KERNEL_K = (1 << 31) - 1
_GROUPED_SWIZZLE_MAX_M = 131072
_GROUPED_SWIZZLE_MAX_N = 16384
_GROUPED_SWIZZLE_MAX_K = 6144


def _default_gemm_a4w6_kernel(M: int, N: int, K: int) -> str:
    """Choose a safe kernel when no shape-tuned A4W6 record is available."""
    padM, padN, padK = _ceil(M, _TILE), _ceil(N, _TILE), _ceil(K, _K_TILE)
    grouped_grid_in_bounds = (
        padM <= _GROUPED_SWIZZLE_MAX_M and padN <= _GROUPED_SWIZZLE_MAX_N
    )
    if M < 2048:
        return _MFMA32_SMALL_KERNEL
    if K > N and (padK > _GROUPED_SWIZZLE_MAX_K or grouped_grid_in_bounds):
        return _MFMA32_LONG_K_KERNEL
    if N > K and padK <= _GROUPED_SWIZZLE_MAX_K and grouped_grid_in_bounds:
        return _MFMA32_GROUPED_KERNEL
    return _MFMA32_SWZ0_KERNEL


@functools.lru_cache(maxsize=1024)
def _get_GEMM_A4W6_config_cached(
    M: int,
    N: int,
    K: int,
    tuned_file: str,
    gfx: str,
    cu_num: int,
) -> dict[str, object] | None:
    configs = _load_mixed_mxfp_configs(tuned_file, "A4W6")
    padM, padN, padK = _ceil(M, _TILE), _ceil(N, _TILE), _ceil(K, _K_TILE)
    match = _find_mixed_mxfp_config(
        configs,
        gfx,
        cu_num,
        M,
        N,
        K,
        padM,
        padN,
        padK,
    )
    if match is not None:
        config, match_kind, matched_shape = match
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                "A4W6 shape M:%s N:%s K:%s matched %s config "
                "M:%s N:%s K:%s in %s: %s",
                M,
                N,
                K,
                match_kind,
                *matched_shape,
                tuned_file,
                config["kernelName"],
            )
        return config
    if AITER_LOG_TUNED_CONFIG:
        logger.info(
            "A4W6 shape M:%s N:%s K:%s has no tuned config in %s; "
            "using the safe default kernel",
            M,
            N,
            K,
            tuned_file,
        )
    return None


def get_GEMM_A4W6_config(
    M: int,
    N: int,
    K: int,
    tuned_file: str | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, object] | None:
    """Return an exact or physical-shape A4W6 tuning record."""
    tuned_file = os.path.abspath(
        tuned_file or AITER_CONFIGS.AITER_CONFIG_GEMM_A4W6_ASM_FILE
    )
    try:
        if device is None:
            gfx, cu_num = get_gfx(), get_cu_num()
        else:
            device_index = (
                torch.cuda.current_device() if device.index is None else device.index
            )
            gfx, cu_num = _get_device_gfx_cu(device_index)
    except (AssertionError, IndexError, KeyError, RuntimeError):
        return None
    return _get_GEMM_A4W6_config_cached(M, N, K, tuned_file, gfx, cu_num)


def clear_gemm_a4w6_config_cache() -> None:
    """Clear cached A4W6 tuning data after a tuner updates its CSV."""
    _load_mixed_mxfp_configs.cache_clear()
    _get_GEMM_A4W6_config_cached.cache_clear()
    _get_device_gfx_cu.cache_clear()


@torch.compiler.assume_constant_result
def _compiled_gemm_a4w6_configs(
    device_index: int,
) -> tuple[tuple[int, int, int, str], ...]:
    """Snapshot current-device tuning rows as compile-time constants."""
    tuned_file = os.path.abspath(AITER_CONFIGS.AITER_CONFIG_GEMM_A4W6_ASM_FILE)
    configs = _load_mixed_mxfp_configs(tuned_file, "A4W6")
    try:
        gfx, cu_num = _get_device_gfx_cu(device_index)
    except (AssertionError, IndexError, KeyError, RuntimeError):
        return ()
    return tuple(
        (M, N, K, str(config["kernelName"]))
        for (config_gfx, config_cu, M, N, K), config in configs.items()
        if config_gfx == gfx and config_cu == cu_num
    )


def _select_gemm_a4w6_kernel(
    M: int,
    N: int,
    K: int,
    kernelName: str | None,
    device: torch.device | None = None,
) -> str:
    if kernelName:
        return kernelName
    if torch.compiler.is_compiling():
        device_index = (
            torch.cuda.current_device()
            if device is None or device.index is None
            else device.index
        )
        configs = _compiled_gemm_a4w6_configs(device_index)
        for tuned_M, tuned_N, tuned_K, tuned_kernel in configs:
            if M == tuned_M and N == tuned_N and K == tuned_K:
                return tuned_kernel
        padM, padN, padK = (
            _ceil(M, _TILE),
            _ceil(N, _TILE),
            _ceil(K, _K_TILE),
        )
        for tuned_M, tuned_N, tuned_K, tuned_kernel in configs:
            if padM == tuned_M and padN == tuned_N and padK == tuned_K:
                return tuned_kernel
        return _default_gemm_a4w6_kernel(M, N, K)
    config = get_GEMM_A4W6_config(M, N, K, device=device)
    if config is not None:
        return str(config["kernelName"])
    return _default_gemm_a4w6_kernel(M, N, K)


@compile_ops(
    "module_gemm_a4w6_asm",
    fc_name="gemm_a4w6_asm",
    ffi_type="ctypes",
)
def _gemm_a4w6_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernelName: str,
    alpha: float,
) -> None: ...


def gemm_a4w6_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernelName: str | None = None,
    alpha: float = 1.0,
) -> Tensor:
    """Launch A4W6 on physical GEMM-layout buffers.

    ``out`` dimensions must be multiples of 256, ``K`` must be the padded
    multiple of 128, and all packed buffers must exactly match those physical
    dimensions. Prefer :func:`gemm_a4w6` when starting from logical shapes.
    """
    if float(alpha) != 1.0:
        raise ValueError("gemm_a4w6 currently supports only alpha=1.0")
    if K <= 0 or K > _MAX_KERNEL_K:
        raise ValueError(f"gemm_a4w6 K is outside the int32 kernel ABI: {K}")
    if out.ndim != 2:
        raise ValueError(f"gemm_a4w6_asm expects a 2D output, got {out.ndim}D")
    kernelName = kernelName or _default_gemm_a4w6_kernel(*out.shape, K)
    _gemm_a4w6_asm(A, B, A_scale, B_scale, out, K, kernelName, alpha)
    return out


def gemm_a4w6(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
    alpha: float = 1.0,
    kernelName: str | None = None,
) -> Tensor:
    """Run MXFP4 activations by MXFP6 weights and return logical ``[M, N]``.

    The returned slice is non-contiguous when ``N`` requires physical padding.
    """
    if dtype != torch.bfloat16:
        raise ValueError(f"gemm_a4w6 only supports torch.bfloat16, got {dtype}")
    if float(alpha) != 1.0:
        raise ValueError("gemm_a4w6 currently supports only alpha=1.0")
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"gemm_a4w6 requires positive dimensions, got {(M, N, K)}")
    if K > _MAX_KERNEL_K:
        raise ValueError(f"gemm_a4w6 K is outside the int32 kernel ABI: {K}")
    padK = _ceil(K, _K_TILE)
    if padK > _MAX_KERNEL_K:
        raise ValueError(f"gemm_a4w6 padded K is outside the int32 kernel ABI: {padK}")
    selected_kernel = _select_gemm_a4w6_kernel(M, N, K, kernelName, device=A.device)
    padM, padN = _ceil(M, _TILE), _ceil(N, _TILE)
    if padM * padN * torch.bfloat16.itemsize > _MAX_BUFFER_BYTES:
        raise ValueError("gemm_a4w6 output exceeds the kernel's 2 GiB address range")
    out = torch.empty((padM, padN), dtype=dtype, device=A.device)
    gemm_a4w6_asm(
        A,
        B,
        A_scale,
        B_scale,
        out,
        padK,
        kernelName=selected_kernel,
        alpha=alpha,
    )
    return out[:M, :N]


__all__ = [
    "clear_gemm_a4w6_config_cache",
    "gemm_a4w6",
    "gemm_a4w6_asm",
    "get_GEMM_A4W6_config",
]
