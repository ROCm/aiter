# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 A6W4 GEMM.

Activations use the existing Hadamard-rotated MXFP6 packing contract. Weights
use the same per-1x32 rotation and E8M0 scale layout, but store E2M1 values in a
compact 16-byte-per-lane C0 tile.
"""

import functools
import os

import torch
from torch import Tensor

from aiter import logger

from ..jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG, compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..utility.mx_types import MX_DEFAULT_ROUND_MODE
from .gemm_op_a6w6 import (
    _K_TILE,
    _PADK,
    _SCALE_GROUP_SIZE,
    _TILE,
    _ceil,
    _rotate_k32_torch,
    pack_scale_torch,
)
from .gemm_op_mixed_mxfp import (
    _find_mixed_mxfp_config,
    _get_device_gfx_cu,
    _load_mixed_mxfp_configs,
)
from .quant import quant_mxfp4_hip

_MFMA32_SMALL_KERNEL = "f6f4gemm_m32_s0_a4_nt_kernel_func"
_MFMA32_SWZ0_KERNEL = "f6f4gemm_m32_s0_a4_t_kernel_func"
_MFMA32_GROUPED_KERNEL = "f6f4gemm_m32_s3_a4_t_kernel_func"
_MFMA32_LONG_K_KERNEL = "f6f4gemm_m32_s3_a5_t_kernel_func"
_PACKED_W_TILE_BYTES = 16384
_SCALE_TILE_BYTES = 1024
_MAX_BUFFER_BYTES = 1 << 31
_MAX_KERNEL_K = (1 << 31) - 1
_GROUPED_SWIZZLE_MAX_M = 131072
_GROUPED_SWIZZLE_MAX_N = 16384
_GROUPED_SWIZZLE_MAX_K = 6144


@compile_ops("module_quant", fc_name="quant_mxfp4_gemm_hip_out", develop=True)
def quant_mxfp4_gemm_hip_out(
    input: Tensor,
    packed: Tensor,
    packed_scale: Tensor,
    round_mode: int = MX_DEFAULT_ROUND_MODE,
) -> None: ...


_native_quant_mxfp4_gemm_hip_out = quant_mxfp4_gemm_hip_out


def quant_mxfp4_gemm_hip_out(
    input: Tensor,
    packed: Tensor,
    packed_scale: Tensor,
    round_mode: int = MX_DEFAULT_ROUND_MODE,
) -> None:
    """Launch the fused packer on the input tensor's current HIP stream."""
    if input.ndim != 2:
        raise ValueError(
            f"quant_mxfp4_gemm_hip_out expects a 2D input, got {input.ndim}D"
        )
    if input.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError("quant_mxfp4_gemm_hip_out requires a floating input")
    if not input.is_cuda or not packed.is_cuda or not packed_scale.is_cuda:
        raise ValueError("quant_mxfp4_gemm_hip_out requires GPU tensors")
    if input.device != packed.device or input.device != packed_scale.device:
        raise ValueError("quant_mxfp4_gemm_hip_out requires tensors on one GPU")
    if not input.is_contiguous():
        raise ValueError("quant_mxfp4_gemm_hip_out requires a contiguous input")
    if packed.dtype != torch.uint8 or packed_scale.dtype != torch.uint8:
        raise ValueError("quant_mxfp4_gemm_hip_out requires uint8 output buffers")
    if not packed.is_contiguous() or not packed_scale.is_contiguous():
        raise ValueError("quant_mxfp4_gemm_hip_out requires contiguous outputs")
    _, _, expected_packed, expected_scale = _validate_mxfp4_gemm_input(input)
    if packed.numel() != expected_packed or packed_scale.numel() != expected_scale:
        raise ValueError(
            "quant_mxfp4_gemm_hip_out buffers have wrong size: "
            f"got ({packed.numel()}, {packed_scale.numel()}), "
            f"expected ({expected_packed}, {expected_scale})"
        )
    if not torch.compiler.is_compiling() and (
        packed.data_ptr() % 16 or packed_scale.data_ptr() % 16
    ):
        raise ValueError("quant_mxfp4_gemm_hip_out requires 16-byte-aligned outputs")
    round_mode_int = _normalize_round_mode(round_mode)
    with torch.cuda.device(input.device):
        _native_quant_mxfp4_gemm_hip_out(input, packed, packed_scale, round_mode_int)


def _default_gemm_a6w4_kernel(M: int, N: int, K: int) -> str:
    """Choose a safe kernel when no shape-tuned A6W4 record is available."""
    padM, padN, padK = _ceil(M, _TILE), _ceil(N, _TILE), _ceil(K, _K_TILE)
    grouped_grid_in_bounds = (
        padM <= _GROUPED_SWIZZLE_MAX_M and padN <= _GROUPED_SWIZZLE_MAX_N
    )
    if M < 2048:
        return _MFMA32_SMALL_KERNEL
    if K > N and (padK > _GROUPED_SWIZZLE_MAX_K or grouped_grid_in_bounds):
        return _MFMA32_LONG_K_KERNEL
    if padK <= _GROUPED_SWIZZLE_MAX_K and grouped_grid_in_bounds:
        return _MFMA32_GROUPED_KERNEL
    return _MFMA32_SWZ0_KERNEL


@functools.lru_cache(maxsize=1024)
def _get_GEMM_A6W4_config_cached(
    M: int,
    N: int,
    K: int,
    tuned_file: str,
    gfx: str,
    cu_num: int,
) -> dict[str, object] | None:
    configs = _load_mixed_mxfp_configs(tuned_file, "A6W4")
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
                "A6W4 shape M:%s N:%s K:%s matched %s config "
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
            "A6W4 shape M:%s N:%s K:%s has no tuned config in %s; "
            "using the safe default kernel",
            M,
            N,
            K,
            tuned_file,
        )
    return None


def get_GEMM_A6W4_config(
    M: int,
    N: int,
    K: int,
    tuned_file: str | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, object] | None:
    """Return an exact or physical-shape A6W4 tuning record."""
    tuned_file = os.path.abspath(
        tuned_file or AITER_CONFIGS.AITER_CONFIG_GEMM_A6W4_ASM_FILE
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
    return _get_GEMM_A6W4_config_cached(M, N, K, tuned_file, gfx, cu_num)


def clear_gemm_a6w4_config_cache() -> None:
    """Clear cached A6W4 tuning data after a tuner updates its CSV."""
    _load_mixed_mxfp_configs.cache_clear()
    _get_GEMM_A6W4_config_cached.cache_clear()
    _get_device_gfx_cu.cache_clear()


@torch.compiler.assume_constant_result
def _compiled_gemm_a6w4_configs(
    device_index: int,
) -> tuple[tuple[int, int, int, str], ...]:
    """Snapshot current-device tuning rows as compile-time constants."""
    tuned_file = os.path.abspath(AITER_CONFIGS.AITER_CONFIG_GEMM_A6W4_ASM_FILE)
    configs = _load_mixed_mxfp_configs(tuned_file, "A6W4")
    try:
        gfx, cu_num = _get_device_gfx_cu(device_index)
    except (AssertionError, IndexError, KeyError, RuntimeError):
        return ()
    return tuple(
        (M, N, K, str(config["kernelName"]))
        for (config_gfx, config_cu, M, N, K), config in configs.items()
        if config_gfx == gfx and config_cu == cu_num
    )


def _select_gemm_a6w4_kernel(
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
        configs = _compiled_gemm_a6w4_configs(device_index)
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
        return _default_gemm_a6w4_kernel(M, N, K)
    config = get_GEMM_A6W4_config(M, N, K, device=device)
    if config is not None:
        return str(config["kernelName"])
    return _default_gemm_a6w4_kernel(M, N, K)


def mxfp4_gemm_pack_size(rows: int, K: int) -> tuple[int, int]:
    """Return compact MXFP4 operand and packed-scale byte counts."""
    if rows <= 0 or K <= 0:
        raise ValueError(
            f"mxfp4_gemm_pack_size requires positive dimensions, got {(rows, K)}"
        )
    padR, padK = _ceil(rows, _TILE), _ceil(K, _K_TILE)
    nt = padR // _TILE
    nk_pad = padK // _K_TILE + _PADK
    sizes = (
        nt * nk_pad * _PACKED_W_TILE_BYTES,
        nt * nk_pad * _SCALE_TILE_BYTES,
    )
    if max(sizes) > _MAX_BUFFER_BYTES:
        raise ValueError("mxfp4_gemm_pack_size exceeds the kernel's 2 GiB range")
    return sizes


def _pack_mxfp4_gemm_torch(packed_codes: Tensor, padK: int = _PADK) -> Tensor:
    """Re-tile natural [R, K/2] E2M1 pairs into the kernel's compact C0 blob."""
    raw = packed_codes.contiguous().view(torch.uint8)
    if raw.ndim != 2:
        raise ValueError(
            f"_pack_mxfp4_gemm_torch expects a 2D [rows, K/2] tensor, got {raw.ndim}D"
        )
    rows, packed_k = raw.shape
    K = packed_k * 2
    if rows % _TILE or K % _K_TILE:
        raise ValueError(
            f"_pack_mxfp4_gemm_torch requires rows%{_TILE}=0 and K%{_K_TILE}=0, "
            f"got rows={rows}, K={K}"
        )

    device = raw.device
    nt, nk = rows // _TILE, K // _K_TILE
    rb = torch.arange(16, device=device).repeat_interleave(64)
    lane = torch.arange(64, device=device).repeat(16)
    r16 = lane % 16
    kg = lane // 16
    local_row = rb * 16 + r16

    tile = torch.arange(nt, device=device).view(nt, 1, 1)
    step = torch.arange(nk, device=device).view(1, nk, 1)
    row = tile * _TILE + local_row.view(1, 1, _SCALE_TILE_BYTES)
    byte_col = step * (_K_TILE // 2) + (kg * (_SCALE_GROUP_SIZE // 2)).view(
        1, 1, _SCALE_TILE_BYTES
    )
    row, byte_col = torch.broadcast_tensors(row, byte_col)
    byte = torch.arange(_SCALE_GROUP_SIZE // 2, device=device)
    blocks = raw[row.unsqueeze(-1), byte_col.unsqueeze(-1) + byte]

    out = torch.zeros(
        (nt, nk + padK, _PACKED_W_TILE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    out[:, :nk, :] = blocks.reshape(nt, nk, _PACKED_W_TILE_BYTES)
    return out.reshape(-1).contiguous()


def _validate_mxfp4_gemm_input(
    operand: Tensor,
) -> tuple[int, int, int, int]:
    if operand.ndim != 2:
        raise ValueError(
            f"quant_mxfp4_gemm expects a 2D [rows, K] tensor, got {operand.ndim}D"
        )
    if operand.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError(
            f"quant_mxfp4_gemm requires a floating input, got {operand.dtype}"
        )
    if not operand.is_cuda:
        raise ValueError("quant_mxfp4_gemm requires a CUDA/HIP tensor")
    rows, K = operand.shape
    if rows <= 0 or K <= 0:
        raise ValueError(
            f"quant_mxfp4_gemm requires positive dimensions, got {(rows, K)}"
        )
    packed_bytes, scale_bytes = mxfp4_gemm_pack_size(rows, K)
    return rows, K, packed_bytes, scale_bytes


def _normalize_round_mode(round_mode: int) -> int:
    try:
        round_mode_int = int(round_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"quant_mxfp4_gemm expected round_mode in {{0,1,2,3}}, got {round_mode!r}"
        ) from error
    if round_mode_int != round_mode:
        raise ValueError(
            f"quant_mxfp4_gemm expected an integral round_mode, got {round_mode!r}"
        )
    if round_mode_int not in range(4):
        raise ValueError(
            f"quant_mxfp4_gemm expected round_mode in {{0,1,2,3}}, got {round_mode!r}"
        )
    return round_mode_int


def _quant_mxfp4_gemm_torch_legacy(
    operand: Tensor,
    round_mode: int = MX_DEFAULT_ROUND_MODE,
) -> tuple[Tensor, Tensor]:
    """Legacy padded torch rotate/quantize/repack path for differential tests."""
    rows, K, _, _ = _validate_mxfp4_gemm_input(operand)
    round_mode_int = _normalize_round_mode(round_mode)

    padR, padK = _ceil(rows, _TILE), _ceil(K, _K_TILE)
    padded = torch.nn.functional.pad(operand.detach(), (0, padK - K, 0, padR - rows))
    # Match quant_mxfp6_gemm's fused BF16 Hadamard transform. Applying the same
    # orthonormal block rotation to A and W preserves the underlying GEMM.
    rotated = _rotate_k32_torch(padded).to(torch.bfloat16)
    codes, scales = quant_mxfp4_hip(
        rotated,
        round_mode=round_mode_int,
    )
    packed = _pack_mxfp4_gemm_torch(codes)
    packed_scale = pack_scale_torch(scales.view(torch.uint8), padR)
    return packed, packed_scale


def quant_mxfp4_gemm(
    operand: Tensor,
    round_mode: int = MX_DEFAULT_ROUND_MODE,
) -> tuple[Tensor, Tensor]:
    """Fused H32 + MXFP4 quantize + compact GEMM pack for gfx950.

    The physical layout includes two trailing K tiles used only as addressable
    prefetch spacing. Their contents are intentionally unspecified: every
    mixed ASM kernel bounds accumulation by the padded logical K and never
    accumulates either guard tile.
    """
    _, _, packed_bytes, scale_bytes = _validate_mxfp4_gemm_input(operand)
    round_mode_int = _normalize_round_mode(round_mode)
    operand = operand.detach().contiguous()
    packed = torch.empty(packed_bytes, dtype=torch.uint8, device=operand.device)
    packed_scale = torch.empty(scale_bytes, dtype=torch.uint8, device=operand.device)
    quant_mxfp4_gemm_hip_out(
        operand,
        packed,
        packed_scale,
        round_mode_int,
    )
    return packed, packed_scale


@compile_ops(
    "module_gemm_a6w4_asm",
    fc_name="gemm_a6w4_asm",
    ffi_type="ctypes",
)
def _gemm_a6w4_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernelName: str,
    alpha: float,
) -> None: ...


def gemm_a6w4_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernelName: str | None = None,
    alpha: float = 1.0,
) -> Tensor:
    """Launch A6W4 on physical GEMM-layout buffers.

    ``out`` dimensions must be multiples of 256, ``K`` must be the padded
    multiple of 128, and all packed buffers must exactly match those physical
    dimensions. Prefer :func:`gemm_a6w4` when starting from logical shapes.
    """
    if float(alpha) != 1.0:
        raise ValueError("gemm_a6w4 currently supports only alpha=1.0")
    if K <= 0 or K > _MAX_KERNEL_K:
        raise ValueError(f"gemm_a6w4 K is outside the int32 kernel ABI: {K}")
    if out.ndim != 2:
        raise ValueError(f"gemm_a6w4_asm expects a 2D output, got {out.ndim}D")
    kernelName = kernelName or _default_gemm_a6w4_kernel(*out.shape, K)
    _gemm_a6w4_asm(A, B, A_scale, B_scale, out, K, kernelName, alpha)
    return out


def gemm_a6w4(
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
    """Run MXFP6 activations by MXFP4 weights and return logical ``[M, N]``.

    The returned slice is non-contiguous when ``N`` requires physical padding.
    """
    if dtype != torch.bfloat16:
        raise ValueError(f"gemm_a6w4 only supports torch.bfloat16, got {dtype}")
    if float(alpha) != 1.0:
        raise ValueError("gemm_a6w4 currently supports only alpha=1.0")
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"gemm_a6w4 requires positive dimensions, got {(M, N, K)}")
    if K > _MAX_KERNEL_K:
        raise ValueError(f"gemm_a6w4 K is outside the int32 kernel ABI: {K}")
    selected_kernel = _select_gemm_a6w4_kernel(M, N, K, kernelName, device=A.device)
    padM, padN, padK = _ceil(M, _TILE), _ceil(N, _TILE), _ceil(K, _K_TILE)
    if padM * padN * torch.bfloat16.itemsize > _MAX_BUFFER_BYTES:
        raise ValueError("gemm_a6w4 output exceeds the kernel's 2 GiB address range")
    out = torch.empty((padM, padN), dtype=dtype, device=A.device)
    gemm_a6w4_asm(
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
    "clear_gemm_a6w4_config_cache",
    "gemm_a6w4",
    "gemm_a6w4_asm",
    "get_GEMM_A6W4_config",
    "mxfp4_gemm_pack_size",
    "quant_mxfp4_gemm",
    "quant_mxfp4_gemm_hip_out",
]
