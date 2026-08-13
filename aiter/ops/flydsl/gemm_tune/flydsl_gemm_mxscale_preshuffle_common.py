# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune catalog for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM."""

from __future__ import annotations

import re
from dataclasses import dataclass

_DTYPE_SHORT = {"fp8": "F8", "fp6": "F6", "fp4": "F4", "bf16": "B16", "fp16": "F16"}
_SHORT_DTYPE = {value: key for key, value in _DTYPE_SHORT.items()}

_COMBOS = [("fp4", "fp4"), ("fp6", "fp4"), ("fp8", "fp8")]
_TILE_M = (32, 64, 96, 128, 256)
_TILE_N = (16, 32, 64, 128, 256, 512)
_TILE_K = (128, 256)
_WAVES_PER_EU = (0, 1, 2, 3, 4)
_XCD_SWIZZLE = (0, 4)
MAX_SPLIT_K = 32
_SPLIT_K = tuple(range(1, MAX_SPLIT_K + 1))
_SPLITK_MAX_TMP_BYTES = 1 << 32
_GFX950_CU_NUM = 256


def _a_row_bytes(a_dtype: str, tile_k: int) -> int:
    """A bytes per row in a K tile."""
    return tile_k // 2 if a_dtype == "fp4" else tile_k


@dataclass
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    a_dtype: str
    b_dtype: str
    out_dtype: str
    waves_per_eu: int
    xcd_swizzle: int = 0
    split_k: int = 1

    @property
    def name(self) -> str:
        a = _DTYPE_SHORT[self.a_dtype]
        b = _DTYPE_SHORT[self.b_dtype]
        out = _DTYPE_SHORT[self.out_dtype]
        return (
            f"flydsl_mxpsh_{self.tile_m}x{self.tile_n}x{self.tile_k}"
            f"_{a}_{b}_{out}_w{self.waves_per_eu}_x{self.xcd_swizzle}_sk{self.split_k}"
        )


_NAME_RE = re.compile(
    r"^flydsl_mxpsh_(\d+)x(\d+)x(\d+)_([A-Z0-9]+)_([A-Z0-9]+)_([A-Z0-9]+)_w(\d+)"
    r"(?:_x(\d+))?(?:_sk(\d+))?$"
)


def parse_kernel_name(name: str):
    """Parse one mxscale-preshuffle kernel name."""
    match = _NAME_RE.match(name.strip())
    if not match:
        return None
    tile_m, tile_n, tile_k, a, b, out, wpe, xcd, split_k = match.groups()
    try:
        return {
            "tile_m": int(tile_m),
            "tile_n": int(tile_n),
            "tile_k": int(tile_k),
            "a_dtype": _SHORT_DTYPE[a],
            "b_dtype": _SHORT_DTYPE[b],
            "out_dtype": _SHORT_DTYPE[out],
            "waves_per_eu": int(wpe),
            "xcd_swizzle": int(xcd) if xcd is not None else 0,
            "split_k": int(split_k) if split_k is not None else 1,
        }
    except KeyError:
        return None


def estimated_lds_bytes(instance: kernelInstance) -> int:
    """Double-buffered A-tile LDS footprint."""
    return 2 * instance.tile_m * _a_row_bytes(instance.a_dtype, instance.tile_k)


def _max_lds_bytes() -> int:
    try:
        from aiter.ops.flydsl.utils import get_shared_memory_per_block

        return int(get_shared_memory_per_block(fallback_gfx="gfx950"))
    except Exception:  # noqa: BLE001 - conservative fallback without a device
        return 160 * 1024


def instance_valid(instance: kernelInstance) -> bool:
    """Check shape-independent kernel constraints."""
    if instance.tile_k not in (128, 256):
        return False
    if instance.tile_m % 32 != 0 or instance.tile_n % 16 != 0:
        return False
    num_waves = min(4, instance.tile_n // 16)
    if (instance.tile_n // num_waves) % 16 != 0:
        return False
    a_row_bytes = _a_row_bytes(instance.a_dtype, instance.tile_k)
    if (instance.tile_m * a_row_bytes) % 4096 != 0:
        return False
    return estimated_lds_bytes(instance) <= _max_lds_bytes()


def fits_shape(instance: kernelInstance, M: int, N: int, K: int) -> bool:
    """Check shape-specific tile and split-K constraints."""
    if K % 128 != 0:
        return False
    if N % instance.tile_n != 0 or K % instance.tile_k != 0:
        return False
    if instance.split_k > 1:
        k_per_split = K // instance.split_k
        if (
            K % instance.split_k != 0
            or k_per_split % instance.tile_k != 0
            or k_per_split % 256 != 0
        ):
            return False
        if M * N * instance.split_k * 4 >= _SPLITK_MAX_TMP_BYTES:
            return False
        base_workgroups = ((M + instance.tile_m - 1) // instance.tile_m) * (
            N // instance.tile_n
        )
        if base_workgroups >= _GFX950_CU_NUM:
            return False
    return True


def _build_kernels_list():
    out = {}
    index = 0
    for a_dtype, b_dtype in _COMBOS:
        for tile_m in _TILE_M:
            for tile_n in _TILE_N:
                for tile_k in _TILE_K:
                    for waves_per_eu in _WAVES_PER_EU:
                        for xcd_swizzle in _XCD_SWIZZLE:
                            for split_k in _SPLIT_K:
                                instance = kernelInstance(
                                    tile_m,
                                    tile_n,
                                    tile_k,
                                    a_dtype,
                                    b_dtype,
                                    "bf16",
                                    waves_per_eu,
                                    xcd_swizzle,
                                    split_k,
                                )
                                if instance_valid(instance):
                                    out[index] = instance
                                    index += 1
    return out


kernels_list = _build_kernels_list()


def candidates_for(a_dtype: str, b_dtype: str, M: int, N: int, K: int):
    """Return catalog entries matching one exact operation signature."""
    return [
        (index, instance)
        for index, instance in kernels_list.items()
        if instance.a_dtype == a_dtype
        and instance.b_dtype == b_dtype
        and fits_shape(instance, M, N, K)
    ]
