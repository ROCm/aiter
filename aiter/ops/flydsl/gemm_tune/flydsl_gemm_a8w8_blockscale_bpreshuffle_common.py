# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune metadata for the FlyDSL fp8 blockscale preshuffle-B GEMM kernel.

Sister of ``flydsl_gemm_a8w8_bpreshuffle_common.py``, but for the blockscale
variant (``preshuffle_gemm_blockscaled``). Only the ``fp32_post_mfma`` scale
format is CK-equivalent numerically, so it's the only scale_format wired here.

The underlying kernel is gfx950-only.
"""
from dataclasses import dataclass
import os

from aiter.ops.flydsl.utils import (
    addressable_lds_bytes_for_gfx as _addressable_lds_bytes_for_gfx,
    get_shared_memory_per_block,
)


def get_gfx():
    """Detect GPU arch: honour GPU_ARCHS env, fall back to chip_info, default gfx942."""
    env = os.environ.get("GPU_ARCHS", "")
    if env and env != "native":
        return env.split(";")[-1].strip()
    try:
        from aiter.jit.utils.chip_info import get_gfx as _get_gfx

        return _get_gfx()
    except Exception:
        return "gfx942"


_DTYPE_SHORT = {
    "fp8": "F8",
    "bf16": "B16",
    "fp16": "F16",
}

_SCALE_FORMAT_SHORT = {
    "ue8m0": "ue8m0",
    "fp32": "fp32",
    "fp32_post_mfma": "postmfma",
}


@dataclass
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    block_swizzle_n: int
    q_dtype_a: str  # "fp8"
    q_dtype_w: str  # "fp8"
    dtype: str  # output dtype: "bf16"
    scale_format: str = "fp32_post_mfma"

    @property
    def name(self) -> str:
        qa = _DTYPE_SHORT.get(self.q_dtype_a, self.q_dtype_a.upper())
        qw = _DTYPE_SHORT.get(self.q_dtype_w, self.q_dtype_w.upper())
        dt = _DTYPE_SHORT.get(self.dtype, self.dtype.upper())
        sf = _SCALE_FORMAT_SHORT.get(self.scale_format, self.scale_format)
        return "_".join(
            [
                "flydsl",
                "blockscale",
                "bpreshuffle",
                "x".join(map(str, [self.tile_m, self.tile_n, self.tile_k])),
                f"bsw{self.block_swizzle_n}",
                sf,
                qa,
                qw,
                dt,
            ]
        )


def _ki(
    tile_m,
    tile_n,
    tile_k,
    block_swizzle_n=0,
    scale_format="fp32_post_mfma",
    q_dtype_a="fp8",
    q_dtype_w="fp8",
    dtype="bf16",
):
    return kernelInstance(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_swizzle_n=block_swizzle_n,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        dtype=dtype,
        scale_format=scale_format,
    )


# ---------------------------------------------------------------------------
# LDS estimate
# ---------------------------------------------------------------------------
def _smem_align(ptr: int, align: int = 16) -> int:
    if ptr % align == 0:
        return ptr
    return (ptr + align - 1) // align * align


# Assumed K for the LDS-sizing upper bound (kernels_list entries don't carry K
# — the tuner picks per-shape, but we need a conservative-but-realistic value
# so we don't skip valid tiles. K=2048 → K/128 = 16 sa scales per row, which
# matches the canonical test shape.
_ASSUMED_K_FOR_SA_ESTIMATE = 2048
_FP32_BYTES = 4
_FP8_BYTES = 1


def kernel_instance_estimated_lds_bytes(ki: kernelInstance) -> int:
    """Estimate LDS usage for a fp32_post_mfma preshuffle blockscale kernel.

    LDS layout per the kernel:
      * A: ping/pong fp8 slab — 2 * tile_m * tile_k bytes (ping + pong)
      * sa: fp32 slab — tile_m * (K/128) * 4 bytes (single-buffered, post-mfma)
      * sb: held in registers (no LDS)
    """
    a_bytes = 2 * int(ki.tile_m) * int(ki.tile_k) * _FP8_BYTES
    sa_per_row = _ASSUMED_K_FOR_SA_ESTIMATE // 128
    sa_bytes = int(ki.tile_m) * sa_per_row * _FP32_BYTES
    return _smem_align(a_bytes, 128) + _smem_align(sa_bytes, 128)


def addressable_lds_bytes_for_gfx(gfx: str) -> int:
    return _addressable_lds_bytes_for_gfx(gfx)


def max_lds_bytes_for_tune() -> int:
    """Addressable LDS limit for the current target."""
    return get_shared_memory_per_block(fallback_gfx=get_gfx())


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------
# (tile_m, tile_n, tile_k) grid that we've benched well on gfx950.
_TILE_GRID = [
    (tm, tn, tk)
    for tm in (64, 128, 256)
    for tn in (128, 256)
    for tk in (128, 256)
]
_BSW_VALS = (0, 1, 2, 3, 4, 8)
_SCALE_FORMATS = ("fp32_post_mfma",)


def _build_kernels_list():
    kl = {}
    idx = 0
    for sf in _SCALE_FORMATS:
        for bsw in _BSW_VALS:
            for tm, tn, tk in _TILE_GRID:
                kl[idx] = _ki(tm, tn, tk, block_swizzle_n=bsw, scale_format=sf)
                idx += 1
    return kl


kernels_list_950 = _build_kernels_list()

# Defaults — small dict of robust picks for fallback.
default_kernels_dict_950 = {
    (-1): _ki(128, 128, 128, block_swizzle_n=2),
    (-2): _ki(64, 256, 256, block_swizzle_n=4),
    (-3): _ki(128, 256, 128, block_swizzle_n=2),
    (-4): _ki(256, 256, 128, block_swizzle_n=0),
    (-5): _ki(128, 128, 256, block_swizzle_n=1),
}

# Architecture gate: the kernel is gfx950-only. On other arches we expose an
# empty dict so the tuner simply produces no flydsl tasks.
_arch = get_gfx()
if str(_arch).startswith("gfx95"):
    kernels_list = kernels_list_950
    default_kernels_dict = default_kernels_dict_950
else:
    kernels_list = {}
    default_kernels_dict = {}
