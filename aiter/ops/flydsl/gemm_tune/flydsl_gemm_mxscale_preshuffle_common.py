# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune catalog for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA).

Mirrors the aiter a8w8-bpreshuffle tune catalog style (kernelInstance + name +
build_kernels_list), but the mxscale_preshuffle kernel exposes only tile_m/n/k +
waves_per_eu as perf knobs (no async/lds/cshuffle/xcd/scheduler/split-K), so the
candidate schema and kernelName grammar are correspondingly small.

kernelName grammar (distinct `mxpsh` prefix, never collides with flydsl_bpreshuffle_*):
    flydsl_mxpsh_{tm}x{tn}x{tk}_{A}_{B}_{OUT}_w{wpe}_x{xcd}
e.g. flydsl_mxpsh_64x128x128_F8_F8_B16_w0_x0   (a8w8)
     flydsl_mxpsh_64x128x256_F4_F4_B16_w2_x4   (a4w4)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DTYPE_SHORT = {"fp8": "F8", "fp6": "F6", "fp4": "F4", "bf16": "B16", "fp16": "F16"}
_SHORT_DTYPE = {v: k for k, v in _DTYPE_SHORT.items()}

# a/b operand combos the kernel supports (a4w4 / a6w4 / a8w8).
_COMBOS = [("fp4", "fp4"), ("fp6", "fp4"), ("fp8", "fp8")]
_TILE_M = (32, 64, 96, 128, 256)
_TILE_N = (64, 128, 256, 512)
_TILE_K = (128, 256)
_WAVES_PER_EU = (0, 1, 2, 3, 4)
_XCD_SWIZZLE = (0, 4)  # L2-rasterization XCD swizzle group size (0=off)
MAX_SPLIT_K = 32
_SPLIT_K = tuple(range(1, MAX_SPLIT_K + 1))
_SPLITK_MAX_TMP_BYTES = 1 << 32
_GFX950_CU_NUM = 256


def _a_row_bytes(a_dtype: str, tile_k: int) -> int:
    """A bytes per row in a K-tile: fp4 packs 2 codes/byte, fp6/fp8 = 1 byte/code."""
    return tile_k // 2 if a_dtype == "fp4" else tile_k


@dataclass
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    a_dtype: str  # "fp4" | "fp6" | "fp8"
    b_dtype: str  # "fp4" | "fp8"
    out_dtype: str  # "bf16" | "fp16"
    waves_per_eu: int  # 0=no hint, 1-4=occupancy limit
    xcd_swizzle: int = 0  # 0=off, >0=XCD L2-rasterization group size
    split_k: int = 1  # 1=no split-K, >1=K-batch (partials reduced in fp32)

    @property
    def name(self) -> str:
        a = _DTYPE_SHORT[self.a_dtype]
        b = _DTYPE_SHORT[self.b_dtype]
        o = _DTYPE_SHORT[self.out_dtype]
        return (
            f"flydsl_mxpsh_{self.tile_m}x{self.tile_n}x{self.tile_k}"
            f"_{a}_{b}_{o}_w{self.waves_per_eu}_x{self.xcd_swizzle}_sk{self.split_k}"
        )


_NAME_RE = re.compile(
    r"^flydsl_mxpsh_(\d+)x(\d+)x(\d+)_([A-Z0-9]+)_([A-Z0-9]+)_([A-Z0-9]+)_w(\d+)"
    r"(?:_x(\d+))?(?:_sk(\d+))?$"
)


def parse_kernel_name(name: str):
    """kernelName string -> dict of launch params (None if it isn't an mxpsh name).

    The _x{xcd} and _sk{split_k} suffixes are optional (older names without them
    default to xcd_swizzle=0 / split_k=1).
    """
    m = _NAME_RE.match(name.strip())
    if not m:
        return None
    tm, tn, tk, a, b, o, wpe, xcd, sk = m.groups()
    return dict(
        tile_m=int(tm),
        tile_n=int(tn),
        tile_k=int(tk),
        a_dtype=_SHORT_DTYPE[a],
        b_dtype=_SHORT_DTYPE[b],
        out_dtype=_SHORT_DTYPE[o],
        waves_per_eu=int(wpe),
        xcd_swizzle=int(xcd) if xcd is not None else 0,
        split_k=int(sk) if sk is not None else 1,
    )


def estimated_lds_bytes(ki: kernelInstance) -> int:
    """Double-buffered A tile in LDS (SharedA.a0/a1), row-major [tile_m][a_row_bytes]."""
    return 2 * ki.tile_m * _a_row_bytes(ki.a_dtype, ki.tile_k)


def _max_lds_bytes() -> int:
    try:
        from aiter.ops.flydsl.utils import get_shared_memory_per_block

        return int(get_shared_memory_per_block(fallback_gfx="gfx950"))
    except Exception:
        return 160 * 1024  # gfx950 LDS


def instance_valid(ki: kernelInstance) -> bool:
    """Shape-independent legality against the mxscale_preshuffle kernel constraints."""
    if ki.tile_k not in (128, 256):
        return False
    if ki.tile_m % 32 != 0:  # microscale packs M by 2 -> m_chunks = tile_m//16 even
        return False
    if ki.tile_n % 64 != 0:  # 16-col n-subblocks per wave: tile_n multiple of 64
        return False
    arb = _a_row_bytes(ki.a_dtype, ki.tile_k)
    if (
        ki.tile_m * arb
    ) % 4096 != 0:  # A coop load: n_coop = tile_m*arb//4096 must be integral
        return False
    if estimated_lds_bytes(ki) > _max_lds_bytes():
        return False
    return True


def fits_shape(ki: kernelInstance, M: int, N: int, K: int) -> bool:
    """M is ragged (grid ceil + OOB clip). K must be a multiple of 128: each e8m0
    microscale half is 128-K, and tile_k=128 pairs two halves into one 256-K scale
    word (shuffle_scale rounds K up to a whole 256-K chunk). K%tile_k excludes
    tile_k=256 when K%256!=0, so a K=384 shape only matches tile_k=128 kernels.

    split-K legality (split_k>1): the per-split K length (K/split_k) must stay a
    whole number of tile_k K-tiles AND a whole number of 256-K e8m0 scale chunks,
    so the split boundary never straddles a tile or a microscale word."""
    if K % 128 != 0:
        return False
    if (N % ki.tile_n != 0) or (K % ki.tile_k != 0):
        return False
    if ki.split_k > 1:
        k_per_split = K // ki.split_k
        if (
            K % ki.split_k != 0
            or k_per_split % ki.tile_k != 0
            or k_per_split % 256 != 0
        ):
            return False
        # fp32 scratch tmp[split_k, M, N] must fit the 32-bit num_records field.
        if M * N * ki.split_k * 4 >= _SPLITK_MAX_TMP_BYTES:
            return False
        # occupancy guard: skip split_k>1 when the base grid already fills the CUs.
        base_wg = ((M + ki.tile_m - 1) // ki.tile_m) * (N // ki.tile_n)
        if base_wg >= _GFX950_CU_NUM:
            return False
    return True


def _build_kernels_list():
    out = {}
    idx = 0
    for a_dtype, b_dtype in _COMBOS:
        for tm in _TILE_M:
            for tn in _TILE_N:
                for tk in _TILE_K:
                    for wpe in _WAVES_PER_EU:
                        for xcd in _XCD_SWIZZLE:
                            for sk in _SPLIT_K:
                                ki = kernelInstance(
                                    tm, tn, tk, a_dtype, b_dtype, "bf16", wpe, xcd, sk
                                )
                                if instance_valid(ki):
                                    out[idx] = ki
                                    idx += 1
    return out


kernels_list = _build_kernels_list()


def candidates_for(a_dtype: str, b_dtype: str, M: int, N: int, K: int):
    """(kernel_id, kernelInstance) that match dtypes and fit the shape."""
    return [
        (i, ki)
        for i, ki in kernels_list.items()
        if ki.a_dtype == a_dtype and ki.b_dtype == b_dtype and fits_shape(ki, M, N, K)
    ]
