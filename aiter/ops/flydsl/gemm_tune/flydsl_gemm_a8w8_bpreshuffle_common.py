# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
import math
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
    "int8": "I8",
    "bf16": "B16",
    "fp16": "F16",
}


@dataclass
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    q_dtype_a: str  # "fp8" | "int8"
    q_dtype_w: str  # "fp8" | "int8"
    dtype: str  # output dtype: "bf16" | "fp16"
    lds_stage: int  # 1 or 2
    use_cshuffle_epilog: int  # 0 or 1
    use_async_copy: int  # 0 or 1
    waves_per_eu: int  # 0=no hint, 1-4=occupancy limit
    sScheduler: str  # "Default"

    @property
    def name(self) -> str:
        qa = _DTYPE_SHORT.get(self.q_dtype_a, self.q_dtype_a.upper())
        qw = _DTYPE_SHORT.get(self.q_dtype_w, self.q_dtype_w.upper())
        dt = _DTYPE_SHORT.get(self.dtype, self.dtype.upper())
        return "_".join(
            [
                "flydsl",
                "bpreshuflle",
                "x".join(map(str, [self.tile_m, self.tile_n, self.tile_k])),
                qa,
                qw,
                dt,
                "x".join(
                    map(
                        str,
                        [
                            self.lds_stage,
                            self.use_cshuffle_epilog,
                            self.use_async_copy,
                            self.waves_per_eu,
                        ],
                    )
                ),
                self.sScheduler.lower(),
            ]
        )


def _ki(
    tile_m,
    tile_n,
    tile_k,
    lds_stage,
    cshuffle=0,
    async_copy=0,
    waves_per_eu=0,
    scheduler="Default",
    q_dtype_a="fp8",
    q_dtype_w="fp8",
    dtype="bf16",
):
    return kernelInstance(
        tile_m,
        tile_n,
        tile_k,
        q_dtype_a,
        q_dtype_w,
        dtype,
        lds_stage,
        cshuffle,
        async_copy,
        waves_per_eu,
        scheduler,
    )


def _smem_align(ptr: int, align: int = 16) -> int:
    if ptr % align == 0:
        return ptr
    return (ptr + align - 1) // align * align


def _smem_finalize_size(used_ptr: int) -> int:
    """Match FlyDSL SmemAllocator.finalize: align ptr to 128, min 128."""
    total = _smem_align(used_ptr, 128)
    if total == 0:
        return 128
    return total


def preshuffle_gemm_estimated_lds_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    lds_stage: int = 2,
    use_cshuffle_epilog: int = 0,
) -> int:
    """Estimated total LDS (bytes) for preshuffle_gemm: sum of two smem globals.

    Mirrors ``preshuffle_gemm.py`` ping/pong allocation; used to skip tune
    instances that exceed AMDGPU per-kernel LDS limits (e.g. 64 KiB on gfx942).
    """
    is_fp4 = in_dtype == "fp4"
    elem_bytes = 1 if in_dtype in ("fp8", "int8", "int4", "fp4") else 2
    a_elem_vec_pack = 2 if is_fp4 else 1
    tile_k_bytes = int(tile_k) * elem_bytes
    lds_tile_bytes = int(tile_m) * tile_k_bytes // a_elem_vec_pack
    # Epilogue staging in LDS is fp16/bf16-sized (2 bytes per element).
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if int(use_cshuffle_epilog) else 0

    ptr_pong = 0
    ptr_ping = 0
    if int(lds_stage) == 2:
        buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // 2)
        buffer_size_elems = (
            buffer_size_bytes if elem_bytes == 1 else buffer_size_bytes // 2
        )
        bsz = buffer_size_elems * elem_bytes
        ptr_pong = _smem_align(ptr_pong) + bsz
        ptr_ping = _smem_align(ptr_ping) + bsz
    else:
        lds_total_bytes = max(lds_tile_bytes, lds_out_bytes)
        lds_total_elems = lds_total_bytes if elem_bytes == 1 else lds_total_bytes // 2
        ptr_pong = _smem_align(ptr_pong) + lds_total_elems * elem_bytes

    return _smem_finalize_size(ptr_pong) + _smem_finalize_size(ptr_ping)


def kernel_instance_estimated_lds_bytes(ki: kernelInstance) -> int:
    """LDS estimate using dtypes from a tune ``kernelInstance``."""
    return preshuffle_gemm_estimated_lds_bytes(
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        in_dtype=ki.q_dtype_a,
        out_dtype=ki.dtype,
        lds_stage=ki.lds_stage,
        use_cshuffle_epilog=ki.use_cshuffle_epilog,
    )


def addressable_lds_bytes_for_gfx(gfx: str) -> int:
    return _addressable_lds_bytes_for_gfx(gfx)


def max_lds_bytes_for_tune() -> int:
    """Addressable LDS limit for current target."""
    return get_shared_memory_per_block(fallback_gfx=get_gfx())


# fmt: off
# ---------------------------------------------------------------------------
# Base tile configurations: (tile_m, tile_n, tile_k)
# ---------------------------------------------------------------------------

# lds_stage=2 tiles shared by gfx942 and gfx950
_base_tiles_lds2_common = [
    # small M (decode / token-gen)
    (16,  64,  256), (16,  64,  512),
    (16,  128, 256), (16,  128, 512), (16,  256, 256), (16,  256, 512),
    (16,  512, 256), (16,  192, 256),
    # M=32
    (32,  64,  128), (32,  64,  256), (32,  64,  512), (32,  128, 128),
    (32,  128, 256), (32,  192, 128), (32,  192, 256), (32,  256, 128),
    (32,  256, 256),
    # M=48
    (48,  64,  256), (48,  128, 256), (48,  192, 256), (48,  256, 256),
    # M=64
    (64,  64,  128), (64,  64,  256), (64,  128, 128), (64,  128, 256),
    (64,  192, 128), (64,  192, 256), (64,  256, 128),
    (64,  256, 256),
    # M=80
    (80,  64,  256), (80,  128, 256), (80,  192, 256), (80,  256, 256),
    # M=96
    (96,  64,  128), (96,  64,  256), (96,  128, 128), (96,  128, 256),
    (96,  192, 128), (96,  192, 256), (96,  256, 128), (96,  256, 256),
    # M=112
    (112, 64,  256), (112, 128, 256), (112, 192, 256), (112, 256, 256),
    # M=128
    (128, 64,  128), (128, 64,  256), (128, 128, 128),
    (128, 128, 256), (128, 192, 128), (128, 192, 256), (128, 256, 128),
    # M=160/192/224/256
    (160, 192, 128),
    (192, 64,  128), (192, 128, 128),
    (224, 64,  128), (224, 128, 128), (224, 192, 128),
    (256, 64,  128), (256, 128, 128), (256, 192, 128),
]

# gfx942-only lds_stage=2 tiles (tile_k=64 not supported on gfx950)
_base_tiles_lds2_942_extra = [
    (64,  256, 64),
    (128, 128, 64),
]

# gfx950-only lds_stage=2 tile
_base_tiles_lds2_950_extra = [
    (256, 256, 128),
]

# lds_stage=1 tiles (same for both archs)
_base_tiles_lds1 = [
    (16,  64,  256), (16,  64,  512),
    (16,  128, 256), (16,  128, 512), (16,  256, 256), (16,  256, 512),
    (16,  512, 256),
    (32,  64,  128), (32,  64,  256), (32,  64,  512), (32,  128, 128),
    (32,  128, 256),
    (64,  64,  128), (64,  64,  256), (64,  128, 128), (64,  128, 256),
    (64,  256, 128),
    (128, 64,  128), (128, 128, 128), (128, 128, 256), (128, 256, 128),
]

# ---------------------------------------------------------------------------
# Combo sweep: lds_stage x cshuffle x async_copy x waves_per_eu
# ---------------------------------------------------------------------------
_LDS_STAGES      = (1, 2)
_CSHUFFLE_VALS   = (0, 1)
_ASYNC_COPY_VALS = (0, 1)
_WAVES_PER_EU    = (0, 1, 2, 3, 4)

_WAVES_PER_WG = 4  # typical wavefronts per workgroup in FlyDSL preshuffle GEMM


def _vgpr_per_simd(gfx: str) -> int:
    """VGPRs per SIMD unit for the given GPU architecture."""
    g = (gfx or "").strip().lower()
    if g.startswith("gfx9"):
        return 512
    return 512


_MFMA_M = 16
_MFMA_N = 16
_THREADS_PER_TG = _WAVES_PER_WG * 64


def _estimate_max_wpe(tile_m: int, tile_n: int, total_vgpr: int = 512) -> int:
    """Estimate max achievable waves_per_eu from C-accumulator VGPR pressure.

    Preshuffle GEMM always uses 16x16 MFMA (4 VGPRs per thread per block).
    Per-thread accum VGPRs = round_up(tile_m, 16) * round_up(tile_n, 16) / 256.
    Estimated total ? accum * 1.5 (pipeline overhead for A/B buffers).
    Returns the max waves_per_eu that the register file can support.
    """
    padded_m = math.ceil(tile_m / _MFMA_M) * _MFMA_M
    padded_n = math.ceil(tile_n / _MFMA_N) * _MFMA_N
    c_per_thread = padded_m * padded_n // _THREADS_PER_TG
    est_per_wave = c_per_thread * 1.5
    return int(total_vgpr / max(est_per_wave, 1))


def _build_kernels_list(tiles_lds2, tiles_lds1, total_vgpr=512):
    tiles_by_lds = {2: tiles_lds2, 1: tiles_lds1}
    kl = {}
    idx = 0
    for wpe in _WAVES_PER_EU:
        for csh in _CSHUFFLE_VALS:
            for acp in _ASYNC_COPY_VALS:
                for lds in _LDS_STAGES:
                    for tm, tn, tk in tiles_by_lds[lds]:
                        if wpe > 0 and wpe > _estimate_max_wpe(tm, tn, total_vgpr):
                            continue
                        kl[idx] = _ki(tm, tn, tk, lds, csh, acp, wpe)
                        idx += 1
    return kl


kernels_list_942 = _build_kernels_list(
    _base_tiles_lds2_common + _base_tiles_lds2_942_extra, _base_tiles_lds1,
    total_vgpr=_vgpr_per_simd("gfx942"))
kernels_list_950 = _build_kernels_list(
    _base_tiles_lds2_common + _base_tiles_lds2_950_extra, _base_tiles_lds1,
    total_vgpr=_vgpr_per_simd("gfx950"))
# fmt: on

default_kernels_dict_942 = {
    (-1): _ki(128, 128, 128, 2, 0, 0, 2, "Default"),
    (-2): _ki(16, 64, 512, 2, 0, 0, 2, "Default"),
    (-3): _ki(32, 64, 512, 2, 0, 0, 2, "Default"),
    (-4): _ki(64, 256, 64, 2, 0, 0, 2, "Default"),
    (-5): _ki(128, 128, 64, 2, 0, 0, 2, "Default"),
    (-6): _ki(128, 64, 128, 2, 0, 0, 2, "Default"),
    (-7): _ki(64, 256, 128, 2, 0, 0, 2, "Default"),
}

default_kernels_dict_950 = {
    (-1): _ki(128, 256, 256, 2, 0, 0, 2, "Default"),
    (-2): _ki(16, 64, 512, 2, 0, 0, 2, "Default"),
    (-3): _ki(32, 64, 512, 2, 0, 0, 2, "Default"),
    (-4): _ki(128, 128, 128, 2, 0, 0, 2, "Default"),
}

arch = get_gfx()
if arch == "gfx942":
    kernels_list = kernels_list_942
    default_kernels_dict = default_kernels_dict_942
else:
    kernels_list = kernels_list_950
    default_kernels_dict = default_kernels_dict_950
