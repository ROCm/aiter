# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune spaces for every FlyDSL a8w8 (ptpc) bpreshuffle pipeline on CDNA.

One operator, two pipelines, swept together under the same ``flydsl`` libtype so
a single ``--libtype flydsl`` run picks one winner per shape:

* ``preshuffle`` -- 4-wave MFMA, gfx942 + gfx950, fp8 or int8 weights.
* ``8wave``      -- 8-wave CDNA4 ``MFMA_Scale``, gfx950 only, fp8 only.

Each exposes the same pair -- a ``{kernelId: instance}`` table and a
``fits(ki, M, N, K)`` predicate -- and both are listed in :data:`PIPELINES`, so
the tuner iterates instead of carrying one hand-written branch per pipeline.
Runners stay on the tuner side: this module must remain importable without
flydsl (the tuner reads it to name candidates on hosts that cannot compile).

The gfx1250 WMMA pipeline has its own file and is not part of PIPELINES yet.
"""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

from aiter.ops.flydsl.utils import (
    addressable_lds_bytes_for_gfx as _addressable_lds_bytes_for_gfx,
)
from aiter.ops.flydsl.utils import (
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
    except Exception:  # noqa: BLE001
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
    use_async_copy: int  # 0 or 1
    waves_per_eu: int  # 0=no hint, 1-4=occupancy limit
    xcd_swizzle: int  # 0=off, >0=group size for XCD remap
    lds_stage: int = 2  # 2=double-buffer ping-pong, 1=single A-LDS buffer (half LDS)
    sScheduler: str = "Default"  # scheduler hints on; "Off" = compiler default

    @property
    def enable_scheduler(self) -> bool:
        """Map the scheduler name token to compile_preshuffle_gemm(enable_scheduler=)."""
        return str(self.sScheduler).lower() != "off"

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
                            self.use_async_copy,
                            self.waves_per_eu,
                            self.xcd_swizzle,
                            self.lds_stage,
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
    async_copy=0,
    waves_per_eu=0,
    xcd_swizzle=0,
    lds_stage=2,
    q_dtype_a="fp8",
    q_dtype_w="fp8",
    dtype="bf16",
    scheduler="Default",
):
    return kernelInstance(
        tile_m,
        tile_n,
        tile_k,
        q_dtype_a,
        q_dtype_w,
        dtype,
        async_copy,
        waves_per_eu,
        xcd_swizzle,
        lds_stage,
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
) -> int:
    """Estimated total LDS (bytes) for preshuffle_gemm: sum of smem globals.

    Mirrors ``preshuffle_gemm.py`` A-tile allocation (``SharedStorage`` holds
    ``lds_stage`` × ``tile_m x tile_k`` buffers: 2 for ping-pong, 1 for the
    single-buffer path); used to skip tune instances that exceed AMDGPU
    per-kernel LDS limits (e.g. 64 KiB on gfx942).
    """
    elem_bytes = 1 if in_dtype in ("fp8", "int8") else 2
    a_tile_bytes = int(tile_m) * int(tile_k) * elem_bytes

    # lds_stage A-tile buffers, each finalized to a 128B-aligned smem global.
    return _smem_finalize_size(_smem_align(a_tile_bytes)) * (
        2 if int(lds_stage) == 2 else 1
    )


def kernel_instance_estimated_lds_bytes(ki: kernelInstance) -> int:
    """LDS estimate using dtypes from a tune ``kernelInstance``."""
    return preshuffle_gemm_estimated_lds_bytes(
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        in_dtype=ki.q_dtype_a,
        out_dtype=ki.dtype,
        lds_stage=ki.lds_stage,
    )


def addressable_lds_bytes_for_gfx(gfx: str) -> int:
    return _addressable_lds_bytes_for_gfx(gfx)


@cache
def max_lds_bytes_for_tune() -> int:
    """Addressable LDS limit for current target.

    Cached because ``kernel_fits_shape`` calls it per candidate (thousands of
    times per shape) and the uncached path does a ``torch.cuda.current_device()``
    round trip each time. The arch is already resolved once at import below, so
    a process-lifetime cache changes nothing.
    """
    return get_shared_memory_per_block(fallback_gfx=get_gfx())


def _padded_m(M: int) -> int:
    """Round M up to the bucket the tuner tunes at.

    Moved verbatim from ``GemmA8W8BpreShuffleTuner`` (was ``_get_padded_m``); it
    had no other caller there.
    """
    if M <= 256:
        return (M + 15) // 16 * 16
    elif M <= 1024:
        return (M + 31) // 32 * 32
    elif M <= 4096:
        return (M + 63) // 64 * 64
    else:
        return (M + 127) // 128 * 128


def kernel_fits_shape(ki: kernelInstance, M: int, N: int, K: int) -> bool:
    """Whether a preshuffle candidate is worth tuning for this shape.

    Every predicate is verbatim from the tuner's inline filter, in the original
    order, so the enumerated candidate set is unchanged. **Do not tune these
    here**: they decide the search space, so relaxing or tightening any of them
    invalidates the committed tuned CSVs and requires a re-tune. It lives beside
    ``kernels_list`` only so both a8w8 bpreshuffle pipelines expose the same
    ``(kernels_list, kernel_fits_shape)`` pair.
    """
    if kernel_instance_estimated_lds_bytes(ki) > max_lds_bytes_for_tune():
        return False
    if N % ki.tile_n != 0 or K % ki.tile_k != 0:
        return False
    if _padded_m(M) % ki.tile_m != 0:
        return False
    num_ctas = ((M + ki.tile_m - 1) // ki.tile_m) * (N // ki.tile_n)
    if num_ctas < max(4, min(16, N // 64)):
        return False
    if ki.tile_m == 16 and ki.tile_n == 512:
        return False
    if M >= 8192 and ki.tile_m < 64:
        return False
    if M >= 4096 and ki.tile_m < 32:
        return False
    return not (M >= 2048 and ki.tile_m == 16 and ki.tile_n <= 128)


# fmt: off
# ---------------------------------------------------------------------------
# Base tile configurations: (tile_m, tile_n, tile_k)
# ---------------------------------------------------------------------------

# Tiles shared by gfx942 and gfx950
_base_tiles_common = [
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

# gfx942-only tiles (tile_k=64 not supported on gfx950)
_base_tiles_942_extra = [
    (64,  256, 64),
    (128, 128, 64),
]

# gfx950-only tile
_base_tiles_950_extra = [
    (256, 256, 128),
]

# ---------------------------------------------------------------------------
# Combo sweep: lds_stage x waves_per_eu x async_copy x xcd_swizzle
# ---------------------------------------------------------------------------
_ASYNC_COPY_VALS = (0, 1)
_WAVES_PER_EU    = (0, 1, 2, 3, 4)
_XCD_SWIZZLE_VALS = (0, 4)  # 0=off, >0=XCD remap group size
_LDS_STAGES      = (2, 1)  # 2=double-buffer ping-pong, 1=single A-LDS buffer

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
    Estimated total ~= accum * 1.5 (pipeline overhead for A/B buffers).
    Returns the max waves_per_eu that the register file can support.
    """
    padded_m = math.ceil(tile_m / _MFMA_M) * _MFMA_M
    padded_n = math.ceil(tile_n / _MFMA_N) * _MFMA_N
    c_per_thread = padded_m * padded_n // _THREADS_PER_TG
    est_per_wave = c_per_thread * 1.5
    return int(total_vgpr / max(est_per_wave, 1))


def _build_kernels_list(tiles, total_vgpr=512):
    kl = {}
    idx = 0

    for lds in _LDS_STAGES:
        for wpe in _WAVES_PER_EU:
            for acp in _ASYNC_COPY_VALS:
                for xcd in _XCD_SWIZZLE_VALS:
                    for tm, tn, tk in tiles:
                        if wpe > 0 and wpe > _estimate_max_wpe(tm, tn, total_vgpr):
                            continue
                        kl[idx] = _ki(tm, tn, tk, acp, wpe, xcd, lds_stage=lds)
                        idx += 1
    return kl


kernels_list_942 = _build_kernels_list(
    _base_tiles_common + _base_tiles_942_extra,
    total_vgpr=_vgpr_per_simd("gfx942"))
kernels_list_950 = _build_kernels_list(
    _base_tiles_common + _base_tiles_950_extra,
    total_vgpr=_vgpr_per_simd("gfx950"))
# fmt: on

default_kernels_dict_942 = {
    (-1): _ki(128, 128, 128, 0, 2, 0, 2, scheduler="Default"),
    (-2): _ki(16, 64, 512, 0, 2, 0, 2, scheduler="Default"),
    (-3): _ki(32, 64, 512, 0, 2, 0, 2, scheduler="Default"),
    (-4): _ki(64, 256, 64, 0, 2, 0, 2, scheduler="Default"),
    (-5): _ki(128, 128, 64, 0, 2, 0, 2, scheduler="Default"),
    (-6): _ki(128, 64, 128, 0, 2, 0, 2, scheduler="Default"),
    (-7): _ki(64, 256, 128, 0, 2, 0, 2, scheduler="Default"),
}

default_kernels_dict_950 = {
    (-1): _ki(128, 256, 256, 0, 2, 0, 2, scheduler="Default"),
    (-2): _ki(16, 64, 512, 0, 2, 0, 2, scheduler="Default"),
    (-3): _ki(32, 64, 512, 0, 2, 0, 2, scheduler="Default"),
    (-4): _ki(128, 128, 128, 0, 2, 0, 2, scheduler="Default"),
}

arch = get_gfx()
if arch == "gfx942":
    kernels_list = kernels_list_942
    default_kernels_dict = default_kernels_dict_942
else:
    kernels_list = kernels_list_950
    default_kernels_dict = default_kernels_dict_950


# ===========================================================================
# Pipeline 2: 8wave (CDNA4 MFMA_Scale), gfx950 only
# ===========================================================================

try:
    from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import (
        BLOCK_K as BLOCK_K_8WAVE,
    )
    from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import (
        MIN_K as MIN_K_8WAVE,
    )
    from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import (
        lds_bytes as _lds_bytes_8wave,
    )
except Exception as _exc:  # noqa: BLE001
    print(f"[FlyDSL] 8wave op module unavailable ({_exc}); 8wave candidates disabled")
    BLOCK_K_8WAVE, MIN_K_8WAVE, _lds_bytes_8wave = 128, 256, None

NAME_PREFIX_8WAVE = "flydsl_bpreshuffle_8w"

# 8wave ids share the ``flydsl`` libtype with the preshuffle pipeline and can
# co-occur on the same arch, so they are offset out of that pipeline's dense
# 0-based range. Routing is by Pipeline, not by id range -- the offset only
# keeps CSV rows unambiguous to a human.
KERNEL_ID_BASE_8WAVE = 1_000_000

LDS_BYTES_8WAVE = get_shared_memory_per_block(fallback_gfx="gfx950")
_I32_MAX = 2**31

_TILES_8WAVE = ((128, 256), (128, 512), (256, 256))
# The kernel always emits the rocdl.waves_per_eu attribute, so 0 ("no hint",
# as used by the preshuffle pipeline) is not expressible here.
_WAVES_PER_EU_8WAVE = (1, 2, 3, 4)
_XCD_SWIZZLE_VALS_8WAVE = (0, 4, 8)


@dataclass
class EightWaveKernelInstance:
    block_m: int
    block_n: int
    waves_per_eu: int
    xcd_swizzle: int
    q_dtype_a: str = "fp8"
    q_dtype_w: str = "fp8"
    dtype: str = "bf16"

    @property
    def name(self) -> str:
        return (
            f"{NAME_PREFIX_8WAVE}_{self.block_m}x{self.block_n}x{BLOCK_K_8WAVE}_"
            f"F8_F8_B16_{self.waves_per_eu}x{self.xcd_swizzle}"
        )


def kernel_instance_estimated_lds_bytes_8wave(ki: EightWaveKernelInstance) -> int:
    """Exact (not estimated) LDS footprint for a candidate."""
    return _lds_bytes_8wave(ki.block_m, ki.block_n)


def kernel_fits_shape_8wave(
    ki: EightWaveKernelInstance, M: int, N: int, K: int
) -> bool:
    """Whether a candidate can run this shape at all.

    Deliberately does NOT require ``M % block_m == 0`` or ``N % block_n == 0``:
    the 8-wave kernel handles ragged M and N through its buffer descriptors and
    an explicit column guard, and the best tile for e.g. M=11256 is ragged.
    """
    if _lds_bytes_8wave is None:
        return False
    if M <= 0 or N <= 0 or K <= 0:
        return False
    if K % BLOCK_K_8WAVE != 0 or K < MIN_K_8WAVE:
        return False
    # shuffle_weight(layout=(16, 16)) precondition, and the kernel's B swizzle
    # addresses B in 16-row groups.
    if N % 16 != 0:
        return False
    if kernel_instance_estimated_lds_bytes_8wave(ki) > LDS_BYTES_8WAVE:
        return False
    # Buffer descriptors index in 32 bits; C is bf16 so its byte count doubles.
    return not (M * N * 2 >= _I32_MAX or M * K >= _I32_MAX or N * K >= _I32_MAX)


def is_8wave_enabled() -> bool:
    """8wave candidates are gfx950-only (the MMA atom is CDNA4).

    No env kill switch on purpose: this gates the tuner's candidate list only,
    so it could not stop a tuned CSV row from running 8wave in production --
    ``gemm_a8w8_bpreshuffle_flydsl`` dispatches on the kernelName prefix alone.
    To exclude the pipeline, drop its rows from the tuned CSV.
    """
    return _lds_bytes_8wave is not None and get_gfx().startswith("gfx950")


def _build_kernels_list_8wave() -> dict[int, EightWaveKernelInstance]:
    kl: dict[int, EightWaveKernelInstance] = {}
    idx = KERNEL_ID_BASE_8WAVE
    for block_m, block_n in _TILES_8WAVE:
        if _lds_bytes_8wave(block_m, block_n) > LDS_BYTES_8WAVE:
            continue
        for wpe in _WAVES_PER_EU_8WAVE:
            for xcd in _XCD_SWIZZLE_VALS_8WAVE:
                kl[idx] = EightWaveKernelInstance(block_m, block_n, wpe, xcd)
                idx += 1
    return kl


kernels_list_8wave: dict[int, EightWaveKernelInstance] = (
    _build_kernels_list_8wave() if is_8wave_enabled() else {}
)


# ===========================================================================
# The protocol the tuner iterates
# ===========================================================================


@dataclass(frozen=True)
class Pipeline:
    """One tunable backend of the a8w8 bpreshuffle operator.

    ``q_dtypes_w`` is spelled with plain strings rather than ``torch.dtype`` so
    this module stays importable without torch/flydsl; the tuner maps its
    ``dtypes.fp8`` / ``dtypes.i8`` onto these names.
    """

    name: str
    kernels_list: dict[int, Any]
    fits: Callable[[Any, int, int, int], bool]
    q_dtypes_w: tuple[str, ...]


# Order is load-bearing: it is the order candidates are emitted per shape, and
# preshuffle-before-8wave matches what the tuner produced before PIPELINES
# existed.
PIPELINES: tuple[Pipeline, ...] = (
    Pipeline("preshuffle", kernels_list, kernel_fits_shape, ("fp8", "int8")),
    Pipeline("8wave", kernels_list_8wave, kernel_fits_shape_8wave, ("fp8",)),
)
