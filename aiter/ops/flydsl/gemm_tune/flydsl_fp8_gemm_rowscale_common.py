# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune-side configuration for FlyDSL FP8 row-scale GEMM kernels.

Wraps both the 4-wave interleave variant (``fp8_gemm_4wave.py``) and the
8-wave ping-pong variant (``fp8_gemm_8wave.py``). Both share:

- fp8_e4m3fn input (A and B)
- bf16 output
- per-token (row) scale on A and B
- fixed BLOCK_K = 128
- gfx95* (CDNA4) only

Diffs between variants:

- 4-wave: ``block=(256, 1, 1)``, waves_per_eu=1, exposes ``use_xcd_remap``,
  enforces ``BLOCK_M >= 64 (mod 64) and BLOCK_N >= 64 (mod 64)``
- 8-wave: ``block=(512, 1, 1)``, waves_per_eu=2, NO ``use_xcd_remap`` knob,
  enforces ``BLOCK_M >= 128 (mod 128) and BLOCK_N >= 256 (mod 256)``
"""

from dataclasses import dataclass, field
import os

from aiter.ops.flydsl.utils import (
    addressable_lds_bytes_for_gfx as _addressable_lds_bytes_for_gfx,
    get_shared_memory_per_block,
)


def get_gfx():
    """Detect GPU arch: honour GPU_ARCHS env, fall back to chip_info, default gfx950."""
    env = os.environ.get("GPU_ARCHS", "")
    if env and env != "native":
        return env.split(";")[-1].strip()
    try:
        from aiter.jit.utils.chip_info import get_gfx as _get_gfx

        return _get_gfx()
    except Exception:
        return "gfx950"


# FP8 row-scale GEMM is gfx95* only; BLOCK_K and dtypes are baked into the kernel.
_FIXED_TILE_K = 128
_FIXED_IN_DTYPE = "fp8"
_FIXED_OUT_DTYPE = "bf16"
_FIXED_ARCH_PREFIX = "gfx95"

# Per-variant hard-coded launch attributes from the kernel sources.
_BLOCK_SIZE_BY_WAVE = {4: 256, 8: 512}
_WAVES_PER_EU_BY_WAVE = {4: 1, 8: 2}


@dataclass
class kernelInstance:
    """Identity of one compiled FP8 row-scale GEMM kernel.

    Only fields that actually change the compiled artifact are exposed.
    Implicit fields (``tile_k``, dtypes, arch, ``waves_per_eu``, block size)
    are kept here for documentation/validation but are not tune dimensions.

    B is always passed in the preshuffled layout produced by
    ``aiter.ops.flydsl.kernels.fp8_gemm_utils.preshuffle_b``; the row-major
    path is no longer a tune dimension.
    """

    # ── Tune dimensions ──────────────────────────────────────────────────
    tile_m: int                  # BLOCK_M
    tile_n: int                  # BLOCK_N
    wave_variant: int            # 4 or 8 (selects 4wave / 8wave kernel)
    xcd_remap: int = 1           # 0 or 1; only meaningful for 4-wave (8-wave ignores)

    # ── Implicit/fixed (kept for clarity, do not vary) ───────────────────
    tile_k: int = _FIXED_TILE_K
    q_dtype_a: str = _FIXED_IN_DTYPE
    q_dtype_w: str = _FIXED_IN_DTYPE
    dtype: str = _FIXED_OUT_DTYPE         # output dtype
    waves_per_eu: int = field(init=False)
    block_size: int = field(init=False)

    def __post_init__(self):
        if self.wave_variant not in (4, 8):
            raise ValueError(f"wave_variant must be 4 or 8, got {self.wave_variant}")
        if self.xcd_remap not in (0, 1):
            raise ValueError(f"xcd_remap must be 0 or 1, got {self.xcd_remap}")
        if self.tile_k != _FIXED_TILE_K:
            raise ValueError(f"tile_k is fixed at {_FIXED_TILE_K}, got {self.tile_k}")
        if self.dtype != _FIXED_OUT_DTYPE:
            raise ValueError(f"output dtype must be {_FIXED_OUT_DTYPE!r}, got {self.dtype!r}")
        if self.q_dtype_a != _FIXED_IN_DTYPE or self.q_dtype_w != _FIXED_IN_DTYPE:
            raise ValueError(f"input dtype must be {_FIXED_IN_DTYPE!r}")

        if self.wave_variant == 8:
            # Source-of-truth: kernels/fp8_gemm_8wave.py:30
            if self.tile_m < 128 or self.tile_m % 128 != 0:
                raise ValueError(
                    f"8-wave requires tile_m >= 128 and divisible by 128, got {self.tile_m}"
                )
            if self.tile_n < 256 or self.tile_n % 256 != 0:
                raise ValueError(
                    f"8-wave requires tile_n >= 256 and divisible by 256, got {self.tile_n}"
                )
        else:
            # Source-of-truth: kernels/fp8_gemm_4wave.py:84
            #   assert BLOCK_M >= 64 and BLOCK_M % 64 == 0
            #   assert BLOCK_N >= 64 and BLOCK_N % 64 == 0
            # (the previous "% 16" claim here was wrong: the 2x2 wave layout
            # plus the 16x16x128 MFMA atom requires each BLOCK dim to cover
            # at least one 64-row/64-col wave partition.)
            if self.tile_m < 64 or self.tile_m % 64 != 0:
                raise ValueError(
                    f"4-wave requires tile_m >= 64 and divisible by 64, got {self.tile_m}"
                )
            if self.tile_n < 64 or self.tile_n % 64 != 0:
                raise ValueError(
                    f"4-wave requires tile_n >= 64 and divisible by 64, got {self.tile_n}"
                )

        # Derived
        self.waves_per_eu = _WAVES_PER_EU_BY_WAVE[self.wave_variant]
        self.block_size = _BLOCK_SIZE_BY_WAVE[self.wave_variant]

    @property
    def name(self) -> str:
        """Stable kernel name used in tune CSVs and dispatcher.

        Format (4-wave):  flydsl_fp8_4w_<tm>x<tn>_<xcd|noxcd>
        Format (8-wave):  flydsl_fp8_8w_<tm>x<tn>
        """
        wave = f"{self.wave_variant}w"
        if self.wave_variant == 4:
            xcd = "xcd" if self.xcd_remap else "noxcd"
            return f"flydsl_fp8_{wave}_{self.tile_m}x{self.tile_n}_{xcd}"
        return f"flydsl_fp8_{wave}_{self.tile_m}x{self.tile_n}"


def _ki(
    tile_m,
    tile_n,
    wave_variant,
    *,
    xcd_remap=1,
):
    return kernelInstance(
        tile_m=tile_m,
        tile_n=tile_n,
        wave_variant=wave_variant,
        xcd_remap=xcd_remap,
    )


# ---------------------------------------------------------------------------
# LDS sizing (mirrors the 8-buffer pipeline in fp8_gemm_4wave / 8wave)
# ---------------------------------------------------------------------------

def _smem_align(ptr: int, align: int = 16) -> int:
    if ptr % align == 0:
        return ptr
    return (ptr + align - 1) // align * align


def _smem_finalize_size(used_ptr: int) -> int:
    """Match FlyDSL SmemAllocator.finalize: align to 128 bytes, min 128."""
    total = _smem_align(used_ptr, 128)
    if total == 0:
        return 128
    return total


def fp8_gemm_rowscale_estimated_lds_bytes(tile_m: int, tile_n: int) -> int:
    """Estimated total LDS bytes for the 8-buffer A/B pipeline.

    Layout (both 4-wave and 8-wave variants):
      A side: 4 buffers of (BLOCK_M / 2) * BLOCK_K bytes  (cur0/cur1/next0/next1)
      B side: 4 buffers of (BLOCK_N / 2) * BLOCK_K bytes
    FP8 = 1 byte per element, BLOCK_K = 128.
    Each buffer is its own SmemAllocator global, so each is independently
    finalized (aligned/padded to 128 bytes, min 128).
    """
    lds_block_m = tile_m // 2
    lds_block_n = tile_n // 2
    a_buf_bytes = lds_block_m * _FIXED_TILE_K  # fp8: 1 byte/elem
    b_buf_bytes = lds_block_n * _FIXED_TILE_K
    total = 0
    for _ in range(4):  # a_cur0, a_cur1, a_next0, a_next1
        total += _smem_finalize_size(a_buf_bytes)
    for _ in range(4):  # b_cur0, b_cur1, b_next0, b_next1
        total += _smem_finalize_size(b_buf_bytes)
    return total


def kernel_instance_estimated_lds_bytes(ki: kernelInstance) -> int:
    return fp8_gemm_rowscale_estimated_lds_bytes(ki.tile_m, ki.tile_n)


def addressable_lds_bytes_for_gfx(gfx: str) -> int:
    return _addressable_lds_bytes_for_gfx(gfx)


def max_lds_bytes_for_tune() -> int:
    """Addressable LDS limit for the current target arch."""
    return get_shared_memory_per_block(fallback_gfx=get_gfx())


# ---------------------------------------------------------------------------
# Base tile menus (per the test_fp8_gemm_rowscale.py parametrize space)
# ---------------------------------------------------------------------------

# 4-wave: BLOCK_M, BLOCK_N must be >= 64 and multiples of 64 (see
# kernels/fp8_gemm_4wave.py:84). LDS budget on gfx950 (160 KiB addressable):
#   per-shape LDS = 256 * (tile_m + tile_n) bytes
#   so tile_m + tile_n <= 640. All entries below respect that.
#
# Coverage rationale (extends the previous 5-tile list):
#   - tile_n = 192 is added to cover N divisible by 192 (e.g. N=2112, 3072,
#     6144, 7168 in deepseek shapes); the previous menu had no non-power-of-2
#     tile_n at all.
#   - tile_m = 64 row provides a full N sweep for decode / token-gen shapes
#     where M is small.
#   - tile_m in {192, 320, 384} added for tall-M shapes where the 256-step is
#     too coarse and 128-step underutilises CUs.
_BASE_TILES_4WAVE = [
    # M = 64  (decode / token-gen)
    (64, 64), (64, 128), (64, 192), (64, 256), (64, 384), (64, 512),
    # M = 128
    (128, 64), (128, 128), (128, 192), (128, 256), (128, 320), (128, 384),
    # M = 192
    (192, 64), (192, 128), (192, 192), (192, 256), (192, 384),
    # M = 256
    (256, 64), (256, 128), (256, 192), (256, 256), (256, 320),
    # M = 320
    (320, 64), (320, 128), (320, 192),
    # M = 384
    (384, 64), (384, 128), (384, 192),
]

# 8-wave: BLOCK_M multiple of 128 (>=128); BLOCK_N multiple of 256 (>=256).
_BASE_TILES_8WAVE = [
    (128, 256),
    (256, 256),
    (256, 512),
]

_XCD_REMAP_VALS = (0, 1)


def _build_kernels_list():
    kl = {}
    idx = 0
    for xcd in _XCD_REMAP_VALS:
        for tm, tn in _BASE_TILES_4WAVE:
            kl[idx] = _ki(tm, tn, 4, xcd_remap=xcd)
            idx += 1
    # 8-wave has no xcd_remap dimension
    for tm, tn in _BASE_TILES_8WAVE:
        kl[idx] = _ki(tm, tn, 8, xcd_remap=0)
        idx += 1
    return kl


kernels_list_950 = _build_kernels_list()

# Defaults indexed by negative ids, matching the preshuffle convention.
# Selected from the test harness's parametrize space.
default_kernels_dict_950 = {
    (-1): _ki(256, 256, 8),                # large-shape SoTA pick
    (-2): _ki(128, 256, 8),                # skinny-M 8-wave
    (-3): _ki(256, 256, 4, xcd_remap=1),   # 4-wave fallback
    (-4): _ki(64, 64, 4, xcd_remap=1),     # small shape
}

# FP8 row-scale GEMM is gfx95* only; expose only the 950 lists for symmetry
# with the preshuffle module's naming.
arch = get_gfx()
if not arch.startswith(_FIXED_ARCH_PREFIX):
    # On non-gfx95* arch we still expose the lists (compile/test won't reach
    # the kernel anyway), but warn at import time via stderr is overkill.
    pass

kernels_list = kernels_list_950
default_kernels_dict = default_kernels_dict_950
