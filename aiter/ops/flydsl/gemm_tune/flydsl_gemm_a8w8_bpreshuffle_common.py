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
    "mxfp8": "MXF8",
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
    xcd_swizzle: int = 0  # 0=off, >0=group size for XCD remap
    split_k: int = 1  # 1=single-pass; >1=in-kernel bf16-atomic split-K (mxfp8)

    @property
    def name(self) -> str:
        qa = _DTYPE_SHORT.get(self.q_dtype_a, self.q_dtype_a.upper())
        qw = _DTYPE_SHORT.get(self.q_dtype_w, self.q_dtype_w.upper())
        dt = _DTYPE_SHORT.get(self.dtype, self.dtype.upper())
        # split_k == 1 emits no suffix, so single-pass names are byte-for-byte
        # identical to the legacy format (and the AOT/runtime parsers stay
        # backward compatible). split_k > 1 appends "_spk{N}".
        name = "_".join(
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
                            self.xcd_swizzle,
                        ],
                    )
                ),
                self.sScheduler.lower(),
            ]
        )
        if int(self.split_k) > 1:
            name += f"_spk{int(self.split_k)}"
        return name


def _ki(
    tile_m,
    tile_n,
    tile_k,
    lds_stage,
    cshuffle=0,
    async_copy=0,
    waves_per_eu=0,
    xcd_swizzle=0,
    scheduler="Default",
    q_dtype_a="fp8",
    q_dtype_w="fp8",
    dtype="bf16",
    split_k=1,
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
        xcd_swizzle,
        split_k,
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
    # mxfp8 carries 1-byte fp8 data (the e8m0 microscale is applied in-MMA, not
    # staged in LDS), so it sizes like fp8, not like a 2-byte type.
    elem_bytes = 1 if in_dtype in ("fp8", "int8", "int4", "fp4", "mxfp8") else 2
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
_XCD_SWIZZLE_VALS = (0, 4)

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


def _build_kernels_list(tiles_lds2, tiles_lds1, total_vgpr=512):
    tiles_by_lds = {2: tiles_lds2, 1: tiles_lds1}
    kl = {}
    idx = 0
    for wpe in _WAVES_PER_EU:
        for csh in _CSHUFFLE_VALS:
            for acp in _ASYNC_COPY_VALS:
                for xcd in _XCD_SWIZZLE_VALS:
                    for lds in _LDS_STAGES:
                        for tm, tn, tk in tiles_by_lds[lds]:
                            if wpe > 0 and wpe > _estimate_max_wpe(tm, tn, total_vgpr):
                                continue
                            kl[idx] = _ki(tm, tn, tk, lds, csh, acp, wpe, xcd)
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
    (-1): _ki(128, 128, 128, 2, 0, 0, 2, 0, "Default"),
    (-2): _ki(16, 64, 512, 2, 0, 0, 2, 0, "Default"),
    (-3): _ki(32, 64, 512, 2, 0, 0, 2, 0, "Default"),
    (-4): _ki(64, 256, 64, 2, 0, 0, 2, 0, "Default"),
    (-5): _ki(128, 128, 64, 2, 0, 0, 2, 0, "Default"),
    (-6): _ki(128, 64, 128, 2, 0, 0, 2, 0, "Default"),
    (-7): _ki(64, 256, 128, 2, 0, 0, 2, 0, "Default"),
}

default_kernels_dict_950 = {
    (-1): _ki(128, 256, 256, 2, 0, 0, 2, 0, "Default"),
    (-2): _ki(16, 64, 512, 2, 0, 0, 2, 0, "Default"),
    (-3): _ki(32, 64, 512, 2, 0, 0, 2, 0, "Default"),
    (-4): _ki(128, 128, 128, 2, 0, 0, 2, 0, "Default"),
}

arch = get_gfx()
if arch == "gfx942":
    kernels_list = kernels_list_942
    default_kernels_dict = default_kernels_dict_942
else:
    kernels_list = kernels_list_950
    default_kernels_dict = default_kernels_dict_950


# ---------------------------------------------------------------------------
# mxfp8 (fp8 E4M3 data + per-1x32 e8m0 microscale) -- gfx950 only
# ---------------------------------------------------------------------------
# mxfp8 reuses the SAME tile/combo sweep as the plain-fp8 list above, with two
# differences: (1) the dtype is "mxfp8", and (2) split_k is swept (the in-kernel
# bf16-atomic reduction helps the skinny-M / huge-K shapes, e.g. dsv4 M=1). The
# fp8 ``kernels_list`` is left byte-for-byte unchanged -- this is a separate dict
# with its own kernel ids so the two paths never collide.
#
# mxfp8 tile_k must be 128 or a multiple of 256 (each scaled MFMA spans 128 K;
# pairs of 128-K MFMAs share one 256-K e8m0 scale). split_k > 1 needs lds_stage=2,
# no cshuffle, and tile_k != 128 (mirrors preshuffle_gemm.py's kernel guards).
# Mirror of the runtime cap in gemm_kernels.PRESHUFFLE_SPLIT_K_MAX_TILES (kept as
# a local literal so this tune-side module stays import-light).
_MXFP8_SPLIT_K_MAX_TILES = 256
# split_k sweep is DISABLED for now (single-pass only). On gfx950 the mxfp8
# split-K path currently (1) drifts 5-11% from the fp32 reference for skinny-M /
# huge-K shapes -- inherent to its bf16 atomic reduction, which exceeds the
# tuner's error gate -- and (2) hits a GPU memory fault for at least one config
# (32x128x256, split_k=4). The instance/fits/validity machinery below still
# handles split_k > 1, so re-enable by widening this tuple once the kernel
# split-K path is hardened and the tuner compares split_k>1 against the
# split_k=1 output (not the fp32 ref).
_MXFP8_SPLIT_K_OPTIONS = (1,)


def _mxfp8_tile_k_ok(tk: int) -> bool:
    return int(tk) == 128 or (int(tk) % 256) == 0


def _mxfp8_tile_n_ok(tn: int) -> bool:
    # Root cause: the microscale path packs N by 2 (pack_N=2) via INTEGER
    # division -- preshuffle_gemm.py computes num_acc_n = tile_n // 64 (4 waves x
    # 16-wide MFMA) then _num_acc_n_packed = num_acc_n // 2. If num_acc_n is odd
    # the kernel silently drops the trailing acc_n (no error), so that slice of N
    # is never written -> wrong/zero output. num_acc_n must be even, i.e. tile_n
    # must be a multiple of 128. (plain fp8 uses pack_N=1, so tile_n % 64 is
    # enough -- that's why fp8 supports tile_n=64 but mxfp8 does not.)
    # Verified gfx950: tile_n=64 -> all-zero, 96 -> wrong, 192 -> ~mismatch.
    return (int(tn) % 128) == 0


def _mxfp8_tile_m_ok(tm: int) -> bool:
    # Root cause (same mechanism as _mxfp8_tile_n_ok, M side): the microscale path
    # packs M by 2 (pack_M=2) via integer division -- m_repeat = tile_m // 16 then
    # _m_repeat_packed = m_repeat // 2. m_repeat must be even, i.e. tile_m must be
    # a multiple of 32. Verified gfx950: tile_m=16 -> all-zero, tile_m=48 -> ~48%
    # mismatch.
    return (int(tm) % 32) == 0


def _build_mxfp8_kernels_list(tiles_lds2, tiles_lds1, total_vgpr=512):
    """Mirror of ``_build_kernels_list`` for mxfp8 (dtype + split_k sweep)."""
    tiles_by_lds = {2: tiles_lds2, 1: tiles_lds1}
    kl = {}
    idx = 0
    for wpe in _WAVES_PER_EU:
        for csh in _CSHUFFLE_VALS:
            for acp in _ASYNC_COPY_VALS:
                for xcd in _XCD_SWIZZLE_VALS:
                    for lds in _LDS_STAGES:
                        for tm, tn, tk in tiles_by_lds[lds]:
                            if not _mxfp8_tile_k_ok(tk):
                                continue
                            if not _mxfp8_tile_n_ok(tn):
                                continue
                            if not _mxfp8_tile_m_ok(tm):
                                continue
                            if tk == 128 and lds != 2:
                                # mxfp8 tile_k=128 requires lds_stage=2.
                                continue
                            if wpe > 0 and wpe > _estimate_max_wpe(tm, tn, total_vgpr):
                                continue
                            for spk in _MXFP8_SPLIT_K_OPTIONS:
                                if spk > 1 and (lds != 2 or csh != 0 or tk == 128):
                                    # split_k>1: lds=2, no cshuffle, tile_k!=128.
                                    continue
                                kl[idx] = _ki(
                                    tm, tn, tk, lds, csh, acp, wpe, xcd,
                                    q_dtype_a="mxfp8", q_dtype_w="mxfp8",
                                    dtype="bf16", split_k=spk,
                                )
                                idx += 1
    return kl


# fmt: off
kernels_list_mxfp8_950 = _build_mxfp8_kernels_list(
    _base_tiles_lds2_common + _base_tiles_lds2_950_extra, _base_tiles_lds1,
    total_vgpr=_vgpr_per_simd("gfx950"))
# fmt: on

# mxfp8 is gfx950-only (mfma_scale_f32_16x16x128_f8f6f4); no gfx942 list.
kernels_list_mxfp8 = kernels_list_mxfp8_950


def mxfp8_instance_valid(ki: kernelInstance) -> bool:
    """Shape-independent validity of an mxfp8 kernelInstance config.

    Encodes the kernel's internal constraints (tile_k, split_k vs lds/cshuffle,
    LDS budget) so an invalid instance is never emitted into the candidate pool.
    """
    if ki.q_dtype_a != "mxfp8" or ki.q_dtype_w != "mxfp8":
        return False
    if not _mxfp8_tile_k_ok(ki.tile_k):
        return False
    if not _mxfp8_tile_n_ok(ki.tile_n):
        return False
    if not _mxfp8_tile_m_ok(ki.tile_m):
        return False
    if ki.tile_k == 128 and ki.lds_stage != 2:
        return False
    if int(ki.split_k) > 1:
        if ki.lds_stage != 2 or ki.use_cshuffle_epilog != 0 or ki.tile_k == 128:
            return False
    if kernel_instance_estimated_lds_bytes(ki) > max_lds_bytes_for_tune():
        return False
    return True


def mxfp8_fits_shape(ki: kernelInstance, M: int, N: int, K: int) -> bool:
    """Shape-dependent fit: N % tile_n, K % tile_k, split-K divisibility + pool cap.

    M may be ragged (the kernel OOB-clips along M), so no M divisibility needed.
    """
    spk = int(ki.split_k)
    if N % ki.tile_n != 0 or K % ki.tile_k != 0:
        return False
    if spk > 1:
        if K % spk != 0 or (K // spk) % ki.tile_k != 0:
            return False
        n_tiles = ((M + ki.tile_m - 1) // ki.tile_m) * (N // ki.tile_n)
        if n_tiles > _MXFP8_SPLIT_K_MAX_TILES:
            return False
    return True
