# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import List

# Legacy cache policy = traits default for split-barrier & persistent a16w16 (see
# opus_gemm_traits_a16w16_gfx950.cuh).
_LEGACY_CACHECTL = (0, 17)

_GFX942_KERNEL_NAME_TAGS = {
    "a16w16_kbuf1_sk": "splitk_legacy",
    "a16w16_kbuf2v_sk": "splitk_p1",
    "a16w16_kbuf2v_bk128_sk": "splitk_p1_bk128",
    "a16w16_em3en4_lds1_pgr2_sk": "splitk_em3en4_lds1_pgr2",
    "a16w16_wave_k_coop": "wkc",
    "a16w16_wave_k_coop_accum": "wkc_accum",
    "a16w16_kbuf2v": "p1",
    "a16w16_kbuf2v_bk128": "p1_bk128",
    "a16w16_kbuf1": "legacy",
    "a16w16_quad_mfma32_kbuf1": "quad_mfma32",
    "a16w16_quad_mfma32_kbuf1_sk": "splitk_quad_mfma32_bf16ws",
}


@dataclass
class OpusGemmInstance:
    BLOCK_SIZE: int
    B_M: int
    B_N: int
    B_K: int
    T_M: int
    T_N: int
    W_M: int
    W_N: int
    W_K: int
    VEC_A: int
    VEC_B: int
    VEC_C: int
    GROUP_M: int
    GROUP_N: int
    GROUP_K: int
    kernel_tag: str
    output_dtypes: List[str] = field(default_factory=lambda: ["fp32_t"])
    # Flatmm-only. Defaults to 2 (match existing behavior for non-flatmm kernels).
    # Only emitted in the generated instance name when kernel_tag == "a16w16_flatmm".
    WG_PER_CU: int = 2
    # Compile-time OOB (out-of-bounds) tail handling.
    has_oob: bool = True
    # Cache policy for A/B loads (CDNA4 ISA Table 49). -1 = use traits default.
    # 0=LRU, 1=SC0(LLC Evict), 17=SC0+SC1(L2 Bypass).
    cachectl_a: int = -1
    cachectl_b: int = -1
    # 4g_safe variant flag. True = use the *_4g_safe_gfx950.cuh pipeline
    # header (per-WG-tight buffer-resource sizing -- safe for tensors
    # whose full extent exceeds 4 GiB). False = use the legacy header
    # (full row/col-band BR sizing which wraps at 4 GiB). Same Traits
    # struct and kargs struct either way; only the pipeline body and
    # kernel symbol differ. The kid families that set this to True live
    # under SPLITK_4G_SAFE_KIDS / NON_SPLITK_4G_SAFE_KIDS.
    is_4g_safe: bool = False

    # Optional arch prefix (e.g.
    arch_prefix: str = ""
    # Optional generated name tag override for same-pipeline variants.
    name_tag: str = ""
    # SplitK workspace storage dtype; splitK launchers still use fp32 tune dispatch.
    splitk_workspace_dtype: str = "fp32_t"

    # gfx1250 cluster/TDM split-K consumer tiling: "tileN" (split N) or
    # "tileM" (split M). Only consumed by the a16w16_cluster_tdm_splitk_ws tag.
    ctdm_layout: str = "tileN"

    # gfx1250 cluster_tdm_splitk_ws prefetch depth P (== LDS slots == in-flight
    # TDM count; producer keeps exactly this many TDMs in flight). 2 or 3.
    num_slots: int = 3
    # gfx1250 cluster_tdm_splitk_ws target WG/CU co-residency (1 or 2). 1 is
    # enforced via LDS padding in the traits; chosen by _ctdm_pick_configs() so
    # two WGs never oversubscribe a SIMD-pair's 256-request direct-copy budget.
    wg_per_cu: int = 2

    @property
    def name(self) -> str:
        parts = [
            "opus_gemm",
            "x".join(map(str, [self.BLOCK_SIZE, self.B_M, self.B_N, self.B_K])),
            "x".join(map(str, [self.T_M, self.T_N])),
            "x".join(map(str, [self.W_M, self.W_N, self.W_K])),
            "x".join(map(str, [self.GROUP_M, self.GROUP_N, self.GROUP_K])),
        ]
        if self.arch_prefix:
            parts.insert(1, self.arch_prefix)
        # tag inserts shift right by one slot when arch_prefix is set
        tag_at = 1 + (1 if self.arch_prefix else 0)
        if self.kernel_tag == "a16w16_flatmm":
            parts.insert(tag_at, "flatmm")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a16w16_flatmm_splitk":
            parts.insert(tag_at, "flatmm_splitk")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a16w16_persistent":
            parts.insert(tag_at, "persistent")
        elif self.kernel_tag == "a16w16_mono_tile":
            parts.insert(tag_at, "mono_tile")
        elif self.kernel_tag == "a16w16_cluster_tdm_splitk_ws":
            # gfx1250 fp32-workspace split-K with a separate reduce kernel.
            # Name it opus_gemm_gfx1250_splitk_* (note the "splitk_" segment) so
            # the reduce-TU arch detection in gen_instances.py -- which keys on
            # "opus_gemm_<arch>_splitk_" -- buckets it like the gfx942 splitk kids.
            # The T_M x T_N segment (1x2 for tileN, 2x1 for tileM) keeps the name
            # unique between the two consumer-tiling layouts.
            parts.insert(tag_at, "splitk_cluster_tdm_ws")
            # Prefetch depth P and WG/CU occupancy make each (tile, P, wg) symbol
            # unique (the producer + LDS-pad differ by these).
            parts.append(f"p{self.num_slots}w{self.wg_per_cu}")
        elif self.name_tag:
            parts.insert(tag_at, self.name_tag)
        elif self.kernel_tag in _GFX942_KERNEL_NAME_TAGS:
            name_tag = _GFX942_KERNEL_NAME_TAGS[self.kernel_tag]
            parts.insert(tag_at, name_tag)
        if not self.has_oob:
            parts.append("nooob")
        if self.is_4g_safe:
            parts.append("4g_safe")
        # Legacy cache policy = traits default for split-barrier & persistent a16w16: CACHECTL_A=0
        # (LRU), CACHECTL_B=17 (BYPASS_L2).
        if (self.cachectl_a, self.cachectl_b) != _LEGACY_CACHECTL and (
            self.cachectl_a >= 0 or self.cachectl_b >= 0
        ):
            parts.append(f"cA{self.cachectl_a}cB{self.cachectl_b}")
        return "_".join(parts)


def _a16w16(bs, bm, bn, bk, tn, wm, wn, wk, has_oob=True, cachectl_a=0, cachectl_b=17):
    """Factory for a16w16 split-barrier kid instances.

    cachectl_a / cachectl_b default to (0, 17) = (LRU, BYPASS_L2), which
    matches the traits-default cache policy for the split-barrier pipeline
    (see opus_gemm_a16w16_traits_gfx950 in
    csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh).
    This is the "legacy" policy used by KID 4..9 and 1004..1009 -- the
    `_LEGACY_CACHECTL` special-case in OpusGemmInstance.name keeps these
    kids emitting the bare `..._0x0x0` symbol (no `_cA0cB17` suffix) so
    the production heuristic dispatcher and the opus tuned CSV stay
    bit-compatible.
    """
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    inst = OpusGemmInstance(
        bs,
        bm,
        bn,
        bk,
        2,
        tn,
        wm,
        wn,
        wk,
        vec,
        vec,
        4,
        0,
        0,
        0,
        "a16w16",
        ["fp32_t", "bf16_t"],
        has_oob=has_oob,
    )
    inst.cachectl_a = cachectl_a
    inst.cachectl_b = cachectl_b
    return inst


def _a16w16_flatmm_splitk(bm, bn, bk, wg_per_cu, has_oob=True):
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        256,
        bm,
        bn,
        bk,
        2,
        1,  # T_M, T_N
        16,
        16,
        32,  # MFMA 16x16x32
        vec,
        vec,
        4,  # VEC
        0,
        0,
        0,  # GROUP (unused)
        "a16w16_flatmm_splitk",
        ["fp32_t"],
        wg_per_cu,
        has_oob=has_oob,
    )


def _a16w16_flatmm(bm, bn, bk, wg_per_cu):
    # Flatmm locked config (per gcnasm/opus_fmm/INTEGRATION.md): BLOCK_SIZE=256, T_M=2, T_N=1,
    # MFMA=(16,16,32), VEC=(8,8,4), HAS_BIAS...
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        256,
        bm,
        bn,
        bk,
        2,
        1,  # T_M, T_N (T_N hardcoded to 1 for the warp-spec pipeline)
        16,
        16,
        32,  # MFMA 16x16x32
        vec,
        vec,
        4,  # VEC
        0,
        0,
        0,  # GROUP (unused)
        "a16w16_flatmm",
        ["bf16_t", "fp32_t"],
        wg_per_cu,
    )


# fmt: off
# --- per-pipeline kernel instance lists ---
a8w8_scale_kernels_list = {
    1: OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
}

a8w8_kernels_list = {
    2: OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0, "a8w8", ["fp32_t"]),
}

a16w16_kernels_list = {
    # -- MFMA 16x16x32, T_N=2, BS=256 (2-block/CU capable) --
    # 3:  _a16w16(256, 128, 128, 32,  2, 16, 16, 32),  # disabled: intermittent accuracy (suspected compiler issue with VGPR=104/AGPR=64)
    4:  _a16w16(256, 128, 256, 32,  2, 16, 16, 32),
    5:  _a16w16(256, 256, 128, 32,  2, 16, 16, 32),
    # -- MFMA 16x16x32, T_N=4, BS=512 (1-block/CU) --
    6:  _a16w16(512, 128, 128, 64,  4, 16, 16, 32),
    7:  _a16w16(512, 256, 128, 64,  4, 16, 16, 32),
    8:  _a16w16(512, 128, 256, 64,  4, 16, 16, 32),
    9:  _a16w16(512, 256, 256, 64,  4, 16, 16, 32),  # existing / current default
}

# Removed (kids 100-115, a16w16_flatmm non-splitk): Rationale: the non-splitk a16w16_flatmm
# pipeline has two latent correctness b...
a16w16_flatmm_kernels_list = {}

# 11 splitk tiles mirroring gcnasm/opus_fmm/flatmm_a16w16_4wave_wasp_splitk.cc -t 0..10 dispatch
# exactly: * 8 WG_PER_CU=2 tiles (...
a16w16_flatmm_splitk_kernels_list = {
    # WG_PER_CU=2, cc tile 0..7
    200: _a16w16_flatmm_splitk( 64,  64,  64, 2),   # cc tile 0: M>=128 sweet spot (default)
    201: _a16w16_flatmm_splitk( 32,  32,  64, 2),   # cc tile 1
    202: _a16w16_flatmm_splitk( 32,  32, 128, 2),   # cc tile 2
    203: _a16w16_flatmm_splitk( 32,  64,  64, 2),   # cc tile 3
    204: _a16w16_flatmm_splitk( 32, 128,  64, 2),   # cc tile 4
    205: _a16w16_flatmm_splitk( 64,  32,  64, 2),   # cc tile 5
    206: _a16w16_flatmm_splitk( 64,  32, 128, 2),   # cc tile 6: recommended for medium M
    207: _a16w16_flatmm_splitk(128,  32,  64, 2),   # cc tile 7
    # WG_PER_CU=1, cc tile 8..10 (160 KB/wg LDS; zero VGPR spill only)
    208: _a16w16_flatmm_splitk( 64,  64, 128, 1),   # cc tile 8: deep K, high compute/load ratio
    209: _a16w16_flatmm_splitk(256,  32,  64, 1),   # cc tile 9: very tall, narrow N
    210: _a16w16_flatmm_splitk( 32, 256,  64, 1),   # cc tile 10: very wide, narrow M
    # Tile coverage extension (kids 211..223): B_M=96 OR B_N=96 lanes for shapes whose M or N is a
    # multiple of 96.
    211: _a16w16_flatmm_splitk( 32,  96,  64, 1),   # pfk=9, VGPR=176/512, AGPR=24
    212: _a16w16_flatmm_splitk( 32,  96,  64, 2),   # pfk=4, VGPR=176/256, AGPR=24
    213: _a16w16_flatmm_splitk( 32,  96, 128, 1),   # pfk=4, VGPR=288/512, AGPR=24
    214: _a16w16_flatmm_splitk( 64,  96,  64, 1),   # pfk=7, VGPR=192/512, AGPR=48
    215: _a16w16_flatmm_splitk( 64,  96,  64, 2),   # pfk=3, VGPR=192/256, AGPR=48
    216: _a16w16_flatmm_splitk( 64,  96, 128, 1),   # pfk=3, VGPR=320/512, AGPR=48
    217: _a16w16_flatmm_splitk( 96,  32,  64, 1),   # pfk=9, VGPR=144/512, AGPR=24
    218: _a16w16_flatmm_splitk( 96,  32,  64, 2),   # pfk=4, VGPR=144/256, AGPR=24
    219: _a16w16_flatmm_splitk( 96,  32, 128, 1),   # pfk=4, VGPR=224/512, AGPR=24
    220: _a16w16_flatmm_splitk( 96,  64,  64, 1),   # pfk=7, VGPR=176/512, AGPR=48
    221: _a16w16_flatmm_splitk( 96,  64,  64, 2),   # pfk=3, VGPR=176/256, AGPR=48
    222: _a16w16_flatmm_splitk( 96,  64, 128, 1),   # pfk=3, VGPR=288/512, AGPR=48
    223: _a16w16_flatmm_splitk( 96,  96,  64, 2),   # pfk=3, VGPR=208/256, AGPR=72  (81% VGPR -- watch)
}

# non-OOB variants: kid + 1000, same tile but HAS_OOB=false.
a16w16_kernels_list_nooob = {
    kid + 1000: _a16w16(
        inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
        inst.T_N, inst.W_M, inst.W_N, inst.W_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_kernels_list.items()
}

# CPOL variants for a16w16: 3 policies per kid, tuner picks best per shape.
_CACHECTL_CONFIGS = [
    (2000, 1, 17, "Mheavy"),   # kid_offset, cachectl_a, cachectl_b
    (3000, 17, 1, "Nheavy"),
    (4000, 0,  0, "balanced"),
]
a16w16_kernels_list_cpol = {}
for offset, ca, cb, _tag in _CACHECTL_CONFIGS:
    for kid, inst in a16w16_kernels_list.items():
        new_inst = _a16w16(
            inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
            inst.T_N, inst.W_M, inst.W_N, inst.W_K,
        )
        new_inst.cachectl_a = ca
        new_inst.cachectl_b = cb
        a16w16_kernels_list_cpol[kid + offset] = new_inst

a16w16_kernels_list_cpol_nooob = {}
for offset, ca, cb, _tag in _CACHECTL_CONFIGS:
    for kid, inst in a16w16_kernels_list.items():
        new_inst = _a16w16(
            inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
            inst.T_N, inst.W_M, inst.W_N, inst.W_K, has_oob=False,
        )
        new_inst.cachectl_a = ca
        new_inst.cachectl_b = cb
        a16w16_kernels_list_cpol_nooob[kid + offset + 1000] = new_inst

a16w16_flatmm_splitk_kernels_list_nooob = {
    kid + 1000: _a16w16_flatmm_splitk(
        inst.B_M, inst.B_N, inst.B_K, inst.WG_PER_CU, has_oob=False,
    )
    for kid, inst in a16w16_flatmm_splitk_kernels_list.items()
}

# -- a16w16 persistent (M-outer + N-fast XCD swizzle) ---------------------- Pipeline:
# csrc/opus_gemm/include/gfx950/opus_gemm_pi...


def _a16w16_persistent(bm, bn, bk, has_oob=True,
                       cachectl_a=0, cachectl_b=17):
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    inst = OpusGemmInstance(
        512,         # BLOCK_SIZE
        bm, bn, bk,  # BLOCK
        2, 4,        # T_M, T_N
        16, 16, 32,  # W_M, W_N, W_K  (MFMA 16x16x32)
        vec, vec, 4, # VEC
        0, 0, 0,     # GROUP (unused for persistent)
        "a16w16_persistent",
        ["bf16_t", "fp32_t"],
        has_oob=has_oob,
    )
    inst.cachectl_a = cachectl_a
    inst.cachectl_b = cachectl_b
    return inst


# 4-tile sweep, all B_K=64.
_PERSISTENT_TILES = [
    # (B_M, B_N, B_K)
    (256, 256, 64),  # tile 0: mouter default; 32Kx2Kx7K best 1208 TFLOPS
    (128, 256, 64),  # tile 1: narrow M
    (256, 128, 64),  # tile 2: narrow N
    (128, 128, 64),  # tile 3: small
]

# Legacy (300..303): cachectl == (0, 17).
a16w16_persistent_kernels_list = {
    300 + i: _a16w16_persistent(bm, bn, bk)
    for i, (bm, bn, bk) in enumerate(_PERSISTENT_TILES)
}

# Cpol variants (304..315): 3 groups x 4 tiles, mirroring _CACHECTL_CONFIGS but with a single
# compact base offset per cpol group.
_PERSISTENT_CPOL_GROUPS = [
    # (base_kid, cachectl_a, cachectl_b)
    (304,  1, 17),   # Mheavy
    (308, 17,  1),   # Nheavy
    (312,  0,  0),   # balanced
]
a16w16_persistent_kernels_list_cpol = {}
for _base, _ca, _cb in _PERSISTENT_CPOL_GROUPS:
    for i, (bm, bn, bk) in enumerate(_PERSISTENT_TILES):
        a16w16_persistent_kernels_list_cpol[_base + i] = _a16w16_persistent(
            bm, bn, bk, cachectl_a=_ca, cachectl_b=_cb
        )

# Nooob mirrors at +1000 for both legacy (1300..1305) and cpol (1306..1323).
# Explicit cachectl inheritance keeps name() consistent with parents.
a16w16_persistent_kernels_list_nooob = {
    kid + 1000: _a16w16_persistent(
        inst.B_M, inst.B_N, inst.B_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_persistent_kernels_list.items()
}
a16w16_persistent_kernels_list_cpol_nooob = {
    kid + 1000: _a16w16_persistent(
        inst.B_M, inst.B_N, inst.B_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_persistent_kernels_list_cpol.items()
}

# -- a16w16 mono-tile (single-MMA-per-K-iter, 8 waves) ---------------------
#
# Pipeline:
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh
# Traits:
#   csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh
#   :: opus_gemm_a16w16_mono_tile_traits_gfx950
#
# Locks: BLOCK_SIZE=512, T_M=2, T_N=4, T_K=1, W_M=W_N=16, W_K=32 (MFMA
# 16x16x32 BF16), VEC=8. Single v_c accumulator over the full B_M x B_N
# tile per K iter (no quad-subtile, no split barrier). Intrinsically
# non-OOB (launcher enforces M%B_M==N%B_N==K%B_K==0) and HAS_BIAS=false
# (launcher rejects non-empty bias up front). No splitK.
#
# B_M <= 192 hard cap. The 7 tiles below were picked to cover
# (M-bucket x N-bucket) combinations not already served well by the
# persistent / splitk families.


def _a16w16_mono_tile(bm, bn, bk):
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        512,         # BLOCK_SIZE (8 waves * 64)
        bm, bn, bk,  # BLOCK
        2, 4,        # T_M, T_N
        16, 16, 32,  # W_M, W_N, W_K  (MFMA 16x16x32)
        vec, vec, vec,  # VEC_A=VEC_B=VEC_C=8
        0, 0, 0,     # GROUP (unused)
        "a16w16_mono_tile",
        ["bf16_t", "fp32_t"],
        has_oob=False,
    )


# 5 mono-tile tiles, kids 1400..1404. Kid range deliberately starts at
# 1400 (above the persistent +1000 nooob mirror range that ends at 1323)
# and below the next reserved family slot. No "base/nooob" mirror split:
# mono-tile is non-OOB by construction, so kids land in the >=1000 band
# the way other families' nooob mirrors do.
#
# B_K=128 tiles (e.g. (64,256,128), (128,128,128)) are intentionally
# excluded: the pipeline uses 2x smem_a + 3x smem_b (A double-buffered,
# B triple-buffered as r0/r1/w), which pushes those tiles to 165-231 KiB
# of LDS -- over gfx950's 160 KiB budget. Re-enable only after the
# pipeline drops B to two slots.
_MONO_TILE_TILES = [
    # (B_M, B_N, B_K)
    (192, 256, 64),   # 1400
    (128, 256, 64),   # 1401
    (192, 128, 64),   # 1402
    (128, 128, 64),   # 1403
    ( 64, 128, 64),   # 1404
]
a16w16_mono_tile_kernels_list = {
    1400 + i: _a16w16_mono_tile(bm, bn, bk)
    for i, (bm, bn, bk) in enumerate(_MONO_TILE_TILES)
}

# -- 4g_safe variants (offset +5000) ---------------------------------------
#
# Per-WG-tight buffer-resource sizing pipelines that handle tensors whose
# full extent exceeds 4 GiB without buffer_inst num_records wrap. Same
# Traits / kargs as their legacy siblings; only the pipeline header and
# kernel symbol differ. See
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_4g_safe_gfx950.cuh
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_4g_safe_gfx950.cuh
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_4g_safe_gfx950.cuh
#
# Offset choice: +5000 sits above the cpol band (which uses +2000/+3000/+4000)
# and well clear of the nooob mirror band (+1000). 4g_safe kids carry HAS_OOB
# from their parent (M/N tail is absorbed by the per-WG BR num_records, so
# the per-thread predicate is structurally a no-op for valid in-tile threads;
# we still emit both has_oob variants for consistency with the legacy axis).
_FOUR_G_SAFE_OFFSET = 5000


def _make_4g_safe(inst: "OpusGemmInstance") -> "OpusGemmInstance":
    """Clone an OpusGemmInstance with is_4g_safe=True; everything else
    (kernel_tag, traits, kargs, BLOCK/B_*/T_*/W_*/VEC_*, cachectl, has_oob)
    is inherited verbatim. The codegen dispatch in gen_instances.py reads
    is_4g_safe to pick the 4g_safe pipeline header + kernel symbol."""
    from dataclasses import replace
    return replace(inst, is_4g_safe=True)


a16w16_kernels_list_4g_safe = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_kernels_list.items()
}
a16w16_kernels_list_4g_safe_nooob = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_kernels_list_nooob.items()
}
a16w16_persistent_kernels_list_4g_safe = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_persistent_kernels_list.items()
}
a16w16_persistent_kernels_list_4g_safe_nooob = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_persistent_kernels_list_nooob.items()
}
a16w16_mono_tile_kernels_list_4g_safe = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_mono_tile_kernels_list.items()
}


# -- gfx942 kernel lists ------------------------------------------------ Kid offset: gfx942
GFX942_KID_OFFSET = 10000


def _a16w16_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Factory for gfx942 a16w16 kbuf1-large-tile kid instances (kid 10000,
    MFMA 16x16x16). Same algorithm family as kbuf1 (4-phase, 2 barriers/iter)
    but with a larger tile + BS=512 + inline LDS-staged epilogue.
    """
    vec = 16 // 2  # bf16
    return OpusGemmInstance(
        bs, bm, bn, bk,
        2, tn,            # T_M, T_N
        wm, wn, wk,       # MFMA
        vec, vec, 4,      # VEC
        0, 0, 0,          # GROUP (unused)
        "a16w16_kbuf1_large_tile",
        ["fp32_t", "bf16_t"],
        arch_prefix="gfx942",
    )


def _a16w16_quad_mfma32_gfx942(bs, bm, bn, bk, tm, tn, wm, wn, wk):
    """gfx942 quad MFMA32 path."""
    vec = 16 // 2  # bf16
    return OpusGemmInstance(
        bs, bm, bn, bk,
        tm, tn,
        wm, wn, wk,
        vec, vec, 4,
        0, 0, 0,
        "a16w16_quad_mfma32_kbuf1",
        ["bf16_t"],
        arch_prefix="gfx942",
    )


def _a16w16_quad_mfma32_sk_bf16ws_gfx942(bs, bm, bn, bk, tm, tn, wm, wn, wk, group_m=0):
    """gfx942 quad MFMA32 splitK path with bf16 workspace."""
    vec = 16 // 2  # bf16
    inst = OpusGemmInstance(
        bs, bm, bn, bk,
        tm, tn,
        wm, wn, wk,
        vec, vec, 4,
        group_m, 0, 0,
        "a16w16_quad_mfma32_kbuf1_sk",
        ["fp32_t"],
        arch_prefix="gfx942",
    )
    inst.splitk_workspace_dtype = "bf16_t"
    return inst


def _a16w16_splitk_tag_gfx942(bs, bm, bn, bk, tn, wm, wn, wk, tag):
    """Factory for gfx942 splitK kids that write fp32 workspace + reduce."""
    vec = 16 // 2  # bf16
    return OpusGemmInstance(
        bs, bm, bn, bk,
        2, tn,
        wm, wn, wk,
        vec, vec, 4,
        0, 0, 0,
        tag,
        ["fp32_t"],
        arch_prefix="gfx942",
    )


def _a16w16_kbuf1_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK 4-phase split-barrier, E_M>=2 OK."""
    return _a16w16_splitk_tag_gfx942(
        bs, bm, bn, bk, tn, wm, wn, wk, "a16w16_kbuf1_sk"
    )


def _with_bf16_splitk_workspace(inst, name_tag):
    """Variant marker: same splitK pipeline, bf16 workspace + generated name tag."""
    inst.name_tag = name_tag
    inst.splitk_workspace_dtype = "bf16_t"
    return inst


def _a16w16_kbuf1_sk_bf16ws_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK 4-phase split-barrier with bf16 workspace."""
    inst = _a16w16_kbuf1_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk)
    return _with_bf16_splitk_workspace(inst, "splitk_legacy_bf16ws")


def _a16w16_kbuf2v_sk_bf16ws_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK P1 with bf16 workspace."""
    inst = _a16w16_kbuf2v_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk)
    return _with_bf16_splitk_workspace(inst, "splitk_p1_bf16ws")


def _a16w16_kbuf2v_bk128_sk_bf16ws_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK P1 B_K=128 with bf16 workspace."""
    inst = _a16w16_kbuf2v_bk128_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk)
    return _with_bf16_splitk_workspace(inst, "splitk_p1_bk128_bf16ws")


# gfx942 P1-family non-splitK factories (siblings of corresponding splitK kids).
def _a16w16_p1_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Non-splitK P1 (K-dbuf depth=2 + V-dbuf), sibling of 10201."""
    vec = 16 // 2  # bf16
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_kbuf2v_bk128_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Non-splitK P1 + B_K=128 sub-K decomp, sibling of 10203."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v_bk128", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_kbuf1_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Non-splitK 4-phase legacy (E_M=2 supported), sibling of 10202."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf1", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_kbuf2v_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK P1 (K-dbuf depth=2 + V-dbuf), fp32 workspace + reduce."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v_sk", ["fp32_t"], arch_prefix="gfx942",
    )


def _a16w16_kbuf2v_bk128_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK P1 + B_K=128 sub-K decomp."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v_bk128_sk", ["fp32_t"], arch_prefix="gfx942",
    )


def _a16w16_wave_k_coop_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Wave-K-cooperative small-M/N kid; tn partitions waves over N."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 1, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_wave_k_coop", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_wave_k_coop_accum_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Wave-K-cooperative splitK atomic accumulate path."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 1, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_wave_k_coop_accum", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_em3en4_lds1_pgr2_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK EM3EN4: host 128x96, device 96x128 LDSB1."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_em3en4_lds1_pgr2_sk", ["fp32_t"], arch_prefix="gfx942",
    )


# gfx942 kid registry -- flat two-bucket layout.

gfx942_nosplit_kernels_list = {
    10000: _a16w16_gfx942        (512, 128, 128,  64,    4, 16, 16, 16),   # kbuf1_large_tile (4-phase, big tile)
    10001: _a16w16_p1_gfx942     (256,  64,  64,  64,    2, 16, 16, 16),   # P1 depth=2 sibling of 10201
    10002: _a16w16_kbuf1_gfx942 (256, 128,  64,  64,    2, 16, 16, 16),   # legacy 4-phase E_M=2 sibling of 10202
    10003: _a16w16_kbuf2v_bk128_gfx942(256, 64,  64, 128,    2, 16, 16, 16),   # P1 B_K=128 sibling of 10203
    10006: _a16w16_quad_mfma32_gfx942(256, 256, 256, 32, 2, 2, 32, 32, 8), # quad MFMA32 pipeline
    10300: _a16w16_wave_k_coop_gfx942(512, 16, 16, 64,    1, 16, 16, 16),  # wave-K-coop 16x16, T_K=8
    10301: _a16w16_wave_k_coop_gfx942(512, 16, 32, 32,    1, 16, 16, 16),  # WKC 16x32, B_K=32
    10302: _a16w16_wave_k_coop_gfx942(512, 32, 16, 64,    1, 16, 16, 16),  # WKC 32x16, aliased partial
    10303: _a16w16_wave_k_coop_gfx942(256, 32, 32, 64,    1, 16, 16, 16),  # WKC 32x32, T_K=4
    10305: _a16w16_wave_k_coop_gfx942(512, 16, 32, 64,    1, 16, 16, 16),  # WKC 16x32, B_K=64
    10310: _a16w16_wave_k_coop_accum_gfx942(256, 16, 16, 64, 1, 16, 16, 16),  # WKC 16x16 split8 atomic accumulate
    10311: _a16w16_wave_k_coop_accum_gfx942(512, 16, 32, 32, 1, 16, 16, 16),  # WKC 16x32 split8 atomic accumulate
    10312: _a16w16_wave_k_coop_accum_gfx942(512, 32, 16, 64, 1, 16, 16, 16),  # WKC 32x16 split8 atomic accumulate
    10313: _a16w16_wave_k_coop_accum_gfx942(256, 32, 32, 64, 1, 16, 16, 16),  # WKC 32x32 split8 atomic accumulate
    10314: _a16w16_wave_k_coop_accum_gfx942(256, 64, 16, 64, 1, 16, 16, 16),  # WKC 64x16 split8 atomic accumulate
}

gfx942_splitk_kernels_list = {
    10200: _a16w16_kbuf1_sk_gfx942      (512, 128, 128,  64,    4, 16, 16, 16),                # legacy 4-phase large tile
    10201: _a16w16_kbuf2v_sk_gfx942     (256,  64,  64,  64,    2, 16, 16, 16),                # P1 depth=2 + V-dbuf
    10202: _a16w16_kbuf1_sk_gfx942      (256, 128,  64,  64,    2, 16, 16, 16),                # legacy 4-phase mid tile
    10203: _a16w16_kbuf2v_bk128_sk_gfx942(256, 64,  64, 128,    2, 16, 16, 16),                # P1 B_K=128 sub-K decomp
    10204: _a16w16_em3en4_lds1_pgr2_sk_gfx942 (256, 128,  96, 128,    2, 16, 16, 16),                # EM3EN4 LDS1/PGR2 hipb-orientation (host 128M x 96N)
    10205: _a16w16_kbuf1_sk_gfx942      (512,  64, 128,  64,    4, 16, 16, 16),                # legacy 4-phase M64 x N128
    10210: _a16w16_kbuf1_sk_bf16ws_gfx942(512, 128, 128,  64,    4, 16, 16, 16),                # legacy 4-phase large tile + bf16 workspace
    10211: _a16w16_kbuf2v_sk_bf16ws_gfx942(256,  64,  64,  64,    2, 16, 16, 16),                # P1 depth=2 + bf16 workspace
    10213: _a16w16_kbuf2v_bk128_sk_bf16ws_gfx942(256, 64,  64, 128,    2, 16, 16, 16),           # P1 B_K=128 + bf16 workspace
    10216: _a16w16_quad_mfma32_sk_bf16ws_gfx942(256, 256, 256, 32, 2, 2, 32, 32, 8, group_m=6),  # 10006 bf16 splitK sibling, group-M=6
}

# NOTE: 10402 (a16w16_naive_64x64) was removed -- 32.85us never matched WKC's
# 11.88us on tuned shapes (bf16_tuned_ge...

gfx942_kernels_list = {**gfx942_nosplit_kernels_list, **gfx942_splitk_kernels_list}

# -- gfx1250 kernel lists ----------------------------------------------------
# Kid offset: gfx1250 kids live in the 20000+ range, disjoint from gfx950
# (<10000) and gfx942 (50000+). Today only the cluster/TDM split-K (atomic
# fp32 reduction) pipeline is wired (打通阶段：fp32 output, no bias).
GFX1250_KID_OFFSET = 20000


def _a16w16_cluster_tdm_splitk_ws_gfx1250(bm, bn, bk, layout, num_slots=3, wg_per_cu=2):
    """Factory for the gfx1250 a16w16 cluster/TDM split-K (workspace + reduce) kid.

    Locked geometry from the kernel base
    (demon_gcn/wmma_opus_rdna4/gemm_a16w16_cluster_tdm_splitk_reduce_4wave.cc):
    BLOCK_SIZE=128 (4 waves x 32 = 2 producer + 2 consumer), MFMA 16x16x32,
    NO-CLUSTER (one WG per B_M x B_N tile). The main kernel WMMA-accumulates in
    fp32 and PLAIN-stores each split's partial into an fp32 workspace; a separate
    reduce kernel sums the split slices, folds bias, and casts to the Y dtype.
    output_dtypes = ["fp32_t"] (only the fp32-workspace main kernel is
    instantiated; Y bf16/fp32 is a runtime decision in the reduce kernel).

    layout: "tileN" (consumers split N; B_N>=32) -> T_M=1, T_N=2;
            "tileM" (consumers split M; B_M>=32) -> T_M=2, T_N=1.
    """
    vec = 16 // 2  # bf16 -> VEC_A = VEC_B = 8
    t_m, t_n = (2, 1) if layout == "tileM" else (1, 2)
    return OpusGemmInstance(
        128,            # BLOCK_SIZE (4 waves x 32 lanes)
        bm, bn, bk,
        t_m, t_n,       # T_M, T_N (encodes the consumer tiling layout)
        16, 16, 32,     # MFMA 16x16x32
        vec, vec, 8,    # VEC_A, VEC_B, VEC_C
        0, 0, 0,        # GROUP (unused)
        "a16w16_cluster_tdm_splitk_ws",
        ["fp32_t"],
        arch_prefix="gfx1250",
        ctdm_layout=layout,
        num_slots=num_slots,
        wg_per_cu=wg_per_cu,
    )


def _ctdm_pick_configs(bm, bn, bk):
    """Resource-feasible (P, wg_per_cu) configs for a gfx1250 cluster_tdm tile.

    Hardware prerequisites (gfx1250, per CU):
      * Direct-copy TDM budget: 256 256-byte requests per SIMD-pair (A and B sit
        on separate pairs). The per-TDM (one B_K slot) request count is
            req = rows * B_K * 2 / 256        (rows = B_M for A, B_N for B)
        2 WG/CU share a pair UNCONTROLLED -> each operand must be < 128; a single
        WG must be < 256. (req == 256 deadlocks the TDM engine -- the original
        32x256x128 hang.)
      * LDS: 320 KB / CU. LDS(P) = P * (B_M + B_N) * (B_K + 8) * 2 bytes.
        2 WG/CU need LDS(P) <= 160 KB; 1 WG/CU needs <= 320 KB.
      * VGPR (1024/SIMD, 512/wave at 2 WG/CU) is not the binding constraint for
        the current tiles and is left to the compiler.

    Returns a list of (num_slots P, wg_per_cu) for P in {3, 2}, picking the max
    feasible wg per P. Empty if the tile cannot run at any P (req >= 256).
    """
    rpr = bk // 128                       # 256B-req rows-multiplier (B_K/128)
    req_a = bm * rpr                       # per-TDM A request count
    req_b = bn * rpr                       # per-TDM B request count
    pitch = bk + 8                         # bf16 padded row pitch
    out = []
    # Prefetch depth P in {3, 2}: the run-ahead producer supports both (lower P
    # = lower LDS, can enable 2 WG/CU when P=3 LDS > 160 KB).
    for P in (3, 2):
        lds = P * (bm + bn) * pitch * 2
        if lds > 320 * 1024:
            continue                       # won't fit even 1 WG/CU
        if req_a < 128 and req_b < 128 and lds <= 160 * 1024:
            out.append((P, 2))             # 2 WG/CU safe
        elif req_a < 256 and req_b < 256:
            out.append((P, 1))             # force 1 WG/CU (LDS-pad in traits)
        # else: req >= 256 on some operand -> not runnable at this P
    return out


# Initial tile set seeded from the feasible no-cluster sweep
# (demon_gcn/wmma_opus_rdna4/instances_full_nocluster_feasible.csv), curated to
# the gfx1250 untuned shapes (small M / large N / large K).
#
# SCOPE (on-hardware validated, see op_tests/test_opus_gfx1250_ws.py -- 156/156):
# both the small-M tileN tiles and the fully generalized M/N tiles are wired:
#   * tileN: B_M==16, B_N>=32 (kExpN = B_N/32; N-wave-split + register-expand).
#   * tileM: B_M>=32 (kExpM = B_M/32) with any B_N (kExpN = B_N/16).
# Two earlier generalization bugs have been FIXED (2026-06):
#   (a) kExpM>1 && kExpN>1 -> NaN at the software-pipeline tail (per-split
#       k_steps%3==2): the sched_group_barrier DS/WMMA counts were hard-coded
#       for the kExpM==kExpN==1 base; now scaled by the register expansion in
#       the traits header (kSchedDsCount / kSchedWmmaCount).
#   (b) tileN with kExpN>1 (B_N>32, kTileN=2) -> wrong values: the B-read
#       N-decomposition order (make_layout_rb_ctdm) disagreed with the C-store
#       order; B now mirrors A (kExpN outer, kTileN=wave_n inner).
# Candidate tiles (B_M, B_N, B_K, layout). Each is expanded across its
# resource-feasible (P, wg_per_cu) configs by _ctdm_pick_configs(); tiles whose
# per-TDM request count hits the 256 direct-copy limit on some operand (e.g.
# 32x256x128, 32x128x256) yield no config and are dropped automatically.
_GFX1250_CTDM_TILES = [
    # tileN family (B_M=16)
    (16, 32, 128, "tileN"),
    (16, 32, 256, "tileN"),
    (16, 32, 512, "tileN"),
    (16, 64, 128, "tileN"),
    (16, 128, 128, "tileN"),
    # tileM family (B_M>=32)
    (32, 32, 128, "tileM"),
    (32, 64, 128, "tileM"),
    (32, 128, 128, "tileM"),
    (32, 64, 256, "tileM"),
    (64, 16, 128, "tileM"),
    (64, 64, 128, "tileM"),
]

# Deterministic kid numbering: base 20000 + tile_index*8 + config_index. Each
# tile contributes up to 2 configs (P=3, P=2). Stride 8 keeps room.
gfx1250_kernels_list = {}
_GFX1250_KID_BASE = 20000
_GFX1250_KID_STRIDE = 8
# Stability gate on the consumer N-register-expansion kExpN = B_N/(16*kTileN).
# Two compounding hazards on high-expansion tiles, both now fixed in the consumer:
#  (1) WMMA-source WAR (MI400 SPG 4.6.12.1): v_wmma_*_16x16x32_bf16 reads its
#      Matrix A/B over 16 passes; an instruction overwriting those source VGPRs
#      within ~8 co-exec slots corrupts. The compiler only inserts the required
#      gap for VALU/XDL writers, NOT for ds_load (VMEM) writers, so a next-round
#      ds_load overwriting the source silently raced.
#  (2) DScnt overflow: a 2-round-overlap prefetch keeps 2 rounds of ds in flight
#      (up to 72 > the 6-bit DScnt limit of 63 for kExpN=8).
# Fix: round-granular ping-pong prefetch -- load round i+1 into the OTHER buffer
# BEFORE round i's WMMA (distinct VGPRs => no WAR), but DRAIN each round's ds
# immediately (<=1 round = <=36 in flight => no DScnt overflow). Stress-verified
# 8/8 at the hardest config (splitK=4,K=2880) for kExpN up to 8 (32x128 incl.).
# Guard kept as a safety valve; current tiles top out at kExpN=8.
_GFX1250_MAX_KEXPN = 8
for _ti, (_bm, _bn, _bk, _layout) in enumerate(_GFX1250_CTDM_TILES):
    _ktile_n = 1 if _layout == "tileM" else 2
    _kexp_n = _bn // (16 * _ktile_n)
    if _kexp_n > _GFX1250_MAX_KEXPN:
        continue  # unstable high-expansion tile: skip (kids disabled)
    for _ci, (_P, _wg) in enumerate(_ctdm_pick_configs(_bm, _bn, _bk)):
        _kid = _GFX1250_KID_BASE + _ti * _GFX1250_KID_STRIDE + _ci
        gfx1250_kernels_list[_kid] = _a16w16_cluster_tdm_splitk_ws_gfx1250(
            _bm, _bn, _bk, _layout, num_slots=_P, wg_per_cu=_wg
        )

GFX1250_BASE_KIDS = frozenset(gfx1250_kernels_list.keys())

# combined list (used by production gen_instances / dispatch)
kernels_list = {
    **a8w8_scale_kernels_list,
    **a8w8_kernels_list,
    **a16w16_kernels_list,
    **a16w16_kernels_list_nooob,
    **a16w16_kernels_list_cpol,
    **a16w16_kernels_list_cpol_nooob,
    **a16w16_flatmm_kernels_list,
    **a16w16_flatmm_splitk_kernels_list,
    **a16w16_flatmm_splitk_kernels_list_nooob,
    **a16w16_persistent_kernels_list,
    **a16w16_persistent_kernels_list_cpol,
    **a16w16_persistent_kernels_list_nooob,
    **a16w16_persistent_kernels_list_cpol_nooob,
    **a16w16_mono_tile_kernels_list,
    **a16w16_kernels_list_4g_safe,
    **a16w16_kernels_list_4g_safe_nooob,
    **a16w16_persistent_kernels_list_4g_safe,
    **a16w16_persistent_kernels_list_4g_safe_nooob,
    **a16w16_mono_tile_kernels_list_4g_safe,
    **gfx942_kernels_list,
    **gfx1250_kernels_list,
}

default_kernels_dict = {
    (-1): OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
    (-2): OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0,     "a8w8",       ["fp32_t"]),
    (-3): _a16w16(512, 256, 256, 64, 4, 16, 16, 32),  # same as a16w16 #9
}
# fmt: on


# Subset-compile kid taxonomy (consumed by gen_instances.py for the `HEURISTIC_DEFAULT_KIDS ?

# Splitk kids: a16w16_flatmm_splitk pipeline (kid 200..223 + nooob mirror).
SPLITK_KIDS = (
    frozenset(a16w16_flatmm_splitk_kernels_list.keys())
    | frozenset(a16w16_flatmm_splitk_kernels_list_nooob.keys())
    | frozenset(gfx942_splitk_kernels_list.keys())
    | frozenset(gfx1250_kernels_list.keys())
)

# Non-splitk a16w16-family kids: split-barrier 4..9 + cpol/nooob mirrors, persistent 300..315 +
# cpol/nooob mirrors.
NON_SPLITK_KIDS = (
    frozenset(a16w16_kernels_list.keys())
    | frozenset(a16w16_kernels_list_nooob.keys())
    | frozenset(a16w16_kernels_list_cpol.keys())
    | frozenset(a16w16_kernels_list_cpol_nooob.keys())
    | frozenset(a16w16_persistent_kernels_list.keys())
    | frozenset(a16w16_persistent_kernels_list_cpol.keys())
    | frozenset(a16w16_persistent_kernels_list_nooob.keys())
    | frozenset(a16w16_persistent_kernels_list_cpol_nooob.keys())
    | frozenset(a16w16_mono_tile_kernels_list.keys())
    | frozenset(gfx942_nosplit_kernels_list.keys())  # 10000/10001/10002/10003/10300
)

# 4g_safe kid families. Per-WG-tight BR sizing -- selectable for any shape
# (M/N/K tail safe by BR num_records). All current 4g_safe kids are non-splitk
# (split-barrier / persistent / mono_tile variants). flatmm_splitk_4g_safe
# can be added later if needed.
SPLITK_4G_SAFE_KIDS = frozenset()
NON_SPLITK_4G_SAFE_KIDS = (
    frozenset(a16w16_kernels_list_4g_safe.keys())
    | frozenset(a16w16_kernels_list_4g_safe_nooob.keys())
    | frozenset(a16w16_persistent_kernels_list_4g_safe.keys())
    | frozenset(a16w16_persistent_kernels_list_4g_safe_nooob.keys())
    | frozenset(a16w16_mono_tile_kernels_list_4g_safe.keys())
)
# Per the opus kid pruning policy (project memory), 4g_safe kids are added
# additively -- they do NOT shadow or replace any existing kid.
NON_SPLITK_KIDS = NON_SPLITK_KIDS | NON_SPLITK_4G_SAFE_KIDS

# All-4g_safe-kids superset, consumed by the per-kid 4 GiB filter in
# opus_gemm_tune.py (legacy kids are dropped from the candidate pool when
# A/B/C bytes exceed UINT32_MAX; 4g_safe kids stay).
FOUR_G_SAFE_KIDS = SPLITK_4G_SAFE_KIDS | NON_SPLITK_4G_SAFE_KIDS

# Bias-aware kids: gfx950 split-barrier (4..9 + cpol/nooob mirrors), 4g_safe
# mirrors, and the entire splitk family (gfx950 a16w16_flatmm_splitk + gfx942
# splitk). Persistent excluded (launcher rejects bias).
BIAS_AWARE_KIDS = (
    frozenset(a16w16_kernels_list.keys())
    | frozenset(a16w16_kernels_list_nooob.keys())
    | frozenset(a16w16_kernels_list_cpol.keys())
    | frozenset(a16w16_kernels_list_cpol_nooob.keys())
    | frozenset(a16w16_kernels_list_4g_safe.keys())
    | frozenset(a16w16_kernels_list_4g_safe_nooob.keys())
    | SPLITK_KIDS
)

# Heuristic-dispatch fallback kids (gfx950).
HEURISTIC_DEFAULT_KIDS_GFX950 = frozenset(
    {
        # splitk fallback (small M / non-aligned big M)
        200,
        1200,  # cc tile 0: (64, 64, 64) WG=2
        206,
        1206,  # cc tile 6: (64, 32, 128) WG=2
        208,
        1208,  # cc tile 8: (64, 64, 128) WG=1
        # persistent fallback (large M, tile-aligned)
        300,
        1300,  # persistent (256, 256, 64)
    }
)

HEURISTIC_DEFAULT_KIDS_GFX942 = frozenset(
    {
        # gfx942 heuristic dispatcher fallbacks.
        10000,  # gfx942 split-barrier    512x128x128x64 16x16x16 (large problem)
        10001,  # gfx942 p1               256x64x64x64
        10002,  # gfx942 legacy           256x128x64x64
        10003,  # gfx942 p1_bk128         256x64x64x128
        10200,  # gfx942 splitk          512x128x128x64 16x16x16 (N > 128)
        10201,  # gfx942 splitk_p1        256x64x64x64  (depth=2 + workspace + reduce)
        10202,  # gfx942 splitk          256x128x64x64  16x16x16 (64 < N <= 128)
        10203,  # gfx942 splitk_p1_bk128  256x64x64x128 (B_K=128 Option B; dev/bench)
        10204,  # gfx942 splitk_em3en4_lds1_pgr2 256x128x96x128 hipb-orientation
        10205,  # gfx942 splitk_legacy    512x64x128x64 16x16x16
        10210,  # gfx942 splitk_legacy_bf16ws 512x128x128x64
        10211,  # gfx942 splitk_p1_bf16ws 256x64x64x64
        10213,  # gfx942 splitk_p1_bk128_bf16ws 256x64x64x128
        10300,  # gfx942 wave_k_coop     512x16x16x64 T_K=8
        10301,  # gfx942 wave_k_coop     512x16x32x32 T_K=8
        10302,  # gfx942 wave_k_coop     512x32x16x64 T_K=8
        10303,  # gfx942 wave_k_coop     256x32x32x64 T_K=4
        10305,  # gfx942 wave_k_coop     512x16x32x64 T_K=8
    }
)

# gfx1250 has no shape-heuristic dispatch yet (tune-id entry only). This set
# is used purely to keep the kid in the subset-compile set S so the tune-id
# path can always reach it.
HEURISTIC_DEFAULT_KIDS_GFX1250 = frozenset(gfx1250_kernels_list.keys())

HEURISTIC_DEFAULT_KIDS = (
    HEURISTIC_DEFAULT_KIDS_GFX950
    | HEURISTIC_DEFAULT_KIDS_GFX942
    | HEURISTIC_DEFAULT_KIDS_GFX1250
)

HEURISTIC_DEFAULT_KIDS_BY_ARCH = {
    "gfx950": HEURISTIC_DEFAULT_KIDS_GFX950,
    "gfx942": HEURISTIC_DEFAULT_KIDS_GFX942,
    "gfx1250": HEURISTIC_DEFAULT_KIDS_GFX1250,
}


def heuristic_kids_for_arch(arches):
    """Return the heuristic-default kid subset whose arch_prefix matches.

    ``arches`` is an iterable of lowercase arch strings (e.g. ``{"gfx942"}``)
    or ``None`` (caller does not know / multi-arch build) -- in the ``None``
    case the full union is returned so the legacy multi-arch behaviour is
    preserved.
    """
    if arches is None:
        return HEURISTIC_DEFAULT_KIDS
    arches = {a.lower() for a in arches}
    out = frozenset()
    for arch in arches:
        out = out | HEURISTIC_DEFAULT_KIDS_BY_ARCH.get(arch, frozenset())
    return out


def _opus_sidecar_path():
    """Return the on-disk path of the subset-compile sidecar.

    Lives in ``{bd_dir}/`` (one level above the per-module build dir) so
    it survives ``aiter.jit.core.clear_build("module_deepgemm_opus")`` --
    which ``build_module()`` calls when ``AITER_REBUILD == 1`` -- and is
    therefore the canonical "what kids should be in the next .so" source
    that ``gen_instances.py`` consumes. The tuner expands this sidecar
    BEFORE triggering the rebuild; if it lived inside the build dir,
    clear_build would wipe it out before gen_instances could read it.
    """
    # Import lazily to avoid circular import at module load (aiter imports
    # opus_gemm_common, opus_gemm_common imports aiter.jit.core).
    from aiter.jit.core import bd_dir

    return os.path.join(bd_dir, "compiled_kids_opus.json")
