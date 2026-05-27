# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import List

# Legacy cache policy = traits default for split-barrier & persistent
# a16w16 (see opus_gemm_traits_a16w16_gfx950.cuh).
# Instances carrying this exact (cachectl_a, cachectl_b) tuple are
# treated as "the baseline" and emit the bare .so symbol with NO
# `_cA0cB17` suffix; see OpusGemmInstance.name.
_LEGACY_CACHECTL = (0, 17)


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
    # Compile-time OOB (out-of-bounds) tail handling. True = full boundary
    # checks (mask_va_tail, store_if pred, reduce N-tail). False = no tail
    # checks, only valid when shape is tile-aligned (M%B_M==N%B_N==K%B_K==0).
    has_oob: bool = True
    # Cache policy for A/B loads (CDNA4 ISA Table 49). -1 = use traits default.
    # 0=LRU, 1=SC0(LLC Evict), 17=SC0+SC1(L2 Bypass).
    cachectl_a: int = -1
    cachectl_b: int = -1

    @property
    def name(self) -> str:
        parts = [
            "opus_gemm",
            "x".join(map(str, [self.BLOCK_SIZE, self.B_M, self.B_N, self.B_K])),
            "x".join(map(str, [self.T_M, self.T_N])),
            "x".join(map(str, [self.W_M, self.W_N, self.W_K])),
            "x".join(map(str, [self.GROUP_M, self.GROUP_N, self.GROUP_K])),
        ]
        if self.kernel_tag == "a16w16_flatmm":
            parts.insert(1, "flatmm")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a16w16_flatmm_splitk":
            parts.insert(1, "flatmm_splitk")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a16w16_persistent":
            parts.insert(1, "persistent")
        elif self.kernel_tag == "a16w16_mono_tile":
            parts.insert(1, "mono_tile")
        if not self.has_oob:
            parts.append("nooob")
        # Legacy cache policy = traits default for split-barrier &
        # persistent a16w16: CACHECTL_A=0 (LRU), CACHECTL_B=17
        # (BYPASS_L2). When a kid carries this exact policy, suppress
        # the `_cA0cB17` suffix so the emitted .so symbol stays
        # bit-identical to the pre-cpol baseline. This keeps:
        #   * opus_gemm_heuristic_dispatch_gfx950.cuh hardcoded symbol
        #     names (line 118-119) resolvable;
        #   * aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv
        #     `kernelName` column unchanged for legacy 4..9 winners
        #     (80 rows in production).
        # C++ side, `Traits<..., 0, 17>` and `Traits<...>` (with traits
        # defaults `CACHECTL_A_=0, CACHECTL_B_=17`) are ODR-equivalent
        # template specializations and produce one and only one
        # instantiation, so the legacy and explicit forms are
        # bit-identical in the compiled .so.
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
    This is the "legacy" policy used by KID 4..9 and 1004..1009 — the
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
    # Flatmm locked config (per gcnasm/opus_fmm/INTEGRATION.md):
    # BLOCK_SIZE=256, T_M=2, T_N=1, MFMA=(16,16,32), VEC=(8,8,4), HAS_BIAS=false.
    # Emit both bf16 and fp32 output variants so the tune lookup map can
    # instantiate <fp32_t> when Y.dtype is torch.float32 (mirrors a16w16).
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
    # ── MFMA 16x16x32, T_N=2, BS=256 (2-block/CU capable) ──
    # 3:  _a16w16(256, 128, 128, 32,  2, 16, 16, 32),  # disabled: intermittent accuracy (suspected compiler issue with VGPR=104/AGPR=64)
    4:  _a16w16(256, 128, 256, 32,  2, 16, 16, 32),
    5:  _a16w16(256, 256, 128, 32,  2, 16, 16, 32),
    # ── MFMA 16x16x32, T_N=4, BS=512 (1-block/CU) ──
    6:  _a16w16(512, 128, 128, 64,  4, 16, 16, 32),
    7:  _a16w16(512, 256, 128, 64,  4, 16, 16, 32),
    8:  _a16w16(512, 128, 256, 64,  4, 16, 16, 32),
    9:  _a16w16(512, 256, 256, 64,  4, 16, 16, 32),  # existing / current default
}

# Removed (kids 100-115, a16w16_flatmm non-splitk):
#
# Rationale: the non-splitk a16w16_flatmm pipeline has two latent
# correctness bugs in its N%16 vector store and K%B_K tail handling
# (see opus_gemm_tune._kid_rejects_shape rules (b)), so the tunner
# already rejects these kids for the vast majority of shapes. For
# the remaining shapes (N%16==0 AND K%B_K==0), the splitk pipeline
# with splitK=0 (->KBatch=1) produces bit-identical results via the
# same underlying MMA, at a small cost (one extra reduce kernel
# launch + one fp32 workspace write pass) that is dwarfed by the
# ~70% reduction in JIT compile units (from 57 down to ~26).
#
# Kept as an empty dict so the three merges in opus_gemm_common.py
# (kernels_list below) and opus_gemm_tune.py / gen_instances.py
# stay valid. The `a16w16_flatmm` kernel_tag remains in the schema
# and validators in case a future kid needs it, but no instances
# are emitted by default.
a16w16_flatmm_kernels_list = {}

# 11 splitk tiles mirroring gcnasm/opus_fmm/flatmm_a16w16_4wave_wasp_splitk.cc
# -t 0..10 dispatch exactly:
#   * 8 WG_PER_CU=2 tiles (kids 200..207) - occupancy 2, 80 KB LDS/wg budget
#   * 3 WG_PER_CU=1 tiles (kids 208..210) - hand-picked large/extreme-aspect
#     tiles that fit only in 160 KB/wg LDS. Larger WG=1 combos (128x128x64,
#     128x64x128, 64x128x128, 256x64x64, 64x256x64) spill 100+ VGPRs to
#     scratch and run 1000x slower (cc lines 1143-1150); the validator in
#     gen_instances.py enforces COM_REP_M*COM_REP_N<=16 for WG=1.
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
    # Tile coverage extension (kids 211..223): B_M=96 OR B_N=96 lanes for shapes
    # whose M or N is a multiple of 96. Validated against the codegen
    # _validate_a16w16_flatmm_splitk constraints (LDS pfk>=3, cc-spill guard
    # COM_REP_M*COM_REP_N<=16 for WG=1) plus a refined VGPR estimate (consumer
    # wave: 2*v_a + 2*v_b + ~64 overhead; cap=512/WG_PER_CU). All 13 fit.
    # NOT YET PROFILED: COM_REP_M=3 / COM_REP_N=6 odd-loop unrolls are a new
    # path; precision must be validated via op_tests/test_opus_a16w16_gemm.py
    # (drive the kid through gemm_a16w16_opus by adding the relevant shape
    # to the tuned CSV) before relying on these in production heuristics.
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
# Only valid when shape is tile-aligned (M%B_M==N%B_N==K%B_K==0).
# Explicitly inherits cachectl from the parent so the legacy (0, 17)
# policy is propagated; the _LEGACY_CACHECTL special-case in name
# keeps the .so symbol bare (`..._nooob`, no `_cA0cB17`).
a16w16_kernels_list_nooob = {
    kid + 1000: _a16w16(
        inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
        inst.T_N, inst.W_M, inst.W_N, inst.W_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_kernels_list.items()
}

# CPOL variants for a16w16: 3 policies per kid, tuner picks best per shape.
#   M-heavy: A=SC0(1, LLC Evict), B=BYPASS_L2(17) — large A streams, small B cached
#   N-heavy: A=BYPASS_L2(17), B=SC0(1, LLC Evict) — swapped
#   Balanced: A=LRU(0), B=LRU(0) — both cached normally
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

# ── a16w16 persistent (M-outer + N-fast XCD swizzle) ──────────────────────
#
# Pipeline:
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh
# Traits:
#   csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh
#   :: opus_gemm_a16w16_persistent_traits_gfx950
#
# Compact KID layout:
#   kid = 300 + cpol_group * 6 + tile_idx
#     cpol_group: 0=legacy (0,17), 1=Mheavy (1,17), 2=Nheavy (17,1), 3=balanced (0,0)
#     tile_idx:   0..5 (see _PERSISTENT_TILES below)
#   nooob mirror at +1000: kid range [1300, 1324).
#
# Total: 6 tile × 4 cpol × {has_oob, nooob} = 48 kid.
#
# Persistent kernel locks BLOCK_SIZE=512, T_M=2, T_N=4, MFMA 16x16x32
# (matches the standalone reference gemm_a16w16_8wave_mouter.cc).


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


# 4-tile sweep, all B_K=64. The ra/rb LDS read layout in the shared
# split-barrier traits enforces B_K == T_N * W_K / 2 = 4 * 32 / 2 = 64
# (see _validate_a16w16 in csrc/opus_gemm/gen_instances.py), so other
# B_K values are not currently representable on the T_M=2, T_N=4,
# W_K=32 axis we lock for persistent. Shallow/deep-K variants would
# require a different T_N or W_K axis -- left as future work.
_PERSISTENT_TILES = [
    # (B_M, B_N, B_K)
    (256, 256, 64),  # tile 0: mouter default; 32K×2K×7K best 1208 TFLOPS
    (128, 256, 64),  # tile 1: narrow M
    (256, 128, 64),  # tile 2: narrow N
    (128, 128, 64),  # tile 3: small
]

# Legacy (300..303): cachectl == (0, 17). Same as traits default, so the
# _LEGACY_CACHECTL special-case in OpusGemmInstance.name suppresses the
# `_cA0cB17` suffix and the emitted .so symbol is just
# `opus_gemm_persistent_512x..._0x0x0`.
a16w16_persistent_kernels_list = {
    300 + i: _a16w16_persistent(bm, bn, bk)
    for i, (bm, bn, bk) in enumerate(_PERSISTENT_TILES)
}

# Cpol variants (304..315): 3 groups × 4 tiles, mirroring _CACHECTL_CONFIGS
# but with a single compact base offset per cpol group. Each emitted .so
# carries its `_cA*cB*` suffix.
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

# ── a16w16 mono-tile (single-MMA-per-K-iter, 8 waves) ─────────────────────
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
# B_M ≤ 192 hard cap. The 7 tiles below were picked to cover
# (M-bucket × N-bucket) combinations not already served well by the
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
}

default_kernels_dict = {
    (-1): OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
    (-2): OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0,     "a8w8",       ["fp32_t"]),
    (-3): _a16w16(512, 256, 256, 64, 4, 16, 16, 32),  # same as a16w16 #9
}
# fmt: on


# =============================================================================
# Subset-compile kid taxonomy (consumed by gen_instances.py for the
# `HEURISTIC_DEFAULT_KIDS ⊆ S` assert + the per-pipeline classifier sets).
#
# These are pure data constants -- no tuner / runtime logic lives here. The
# tune-time helpers (candidate_kids_for_shape, candidate_splitK,
# kid_rejects_shape, kid_rejects_bias, _ensure_kids_compiled, ...) live in
# csrc/opus_gemm/opus_gemm_tune.py and are imported by gradlib's GemmTuner
# and the debug-only opus_gemm_tune.py main entry from there.
# =============================================================================

# Splitk kids: a16w16_flatmm_splitk pipeline (kid 200..223 + nooob mirror).
# These are bias-aware. They are the only kids that consume a literal `splitK`
# KBatch argument.
SPLITK_KIDS = frozenset(a16w16_flatmm_splitk_kernels_list.keys()) | frozenset(
    a16w16_flatmm_splitk_kernels_list_nooob.keys()
)

# Non-splitk a16w16-family kids: split-barrier 4..9 + cpol/nooob mirrors,
# persistent 300..315 + cpol/nooob mirrors.
# Note: persistent currently does NOT support bias.
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
)

# Bias-aware kids: split-barrier (4..9 + cpol/nooob mirrors) and the entire
# splitk family. Persistent is excluded because its launcher currently
# rejects any non-empty bias up front.
BIAS_AWARE_KIDS = (
    frozenset(a16w16_kernels_list.keys())
    | frozenset(a16w16_kernels_list_nooob.keys())
    | frozenset(a16w16_kernels_list_cpol.keys())
    | frozenset(a16w16_kernels_list_cpol_nooob.keys())
    | SPLITK_KIDS
)

# Heuristic-dispatch fallback kids (gfx950). MUST match the integer returns
# of opus_a16w16_heuristic_kid_gfx950() in
# csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh.
# These kids MUST always be in the subset-compile set S, otherwise heuristic
# fallback for an unbaked (M,N,K) shape will fail at runtime with
# `AITER_CHECK: Kernel id X not found in a16w16 tune lookup table`.
#
# gen_instances.py asserts HEURISTIC_DEFAULT_KIDS.issubset(S) before writing
# the sidecar, so any drift here vs the C++ side surfaces at codegen time
# rather than at runtime.
HEURISTIC_DEFAULT_KIDS = frozenset(
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
