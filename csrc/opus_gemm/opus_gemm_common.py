# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass, field
from typing import List


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

    @property
    def name(self) -> str:
        parts = [
            "opus_gemm",
            "x".join(
                map(str, [self.BLOCK_SIZE, self.B_M, self.B_N, self.B_K])
            ),
            "x".join(map(str, [self.T_M, self.T_N])),
            "x".join(map(str, [self.W_M, self.W_N, self.W_K])),
            "x".join(map(str, [self.GROUP_M, self.GROUP_N, self.GROUP_K])),
        ]
        if self.kernel_tag == "a16w16_flatmm":
            # Disambiguate by pipeline (flatmm vs split-barrier) and occupancy.
            parts.insert(1, "flatmm")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a16w16_flatmm_splitk":
            # Distinguish from non-splitk flatmm by inserting both tags and
            # appending the WG_PER_CU suffix (same scheme as flatmm).
            parts.insert(1, "flatmm_splitk")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        return "_".join(parts)


def _a16w16(bs, bm, bn, bk, tn, wm, wn, wk):
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk,
        vec, vec, 4, 0, 0, 0, "a16w16", ["fp32_t", "bf16_t"],
    )


def _a16w16_flatmm_splitk(bm, bn, bk, wg_per_cu):
    # Flatmm split-K locked config (per cc -t 0..10 dispatch):
    # BLOCK_SIZE=256, T_M=2, T_N=1, MFMA=(16,16,32), VEC=(8,8,4), HAS_BIAS=false.
    # output_dtypes=["fp32_t"]: main kernel writes fp32 workspace; Y is bf16
    # (reduce kernel does the fp32->bf16 cast). Only the fp32_t template
    # instantiation is generated; opus_gemm.cu forces the <fp32_t> dispatch
    # branch for splitk kids regardless of Y.dtype (which must be bf16).
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        256, bm, bn, bk,
        2, 1,            # T_M, T_N
        16, 16, 32,      # MFMA 16x16x32
        vec, vec, 4,     # VEC
        0, 0, 0,         # GROUP (unused)
        "a16w16_flatmm_splitk",
        ["fp32_t"],
        wg_per_cu,
    )


def _a16w16_flatmm(bm, bn, bk, wg_per_cu):
    # Flatmm locked config (per gcnasm/opus_fmm/INTEGRATION.md):
    # BLOCK_SIZE=256, T_M=2, T_N=1, MFMA=(16,16,32), VEC=(8,8,4), HAS_BIAS=false.
    # Emit both bf16 and fp32 output variants so the tune lookup map can
    # instantiate <fp32_t> when Y.dtype is torch.float32 (mirrors a16w16).
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        256, bm, bn, bk,
        2, 1,            # T_M, T_N (T_N hardcoded to 1 for the warp-spec pipeline)
        16, 16, 32,      # MFMA 16x16x32
        vec, vec, 4,     # VEC
        0, 0, 0,         # GROUP (unused)
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
}

# combined list (used by production gen_instances / dispatch)
kernels_list = {
    **a8w8_scale_kernels_list,
    **a8w8_kernels_list,
    **a16w16_kernels_list,
    **a16w16_flatmm_kernels_list,
    **a16w16_flatmm_splitk_kernels_list,
}

default_kernels_dict = {
    (-1): OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
    (-2): OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0,     "a8w8",       ["fp32_t"]),
    (-3): _a16w16(512, 256, 256, 64, 4, 16, 16, 32),  # same as a16w16 #9
}
# fmt: on
