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
        return "_".join(parts)


def _a16w16(bs, bm, bn, bk, tn, wm, wn, wk):
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk,
        vec, vec, 4, 0, 0, 0, "a16w16", ["fp32_t", "bf16_t"],
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

# 8 tiles validated in gcnasm/opus_fmm/INTEGRATION.md x WG_PER_CU in {1, 2}.
# Kids start at 100 to reserve room above the split-barrier a16w16 range.
a16w16_flatmm_kernels_list = {
    100: _a16w16_flatmm( 32,  32,  64, 2),
    101: _a16w16_flatmm( 32,  32,  64, 1),
    102: _a16w16_flatmm( 32,  32, 128, 2),
    103: _a16w16_flatmm( 32,  32, 128, 1),
    104: _a16w16_flatmm( 32,  64,  64, 2),
    105: _a16w16_flatmm( 32,  64,  64, 1),
    106: _a16w16_flatmm( 32, 128,  64, 2),
    107: _a16w16_flatmm( 32, 128,  64, 1),
    108: _a16w16_flatmm( 64,  32,  64, 2),
    109: _a16w16_flatmm( 64,  32,  64, 1),
    110: _a16w16_flatmm( 64,  32, 128, 2),  # INTEGRATION.md recommended for M >= 64
    111: _a16w16_flatmm( 64,  32, 128, 1),
    112: _a16w16_flatmm( 64,  64,  64, 2),  # INTEGRATION.md best for M >= 128
    113: _a16w16_flatmm( 64,  64,  64, 1),
    114: _a16w16_flatmm(128,  32,  64, 2),
    115: _a16w16_flatmm(128,  32,  64, 1),
}

# combined list (used by production gen_instances / dispatch)
kernels_list = {
    **a8w8_scale_kernels_list,
    **a8w8_kernels_list,
    **a16w16_kernels_list,
    **a16w16_flatmm_kernels_list,
}

default_kernels_dict = {
    (-1): OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
    (-2): OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0,     "a8w8",       ["fp32_t"]),
    (-3): _a16w16(512, 256, 256, 64, 4, 16, 16, 32),  # same as a16w16 #9
}
# fmt: on
