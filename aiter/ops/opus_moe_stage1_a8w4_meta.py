# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Structured metadata for Opus MoE A8W4 stage1 kernels.

This module is torch-free so runtime wrappers, tuner code, and csrc codegen can
share the same stage1 kid table without drifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW2_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_MIN2_PAIR_GATEUP_A_REUSE_MFMA = 1006
OPUS_A8W4_STAGE1_KID_P0_BM128_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP_MIN2_NOSCALEGUARD_FULL_NEXT_A = 1009
OPUS_A8W4_STAGE1_KID_P0_BM128_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP_MIN3_NOSCALEGUARD_FULL_NEXT_A = 1011
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_MIN1_PAIR_GATEUP_A_REUSE_MFMA = 1013
OPUS_A8W4_STAGE1_KID_P0_BM32_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP = 1014
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_MIN2_PAIR_GATEUP_A_REUSE_MFMA = 1015
OPUS_A8W4_STAGE1_KID_P0_BM128_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP_MIN4_NOSCALEGUARD_FULL_NEXT_A = 1017
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_MIN3_PAIR_GATEUP_A_REUSE_MFMA = 1021
OPUS_A8W4_STAGE1_KID_P0_BM64_BN384_GATE_UP_GROUP_SPLIT_T4096_NOCLAMP_MIN1_ASYNC_A_CAP_ROUTES_ASSUME_ROUTE_SPLIT_B = 1022
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_MIN1_PAIR_GATEUP_A_REUSE_MFMA = 1026
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_MIN2_PAIR_GATEUP_A_REUSE_MFMA = 1027
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_MIN3_PAIR_GATEUP_A_REUSE_MFMA = 1028
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_MIN4_PAIR_GATEUP_A_REUSE_MFMA = 1029
OPUS_A8W4_STAGE1_KID_P0_BM128_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP_MIN3_NOSCALEGUARD_SPLIT_B = 1031
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_A_REUSE_MFMA = 1074
OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G6_KW1_NOCLAMP_GROUP_SPLIT_MIN1_A_REUSE_MFMA = 1099


@dataclass(frozen=True)
class OpusA8W4Stage1Instance:
    kid: int
    name: str
    trait: str
    block_m: int
    block_n: int
    block_k: int
    sort_block_m: int
    tuner_candidate: bool = True
    tuner_tokens: tuple[int, ...] = ()
    tuner_min_token: Optional[int] = None
    tuner_max_token: Optional[int] = None

    def supports_token(self, token: Optional[int]) -> bool:
        # Keep formal tuning open across token counts; token lists are historical
        # hints, not hard candidate filters.
        return self.tuner_candidate

    def tuner_params(self) -> dict[str, object]:
        return {
            "kid": self.kid,
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "sort_block_m": self.sort_block_m,
            "trait": self.trait,
        }


OPUS_A8W4_STAGE1_INSTANCES = (
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM32_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP,
        name="opus_moe1_a8w4_bm32_bn384_gateup_groupsplit_noclamp",
        trait="OpusMoeStage1A8W4P0Bm32Bn384GateUpGroupSplitNoClamp",
        block_m=32,
        block_n=384,
        block_k=256,
        sort_block_m=32,
        tuner_tokens=(512, 1024, 2048),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM64_BN384_GATE_UP_GROUP_SPLIT_T4096_NOCLAMP_MIN1_ASYNC_A_CAP_ROUTES_ASSUME_ROUTE_SPLIT_B,
        name="opus_moe1_a8w4_bm64_bn384_gateup_groupsplit_t4096_noclamp_min1_asynca_caproutes_assumeroute_splitb",
        trait="OpusMoeStage1A8W4P0Bm64Bn384GateUpGroupSplitT4096NoClampMin1AsyncACapRoutesAssumeRouteSplitB",
        block_m=64,
        block_n=384,
        block_k=256,
        sort_block_m=64,
        tuner_tokens=(4096, 32768),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G6_KW1_NOCLAMP_GROUP_SPLIT_MIN1_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g6_kw1_noclamp_groupsplit_min1_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G6KWave1NoClampGroupSplitMin1AReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(256, 1024),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_splitselectorb_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampSplitSelectorBAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(2, 64, 128),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW2_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_MIN2_PAIR_GATEUP_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw2_caproutes_noclamp_splitselectorb_min2_pairgateup_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave2CapRoutesNoClampSplitSelectorBMin2PairGateUpAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(8,),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_MIN1_PAIR_GATEUP_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_splitselectorb_min1_pairgateup_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampSplitSelectorBMin1PairGateUpAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(1, 4, 16, 32, 64, 128, 256, 512),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_MIN2_PAIR_GATEUP_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_splitselectorb_min2_pairgateup_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampSplitSelectorBMin2PairGateUpAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(4, 16, 32, 64, 128, 256, 512),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_SPLIT_SELECTOR_B_MIN3_PAIR_GATEUP_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_splitselectorb_min3_pairgateup_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampSplitSelectorBMin3PairGateUpAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(256,),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_MIN1_PAIR_GATEUP_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_min1_pairgateup_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampMin1PairGateUpAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(256,),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_MIN2_PAIR_GATEUP_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_min2_pairgateup_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampMin2PairGateUpAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(256,),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_MIN3_PAIR_GATEUP_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_min3_pairgateup_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampMin3PairGateUpAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(256,),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_CAP_ROUTES_NOCLAMP_MIN4_PAIR_GATEUP_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_min4_pairgateup_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampMin4PairGateUpAReuse",
        block_m=16,
        block_n=384,
        block_k=256,
        sort_block_m=16,
        tuner_tokens=(256,),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM128_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP_MIN3_NOSCALEGUARD_SPLIT_B,
        name="opus_moe1_a8w4_bm128_bn384_gateup_groupsplit_noclamp_min3_noscaleguard_splitb",
        trait="OpusMoeStage1A8W4P0Bm128Bn384GateUpGroupSplitNoClampMin3NoScaleGuardSplitB",
        block_m=128,
        block_n=384,
        block_k=256,
        sort_block_m=128,
        tuner_tokens=(4096,),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM128_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP_MIN2_NOSCALEGUARD_FULL_NEXT_A,
        name="opus_moe1_a8w4_bm128_bn384_gateup_groupsplit_noclamp_min2_noscaleguard_fullnexta",
        trait="OpusMoeStage1A8W4P0Bm128Bn384GateUpGroupSplitNoClampMin2NoScaleGuardFullNextA",
        block_m=128,
        block_n=384,
        block_k=256,
        sort_block_m=128,
        tuner_tokens=(8192, 16384),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM128_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP_MIN3_NOSCALEGUARD_FULL_NEXT_A,
        name="opus_moe1_a8w4_bm128_bn384_gateup_groupsplit_noclamp_min3_noscaleguard_fullnexta",
        trait="OpusMoeStage1A8W4P0Bm128Bn384GateUpGroupSplitNoClampMin3NoScaleGuardFullNextA",
        block_m=128,
        block_n=384,
        block_k=256,
        sort_block_m=128,
        tuner_tokens=(8192, 16384),
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM128_BN384_GATE_UP_GROUP_SPLIT_NOCLAMP_MIN4_NOSCALEGUARD_FULL_NEXT_A,
        name="opus_moe1_a8w4_bm128_bn384_gateup_groupsplit_noclamp_min4_noscaleguard_fullnexta",
        trait="OpusMoeStage1A8W4P0Bm128Bn384GateUpGroupSplitNoClampMin4NoScaleGuardFullNextA",
        block_m=128,
        block_n=384,
        block_k=256,
        sort_block_m=128,
        tuner_tokens=(8192, 16384),
    ),
)

OPUS_A8W4_STAGE1_BY_KID = {
    inst.kid: inst for inst in OPUS_A8W4_STAGE1_INSTANCES
}
OPUS_A8W4_STAGE1_BY_NAME = {
    inst.name: inst for inst in OPUS_A8W4_STAGE1_INSTANCES
}
OPUS_A8W4_STAGE1_TUNER_INSTANCES = tuple(
    inst for inst in OPUS_A8W4_STAGE1_INSTANCES if inst.tuner_candidate
)


def opus_a8w4_stage1_instance(kid: int) -> Optional[OpusA8W4Stage1Instance]:
    return OPUS_A8W4_STAGE1_BY_KID.get(int(kid))


def opus_a8w4_stage1_kid_from_name(name) -> Optional[int]:
    inst = OPUS_A8W4_STAGE1_BY_NAME.get(str(name))
    return None if inst is None else inst.kid


def opus_a8w4_stage1_kid_name(kid: int) -> str:
    inst = opus_a8w4_stage1_instance(kid)
    return "unknown" if inst is None else inst.name


def opus_a8w4_stage1_kid_block_m(kid: int) -> int:
    inst = opus_a8w4_stage1_instance(kid)
    return -1 if inst is None else inst.block_m


def opus_a8w4_stage1_kid_sort_block_m(kid: int) -> int:
    inst = opus_a8w4_stage1_instance(kid)
    return -1 if inst is None else inst.sort_block_m


def get_opus_a8w4_stage1_kernels(token: Optional[int] = None) -> dict[str, dict[str, object]]:
    return {
        inst.name: inst.tuner_params()
        for inst in OPUS_A8W4_STAGE1_INSTANCES
        if inst.supports_token(token)
    }
