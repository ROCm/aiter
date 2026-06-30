# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Structured metadata for experimental Opus MoE A8W4 stage1 kernels.

This module is torch-free so runtime wrappers, tuner code, and csrc codegen can
share the same stage1 kid table without drifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


OPUS_A8W4_STAGE1_KID_P0_BM32_BN384_A_REUSE_MFMA = 1010
OPUS_A8W4_STAGE1_KID_P0_BM64_BN384_ROW_SPLIT = 1020
OPUS_A8W4_STAGE1_KID_P0_BM64_BN384_GATE_UP_GROUP_SPLIT = 1030
OPUS_A8W4_STAGE1_KID_P0_BM128_BN256_GATE_UP_GROUP_SPLIT = 1040


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
        if token is None:
            return self.tuner_candidate
        if not self.tuner_candidate:
            return False
        token = int(token)
        if self.tuner_tokens and token in self.tuner_tokens:
            return True
        if (
            self.tuner_tokens
            and self.tuner_min_token is None
            and self.tuner_max_token is None
        ):
            return False
        if self.tuner_min_token is not None and token < self.tuner_min_token:
            return False
        if self.tuner_max_token is not None and token > self.tuner_max_token:
            return False
        return True

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
        kid=OPUS_A8W4_STAGE1_KID_P0_BM32_BN384_A_REUSE_MFMA,
        name="opus_moe1_a8w4_bm32_bn384_a_reuse_mfma",
        trait="OpusMoeStage1A8W4P0Bm32Bn384AReuse",
        block_m=32,
        block_n=384,
        block_k=256,
        sort_block_m=32,
        tuner_min_token=1,
        tuner_max_token=1024,
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM64_BN384_ROW_SPLIT,
        name="opus_moe1_a8w4_bm64_bn384_row_split",
        trait="OpusMoeStage1A8W4P0Bm64Bn384RowSplit",
        block_m=64,
        block_n=384,
        block_k=256,
        sort_block_m=64,
        tuner_candidate=False,
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM64_BN384_GATE_UP_GROUP_SPLIT,
        name="opus_moe1_a8w4_bm64_bn384_gateup_groupsplit",
        trait="OpusMoeStage1A8W4P0Bm64Bn384GateUpGroupSplit",
        block_m=64,
        block_n=384,
        block_k=256,
        sort_block_m=64,
        tuner_min_token=1,
    ),
    OpusA8W4Stage1Instance(
        kid=OPUS_A8W4_STAGE1_KID_P0_BM128_BN256_GATE_UP_GROUP_SPLIT,
        name="opus_moe1_a8w4_bm128_bn256_gateup_groupsplit",
        trait="OpusMoeStage1A8W4P0Bm128Bn256GateUpGroupSplit",
        block_m=128,
        block_n=256,
        block_k=256,
        sort_block_m=128,
        tuner_tokens=(32, 128),
        tuner_min_token=2048,
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
