# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus MoE stage2 codegen metadata bridge.

A8W4 stage2 metadata is defined in the Python package because both runtime
fused_moe glue and csrc codegen need the same kid/name/layout table. This file
keeps csrc-side generators close to the opus_gemm_common.py pattern while
re-exporting A8W4 data from the package source of truth.
"""

from __future__ import annotations

import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path


def _load_a8w4_meta_module():
    here = Path(__file__).resolve()
    rel_meta_path = Path("aiter") / "ops" / "opus" / "moe_stage2_a8w4_meta.py"
    candidates = [
        here.parents[2] / rel_meta_path,
    ]
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / rel_meta_path)
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "_opus_moe_stage2_a8w4_meta", path
            )
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError("unable to locate aiter/ops/opus/moe_stage2_a8w4_meta.py")


_a8w4_meta = _load_a8w4_meta_module()
OPUS_A8W4_STAGE2_BY_KID = _a8w4_meta.OPUS_A8W4_STAGE2_BY_KID
OPUS_A8W4_TUNER_INSTANCES = _a8w4_meta.OPUS_A8W4_TUNER_INSTANCES
OPUS_A8W4_DEFAULT_SHAPE_FAMILY_CONTRACT = (
    _a8w4_meta.OPUS_A8W4_DEFAULT_SHAPE_FAMILY_CONTRACT
)
OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT = (
    _a8w4_meta.OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT
)
OPUS_A8W4_SHAPE_FAMILY_CONTRACTS = _a8w4_meta.OPUS_A8W4_SHAPE_FAMILY_CONTRACTS
OPUS_A8W4_ROUTE_REDUCE_INSTANCES = _a8w4_meta.OPUS_A8W4_ROUTE_REDUCE_INSTANCES
OPUS_A8W4_OUT_MODE_ATOMIC = _a8w4_meta.OPUS_A8W4_OUT_MODE_ATOMIC
OpusA8W4Stage2Instance = _a8w4_meta.OpusA8W4Stage2Instance
opus_a8w4_decode_kid = _a8w4_meta.opus_a8w4_decode_kid
opus_a8w4_shape_family_for_shape = _a8w4_meta.opus_a8w4_shape_family_for_shape


@dataclass(frozen=True)
class OpusMoeStage2Instance:
    kid: int
    name: str
    trait: str
    block_m: int
    block_n: int
    block_k: int
    dtype: str = "bf16"
    a2_layout: str = "token_major"
    output_mode: str = "token_slot_route_output_reduce"
    launcher: str = "opus_moe_stage2_gemmstyle_launch_gfx950"


STAGE2_BF16_KERNELS = {
    1: OpusMoeStage2Instance(
        1,
        "bf16_gemmstyle256x256x64_token_slot_route_out_no_oob_nfast",
        "OpusMoeStage2Bf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast",
        block_m=256,
        block_n=256,
        block_k=64,
        output_mode="token_slot_route_output_reduce",
    ),
}

STAGE2_A8W4_KERNELS: dict[int, OpusA8W4Stage2Instance] = dict(OPUS_A8W4_STAGE2_BY_KID)
STAGE2_A8W4_TUNER_KERNELS: dict[int, OpusA8W4Stage2Instance] = {
    inst.kid: inst for inst in OPUS_A8W4_TUNER_INSTANCES
}
