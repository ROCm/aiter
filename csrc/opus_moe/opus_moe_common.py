# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus MoE stage2 codegen metadata bridge.

A8W4 stage2 metadata is defined in the Python package because both runtime
fused_moe glue and csrc codegen need the same kid/name/layout table. This file
keeps csrc-side generators close to the opus_gemm_common.py pattern while
re-exporting A8W4 data from the package source of truth.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path


def _load_ops_meta_module(rel_meta_path: Path, module_name: str):
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / rel_meta_path,
    ]
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / rel_meta_path)
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                module_name, path
            )
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(f"unable to locate {rel_meta_path.as_posix()}")


_a8w4_meta = _load_ops_meta_module(
    Path("aiter") / "ops" / "opus" / "moe_stage2_a8w4_meta.py",
    "_opus_moe_stage2_a8w4_meta",
)
_a8w4_stage1_meta = _load_ops_meta_module(
    Path("aiter") / "ops" / "opus_moe_stage1_a8w4_meta.py",
    "_opus_moe_stage1_a8w4_meta",
)
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

DEFAULT_STAGE2_KIDS = tuple(STAGE2_BF16_KERNELS)

STAGE2_KERNELS_BY_DTYPE = {
    "bf16": STAGE2_BF16_KERNELS,
}

STAGE2_A8W4_KERNELS: dict[int, OpusA8W4Stage2Instance] = dict(OPUS_A8W4_STAGE2_BY_KID)
STAGE2_A8W4_TUNER_KERNELS: dict[int, OpusA8W4Stage2Instance] = {
    inst.kid: inst for inst in OPUS_A8W4_TUNER_INSTANCES
}
OPUS_A8W4_STAGE1_BY_KID = _a8w4_stage1_meta.OPUS_A8W4_STAGE1_BY_KID
OPUS_A8W4_STAGE1_TUNER_INSTANCES = _a8w4_stage1_meta.OPUS_A8W4_STAGE1_TUNER_INSTANCES
OpusA8W4Stage1Instance = _a8w4_stage1_meta.OpusA8W4Stage1Instance

STAGE1_A8W4_KERNELS: dict[int, OpusA8W4Stage1Instance] = dict(
    OPUS_A8W4_STAGE1_BY_KID
)
STAGE1_A8W4_TUNER_KERNELS: dict[int, OpusA8W4Stage1Instance] = {
    inst.kid: inst for inst in OPUS_A8W4_STAGE1_TUNER_INSTANCES
}

STAGE2_TUNE_KEY_COLUMNS = [
    "arch",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "dtype",
    "a2_layout",
    "output_mode",
    "block_m",
]

STAGE2_TUNE_RESULT_COLUMNS = [
    "kid",
    "kernel_name",
    "block_n",
    "block_k",
    "us",
    "max_abs",
    "mean_abs",
    "valid",
]

STAGE2_TUNE_COLUMNS = STAGE2_TUNE_KEY_COLUMNS + STAGE2_TUNE_RESULT_COLUMNS


def default_stage2_tuned_csv() -> str:
    env_path = os.environ.get("OPUS_MOE_STAGE2_TUNED_CSV")
    if env_path:
        return env_path

    for data_dir in (
        Path("/shared/amdgpu/home/hyi_qle/yifehuan_temp/data"),
        Path("/app/yifehuan_temp/data"),
    ):
        if data_dir.exists():
            return str(data_dir / "opus_moe_stage2_tuned.csv")

    return "/tmp/opus_moe_stage2_tuned.csv"


def candidate_stage2_kids_for_shape(
    *,
    model_dim: int,
    inter_dim: int,
    block_m: int,
    dtype: str = "bf16",
    requested_kids: list[int] | tuple[int, ...] | None = None,
) -> list[OpusMoeStage2Instance]:
    kernel_table = STAGE2_KERNELS_BY_DTYPE.get(dtype, {})
    default_kids = tuple(kernel_table)
    kids = list(requested_kids or default_kids)
    instances: list[OpusMoeStage2Instance] = []

    for kid in kids:
        inst = kernel_table.get(int(kid))
        if inst is None:
            continue
        if block_m % inst.block_m != 0:
            continue
        if model_dim % inst.block_n != 0 or inter_dim % inst.block_k != 0:
            continue
        instances.append(inst)

    return instances
