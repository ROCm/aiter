# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus MoE codegen metadata.

Keep Opus MoE kernel tables in csrc, matching the opus_gemm_common.py pattern.
Runtime Python wrappers consume public kernel names; internal numeric kids stay
inside codegen/generated C++ dispatch.
"""

from __future__ import annotations

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
            spec = importlib.util.spec_from_file_location(module_name, path)
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
OPUS_A8W4_CODEGEN_SEED_EFFECTIVE_INTER_DIMS = (
    _a8w4_meta.OPUS_A8W4_CODEGEN_SEED_EFFECTIVE_INTER_DIMS
)
OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT = (
    _a8w4_meta.OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT
)
OPUS_A8W4_ROUTE_REDUCE_INSTANCES = _a8w4_meta.OPUS_A8W4_ROUTE_REDUCE_INSTANCES
OPUS_A8W4_OUT_MODE_ATOMIC = _a8w4_meta.OPUS_A8W4_OUT_MODE_ATOMIC
opus_a8w4_decode_kid = _a8w4_meta.opus_a8w4_decode_kid
opus_a8w4_effective_inter_dim = _a8w4_meta.opus_a8w4_effective_inter_dim

OPUS_A8W4_STAGE1_SCALE_GROUP_LOGICAL_K = 32
OPUS_A8W4_STAGE1_MFMA_K = 128
OPUS_A8W4_STAGE1_SCALE_K_PACK = 2


@dataclass(frozen=True)
class OpusA8W4Stage1Instance:
    kid: int
    name: str
    block_m: int
    block_n: int
    gate_up_group_split: bool = False
    k_wave: int = 1
    min_blocks_per_cu_override: int = 0
    skip_invalid_a_scale_guard: bool = False
    quant_group_blocks: int = 1
    activation: str = "silu"
    block_threads: int = 256

    @property
    def profile_name(self) -> str:
        # CSV/profile name: reuse the tuner's existing _fp8 convention while
        # keeping the device kid name stable.
        return self.name if self.name.endswith("_fp8") else f"{self.name}_fp8"


OPUS_A8W4_STAGE1_INSTANCES = (
    OpusA8W4Stage1Instance(
        kid=1000,
        name="opus_moe1_a8w4_bm32_bn384_gateup_groupsplit_caproutes_assumeroute_splitb_noclamp_min1",
        block_m=32,
        block_n=384,
        gate_up_group_split=True,
        min_blocks_per_cu_override=1,
        quant_group_blocks=6,
    ),
    OpusA8W4Stage1Instance(
        kid=1001,
        name="opus_moe1_a8w4_bm32_bn256_gateup_groupsplit_noscaleguard_qgb4",
        block_m=32,
        block_n=256,
        gate_up_group_split=True,
        skip_invalid_a_scale_guard=True,
        quant_group_blocks=4,
    ),
    OpusA8W4Stage1Instance(
        kid=1002,
        name="opus_moe1_a8w4_bm64_bn384_gateup_groupsplit_t4096_noclamp_min1_asynca_caproutes_assumeroute_splitb",
        block_m=64,
        block_n=384,
        gate_up_group_split=True,
        min_blocks_per_cu_override=1,
        quant_group_blocks=3,
    ),
    OpusA8W4Stage1Instance(
        kid=1003,
        name="opus_moe1_a8w4_bm64_bn256_gateup_groupsplit_t4096_noclamp_min1_asynca_caproutes_assumeroute_splitb_noscaleguard_qgb2",
        block_m=64,
        block_n=256,
        gate_up_group_split=True,
        min_blocks_per_cu_override=1,
        skip_invalid_a_scale_guard=True,
        quant_group_blocks=2,
    ),
    OpusA8W4Stage1Instance(
        kid=1004,
        name="opus_moe1_a8w4_bm64_bn256_gateup_groupsplit_t4096_noclamp_min1_asynca_caproutes_assumeroute_splitb_noscaleguard_qgb4",
        block_m=64,
        block_n=256,
        gate_up_group_split=True,
        min_blocks_per_cu_override=1,
        skip_invalid_a_scale_guard=True,
        quant_group_blocks=4,
    ),
    OpusA8W4Stage1Instance(
        kid=1005,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_min2_pairgateup_a_reuse_mfma",
        block_m=16,
        block_n=384,
        k_wave=4,
        min_blocks_per_cu_override=2,
    ),
    OpusA8W4Stage1Instance(
        kid=1006,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw4_caproutes_noclamp_min4_pairgateup_a_reuse_mfma",
        block_m=16,
        block_n=384,
        k_wave=4,
        min_blocks_per_cu_override=4,
    ),
    OpusA8W4Stage1Instance(
        kid=1007,
        name="opus_moe1_a8w4_bm32_bn384_g1_kw4_caproutes_noclamp_min3_pairgateup_a_reuse_mfma",
        block_m=32,
        block_n=384,
        k_wave=4,
        min_blocks_per_cu_override=3,
    ),
    OpusA8W4Stage1Instance(
        kid=1008,
        name="opus_moe1_a8w4_bm32_bn384_g1_kw7_caproutes_noclamp_min2_pairgateup_a_reuse_mfma",
        block_m=32,
        block_n=384,
        k_wave=7,
        min_blocks_per_cu_override=2,
    ),
    OpusA8W4Stage1Instance(
        kid=1009,
        name="opus_moe1_a8w4_bm32_bn384_g1_kw7_caproutes_noclamp_min3_pairgateup_a_reuse_mfma",
        block_m=32,
        block_n=384,
        k_wave=7,
        min_blocks_per_cu_override=3,
    ),
    OpusA8W4Stage1Instance(
        kid=1010,
        name="opus_moe1_a8w4_bm32_bn384_g1_kw4_caproutes_noclamp_min1_pairgateup_a_reuse_mfma",
        block_m=32,
        block_n=384,
        k_wave=4,
        min_blocks_per_cu_override=1,
    ),
    OpusA8W4Stage1Instance(
        kid=1011,
        name="opus_moe1_a8w4_bm128_bn256_gateup_groupsplit_noclamp_min2_noscaleguard_fullnexta",
        block_m=128,
        block_n=256,
        gate_up_group_split=True,
        skip_invalid_a_scale_guard=True,
        quant_group_blocks=2,
    ),
    OpusA8W4Stage1Instance(
        kid=1012,
        name="opus_moe1_a8w4_bm16_bn384_g1_kw2_min2_pairgateup_a_reuse_mfma_swiglu",
        block_m=16,
        block_n=384,
        k_wave=2,
        min_blocks_per_cu_override=2,
        activation="swiglu",
    ),
    OpusA8W4Stage1Instance(
        kid=1013,
        name="opus_moe1_a8w4_bm32_bn256_gateup_groupsplit_qgb4_swiglu",
        block_m=32,
        block_n=256,
        gate_up_group_split=True,
        skip_invalid_a_scale_guard=True,
        quant_group_blocks=4,
        activation="swiglu",
        block_threads=512,
    ),
)

OPUS_A8W4_STAGE1_BY_KID = {inst.kid: inst for inst in OPUS_A8W4_STAGE1_INSTANCES}


def opus_a8w4_stage1_output_cols_per_tile(
    inst: OpusA8W4Stage1Instance,
) -> int:
    return (
        inst.block_n // 2
        if inst.gate_up_group_split
        else OPUS_A8W4_STAGE1_SCALE_GROUP_LOGICAL_K
    )


def opus_a8w4_stage1_effective_inter_dim(
    logical_inter_dim: int,
    inter_dim_pad: int,
) -> int | None:
    logical_inter_dim = int(logical_inter_dim)
    inter_dim_pad = int(inter_dim_pad)
    if inter_dim_pad < 0 or logical_inter_dim <= inter_dim_pad:
        return None
    return logical_inter_dim - inter_dim_pad


def opus_a8w4_stage1_shape_values(
    *,
    model_dim: int,
    logical_inter_dim: int,
    inter_dim_pad: int,
) -> tuple[int, int] | None:
    model_dim = int(model_dim)
    logical_inter_dim = int(logical_inter_dim)
    effective_inter_dim = opus_a8w4_stage1_effective_inter_dim(
        logical_inter_dim, inter_dim_pad
    )
    if (
        effective_inter_dim is None
        or model_dim % OPUS_A8W4_STAGE1_MFMA_K != 0
        or logical_inter_dim % OPUS_A8W4_STAGE1_SCALE_GROUP_LOGICAL_K != 0
    ):
        return None
    return model_dim // OPUS_A8W4_STAGE1_MFMA_K, effective_inter_dim


def opus_a8w4_stage1_shape_requirements(
    inst: OpusA8W4Stage1Instance,
) -> tuple[int, int] | None:
    if (
        inst.gate_up_group_split
        and (inst.block_n // 2) % OPUS_A8W4_STAGE1_SCALE_GROUP_LOGICAL_K != 0
    ):
        return None
    return (
        int(inst.k_wave) * OPUS_A8W4_STAGE1_SCALE_K_PACK,
        opus_a8w4_stage1_output_cols_per_tile(inst),
    )


def opus_a8w4_stage1_instance_supports_shape(
    inst: OpusA8W4Stage1Instance,
    *,
    model_dim: int,
    logical_inter_dim: int,
    inter_dim_pad: int,
) -> bool:
    shape_values = opus_a8w4_stage1_shape_values(
        model_dim=model_dim,
        logical_inter_dim=logical_inter_dim,
        inter_dim_pad=inter_dim_pad,
    )
    requirements = opus_a8w4_stage1_shape_requirements(inst)
    if shape_values is None or requirements is None:
        return False
    k_steps, effective_inter_dim = shape_values
    k_step_multiple, output_cols_per_tile = requirements
    return (
        k_steps % k_step_multiple == 0
        and effective_inter_dim % output_cols_per_tile == 0
    )


def opus_a8w4_stage1_instances_for_shape(
    *,
    model_dim: int,
    logical_inter_dim: int,
    inter_dim_pad: int,
    activation: str | None = None,
    block_m: int | None = None,
) -> tuple[OpusA8W4Stage1Instance, ...]:
    act = None if activation is None else str(activation).strip().lower()
    bm = None if block_m is None else int(block_m)
    return tuple(
        inst
        for inst in OPUS_A8W4_STAGE1_INSTANCES
        if (act is None or str(inst.activation).lower() == act)
        and (bm is None or int(inst.block_m) == bm)
        and opus_a8w4_stage1_instance_supports_shape(
            inst,
            model_dim=model_dim,
            logical_inter_dim=logical_inter_dim,
            inter_dim_pad=inter_dim_pad,
        )
    )


def opus_a8w4_stage1_instance_requires_bias(
    inst: OpusA8W4Stage1Instance,
) -> bool:
    return str(inst.activation).lower() == "swiglu"


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

STAGE2_A8W4_KERNELS = dict(_a8w4_meta.OPUS_A8W4_STAGE2_BY_KID)
STAGE1_A8W4_KERNELS: dict[int, OpusA8W4Stage1Instance] = dict(OPUS_A8W4_STAGE1_BY_KID)
