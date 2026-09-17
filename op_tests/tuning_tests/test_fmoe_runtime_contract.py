# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import importlib
from pathlib import Path

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import _resolve_tuning_topk
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

_DSV41_STAGE1 = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw2_fp8"
_MODEL_CONFIG = (
    Path(__file__).parents[2]
    / "aiter/configs/model_configs/dsv41_flash_fp8fp4_tuned_fmoe.csv"
)


def test_ep_tuning_topk_preserves_routes_without_fake_expert():
    assert _resolve_tuning_topk(6, is_ep=True, ep_has_fake_expert=False) == 6
    assert _resolve_tuning_topk(7, is_ep=True, ep_has_fake_expert=True) == 6
    assert _resolve_tuning_topk(6, is_ep=False) == 6


def test_a8w4_stage1_runtime_contract_uses_real_scales():
    params = get_flydsl_kernel_params(_DSV41_STAGE1)
    assert params is not None
    assert params["a_dtype"] == "fp8"
    assert params["b_dtype"] == "fp4"
    assert params["out_dtype"] == "fp8"
    assert params["gate_mode"] == "interleave"
    assert params.get("a_scale_one", False) is False


def test_dsv41_rows_round_trip_through_production_dispatch(monkeypatch):
    fused_moe = importlib.import_module("aiter.fused_moe")
    with _MODEL_CONFIG.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert {int(row["token"]) for row in rows} == {
        1,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    }
    assert {row["topk"] for row in rows} == {"6"}

    monkeypatch.delenv("AITER_CONFIG_FMOE", raising=False)
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    old_cfg = fused_moe.cfg_2stages
    old_by_file = dict(fused_moe.cfg_2stages_by_file)
    AITER_CONFIGS.get_config_file.cache_clear()
    fused_moe.get_2stage_cfgs.cache_clear()
    fused_moe.cfg_2stages = None
    fused_moe.cfg_2stages_by_file.clear()
    try:
        for row in rows:
            metadata = fused_moe.get_2stage_cfgs(
                int(row["token"]),
                5120,
                2304,
                96,
                6,
                torch.bfloat16,
                dtypes.fp8,
                dtypes.fp4x2,
                QuantType.per_1x32,
                True,
                ActivationType.Silu,
                False,
                0,
                0,
                True,
                "interleave",
                is_ep=True,
                ep_has_fake_expert=False,
            )
            stage1 = metadata.stage1.keywords["kernelName"]
            stage2 = metadata.stage2.keywords["kernelName"]
            assert (stage1, stage2) == (
                row["kernelName1"],
                row["kernelName2"],
            )
            assert metadata.fuse_quant == "fp8"
            assert get_flydsl_kernel_params(stage1).get("a_scale_one", False) is False
    finally:
        fused_moe.get_2stage_cfgs.cache_clear()
        fused_moe.cfg_2stages = old_cfg
        fused_moe.cfg_2stages_by_file.clear()
        fused_moe.cfg_2stages_by_file.update(old_by_file)
        AITER_CONFIGS.get_config_file.cache_clear()
