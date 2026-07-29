# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
from pathlib import Path

import torch

import aiter.fused_moe as fused_moe  # noqa: PLR0402
from aiter import ActivationType, QuantType, dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.moe_common import GateMode

MODEL_CONFIG = (
    Path(__file__).parents[1] / "aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv"
)
PROMOTED_PAIR = (
    "flydsl_moe1_afp8_wfp4_bf16_t128x256x256_bnt0_gui_fp8",
    "flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_persist_sbm128",
)
INDEX_COLUMNS = (
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)


def _kernel_pair(metadata):
    return tuple(
        getattr(stage, "keywords", {}).get("kernelName", "")
        for stage in (metadata.stage1, metadata.stage2)
    )


def test_dsv4_promoted_model_rows_are_exact_and_unique():
    with MODEL_CONFIG.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    active_rows = [row for row in rows if not row["_tag"]]
    keys = [tuple(row[column] for column in INDEX_COLUMNS) for row in active_rows]
    assert len(keys) == len(set(keys))

    promoted_rows = {
        int(row["token"]): row
        for row in active_rows
        if row["cu_num"] == "256"
        and row["model_dim"] == "7168"
        and row["inter_dim"] == "3072"
        and row["expert"] == "48"
        and row["topk"] == "6"
        and row["token"] in {"16384", "32768"}
    }
    assert set(promoted_rows) == {16384, 32768}
    for row in promoted_rows.values():
        assert (row["kernelName1"], row["kernelName2"]) == PROMOTED_PAIR
    assert {
        key: value for key, value in promoted_rows[32768].items() if key != "token"
    } == {key: value for key, value in promoted_rows[16384].items() if key != "token"}


def test_dsv4_promoted_rows_load_through_default_model_config_merge(
    monkeypatch,
):
    monkeypatch.delenv("AITER_CONFIG_FMOE", raising=False)
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    old_cfg = fused_moe.cfg_2stages
    AITER_CONFIGS.get_config_file.cache_clear()
    fused_moe.get_2stage_cfgs.cache_clear()
    fused_moe.cfg_2stages = None
    try:
        for runtime_m in (32768, 32769, 65535, 65536, 131071):
            padded_m = fused_moe.get_padded_M(runtime_m)
            assert padded_m == 32768
            metadata = fused_moe.get_2stage_cfgs(
                padded_m,
                7168,
                3072,
                48,
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
                GateMode.INTERLEAVE.value,
                is_ep=True,
                ep_has_fake_expert=False,
            )
            assert _kernel_pair(metadata) == PROMOTED_PAIR
            assert metadata.fuse_quant == "fp8"
    finally:
        fused_moe.get_2stage_cfgs.cache_clear()
        fused_moe.cfg_2stages = old_cfg
        AITER_CONFIGS.get_config_file.cache_clear()


def test_resolve_tuning_topk_contract():
    assert fused_moe._resolve_tuning_topk(7, is_ep=True) == 6
    assert fused_moe._resolve_tuning_topk(6, is_ep=True, ep_has_fake_expert=False) == 6
    assert fused_moe._resolve_tuning_topk(6, is_ep=False) == 6


def test_ep_fake_expert_env_is_default_on_and_explicitly_disabled(
    monkeypatch,
):
    monkeypatch.delenv("AITER_FLYDSL_EP_NO_FAKE_EXPERT", raising=False)
    assert fused_moe._ep_has_fake_expert_for_tuning() is True

    monkeypatch.setenv("AITER_FLYDSL_EP_NO_FAKE_EXPERT", "1")
    assert fused_moe._ep_has_fake_expert_for_tuning() is False


def test_get_2stage_cfgs_cache_separates_fake_expert_contract(monkeypatch):
    common_key = (
        "gfx950",
        256,
        512,
        7168,
        3072,
        48,
    )
    common_types = (
        ActivationType.Silu,
        str(torch.bfloat16),
        str(dtypes.fp8),
        str(dtypes.fp4x2),
        str(QuantType.per_1x32),
        True,
        False,
    )

    def cfg(block_m):
        return {
            "block_m": block_m,
            "ksplit": 0,
            "kernelName1": "test_stage1",
            "kernelName2": "test_stage2",
            "run_1stage": False,
        }

    monkeypatch.setattr(
        fused_moe,
        "cfg_2stages",
        (
            {
                common_key + (6,) + common_types: cfg(64),
                common_key + (7,) + common_types: cfg(128),
            },
            {},
        ),
    )
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe, "is_flydsl_available", lambda: False)

    args = (
        512,
        7168,
        3072,
        48,
        7,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
    )
    fused_moe.get_2stage_cfgs.cache_clear()
    try:
        legacy = fused_moe.get_2stage_cfgs(*args, is_ep=True)
        candidate = fused_moe.get_2stage_cfgs(
            *args, is_ep=True, ep_has_fake_expert=False
        )
        assert legacy.block_m == 64
        assert candidate.block_m == 128
        assert fused_moe.get_2stage_cfgs.cache_info().misses == 2

        assert fused_moe.get_2stage_cfgs(*args, is_ep=True) is legacy
        assert (
            fused_moe.get_2stage_cfgs(*args, is_ep=True, ep_has_fake_expert=False)
            is candidate
        )
        assert fused_moe.get_2stage_cfgs.cache_info().hits == 2
    finally:
        fused_moe.get_2stage_cfgs.cache_clear()
