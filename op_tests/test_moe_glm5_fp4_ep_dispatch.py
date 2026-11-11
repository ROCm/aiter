# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch coverage for the GLM-5 MXFP4 EP4 fused-MoE rows."""

import importlib
from pathlib import Path

import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_2stage_cfgs, get_padded_M

# With the shared expert fused, an EP4 rank holds 64 routed + 1 shared experts
# and topk_ids carries 8 routed + 1 shared column; fused_moe keys on
# topk_ids.shape[1].
EXPERTS = 65
TOPK = 9

DECODE = (
    "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_fp4",
    "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2",
    32,
)
PREFILL_16K = (
    "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_bnt0_fp4",
    "flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_atomic_persist_sbm64",
    64,
)
PREFILL_32K = (
    "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_bnt0_fp4",
    "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128",
    128,
)


@pytest.fixture
def gfx950_dispatch(monkeypatch):
    fused_moe = importlib.import_module("aiter.fused_moe")
    config = (
        Path(__file__).resolve().parents[1]
        / "aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv"
    )
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(config))
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe, "cfg_2stages", None)
    fused_moe.AITER_CONFIGS.get_config_file.cache_clear()
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()
    fused_moe.AITER_CONFIGS.get_config_file.cache_clear()


@pytest.mark.parametrize(
    ("m", "tier", "expected"),
    (
        (5, 8, DECODE),
        (12, 16, DECODE),
        (24, 32, DECODE),
        (36, 64, DECODE),
        (64, 64, DECODE),
        (8193, 16384, PREFILL_16K),
        (16385, 32768, PREFILL_32K),
        (131072, 131072, PREFILL_32K),
    ),
)
def test_ep4_tiers_use_fused_fp4_stage1(gfx950_dispatch, m, tier, expected):
    assert get_padded_M(m) == tier
    meta = get_2stage_cfgs(
        tier,
        6144,
        2048,
        EXPERTS,
        TOPK,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
        opus_weights_shuffled=True,
        is_ep=True,
    )
    stage1, stage2, block_m = expected
    assert meta.stage1.keywords["kernelName"] == stage1
    assert meta.stage2.keywords["kernelName"] == stage2
    assert meta.fuse_quant == "fp4"
    assert meta.block_m == block_m


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
