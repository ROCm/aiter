# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch coverage for the GLM-5 MXFP4 EP4 decode rows."""

import importlib
from pathlib import Path

import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_2stage_cfgs, get_padded_M

STAGE1 = "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_fp4"
STAGE2 = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2"


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
    ("m", "tier"),
    ((5, 8), (12, 16), (24, 32), (36, 64), (64, 64)),
)
def test_decode_tiers_use_fused_fp4_stage1(gfx950_dispatch, m, tier):
    assert get_padded_M(m) == tier
    meta = get_2stage_cfgs(
        tier,
        6144,
        2048,
        65,
        8,
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
    assert meta.stage1.keywords["kernelName"] == STAGE1
    assert meta.stage2.keywords["kernelName"] == STAGE2
    assert meta.fuse_quant == "fp4"
    assert meta.block_m == 32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
