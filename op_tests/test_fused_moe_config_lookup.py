# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect
from unittest import mock

import pytest
import torch

import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe as fused_moe_api


def _key(topk):
    return (
        "gfx950",
        256,
        4,
        6144,
        3072,
        32,
        topk,
        ActivationType.Swiglu,
        str(dtypes.bf16),
        str(dtypes.fp8),
        str(dtypes.fp8),
        str(QuantType.per_1x32),
        True,
        False,
    )


def _config(block_m, label):
    return {
        "block_m": block_m,
        "ksplit": 0,
        "kernelName1": label,
        "kernelName2": "",
        "run_1stage": True,
        "xbf16": 0,
        "flat": 0,
    }


def _lookup(runtime_topk, rows, has_fake_topk_slot=None):
    old_cfg = fused_moe_module.cfg_2stages
    fused_moe_module.get_2stage_cfgs.cache_clear()
    fused_moe_module.cfg_2stages = (rows, {})
    try:
        with (
            mock.patch.object(fused_moe_module, "get_cu_num", return_value=256),
            mock.patch.object(
                fused_moe_module, "get_gfx_runtime", return_value="gfx950"
            ),
        ):
            return fused_moe_module.get_2stage_cfgs(
                4,
                6144,
                3072,
                32,
                runtime_topk,
                dtypes.bf16,
                dtypes.fp8,
                dtypes.fp8,
                QuantType.per_1x32,
                True,
                ActivationType.Swiglu,
                False,
                0,
                0,
                is_ep=True,
                has_fake_topk_slot=has_fake_topk_slot,
            )
    finally:
        fused_moe_module.cfg_2stages = old_cfg
        fused_moe_module.get_2stage_cfgs.cache_clear()


def test_standard_ep_prefers_actual_topk_width():
    rows = {
        _key(4): _config(64, "standard_topk4"),
        _key(3): _config(32, "legacy_topk3"),
    }

    metadata = _lookup(4, rows, has_fake_topk_slot=False)

    assert metadata.block_m == 64
    assert metadata.stage1.keywords["kernelName"] == "standard_topk4"


def test_explicit_fake_slot_selects_routed_topk_width():
    rows = {
        _key(5): _config(32, "actual_topk5"),
        _key(4): _config(64, "fake_slot_routed_topk4"),
    }

    metadata = _lookup(5, rows, has_fake_topk_slot=True)

    assert metadata.block_m == 64
    assert metadata.stage1.keywords["kernelName"] == "fake_slot_routed_topk4"


def test_legacy_fake_slot_fallback_remains_available():
    rows = {
        _key(5): _config(32, "runtime_width"),
        _key(4): _config(64, "legacy_fake_slot"),
    }
    metadata = _lookup(5, rows)

    assert metadata.block_m == 64
    assert metadata.stage1.keywords["kernelName"] == "legacy_fake_slot"


def test_fake_slot_flag_requires_ep_and_an_extra_slot():
    candidates = fused_moe_module._fmoe_config_topk_candidates
    assert candidates(4, is_ep=True, has_fake_topk_slot=False) == (4,)
    assert candidates(4, is_ep=True, has_fake_topk_slot=None) == (3, 4)
    assert candidates(5, is_ep=True, has_fake_topk_slot=True) == (4,)
    with pytest.raises(ValueError, match="only valid"):
        candidates(4, is_ep=False, has_fake_topk_slot=True)
    with pytest.raises(ValueError, match="runtime topk"):
        candidates(1, is_ep=True, has_fake_topk_slot=True)


def test_public_api_exposes_explicit_fake_slot_keyword():
    assert "has_fake_topk_slot" in fused_moe_api.__code__.co_varnames
    assert "has_fake_topk_slot" in inspect.signature(
        fused_moe_module.fused_moe_fake
    ).parameters
    schema = torch._C._dispatch_find_schema_or_throw(
        "aiter::fused_moe_", ""
    ).schema()
    assert "bool? has_fake_topk_slot=None" in str(schema)
