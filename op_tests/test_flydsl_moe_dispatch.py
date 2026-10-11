# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for untuned MX MoE dispatch policy."""

from __future__ import annotations

import pytest

import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes


@pytest.fixture(autouse=True)
def _clear_dispatch_cache():
    fm.get_2stage_cfgs.cache_clear()
    yield
    fm.get_2stage_cfgs.cache_clear()


def _mock_dispatch(monkeypatch, q_dtype_a, q_dtype_w, activation):
    monkeypatch.setattr(fm, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fm, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(fm, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fm, "is_flydsl_available", lambda: True)
    monkeypatch.setattr(fm, "cfg_2stages", ({}, {}))

    return fm.get_2stage_cfgs(
        64,
        2560,
        768,
        64,
        10,
        dtypes.bf16,
        q_dtype_a,
        q_dtype_w,
        QuantType.per_1x32,
        True,
        activation,
        False,
        0,
        128,
        is_shuffled=True,
        is_ep=True,
    )


def test_silu_a4w4_uses_native_ck(monkeypatch):
    metadata = _mock_dispatch(
        monkeypatch,
        dtypes.fp4x2,
        dtypes.fp4x2,
        ActivationType.Silu,
    )

    assert metadata.stage1.func is fm.ck_moe_stage1


@pytest.mark.parametrize(
    "q_dtype_w",
    [
        pytest.param(dtypes.fp4x2, id="a8w4"),
        pytest.param(dtypes.fp8, id="a8w8"),
    ],
)
def test_silu_a8w_flydsl_remains_default(monkeypatch, q_dtype_w):
    metadata = _mock_dispatch(
        monkeypatch,
        dtypes.fp8,
        q_dtype_w,
        ActivationType.Silu,
    )

    assert metadata.stage1.func is fm._flydsl_stage1_wrapper


@pytest.mark.parametrize(
    "activation",
    [
        pytest.param(ActivationType.Swiglu, id="swiglu"),
        pytest.param(ActivationType.Situv2, id="situv2"),
    ],
)
def test_explicit_a4w4_flydsl_activations_remain_enabled(monkeypatch, activation):
    metadata = _mock_dispatch(
        monkeypatch,
        dtypes.fp4x2,
        dtypes.fp4x2,
        activation,
    )

    assert metadata.stage1.func is fm._flydsl_stage1_wrapper


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
