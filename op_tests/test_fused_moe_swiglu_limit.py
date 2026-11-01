# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""An active swiglu_limit must not be silently dropped.

The predicate and dispatch-guard tests do not launch a kernel. The bf16
QuantType.No / Silu case at the bottom is the ticket repro and needs a GPU;
it is skipped when CUDA is unavailable or the arch is not gfx942/gfx950.
"""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _flydsl_stage1_wrapper,
    _mxfp4_a4w4_stage1_fw,
    _opus_a8w4_stage1_wrapper,
    _raise_if_swiglu_limit_dropped,
    _raise_if_whole_graph_drops_swiglu_limit,
    _swiglu_limit_is_active,
    asm_stage1,
    ck_moe_stage1,
    cktile_moe_stage1,
    fused_moe,
    fused_moe_1stage,
    get_2stage_cfgs,
)

_INACTIVE = (None, 0.0, 0, float("inf"))
_NONCONSUMING = (ck_moe_stage1, cktile_moe_stage1, asm_stage1, fused_moe_1stage)
_CONSUMING = (
    _flydsl_stage1_wrapper,
    _opus_a8w4_stage1_wrapper,
    _mxfp4_a4w4_stage1_fw,
)


@pytest.fixture
def gfx942(monkeypatch):
    monkeypatch.setattr("aiter.fused_moe.get_gfx", lambda: "gfx942")


@pytest.mark.parametrize(
    ("limit", "active"),
    [
        (None, False),
        (0.0, False),
        (0, False),
        (float("inf"), False),
        (float("-inf"), False),
        (float("nan"), False),
        (-1.0, False),
        (1.0, True),
        (7.0, True),
        (10.0, True),
    ],
)
def test_swiglu_limit_is_active(limit, active):
    assert _swiglu_limit_is_active(limit) is active


@pytest.mark.parametrize("limit", _INACTIVE)
@pytest.mark.parametrize("stage1", _NONCONSUMING)
def test_inactive_sentinels_do_not_raise(gfx942, stage1, limit):
    _raise_if_swiglu_limit_dropped(stage1, limit, ActivationType.Silu)


@pytest.mark.parametrize("stage1", _NONCONSUMING)
def test_nonconsuming_stage1_raises_for_finite_limit(gfx942, stage1):
    with pytest.raises(NotImplementedError, match="silently dropped") as exc:
        _raise_if_swiglu_limit_dropped(stage1, 10.0, ActivationType.Silu)
    message = str(exc.value)
    assert stage1.__name__ in message
    assert "swiglu_limit=10.0" in message
    assert "ActivationType.Silu" in message
    assert "gfx942" in message
    assert "FlyDSL and Opus" in message


@pytest.mark.parametrize("stage1", _CONSUMING)
@pytest.mark.parametrize(
    "activation",
    [ActivationType.Silu, ActivationType.Swiglu],
)
def test_consuming_stage1_accepts_finite_limit(gfx942, stage1, activation):
    _raise_if_swiglu_limit_dropped(stage1, 7.0, activation)


def test_mxfp4_situv2_cannot_apply_finite_limit(gfx942):
    with pytest.raises(NotImplementedError, match="_mxfp4_a4w4_stage1_fw"):
        _raise_if_swiglu_limit_dropped(
            _mxfp4_a4w4_stage1_fw, 10.0, ActivationType.Situv2
        )


def test_fhmoe_wrapper_accepts_finite_limit(gfx942):
    from aiter.fhmoe import _flydsl_fhmoe_stage1_wrapper

    _raise_if_swiglu_limit_dropped(
        _flydsl_fhmoe_stage1_wrapper, 10.0, ActivationType.Silu
    )


def test_gfx942_whole_graph_silu_raises(gfx942):
    with pytest.raises(NotImplementedError, match="flydsl_gfx942") as exc:
        _raise_if_whole_graph_drops_swiglu_limit(1.0, ActivationType.Silu)
    assert "SwiGLU only" in str(exc.value)


@pytest.mark.parametrize("limit", _INACTIVE)
def test_gfx942_whole_graph_inactive_limit_does_not_raise(gfx942, limit):
    _raise_if_whole_graph_drops_swiglu_limit(limit, ActivationType.Silu)


def test_gfx942_whole_graph_swiglu_does_not_raise(gfx942):
    _raise_if_whole_graph_drops_swiglu_limit(10.0, ActivationType.Swiglu)


@pytest.mark.parametrize("token", [4, 256])
def test_bf16_noquant_silu_repro_raises(monkeypatch, token):
    """Repro: QuantType.No + Silu used to return an unclamped tensor.

    AITER_BYPASS_TUNE_CONFIG forces the empty-kernelName1 fallthrough. On
    gfx942, token < 256 selects fused_moe_1stage and token >= 256 selects
    ck_moe_stage1. gfx950 has no 1-stage row for this quant, so both sizes
    select ck_moe_stage1. Either way the bound used to be dropped.
    """
    if not torch.cuda.is_available():
        pytest.skip("fused_moe launch needs a GPU")
    from aiter.jit.utils.chip_info import get_gfx

    arch = get_gfx()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"bf16 no-quant fallthrough is not asserted on {arch}")

    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    try:
        experts, model_dim, inter_dim, topk = 4, 128, 128, 2
        hidden = torch.randn(token, model_dim, dtype=dtypes.bf16, device="cuda")
        w1 = torch.randn(
            experts, inter_dim, model_dim, dtype=dtypes.bf16, device="cuda"
        )
        w2 = torch.randn(
            experts, model_dim, inter_dim, dtype=dtypes.bf16, device="cuda"
        )
        topk_ids = torch.randint(0, experts, (token, topk), device="cuda")
        topk_weights = torch.softmax(torch.randn(token, topk, device="cuda"), dim=-1)
        with pytest.raises(NotImplementedError, match="cannot apply swiglu_limit"):
            fused_moe(
                hidden,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation=ActivationType.Silu,
                quant_type=QuantType.No,
                swiglu_limit=1.0,
            )
    finally:
        get_2stage_cfgs.cache_clear()
