# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import aiter
import pytest
from aiter import dtypes
from aiter.fused_moe import get_2stage_cfgs
from aiter.ops.flydsl.moe_common import GateMode


@pytest.fixture(autouse=True)
def _bypass_tuned_cfg(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")


def test_swiglu_bf16_fp4_cktile_uses_zero_npad_for_padded_separated_w13():
    metadata = get_2stage_cfgs(
        token=128,
        model_dim=6144,
        inter_dim=768,
        expert=129,
        topk=5,
        dtype=dtypes.bf16,
        q_dtype_a=dtypes.bf16,
        q_dtype_w=dtypes.fp4x2,
        q_type=aiter.QuantType.per_1x32,
        use_g1u1=True,
        activation=aiter.ActivationType.Swiglu,
        doweight_stage1=False,
        hidden_pad=0,
        intermediate_pad=128,
        is_shuffled=True,
        gate_mode=GateMode.SEPARATED.value,
    )

    assert metadata.stage1.func.__name__ == "cktile_moe_stage1"
    assert metadata.stage1.keywords["n_pad_zeros"] == 0
    assert metadata.stage1.keywords["post_activation_layout"] == "standard"


def test_non_swiglu_ksplit_cktile_preserves_legacy_npad_formula():
    token = 8
    topk = 1
    inter_dim = 3072
    model_dim = 3072
    intermediate_pad = 128

    metadata = get_2stage_cfgs(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        expert=128,
        topk=topk,
        dtype=dtypes.bf16,
        q_dtype_a=dtypes.bf16,
        q_dtype_w=dtypes.fp4x2,
        q_type=aiter.QuantType.per_1x32,
        use_g1u1=True,
        activation=aiter.ActivationType.Silu,
        doweight_stage1=False,
        hidden_pad=0,
        intermediate_pad=intermediate_pad,
        is_shuffled=True,
        gate_mode=GateMode.SEPARATED.value,
    )

    assert metadata.stage1.func.__name__ == "cktile_moe_stage1"
    assert metadata.ksplit > 1
    assert metadata.stage1.keywords["post_activation_layout"] == "auto"
    expected_npad = intermediate_pad // 64 * 64 * 2
    assert metadata.stage1.keywords["n_pad_zeros"] == expected_npad
