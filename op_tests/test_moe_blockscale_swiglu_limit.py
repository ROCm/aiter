# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 per_1x128 blockscale MoE must apply swiglu_limit.

Before the fix the limit was dropped silently: ASM 1-stage and CK 2-stage
ran an unclamped SiLU. Uses a tuned gfx942 shape so the tuned ASM rows in the
default config are hit and must be discarded when a limit is set.
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    asm_stage1,
    ck_moe_stage1,
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="Exercises the gfx942 FP8 per_1x128 MoE dispatch",
)

MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 4096, 1024, 289, 9
# Gate/up pre-activations need |x| well above the limit for the clamp to matter.
GATE_STD = 8.0


@pytest.fixture(autouse=True)
def fresh_dispatch():
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _quant_128x128(w):
    e, d1, d2 = w.shape
    blocks = (
        w.view(e, d1 // 128, 128, d2 // 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(e, -1, 128 * 128)
    )
    q, s = aiter.pertoken_quant(blocks, quant_dtype=dtypes.fp8)
    q = (
        q.view(e, d1 // 128, d2 // 128, 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(e, d1, d2)
    )
    return q, s.view(e, d1 // 128, d2 // 128)


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    w1 = torch.randn(
        EXPERTS, 2 * INTER_DIM, MODEL_DIM, device="cuda", dtype=dtypes.bf16
    ) * (GATE_STD / MODEL_DIM**0.5)
    w1_qt, w1_scale = _quant_128x128(w1)
    del w1
    w2 = (
        torch.randn(EXPERTS, MODEL_DIM, INTER_DIM, device="cuda", dtype=dtypes.bf16)
        / INTER_DIM**0.5
    )
    w2_qt, w2_scale = _quant_128x128(w2)
    del w2
    return dict(
        w1_qt=w1_qt,
        w2_qt=w2_qt,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_sh=shuffle_weight(w1_qt, layout=(16, 16)),
        w2_sh=shuffle_weight(w2_qt, layout=(16, 16)),
    )


def _calc_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def _reference(w, a1_qt, a1_scale, topk_weights, topk_ids, limit):
    out1 = torch_moe_stage1(
        a1_qt,
        w["w1_qt"],
        w["w2_qt"],
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        a1_scale=a1_scale,
        w1_scale=w["w1_scale"],
        swiglu_limit=limit,
    )
    tokens, topk = topk_ids.shape
    a2_qt, a2_scale = aiter.pertoken_quant(
        out1.view(tokens, -1, 128), quant_dtype=dtypes.fp8
    )
    return torch_moe_stage2(
        a2_qt.view(tokens, topk, -1),
        w["w1_qt"],
        w["w2_qt"],
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x128,
        w2_scale=w["w2_scale"],
        a2_scale=a2_scale.view(tokens, topk, -1),
        doweight=True,
    )


# 7.0 selects CK swiglustep_and_mul; 5.0 needs split-K + silu_and_mul(limit).
@pytest.mark.parametrize("limit", [7.0, 5.0, None])
@pytest.mark.parametrize("tokens", [1, 32, 1024])
def test_fused_moe_per_1x128_swiglu_limit(weights, tokens, limit):
    torch.manual_seed(tokens)
    x = torch.randn(tokens, MODEL_DIM, device="cuda", dtype=dtypes.bf16)
    score = torch.randn(tokens, EXPERTS, device="cuda", dtype=dtypes.bf16)
    topk_weights, topk_ids = fused_topk(x, score, TOPK, True)
    a1_qt, a1_scale = aiter.pertoken_quant(
        x.view(tokens, -1, 128), quant_dtype=dtypes.fp8
    )
    a1_qt, a1_scale = a1_qt.view(tokens, MODEL_DIM), a1_scale.squeeze(-1)

    out = fused_moe(
        x,
        weights["w1_sh"],
        weights["w2_sh"],
        topk_weights,
        topk_ids,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
        quant_type=QuantType.per_1x128,
        activation=ActivationType.Silu,
        swiglu_limit=limit,
    )
    torch.cuda.synchronize()
    assert not out.isnan().any()

    expected = _reference(weights, a1_qt, a1_scale, topk_weights, topk_ids, limit)
    logits_diff = _calc_diff(expected, out)
    assert logits_diff < 1e-3, f"logits_diff={logits_diff}"
    if limit:
        unclamped = _reference(weights, a1_qt, a1_scale, topk_weights, topk_ids, None)
        assert (
            _calc_diff(unclamped, out) > 1e-2
        ), "output matches the unclamped reference; swiglu_limit was not applied"


def test_asm_stage1_rejects_limit_without_splitk():
    with pytest.raises(NotImplementedError, match="swiglu_limit"):
        asm_stage1(
            *([None] * 6),
            torch.empty(1, 1, 1, device="cuda"),
            TOPK,
            32,
            ksplit=0,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x128,
            swiglu_limit=5.0,
        )


def test_ck_stage1_rejects_unsupported_limit_without_splitk():
    with pytest.raises(NotImplementedError, match="swiglu_limit"):
        ck_moe_stage1(
            torch.empty(1, MODEL_DIM, device="cuda"),
            *([None] * 6),
            TOPK,
            32,
            None,
            None,
            quant_type=QuantType.per_1x128,
            activation=ActivationType.Silu,
            splitk=1,
            swiglu_limit=5.0,
        )
