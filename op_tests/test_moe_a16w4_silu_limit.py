# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""a16w4 (bf16 A x MXFP4 W) SiLU MoE with a swiglu_limit, the small-batch route of
DeepSeek-V4.1 on gfx950 (token count below AITER_BF16_FP8_MOE_BOUND).

CK2stages has no A16W4 SiLU kernel and no clamp, so this route must reach CK-Tile's
split-K post-activation path and apply swiglu_limit there. Checked end to end through
fused_moe against the clamped torch reference, on gate/up-interleaved weights.

Run:
    pytest op_tests/test_moe_a16w4_silu_limit.py -q
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="gfx950 required for a16w4 MXFP4 MoE"
)

# DeepSeek-V4.1-Flash EP4: model_dim 5120, inter_dim 2304; fewer experts keep it quick
MODEL_DIM, INTER_DIM, E, TOPK = 5120, 2304, 32, 6
SWIGLU_LIMIT = 7.0


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


@_SKIP
@pytest.mark.parametrize("token", [1, 32])
def test_a16w4_silu_applies_swiglu_limit(token):
    gate_mode = GateMode.INTERLEAVE  # the layout the SiLU route is verified on
    dtype = dtypes.bf16
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # scaled so a good share of gate/up values exceed the limit
    inp = torch.randn((token, MODEL_DIM), dtype=dtype, device="cuda") * 4
    w1 = torch.randn((E, INTER_DIM * 2, MODEL_DIM), dtype=dtype, device="cuda") / 8
    w2 = torch.randn((E, MODEL_DIM, INTER_DIM), dtype=dtype, device="cuda") / 16
    score = torch.randn((token, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, TOPK, True)

    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, INTER_DIM * 2, MODEL_DIM // 2)
    w2_qt = w2_qt.view(E, MODEL_DIM, INTER_DIM // 2)

    def reference(swiglu_limit):
        o1 = torch_moe_stage1(
            inp,
            w1_qt.view(dtypes.fp4x2),
            w2_qt.view(dtypes.fp4x2),
            topk_weights,
            topk_ids,
            dtype=dtype,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale.view(E, INTER_DIM * 2, MODEL_DIM // 32),
            swiglu_limit=swiglu_limit,
        )
        return torch_moe_stage2(
            o1.view(token, TOPK, INTER_DIM),
            w1_qt.view(dtypes.fp4x2),
            w2_qt.view(dtypes.fp4x2),
            topk_weights,
            topk_ids,
            dtype=dtype,
            quant_type=QuantType.per_1x32,
            w2_scale=w2_scale.view(E, MODEL_DIM, INTER_DIM // 32),
            doweight=True,
        )

    ref = reference(SWIGLU_LIMIT)
    unclamped = reference(None)
    assert (
        _cos_diff(ref.float(), unclamped.float()) > 1e-2
    ), "inputs never reach the limit"

    gate_up = gate_mode == GateMode.INTERLEAVE
    out = fused_moe(
        inp,
        shuffle_weight_a16w4(w1_qt, 16, gate_up),
        shuffle_weight_a16w4(w2_qt, 16, False),
        topk_weights,
        topk_ids,
        w1_scale=shuffle_scale_a16w4(w1_scale, E, gate_up),
        w2_scale=shuffle_scale_a16w4(w2_scale, E, False),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
        doweight_stage1=False,
        gate_mode=gate_mode.value,
        swiglu_limit=SWIGLU_LIMIT,
    )
    assert not out.isnan().any().item(), "a16w4 SiLU output contains NaN"
    diff = _cos_diff(ref.float(), out.float())
    assert diff < 1e-2, f"a16w4 SiLU with swiglu_limit: cos diff {diff:.3e}"
