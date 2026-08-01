# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness test for the ported FlyDSL a16w4 (bf16 A x MXFP4 W) SiTUv2 path.

Exercises the production ``fused_moe`` SiTUv2 SEPARATED dispatch (routed to
``aiter/ops/flydsl/kernels/moe_2stage_a16wmix`` via
``aiter/ops/flydsl/moe_2stage_a16w4_dispatch.py``) against a bf16 SiTUv2 torch
reference, using the standard guinterleave (a8w4/mxfp8-shared) caller weight
layout. The replaced aiter kernel failed this same strict gate at
logits_diff ~1.0; the port passes at ~1.5e-5.

Run:
    pytest op_tests/flydsl_tests/test_a16w4_flydsl_situv2.py -q
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950") or not is_flydsl_available(),
    reason="CDNA (gfx942/gfx950) + FlyDSL required for a16w4 SiTUv2",
)


def _logits_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom)


@_SKIP
@pytest.mark.parametrize("model_dim,inter_dim", [(3584, 512), (3584, 384)])
@pytest.mark.parametrize("token", [1, 16, 128])
def test_a16w4_situv2_separated(model_dim, inter_dim, token):
    E, topk = 896, 16
    dtype = dtypes.bf16
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w1_scale_e = w1_scale.view(E, inter_dim * 2, model_dim // 32)
    w2_scale_e = w2_scale.view(E, model_dim, inter_dim // 32)

    # bf16 SiTUv2 SEPARATED reference
    o1 = torch_moe_stage1(
        input.to(dtype),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale_e,
        doweight=False,
        situ_beta=1.0,
        situ_linear_beta=1.0,
    )
    ref = torch_moe_stage2(
        o1.view(token, topk, inter_dim),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale_e,
        a2_scale=None,
        doweight=True,
    )

    # caller contract: standard guinterleave (a8w4/mxfp8-shared) weight layout
    w1_gui = shuffle_weight_a16w4(w1_qt, 16, True)
    w2_gui = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale_gui = shuffle_scale_a16w4(w1_scale, E, True)
    w2_scale_gui = shuffle_scale_a16w4(w2_scale, E, False)

    out = fused_moe(
        input,
        w1_gui,
        w2_gui,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale_gui,
        w2_scale=w2_scale_gui,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Situv2,
        doweight_stage1=False,
        gate_mode=GateMode.SEPARATED.value,
    )

    assert not out.isnan().any().item(), "a16w4 SiTUv2 output contains NaN"
    ld = _logits_diff(ref.float(), out.float())
    assert ld < 1e-2, f"a16w4 SiTUv2 logits_diff too large: {ld:.3e}"
