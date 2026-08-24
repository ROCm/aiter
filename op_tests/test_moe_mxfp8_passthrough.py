# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical test for the MXFP8 activation passthrough in fused_moe.

The passthrough lets a caller hand over activations that are already fp8 with
group-32 e8m0 microscales, so fused_moe sorts the scale and skips
requantization. Its entire claim is that taking the shortcut changes nothing,
and that is testable exactly: pre-quantize with the same HIP MX quantizer the
internal path uses, and both sides feed identical fp8 bytes and identical scale
values to the same GEMM. The only thing left varying is which code sorts the
scale, so the two outputs must be *bit-identical* -- not merely close.

Exactness matters here because the failure mode is silent. The scale is consumed
bytewise, so a mis-sorted or mis-strided scale returns plausible-looking wrong
numbers rather than raising, and any tolerance wide enough to absorb quantizer
noise would also absorb a real defect. There is no quantizer noise to absorb.

Scope, stated so it is not overclaimed: this pins the passthrough against the
path it replaces. It is not an absolute-correctness test for the a8w4 MoE, which
op_tests/test_moe_2stage.py and op_tests/flydsl_tests/test_flydsl_moe_a8w4.py
already cover against a torch reference.

The harness mirrors how SGLang's MoRI dispatch calls this: per_1x32 MXFP4 expert
weights in the shuffled a8w4 layout, Swiglu with GateMode.INTERLEAVE, and
AITER_BF16_FP8_MOE_BOUND=0 so decode-sized batches resolve q_dtype_a to fp8.
Those knobs are load-bearing, and not merely for reaching the branch: under
GateMode.SEPARATED, Swiglu resolves q_dtype_a to bf16, and fp8 activations are
then swallowed by the earlier `a1 = hidden_states.to(dtype)` branch, which casts
the fp8 bytes numerically and drops a1_scale. That is pre-existing behaviour and
not what this branch does, but it is why the gate mode here is not incidental.
"""

import os

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_mx_quant_hip
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm device"),
    pytest.mark.skipif(get_gfx() not in ("gfx950",), reason="gfx950 a8w4 MoE required"),
]

# Decode-range token counts, including the single-token EP rank case.
TOKENS = [1, 8, 112]
MODEL_DIM = 2048
INTER_DIM = 512
EXPERTS = 8
TOPK = 2


@pytest.fixture(autouse=True)
def _force_fp8_activation_path():
    """Match serving: keep q_dtype_a at fp8 even for decode-sized batches."""
    old = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    yield
    if old is None:
        os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
    else:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = old


def _build(tokens, device="cuda", dtype=dtypes.bf16):
    torch.manual_seed(0)
    x = torch.randn(tokens, MODEL_DIM, dtype=dtype, device=device) / 10
    w1 = torch.randn(EXPERTS, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=device) / 10
    w2 = torch.randn(EXPERTS, MODEL_DIM, INTER_DIM, dtype=dtype, device=device) / 10
    score = torch.randn(tokens, EXPERTS, dtype=dtype, device=device)
    topk_weight, topk_ids = fused_topk(x, score, TOPK, True)

    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(EXPERTS, INTER_DIM * 2, MODEL_DIM // 2)
    w2_q = w2_q.view(EXPERTS, MODEL_DIM, INTER_DIM // 2)

    return (
        x,
        shuffle_weight_a16w4(w1_q, 16, True),
        shuffle_weight_a16w4(w2_q, 16, False),
        {
            "topk_weight": topk_weight,
            "topk_ids": topk_ids,
            "quant_type": QuantType.per_1x32,
            "activation": ActivationType.Swiglu,
            "gate_mode": GateMode.INTERLEAVE.value,
            "w1_scale": shuffle_scale_a16w4(w1_scale, EXPERTS, True),
            "w2_scale": shuffle_scale_a16w4(w2_scale, EXPERTS, False),
            "dtype": dtypes.bf16,
        },
    )


@pytest.mark.parametrize("tokens", TOKENS)
def test_passthrough_is_bit_identical_to_internal_quantization(tokens):
    """Taking the shortcut must change nothing at all."""
    x, w1, w2, kwargs = _build(tokens)

    internal = fused_moe(x, w1, w2, **kwargs)

    # Same quantizer the internal path uses, so the GEMM sees identical bytes.
    a1, a1_scale = per_1x32_mx_quant_hip(
        x,
        scale=None,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    assert a1.dtype == dtypes.fp8 and a1_scale.dtype == dtypes.fp8_e8m0
    passthrough = fused_moe(a1, w1, w2, a1_scale=a1_scale, **kwargs)

    assert torch.isfinite(passthrough).all(), "passthrough produced non-finite output"
    assert torch.equal(passthrough, internal), (
        "passthrough is not bit-identical to internal requantization; the sorted "
        "activation scale is not the scale that belongs to these tokens "
        f"(max |delta| = {(passthrough.float() - internal.float()).abs().max().item():.3e})"
    )
