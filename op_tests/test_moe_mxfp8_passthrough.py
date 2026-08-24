"""Numerical test for the MXFP8 activation passthrough in fused_moe.

The passthrough lets a caller hand over activations that are already fp8 with
group-32 e8m0 microscales, so fused_moe sorts the scale and skips
requantization. Correctness means one thing: taking that shortcut must produce
the same result as letting fused_moe quantize the activations itself.

This mirrors the SGLang MoRI a8w4 call pattern: PER_1X32 MXFP4 weights,
GateMode.INTERLEAVE, and fp8 activations even at decode batch sizes (the
production path sets AITER_BF16_FP8_MOE_BOUND=0). Without those knobs a
standalone harness routes fp8 inputs into the mxfp4 quant branch and aborts
inside the HIP launcher — that is a test-setup failure, not evidence that the
passthrough is broken in serving.
"""

from __future__ import annotations

import os

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm device"),
    pytest.mark.skipif(get_gfx() not in ("gfx950",), reason="gfx950 a8w4 MoE required"),
]

# Decode-range token counts, including the single-token EP rank case.
TOKENS = [1, 8, 112]
# Small tuned-friendly shape; full DSV4 dims (7168, 256) also work but are slow.
MODEL_DIM = 2048
INTER_DIM = 512
EXPERTS = 8
TOPK = 2


@pytest.fixture(autouse=True)
def _force_fp8_activation_path():
    """Match serving: always pick the a8w4 fp8 activation branch, even for M=1."""
    old = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    yield
    if old is None:
        os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
    else:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = old


def _prepare_a8w4_weights(w1, w2):
    """MXFP4 expert weights in the layout fused_moe expects for a8w4."""
    e = w1.shape[0]
    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(e, w1.shape[1], w1.shape[2] // 2)
    w2_q = w2_q.view(e, w2.shape[1], w2.shape[2] // 2)
    w1_q = shuffle_weight_a16w4(w1_q, 16, gate_up=True)
    w2_q = shuffle_weight_a16w4(w2_q, 16, gate_up=False)
    w1_scale = shuffle_scale_a16w4(w1_scale, e, gate_up=True)
    w2_scale = shuffle_scale_a16w4(w2_scale, e, gate_up=False)
    return w1_q, w1_scale, w2_q, w2_scale


def _build(tokens, device, dtype=dtypes.bf16):
    torch.manual_seed(0)
    x = torch.randn(tokens, MODEL_DIM, dtype=dtype, device=device) / 10
    w1 = torch.randn(EXPERTS, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=device) / 10
    w2 = torch.randn(EXPERTS, MODEL_DIM, INTER_DIM, dtype=dtype, device=device) / 10

    score = torch.randn(tokens, EXPERTS, dtype=dtype, device=device)
    topk_weight, topk_ids = torch.topk(torch.softmax(score.float(), dim=-1), TOPK)
    w1_q, w1_scale, w2_q, w2_scale = _prepare_a8w4_weights(w1, w2)
    return x, w1_q, w2_q, w1_scale, w2_scale, topk_weight, topk_ids.to(torch.int32)


def _fused_moe_kwargs(topk_weight, topk_ids, w1_scale, w2_scale):
    return dict(
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        gate_mode=GateMode.INTERLEAVE.value,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        dtype=dtypes.bf16,
    )


@pytest.mark.parametrize("tokens", TOKENS)
def test_passthrough_matches_internal_quantization(tokens):
    """Pre-quantized fp8 + e8m0 activations must match internal requantization."""
    device = "cuda"
    x, w1_q, w2_q, w1_scale, w2_scale, topk_weight, topk_ids = _build(tokens, device)
    common = _fused_moe_kwargs(topk_weight, topk_ids, w1_scale, w2_scale)

    ref = fused_moe(x, w1_q, w2_q, **common)

    a1, a1_scale = per_1x32_f8_scale_f8_quant(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    got = fused_moe(a1, w1_q, w2_q, a1_scale=a1_scale, **common)

    assert got.shape == ref.shape
    assert torch.isfinite(got).all(), "passthrough produced non-finite output"
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_passthrough_is_not_taken_without_a_scale():
    """Without a1_scale, bf16 activations must still run through internal quant."""
    device = "cuda"
    x, w1_q, w2_q, w1_scale, w2_scale, topk_weight, topk_ids = _build(8, device)
    out = fused_moe(
        x,
        w1_q,
        w2_q,
        **_fused_moe_kwargs(topk_weight, topk_ids, w1_scale, w2_scale),
    )
    assert torch.isfinite(out).all()
