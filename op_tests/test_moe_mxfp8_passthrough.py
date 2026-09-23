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
op_tests/test_moe_2stage.py and op_tests/flydsl_tests/test_flydsl_moe.py
already cover against a torch reference.

The harness mirrors how SGLang's MoRI dispatch calls this: per_1x32 MXFP4 expert
weights in the shuffled a8w4 layout and GateMode.INTERLEAVE. Swiglu tests force
the decode-sized activation policy to FP8. Plain-SiLU interleaved MXFP4 selects
FP8 directly on gfx950, so its regression runs with the default threshold.
"""

import os

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import (
    dynamic_per_group_scaled_quant,
    per_1x32_f4_quant,
    per_1x32_mx_quant_hip,
)
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

    w1_shuffled = shuffle_weight_a16w4(w1_q, 16, True)
    w2_shuffled = shuffle_weight_a16w4(w2_q, 16, False)
    w1_shuffled.is_shuffled = True
    w2_shuffled.is_shuffled = True

    return (
        x,
        w1_shuffled,
        w2_shuffled,
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


def test_silu_interleave_uses_a8w4_below_default_threshold(monkeypatch):
    """Plain-SiLU interleaved MXFP4 must not fall into unsupported A16W4."""
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "256")
    monkeypatch.setenv("AITER_FLYDSL_FORCE", "0")
    x, w1, w2, kwargs = _build(8)
    kwargs["activation"] = ActivationType.Silu

    internal = fused_moe(x, w1, w2, **kwargs)
    a1, a1_scale = per_1x32_mx_quant_hip(
        x,
        scale=None,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    passthrough = fused_moe(a1, w1, w2, a1_scale=a1_scale, **kwargs)

    assert torch.isfinite(internal).all()
    assert torch.equal(passthrough, internal)


def test_ep_config_key_supports_legacy_and_routed_only_topk():
    """Both EP route conventions must select the routed-top-k=8 MiMo row."""
    common = (8192, 6144, 2048, 24)
    kwargs = {
        "dtype": torch.bfloat16,
        "q_dtype_a": dtypes.fp8,
        "q_dtype_w": dtypes.fp4x2,
        "q_type": QuantType.per_1x32,
        "use_g1u1": True,
        "activation": ActivationType.Silu,
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "is_shuffled": True,
        "gate_mode": GateMode.INTERLEAVE.value,
        "is_ep": True,
    }

    get_2stage_cfgs.cache_clear()
    legacy = get_2stage_cfgs(*common, 9, ep_has_fake_route=True, **kwargs)
    routed_only = get_2stage_cfgs(*common, 8, ep_has_fake_route=False, **kwargs)

    legacy_names = (
        legacy.stage1.keywords["kernelName"],
        legacy.stage2.keywords["kernelName"],
    )
    routed_only_names = (
        routed_only.stage1.keywords["kernelName"],
        routed_only.stage2.keywords["kernelName"],
    )
    assert legacy_names == routed_only_names
    assert legacy_names == (
        "flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_gui_persist_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_persist",
    )


def test_routed_only_ep_topk_reaches_fused_moe():
    """The public API must accept an EP mask without requiring a fake route."""
    x, w1, w2, kwargs = _build(8)
    kwargs["activation"] = ActivationType.Silu
    expert_mask = torch.ones(EXPERTS, dtype=torch.int32, device=x.device)

    reference = fused_moe(x, w1, w2, **kwargs)
    routed_only = fused_moe(
        x,
        w1,
        w2,
        expert_mask=expert_mask,
        ep_has_fake_route=False,
        **kwargs,
    )

    torch.testing.assert_close(routed_only, reference, atol=1.0, rtol=0.05)


def test_device_row_limited_mxfp8_quant_graph_replay():
    """Persistent row-limited quantization must track the replay-time row count."""
    torch.manual_seed(13)
    capacity, cols = 8192, 512
    x = torch.randn((capacity, cols), dtype=torch.bfloat16, device="cuda") / 8
    full_out = torch.empty((capacity, cols), dtype=dtypes.fp8, device="cuda")
    full_scale = torch.empty(
        (capacity, cols // 32), dtype=dtypes.fp8_e8m0, device="cuda"
    )
    dynamic_per_group_scaled_quant(full_out, x, full_scale, 32, False)

    out = torch.empty_like(full_out)
    scale = torch.empty_like(full_scale)
    num_rows = torch.tensor([257], dtype=torch.int32, device="cuda")
    dynamic_per_group_scaled_quant(
        out, x, scale, 32, False, num_rows=num_rows, num_rows_factor=1
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dynamic_per_group_scaled_quant(
            out, x, scale, 32, False, num_rows=num_rows, num_rows_factor=1
        )

    for live_rows in (257, 1025):
        num_rows.fill_(live_rows)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:live_rows], full_out[:live_rows])
        assert torch.equal(scale[:live_rows], full_scale[:live_rows])
