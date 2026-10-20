# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression for gfx950 fp8 CK stage2 at intermediate size 384.

The default preshuffle_off heuristic used a K tile of 256. The gfx950 fp8
16x16 matrix instruction consumes 128 K elements, so 384 was rejected while
512 was accepted. See https://github.com/ROCm/aiter/issues/6452.
"""

import pytest
import torch

from aiter import ActivationType, QuantType
from aiter import dtypes as aiter_dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="Exercises the gfx950 fp8 CK default dispatch",
)


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    # A tuned CSV row must not hide the heuristic bug.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _quantize(weight, quant_type):
    fp8 = aiter_dtypes.fp8
    fp8_max = torch.finfo(fp8).max
    dims = 2 if quant_type == QuantType.per_Token else (1, 2)
    amax = weight.abs().amax(dim=dims, keepdim=True).clamp(min=1e-6)
    scale = amax / fp8_max
    return (weight / scale).to(fp8), scale


def _run(inter_dim, quant_type, static_a1):
    torch.manual_seed(0)
    tokens, experts, topk, model_dim = 16, 8, 2, 896
    hidden = torch.randn(tokens, model_dim, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(
        experts, inter_dim * 2, model_dim, device="cuda", dtype=torch.bfloat16
    )
    w2 = torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=torch.bfloat16)
    score = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)
    w1_q, w1_scale = _quantize(w1, quant_type)
    w2_q, w2_scale = _quantize(w2, quant_type)
    a1_scale = (
        torch.tensor([1.0], dtype=torch.float32, device="cuda") if static_a1 else None
    )
    # Weights stay unshuffled: this is the JIT preshuffle_off path from the issue.
    actual = fused_moe(
        hidden,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        None,
        ActivationType.Silu,
        quant_type,
        False,
        w1_scale,
        w2_scale,
        a1_scale,
        None,
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    assert actual.abs().sum().item() > 0


@pytest.mark.parametrize("inter_dim", [384, 512])
@pytest.mark.parametrize(
    "quant_type,static_a1",
    [
        (QuantType.per_Token, False),
        (QuantType.per_Tensor, True),
        (QuantType.per_Tensor, False),
    ],
)
def test_gfx950_fp8_default_dispatch(inter_dim, quant_type, static_a1):
    _run(inter_dim, quant_type, static_a1)
