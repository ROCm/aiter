# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx1201",
    reason="gfx1201 selected large-M BF16 G1U1 Triton MoE coverage",
)

_RTOL = 1e-2
_ATOL = 128


def _make_case(tokens: int):
    torch.manual_seed(2026 + tokens)
    model_dim = 2048
    inter_dim = 128
    expert = 256
    topk = 8
    hidden = torch.randn((tokens, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(
        (expert, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda")
    ids = torch.randint(0, expert, (tokens, topk), dtype=torch.int32, device="cuda")
    weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return hidden, w1.contiguous(), w2.contiguous(), weights.contiguous(), ids.contiguous()


@pytest.mark.parametrize("tokens", [1152, 1543, 1728, 2048])
def test_public_fused_moe_bf16_g1u1_selected_large_m(tokens):
    hidden, w1, w2, weights, ids = _make_case(tokens)

    actual = fused_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
    )
    expected = torch_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
    )
    assert checkAllclose(actual, expected, rtol=_RTOL, atol=_ATOL) == 0


def test_public_fused_moe_bf16_g1u1_selected_large_m_output_buffer():
    hidden, w1, w2, weights, ids = _make_case(1152)
    expected = torch_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
    )
    output = torch.empty_like(expected)
    actual = fused_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        output=output,
    )
    assert actual is output
    assert checkAllclose(output, expected, rtol=_RTOL, atol=_ATOL) == 0
