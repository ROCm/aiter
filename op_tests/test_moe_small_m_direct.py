# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.flydsl.moe_common import GateMode
from aiter.test_common import checkAllclose


pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx1201",
    reason="gfx1201 small-M direct MoE coverage",
)


def _make_balanced_topk(token: int, topk: int, expert: int):
    rows = torch.arange(token, device="cuda", dtype=torch.int32)[:, None]
    slots = torch.arange(topk, device="cuda", dtype=torch.int32)[None, :]
    topk_ids = (rows * topk + slots) % expert
    topk_weights = torch.rand((token, topk), dtype=torch.float32, device="cuda")
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.contiguous(), topk_weights.contiguous()


@pytest.mark.parametrize("token", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("inter_dim", [128, 256])
@pytest.mark.parametrize(
    "preshuffle_w1,preshuffle_w2",
    [(False, False), (True, False), (True, True)],
)
def test_bf16_g1u1_small_m_direct(token, inter_dim, preshuffle_w1, preshuffle_w2):
    torch.manual_seed(2026 + token + inter_dim)
    model_dim = 2048
    expert = 256
    topk = 8

    hidden_states = torch.randn(
        (token, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w1 = torch.randn(
        (expert, inter_dim * 2, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w2 = torch.randn(
        (expert, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda"
    )
    topk_ids, topk_weights = _make_balanced_topk(token, topk, expert)
    w1_input = shuffle_weight(w1, layout=(16, 16)) if preshuffle_w1 else w1
    w2_input = shuffle_weight(w2, layout=(16, 16)) if preshuffle_w2 else w2

    actual = fused_moe(
        hidden_states,
        w1_input,
        w2_input,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        gate_mode=GateMode.SEPARATED.value,
    )
    expected = torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
    )

    checkAllclose(actual, expected, rtol=1e-2, atol=1e-2)
