# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Persistent asm fused_moe: a token's output must not depend on how other tokens are routed."""

import pytest
import torch

from aiter import QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx942", reason="gfx942 fp8 blockscale asm MoE kernels"
)

MODEL_DIM = 2048
E = 128
TOPK = 8


@_SKIP
@pytest.mark.parametrize("inter_dim", [768, 1536])
@pytest.mark.parametrize("token", [64, 256])
def test_fused_moe_blockscale_token_ignores_other_routing(inter_dim, token):
    """inter_dim / 256 of 3 or 6 does not divide a persistent grid of 304 TGs, the
    case where the launch had more TGs than the kernel's total_tgs."""
    torch.manual_seed(0)
    w1 = shuffle_weight(
        (torch.randn(E, 2 * inter_dim, MODEL_DIM, device="cuda") * 0.05).to(dtypes.fp8),
        (16, 16),
    )
    w2 = shuffle_weight(
        (torch.randn(E, MODEL_DIM, inter_dim, device="cuda") * 0.05).to(dtypes.fp8),
        (16, 16),
    )
    w1_scale = torch.rand(E, 2 * inter_dim // 128, MODEL_DIM // 128, device="cuda")
    w2_scale = torch.rand(E, MODEL_DIM // 128, inter_dim // 128, device="cuda")
    x = torch.randn(token, MODEL_DIM, device="cuda").to(dtypes.bf16)
    probs = torch.softmax(torch.randn(token, E, device="cuda"), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, TOPK, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).float()
    topk_ids = topk_ids.to(dtypes.i32)

    def run(ids):
        return fused_moe(
            x,
            w1,
            w2,
            topk_weights,
            ids,
            quant_type=QuantType.per_128x128,
            w1_scale=w1_scale * 0.02 + 0.005,
            w2_scale=w2_scale * 0.02 + 0.005,
        ).float()

    base = run(topk_ids)
    for moved, first in ((2, 40), (8, 0), (32, 0), (32, 40)):
        ids = topk_ids.clone()
        ids[-moved:] = torch.arange(first, first + TOPK, dtype=dtypes.i32)
        kept = token - moved
        err = (run(ids)[:kept] - base[:kept]).abs().amax() / base[:kept].abs().amax()
        assert err < 0.05, (
            f"rerouting the last {moved} tokens to experts {first}..{first + TOPK - 1} "
            f"moved the others by {err:.2f} of the output range"
        )
