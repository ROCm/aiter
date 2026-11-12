# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: stale memory in a2 slots that stage1 skips must not set the
dynamic per-tensor fp8 a2 scale."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="Needs a ROCm GPU",
)

TOKEN, MODEL_DIM, INTER_DIM, E, TOPK = 32, 1024, 512, 8, 2


def _leave_stale_memory(value):
    # Freed caching-allocator blocks keep their contents, so the a2 buffer
    # allocated next by fused_moe starts out holding `value`.
    stale = [
        torch.full((TOKEN, TOPK, INTER_DIM), value, dtype=dtypes.bf16, device="cuda")
        for _ in range(64)
    ]
    del stale


@pytest.mark.parametrize("stale_value", [1e30, float("inf")])
@pytest.mark.parametrize("unrouted", ["padding_rows", "ep_non_local_experts"])
def test_per_tensor_a2_ignores_unrouted_slots(unrouted, stale_value):
    torch.manual_seed(0)
    hidden = torch.randn((TOKEN, MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn((E, INTER_DIM * 2, MODEL_DIM), device="cuda") / 10
    w2 = torch.randn((E, MODEL_DIM, INTER_DIM), device="cuda") / 10
    score = torch.randn((TOKEN, E), device="cuda")
    topk_weights, topk_ids = torch.softmax(score, dim=-1).topk(TOPK, dim=-1)
    topk_ids = topk_ids.to(dtypes.i32)

    expert_mask = None
    if unrouted == "padding_rows":
        topk_ids[1::4] = -1
    else:
        expert_mask = torch.tensor([1, 0] * (E // 2), dtype=dtypes.i32, device="cuda")
        w1 = w1[expert_mask.bool()]
        w2 = w2[expert_mask.bool()]

    w1_qt, w1_scale = aiter.pertoken_quant(
        w1.view(w1.shape[0], -1), quant_dtype=dtypes.fp8
    )
    w2_qt, w2_scale = aiter.pertoken_quant(
        w2.view(w2.shape[0], -1), quant_dtype=dtypes.fp8
    )
    w1_qt = w1_qt.view(w1.shape)
    w2_qt = w2_qt.view(w2.shape)

    ref = torch_moe(
        hidden,
        w1_qt.float() * w1_scale.view(-1, 1, 1),
        w2_qt.float() * w2_scale.view(-1, 1, 1),
        topk_weights,
        topk_ids,
        expert_mask=expert_mask,
    )

    w1_qt = shuffle_weight(w1_qt, layout=(16, 16))
    w2_qt = shuffle_weight(w2_qt, layout=(16, 16))

    _leave_stale_memory(stale_value)
    out = fused_moe(
        hidden,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        expert_mask=expert_mask,
        quant_type=aiter.QuantType.per_Tensor,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    assert torch.isfinite(out).all()
    out, ref = out.double(), ref.double()
    diff = 1 - 2 * (out * ref).sum() / (out * out + ref * ref).sum()
    assert diff < 1e-2, f"{diff=}, mean|out|={out.abs().mean():.6f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
