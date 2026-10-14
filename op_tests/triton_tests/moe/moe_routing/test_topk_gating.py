# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import topk as topk_ops


def _is_gfx1250() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return get_gfx() == "gfx1250"


pytestmark = pytest.mark.skipif(not _is_gfx1250(), reason="requires gfx1250")


def _torch_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    need_renorm: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the legacy wave32 tie priority with ordinary torch operations."""
    rows, experts = logits.shape
    scores = torch.sigmoid(logits.float())
    choice = scores + bias.float()
    choice = torch.where(torch.isnan(choice), float("-inf"), choice)

    cols = torch.arange(experts, device=logits.device, dtype=torch.int64)
    lane = (cols // 4) % 32
    local_pos = (cols // 128) * 4 + cols % 4
    low2 = (lane & 3) ^ (((lane & 4) != 0).to(torch.int64) * 3)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    lane_rank = ((lane ^ 0x38) & 0x38) | bit2 | low2
    tie_rank = lane_rank * 32 + local_pos
    tie_rank = tie_rank.expand(rows, -1)

    selected_ids = []
    selected_weights = []
    for _ in range(topk):
        max_choice = choice.max(dim=1, keepdim=True).values
        ranks = torch.where(
            choice == max_choice,
            tie_rank,
            torch.iinfo(torch.int64).max,
        )
        ids = ranks.argmin(dim=1)
        selected_ids.append(ids)
        selected_weights.append(scores.gather(1, ids[:, None]).squeeze(1))
        choice.scatter_(1, ids[:, None], float("-inf"))

    ids = torch.stack(selected_ids, dim=1).to(torch.int32)
    weights = torch.stack(selected_weights, dim=1)
    if need_renorm:
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights *= torch.where(weight_sum > 0.0, scale / weight_sum, 0.0)
    else:
        weights *= scale
    return weights, ids


@pytest.mark.parametrize(
    "experts,topk,rows,case,need_renorm,scale",
    [
        (512, 4, 1, "random", True, 1.0),
        (768, 8, 8, "ties", True, 1.0),
        (768, 8, 2, "small_scores", True, 1.0),
        (896, 16, 32, "random", False, 2.5),
        (900, 8, 8, "nan", True, 1.0),
        (1024, 16, 128, "extreme", False, 0.75),
    ],
)
def test_exact_sigmoid_topk_matches_torch(
    experts, topk, rows, case, need_renorm, scale
):
    torch.manual_seed(experts + topk + rows)
    backing = torch.randn(
        (rows, experts + 17),
        dtype=torch.bfloat16,
        device="cuda",
    )
    logits = backing[:, 7 : 7 + experts]
    bias = (torch.randn(experts, device="cuda") * 0.1).to(torch.bfloat16)

    if case == "ties":
        logits.zero_()
        bias.zero_()
    elif case == "small_scores":
        logits[0].fill_(-100)
        logits[1].fill_(-50)
        bias.zero_()
    elif case == "nan":
        logits[:, ::113] = float("nan")
        bias[::127] = float("nan")
    elif case == "extreme":
        logits[:, 0::4] = 100
        logits[:, 1::4] = -100
        logits[:, 2::4] = 0
        logits[:, 3::4] = torch.linspace(
            -20,
            20,
            experts // 4,
            device="cuda",
        )

    expected_weights, expected_ids = _torch_reference(
        logits,
        bias,
        topk,
        need_renorm,
        scale,
    )
    weights = torch.empty_strided(
        (rows, topk),
        (topk + 3, 1),
        dtype=torch.float32,
        device="cuda",
    )
    ids = torch.empty_strided(
        (rows, topk),
        (topk + 3, 1),
        dtype=torch.int32,
        device="cuda",
    )

    assert topk_ops._can_use_gfx1250_exact_sigmoid_topk(
        logits,
        bias,
        weights,
        ids,
    )
    topk_ops.biased_grouped_topk(
        logits,
        bias,
        weights,
        ids,
        1,
        1,
        need_renorm,
        scale,
    )
    hip_weights = torch.empty_like(weights)
    hip_ids = torch.empty_like(ids)
    topk_ops.biased_grouped_topk_hip(
        logits,
        bias,
        hip_weights,
        hip_ids,
        1,
        1,
        need_renorm,
        scale,
    )

    assert torch.equal(ids, expected_ids)
    assert torch.equal(ids, hip_ids)
    torch.testing.assert_close(
        weights,
        expected_weights,
        atol=2e-5,
        rtol=2e-5,
        equal_nan=True,
    )
    torch.testing.assert_close(
        weights,
        hip_weights,
        atol=2e-5,
        rtol=2e-5,
        equal_nan=True,
    )
