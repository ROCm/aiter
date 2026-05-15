# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter import dtypes
from aiter.fused_moe import (
    _add_valid_expert_bias,
    moe_sorting,
)
from aiter.test_common import checkAllclose


def test_moe_sorting_return_local_topk_ids():
    topk_ids = torch.tensor(
        [[0, 1, 5], [2, 4, 3], [5, 0, 2]],
        dtype=dtypes.i32,
        device="cuda",
    )
    topk_weights = torch.ones(topk_ids.shape, dtype=dtypes.fp32, device="cuda")
    expert_mask = torch.tensor([0, 1, 1, 0, 1, 0], dtype=dtypes.i32, device="cuda")
    expected_local_topk_ids = torch.tensor(
        [[-1, 0, -1], [1, 2, -1], [-1, -1, 1]],
        dtype=dtypes.i32,
        device="cuda",
    )

    for dispatch_policy in [0, 1, 2]:
        *_, local_topk_ids = moe_sorting(
            topk_ids,
            topk_weights,
            expert_mask.numel(),
            model_dim=16,
            moebuf_dtype=dtypes.bf16,
            block_size=8,
            expert_mask=expert_mask,
            dispatch_policy=dispatch_policy,
            return_local_topk_ids=True,
        )
        checkAllclose(
            expected_local_topk_ids,
            local_topk_ids,
            atol=0,
            msg=f"local_topk_ids dispatch_policy={dispatch_policy}",
        )

    *_, local_topk_ids_without_mask = moe_sorting(
        topk_ids,
        topk_weights,
        expert_mask.numel(),
        model_dim=16,
        moebuf_dtype=dtypes.bf16,
        block_size=8,
        return_local_topk_ids=True,
    )
    checkAllclose(
        topk_ids,
        local_topk_ids_without_mask,
        atol=0,
        msg="local_topk_ids without expert_mask",
    )


def test_invalid_expert_bias_is_masked():
    valid_out = torch.zeros((3, 4), dtype=dtypes.bf16, device="cuda")
    expert_ids = torch.tensor([-1, 0, 2], dtype=dtypes.i32, device="cuda")
    bias = torch.arange(8, dtype=dtypes.fp32, device="cuda").view(2, 4)

    actual = _add_valid_expert_bias(valid_out, expert_ids, bias)
    expected = torch.zeros_like(actual)
    expected[1] = bias[0].to(actual.dtype)
    checkAllclose(expected, actual, atol=0, msg="invalid expert bias mask")
