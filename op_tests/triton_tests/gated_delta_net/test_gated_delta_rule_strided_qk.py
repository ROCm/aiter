# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk


@pytest.mark.parametrize("tokens", [64, 128])
def test_chunk_opt_vk_l2norm_accepts_strided_qk(tokens: int):
    torch.manual_seed(tokens)
    batch, heads, head_dim = 2, 2, 128
    shape = (batch, tokens, heads, head_dim)

    q = torch.randn(
        tokens,
        batch,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).transpose(0, 1)
    k = torch.randn(
        tokens,
        batch,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).transpose(0, 1)
    assert q.shape == k.shape == shape
    assert q.stride(-1) == k.stride(-1) == 1
    assert not q.is_contiguous()
    assert not k.is_contiguous()

    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.rand(
        batch, tokens, heads, dtype=torch.bfloat16, device="cuda"
    ).sigmoid()
    g = F.logsigmoid(
        torch.rand(batch, tokens, heads, dtype=torch.float32, device="cuda")
    )
    initial_state = (
        torch.randn(
            batch,
            heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device="cuda",
        )
        .transpose(-1, -2)
        .contiguous()
    )

    def run(q_input: torch.Tensor, k_input: torch.Tensor):
        return chunk_gated_delta_rule_opt_vk(
            q=q_input,
            k=k_input,
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            initial_state=initial_state.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_dtype=torch.float32,
            snapshot_dtype=torch.float32,
        )

    actual, actual_state = run(q, k)
    expected, expected_state = run(q.contiguous(), k.contiguous())

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
