# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Smoke tests for ``causal_conv1d_update_fast`` / ``fused_reshape_causal_conv1d_update_fast``.

``causal_conv1d_update_fast`` updates ``conv_state`` in place inside the kernel; the fast
kernels use a different state layout than ``causal_conv1d_update_ref`` in ``test_causal_conv1d``.
"""

import pytest
import torch

from aiter.ops.triton.causal_conv1d_update_fast import (
    causal_conv1d_update_fast,
    fused_reshape_causal_conv1d_update_fast,
)

cuda_ok = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device required"
)


@cuda_ok
@pytest.mark.parametrize("batch,dim,width,seqlen", [(2, 64, 3, 1), (1, 128, 4, 2)])
def test_causal_conv1d_update_fast_smoke(batch, dim, width, seqlen):
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.bfloat16)
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device=device, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16)
    out = causal_conv1d_update_fast(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=None,
    )
    assert out.shape == x.shape
    assert torch.isfinite(out.to(torch.float32)).all()


@cuda_ok
def test_fused_reshape_causal_conv1d_update_fast_smoke():
    device = "cuda"
    torch.manual_seed(0)
    num_k_heads = 2
    num_v_heads = 2
    head_k_dim = 8
    head_v_dim = 8
    head_dim = head_k_dim + head_k_dim + head_v_dim * num_v_heads // num_k_heads
    head_qkvz_dim = head_dim + head_v_dim * num_v_heads // num_k_heads
    qkvz_dim = num_k_heads * head_qkvz_dim
    num_tokens = 4
    num_actual_tokens = 2
    width = 3
    seqlen = 1
    dim = num_k_heads * head_dim

    x = torch.randn(num_tokens, qkvz_dim, seqlen, device=device, dtype=torch.bfloat16)
    ba = torch.randn(num_tokens, 2 * num_v_heads, device=device, dtype=torch.bfloat16)
    z_out = torch.zeros(
        num_tokens, num_v_heads, head_v_dim, device=device, dtype=torch.bfloat16
    )
    core = torch.zeros_like(z_out)
    conv_state = torch.randn(
        num_actual_tokens, dim, width - 1, device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(dim, width, device=device, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16)

    out, b_out, a_out = fused_reshape_causal_conv1d_update_fast(
        x,
        num_actual_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        ba,
        z_out,
        core,
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=None,
    )
    # With 2D packed `x`, the launcher unsqueezes then squeezes `seqlen==1` on output.
    assert out.shape == (num_actual_tokens, dim)
    assert b_out.shape == (num_actual_tokens, num_v_heads)
    assert a_out.shape == (num_actual_tokens, num_v_heads)
    assert torch.isfinite(out.to(torch.float32)).all()
