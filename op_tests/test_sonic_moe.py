# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
from torch.testing import assert_close

from aiter.sonic_moe import ActivationType, KernelBackendMoE, MoE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_sonic_moe_aiter_matches_torch_swiglu():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    token, hidden, intermediate, experts, topk = 128, 128, 64, 8, 2

    moe = MoE(
        num_experts=experts,
        num_experts_per_tok=topk,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(device="cuda", dtype=dtype)
    moe.eval()

    x = 0.02 * torch.randn(token, hidden, device="cuda", dtype=dtype)
    with torch.no_grad():
        actual, aux_actual = moe(x, kernel_backend_moe=KernelBackendMoE.aiter)
        expected, aux_expected = moe(x, kernel_backend_moe=KernelBackendMoE.torch)

    assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)
    assert_close(aux_actual.float(), aux_expected.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_sonic_moe_aiter_inference_returns_no_aux_loss():
    torch.manual_seed(0)
    moe = MoE(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=128,
        intermediate_size=64,
        activation_function="swiglu",
        add_bias=False,
        std=0.02,
    ).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    y, aux_loss = moe(
        x,
        kernel_backend_moe="aiter",
        is_inference_mode=True,
    )

    assert y.shape == x.shape
    assert aux_loss is None
