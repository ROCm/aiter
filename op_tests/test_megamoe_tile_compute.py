# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch

from aiter.ops.flydsl.kernels.megamoe_tile import (
    a4w4_dense_reference,
    prepare_local_a4w4_weights,
    run_local_ep_a4w4,
    run_local_ep_a4w4_silu,
)


def _logits_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    x, y = lhs.double(), rhs.double()
    return float(1.0 - (2.0 * (x * y).sum() / (x.square().sum() + y.square().sum())))


def test_local_a4w4_silu_v2_layout_baseline():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("ROCm GPU required")
    torch.manual_seed(11)
    dev = torch.device("cuda", 0)
    m, h, inter, experts, topk = 4, 1024, 256, 8, 2
    x = (torch.randn(m, h, device=dev) * 0.1).to(torch.bfloat16)
    w1 = (torch.randn(experts, 2 * inter, h, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(experts, h, inter, device=dev) * 0.05).to(torch.bfloat16)
    ids = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.int32, device=dev)
    weights = torch.tensor([[0.6, 0.4]] * m, dtype=torch.float32, device=dev)

    prepared = prepare_local_a4w4_weights(w1, w2)
    mask = torch.ones(experts, dtype=torch.int32, device=dev)
    out = run_local_ep_a4w4_silu(
        x, ids, weights, prepared, global_experts=experts, local_expert_mask=mask
    )
    ref = a4w4_dense_reference(x, ids, weights, prepared)
    assert torch.isfinite(out).all()
    assert _logits_diff(out.float(), ref) < 1e-2


def test_local_a4w4_all_activation_variants():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("ROCm GPU required")
    torch.manual_seed(17)
    dev = torch.device("cuda", 0)
    m, h, inter, experts, topk = 2, 1024, 256, 4, 2
    x = (torch.randn(m, h, device=dev) * 0.1).to(torch.bfloat16)
    w1 = (torch.randn(experts, 2 * inter, h, device=dev) * 0.04).to(torch.bfloat16)
    w2 = (torch.randn(experts, h, inter, device=dev) * 0.04).to(torch.bfloat16)
    ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32, device=dev)
    route_weights = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32, device=dev
    )
    prepared = prepare_local_a4w4_weights(w1, w2)
    mask = torch.ones(experts, dtype=torch.int32, device=dev)

    for activation in ("silu", "swiglu", "situv2"):
        out = run_local_ep_a4w4(
            x,
            ids,
            route_weights,
            prepared,
            global_experts=experts,
            local_expert_mask=mask,
            activation=activation,
        )
        ref = a4w4_dense_reference(
            x, ids, route_weights, prepared, activation=activation
        )
        assert torch.isfinite(out).all()
        diff = _logits_diff(out.float(), ref)
        print(f"{activation}_logits_diff={diff:.6e}")
        assert diff < 2e-2, activation
