# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kimi_k3_mla_gate import (
    kimi_k3_mla_gate,
    supports_kimi_k3_mla_gate,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="Kimi-K3 MLA gate specialization requires gfx950",
)


def _inputs(batch: int, seed: int = 17):
    torch.manual_seed(seed)
    hidden = torch.randn((batch, 7168), device="cuda", dtype=torch.bfloat16) * 0.15
    weight = torch.randn((1536, 7168), device="cuda", dtype=torch.bfloat16) * 0.02
    attention = torch.randn((batch, 1536), device="cuda", dtype=torch.bfloat16)
    return hidden.contiguous(), weight.contiguous(), attention.contiguous()


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = actual.float() - expected.float()
    return (
        torch.sqrt(torch.mean(error.square()))
        / torch.sqrt(torch.mean(expected.float().square())).clamp_min(1e-30)
    ).item()


def test_kimi_k3_mla_gate_primary_dispatch_and_accuracy():
    hidden, weight, attention = _inputs(1)
    assert supports_kimi_k3_mla_gate(hidden, weight, attention)
    output = kimi_k3_mla_gate(hidden, weight, attention)
    reference = (
        torch.sigmoid(hidden.float() @ weight.float().T) * attention.float()
    ).to(torch.bfloat16)
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous()
    assert _relative_rmse(output, reference) <= 0.01
    assert (
        torch.nn.functional.cosine_similarity(
            output.float().flatten(),
            reference.float().flatten(),
            dim=0,
        ).item()
        >= 0.999
    )


def test_kimi_k3_mla_gate_reuses_valid_output():
    hidden, weight, attention = _inputs(1)
    output = torch.empty_like(attention)

    actual = kimi_k3_mla_gate(hidden, weight, attention, out=output)
    torch.cuda.synchronize()

    assert actual is output


def test_kimi_k3_mla_gate_supports_attention_output_alias():
    hidden, weight, attention = _inputs(1)
    attention_input = attention.clone()
    reference = (
        torch.sigmoid(hidden.float() @ weight.float().T) * attention_input.float()
    ).to(torch.bfloat16)

    actual = kimi_k3_mla_gate(hidden, weight, attention, out=attention)
    torch.cuda.synchronize()

    assert actual is attention
    assert _relative_rmse(actual, reference) <= 0.01


def test_kimi_k3_mla_gate_graph_replay_uses_changed_inputs():
    hidden, weight, attention = _inputs(1)
    output = torch.empty_like(attention)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kimi_k3_mla_gate(hidden, weight, attention, out=output)

    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    hidden.copy_(torch.randn_like(hidden))
    attention.copy_(torch.randn_like(attention))
    graph.replay()
    torch.cuda.synchronize()
    second = output.clone()

    assert not torch.equal(first, second)
    reference = (
        torch.sigmoid(hidden.float() @ weight.float().T) * attention.float()
    ).to(torch.bfloat16)
    assert _relative_rmse(second, reference) <= 0.01


def test_kimi_k3_mla_gate_support_is_narrow():
    hidden, weight, attention = _inputs(1)
    noncontiguous_attention = torch.empty(
        (1, 3072),
        device="cuda",
        dtype=torch.bfloat16,
    )[:, ::2]

    assert supports_kimi_k3_mla_gate(hidden, weight, attention)
    assert not supports_kimi_k3_mla_gate(hidden.expand(2, -1), weight, attention)
    assert not supports_kimi_k3_mla_gate(hidden, weight, noncontiguous_attention)


def test_kimi_k3_mla_gate_rejects_unsupported_shape():
    hidden, weight, attention = _inputs(2)

    with pytest.raises(NotImplementedError, match="requires contiguous gfx950 BF16"):
        kimi_k3_mla_gate(hidden, weight, attention)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1536,), torch.bfloat16),
        ((1, 1536), torch.float32),
    ],
)
def test_kimi_k3_mla_gate_rejects_invalid_output(shape, dtype):
    hidden, weight, attention = _inputs(1)
    output = torch.empty(shape, device="cuda", dtype=dtype)

    with pytest.raises(ValueError, match="out must be contiguous BF16"):
        kimi_k3_mla_gate(hidden, weight, attention, out=output)
