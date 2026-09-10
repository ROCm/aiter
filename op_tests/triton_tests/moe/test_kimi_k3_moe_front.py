# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.kimi_k3_moe_front import (
    kimi_k3_moe_front_bf16_epilogue,
    kimi_k3_moe_front_large_m_bf16,
    merge_kimi_k3_moe_front_weights,
)

SHARED_GATE_UP = 1536
SHARED_INTERMEDIATE = 768
NUM_EXPERTS = 896
ROUTED_LATENT = 3584
HIDDEN = 7168
FRONT = SHARED_GATE_UP + NUM_EXPERTS + ROUTED_LATENT


def _gfx950_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return get_gfx_runtime() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _gfx950_available(),
    reason="Kimi-K3 large-M front requires gfx950",
)


def _reference(front: torch.Tensor):
    shared = front[:, :SHARED_GATE_UP].to(torch.bfloat16).float()
    gate = shared[:, :SHARED_INTERMEDIATE]
    up = shared[:, SHARED_INTERMEDIATE:]
    shared = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    ).to(torch.bfloat16)
    router = front[:, SHARED_GATE_UP : SHARED_GATE_UP + NUM_EXPERTS]
    routed = front[:, SHARED_GATE_UP + NUM_EXPERTS :].to(torch.bfloat16)
    return shared, router.contiguous(), routed.contiguous()


@pytest.mark.parametrize("m", [1, 32, 512])
def test_bf16_epilogue_matches_reference(m: int):
    generator = torch.Generator(device="cuda").manual_seed(20261006 + m)
    front = torch.randn(
        (m, FRONT),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    actual = kimi_k3_moe_front_bf16_epilogue(front)
    expected = _reference(front)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0], expected[0], rtol=0.01, atol=0.005)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_bf16_epilogue_graph_reuses_outputs():
    m = 32
    front = torch.randn((m, FRONT), dtype=torch.float32, device="cuda")
    outputs = (
        torch.empty((m, SHARED_INTERMEDIATE), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, NUM_EXPERTS), dtype=torch.float32, device="cuda"),
        torch.empty((m, ROUTED_LATENT), dtype=torch.bfloat16, device="cuda"),
    )

    def run():
        return kimi_k3_moe_front_bf16_epilogue(
            front,
            shared_out=outputs[0],
            router_out=outputs[1],
            routed_out=outputs[2],
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    graph.replay()
    torch.cuda.synchronize()

    assert all(a is b for a, b in zip(actual, outputs, strict=True))
    expected = _reference(front)
    torch.testing.assert_close(actual[0], expected[0], rtol=0.01, atol=0.005)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_merge_weights_preserves_native_row_order():
    gate = (
        torch.arange(
            SHARED_INTERMEDIATE * HIDDEN,
            dtype=torch.float32,
            device="cuda",
        )
        .to(torch.bfloat16)
        .view(SHARED_INTERMEDIATE, HIDDEN)
    )
    up = gate + 1
    shared = torch.cat((gate, up), dim=0)
    router = torch.zeros(
        (NUM_EXPERTS, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
    )
    routed = torch.zeros(
        (ROUTED_LATENT, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
    )

    merged = merge_kimi_k3_moe_front_weights(shared, router, routed)

    torch.testing.assert_close(merged[:SHARED_GATE_UP], shared)
    torch.testing.assert_close(
        merged[SHARED_GATE_UP : SHARED_GATE_UP + NUM_EXPERTS],
        router,
    )
    torch.testing.assert_close(
        merged[SHARED_GATE_UP + NUM_EXPERTS :],
        routed,
    )


# 1, 7, 14 and 16 are the single-sequence and small-batch decode shapes. 32
# through 192 are the MTP decode bucket: with num_speculative_tokens=3 a
# pure-decode step submits num_seqs * 4 tokens, so a server serving 8 to 48
# concurrent sequences lands here rather than in the 1..16 range.
@pytest.mark.parametrize("m", [1, 7, 14, 16, 32, 48, 64, 80, 96, 112, 128, 192])
def test_decode_full_front_matches_reference(m: int):
    generator = torch.Generator(device="cuda").manual_seed(20260908)
    hidden_states = torch.randn(
        (m, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    shared_weight = (
        torch.randn(
            (SHARED_GATE_UP, HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    router_weight = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    routed_weight = (
        torch.randn(
            (ROUTED_LATENT, HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    merged_weight = merge_kimi_k3_moe_front_weights(
        shared_weight,
        router_weight,
        routed_weight,
    )

    actual = kimi_k3_moe_front_large_m_bf16(
        hidden_states,
        merged_weight,
    )
    front_reference = torch.mm(
        hidden_states,
        merged_weight.t(),
        out_dtype=torch.float32,
    )
    expected = _reference(front_reference)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0], expected[0], rtol=0.01, atol=0.005)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.01, atol=0.01)
    torch.testing.assert_close(actual[2], expected[2], rtol=0.01, atol=0.005)
