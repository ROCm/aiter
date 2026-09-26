# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.moe.moe_situ_epilogue import (
    moe_situ_epilogue,
)


def _gfx950_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return get_gfx_runtime() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _gfx950_available(),
    reason="Merged MoE SiTU epilogue requires gfx950",
)


def _reference(
    merged_projection: torch.Tensor,
    shared_intermediate_size: int,
    num_experts: int,
):
    shared_gate_up = 2 * shared_intermediate_size
    shared = merged_projection[:, :shared_gate_up].to(torch.bfloat16).float()
    gate = shared[:, :shared_intermediate_size]
    up = shared[:, shared_intermediate_size:]
    shared = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    ).to(torch.bfloat16)
    router = merged_projection[:, shared_gate_up : shared_gate_up + num_experts]
    routed = merged_projection[:, shared_gate_up + num_experts :].to(torch.bfloat16)
    return shared, router.contiguous(), routed.contiguous()


@pytest.mark.parametrize(
    "m,shared_intermediate_size,num_experts,routed_latent_size,tile",
    [
        (1, 63, 37, 129, 128),
        (32, 129, 257, 513, 256),
        (512, 768, 896, 3584, 512),
        (1024, 768, 896, 3584, 512),
        (1536, 768, 896, 3584, 512),
        (2048, 768, 896, 3584, 512),
    ],
)
def test_situ_epilogue_matches_reference(
    m: int,
    shared_intermediate_size: int,
    num_experts: int,
    routed_latent_size: int,
    tile: int,
):
    generator = torch.Generator(device="cuda").manual_seed(20261006 + m)
    projection_size = 2 * shared_intermediate_size + num_experts + routed_latent_size
    merged_projection = torch.randn(
        (m, projection_size),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    actual = moe_situ_epilogue(
        merged_projection,
        shared_intermediate_size=shared_intermediate_size,
        num_experts=num_experts,
        routed_latent_size=routed_latent_size,
        tile=tile,
    )
    expected = _reference(
        merged_projection,
        shared_intermediate_size,
        num_experts,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0], expected[0], rtol=0.01, atol=0.005)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_situ_epilogue_graph_reuses_outputs():
    m = 32
    shared_intermediate_size = 129
    num_experts = 257
    routed_latent_size = 513
    projection_size = 2 * shared_intermediate_size + num_experts + routed_latent_size
    merged_projection = torch.randn(
        (m, projection_size), dtype=torch.float32, device="cuda"
    )
    outputs = (
        torch.empty((m, shared_intermediate_size), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, num_experts), dtype=torch.float32, device="cuda"),
        torch.empty((m, routed_latent_size), dtype=torch.bfloat16, device="cuda"),
    )

    def run():
        return moe_situ_epilogue(
            merged_projection,
            shared_intermediate_size=shared_intermediate_size,
            num_experts=num_experts,
            routed_latent_size=routed_latent_size,
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
    expected = _reference(
        merged_projection,
        shared_intermediate_size,
        num_experts,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.01, atol=0.005)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
