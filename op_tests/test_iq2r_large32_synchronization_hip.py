# SPDX-License-Identifier: MIT
"""Regressions for shared-codebook reuse and partial-M loads in large32."""

import pytest
import torch

from aiter.ops.iq2r import (
    iq2r_encode_device,
    iq2r_route_gather_quant_out,
    iq2r_route_sort_tasks_out,
    iq2r_task_capacity,
    iq2r_task_gemm_out,
)
from aiter.ops.iq2r_encoder import iq2r_initial_codebook
from aiter.ops.iq2r_format import IQ2R_CODEBOOK_BYTES, IQ2RMetadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="requires gfx950",
)


@pytest.fixture(scope="module")
def distinct_expert_weights():
    generator = torch.Generator(device="cuda").manual_seed(0xE063)
    weight = torch.randn((6144, 256), device="cuda", generator=generator) * 0.03
    data, auxiliary = iq2r_encode_device(
        weight,
        torch.ones(256, device="cuda"),
        iq2r_initial_codebook("cuda"),
        exponent_radius=8,
    )
    data = data.expand(257, -1).contiguous()
    auxiliary = auxiliary.expand(257, -1).contiguous()
    # Repeating an identical codebook across experts hides overwrite races.
    # Vary finite codebook values while preserving entry zero and the IQ2R ABI.
    codebook = auxiliary[0, :IQ2R_CODEBOOK_BYTES].view(torch.float8_e4m3fn).float()
    factors = torch.linspace(0.6, 1.3, 257, device="cuda")[:, None]
    values = (codebook[None, :] * factors).to(torch.float8_e4m3fn)
    auxiliary[:, :IQ2R_CODEBOOK_BYTES].copy_(values.view(torch.uint8))
    assert torch.isfinite(values.float()).all()
    return data, auxiliary


@pytest.mark.parametrize("routing", ["spread", "hot"])
def test_large32_repeated_graph_is_stable(
    distinct_expert_weights, routing, monkeypatch
):
    data, auxiliary = distinct_expert_weights
    metadata = IQ2RMetadata(logical_n=6144, logical_k=256)
    tokens, topk, routes = 256, 9, 2304
    generator = torch.Generator(device="cuda").manual_seed(0x256E063)
    if routing == "spread":
        ids = torch.stack(
            [
                torch.randperm(256, device="cuda", generator=generator)[:8]
                for _ in range(tokens)
            ]
        ).int()
    else:
        ids = torch.tensor(
            [0, 17, 65, 127, 160, 223, 254, 255], dtype=torch.int32, device="cuda"
        ).repeat(tokens, 1)
    ids = torch.cat(
        (ids, torch.full((tokens, 1), 256, dtype=torch.int32, device="cuda")), 1
    )
    sorted_ids, gather, scatter = [
        torch.empty(routes, dtype=torch.int32, device="cuda") for _ in range(3)
    ]
    tasks = torch.empty(
        (iq2r_task_capacity(routes, 257, 32), 3), dtype=torch.int32, device="cuda"
    )
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    iq2r_route_sort_tasks_out(
        ids.flatten(),
        sorted_ids,
        gather,
        scatter,
        tasks,
        count,
        expert_count=257,
        task_rows=32,
    )
    hidden = torch.randn((tokens, 256), device="cuda", generator=generator).bfloat16()
    fp8 = torch.empty((routes, 256), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty((routes, 8), dtype=torch.uint8, device="cuda")
    output = torch.empty((routes, 6144), dtype=torch.bfloat16, device="cuda")
    iq2r_route_gather_quant_out(hidden, gather, fp8, scales, topk=topk)

    def run():
        iq2r_task_gemm_out(
            fp8, scales, data, auxiliary, tasks, count, metadata, output, tile_n=128
        )

    monkeypatch.setenv("IQ2R_GEMM_DOWN_FAMILY", "3x2m2")
    run()
    reference = output.clone()
    monkeypatch.setenv("IQ2R_GEMM_DOWN_FAMILY", "large32")
    monkeypatch.setenv("IQ2R_GEMM_DOWN_GRID_MULTIPLIER", "4")
    run()
    # Two-wave K reduction and serial-K large32 may differ by one BF16 ULP.
    torch.testing.assert_close(output, reference, rtol=1 / 128, atol=1e-6)
    expected = output.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(128):
        graph.replay()
        assert torch.equal(
            output, expected
        ), "fixed inputs changed across large32 graph replays"
