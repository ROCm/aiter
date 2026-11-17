# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.moe.moe_routing import topk_align

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is None
    or get_gfx() != "gfx1201",
    reason="gfx1201 selected large-M BF16 G1U1 route alignment coverage",
)


def _max_tiles(route_count: int, num_experts: int, block_m: int) -> int:
    if route_count <= num_experts:
        return route_count
    return (
        (max(route_count - num_experts + 1, 0) + block_m - 1) // block_m
        + num_experts
        - 1
    )


def _reference_offsets(hist: torch.Tensor, block_m: int):
    raw_offsets = torch.empty(hist.numel() + 1, dtype=torch.int32, device=hist.device)
    raw_offsets[0] = 0
    raw_offsets[1:] = torch.cumsum(hist, dim=0)

    tile_counts = torch.div(hist + block_m - 1, block_m, rounding_mode="floor")
    tile_offsets = torch.empty(hist.numel() + 1, dtype=torch.int32, device=hist.device)
    tile_offsets[0] = 0
    tile_offsets[1:] = torch.cumsum(tile_counts, dim=0)
    return raw_offsets, tile_offsets


def _reference_block_pid_map(
    hist: torch.Tensor,
    route_count: int,
    num_experts: int,
    block_m: int,
    tile_offsets: torch.Tensor,
):
    max_blocks = _max_tiles(route_count, num_experts, block_m)
    expected = torch.full((max_blocks,), -1, dtype=torch.int32, device=hist.device)
    for expert in range(num_experts):
        count = int(hist[expert].item())
        for block_id in range((count + block_m - 1) // block_m):
            tile = int(tile_offsets[expert].item()) + block_id
            expected[tile] = (block_id << 16) | expert
    return expected


@pytest.fixture
def isolated_scratch(monkeypatch):
    scratch = {}
    monkeypatch.setattr(topk_align, "_TOPK_ALIGN_SCRATCH", scratch)
    yield scratch
    torch.cuda.synchronize()


def _make_topk_case(tokens: int, *, skewed: bool = False):
    torch.manual_seed(2026 + tokens)
    topk_ids = torch.randint(0, 256, (tokens, 8), dtype=torch.int32, device="cuda")
    if skewed:
        topk_ids = topk_ids.remainder(16).contiguous()
    topk_weights = torch.rand((tokens, 8), dtype=torch.float32, device="cuda")
    return topk_ids, topk_weights


def _assert_align_result_matches_reference(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    result,
    *,
    num_experts: int,
    block_m: int,
):
    routing_data, gather, inverse, sorted_weights = result
    torch.cuda.synchronize()

    flat_ids = topk_ids.flatten()
    flat_weights = topk_weights.flatten()
    route_count = flat_ids.numel()
    hist = torch.bincount(flat_ids.long(), minlength=num_experts).to(torch.int32)
    raw_offsets, tile_offsets = _reference_offsets(hist, block_m)
    block_pid_map = _reference_block_pid_map(
        hist, route_count, num_experts, block_m, tile_offsets
    )

    torch.testing.assert_close(routing_data.expt_data.hist, hist, rtol=0, atol=0)
    torch.testing.assert_close(
        routing_data.expt_data.token_offs_raw, raw_offsets, rtol=0, atol=0
    )
    torch.testing.assert_close(
        routing_data.expt_data.token_offs_pad, tile_offsets, rtol=0, atol=0
    )
    torch.testing.assert_close(
        routing_data.expt_data.block_pid_map, block_pid_map, rtol=0, atol=0
    )

    route_arange = torch.arange(route_count, device=topk_ids.device, dtype=torch.int32)
    torch.testing.assert_close(torch.sort(gather).values, route_arange, rtol=0, atol=0)
    torch.testing.assert_close(gather[inverse.long()], route_arange, rtol=0, atol=0)
    torch.testing.assert_close(
        sorted_weights[inverse.long()],
        flat_weights,
        rtol=0,
        atol=0,
    )

    sorted_experts = flat_ids[gather.long()]
    for expert in range(num_experts):
        start = int(raw_offsets[expert].item())
        end = int(raw_offsets[expert + 1].item())
        if start == end:
            continue
        assert torch.all(sorted_experts[start:end] == expert)

    assert routing_data.block_m == block_m
    assert routing_data.n_expts_tot == num_experts
    assert routing_data.n_expts_act == topk_ids.shape[1]
    assert routing_data.gate_scal.data_ptr() == sorted_weights.data_ptr()


def _assert_align_matches_reference(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_m: int,
):
    result = topk_align.align_topk_ids(
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        block_m=block_m,
    )
    _assert_align_result_matches_reference(
        topk_ids,
        topk_weights,
        result,
        num_experts=num_experts,
        block_m=block_m,
    )


def test_align_topk_ids_metadata_with_empty_and_uneven_experts():
    topk_ids = torch.tensor(
        [
            [0, 3, 3],
            [3, 7, 7],
            [7, 7, 7],
            [5, 5, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.arange(
        topk_ids.numel(), dtype=torch.float32, device="cuda"
    ).reshape_as(topk_ids)

    _assert_align_matches_reference(
        topk_ids,
        topk_weights,
        num_experts=8,
        block_m=4,
    )


def test_align_topk_ids_uses_exclusive_scratch_per_stream(isolated_scratch):
    topk_ids = torch.tensor(
        [[0, 1, 1, 7], [7, 7, 3, 3]], dtype=torch.int32, device="cuda"
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    default_stream = torch.cuda.current_stream()

    first, gather_first, _inverse_first, _weights_first = topk_align.align_topk_ids(
        topk_ids,
        topk_weights,
        num_experts=8,
        block_m=4,
    )

    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream):
        other_stream.wait_stream(default_stream)
        second, gather_second, _inverse_second, _weights_second = (
            topk_align.align_topk_ids(
                topk_ids,
                topk_weights,
                num_experts=8,
                block_m=4,
            )
        )
    other_stream.synchronize()

    assert first.expt_data.hist.data_ptr() != second.expt_data.hist.data_ptr()
    assert gather_first.data_ptr() != gather_second.data_ptr()


def test_scratch_cache_bounded_across_large_m_policy(isolated_scratch):
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for tokens in range(1024, 4097):
            topk_align._scratch(device, tokens * 8, 256, 32)
    stream.synchronize()

    keys = [
        key
        for key in isolated_scratch
        if key[0] == device and key[1] == stream.cuda_stream and key[3:] == (256, 32)
    ]
    assert len(keys) == 3
    assert {key[2] for key in keys} == {8192, 16384, 32768}

    expected_blocks = {8192: 504, 16384: 760, 32768: 1272}
    for key in keys:
        scratch = isolated_scratch[key]
        capacity = key[2]
        assert scratch["block_pid_map"].numel() == expected_blocks[capacity]
        for name in ("sorted_routes", "sorted_weights", "inverse"):
            assert scratch[name].numel() == capacity

    bytes_retained = sum(
        tensor.numel() * tensor.element_size()
        for scratch in isolated_scratch.values()
        for tensor in scratch.values()
    )
    assert bytes_retained == 710_596


def test_scratch_bucket_reuse_and_exact_output_shapes(isolated_scratch):
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        scratch_a = topk_align._scratch(device, 1152 * 8, 256, 32)
        scratch_b = topk_align._scratch(device, 1544 * 8, 256, 32)
        scratch_c = topk_align._scratch(device, 2049 * 8, 256, 32)
        assert scratch_a is scratch_b
        assert scratch_a is not scratch_c
        for name in ("block_pid_map", "sorted_routes", "sorted_weights", "inverse"):
            assert scratch_a[name].data_ptr() == scratch_b[name].data_ptr()
            assert scratch_a[name].data_ptr() != scratch_c[name].data_ptr()

        ids, weights = _make_topk_case(1152)
        routing_data, gather, inverse, sorted_weights = topk_align.align_topk_ids(
            ids,
            weights,
            num_experts=256,
            block_m=32,
        )
    stream.synchronize()

    assert gather.numel() == 1152 * 8
    assert inverse.numel() == 1152 * 8
    assert sorted_weights.numel() == 1152 * 8
    assert routing_data.expt_data.block_pid_map.numel() == topk_align._max_tiles(
        1152 * 8, 256, 32
    )
    assert routing_data.gate_scal.data_ptr() == sorted_weights.data_ptr()


@pytest.mark.parametrize(
    "sequence",
    [
        (1152, 1544, 1152),
        (1024, 1025, 1024),
        (2048, 2049, 2048),
        (3072, 4096, 3072),
    ],
)
def test_align_topk_ids_alternating_m(sequence, isolated_scratch):
    for index, tokens in enumerate(sequence):
        ids, weights = _make_topk_case(tokens, skewed=index == 1)
        _assert_align_matches_reference(
            ids,
            weights,
            num_experts=256,
            block_m=32,
        )


@pytest.mark.parametrize("m_a,m_b", [(1152, 1544), (2048, 2049)])
def test_align_topk_ids_graph_replay_buckets(m_a, m_b, isolated_scratch):
    if not hasattr(torch.cuda, "CUDAGraph"):
        pytest.skip("CUDA/HIP graphs are not available in this torch build")

    stream = torch.cuda.Stream()
    ids_a, weights_a = _make_topk_case(m_a)
    ids_b, weights_b = _make_topk_case(m_b, skewed=True)

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        topk_align.align_topk_ids(ids_a, weights_a, num_experts=256, block_m=32)
        topk_align.align_topk_ids(ids_b, weights_b, num_experts=256, block_m=32)
    stream.synchronize()

    graph_a = graph_b = result_a = result_b = None
    try:
        graph_a = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_a, stream=stream):
            result_a = topk_align.align_topk_ids(
                ids_a, weights_a, num_experts=256, block_m=32
            )

        graph_b = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_b, stream=stream):
            result_b = topk_align.align_topk_ids(
                ids_b, weights_b, num_experts=256, block_m=32
            )

        for graph, ids, weights, result in (
            (graph_a, ids_a, weights_a, result_a),
            (graph_b, ids_b, weights_b, result_b),
            (graph_a, ids_a, weights_a, result_a),
        ):
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            _assert_align_result_matches_reference(
                ids,
                weights,
                result,
                num_experts=256,
                block_m=32,
            )
    finally:
        torch.cuda.synchronize()
        del graph_a, graph_b, result_a, result_b
