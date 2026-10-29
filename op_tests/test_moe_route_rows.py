# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Ordinary TP MoE route rows: integer contracts and public-wrapper timing."""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_topids_to_rows
from aiter.test_common import benchmark, checkAllclose, run_perftest


def make_ids(tokens, topk, experts, distribution):
    ids = torch.randint(experts, (tokens, topk), dtype=torch.int32, device="cuda")
    if distribution == "shared":
        ids[:, -1] = experts - 1
    elif distribution == "hot":
        ids.fill_(experts - 1)
    elif distribution == "dropped":
        ids.view(-1)[::3] = -1
    return ids


def route_reference(ids, experts, max_m, initial=None):
    ids = ids.to(device="cpu", dtype=torch.int64).flatten()
    counts = torch.bincount(ids[ids >= 0], minlength=experts)
    start = (
        torch.zeros(experts, dtype=torch.int64, device="cpu")
        if initial is None
        else initial.cpu().long()
    )
    expected = [
        torch.full((int((ids < 0).sum()),), -1, dtype=torch.int64, device="cpu")
    ]
    for e in range(experts):
        expected.append(
            e * max_m + torch.arange(start[e], start[e] + counts[e], device="cpu")
        )
    return (counts + start).int(), torch.cat(expected).sort().values


def check_routes(ids, counter, rows, experts, max_m, initial=None):
    counts, expected_rows = route_reference(ids, experts, max_m, initial)
    torch.testing.assert_close(counter.cpu(), counts, rtol=0, atol=0)
    actual = rows.cpu().long().flatten()
    torch.testing.assert_close(actual.sort().values, expected_rows, rtol=0, atol=0)
    flat_ids = ids.cpu().long().flatten()
    assert torch.equal(actual[flat_ids < 0], torch.full_like(actual[flat_ids < 0], -1))
    keep = flat_ids >= 0
    assert torch.equal(actual[keep] // max_m, flat_ids[keep])


@benchmark()
def test_route_rows(tokens, topk, experts, distribution):
    torch.manual_seed(123)
    ids = make_ids(tokens, topk, experts, distribution)
    max_m = max(1, ids.numel())
    ref_counts, _ = route_reference(ids, experts, max_m)
    candidates = {"flydsl": lambda: flydsl_moe_topids_to_rows(ids, experts, max_m)}
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (counts, rows), us = run_perftest(fn)
        check_routes(ids, counts, rows, experts, max_m)
        err = checkAllclose(
            ref_counts.to(device=counts.device, dtype=dtypes.fp32),
            counts.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"{name}: route counts",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = 0.0  # Integer routing, no floating-point operations.
        ret[f"{name} TB/s"] = (8 * ids.numel() + 8 * experts) / us / 1e6
        ret[f"{name} err"] = err
    return ret


# The benchmark wrapper accepts CLI sweep arguments, not pytest fixtures.
test_route_rows.__test__ = False


@pytest.mark.parametrize("experts", [1, 17, 129, 512, 513])
@pytest.mark.parametrize("tokens,topk", [(1, 1), (255, 1), (205, 5), (2517, 5)])
@pytest.mark.parametrize("distribution", ["uniform", "shared", "hot", "dropped"])
def test_route_rows_graph_and_counter(experts, tokens, topk, distribution):
    if not torch.cuda.is_available() or get_gfx() not in (
        "gfx942",
        "gfx950",
        "gfx1250",
    ):
        pytest.skip("supported ROCm GPU required")
    torch.manual_seed(456)
    ids = make_ids(tokens, topk, experts, distribution)
    initial = torch.arange(experts, dtype=torch.int32, device="cuda") % 4
    max_m = max(1, ids.numel() + 4)
    counter = initial.clone()

    def launch():
        counter.copy_(initial)
        return flydsl_moe_topids_to_rows(ids, experts, max_m, counter=counter)

    counts, rows = launch()
    assert counts.data_ptr() == counter.data_ptr()
    check_routes(ids, counts, rows, experts, max_m, initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        counts, rows = launch()
    for i in range(8):
        ids.copy_(make_ids(tokens, topk, experts, distribution))
        initial.fill_(i % 4)
        rows.fill_(-987654)
        graph.replay()
        torch.cuda.synchronize()
        check_routes(ids, counts, rows, experts, max_m, initial)


@pytest.mark.parametrize(
    "experts", [1, 129, 512, 513, 768, 896, 1024, 1536, 2048, 4096]
)
def test_tp_route_capacity_and_boundary(experts, monkeypatch):
    """Exercise both sides of the capacity-dependent public dispatch gate."""
    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        pytest.skip("gfx1250 TP histogram dispatch required")
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.kernels.moe_route_maps import (
        MAX_ROUTE_BUCKETS,
        tp_route_bucket_capacity,
        tp_route_min_routes,
    )

    capacity = tp_route_bucket_capacity(experts)
    assert experts <= capacity <= 4096 and capacity % 256 == 0
    assert MAX_ROUTE_BUCKETS == 512  # EP scratch remains unchanged.
    boundary = tp_route_min_routes(experts)
    selected = []
    getter = moe_kernels._get_compiled_topids_to_rows_lds

    def record(n):
        selected.append(n)
        return getter(n)

    monkeypatch.setattr(moe_kernels, "_get_compiled_topids_to_rows_lds", record)
    for routes in (boundary - 1, boundary, boundary + 1):
        ids = make_ids(routes, 1, experts, "dropped")
        initial = torch.arange(experts, dtype=torch.int32, device="cuda") % 4
        counter = initial.clone()
        selected.clear()
        counts, rows = moe_kernels.flydsl_moe_topids_to_rows(
            ids, experts, routes + 4, counter=counter
        )
        check_routes(ids, counts, rows, experts, routes + 4, initial)
        assert bool(selected) == (routes >= boundary)


@pytest.mark.parametrize("experts", [513, 896, 1024, 1536, 2048, 4096, 4097])
def test_wide_tp_route_changed_graph_inputs(experts):
    """Real top-16 IDs, then all dropped/hot inputs and changed preset offsets."""
    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        pytest.skip("gfx1250 TP routing required")
    from aiter.ops.flydsl.kernels.moe_route_maps import tp_route_min_routes

    routes = tp_route_min_routes(experts) if experts <= 4096 else 4096
    tokens = (routes + 15) // 16
    ids = torch.rand(tokens, experts, device="cuda").topk(16, dim=1).indices.int()
    initial = torch.zeros(experts, dtype=torch.int32, device="cuda")
    counter = initial.clone()
    max_m = ids.numel() + 4

    def call():
        counter.copy_(initial)
        return flydsl_moe_topids_to_rows(ids, experts, max_m, counter=counter)

    counts, out = call()
    check_routes(ids, counts, out, experts, max_m, initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        counts, out = call()
    for value in (-7, experts - 1, -1, 0):
        ids.fill_(value)
        initial.fill_(3)
        out.fill_(-98765)
        graph.replay()
        torch.cuda.synchronize()
        check_routes(ids, counts, out, experts, max_m, initial)


def test_tp_route_capacity_rejects_invalid_experts():
    from aiter.ops.flydsl.kernels.moe_route_maps import tp_route_bucket_capacity

    for experts in (-1, 0, 4097):
        with pytest.raises(ValueError, match="bucket count out of range"):
            tp_route_bucket_capacity(experts)


def main():
    if get_gfx() not in ("gfx942", "gfx950", "gfx1250"):
        aiter.logger.warning("MoE route rows unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-b", "--batch", nargs="+", type=int, default=[1, 64, 205, 2517]
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[5])
    parser.add_argument("--experts", nargs="+", type=int, default=[129])
    parser.add_argument(
        "--distribution",
        nargs="+",
        choices=["uniform", "shared", "hot", "dropped"],
        default=["uniform", "shared", "hot", "dropped"],
    )
    args = parser.parse_args()
    rows = [
        test_route_rows(*shape)
        for shape in itertools.product(
            args.batch, args.topk, args.experts, args.distribution
        )
    ]
    aiter.logger.info(
        "MoE route rows:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
