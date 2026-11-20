# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP route wrappers promise None means the full static route extent."""

import importlib

import pytest
import torch

MK = importlib.import_module("aiter.ops.flydsl.moe_kernels")


def gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() not in ("gfx1250", "gfx950", "gfx942"):
        pytest.skip("tested ROCm EP route targets")


@pytest.mark.parametrize("variant", ["plain", "lds", "fused"])
def test_omitted_count_reaches_launcher_as_null(variant, monkeypatch):
    # Stop at the launch boundary: safe even before the fix, and demonstrates
    # why a kernel must handle null rather than blindly load count[0].
    monkeypatch.setenv("AITER_FLYDSL_ROUTE_G2L_LDS", "0" if variant == "plain" else "1")
    seen = []
    sentinel = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: sentinel)
    for name in (
        "_get_compiled_topids_to_rows_g2l",
        "_get_compiled_route_g2l_lds",
        "_get_compiled_route_g2l_fused",
    ):
        monkeypatch.setattr(MK, name, lambda *a, **kw: sentinel)
    monkeypatch.setattr(MK, "_dispatch_compiled", lambda fn, *args: seen.append(args))
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    output = torch.empty_like(weights, dtype=torch.bfloat16)
    kw = (
        {"expert_mask": torch.tensor([1, 0, 1, 0], dtype=torch.int32)}
        if variant == "fused"
        else {"g2l_lut": torch.tensor([0, 2, 1, 2], dtype=torch.int32)}
    )
    MK.flydsl_moe_topids_to_rows(ids, 2, 8, gather_w=output, weight_in=weights, **kw)
    assert len(seen) == 1
    assert seen[0][6].pointer.value is None


def fixture_data(tokens, dtype):
    global_experts = 16
    mask = torch.tensor([1, 0] * 8, device="cuda", dtype=torch.int32)
    experts = int(mask.sum().item())
    lut = torch.where(mask != 0, mask.cumsum(0).int() - 1, experts).int()
    ids = (torch.arange(tokens * 3, device="cuda", dtype=torch.int32) * 7).reshape(
        tokens, 3
    ) % global_experts
    weights = (
        torch.arange(ids.numel(), device="cuda", dtype=torch.float32).reshape_as(ids)
        % 13
    ) / 16
    out_weights = torch.empty_like(weights, dtype=dtype)
    nvalid = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    nvr = torch.tensor([ids.numel()], device="cuda", dtype=torch.int32)
    return mask, lut, experts, ids, weights, out_weights, nvalid, nvr


def check_routes(counts, rows, weights_out, ids, weights, lut, experts, max_m, live):
    flat_ids = ids.reshape(-1)[:live].long()
    local = lut[flat_ids]
    keep = local < experts
    expected_counts = torch.bincount(local[keep].long(), minlength=experts).int()
    assert torch.equal(counts, expected_counts)
    r = rows.reshape(-1)[:live]
    assert (r[~keep] == -1).all()
    assert torch.equal(r[keep] // max_m, local[keep])
    for e in range(experts):
        got = r[local == e] % max_m
        assert torch.equal(
            got.sort().values,
            torch.arange(got.numel(), device="cuda", dtype=torch.int32),
        )
    expected_weights = torch.where(
        keep, weights.reshape(-1)[:live].to(weights_out.dtype), 0
    )
    assert torch.equal(weights_out.reshape(-1)[:live], expected_weights)


@pytest.mark.parametrize("variant", ["plain", "lds", "fused"])
@pytest.mark.parametrize("tokens", [1, 17, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("count_mode", ["omitted", "routes", "tokens"])
def test_optional_count_routes_and_changed_graph(
    variant, tokens, dtype, count_mode, monkeypatch
):
    gpu()
    monkeypatch.setenv("AITER_FLYDSL_ROUTE_G2L_LDS", "0" if variant == "plain" else "1")
    mask, lut, E, ids, weights, out_weights, nvalid, nvr = fixture_data(tokens, dtype)
    max_m = ids.numel() + 3
    kw = {"expert_mask": mask} if variant == "fused" else {"g2l_lut": lut}
    if count_mode == "routes":
        kw["num_valid_routes"] = nvr
    elif count_mode == "tokens":
        kw["num_local_tokens"] = nvalid

    def call():
        return MK.flydsl_moe_topids_to_rows(
            ids, E, max_m, gather_w=out_weights, weight_in=weights, **kw
        )

    call()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            counts, rows = call()
        # Omitted count always means all routes; supplied counts remain dynamic.
        for step, live_tokens in enumerate((tokens, tokens // 2, 0, tokens)):
            if step == 1:
                ids.fill_(1)  # all experts non-local
            elif step == 2:
                ids.fill_(0)  # hotspot with zero valid count
            else:
                ids.copy_(
                    (
                        torch.arange(ids.numel(), device="cuda", dtype=torch.int32) * 7
                    ).reshape_as(ids)
                    % 16
                )
            weights.neg_()
            nvalid.fill_(live_tokens)
            route_count = live_tokens * 3
            if count_mode == "routes" and step == 1:
                route_count = min(ids.numel(), route_count + 1)
            nvr.fill_(route_count)
            out_weights.fill_(-83)
            graph.replay()
            live = (
                ids.numel()
                if count_mode == "omitted"
                else route_count if count_mode == "routes" else live_tokens * 3
            )
            check_routes(counts, rows, out_weights, ids, weights, lut, E, max_m, live)
            # EP dead-tail is intentionally left untouched.
            assert (out_weights.reshape(-1)[live:] == -83).all()
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("tokens", [1, 17, 257])
def test_direct_mask_writes_count_and_keeps_graph_replay(tokens, monkeypatch):
    gpu()
    monkeypatch.setenv("AITER_FLYDSL_ROUTE_G2L_LDS", "1")
    E = 8
    mask = torch.tensor([1] * 8 + [0] * 8, device="cuda", dtype=torch.int32)
    ids = (torch.arange(tokens * 3, device="cuda", dtype=torch.int32) * 7).reshape(
        tokens, 3
    ) % 16
    weights = torch.full(ids.shape, 0.5, device="cuda")
    output = torch.empty_like(weights, dtype=torch.bfloat16)
    nvt = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    nvr = torch.empty(1, device="cuda", dtype=torch.int32)
    max_m = ids.numel() + 1
    lut = torch.where(
        mask != 0, torch.arange(16, device="cuda", dtype=torch.int32) % E, E
    ).int()

    def call():
        return MK.flydsl_moe_topids_to_rows(
            ids,
            E,
            max_m,
            g2l_lut=mask,
            gather_w=output,
            weight_in=weights,
            num_local_tokens=nvt,
            num_valid_routes=nvr,
            contiguous_expert_mask=True,
        )

    call()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            counts, rows = call()
        for live in (tokens, 0, tokens // 2, tokens):
            nvt.fill_(live)
            nvr.fill_(-99)
            output.fill_(-83)
            graph.replay()
            assert nvr.item() == live * 3
            check_routes(counts, rows, output, ids, weights, lut, E, max_m, live * 3)
            assert (output.reshape(-1)[live * 3 :] == -83).all()
    torch.cuda.current_stream().wait_stream(stream)
