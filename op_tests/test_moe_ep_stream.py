# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Local EP mapping kernels only; no inter-rank transport is simulated."""

import importlib
from types import SimpleNamespace

import pytest
import torch

MK = importlib.import_module("aiter.ops.flydsl.moe_kernels")
GM = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
SHIM = importlib.import_module("aiter.ops.flydsl.kernels.tensor_shim")


@pytest.fixture
def t(monkeypatch):
    fast = SHIM.current_stream_arg

    def select(arm):
        getter = torch.cuda.current_stream if arm == "baseline" else fast
        monkeypatch.setattr(MK, "current_stream_arg", getter)
        monkeypatch.setattr(GM, "current_stream_arg", getter)

    return SimpleNamespace(MK=MK, GM=GM, select=select)


@pytest.mark.parametrize(
    "policy", ["plain_lut", "lds_lut", "direct_mask", "fused_mask"]
)
@pytest.mark.parametrize("valid", [0, 17, 37])
def test_ep_route_and_remap_stream_graph(policy, valid, monkeypatch, t):
    if not torch.cuda.is_available():
        pytest.skip("requires ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 EP route/remap test")

    E, EG, M, K, TILE = 4, 8, 37, 2, 16
    ids = (
        torch.arange(M * K, device="cuda", dtype=torch.int32).reshape(M, K) % EG
    ).contiguous()
    weights = torch.linspace(0.125, 0.875, M * K, device="cuda").reshape(M, K)
    mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda", dtype=torch.int32)
    lut = torch.tensor([E, E, E, E, 0, 1, 2, 3], device="cuda", dtype=torch.int32)
    nvt = torch.tensor([valid], device="cuda", dtype=torch.int32)
    nvr = torch.tensor([valid * K], device="cuda", dtype=torch.int32)
    tis = torch.arange(M, device="cuda", dtype=torch.int32)
    cap = M * K + E * TILE
    gw = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
    epmap = torch.empty((cap + 1, 2), device="cuda", dtype=torch.int32)
    inverse = torch.empty(cap, device="cuda", dtype=torch.int32)
    # All non-timing setup explicitly ordered on a non-default stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    monkeypatch.setenv(
        "AITER_FLYDSL_ROUTE_G2L_LDS", "0" if policy == "plain_lut" else "1"
    )

    def invoke():
        nvr.copy_(nvt * K)
        gw.fill_(float("nan"))
        epmap[:, 0].fill_(-1)
        epmap[:, 1].zero_()
        inverse.fill_(-1)
        opts = {
            "gather_w": gw,
            "weight_in": weights,
            "num_valid_routes": nvr,
            "num_local_tokens": nvt,
        }
        if policy == "fused_mask":
            opts["expert_mask"] = mask
        else:
            opts["g2l_lut"] = mask if policy == "direct_mask" else lut
            opts["contiguous_expert_mask"] = policy == "direct_mask"
        counts, rows = t.MK.flydsl_moe_topids_to_rows(ids, E, M * K, **opts)
        masked_rows = rows.clone()
        starts, psum, total = t.GM.contiguous_psum_remap(
            counts,
            rows,
            E,
            M * K,
            TILE,
            num_valid_routes=nvr,
            ep_scatter_params={
                "gather_w": gw,
                "tis": tis,
                "ep_rowmap": epmap,
                "topk": K,
                "max_tok": M,
                "slot_stride": M * K,
            },
            row_to_token=inverse,
            topk=K,
        )
        return counts, masked_rows, rows, starts, psum, total

    def check(result):
        counts, masked, rows, starts, psum, total = [x.cpu() for x in result]
        iid = ids.cpu().flatten()
        ws = weights.cpu().flatten().to(torch.bfloat16)
        active = torch.arange(M * K) < int(nvt.cpu()[0]) * K
        keep = active & (iid >= 4)
        dropped = active & ~keep
        buckets = iid % 4
        expected_counts = torch.bincount(buckets[keep].long(), minlength=E).to(
            torch.int32
        )
        assert torch.equal(counts, expected_counts)
        padded = (counts + TILE - 1) // TILE * TILE
        expected_starts = padded.cumsum(0) - padded
        assert torch.equal(starts, expected_starts)
        assert torch.equal(psum, starts + counts)
        assert int(total) == max(int(padded.sum()), TILE)
        masked = masked.flatten()
        rows = rows.flatten()
        assert bool((masked[dropped] < 0).all())
        assert bool((rows[dropped] < 0).all())
        assert torch.equal(masked[keep] // (M * K), buckets[keep])
        for e in range(E):
            indices = rows[keep & (buckets == e)]
            assert torch.equal(
                indices.sort().values,
                torch.arange(
                    int(starts[e]), int(starts[e] + counts[e]), dtype=torch.int32
                ),
            )
        expected_weights = torch.where(keep, ws, torch.zeros_like(ws))
        got_weights = gw.cpu().flatten()
        assert torch.equal(got_weights[active], expected_weights[active])
        assert torch.isnan(
            got_weights[~active]
        ).all(), "Kernel must leave truncated tail untouched"
        actual_map = epmap.cpu()
        actual_inverse = inverse.cpu()
        flat_tis = tis.cpu()
        for route in torch.where(keep)[0].tolist():
            row = int(rows[route])
            token = route // K
            slot = route % K
            origin = int(flat_tis[token])
            destination = (origin // M) * (M * K) + (origin % M) * K + slot
            assert int(actual_map[row, 0]) == destination
            weight_bits = ws[route : route + 1].float().view(torch.int32)[0]
            assert actual_map[row, 1] == weight_bits
            assert int(actual_inverse[row]) == token
        used = rows[keep].long()
        unused = torch.ones(cap + 1, dtype=torch.bool)
        unused[used] = False
        assert bool((actual_map[unused, 0] == -1).all())
        return counts, starts, psum

    with torch.cuda.stream(stream):
        out = {}
        for arm in ("baseline", "candidate"):
            t.select(arm)
            result = invoke()
            torch.cuda.synchronize()
            out[arm] = check(result)
        assert all(torch.equal(a, b) for a, b in zip(out["baseline"], out["candidate"]))
        for arm in ("baseline", "candidate"):
            t.select(arm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = invoke()
            for nv in (37, 0, 17):
                nvt.fill_(nv)
                ids.copy_((ids + 1) % EG)
                graph.replay()
                torch.cuda.synchronize()
                check(result)
    torch.cuda.current_stream().wait_stream(stream)
    t.select("candidate")
