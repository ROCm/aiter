# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP A-preshuffle must build the inverse route map, including small batches."""

import importlib

import pytest
import torch

from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.mark.parametrize("tokens", [1, 16, 32, 191, 256, 409, 410])
@pytest.mark.parametrize("tile_m", [16, 32, 64])
def test_tp_apreshuffle_matches_row_major_and_replays(tokens, tile_m, monkeypatch):
    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        pytest.skip("requires gfx1250")
    gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
    torch.manual_seed(1193 + tokens)
    experts, hidden, inter, topk = 17, 1024, 1024, 5

    def packed(n, k, gate_up=False):
        raw = torch.randint(
            0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
        )
        return moe_shuffle_weight(
            raw, experts_cnt=experts, is_guinterleave=gate_up, gate_up=gate_up
        )

    def scales(n, k, gate_up=False):
        raw = torch.full((experts, n, k // 32), 121, device="cuda", dtype=torch.uint8)
        return moe_shuffle_scale(
            raw, experts_cnt=experts, is_guinterleave=gate_up, gate_up=gate_up
        )

    x = (torch.randn(tokens, hidden, device="cuda") * 0.5).bfloat16()
    ids = torch.rand(tokens, experts, device="cuda").topk(topk).indices.int()
    weights = torch.rand(tokens, topk, device="cuda")
    weights /= weights.sum(1, keepdim=True)
    kwargs = {
        "hidden_states": x,
        "w1": packed(2 * inter, hidden, True),
        "w2": packed(hidden, inter),
        "topk_weight": weights,
        "topk_ids": ids,
        "E": experts,
        "model_dim": hidden,
        "inter_dim": inter,
        "dtype": torch.bfloat16,
        "activation": ActivationType.Swiglu,
        "w1_scale": scales(2 * inter, hidden, True),
        "w2_scale": scales(hidden, inter),
        "bias1": None,
        "bias2": None,
        "swiglu_limit": 7.0,
        "doweight_stage1": False,
        "tile_m": tile_m,
        "tile_n": 128,
        "tile_k": 128,
        "m_warp": 1,
        "n_warp": 2,
        "num_buffers": 2,
        "tile_m2": tile_m,
        "tile_n2": 128,
        "tile_k2": 128,
        "m_warp2": 1,
        "n_warp2": 2,
        "num_buffers2": 2,
        "cluster_m": 1,
        "cluster_n": 1,
        "cluster_m2": 1,
        "cluster_n2": 1,
        "waves_per_tensor_tdm": 1,
        "waves_per_tensor_tdm2": 1,
        "data_format": "fp4",
        "tiles_configured": True,
    }

    def run(apre):
        return gm._grouped_a8w4_tdm_moe(**kwargs, a_preshuffle=apre)

    fused_calls = []
    fused_route = gm.fused_route_psum_remap

    def record_fused_route(*args, **kw):
        fused_calls.append(True)
        return fused_route(*args, **kw)

    monkeypatch.setattr(gm, "fused_route_psum_remap", record_fused_route)
    reference = run(0).clone()
    expected_fused = tokens * topk <= gm._FUSED_ROUTE_PSUM_MAX_NUMEL
    assert bool(fused_calls) == expected_fused
    fused_calls.clear()
    candidate = run(1)
    assert not fused_calls, "A-preshuffle needs the inverse route map"
    assert torch.count_nonzero(reference).item() > 0
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    graphs, outputs = [], []
    for apre in (0, 1):
        run(apre)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(apre)
        graphs.append(graph)
        outputs.append(out)
    try:
        for rep in range(6):
            x.neg_()
            ids.copy_(
                torch.rand(tokens, experts, device="cuda").topk(topk).indices.int()
            )
            weights.fill_(1.0 / topk)
            # Include live route changes, a dead tail, all dropped, then live.
            if rep == 2:
                ids[tokens // 2 :] = -1
                weights[tokens // 2 :] = 0
            elif rep == 3:
                ids.fill_(-1)
                weights.zero_()
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
            if rep == 3:
                assert torch.count_nonzero(outputs[1]).item() == 0
    finally:
        for graph in graphs:
            graph.reset()


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
