# SPDX-License-Identifier: MIT
"""Same selected GEMMs with different task representation, TP8 and TP4."""

import pytest
import torch
import aiter.iq2r_moe as moe
from aiter.iq2r_quad_pack import repack_gate_quad
from aiter.ops.iq2r_format import IQ2RMetadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="requires gfx950",
)


@pytest.fixture(scope="module", params=[8, 4])
def packed(request):
    tp = request.param
    gen = torch.Generator(device="cuda").manual_seed(163100 + tp)
    gm = IQ2RMetadata(logical_n=4096 // tp, logical_k=6144)
    dm = IQ2RMetadata(logical_n=6144, logical_k=2048 // tp)
    v = []
    for meta in (gm, dm):
        data = torch.randint(
            256, (257, meta.data_bytes), device="cuda", dtype=torch.uint8, generator=gen
        )
        aux = torch.empty((257, meta.auxiliary_bytes), device="cuda", dtype=torch.uint8)
        aux[:, :4096].random_(8, 65, generator=gen)
        aux[:, :8].zero_()
        aux[:, 4096:].fill_(112)
        v.extend((data, aux))
    return tp, *v, gm, dm, repack_gate_quad(v[0], verify=True)


@pytest.mark.parametrize("tokens", [4, 8])
def test_same_moe(packed, tokens, monkeypatch):
    tp, gd, ga, dd, da, gm, dm, quad = packed
    for name in [
        "INDEXED_INPUT",
        "QUAD_GATE",
        "SPARSE_QUAD_GATE",
        "SMALL_QUAD_GATE",
        "SCHEDULED",
        "SPARSE_DOWN",
        "FUSED_DOWN",
        "C32",
        "ROUTE9_DOWN",
        "DIRECT_GATE",
    ]:
        monkeypatch.setattr(moe, "IQ2R_GLM53_" + name, True)
    monkeypatch.setattr(moe, "IQ2R_GLM53_CODEBOOK_BATCH", False)
    monkeypatch.setattr(moe, "IQ2R_GLM53_SPLIT4_GATE", False)
    monkeypatch.setattr(moe, "IQ2R_GLM53_TP4", tp == 4)
    monkeypatch.setenv("IQ2R_GLM53_TP4", "1" if tp == 4 else "0")
    monkeypatch.setattr(moe, "IQ2R_GLM53_PAIR_DOWN", 0)
    monkeypatch.setenv("IQ2R_GLM53_BALLOT_ROUTER", "1")
    gen = torch.Generator(device="cuda").manual_seed(163200 + tokens)
    hidden = torch.randn(
        (tokens, 6144), device="cuda", dtype=torch.bfloat16, generator=gen
    )
    ids = torch.empty((tokens, 9), device="cuda", dtype=torch.int32)
    weights = torch.empty((tokens, 9), device="cuda", dtype=torch.float32)

    def change(step):
        hidden.normal_(generator=gen)
        routed = (
            torch.arange(8, device="cuda").repeat(tokens, 1)
            if step % 2
            else torch.stack(
                [
                    torch.randperm(256, device="cuda", generator=gen)[:8]
                    for _ in range(tokens)
                ]
            )
        )
        ids[:, :8].copy_(routed)
        ids[:, 8].fill_(256)
        weights[:, :8].copy_(
            torch.randn((tokens, 8), device="cuda", generator=gen).softmax(-1)
        )
        weights[:, 8].fill_(1)

    change(0)
    states = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("IQ2R_GLM53_STATIC_BALLOT", flag)
        ws = moe.IQ2RMoeWorkspace.allocate(
            tokens + 17,
            9,
            device="cuda",
            max_experts=257,
            hidden_size=6144,
            intermediate_size=2048 // tp,
        )
        output = torch.empty_like(hidden)

        def run():
            moe.iq2r_fused_moe_out(
                hidden,
                gd,
                ga,
                dd,
                da,
                weights,
                ids,
                output,
                gate_up_metadata=gm,
                down_metadata=dm,
                gate_up_tile_n=128,
                down_tile_n=128,
                gate_up_bias=None,
                down_bias=None,
                workspace=ws,
                gate_quad_data=quad,
                swiglu_limit=0.0,
                swiglu_alpha=1.0,
                swiglu_up_offset=0.0,
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        states[flag] = (ws, output, graph)
    for step in range(6):
        change(step)
        snapshots = []
        for flag, (ws, out, graph) in states.items():
            out.fill_(float("nan"))
            ws.intermediate_fp8.view(torch.uint8).fill_(127)
            ws.intermediate_scales.zero_()
            graph.replay()
            order = ws.scatter_indices[: tokens * 9].long()
            snapshots.append(
                (
                    out.view(torch.uint8).clone(),
                    ws.intermediate_fp8[: tokens * 9]
                    .view(torch.uint8)
                    .index_select(0, order),
                    ws.intermediate_scales[: tokens * 9].index_select(0, order),
                )
            )
            assert bool(torch.isfinite(out).all())
        assert all(torch.equal(a, b) for a, b in zip(*snapshots)), (tp, tokens, step)
