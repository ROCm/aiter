# SPDX-License-Identifier: MIT
"""Invalid padding must not address weights or change any active row."""

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
    gen = torch.Generator(device="cuda").manual_seed(181000 + tp)
    gm = IQ2RMetadata(logical_n=4096 // tp, logical_k=6144)
    dm = IQ2RMetadata(logical_n=6144, logical_k=2048 // tp)
    values = []
    for meta in [gm, dm]:
        data = torch.randint(
            256, (257, meta.data_bytes), device="cuda", dtype=torch.uint8, generator=gen
        )
        aux = torch.empty((257, meta.auxiliary_bytes), device="cuda", dtype=torch.uint8)
        aux[:, :4096].random_(8, 65, generator=gen)
        aux[:, :8].zero_()
        aux[:, 4096:].fill_(112)
        values.extend([data, aux])
    return tp, *values, gm, dm, repack_gate_quad(values[0], verify=True)


@pytest.mark.parametrize("tokens", [4, 8])
@pytest.mark.parametrize("pattern", ["hot", "spread"])
@pytest.mark.parametrize("invalid", [-1, 2147483647])
def test_invalid_padding(packed, tokens, pattern, invalid, monkeypatch):
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
    monkeypatch.setattr(moe, "IQ2R_GLM53_TP4", tp == 4)
    monkeypatch.setenv("IQ2R_GLM53_TP4", "1" if tp == 4 else "0")
    monkeypatch.setenv("IQ2R_GLM53_BALLOT_ROUTER", "1")
    monkeypatch.setattr(moe, "IQ2R_GLM53_ADAPTIVE_DOWN", tp == 8)
    monkeypatch.setattr(moe, "IQ2R_GLM53_PAIR_DOWN", 0)
    monkeypatch.setattr(moe, "IQ2R_GLM53_CODEBOOK_BATCH", False)
    monkeypatch.setattr(moe, "IQ2R_GLM53_SPLIT4_GATE", False)
    gen = torch.Generator(device="cuda").manual_seed(181100 + tokens)
    hidden = torch.randn(
        (tokens, 6144), device="cuda", dtype=torch.bfloat16, generator=gen
    )
    ids = torch.empty((tokens, 9), device="cuda", dtype=torch.int32)
    ids[:, :8] = (
        torch.arange(8, device="cuda").repeat(tokens, 1)
        if pattern == "hot"
        else torch.stack(
            [
                torch.randperm(256, device="cuda", generator=gen)[:8]
                for _ in range(tokens)
            ]
        )
    )
    ids[:, 8] = 256
    weights = torch.randn((tokens, 9), device="cuda", generator=gen).softmax(-1)
    weights[:, 8] = 1
    saved_ids, saved_weights = ids.clone(), weights.clone()
    active = tokens - 1
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
    expected = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for dirty in [True, False, True]:
        ids.copy_(saved_ids)
        weights.copy_(saved_weights)
        if dirty:
            ids[active:, :8] = invalid
            weights[active:, :8] = float("nan")
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output[:active], expected[:active]), (
            tp,
            tokens,
            pattern,
            invalid,
        )
        assert bool(torch.isfinite(output[:active]).all())
        if dirty:
            assert bool((ids[active:, :8] == invalid).all())
        else:
            assert torch.equal(output, expected)
