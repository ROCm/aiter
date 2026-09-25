# SPDX-License-Identifier: MIT
"""Native GLM TP8 scheduling: exact bytes, changing graphs and dispatch bounds."""

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


@pytest.fixture(scope="module")
def packed_weights():
    # Distinct codebooks expose shared-memory reuse races. Finite magnitudes,
    # reserved entry zero, canonical padding and per-block scale bases retain
    # the production ABI without requiring a model checkpoint.
    generator = torch.Generator(device="cuda").manual_seed(0xE106)
    metadata = (
        IQ2RMetadata(logical_n=512, logical_k=6144),
        IQ2RMetadata(logical_n=6144, logical_k=256),
    )
    values = []
    for meta in metadata:
        data = torch.randint(
            256,
            (257, meta.data_bytes),
            generator=generator,
            device="cuda",
            dtype=torch.uint8,
        )
        auxiliary = torch.empty(
            (257, meta.auxiliary_bytes), device="cuda", dtype=torch.uint8
        )
        auxiliary[:, :4096].random_(8, 65, generator=generator)
        auxiliary[:, :8].zero_()
        auxiliary[:, 4096:].fill_(112)
        values.extend((data, auxiliary))
    return (*values, *metadata, repack_gate_quad(values[0], verify=True))


@pytest.mark.parametrize(
    "tokens", [1, 2, 3, 4, 8, 16, 17, 31, 32, 33, 64, 128, 255, 256, 257]
)
@pytest.mark.parametrize("pair", [0])
def test_scheduled_moe_exact_dynamic_graph(packed_weights, tokens, monkeypatch, pair):
    monkeypatch.setattr(moe, "IQ2R_GLM53_PAIR_DOWN", pair)
    gd, ga, dd, da, gm, dm, quad = packed_weights
    for setting in (
        "IQ2R_GLM53_INDEXED_INPUT",
        "IQ2R_GLM53_QUAD_GATE",
        "IQ2R_GLM53_SPARSE_QUAD_GATE",
        "IQ2R_GLM53_SMALL_QUAD_GATE",
    ):
        monkeypatch.setattr(moe, setting, True)
    monkeypatch.setattr(moe, "IQ2R_GLM53_SPLIT4_GATE", False)
    monkeypatch.setattr(moe, "IQ2R_GLM53_FUSED_DOWN", True)
    monkeypatch.setattr(moe, "IQ2R_GLM53_C32", True)
    monkeypatch.setattr(moe, "IQ2R_GLM53_ROUTE9_DOWN", True)
    monkeypatch.setattr(moe, "IQ2R_GLM53_DIRECT_GATE", True)
    monkeypatch.setattr(moe, "IQ2R_GLM53_CODEBOOK_BATCH", True)
    generator = torch.Generator(device="cuda").manual_seed(106000 + tokens)
    hidden = torch.randn(
        (tokens, 6144), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    ids = torch.empty((tokens, 9), device="cuda", dtype=torch.int32)
    weights = torch.empty((tokens, 9), device="cuda", dtype=torch.float32)

    def change_inputs(step):
        hidden.normal_(generator=generator)
        routed = (
            torch.arange(8, device="cuda").repeat(tokens, 1)
            if step % 2
            else torch.stack(
                [
                    torch.randperm(256, device="cuda", generator=generator)[:8]
                    for _ in range(tokens)
                ]
            )
        )
        if tokens == 8 and step in (2, 3):
            routed = torch.arange(8, device="cuda").repeat(tokens, 1)
            routed[2, 0] = 11
            if step == 3:
                routed[4, 1] = 12
        ids[:, :8].copy_(routed)
        ids[:, 8].fill_(256)
        weights[:, :8].copy_(
            torch.randn((tokens, 8), device="cuda", generator=generator).softmax(-1)
        )
        weights[:, 8].fill_(1)

    states = {}
    native_calls = []
    for name in (
        "iq2r_gate_quad_scheduled_out",
        "iq2r_down_shortk_out",
        "iq2r_down_sparse_scheduled_out",
        "iq2r_down_token_route9_out",
        "iq2r_down_token_adaptive9_out",
    ):
        original = getattr(moe, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            native_calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(moe, name, counted)

    change_inputs(0)
    for enabled in (False, True):
        monkeypatch.setattr(moe, "IQ2R_GLM53_ADAPTIVE_DOWN", enabled)
        monkeypatch.setattr(moe, "IQ2R_GLM53_SCHEDULED", enabled)
        monkeypatch.setattr(moe, "IQ2R_GLM53_SPARSE_DOWN", enabled)
        workspace = moe.IQ2RMoeWorkspace.allocate(
            tokens + 17,
            9,
            device="cuda",
            max_experts=257,
            hidden_size=6144,
            intermediate_size=256,
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
                workspace=workspace,
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
        states[enabled] = (workspace, output, graph)

    assert native_calls.count("iq2r_down_token_adaptive9_out") == (
        2 if tokens == 8 else 0
    ), native_calls
    for step in range(6):
        change_inputs(step)
        snapshots = []
        for workspace, output, graph in states.values():
            output.fill_(float("nan"))
            workspace.intermediate_fp8.view(torch.uint8).fill_(127)
            workspace.intermediate_scales.zero_()
            graph.replay()
            order = workspace.scatter_indices[: tokens * 9].long()
            snapshots.append(
                (
                    output.view(torch.uint8).clone(),
                    workspace.intermediate_fp8[: tokens * 9]
                    .view(torch.uint8)
                    .index_select(0, order),
                    workspace.intermediate_scales[: tokens * 9].index_select(0, order),
                )
            )
            assert bool(torch.isfinite(output).all())
        assert all(torch.equal(a, b) for a, b in zip(*snapshots)), (tokens, step)
