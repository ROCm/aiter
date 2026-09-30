# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R packed MoE (TP4/TP8, gfx950) against the generic IQ2R MoE."""

from types import SimpleNamespace

import pytest
import torch

from aiter.iq2r_glm53 import (
    IQ2RGlm53Workspace,
    _default_config,
    iq2r_glm53_config,
    iq2r_glm53_moe_out,
    iq2r_glm53_pack,
    iq2r_glm53_slice_gate,
)
from aiter.iq2r_moe import IQ2RMoeWorkspace, iq2r_fused_moe_out
from aiter.ops.iq2r import iq2r_encode_device
from aiter.ops.iq2r_encoder import iq2r_initial_codebook
from aiter.ops.iq2r_format import (
    IQ2RMetadata,
    iq2r_slice_input_data,
    iq2r_slice_output_auxiliary,
)

_EXPERTS = 257
_TOPK = 9
_HIDDEN = 6144


def _has_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName
    )


pytestmark = pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")


def _relative_metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    relative_rmse = (
        actual - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1), expected.reshape(1, -1), dim=-1
    )[0]
    return relative_rmse.item(), cosine.item()


@pytest.fixture(scope="module", params=[256, 512], ids=["tp8", "tp4"])
def glm53_layer(request):
    intermediate = request.param
    gate_metadata = IQ2RMetadata(logical_n=2 * intermediate, logical_k=_HIDDEN)
    down_metadata = IQ2RMetadata(logical_n=_HIDDEN, logical_k=intermediate)
    codebook = iq2r_initial_codebook("cuda")

    def encode(n: int, k: int, seed: int):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        weight = torch.randn((n, k), generator=generator, device="cuda") * 0.02
        importance = torch.ones((k,), dtype=torch.float32, device="cuda")
        return iq2r_encode_device(weight, importance, codebook, exponent_radius=8)

    # Three distinct experts tiled over 257; expert 256 is the shared expert.
    gate = [encode(2 * intermediate, _HIDDEN, 0x5330 + i) for i in range(3)]
    down = [encode(_HIDDEN, intermediate, 0x5340 + i) for i in range(3)]
    pick = torch.arange(_EXPERTS, device="cuda") % 3

    def stack(parts, index):
        return torch.stack([p[index].reshape(-1) for p in parts])[pick].contiguous()

    layer = SimpleNamespace(
        intermediate=intermediate,
        gate_metadata=gate_metadata,
        down_metadata=down_metadata,
        gate_data=stack(gate, 0),
        gate_auxiliary=stack(gate, 1),
        down_data=stack(down, 0),
        down_auxiliary=stack(down, 1),
    )
    layer.packed = iq2r_glm53_pack(
        layer.gate_data,
        layer.gate_auxiliary,
        layer.down_data,
        layer.down_auxiliary,
        intermediate_size=intermediate,
    )
    return layer


def _routing(tokens: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((tokens, _EXPERTS - 1), generator=generator, device="cuda")
    routed = scores.topk(_TOPK - 1, dim=-1).indices
    shared = torch.full_like(routed[:, :1], _EXPERTS - 1)
    topk_ids = torch.cat((routed, shared), dim=-1).to(torch.int32).contiguous()
    topk_weights = torch.softmax(
        torch.randn((tokens, _TOPK), generator=generator, device="cuda"), dim=-1
    ).contiguous()
    hidden = (
        torch.randn((tokens, _HIDDEN), generator=generator, device="cuda") * 0.2
    ).to(torch.bfloat16)
    return hidden, topk_weights, topk_ids


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 64, 128, 256, 300, 2048, 3000])
def test_glm53_moe_matches_generic(glm53_layer, tokens):
    layer = glm53_layer
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5300 + tokens)

    expected = torch.empty_like(hidden)
    generic_workspace = IQ2RMoeWorkspace.allocate(
        tokens,
        _TOPK,
        device="cuda",
        max_experts=_EXPERTS,
        hidden_size=_HIDDEN,
        intermediate_size=layer.intermediate,
    )
    iq2r_fused_moe_out(
        hidden,
        layer.gate_data,
        layer.gate_auxiliary,
        layer.down_data,
        layer.down_auxiliary,
        topk_weights,
        topk_ids,
        expected,
        gate_up_metadata=layer.gate_metadata,
        down_metadata=layer.down_metadata,
        gate_up_tile_n=128,
        down_tile_n=128,
        gate_up_bias=None,
        down_bias=None,
        workspace=generic_workspace,
        # GLM uses unclamped SiLU(gate) * up; the generic kernel always clamps.
        swiglu_limit=1e30,
        swiglu_alpha=1.0,
        swiglu_up_offset=0.0,
    )
    del generic_workspace

    actual = torch.empty_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    iq2r_glm53_moe_out(
        hidden, *layer.packed, topk_weights, topk_ids, actual, workspace
    )
    assert torch.isfinite(actual).all()
    relative_rmse, cosine = _relative_metrics(actual, expected)
    assert relative_rmse < 2e-3
    assert cosine > 0.9999


def test_glm53_moe_chunks_long_prefill(glm53_layer):
    layer = glm53_layer
    tokens = 600
    hidden, topk_weights, topk_ids = _routing(tokens, 0x53C0)
    whole = torch.empty_like(hidden)
    iq2r_glm53_moe_out(
        hidden,
        *layer.packed,
        topk_weights,
        topk_ids,
        whole,
        IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda"),
    )
    # A 300-token workspace runs the batch as two 300-token prefill chunks.
    chunked = torch.empty_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(300, layer.intermediate, device="cuda")
    iq2r_glm53_moe_out(
        hidden, *layer.packed, topk_weights, topk_ids, chunked, workspace
    )
    halves = torch.empty_like(hidden)
    for rows in (slice(0, 300), slice(300, 600)):
        iq2r_glm53_moe_out(
            hidden[rows],
            *layer.packed,
            topk_weights[rows],
            topk_ids[rows],
            halves[rows],
            workspace,
        )
    torch.testing.assert_close(chunked, halves, rtol=0, atol=0)
    relative_rmse, _ = _relative_metrics(chunked, whole)
    assert relative_rmse < 1e-6


@pytest.mark.parametrize("tp", [4, 8])
def test_glm53_pack_commutes_with_tp_slicing(tp):
    """Packing the full checkpoint then slicing equals slicing then packing."""
    full_intermediate = 2048
    intermediate = full_intermediate // tp
    experts = 2
    gate_metadata = IQ2RMetadata(logical_n=2 * full_intermediate, logical_k=_HIDDEN)
    down_metadata = IQ2RMetadata(logical_n=_HIDDEN, logical_k=full_intermediate)
    generator = torch.Generator(device="cuda").manual_seed(0x53D0 + tp)

    def random_bytes(columns: int):
        return torch.randint(
            0, 256, (experts, columns), generator=generator, device="cuda",
            dtype=torch.uint8,
        )  # fmt: skip

    gate = random_bytes(gate_metadata.data_bytes)
    gate_aux = random_bytes(gate_metadata.auxiliary_bytes)
    down = random_bytes(down_metadata.data_bytes)
    down_aux = random_bytes(down_metadata.auxiliary_bytes)
    # The full gate/up checkpoint has 4096 columns: a multiple of 64.
    packed = iq2r_glm53_pack(
        gate, gate_aux, down, down_aux, intermediate_size=full_intermediate
    )
    for rank in range(tp):
        gate_start, gate_len = rank * 2 * intermediate, 2 * intermediate
        down_start = rank * intermediate
        shard = iq2r_glm53_pack(
            _slice_rows(gate, gate_metadata, gate_start, gate_len),
            iq2r_slice_output_auxiliary(gate_aux, gate_metadata, gate_start, gate_len),
            iq2r_slice_input_data(down, down_metadata, down_start, intermediate),
            down_aux,
            intermediate_size=intermediate,
        )
        assert torch.equal(
            iq2r_glm53_slice_gate(packed[0], gate_start, gate_len), shard[0]
        )
        assert torch.equal(
            iq2r_slice_output_auxiliary(packed[1], gate_metadata, gate_start, gate_len),
            shard[1],
        )
        assert torch.equal(
            iq2r_slice_input_data(packed[2], down_metadata, down_start, intermediate),
            shard[2],
        )
        assert torch.equal(packed[3], shard[3])


def _atom_views(group: torch.Tensor, block: int):
    """(8-byte record, metadata byte) views of one N=16 block in a K=128 group."""
    triplet, atom = divmod(block % 6, 3)
    base = triplet * 1792
    third = group[..., base + 1024 : base + 1792].unflatten(-1, (64, 12))
    if atom < 2:
        pair = group[..., base : base + 1024].unflatten(-1, (64, 16))
        record = pair[..., atom * 8 : atom * 8 + 8]
    else:
        record = third[..., :8]
    return record, third[..., 8 + atom]


def _slice_rows(data, metadata, start, length):
    """Canonical output-column slice, regrouping blocks into six-block groups."""
    target = IQ2RMetadata(logical_n=length, logical_k=metadata.logical_k)
    experts, k_tiles = data.shape[0], metadata.k_tiles
    source = data.view(experts, -1, k_tiles, 3584)
    output = torch.zeros(
        (experts, target.data_bytes), dtype=torch.uint8, device=data.device
    ).view(experts, -1, k_tiles, 3584)
    for block in range(length // 16):
        source_block = start // 16 + block
        src = _atom_views(source[:, source_block // 6], source_block)
        dst = _atom_views(output[:, block // 6], block)
        dst[0].copy_(src[0])
        dst[1].copy_(src[1])
    return output.view(experts, -1)


def test_glm53_tuned_config():
    for intermediate in (256, 512):
        assert iq2r_glm53_config(1, intermediate).down_kernel == "route9"
        assert iq2r_glm53_config(64, intermediate).down_kernel == "packed"
        assert iq2r_glm53_config(4096, intermediate).down_chunks == 2
        # Untuned token counts fall back to the default heuristic.
        assert iq2r_glm53_config(77, intermediate) == _default_config(77, intermediate)
    assert iq2r_glm53_config(128, 256).gate_kernel == "nobarrier"
    assert iq2r_glm53_config(256, 256).down_kernel == "ordered"
