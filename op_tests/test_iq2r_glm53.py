# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R packed MoE (TP4/TP8, gfx950) against a dense Torch reference."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from aiter.iq2r_glm53 import (
    IQ2RGlm53Config,
    IQ2RGlm53Workspace,
    _default_config,
    iq2r_glm53_config,
    iq2r_glm53_moe_out,
    iq2r_glm53_pack,
    iq2r_glm53_slice_gate,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.iq2r import iq2r_encode_device, iq2r_materialize_device
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


def _mxfp8(values: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize to MXFP8 E4M3 with one E8M0 scale per 32 values."""
    blocks = values.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, -127))
    scale = torch.exp2(exponent)
    quantized = (blocks / scale).to(torch.float8_e4m3fn).float() * scale
    return quantized.flatten(-2)


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
        pick=pick,
        gate_weights=[
            iq2r_materialize_device(d.view(1, -1), a.view(1, -1), gate_metadata)
            for d, a in gate
        ],
        down_weights=[
            iq2r_materialize_device(d.view(1, -1), a.view(1, -1), down_metadata)
            for d, a in down
        ],
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


def _reference_moe(layer, hidden, topk_weights, topk_ids):
    """MXFP8 activations, SiLU(gate) * up in BF16, BF16 route outputs."""
    a1 = _mxfp8(hidden)
    output = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    route_expert = layer.pick[topk_ids.long()]
    for expert, (gate, down) in enumerate(zip(layer.gate_weights, layer.down_weights)):
        weight = (topk_weights * (route_expert == expert)).sum(dim=-1, keepdim=True)
        gate_up = a1 @ gate.T
        act = (torch.nn.functional.silu(gate_up[:, 0::2]) * gate_up[:, 1::2]).to(
            torch.bfloat16
        )
        route = (_mxfp8(act) @ down.T).to(torch.bfloat16)
        output += weight * route.float()
    return output.to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 24, 64, 100, 128, 256, 2048, 3000])
def test_glm53_moe_matches_reference(glm53_layer, tokens):
    layer = glm53_layer
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5300 + tokens)
    expected = _reference_moe(layer, hidden, topk_weights, topk_ids)

    actual = torch.empty_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    iq2r_glm53_moe_out(hidden, *layer.packed, topk_weights, topk_ids, actual, workspace)
    assert torch.isfinite(actual).all()
    relative_rmse, cosine = _relative_metrics(actual, expected)
    # The intermediate activation is requantized to FP8 E4M3, so small GEMM
    # differences flip a few elements per row to the neighbouring FP8 value
    # (measured: relative RMSE <= 1.7e-2, cosine >= 0.99985).
    assert relative_rmse < 3e-2
    assert cosine > 0.9995


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
    """Tuned rows are used as written; other token counts get the defaults."""
    table = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_IQ2R_GLM53_FILE)
    rows = table[(table.gfx == get_gfx_runtime()) & (table.cu_num == get_cu_num())]
    for row in rows.itertuples():
        assert iq2r_glm53_config(row.token, row.inter_dim) == IQ2RGlm53Config(
            row.gate_kernel,
            row.gate_grid,
            row.down_kernel,
            row.down_grid,
            row.down_chunks,
        )
    for intermediate in (256, 512):
        assert iq2r_glm53_config(77, intermediate) == _default_config(77)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
