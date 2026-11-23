# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R-v2 packed MoE (TP4/TP8, 512- and 1,024-entry codebooks, gfx950)
against a dense Torch reference.

``python3 op_tests/test_iq2r_glm53.py`` times the tuned and default launch
configs over a token sweep, prints a summary table, then runs the tests.
"""

import argparse
import functools
import itertools
from dataclasses import astuple
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import aiter
from aiter.iq2r_glm53 import (
    DOWN_KERNELS,
    FUSED_SHARED_TOKENS,
    MAX_CHUNK_TOKENS,
    MAX_DECODE_TOKENS,
    IQ2RGlm53Config,
    IQ2RGlm53Workspace,
    _config_problem,
    _default_config,
    _mxfp8_quantize,
    _tuned_table,
    iq2r_glm53_config,
    iq2r_glm53_dequantize_shared,
    iq2r_glm53_moe_out,
    iq2r_glm53_pack,
    iq2r_glm53_pack_shared,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_gfx_runtime
from aiter.ops.iq2r_v2 import (
    IQ2R_V2_CODEBOOK_MAX_BYTE,
    IQ2R_V2_CODEBOOK_MIN_BYTE,
    IQ2R_V2_LEVELS_C4,
    iq2r_v2_check_codebook,
    iq2r_v2_codebook_copies,
    iq2r_v2_encode,
    iq2r_v2_gains,
    iq2r_v2_learn_codebook,
    iq2r_v2_materialize,
    iq2r_v2_seed_codebook,
    iq2r_v2_write_auxiliary,
    iq2r_v2_write_data,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]
LEVELS = IQ2R_V2_LEVELS_C4

_EXPERTS = 256
_TOPK = 8
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


def _interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack((a, b), dim=1).reshape(-1, *a.shape[1:])


# A gain far from 1 makes a dropped or misapplied output gain fail the tests.
_TEST_GAIN = 0.8


def _encode_gate_up(gate, up, book, importance):
    """Canonical IQ2R-v2 gate/up: rows interleaved, one shared codebook."""
    weight = _interleave(gate, up)
    enc = iq2r_v2_encode(weight, importance, book, LEVELS)
    gains = iq2r_v2_gains(weight, enc.reconstruction, _TEST_GAIN, interleaved=True)
    data = iq2r_v2_write_data(
        enc.indices, enc.negative, enc.levels, LEVELS, book.shape[0]
    )
    return data, iq2r_v2_write_auxiliary(book[None], enc.d, gains)


def _encode_down(weight, book, importance):
    enc = iq2r_v2_encode(weight, importance, book, LEVELS)
    gains = iq2r_v2_gains(weight, enc.reconstruction, _TEST_GAIN)
    data = iq2r_v2_write_data(
        enc.indices, enc.negative, enc.levels, LEVELS, book.shape[0]
    )
    return data, iq2r_v2_write_auxiliary(book[None], enc.d, gains)


@functools.lru_cache(maxsize=1)
def _glm53_layer(intermediate: int, entries: int = 512):
    seed = iq2r_v2_seed_codebook("cuda", entries)
    books = (seed, (seed * 1.0625).to(torch.float8_e4m3fn).float())

    def weight(n: int, k: int, s: int):
        generator = torch.Generator(device="cuda").manual_seed(s)
        return torch.randn((n, k), generator=generator, device="cuda") * 0.02

    ones_h = torch.ones(_HIDDEN, device="cuda")
    ones_i = torch.ones(intermediate, device="cuda")
    gate = [
        _encode_gate_up(
            weight(intermediate, _HIDDEN, 0x5330 + i),
            weight(intermediate, _HIDDEN, 0x5350 + i),
            books[1],
            ones_h,
        )
        for i in range(3)
    ]
    down = [
        _encode_down(weight(_HIDDEN, intermediate, 0x5340 + i), books[i % 2], ones_i)
        for i in range(3)
    ]
    pick = torch.arange(_EXPERTS, device="cuda") % 3

    def stack(parts, index):
        return torch.stack([p[index].reshape(-1) for p in parts])[pick].contiguous()

    def dense(data, aux, n, k):
        return iq2r_v2_materialize(data, aux, n, k, LEVELS, codebook_entries=entries)

    layer = SimpleNamespace(
        intermediate=intermediate,
        entries=entries,
        pick=pick,
        gate_weights=[dense(d, a, 2 * intermediate, _HIDDEN) for d, a in gate],
        down_weights=[dense(d, a, _HIDDEN, intermediate) for d, a in down],
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


@pytest.fixture(
    scope="module",
    params=[(256, 512), (512, 512), (256, 1024), (512, 1024)],
    ids=["tp8", "tp4", "tp8-book1024", "tp4-book1024"],
)
def glm53_layer(request):
    return _glm53_layer(*request.param)


def _routing(tokens: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((tokens, _EXPERTS), generator=generator, device="cuda")
    topk_ids = scores.topk(_TOPK, dim=-1).indices.to(torch.int32).contiguous()
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


def _check_moe(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.isfinite(actual).all()
    relative_rmse, cosine = _relative_metrics(actual, expected)
    # The intermediate activation is requantized to FP8 E4M3, so small GEMM
    # differences flip a few elements per row to the neighbouring FP8 value
    # (measured: relative RMSE <= 1.7e-2, cosine >= 0.99985).
    assert relative_rmse < 3e-2
    assert cosine > 0.9995


@pytest.mark.parametrize(
    "tokens",
    [1, 2, 4, 8, 16, 24, 32, 64, 100, 128, 256, 512, 1024, 2048, 3000, 4096],
)
def test_glm53_moe_matches_reference(glm53_layer, tokens):
    layer = glm53_layer
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5300 + tokens)
    expected = _reference_moe(layer, hidden, topk_weights, topk_ids)

    actual = torch.empty_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    iq2r_glm53_moe_out(hidden, *layer.packed, topk_weights, topk_ids, actual, workspace)
    _check_moe(actual, expected)


def _one_hot_rows(rows: int, k: int, columns: torch.Tensor):
    """FP8 activations with a single 1.0 per row (at ``columns``), E8M0 scales 1."""
    activations = torch.zeros((rows, k), device="cuda")
    activations[torch.arange(rows, device="cuda"), columns] = 1.0
    scales = torch.full((rows, k // 32), 127, dtype=torch.uint8, device="cuda")
    return activations.to(torch.float8_e4m3fn), scales


def _decoded_columns(layer, experts: torch.Tensor, columns: torch.Tensor):
    """BF16 of gain * W[:, k]: what a down kernel writes for a one-hot row.

    The single product of the one-hot row is exact in the MFMA's FP32
    accumulator, so the kernel's decode must match the reference bit for bit.
    """
    dense = torch.stack(layer.down_weights)[layer.pick[experts.long()]]
    return dense[torch.arange(len(columns), device="cuda"), :, columns].to(
        torch.bfloat16
    )


@pytest.mark.parametrize("kernel", ["m16", "m16_2wg", "m32", "per_token", "prefill"])
def test_glm53_down_decode_bit_exact(glm53_layer, kernel):
    """Every down kernel decodes the packed weights bit-exactly (vs iq2r_v2_materialize)."""
    from aiter.ops.iq2r import (
        iq2r_glm53_down_out,
        iq2r_glm53_down_per_token_out,
        iq2r_glm53_down_reduce_out,
    )

    layer = glm53_layer
    k = layer.intermediate
    _, _, data, aux = layer.packed
    generator = torch.Generator(device="cuda").manual_seed(0x53D0)
    if kernel in DOWN_KERNELS:
        if kernel == "m32" and k != 256:
            pytest.skip("the m32 down kernel is TP8 only")
        if kernel == "m16_2wg" and k != 512:
            pytest.skip("the m16_2wg down kernel is TP4 only")
        tokens = 64
        routes = tokens * 9
        # Rows 32 at a time per expert; row r reads column 37 r mod K, so
        # every column of every expert pattern is covered.
        columns = (torch.arange(routes, device="cuda") * 37) % k
        experts = (torch.arange(routes, device="cuda") // 32) % 3
        activations, scales = _one_hot_rows(routes, k, columns)
        tasks = torch.zeros(
            (routes // 32 + _EXPERTS, 3), dtype=torch.int32, device="cuda"
        )
        for task in range(routes // 32):
            tasks[task] = torch.tensor([task * 32, 32, task % 3], dtype=torch.int32)
        count = torch.tensor([routes // 32], dtype=torch.int32, device="cuda")
        output = torch.full(
            (routes, _HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        none = torch.empty(0, dtype=torch.uint8, device="cuda")
        iq2r_glm53_down_out(
            activations, scales, data, aux, tasks, count, output,
            DOWN_KERNELS[kernel], 4, none, none, none, none,
        )  # fmt: skip
        assert torch.equal(output, _decoded_columns(layer, experts, columns))
        return

    # per_token and prefill sum the eight routes of a token: a one-hot route
    # weight makes the token's output exactly one route's.
    for tokens in (1, 2, 4) if kernel == "per_token" else (64,):
        routes = tokens * 9
        for trial in range(4):
            columns = torch.randint(0, k, (routes,), generator=generator, device="cuda")
            route_experts = (torch.arange(routes, device="cuda") // 64) % 3
            if kernel == "per_token":
                route_experts = torch.randint(
                    0, 3, (routes,), generator=generator, device="cuda"
                )
            activations, scales = _one_hot_rows(routes, k, columns)
            chosen = torch.randint(
                0, _TOPK, (tokens,), generator=generator, device="cuda"
            )
            weights = (
                torch.nn.functional.one_hot(chosen.long(), _TOPK).float().contiguous()
            )
            scatter = torch.arange(routes, dtype=torch.int32, device="cuda")
            picked = torch.arange(tokens, device="cuda") * 9 + chosen
            output = torch.full(
                (tokens, _HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            if kernel == "per_token":
                ids = (
                    route_experts.reshape(tokens, 9)[:, :_TOPK]
                    .to(torch.int32)
                    .contiguous()
                )
                none = torch.empty(0, dtype=torch.uint8, device="cuda")
                iq2r_glm53_down_per_token_out(
                    activations, scales, data, aux, ids, scatter, weights, output,
                    none, none, none,
                )  # fmt: skip
            else:
                tasks = torch.zeros(
                    (routes // 64 + 1 + _EXPERTS, 3), dtype=torch.int32, device="cuda"
                )
                n_tasks = (routes + 63) // 64
                for task in range(n_tasks):
                    begin = task * 64
                    tasks[task] = torch.tensor(
                        [begin, min(64, routes - begin), task % 3], dtype=torch.int32
                    )
                count = torch.tensor([n_tasks], dtype=torch.int32, device="cuda")
                route_output = torch.empty(
                    (routes, _HIDDEN), dtype=torch.bfloat16, device="cuda"
                )
                iq2r_glm53_down_reduce_out(
                    activations, scales, data, aux, tasks, count, route_output,
                    scatter, weights, output, 1 + trial % 2,
                )  # fmt: skip
            expected = _decoded_columns(layer, route_experts[picked], columns[picked])
            assert torch.equal(output, expected), (tokens, trial)


# Each gate and down kernel at grids and chunk counts the shipped table and
# the defaults do not use. Configs a token count cannot run are skipped.
_EXPLICIT_CONFIGS = [
    IQ2RGlm53Config("decode", 4, "per_token", 4, 1),
    IQ2RGlm53Config("decode", 1, "m16", 2, 1),
    IQ2RGlm53Config("decode", 2, "m16", 16, 1),
    IQ2RGlm53Config("decode", 2, "m16_2wg", 4, 1),
    IQ2RGlm53Config("decode", 4, "m32", 8, 1),
    IQ2RGlm53Config("decode", 8, "m32", 1, 1),
    IQ2RGlm53Config("prefill", 1, "prefill", 0, 2),
    IQ2RGlm53Config("prefill", 8, "prefill", 0, 1),
]


@pytest.mark.parametrize(
    "config", _EXPLICIT_CONFIGS, ids=lambda c: "-".join(map(str, astuple(c)))
)
@pytest.mark.parametrize("tokens", [4, 48, 1024])
def test_glm53_moe_explicit_configs(glm53_layer, tokens, config):
    layer = glm53_layer
    problem = _config_problem(config, tokens, layer.intermediate)
    if problem:
        pytest.skip(problem)
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5390 + tokens)
    expected = _reference_moe(layer, hidden, topk_weights, topk_ids)

    actual = torch.empty_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    iq2r_glm53_moe_out(
        hidden, *layer.packed, topk_weights, topk_ids, actual, workspace, config
    )
    _check_moe(actual, expected)


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


@functools.lru_cache(maxsize=2)
def _shared_expert(intermediate: int):
    """Random shared expert, its packed MXFP8 form and its exact FP32 weights."""
    generator = torch.Generator(device="cuda").manual_seed(0x53F0 + intermediate)
    gate, up = (
        torch.randn((intermediate, _HIDDEN), generator=generator, device="cuda") * 0.02
        for _ in range(2)
    )
    down = (
        torch.randn((_HIDDEN, intermediate), generator=generator, device="cuda") * 0.02
    )
    exact = [
        iq2r_glm53_dequantize_shared(*_mxfp8_quantize(w)) for w in (gate, up, down)
    ]
    return iq2r_glm53_pack_shared(gate, up, down), exact


def _reference_shared(exact, hidden):
    """MXFP8 activations, SiLU(gate) * up in BF16, BF16 output (weight 1)."""
    gate, up, down = exact
    a1 = _mxfp8(hidden)
    act = (torch.nn.functional.silu(a1 @ gate.T) * (a1 @ up.T)).to(torch.bfloat16)
    return (_mxfp8(act) @ down.T).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 8, 16, 33, 64, 100, 256, 1000, 1024])
def test_glm53_moe_fused_shared_expert(glm53_layer, tokens):
    """The MXFP8 shared expert as route slot 8: decode paths, deterministic."""
    layer = glm53_layer
    shared, exact = _shared_expert(layer.intermediate)
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5370 + tokens)
    expected = (
        _reference_moe(layer, hidden, topk_weights, topk_ids).float()
        + _reference_shared(exact, hidden).float()
    ).to(torch.bfloat16)
    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    actual, again = torch.empty_like(hidden), torch.empty_like(hidden)
    for out in (actual, again):
        iq2r_glm53_moe_out(
            hidden, *layer.packed, topk_weights, topk_ids, out, workspace, shared=shared
        )
    _check_moe(actual, expected)
    assert torch.equal(actual, again)


def test_glm53_moe_fused_shared_expert_limits(glm53_layer):
    layer = glm53_layer
    shared, _ = _shared_expert(layer.intermediate)
    hidden, topk_weights, topk_ids = _routing(1025, 0x53F8)
    workspace = IQ2RGlm53Workspace.allocate(1025, layer.intermediate, device="cuda")
    output = torch.empty_like(hidden)
    with pytest.raises(ValueError, match="up to 1024 tokens"):
        iq2r_glm53_moe_out(
            hidden,
            *layer.packed,
            topk_weights,
            topk_ids,
            output,
            workspace,
            shared=shared,
        )
    prefill = IQ2RGlm53Config("prefill", 2, "prefill", 0, 1)
    with pytest.raises(ValueError, match="decode kernels only"):
        iq2r_glm53_moe_out(
            hidden[:512], *layer.packed, topk_weights[:512], topk_ids[:512],
            output[:512], workspace, prefill, shared=shared,
        )  # fmt: skip


@pytest.mark.parametrize("tokens", [1, 3, 4, 64, 1024])
def test_glm53_moe_drops_invalid_routes(glm53_layer, tokens):
    """Routes with ids outside 0..255 contribute nothing on every path."""
    layer = glm53_layer
    hidden, topk_weights, topk_ids = _routing(tokens, 0x53E0 + tokens)
    topk_ids[:, 0] = -1
    topk_ids[1::2, 3] = _EXPERTS
    valid = (topk_ids >= 0) & (topk_ids < _EXPERTS)
    expected = _reference_moe(
        layer, hidden, topk_weights * valid, topk_ids.clamp(0, _EXPERTS - 1)
    )

    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    workspace.route_output.fill_(float("nan"))  # a dropped route must not be read
    actual = torch.empty_like(hidden)
    iq2r_glm53_moe_out(hidden, *layer.packed, topk_weights, topk_ids, actual, workspace)
    _check_moe(actual, expected)


def test_glm53_moe_rejects_unfit_configs(glm53_layer):
    layer = glm53_layer
    hidden, topk_weights, topk_ids = _routing(9, 0x53B0)
    output = torch.zeros_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(8, layer.intermediate, device="cuda")

    def run(tokens, config):
        rows = slice(0, tokens)
        iq2r_glm53_moe_out(
            hidden[rows], *layer.packed, topk_weights[rows], topk_ids[rows],
            output[rows], workspace, config,
        )  # fmt: skip

    with pytest.raises(ValueError, match="M1 needs down per_token"):
        run(1, IQ2RGlm53Config("decode", 2, "m16", 4, 1))
    with pytest.raises(ValueError, match="unknown decode down kernel"):
        run(8, IQ2RGlm53Config("decode", 2, "prefill", 0, 1))
    with pytest.raises(ValueError, match="per_token covers"):
        run(3, IQ2RGlm53Config("decode", 2, "per_token", 4, 1))
    with pytest.raises(ValueError, match="unknown gate kernel"):
        run(8, IQ2RGlm53Config("bogus", 2, "m16", 4, 1))
    with pytest.raises(ValueError, match="unknown decode down kernel"):
        run(8, IQ2RGlm53Config("decode", 2, "packed", 4, 1))
    with pytest.raises(ValueError, match="top-8"):  # [tokens, 9] routing
        iq2r_glm53_moe_out(
            hidden[:8], *layer.packed, torch.ones((8, 9), device="cuda"),
            torch.zeros((8, 9), dtype=torch.int32, device="cuda"), output[:8],
            workspace,
        )  # fmt: skip
    # The 1-token tail of a chunked batch cannot reuse an M8 config.
    with pytest.raises(ValueError, match="exceed"):
        run(9, _default_config(8))
    assert not output.any()


def test_glm53_moe_rejects_other_archs(glm53_layer, monkeypatch):
    import aiter.iq2r_glm53 as glm53

    layer = glm53_layer
    hidden, topk_weights, topk_ids = _routing(8, 0x53A0)
    output = torch.zeros_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(8, layer.intermediate, device="cuda")
    monkeypatch.setattr(glm53, "get_gfx", lambda: "gfx942")
    with pytest.raises(RuntimeError, match="requires gfx950"):
        iq2r_glm53_moe_out(
            hidden, *layer.packed, topk_weights, topk_ids, output, workspace
        )
    assert not output.any()


# Copy-formation deltas of the kernels (iq2r_v2_copy_words): copy c of
# a codebook byte b is b + delta_c[b & 7].
_FORMATION_DELTAS = (
    (0,) * 8,
    (2,) * 8,
    (3, 4, 4, 5, 4, 4, 4, 4),
    (5, 6, 6, 6, 6, 6, 6, 6),
)


def test_glm53_copy_formation_matches_codebook_copies():
    """The kernels form copies 1-3 from copy 0 with integer adds; on every byte
    the encoder may emit they equal the packed copies E4M3(codebook * m[c])."""
    allowed = torch.arange(
        IQ2R_V2_CODEBOOK_MIN_BYTE, IQ2R_V2_CODEBOOK_MAX_BYTE + 1, dtype=torch.uint8
    )
    deltas = torch.tensor(_FORMATION_DELTAS, dtype=torch.int32)
    formed = (allowed.int()[None] + deltas[:, allowed.long() & 7]).to(torch.uint8)
    book = allowed.view(torch.float8_e4m3fn).float()
    copies = iq2r_v2_codebook_copies(book, LEVELS)
    assert torch.equal(formed, copies.to(torch.float8_e4m3fn).view(torch.uint8))
    # The range is tight: the subnormal byte below it already forms wrongly.
    below = torch.tensor([IQ2R_V2_CODEBOOK_MIN_BYTE - 1], dtype=torch.uint8)
    copies = iq2r_v2_codebook_copies(below.view(torch.float8_e4m3fn).float(), LEVELS)
    formed = (below.int()[None] + deltas[:, below.long() & 7]).to(torch.uint8)
    assert not torch.equal(formed, copies.to(torch.float8_e4m3fn).view(torch.uint8))


def test_glm53_learned_codebook_stays_in_kernel_range():
    """Zero and tiny weights pull k-means centroids towards 0 (below 2^-6, a
    subnormal byte the kernels' copy formation gets wrong); the learner clamps."""
    generator = torch.Generator(device="cuda").manual_seed(0x53F0)
    weight = torch.randn((256, 1024), generator=generator, device="cuda") * 0.02
    weight[:, :512] = 0.0
    weight[:64, 512:768] *= 1e-4
    book, _ = iq2r_v2_learn_codebook(
        weight, torch.ones(1024, device="cuda"), LEVELS, rounds=3, sample_vectors=8192
    )
    raw = book.to(torch.float8_e4m3fn).view(torch.uint8)
    assert int(raw.min()) >= IQ2R_V2_CODEBOOK_MIN_BYTE
    assert int(raw.max()) <= IQ2R_V2_CODEBOOK_MAX_BYTE
    iq2r_v2_check_codebook(book)


@pytest.mark.parametrize("byte", [0x00, 0x05, 0x79, 0x80])
def test_glm53_pack_rejects_codebook_outside_kernel_range(glm53_layer, byte):
    layer = glm53_layer
    for which in ("gate", "down"):
        aux = {
            "gate": layer.gate_auxiliary.clone(),
            "down": layer.down_auxiliary.clone(),
        }
        aux[which][0, 17] = byte  # one codebook byte of expert 0
        with pytest.raises(ValueError, match="codebook bytes"):
            iq2r_glm53_pack(
                layer.gate_data,
                aux["gate"],
                layer.down_data,
                aux["down"],
                intermediate_size=layer.intermediate,
            )


def test_glm53_tuned_config():
    """Tuned rows are used as written; other token counts round up to a row
    that can run them."""
    table = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_IQ2R_GLM53_FILE)
    rows = table[(table.gfx == get_gfx_runtime()) & (table.cu_num == get_cu_num())]
    for row in rows.itertuples():
        config = iq2r_glm53_config(
            row.token, row.inter_dim, row.codebook, bool(row.shared)
        )
        assert config == IQ2RGlm53Config(
            row.gate_kernel,
            row.gate_grid,
            row.down_kernel,
            row.down_grid,
            row.down_chunks,
        )
    for intermediate in (256, 512):
        lookup = functools.partial(iq2r_glm53_config, intermediate_size=intermediate)
        assert lookup(77) == lookup(128)
        # per_token cannot run M3, so M3 rounds up past a per-token M4 row.
        m4 = lookup(4)
        assert lookup(3) == (lookup(8) if m4.down_kernel == "per_token" else m4)
        for tokens in range(1, MAX_CHUNK_TOKENS + 1):
            for config in (lookup(tokens), _default_config(tokens)):
                assert _config_problem(config, tokens, intermediate) is None
        # With the fused shared expert: decode kernels only.
        for tokens in range(1, MAX_DECODE_TOKENS + 1):
            config = lookup(tokens, codebook_entries=1024, shared=True)
            assert config.gate_kernel != "prefill" or tokens > FUSED_SHARED_TOKENS
            assert _config_problem(config, tokens, intermediate) is None


def test_glm53_tuned_table_checks(tmp_path):
    table = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_IQ2R_GLM53_FILE)
    shipped = _tuned_table(AITER_CONFIGS.AITER_CONFIG_IQ2R_GLM53_FILE)

    # Rows of another model family in the same file are ignored.
    path = tmp_path / "families.csv"
    other = table.assign(expert=129, down_kernel="other")
    pd.concat([table, other]).to_csv(path, index=False)
    assert _tuned_table(str(path)) == shipped

    path = tmp_path / "duplicate.csv"
    pd.concat([table, table.head(1).assign(us=1.0)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than one"):
        _tuned_table(str(path))

    path = tmp_path / "unfit.csv"
    table.assign(down_kernel="m16").to_csv(path, index=False)
    with pytest.raises(ValueError, match="M1 needs down per_token"):
        _tuned_table(str(path))


@benchmark()
def test_glm53_moe_perf(tokens: int = 8, inter_dim: int = 256):
    """Time the tuned and the default launch config against the reference."""
    layer = _glm53_layer(inter_dim)
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5300 + tokens)
    expected = _reference_moe(layer, hidden, topk_weights, topk_ids).float()
    workspace = IQ2RGlm53Workspace.allocate(tokens, inter_dim, device="cuda")
    candidates = {
        "tuned": iq2r_glm53_config(tokens, inter_dim),
        "default": _default_config(tokens),
    }

    # Gate/up and down GEMMs over every route; each distinct expert's packed
    # weights are read once.
    flops = 6 * tokens * _TOPK * _HIDDEN * inter_dim
    expert_bytes = sum(t.nbytes for t in layer.packed) // _EXPERTS
    nbytes = topk_ids.unique().numel() * expert_bytes + 2 * hidden.nbytes
    # Outputs are ~1e-2, so a fixed 1e-2 atol would accept anything.
    atol = 0.1 * expected.square().mean().sqrt().item()

    ret = {"gfx": get_gfx()}
    for name, config in candidates.items():
        output = torch.empty_like(hidden)
        # run_perftest rotates copies of the tensor arguments, weights included,
        # so each call reads cold weights as a serving step does across layers.
        # Serving shares one workspace between layers; so do the copies here.
        # The tuner times graph replays; here us is the summed kernel time.
        moe = functools.partial(iq2r_glm53_moe_out, workspace=workspace, config=config)
        _, us = run_perftest(moe, hidden, *layer.packed, topk_weights, topk_ids, output)
        _check_moe(output, expected)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            expected,
            output.float(),
            rtol=5e-2,
            atol=atol,
            msg=f"{name} M={tokens} I={inter_dim}: ",
        )
    return ret


# main() runs the sweep; pytest collects only the correctness tests above.
test_glm53_moe_perf.__test__ = False


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("GLM-5.3 IQ2R MoE unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Time the GLM-5.3 IQ2R-v2 MoE (hidden 6144, 256 experts, top-8)",
    )
    parser.add_argument(
        "-t",
        "--tokenNum",
        type=int,
        nargs="*",
        default=[2**i for i in range(13)],
        help="Token counts. Default: 1, 2, 4, ..., 4096.",
    )
    parser.add_argument(
        "-i",
        "--inter_dim",
        type=int,
        nargs="*",
        default=[256, 512],
        choices=[256, 512],
        help="Expert intermediate size per rank: 256 (TP8), 512 (TP4).",
    )
    args = parser.parse_args()

    rows = [
        test_glm53_moe_perf(tokens, inter_dim)
        for inter_dim, tokens in itertools.product(args.inter_dim, args.tokenNum)
    ]
    aiter.logger.info(
        "GLM-5.3 IQ2R-v2 MoE summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    pytest_status = pytest.main([__file__, "-v"])
    if pytest_status != pytest.ExitCode.OK:
        raise SystemExit(pytest_status)


if __name__ == "__main__":
    main()
