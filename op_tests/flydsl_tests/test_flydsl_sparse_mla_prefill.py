# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and CUDA-graph coverage for FP8 FlyDSL sparse-MLA prefill."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx

_NUM_HEADS = 16
_V_HEAD_DIM = 512
_ROPE_HEAD_DIM = 64
_HEAD_DIM = _V_HEAD_DIM + _ROPE_HEAD_DIM
_TOPK = 2048
_SOFTMAX_SCALE = 0.0625
_MIN_SNR_DB = 35.3


def _require_gfx950_flydsl():
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        pytest.skip("requires gfx950")


def _make_case(num_tokens, num_pages=4096, seed=1234):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(
        num_tokens,
        _NUM_HEADS,
        _HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(num_pages, _HEAD_DIM, generator=generator, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    indices = torch.randint(
        0,
        num_pages,
        (num_tokens, _TOPK),
        generator=generator,
        dtype=torch.int32,
    )
    indices[:, -64:] = -1
    q = q.cuda()
    return (
        q[:, :, :_V_HEAD_DIM],
        q[:, :, _V_HEAD_DIM:],
        kv.cuda().contiguous(),
        indices.cuda(),
    )


def _reference(q_nope, q_rope, kv, indices, token_ids):
    query = torch.cat((q_nope[token_ids], q_rope[token_ids]), dim=-1).double()
    rows = indices[token_ids].long()
    valid = rows >= 0
    gathered = kv.double()[rows.clamp(min=0)]
    scores = torch.einsum("nhd,nkd->nhk", query, gathered) * _SOFTMAX_SCALE
    scores.masked_fill_(~valid[:, None, :], -float("inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("nhk,nkv->nhv", probabilities, gathered[:, :, :_V_HEAD_DIM])


def _snr_db(actual, expected):
    actual = actual.double()
    expected = expected.double()
    signal = expected.square().sum()
    noise = (actual - expected).square().sum()
    return float(10 * torch.log10(signal / noise))


def test_flydsl_sparse_mla_prefill_correctness():
    _require_gfx950_flydsl()
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    q_nope, q_rope, kv, indices = _make_case(512)
    assert not q_nope.is_contiguous() and not q_rope.is_contiguous()
    assert q_nope.stride() == q_rope.stride() == (16 * 576, 576, 1)
    actual = flydsl_sparse_mla_prefill(q_nope, q_rope, kv, indices, _SOFTMAX_SCALE)
    torch.cuda.synchronize()
    token_ids = torch.linspace(0, 511, 16, device="cuda", dtype=torch.long)
    expected = _reference(q_nope, q_rope, kv, indices, token_ids)
    snr = _snr_db(actual[token_ids], expected)
    assert (
        snr >= _MIN_SNR_DB
    ), f"SNR {snr:.6f} dB is below the {_MIN_SNR_DB} dB no-regression gate"


def test_flydsl_sparse_mla_prefill_cuda_graph_replay():
    _require_gfx950_flydsl()
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    q_nope, q_rope, kv, indices = _make_case(64)
    output = torch.empty(
        (64, _NUM_HEADS, _V_HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    flydsl_sparse_mla_prefill(q_nope, q_rope, kv, indices, _SOFTMAX_SCALE, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_sparse_mla_prefill(
            q_nope, q_rope, kv, indices, _SOFTMAX_SCALE, out=output
        )

    next_q_nope, next_q_rope, _, _ = _make_case(64, seed=4321)
    q_nope.copy_(next_q_nope)
    q_rope.copy_(next_q_rope)
    expected = torch.empty_like(output)
    flydsl_sparse_mla_prefill(q_nope, q_rope, kv, indices, _SOFTMAX_SCALE, out=expected)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_prefill_rejects_kv_above_the_32bit_extent():
    """A KV buffer past 2 GiB must raise, not fault the GPU.

    The launcher passes ``kv.view(torch.int8).reshape(-1)`` and that extent is
    packed as a 32-bit int, so an 8 GiB pool raised
    ``'i' format requires -2147483648 <= number <= 2147483647`` out of the shim
    and the launch that followed reported ``Memory access fault by GPU node-N``.
    Decode has no such limit (it passes kv as a bare pointer), so the asymmetry
    is easy to trip over. Allocates just over 2 GiB, and skips if that does not
    fit.
    """
    import pytest
    import torch

    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    pages = (2**31) // 576 + 1024  # just over the 32-bit byte extent
    need = pages * 576 + (1 << 28)
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip("not enough free device memory for a 2 GiB KV buffer")

    kv = torch.zeros(pages, 576, dtype=torch.float8_e4m3fn, device="cuda")
    tokens = 256
    q_nope = torch.zeros(tokens, 16, 512, dtype=torch.float8_e4m3fn, device="cuda")
    q_rope = torch.zeros(tokens, 16, 64, dtype=torch.float8_e4m3fn, device="cuda")
    indices = torch.zeros(tokens, 2048, dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="32-bit extent"):
        flydsl_sparse_mla_prefill(
            q_nope=q_nope,
            q_rope=q_rope,
            kv=kv,
            indices=indices,
            softmax_scale=1.0,
        )
