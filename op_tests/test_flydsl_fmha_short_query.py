# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Short noncausal cached-prefix split-K, including packed and empty entries."""

from itertools import accumulate

import pytest
import torch

from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
    flydsl_flash_attn_fp8_func,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires gfx950",
)


@pytest.fixture(autouse=True)
def _precision(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(42)


def _inputs(q_shape, kv_shape):
    tensors, scales = [], []
    for shape in (q_shape, kv_shape, (*kv_shape[:-1], 128)):
        value = torch.randn(shape, dtype=torch.float32, device="cuda")
        scale = value.abs().amax().clamp_min(1e-6).reshape(1) / 448
        tensors.append((value / scale).to(torch.float8_e4m3fn))
        scales.append(scale)
    return tensors, dict(zip(("q_descale", "k_descale", "v_descale"), scales))


def _reference(q, k, v, descales, scale):
    q, k, v = (
        (t.float() * descales[name]).transpose(0, 1)
        for t, name in zip((q, k, v), ("q_descale", "k_descale", "v_descale"))
    )
    scores = (q @ k.transpose(-1, -2)) * (192**-0.5 if scale is None else scale)
    return (scores.softmax(-1) @ v).transpose(0, 1), scores.logsumexp(-1)


def _check(out, lse, ref, ref_lse):
    assert torch.isfinite(out).all()
    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    rmse = (out.float() - ref).square().mean().sqrt()
    assert (rmse / ref.square().mean().sqrt().clamp_min(1e-12)).item() < 0.04
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "q_len,kv_len",
    [(1, 257), (70, 16384), (127, 11008), (129, 257), (255, 513), (383, 11008)],
)
@pytest.mark.parametrize("splits", [2, 4, 16, None])
@pytest.mark.parametrize("scale", [None, 0.137])
@torch.no_grad()
def test_short_query_dense(q_len, kv_len, splits, scale):
    tensors, descales = _inputs((1, q_len, 12, 192), (1, kv_len, 12, 192))
    ref, ref_lse = _reference(*(t[0] for t in tensors), descales, scale)
    out = torch.full(
        (1, q_len, 12, 128), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    lse = torch.full((1, 12, q_len), float("nan"), device="cuda")
    result, result_lse = flydsl_flash_attn_fp8_func(
        *tensors,
        **descales,
        causal=False,
        num_kv_splits=splits,
        softmax_scale=scale,
        return_lse=True,
        out=out,
        lse=lse,
    )
    assert result is out and result_lse is lse
    _check(out[0], lse[0], ref, ref_lse)


@pytest.mark.parametrize("splits", [2, 4, 16, None])
@pytest.mark.parametrize("scale", [None, 0.137])
@torch.no_grad()
def test_short_query_varlen(splits, scale):
    # Unequal query lengths exercise stores past the last live query; empty KV
    # entries must produce zero O and -inf LSE for downstream attention merging.
    q_lens, kv_lens = [70, 1, 127, 0], [16384, 0, 257, 128]
    cuq = list(accumulate(q_lens, initial=0))
    cuk = list(accumulate(kv_lens, initial=0))
    tensors, descales = _inputs((cuq[-1], 12, 192), (cuk[-1], 12, 192))
    kwargs = dict(
        **descales,
        causal=False,
        num_kv_splits=splits,
        softmax_scale=scale,
        cu_seqlens_q=torch.tensor(cuq, device="cuda", dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(cuk, device="cuda", dtype=torch.int32),
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        cross_seqlen=True,
    )
    out, lse = flydsl_flash_attn_fp8_func(*tensors, **kwargs, return_lse=True)
    out_only = flydsl_flash_attn_fp8_func(*tensors, **kwargs)
    torch.testing.assert_close(out, out_only, atol=0, rtol=0)
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        if not q_len:
            continue
        q_slice = slice(cuq[b], cuq[b + 1])
        kv_slice = slice(cuk[b], cuk[b + 1])
        if not kv_len:
            assert (out[q_slice] == 0).all()
            assert torch.isneginf(lse[:, q_slice]).all()
        else:
            ref, ref_lse = _reference(
                tensors[0][q_slice],
                tensors[1][kv_slice],
                tensors[2][kv_slice],
                descales,
                scale,
            )
            _check(out[q_slice], lse[:, q_slice], ref, ref_lse)


@torch.no_grad()
def test_short_query_graph_replay_reads_current_descales():
    tensors, descales = _inputs((1, 70, 12, 192), (1, 16384, 12, 192))

    def run():
        return flydsl_flash_attn_fp8_func(
            *tensors,
            **descales,
            causal=False,
            num_kv_splits=4,
            softmax_scale=0.137,
            return_lse=True,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = run()
    for factor in (0.5, 1.5):
        descales["q_descale"].mul_(factor)
        descales["v_descale"].mul_(factor)
        graph.replay()
        ref, ref_lse = _reference(*(t[0] for t in tensors), descales, 0.137)
        _check(out[0], lse[0], ref, ref_lse)


@pytest.mark.parametrize("causal,cross", [(True, True), (True, False), (False, False)])
def test_short_query_unsupported_modes_still_reject(causal, cross):
    tensors, descales = _inputs((1, 70, 12, 192), (1, 512 if cross else 70, 12, 192))
    with pytest.raises(ValueError, match="split-K requires"):
        flydsl_flash_attn_fp8_func(
            *tensors,
            **descales,
            causal=causal,
            cross_seqlen=cross,
            num_kv_splits=4,
        )
