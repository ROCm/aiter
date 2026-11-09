# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public unified-attention routing: FlyDSL BF16 MiMo route and its fallbacks."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

pytest.importorskip("flydsl")

import aiter.ops.flydsl.pa_decode as pa
import aiter.ops.triton.attention.unified_attention as triton_ua
from aiter.ops.flydsl import unified_attention as flydsl_ua
from aiter.ops.unified_attention import unified_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="the FlyDSL unified-attention route requires gfx950",
)

QLEN, HQ, HKV, D, DV = 8, 16, 1, 192, 128
SKEWED = (54000, 8, 9, 63, 64, 65, 127, 128, 129, 255, 257, 2049, 4097, 8191, 16383)


def _inputs(lengths, page, *, qlen=QLEN, hkv=HKV, shared_page=False, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    batch = len(lengths)
    pages_per_seq = max(1, -(-max(lengths) // page))
    num_pages = batch * pages_per_seq + 1
    k = torch.randn(num_pages, page, hkv, D, device=dev).bfloat16()
    v = torch.randn(num_pages, page, hkv, DV, device=dev).bfloat16()
    # Permuted physical pages; page 0 is never referenced by a live token.
    table = (1 + torch.randperm(num_pages - 1, device=dev, dtype=torch.int32)).reshape(
        batch, pages_per_seq
    )
    if shared_page and batch > 1:
        # Two sequences read the same physical first page (prefix sharing).
        table[1, 0] = table[0, 0]
    q = torch.randn(batch * qlen, HQ * hkv // HKV, D, device=dev).bfloat16()
    seqused_k = torch.tensor(lengths, dtype=torch.int32, device=dev)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=dev) * qlen
    out = torch.full(
        (batch * qlen, q.shape[1], DV), float("nan"), device=dev, dtype=torch.bfloat16
    )
    return q, k, v, out, cu_seqlens_q, seqused_k, table


def _call(q, k, v, out, cu, seqused_k, table, *, qlen=QLEN, backend=None, **kw):
    options = {
        "causal": True,
        "window_size": (-1, -1),
        "softcap": 0,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
    }
    options.update(kw)
    return unified_attention(
        q,
        k,
        v,
        out,
        cu,
        qlen,
        seqused_k,
        int(seqused_k.max().item()),
        D**-0.5,
        options.pop("causal"),
        options.pop("window_size"),
        table,
        options.pop("softcap"),
        options.pop("q_descale"),
        options.pop("k_descale"),
        options.pop("v_descale"),
        backend=backend,
        **options,
    )


def _ragged_inputs(query_lengths, lengths, page, *, seed=0):
    """Sequences of varying query lengths; table and caches as in ``_inputs``."""
    q, k, v, _, _, seqused_k, table = _inputs(lengths, page, qlen=1, seed=seed)
    cu = torch.tensor(query_lengths, dtype=torch.int32).cumsum(0)
    cu_seqlens_q = torch.cat((torch.zeros(1, dtype=torch.int32), cu.int())).cuda()
    tokens = int(cu[-1])
    q = torch.randn(tokens, HQ, D, device="cuda").bfloat16()
    out = torch.full((tokens, HQ, DV), float("nan"), device="cuda").bfloat16()
    return q, k, v, out, cu_seqlens_q, seqused_k, table


def _reference(q, k, v, seqused_k, table, qlen=QLEN, cu_seqlens_q=None):
    page, hkv = k.shape[1], k.shape[2]
    group = q.shape[1] // hkv
    out = torch.zeros(q.shape[0], q.shape[1], DV, device=q.device)
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(seqused_k.numel() + 1) * qlen
    starts = cu_seqlens_q.tolist()
    for b, ctx in enumerate(seqused_k.tolist()):
        first, last = starts[b], starts[b + 1]
        if first == last:
            continue
        tokens = torch.arange(ctx, device=q.device)
        phys = table[b, tokens // page].long()
        keys = k[phys, tokens % page].float()  # [ctx, hkv, D]
        values = v[phys, tokens % page].float()
        rows = q[first:last].float()  # [queries, HQ, D]
        pos = ctx - (last - first) + torch.arange(last - first, device=q.device)
        for h in range(q.shape[1]):
            kv_h = h // group
            scores = rows[:, h] @ keys[:, kv_h].T * D**-0.5  # [queries, ctx]
            scores = scores.masked_fill(tokens[None] > pos[:, None], float("-inf"))
            prob = torch.softmax(scores, dim=-1).nan_to_num(0.0)
            out[first:last, h] = prob @ values[:, kv_h]
    return out


def _check(out, ref, atol=1e-2, rtol=1e-2):
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


@pytest.fixture
def route_spy(monkeypatch):
    """Record whether pa_decode (FlyDSL) or the Triton op ran."""
    calls = []
    for module, name, tag in (
        (pa, "pa_decode", "flydsl"),
        (flydsl_ua, "_unified_attention_varlen", "flydsl"),
        (triton_ua, "unified_attention", "triton"),
    ):
        original = getattr(module, name)

        def spy(*a, _original=original, _tag=tag, **kw):
            calls.append(_tag)
            return _original(*a, **kw)

        monkeypatch.setattr(module, name, spy)
    return calls


@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param((4096,), id="b1-4k"),
        pytest.param((16384,) * 8, id="b8-16k"),
        pytest.param((54000,) * 16, id="b16-54k"),
        pytest.param(SKEWED, id="b15-skewed"),
        pytest.param(SKEWED + (16384,), id="b16-skewed"),
    ],
)
def test_flydsl_mimo_matches_reference(page, lengths, route_spy):
    q, k, v, out, cu, used, table = _inputs(lengths, page, shared_page=True)
    result = _call(q, k, v, out, cu, used, table, backend="flydsl")
    assert result is out and route_spy == ["flydsl"]
    ref = _reference(q, k, v, used, table)
    _check(out, ref)


@pytest.mark.parametrize("page", [64, 128])
def test_auto_selects_flydsl_and_matches_triton(page, route_spy):
    q, k, v, out, cu, used, table = _inputs(SKEWED[:8], page)
    _call(q, k, v, out, cu, used, table)
    assert route_spy == ["flydsl"]
    triton_out = torch.full_like(out, float("nan"))
    _call(q, k, v, triton_out, cu, used, table, backend="triton")
    assert route_spy == ["flydsl", "triton"]
    ref = _reference(q, k, v, used, table)
    _check(out, ref)
    _check(triton_out, ref)


@pytest.mark.parametrize("qlen", [1, 4])
def test_flydsl_shorter_query_lengths(qlen, route_spy):
    q, k, v, out, cu, used, table = _inputs((4097, 129, 64), 128, qlen=qlen)
    _call(q, k, v, out, cu, used, table, qlen=qlen, backend="flydsl")
    assert route_spy == ["flydsl"]
    _check(out, _reference(q, k, v, used, table, qlen))


def _unsupported_cases():
    def page16(args):
        q, k, v, out, cu, used, table = args
        split = k.shape[1] // 16
        k = k.reshape(-1, 16, *k.shape[2:])
        v = v.reshape(-1, 16, *v.shape[2:])
        table = (
            table[:, :, None] * split + torch.arange(split, device=table.device)
        ).reshape(table.shape[0], -1)
        return (q, k, v, out, cu, used, table.int().contiguous()), {}, "page size"

    def window(args):
        return args, {"window_size": (127, 0)}, "full attention"

    def softcap(args):
        return args, {"softcap": 30.0}, "softcap"

    def ragged_sinks(args):
        q, k, v, out, cu, used, table = args
        sinks = torch.zeros(q.shape[1], device=q.device)
        return (q[:-1], k, v, out[:-1], cu, used, table), {"sinks": sinks}, "sinks"

    return [page16, window, softcap, ragged_sinks]


@pytest.mark.parametrize("mutate", _unsupported_cases(), ids=lambda f: f.__name__)
def test_unsupported_falls_back_or_raises(mutate, route_spy):
    args = _inputs((300, 129, 64), 64)
    args, kw, match = mutate(args)
    with pytest.raises(RuntimeError, match=match):
        _call(*args, backend="flydsl", **kw)
    assert route_spy == []
    if match == "sinks":
        return  # Triton handles ragged queries, but this oracle does not.
    # Only causal configurations reach Triton, which handles windows itself.
    _call(*args, **kw)
    assert route_spy == ["triton"]


# (query lengths, KV lengths): prefill chunks on prefixes, MTP-style queries
# that are not uniform, queries longer than 8 mixed with decodes, an empty
# sequence, and a fresh prompt that is not a multiple of 8.
RAGGED = {
    "prefill-chunks": ((700, 1300), (2000, 1300)),
    "mixed-decode": ((1, 517, 1, 3, 64), (54000, 4000, 9, 129, 64)),
    "ragged-mtp": ((1, 8, 3, 5, 2), (300, 8191, 64, 65, 2049)),
    "empty-seq": ((0, 9, 1), (100, 9, 1)),
    "fresh-odd": ((1999,), (1999,)),
}


@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("case", list(RAGGED), ids=list(RAGGED))
def test_flydsl_ragged_matches_reference(page, case, route_spy):
    query_lengths, lengths = RAGGED[case]
    q, k, v, out, cu, used, table = _ragged_inputs(query_lengths, lengths, page)
    result = _call(
        q, k, v, out, cu, used, table, qlen=max(query_lengths), backend="flydsl"
    )
    assert result is out and route_spy == ["flydsl"]
    _check(out, _reference(q, k, v, used, table, cu_seqlens_q=cu.cpu()))


def test_ragged_disabled_by_environment(monkeypatch, route_spy):
    monkeypatch.setenv("AITER_UNIFIED_ATTN_FLYDSL_VARLEN", "0")
    q, k, v, out, cu, used, table = _ragged_inputs((1, 30), (300, 129), 64)
    _call(q, k, v, out, cu, used, table, qlen=30)
    assert route_spy == ["triton"]


def test_disabled_by_environment(monkeypatch, route_spy):
    monkeypatch.setenv("AITER_UNIFIED_ATTN_FLYDSL", "0")
    q, k, v, out, cu, used, table = _inputs((300, 129), 64)
    _call(q, k, v, out, cu, used, table)
    assert route_spy == ["triton"]


@pytest.mark.parametrize("page", [64, 128])
def test_graph_replay_follows_new_lengths(page):
    capacity = (16384, 4097, 129, 8)
    q, k, v, out, cu, used, table = _inputs(capacity, page)
    max_k = max(capacity)
    stream = torch.cuda.Stream()

    def call():
        unified_attention(
            q,
            k,
            v,
            out,
            cu,
            QLEN,
            used,
            max_k,
            D**-0.5,
            True,
            (-1, -1),
            table,
            0,
            None,
            None,
            None,
            backend="flydsl",
        )

    with torch.cuda.stream(stream):
        call()  # Uncaptured warmup allocates the workspace on this stream.
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    for lengths in ((16384, 4097, 129, 8), (8, 16384, 300, 2048), (12000,) * 4):
        used.copy_(torch.tensor(lengths, dtype=torch.int32))
        q.normal_()
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        _check(out, _reference(q, k, v, used, table))


def test_cold_capture_raises_for_explicit_flydsl():
    flydsl_ua._WORKSPACES.clear()
    q, k, v, out, cu, used, table = _inputs((300, 129), 64)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="uncaptured warmup"),
        torch.cuda.graph(graph, stream=stream),
    ):
        unified_attention(
            q,
            k,
            v,
            out,
            cu,
            QLEN,
            used,
            300,
            D**-0.5,
            True,
            (-1, -1),
            table,
            0,
            None,
            None,
            None,
            backend="flydsl",
        )


@pytest.mark.parametrize("page", [64, 128])
def test_ragged_graph_replay_follows_new_offsets(page):
    query_lengths, lengths = (40, 1, 300, 7), (16384, 4097, 300, 2048)
    q, k, v, out, cu, used, table = _ragged_inputs(query_lengths, lengths, page)
    max_k = max(lengths)
    stream = torch.cuda.Stream()

    def call():
        unified_attention(
            q,
            k,
            v,
            out,
            cu,
            300,
            used,
            max_k,
            D**-0.5,
            True,
            (-1, -1),
            table,
            0,
            None,
            None,
            None,
            backend="flydsl",
        )

    with torch.cuda.stream(stream):
        call()  # Uncaptured warmup allocates the workspace on this stream.
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    # Same token count, new split and lengths; each length covers its queries.
    for split, kv in (
        ((40, 1, 300, 7), (16384, 4097, 300, 2048)),
        ((1, 1, 1, 345), (8, 16384, 300, 2048)),
        ((100, 100, 100, 48), (12000, 100, 5000, 48)),
    ):
        offsets = torch.tensor((0, *split), dtype=torch.int32).cumsum(0)
        cu.copy_(offsets.int())
        used.copy_(torch.tensor(kv, dtype=torch.int32))
        q.normal_()
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        _check(out, _reference(q, k, v, used, table, cu_seqlens_q=offsets))
