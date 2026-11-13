# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public unified-attention routing: FlyDSL BF16 MiMo route and its fallbacks."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

pytest.importorskip("flydsl")

import aiter.ops.triton.attention.unified_attention as triton_ua
from aiter.ops.flydsl import unified_attention_bf16 as flydsl_ua
from aiter.ops.flydsl import unified_attention_varlen
from aiter.ops.flydsl.kernels.pa_decode_plan import build_query_block_map
from aiter.ops.unified_attention import unified_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="the FlyDSL unified-attention route requires gfx950",
)

QLEN, HQ, HKV, D, DV = 8, 16, 1, 192, 128
SKEWED = (54000, 8, 9, 63, 64, 65, 127, 128, 129, 255, 257, 2049, 4097, 8191, 16383)
WINDOW = (127, 0)


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
    q = torch.randn(batch * qlen, HQ * hkv, D, device=dev).bfloat16()
    seqused_k = torch.tensor(lengths, dtype=torch.int32, device=dev)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=dev) * qlen
    out = torch.full(
        (batch * qlen, q.shape[1], DV), float("nan"), device=dev, dtype=torch.bfloat16
    )
    return q, k, v, out, cu_seqlens_q, seqused_k, table


def _call(
    q,
    k,
    v,
    out,
    cu,
    seqused_k,
    table,
    *,
    qlen=QLEN,
    max_k=None,
    backend=None,
    **kw,
):
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
        int(seqused_k.max().item()) if max_k is None else max_k,
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


def _ragged_inputs(query_lengths, lengths, page, *, hkv=HKV, seed=0):
    """Sequences of varying query lengths; table and caches as in ``_inputs``."""
    q, k, v, _, _, seqused_k, table = _inputs(lengths, page, qlen=1, hkv=hkv, seed=seed)
    cu = torch.tensor(query_lengths, dtype=torch.int32).cumsum(0)
    cu_seqlens_q = torch.cat((torch.zeros(1, dtype=torch.int32), cu.int())).cuda()
    tokens = int(cu[-1])
    q = torch.randn(tokens, HQ * hkv, D, device="cuda").bfloat16()
    out = torch.full((tokens, HQ * hkv, DV), float("nan"), device="cuda").bfloat16()
    return q, k, v, out, cu_seqlens_q, seqused_k, table


def _reference(
    q, k, v, seqused_k, table, qlen=QLEN, cu_seqlens_q=None, window=0, sinks=None
):
    """FP32 attention; ``window`` keys per query including its own (0: all), and
    ``sinks`` one extra logit per query head that only joins the denominator."""
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
        keys = k[phys, tokens % page].float().repeat_interleave(group, 1)
        values = v[phys, tokens % page].float().repeat_interleave(group, 1)
        # Query chunks bound the [heads, queries, ctx] scores of long prefills.
        for lo in range(first, last, 1024):
            hi = min(lo + 1024, last)
            rows = q[lo:hi].float()
            pos = (
                ctx
                - (last - first)
                + (lo - first)
                + torch.arange(hi - lo, device=q.device)
            )
            scores = torch.einsum("qhd,khd->hqk", rows, keys) * D**-0.5
            mask = tokens[None] > pos[:, None]
            if window:
                mask |= tokens[None] <= pos[:, None] - window
            scores = scores.masked_fill(mask, float("-inf"))
            if sinks is not None:
                sink = sinks.float()[:, None, None].expand(-1, hi - lo, 1)
                scores = torch.cat((scores, sink), -1)
            prob = torch.softmax(scores, dim=-1).nan_to_num(0.0)
            if sinks is not None:
                prob = prob[..., :-1]
            out[lo:hi] = torch.einsum("hqk,khd->qhd", prob, values)
    return out


def _check(out, ref, atol=1e-2, rtol=1e-2):
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


@pytest.fixture
def route_spy(monkeypatch):
    """Record which FlyDSL route or the Triton op ran."""
    calls = []
    for module, name, tag in (
        (flydsl_ua, "_unified_attention_varlen", "flydsl"),
        (flydsl_ua, "_unified_attention_window", "flydsl"),
        (flydsl_ua, "_unified_attention_varlen_optimized", "optimized"),
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


@pytest.mark.parametrize("qlen", [1, 2, 4])
def test_flydsl_shorter_query_lengths(qlen, route_spy):
    q, k, v, out, cu, used, table = _inputs((4097, 129, 64), 128, qlen=qlen)
    _call(q, k, v, out, cu, used, table, qlen=qlen, backend="flydsl")
    assert route_spy == ["flydsl"]
    _check(out, _reference(q, k, v, used, table, qlen))


@pytest.mark.parametrize("qlen", [1, 2, 8])
def test_flydsl_two_kv_heads(qlen, route_spy):
    q, k, v, out, cu, used, table = _inputs((4097, 129, 300, 64), 64, qlen=qlen, hkv=2)
    _call(q, k, v, out, cu, used, table, qlen=qlen, backend="flydsl")
    assert route_spy == ["flydsl"]
    _check(out, _reference(q, k, v, used, table, qlen))


def test_flydsl_decode_batches_above_512(route_spy):
    lengths = tuple(
        int(x)
        for x in torch.randint(
            1, 700, (600,), generator=torch.Generator().manual_seed(1)
        )
    )
    q, k, v, out, cu, used, table = _inputs(lengths, 64, qlen=1)
    _call(q, k, v, out, cu, used, table, qlen=1, backend="flydsl")
    assert route_spy == ["flydsl"]
    _check(out, _reference(q, k, v, used, table, 1))


def test_shrinking_batch_reuses_workspace(route_spy):
    """A batch that fills the CUs (one task per block, no partials) followed by a
    smaller one on the same workspace, which needs the partial buffers."""
    flydsl_ua._WORKSPACES.clear()
    for batch in (300, 64):
        lengths = (4097, 129, 64) * (batch // 3) + (300,) * (batch % 3)
        q, k, v, out, cu, used, table = _inputs(lengths, 128, qlen=1)
        _call(q, k, v, out, cu, used, table, qlen=1, backend="flydsl")
        _check(out, _reference(q, k, v, used, table, 1))
    assert route_spy == ["flydsl", "flydsl"]


@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("sinks", [False, True], ids=["no-sinks", "sinks"])
@pytest.mark.parametrize(
    "query_lengths,lengths",
    [
        pytest.param((1,) * 4, (300, 129, 4097, 64), id="uniform-q1"),
        pytest.param((8,) * 4, (300, 129, 4097, 64), id="uniform-q8"),
        pytest.param((40, 1, 300, 7), (16384, 4097, 300, 2048), id="ragged"),
    ],
)
def test_sliding_window_matches_reference(
    page, sinks, query_lengths, lengths, route_spy
):
    q, k, v, out, cu, used, table = _ragged_inputs(query_lengths, lengths, page)
    sink = torch.randn(HQ, device="cuda") * 2 if sinks else None
    _call(
        q,
        k,
        v,
        out,
        cu,
        used,
        table,
        qlen=max(query_lengths),
        backend="flydsl",
        window_size=WINDOW,
        sinks=sink,
    )
    assert route_spy == ["flydsl"]
    ref = _reference(
        q, k, v, used, table, cu_seqlens_q=cu.cpu(), window=WINDOW[0] + 1, sinks=sink
    )
    _check(out, ref)


def test_sliding_window_two_kv_heads(route_spy):
    query_lengths, lengths = (40, 1, 300, 7), (16384, 4097, 300, 2048)
    q, k, v, out, cu, used, table = _ragged_inputs(query_lengths, lengths, 64, hkv=2)
    _call(q, k, v, out, cu, used, table, qlen=300, backend="flydsl", window_size=WINDOW)
    assert route_spy == ["flydsl"]
    _check(
        out,
        _reference(q, k, v, used, table, cu_seqlens_q=cu.cpu(), window=WINDOW[0] + 1),
    )


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
        return args, {"window_size": (127, 5)}, "causal left window"

    def softcap(args):
        return args, {"softcap": 30.0}, "softcap"

    def ragged_sinks(args):
        q, k, v, out, cu, used, table = args
        sinks = torch.zeros(q.shape[1], device=q.device)
        return (q[:-1], k, v, out[:-1], cu, used, table), {"sinks": sinks}, "sinks"

    def cu_seqlens_int64(args):
        q, k, v, out, cu, used, table = args
        return (q, k, v, out, cu.long(), used, table), {}, "cu_seqlens_q"

    return [page16, window, softcap, ragged_sinks, cu_seqlens_int64]


@pytest.mark.parametrize("mutate", _unsupported_cases(), ids=lambda f: f.__name__)
def test_unsupported_falls_back_or_raises(mutate, route_spy):
    args = _inputs((300, 129, 64), 64)
    args, kw, match = mutate(args)
    with pytest.raises(RuntimeError, match=match):
        _call(*args, backend="flydsl", **kw)
    assert route_spy == []
    if match in ("sinks", "causal left window", "cu_seqlens_q"):
        return  # Triton handles these, but this oracle does not.
    # Only causal configurations reach Triton.
    _call(*args, **kw)
    assert route_spy == ["triton"]


def test_uniform_route_requires_cu_seqlens_q():
    q, k, v, out, _, used, table = _inputs((300, 129, 64), 64)
    common = (8, used, 300, D**-0.5, True)
    tail = (table, 0, None, None, None)
    reason = flydsl_ua.flydsl_decline_reason(
        q, k, v, out, None, *common, (-1, -1), *tail
    )
    assert reason is not None and "cu_seqlens_q" in reason
    # The uniform sliding-window route takes query rows from the shapes.
    assert (
        flydsl_ua.flydsl_decline_reason(q, k, v, out, None, *common, WINDOW, *tail)
        is None
    )


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


def _query_map_reference(
    cu_seqlens_q, seqused_k, num_tokens, capacity, query_block, longest_first
):
    """(query_map, contexts) of build_query_block_map, in torch."""
    dev = cu_seqlens_q.device
    query_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
    blocks = (query_lengths + (query_block - 1)) // query_block
    block_ends = torch.cumsum(blocks, 0)
    pseudo = torch.arange(capacity, device=dev)
    seq = torch.searchsorted(block_ends, pseudo, right=True)
    live = seq < blocks.numel()
    seq = seq.clamp_(max=blocks.numel() - 1)
    first = (pseudo - (block_ends[seq] - blocks[seq])) * query_block
    row = cu_seqlens_q[seq].long() + first
    valid = torch.minimum(query_lengths[seq] - first, num_tokens - row)
    valid = torch.where(live, valid.clamp(0, query_block), 0)
    context = torch.where(
        live, seqused_k[seq].long() - query_lengths[seq] + first + valid, 0
    ).clamp_(min=0)
    rows = torch.stack(
        (row.clamp(0, max(num_tokens - 1, 0)), seq, valid, torch.zeros_like(seq)), 1
    )
    if longest_first:
        order = torch.argsort(context, descending=True, stable=True)
        rows, context = rows[order], context[order]
    return rows.int(), context.int()


@pytest.mark.parametrize("longest_first", [True, False], ids=["sorted", "in-order"])
@pytest.mark.parametrize(
    "batch,max_query,query_block",
    [
        (1, 9000, 16),
        (64, 1, 1),
        (600, 1, 1),
        (1024, 8, 8),
        (300, 200, 16),
        (4096, 40, 16),
    ],
)
def test_query_block_map_matches_torch(batch, max_query, query_block, longest_first):
    gen = torch.Generator().manual_seed(batch)
    query_lengths = torch.randint(0, max_query + 1, (batch,), generator=gen)
    query_lengths[0] = max_query
    contexts = query_lengths + torch.randint(0, 5000, (batch,), generator=gen)
    cu = (
        torch.cat((torch.zeros(1, dtype=torch.long), query_lengths.cumsum(0)))
        .int()
        .cuda()
    )
    used = contexts.int().cuda()
    tokens = int(cu[-1])
    capacity = -(-tokens // query_block) + batch
    query_map = torch.full((capacity, 4), -7, dtype=torch.int32, device="cuda")
    ctx = torch.full((capacity,), -7, dtype=torch.int32, device="cuda")
    build_query_block_map(cu, used, tokens, query_block, query_map, ctx, longest_first)
    ref_map, ref_ctx = _query_map_reference(
        cu, used, tokens, capacity, query_block, longest_first
    )
    torch.testing.assert_close(query_map, ref_map, rtol=0, atol=0)
    torch.testing.assert_close(ctx, ref_ctx, rtol=0, atol=0)


def test_varlen_optimized_route_is_narrow(monkeypatch):
    q = torch.empty((8192, HQ, D), dtype=torch.bfloat16)
    k = torch.empty((1, 128, 1, D), dtype=torch.bfloat16)
    out = torch.empty((8192, HQ, DV), dtype=torch.bfloat16)
    eligible = flydsl_ua._use_varlen_optimized

    assert eligible(q, k, out, 1, 8192, 256)
    assert not eligible(q, k[:, :64], out, 1, 8192, 256)
    assert not eligible(q, k.expand(1, 128, 2, D), out, 1, 8192, 256)
    assert not eligible(q, k, out, 1, 4095, 256)
    assert not eligible(q[:16], k, out[:16], 1, 8192, 256)
    assert not eligible(q, k, out, 1, 8192, 255)

    monkeypatch.setenv("AITER_UNIFIED_ATTN_FLYDSL_VARLEN_OPTIMIZED", "0")
    assert not eligible(q, k, out, 1, 8192, 256)
    monkeypatch.setenv("AITER_UNIFIED_ATTN_FLYDSL_VARLEN_OPTIMIZED", "1")
    monkeypatch.setenv("AITER_UNIFIED_ATTN_FLYDSL_VARLEN", "0")
    assert not eligible(q, k, out, 1, 8192, 256)


def test_varlen_optimized_workspace_covers_tail_plan():
    spec = unified_attention_varlen.workspace_requirements(8192, 1, 256)
    assert spec is not None
    need, pseudo, capacity, tail_lo = spec
    assert capacity > pseudo
    assert 0 <= tail_lo < pseudo
    assert need["ticket"] == (1, torch.int32)
    assert need["work_info"] == (capacity * 4, torch.int32)
    assert need["reduce_info"] == (pseudo * 2, torch.int32)
    assert need["tail_max_logits"][1] == torch.float32
    assert need["tail_exp_sums"][1] == torch.float32
    assert need["tail_output"][1] == torch.float32


def test_varlen_optimized_dispatch_and_kill_switch(monkeypatch):
    qlen = 8192
    args = _inputs((qlen,), 128, qlen=qlen)
    calls = []

    def optimized(*call_args, **kwargs):
        calls.append("optimized")
        return call_args[3]

    def baseline(*call_args, **kwargs):
        calls.append("baseline")
        return call_args[3]

    monkeypatch.setattr(flydsl_ua, "_unified_attention_varlen_optimized", optimized)
    monkeypatch.setattr(flydsl_ua, "_unified_attention_varlen", baseline)
    _call(*args, qlen=qlen, backend="flydsl")
    assert calls == ["optimized"]

    calls.clear()
    monkeypatch.setenv("AITER_UNIFIED_ATTN_FLYDSL_VARLEN_OPTIMIZED", "0")
    _call(*args, qlen=qlen, backend="flydsl")
    assert calls == ["baseline"]


@pytest.mark.parametrize(
    "query_lengths,lengths",
    [
        pytest.param((8192,), (8192,), id="prefill-8k"),
        # Prefill chunks on 2k prefixes beside decodes: the shortest blocks split.
        pytest.param((2048,) * 8 + (1,) * 56, (4096,) * 8 + (6000,) * 56, id="mixed"),
    ],
)
def test_varlen_optimized_matches_reference(query_lengths, lengths, route_spy):
    if torch.cuda.get_device_properties(0).multi_processor_count != 256:
        pytest.skip("the optimized varlen kernel runs on 256-CU parts")
    q, k, v, out, cu, used, table = _ragged_inputs(query_lengths, lengths, 128)
    _call(q, k, v, out, cu, used, table, qlen=max(query_lengths), backend="flydsl")
    assert route_spy == ["optimized"]
    _check(out, _reference(q, k, v, used, table, cu_seqlens_q=cu.cpu()))


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
        _call(q, k, v, out, cu, used, table, max_k=max_k, backend="flydsl")

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
        _call(q, k, v, out, cu, used, table, max_k=300, backend="flydsl")


@pytest.mark.parametrize("page", [64, 128])
def test_ragged_graph_replay_follows_new_offsets(page):
    query_lengths, lengths = (40, 1, 300, 7), (16384, 4097, 300, 2048)
    q, k, v, out, cu, used, table = _ragged_inputs(query_lengths, lengths, page)
    max_k = max(lengths)
    stream = torch.cuda.Stream()

    def call():
        _call(q, k, v, out, cu, used, table, qlen=300, max_k=max_k, backend="flydsl")

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


def test_varlen_optimized_graph_replay_follows_new_lengths():
    qlen = 8192
    # The block table holds 2 * qlen tokens.
    q, k, v, out, cu, used, table = _inputs((2 * qlen,), 128, qlen=qlen)
    stream = torch.cuda.Stream()

    def call():
        _call(
            q, k, v, out, cu, used, table, qlen=qlen, max_k=2 * qlen, backend="flydsl"
        )

    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    for context in (qlen, qlen + 4000):
        used.fill_(context)
        q.normal_()
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        _check(out, _reference(q, k, v, used, table, qlen))
