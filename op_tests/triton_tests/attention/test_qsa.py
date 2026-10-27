# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for QSA quantized cache writes, scoring, selection and
sparse paged GQA."""

import pytest
import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _make_paged_scenario(
    rows,
    pages,
    block,
    ratio,
    index_heads,
    index_dim,
    query_heads,
    query_dim,
    kv_heads,
    start=0,
    dtype=torch.bfloat16,
):
    device = "cuda"
    positions = torch.arange(start, start + rows, device=device, dtype=torch.int32)
    tables = torch.randperm(pages, device=device).to(torch.int32)[None]
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    lengths = torch.tensor([start + rows], dtype=torch.int32, device=device)
    index_q = torch.randn(rows, index_heads, index_dim, dtype=dtype, device=device)
    compressed = torch.randn(
        pages,
        block // ratio,
        1,
        index_dim,
        dtype=dtype,
        device=device,
    )
    query = torch.randn(rows, query_heads, query_dim, dtype=dtype, device=device)
    keys = torch.randn(pages, block, kv_heads, query_dim, dtype=dtype, device=device)
    values = torch.randn_like(keys)
    return {
        "positions": positions,
        "tables": tables,
        "requests": requests,
        "lengths": lengths,
        "index_q": index_q,
        "compressed": compressed,
        "query": query,
        "keys": keys,
        "values": values,
    }


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("start", [15, 2047, 2063])
def test_qsa_verification_is_causal_and_matches_individual_queries(k, start):
    """Future draft keys must not enter an earlier query's selected context."""
    from aiter.ops.qsa import qsa_select_paged_tokens
    from aiter.ops.triton.attention.qsa import qsa_sparse_paged_gqa

    torch.manual_seed(93)
    block, ratio, budget = 16, 4, 2048
    rows, pages = k + 1, 132
    s = _make_paged_scenario(
        rows,
        pages,
        block,
        ratio,
        index_heads=4,
        index_dim=128,
        query_heads=24,
        query_dim=256,
        kv_heads=2,
        start=start,
    )
    selected = qsa_select_paged_tokens(
        s["index_q"],
        s["compressed"],
        s["tables"],
        s["requests"],
        s["positions"],
        s["lengths"],
        budget,
        ratio,
    )
    batched = qsa_sparse_paged_gqa(
        s["query"],
        s["keys"],
        s["values"],
        selected,
        s["tables"],
        s["requests"],
        num_decode_requests=1,
    )
    for row in range(rows):
        visible = selected[row][selected[row] >= 0]
        expected_count = (
            min((start + row + 1) // ratio * ratio, budget) + (start + row + 1) % ratio
        )
        assert visible.numel() == expected_count
        assert visible.unique().numel() == visible.numel()
        assert int(visible.max()) <= start + row

        individual = qsa_select_paged_tokens(
            s["index_q"][row : row + 1],
            s["compressed"],
            s["tables"],
            s["requests"][row : row + 1],
            s["positions"][row : row + 1],
            s["lengths"].new_tensor([start + row + 1]),
            budget,
            ratio,
        )
        torch.testing.assert_close(individual, selected[row : row + 1])

        reference = qsa_sparse_paged_gqa(
            s["query"][row : row + 1],
            s["keys"],
            s["values"],
            individual,
            s["tables"],
            s["requests"][row : row + 1],
            num_decode_requests=1,
        )
        torch.testing.assert_close(
            batched[row : row + 1],
            reference,
            rtol=0.001,
            atol=0.001,
        )


@pytest.mark.parametrize("rows", [512, 1024])
def test_qsa_mfma_scorer_matches_vector_path(rows):
    """The MFMA scorer (rows > 256) must match the vector-path scorer row for row."""
    from aiter.ops.triton.attention.qsa import qsa_paged_mqa_logits

    torch.manual_seed(7)
    block, ratio, pages = 64, 4, 128
    index_heads, index_dim = 4, 128
    compressed_page_size = block // ratio

    positions = torch.arange(rows, dtype=torch.int32, device="cuda")
    tables = torch.randperm(pages, device="cuda").to(torch.int32)[None]
    requests = torch.zeros(rows, dtype=torch.int32, device="cuda")
    lengths = torch.tensor([rows], dtype=torch.int32, device="cuda")
    index_q = torch.randn(
        rows, index_heads, index_dim, dtype=torch.bfloat16, device="cuda"
    )
    compressed = torch.randn(
        pages, compressed_page_size, 1, index_dim, dtype=torch.bfloat16, device="cuda"
    )

    # Vector path: score in chunks of 256 to stay on the non-MFMA kernel
    ref_logits_parts = []
    ref_visible_parts = []
    chunk = 256
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        sl = slice(start, end)
        part_lengths = torch.tensor([end], dtype=torch.int32, device="cuda")
        logits_part, visible_part = qsa_paged_mqa_logits(
            index_q[sl],
            compressed,
            tables,
            requests[sl],
            positions[sl],
            part_lengths,
            ratio,
        )
        ref_logits_parts.append(logits_part)
        ref_visible_parts.append(visible_part)
    ref_logits = torch.cat(ref_logits_parts, dim=0)
    ref_visible = torch.cat(ref_visible_parts, dim=0)

    # MFMA path: rows > 256 triggers the MFMA kernel
    mfma_logits, mfma_visible = qsa_paged_mqa_logits(
        index_q,
        compressed,
        tables,
        requests,
        positions,
        lengths,
        ratio,
    )

    torch.testing.assert_close(mfma_visible, ref_visible)
    for row in range(rows):
        vis = int(ref_visible[row])
        if vis > 0:
            torch.testing.assert_close(
                mfma_logits[row, :vis],
                ref_logits[row, :vis],
                rtol=0.01,
                atol=0.01,
            )


# -- quantized caches and the stacked decode scorer -----------------------


def _store(fmt, values, page_rows):
    """Write `values` `[rows, heads, dim]` into a fresh paged cache of `fmt`."""
    from aiter.ops.qsa import qsa_cache_row_layout, qsa_store_quantized_rows

    rows, heads, dim = values.shape
    pages = -(-rows // page_rows)
    vdim, vdtype, tail, sdtype = qsa_cache_row_layout(fmt, dim)
    cache = torch.zeros(pages, page_rows, heads, vdim, device="cuda", dtype=vdtype)
    if tail is None:
        cache.view(-1, heads, dim)[:rows] = values
        return cache, None
    scale = torch.zeros(pages, page_rows, heads, *tail, device="cuda", dtype=sdtype)
    slots = torch.arange(rows, device="cuda")
    qsa_store_quantized_rows(cache, scale, slots, values)
    return cache, scale


def _dequant(fmt, cache, scale):
    from aiter.ops.qsa import FP4_GROUP

    if scale is None:
        return cache.float()
    if fmt == "fp8":
        return cache.float() * scale.unsqueeze(-1)
    grid = torch.tensor(E2M1 + tuple(-x for x in E2M1), device="cuda")
    codes = cache.long()
    values = torch.stack([grid[codes & 15], grid[codes >> 4]], dim=-1).flatten(-2)
    exponent = scale.float().repeat_interleave(FP4_GROUP, dim=-1)
    return values * torch.exp2(exponent - 127)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_qsa_fp8_writer_never_saturates():
    from aiter.ops.qsa import FP8_DTYPE, qsa_store_quantized_rows

    torch.manual_seed(0)
    rows = 2048
    magnitude = torch.logspace(-4, 4, rows, device="cuda")[:, None, None]
    key = (torch.randn(rows, 2, 256, device="cuda") * magnitude).bfloat16()
    value = (torch.randn(rows, 2, 256, device="cuda") * magnitude).bfloat16()
    caches = [torch.zeros(rows // 16, 16, 2, 256, device="cuda", dtype=FP8_DTYPE)]
    caches.append(torch.zeros_like(caches[0]))
    scales = [torch.zeros(rows // 16, 16, 2, device="cuda") for _ in range(2)]
    slots = torch.arange(rows, device="cuda")
    slots[5] = -1
    qsa_store_quantized_rows(
        caches[0], scales[0], slots, key, caches[1], scales[1], value
    )
    for cache, scale, source in zip(caches, scales, (key, value)):
        restored = _dequant("fp8", cache, scale).view(rows, 2, 256)
        assert torch.isfinite(restored).all()
        per_row = (restored - source.float()).norm(dim=-1) / source.float().norm(dim=-1)
        assert per_row[slots >= 0].max() < 0.06
        assert not restored[5].any()


def test_qsa_fp4_writer_is_exact_on_the_grid_and_close_off_it():
    from aiter.ops.qsa import FP4_GROUP

    torch.manual_seed(1)
    rows = 512
    grid = torch.tensor(E2M1 + tuple(-x for x in E2M1), device="cuda")
    # Each 32-dim group holds +-6 so the shared exponent is the group's own.
    codes = torch.randint(0, 16, (rows, 1, 128), device="cuda")
    exact = grid[codes]
    exact[..., ::FP4_GROUP] = 6.0
    exact *= torch.exp2(
        torch.randint(-20, 20, (rows, 1, 4), device="cuda")
    ).repeat_interleave(FP4_GROUP, -1)
    cache, scale = _store("fp4", exact.bfloat16(), 4)
    torch.testing.assert_close(
        _dequant("fp4", cache, scale).view(-1, 1, 128)[:rows], exact, rtol=0, atol=0
    )
    noisy = torch.randn(rows, 1, 128, device="cuda").bfloat16()
    cache, scale = _store("fp4", noisy, 4)
    restored = _dequant("fp4", cache, scale).view(-1, 1, 128)[:rows]
    per_row = (restored - noisy.float()).norm(dim=-1) / noisy.float().norm(dim=-1)
    assert per_row.mean() < 0.15


def _index_case(requests, context, rows_per_request, page_rows, seed=2):
    g = torch.Generator(device="cuda").manual_seed(seed)
    groups = context // 4
    pages_per_request = -(-groups // page_rows)
    pages = requests * pages_per_request + 3
    order = torch.randperm(pages, device="cuda", generator=g).to(torch.int32)
    table = order[: requests * pages_per_request].view(requests, pages_per_request)
    keys = torch.randn(pages * page_rows, 1, 128, device="cuda", generator=g)
    q = torch.randn(requests * rows_per_request, 4, 128, device="cuda", generator=g)
    t2r = torch.arange(requests, device="cuda", dtype=torch.int32)
    t2r = t2r.repeat_interleave(rows_per_request)
    positions = torch.arange(rows_per_request, device="cuda").repeat(requests)
    positions = (positions + context - rows_per_request).to(torch.int64)
    lens = torch.full((requests,), context, device="cuda", dtype=torch.int32)
    return keys.bfloat16(), q.bfloat16(), table, t2r, positions, lens


def _index_reference(q, restored, table, t2r, columns):
    """FP32 scores of every column against dequantized keys `[slots, dim]`."""
    page_rows = restored.shape[1]
    groups = torch.arange(columns, device="cuda")
    ref = torch.empty(q.shape[0], columns, device="cuda")
    for request in range(table.shape[0]):
        rows = t2r == request
        k = restored[table[request, groups // page_rows].long(), groups % page_rows]
        ref[rows] = torch.relu(q[rows].float() @ k.T).sum(1) / q.shape[2] ** 0.5
    return ref


@pytest.mark.parametrize("fmt", ["fp8", "fp4"])
@pytest.mark.parametrize(
    ("rows_per_request", "stacked", "page_rows"),
    [(4, True, 4), (4, False, 4), (4, True, 16), (256, True, 4), (256, True, 16)],
)
def test_qsa_quantized_index_scores_match_dequantized_keys(
    fmt, rows_per_request, stacked, page_rows
):
    """Every scorer (stacked decode, CUDA-core decode, prefill MFMA) scores a
    quantized page as FP32 math on its dequantized keys would."""
    from aiter.ops.qsa import qsa_paged_mqa_logits

    requests = 9 if rows_per_request == 4 else 2
    keys, q, table, t2r, positions, lens = _index_case(
        requests, 8192, rows_per_request, page_rows
    )
    cache, scale = _store(fmt, keys, page_rows)
    out, visible = qsa_paged_mqa_logits(
        q,
        cache,
        table,
        t2r,
        positions,
        lens,
        4,
        compressed_k_scale=scale,
        stacked_decode=stacked,
    )
    restored = _dequant(fmt, cache, scale).view(-1, page_rows, 128)
    ref = _index_reference(q, restored, table, t2r, out.shape[1])
    expected = torch.minimum((positions + 1) // 4, lens[t2r.long()] // 4)
    assert torch.equal(visible, expected.to(visible.dtype))
    groups = torch.arange(out.shape[1], device="cuda")
    seen = groups[None, :] < visible[:, None]
    torch.testing.assert_close(out[seen], ref[seen], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("fmt", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("rows_per_request", [4, 300])
@pytest.mark.parametrize("stacked", [True, False])
def test_qsa_quantized_index_topk_matches_reference(fmt, rows_per_request, stacked):
    """Selection over a quantized index cache keeps groups the FP32 reference
    ranks in its top-k (up to near ties) and appends the causal tail."""
    from aiter.ops.qsa import qsa_select_paged_tokens

    ratio, budget, page_rows = 4, 512, 16
    requests = 9 if rows_per_request == 4 else 1
    keys, q, table, t2r, positions, lens = _index_case(
        requests, 4096, rows_per_request, page_rows, seed=5
    )
    cache, scale = _store(fmt, keys, page_rows)
    selected = qsa_select_paged_tokens(
        q,
        cache,
        table,
        t2r,
        positions,
        lens,
        budget,
        ratio,
        compressed_k_scale=scale,
        stacked_decode=stacked,
    )
    restored = _dequant(fmt, cache, scale).view(-1, page_rows, 128)
    ref = _index_reference(q, restored, table, t2r, table.shape[1] * page_rows)
    block_topk = budget // ratio
    for row in range(q.shape[0]):
        length = int(positions[row]) + 1
        visible = length // ratio
        tokens = selected[row][selected[row] >= 0]
        history = tokens[tokens < visible * ratio]
        tail = tokens[tokens >= visible * ratio]
        assert torch.equal(
            tail.sort().values,
            torch.arange(visible * ratio, length, device="cuda", dtype=tail.dtype),
        )
        groups = (history // ratio).unique()
        keep = min(block_topk, visible)
        assert groups.numel() == keep
        assert history.numel() == keep * ratio
        scores = ref[row, :visible]
        threshold = scores.topk(keep).values[-1]
        assert (scores[groups.long()] >= threshold - 1e-3 * threshold.abs()).all()


@pytest.mark.parametrize(("tokens", "requests"), [(36, 9), (1024, 1)])
def test_qsa_fp8_sparse_gqa_matches_dequantized_kv(tokens, requests):
    from aiter.ops.qsa import (
        FP8_DTYPE,
        qsa_sparse_paged_gqa,
        qsa_store_quantized_rows,
    )

    g = torch.Generator(device="cuda").manual_seed(3)
    pages_per_request, page_rows = 64, 16
    pages = requests * pages_per_request + 2
    order = torch.randperm(pages, device="cuda", generator=g).to(torch.int32)
    table = order[: requests * pages_per_request].view(requests, pages_per_request)
    shape = (pages, page_rows, 2, 256)
    key = torch.randn(shape, device="cuda", generator=g).bfloat16()
    value = (torch.randn(shape, device="cuda", generator=g) * 3).bfloat16()
    q = torch.randn(tokens, 24, 256, device="cuda", generator=g).bfloat16()
    context = pages_per_request * page_rows
    selected = torch.randint(0, context, (tokens, 515), device="cuda", generator=g)
    selected = selected.to(torch.int32)
    selected[:, -3:] = -1
    t2r = (torch.arange(tokens, device="cuda") * requests // tokens).to(torch.int32)

    caches = [torch.empty(shape, device="cuda", dtype=FP8_DTYPE) for _ in "kv"]
    scales = [torch.empty(shape[:3], device="cuda") for _ in "kv"]
    flat = pages * page_rows
    qsa_store_quantized_rows(
        caches[0],
        scales[0],
        torch.arange(flat, device="cuda"),
        key.view(flat, 2, 256),
        caches[1],
        scales[1],
        value.view(flat, 2, 256),
    )
    decode = requests if tokens == 36 else None
    out = qsa_sparse_paged_gqa(
        q,
        caches[0],
        caches[1],
        selected,
        table,
        t2r,
        1 / 16,
        num_decode_requests=decode,
        k_scale=scales[0],
        v_scale=scales[1],
    )
    ref = qsa_sparse_paged_gqa(
        q,
        _dequant("fp8", caches[0], scales[0]).bfloat16(),
        _dequant("fp8", caches[1], scales[1]).bfloat16(),
        selected,
        table,
        t2r,
        1 / 16,
        num_decode_requests=decode,
    )
    assert torch.isfinite(out).all()
    assert _rel(out, ref) < 1e-2


_STACKED_HEADS, _STACKED_DIM, _STACKED_RATIO = 4, 128, 4


def _stacked_case(rows_per_request, page_size, contexts, padded_rows=3, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ratio = _STACKED_RATIO
    width = max(-(-(ctx // ratio) // page_size) for ctx in contexts) + 2
    pages = len(contexts) * width + 5
    table = torch.full((len(contexts), width), -1, dtype=torch.int32, device="cuda")
    order = torch.randperm(pages, generator=g, device="cuda").to(torch.int32)
    for req, ctx in enumerate(contexts):
        used = -(-(ctx // ratio) // page_size)
        table[req, :used] = order[req * width : req * width + used]
    keys = torch.randn(pages, page_size, 1, _STACKED_DIM, generator=g, device="cuda")
    keys = keys.to(torch.bfloat16)
    requests, positions = [], []
    for req, ctx in enumerate(contexts):
        requests += [req] * rows_per_request
        positions += range(ctx - rows_per_request, ctx)
    requests += [-1] * padded_rows
    positions += [0] * padded_rows
    rows = len(requests)
    q = torch.randn(rows, _STACKED_HEADS, _STACKED_DIM, generator=g, device="cuda")
    q = q.to(torch.bfloat16)
    token_to_request = torch.tensor(requests, dtype=torch.int32, device="cuda")
    query_positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
    context_lens = torch.tensor(contexts, dtype=torch.int32, device="cuda")
    return q, keys, table, token_to_request, query_positions, context_lens


def _stacked_reference(q, keys, table, token_to_request, query_positions, lens):
    page_size = keys.shape[1]
    columns = table.shape[1] * page_size
    logits = torch.full((q.shape[0], columns), -float("inf"), device="cuda")
    visible = torch.zeros(q.shape[0], dtype=torch.int32, device="cuda")
    for row in range(q.shape[0]):
        req = int(token_to_request[row])
        if req < 0:
            continue
        n = min(
            (int(query_positions[row]) + 1) // _STACKED_RATIO,
            int(lens[req]) // _STACKED_RATIO,
        )
        visible[row] = n
        groups = torch.arange(n, device="cuda")
        k = keys[table[req, groups // page_size].long(), groups % page_size, 0]
        scores = torch.relu(q[row].float() @ k.float().T).sum(0)
        logits[row, :n] = scores / _STACKED_DIM**0.5
    return logits, visible


def _seen(logits, visible):
    # Top-k reads a row only up to its visible count; nothing past it is kept.
    columns = torch.arange(logits.shape[1], device="cuda")
    return logits[columns[None, :] < visible[:, None]]


def _launched_kernels(fn, *args, **kwargs):
    """Return `fn(*args, **kwargs)` and the names of the GPU kernels that it
    launched."""
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
    return result, {event.name for event in prof.events()}


@pytest.mark.parametrize("page_size", [4, 16])
@pytest.mark.parametrize("rows_per_request", [1, 4])
def test_qsa_stacked_scorer_matches_cuda_core(page_size, rows_per_request):
    from aiter.ops.qsa import qsa_paged_mqa_logits

    # Uneven contexts: a block of stacked rows spans requests and the widest
    # row sets the tile count, the narrower ones must still mask their tails.
    case = _stacked_case(rows_per_request, page_size, [4103, 37, 1990, 8, 12288, 515])
    ref, ref_visible = _stacked_reference(*case)
    out = {}
    for stacked in (True, False):
        row_starts = torch.full(
            (case[0].shape[0],), 7, dtype=torch.int32, device="cuda"
        )
        (logits, visible), kernels = _launched_kernels(
            qsa_paged_mqa_logits,
            *case,
            compress_ratio=_STACKED_RATIO,
            row_starts=row_starts,
            stacked_decode=stacked,
        )
        assert torch.equal(visible, ref_visible)
        assert not row_starts.any()
        used = any("_qsa_paged_mqa_logits_stacked" in name for name in kernels)
        assert used == stacked, sorted(kernels)
        out[stacked] = _seen(logits, visible)
        torch.testing.assert_close(
            out[stacked], _seen(ref, visible), rtol=1e-4, atol=1e-4
        )
    torch.testing.assert_close(out[True], out[False], rtol=5e-5, atol=5e-5)


def test_qsa_stacked_scorer_respects_max_columns():
    from aiter.ops.qsa import qsa_paged_mqa_logits

    case = _stacked_case(4, 16, [9000, 600])
    ref, _ = _stacked_reference(*case)
    logits, visible = qsa_paged_mqa_logits(
        *case, compress_ratio=_STACKED_RATIO, max_columns=1000, stacked_decode=True
    )
    assert logits.shape[1] == 1000
    visible = visible.clamp(max=1000)
    torch.testing.assert_close(
        _seen(logits, visible), _seen(ref[:, :1000], visible), rtol=1e-4, atol=1e-4
    )
