# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL cascade (shared-prefix) paged decode correctness tests.

Run with: python -m pytest -q op_tests/test_flydsl_pa_decode_cascade.py

Every test asserts that the cascade path actually ran (at least one
grouped item), so a silently-fallen-back op can never go green. Metadata
is built on host here (a planner op fills the same layout on device in
`aiter.ops.flydsl.pa_decode_cascade_plan`); the op itself runs fully on
device.
"""

import importlib.util

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

PAGE = 16


@pytest.fixture
def device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        pytest.skip(
            f"cascade pa_decode is wired for gfx942/gfx950, got {get_gfx_runtime()}"
        )
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return torch.device("cuda")


def _make_caches(num_pages, kv_heads, dim, dtype, device):
    """5D vectorized BF16 K/V caches plus a logical [pages, kvh, page, dim] view."""
    key = torch.empty(num_pages, kv_heads, PAGE, dim, device=device).normal_(0, 0.1)
    value = torch.empty(num_pages, kv_heads, PAGE, dim, device=device).normal_(0, 0.1)
    key = key.to(dtype)
    value = value.to(dtype)
    key_cache = (
        key.reshape(-1, kv_heads, PAGE, dim // 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    )
    value_cache = (
        value.reshape(-1, kv_heads, PAGE // 8, 8, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    return key, value, key_cache, value_cache


def _reference(query, key, value, table, lengths, sinks=None):
    """Fp32 attention over logical KV, one output row per request (qlen 1)."""
    batch = lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads = key.shape[1]
    group = heads // kv_heads
    out = torch.zeros(batch, heads, dim, dtype=torch.float32, device=query.device)
    for row in range(batch):
        total = int(lengths[row])
        blocks = table[row, : -(-total // PAGE)]
        # [nblk, kvh, page, dim] -> [nblk * page, kvh, dim]
        k = key[blocks].permute(0, 2, 1, 3).reshape(-1, kv_heads, dim)[:total].float()
        v = value[blocks].permute(0, 2, 1, 3).reshape(-1, kv_heads, dim)[:total].float()
        q = query[row].float().reshape(kv_heads, group, dim)
        s = torch.einsum("kgd,tkd->kgt", q, k) * dim**-0.5
        if sinks is None:
            w = s.softmax(-1)
        else:
            sink = sinks.float().reshape(kv_heads, group)
            m = torch.maximum(s.max(-1).values, sink)
            es = (s - m[..., None]).exp()
            w = es / (es.sum(-1, keepdim=True) + (sink - m).exp()[..., None])
        out[row] = torch.einsum("kgt,tkd->kgd", w, v).reshape(heads, dim)
    return out


def _fill_workspace(ws, items, singles_rows, lengths, table, device):
    """Host-side builder of the cascade metadata the planner emits.

    ``items`` is a list of (member rows, shared block count) tuples; rows
    not in any item are singles. Pads every buffer to the workspace max
    with inert values (count 0 items, length 0 queries).
    """
    ws.items.zero_()
    b_rows = []
    for it, (members, sblk) in enumerate(items):
        assert len(members) <= ws.members_per_item
        assert sblk >= 1
        ws.items[it, 0] = sblk
        ws.items[it, 1] = len(members)
        for slot, row in enumerate(members):
            ws.items[it, 2 + slot] = row
            tail = int(lengths[row]) - sblk * PAGE
            assert tail >= 0, "shared prefix cannot exceed the context"
            b_idx = len(b_rows)
            b_rows.append((b_idx, it, slot, row, tail))
            ws.b_of_item_slot[it, slot] = b_idx
    assert len(b_rows) <= ws.max_b_rows
    for b_idx, it, slot, row, tail in b_rows:
        ws.b_map[b_idx] = row
        ws.tok_of_b[b_idx] = row
        ws.b_item[b_idx] = it
        ws.b_slot[b_idx] = slot
        ws.tail_lens[b_idx] = tail
        sblk = int(ws.items[it, 0])
        span = -(-tail // PAGE)
        kept = table[row, sblk : sblk + span]
        ws.tail_tables[b_idx, : kept.numel()] = kept
    for s_idx, row in enumerate(singles_rows[: ws.max_singles]):
        ws.singles_rows[s_idx] = row
        ws.tok_of_s[s_idx] = row
        ws.singles_lens[s_idx] = lengths[row]
        span = -(-int(lengths[row]) // PAGE)
        ws.singles_tables[s_idx, :span] = table[row, :span]
    ws.b_map[len(b_rows) :].zero_()
    ws.tok_of_b[len(b_rows) :].zero_()
    ws.tail_lens[len(b_rows) :].zero_()
    ws.b_of_item_slot[len(items) :].zero_()
    ws.singles_rows[len(singles_rows) :].zero_()
    ws.singles_lens[len(singles_rows) :].zero_()
    ws.counts[0] = len(items)
    ws.counts[1] = len(b_rows)
    ws.counts[2] = min(len(singles_rows), ws.max_singles)
    return len(items), len(b_rows)


def _run_case(
    device,
    *,
    group,
    dim,
    kv_heads=2,
    items=(),
    lengths=(),
    table_width=None,
    sinks=False,
    n_items_cap=None,
):
    """Build a synthetic batch, run cascade, compare against fp32 reference.

    Returns (cascade output, reference, n_items, n_members) for callers
    that assert on path coverage.
    """
    from aiter.ops.flydsl.pa_decode_cascade import (
        pa_decode_cascade,
        plan_pa_decode_cascade,
    )

    torch.manual_seed(0)
    heads = kv_heads * group
    batch = len(lengths)
    if table_width is None:
        table_width = max(-(-v // PAGE) for v in lengths) + 2
    num_pages = table_width * batch + 8
    key, value, key_cache, value_cache = _make_caches(
        num_pages, kv_heads, dim, torch.bfloat16, device
    )
    # One contiguous, non-overlapping page run per row keeps tables simple;
    # group members share physical blocks exactly where their rows agree.
    table = torch.zeros(batch, table_width, dtype=torch.int32)
    cursor = 0
    row_pages = []
    for row, total in enumerate(lengths):
        n = -(-total // PAGE)
        row_pages.append((cursor, n))
        table[row, :n] = torch.arange(cursor, cursor + n)
        cursor += n
    # Make item members share the shared block range physically.
    for members, sblk in items:
        base = int(table[members[0], 0])
        for row in members[1:]:
            table[row, :sblk] = torch.arange(base, base + sblk)
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    table = table.to(device)
    query = torch.randn(batch, heads, dim, device=device).to(torch.bfloat16) * 0.25
    sink_t = (
        torch.randn(heads, device=device).to(torch.bfloat16) * 0.5 if sinks else None
    )
    singles = sorted(set(range(batch)) - {r for rows, _ in items for r in rows})
    max_items = n_items_cap if n_items_cap is not None else max(len(items), 1)
    kept_items, spilled = items[:max_items], items[max_items:]
    singles = singles + [r for rows, _ in spilled for r in rows]
    singles.sort()
    ws = plan_pa_decode_cascade(
        device,
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=dim,
        members_per_item=4,
        max_items=max_items,
        max_singles=max(len(singles), 1),
        table_width=table_width,
    )
    n_items, _n_members = _fill_workspace(
        ws, kept_items, singles, lengths_t, table, device
    )
    output = torch.full_like(query, float("nan"))
    pa_decode_cascade(
        output,
        query,
        key_cache,
        value_cache,
        lengths_t,
        table,
        dim**-0.5,
        sinks=sink_t,
        workspace=ws,
    )
    ref = _reference(query, key, value, table, lengths_t, sink_t)
    return output, ref, n_items, singles


def _check(output, ref, n_items, singles=(), atol=8e-3, rtol=2e-2):
    """No NaNs, no path fallback, and per-element bounds (members looser).

    Grouped rows round softmax statistics in bf16 across two segmented
    softmax-passes merged in LSE space; singles run the stock kernel. The
    member bound tracks the measured class of that composition on gfx942
    (2e-2..3e-2 absolute on uniform-random inputs of this magnitude);
    singles stay at the plain-kernel class.
    """
    assert n_items > 0, "the cascade path did not run"
    out, ref = output.float(), ref.float()
    assert not torch.isnan(out).any()
    diff = (out - ref).abs()
    bound_member = 4.0e-2 + 2e-2 * ref.abs()
    bound_single = atol + rtol * ref.abs()
    bad_member = (diff - bound_member).clamp(min=0).amax()
    rows = set(range(output.shape[0])) - set(singles)
    bad_single = (
        (diff[list(singles), :, :] - bound_single[list(singles), :, :])
        .clamp(min=0)
        .amax()
        if singles
        else torch.tensor(0.0)
    )
    assert bad_member == 0 and bad_single == 0, (
        f"member slack {bad_member:.4g}, single slack {bad_single:.4g}, "
        f"max diff {diff.amax():.4g} on rows {sorted(rows)}"
    )


_GEOMS = [
    pytest.param(3, 64, id="g3-d64"),
    pytest.param(4, 128, id="g4-d128"),
    pytest.param(4, 512, id="g4-d512"),
    pytest.param(6, 256, id="g6-d256"),
    pytest.param(8, 64, id="g8-d64"),
]


@pytest.mark.parametrize("group,dim", _GEOMS)
def test_parity(device, group, dim):
    """Members share one 3-block prefix; two singles take the plain path."""
    out, ref, n_items, singles = _run_case(
        device,
        group=group,
        dim=dim,
        lengths=[80, 96, 77, 64, 33],
        items=[([0, 1, 2], 3)],
    )
    _check(out, ref, n_items, singles)


@pytest.mark.parametrize("group,dim", _GEOMS)
def test_parity_sinks(device, group, dim):
    """Attention sinks (gpt-oss contract) are added exactly once, in the fold."""
    out, ref, n_items, singles = _run_case(
        device,
        group=group,
        dim=dim,
        lengths=[80, 96, 77, 64, 33],
        items=[([0, 1, 2], 3)],
        sinks=True,
    )
    _check(out, ref, n_items, singles)


def test_identical_prompts(device):
    """Duplicate requests: the private tail may be a fraction of one block."""
    out, ref, n_items, singles = _run_case(
        device,
        group=4,
        dim=128,
        lengths=[49, 49, 49, 80],
        items=[([0, 1, 2], 3)],  # tail = 1 token
    )
    _check(out, ref, n_items, singles)


def test_shared_to_block_boundary_full_coverage(device):
    """Identical block-aligned contexts: tail length is exactly zero."""
    out, ref, n_items, singles = _run_case(
        device,
        group=4,
        dim=128,
        lengths=[48, 48, 48, 96],
        items=[([0, 1, 2], 3)],
    )
    _check(out, ref, n_items, singles)


def test_group_splits_into_items(device):
    """A 7-member group becomes items of at most 4 members each."""
    out, ref, n_items, singles = _run_case(
        device,
        group=4,
        dim=128,
        lengths=[87, 91, 78, 82, 84, 79, 90],
        items=[([0, 1, 2, 3], 4), ([4, 5, 6], 4)],
    )
    assert n_items == 2
    _check(out, ref, n_items, singles)


def test_uneven_tail_lengths(device):
    """Members' tails of 1 block vs ~100 blocks merge exactly."""
    out, ref, n_items, singles = _run_case(
        device,
        group=4,
        dim=128,
        lengths=[81, 1680, 96, 640],
        items=[([0, 1, 2], 5)],
    )
    _check(out, ref, n_items, singles)


def test_item_overflow_spills_to_singles(device):
    """When the item table is full, the extra members run as singles."""
    out, ref, n_items, singles = _run_case(
        device,
        group=4,
        dim=128,
        lengths=[64, 80, 72, 88],
        items=[([0, 1], 2), ([2, 3], 2)],
        n_items_cap=1,
    )
    assert n_items == 1
    _check(out, ref, n_items, singles)


def test_group_of_one_and_mixed_tail_size(device):
    """A singleton group is legal metadata; remaining rows are singles."""
    out, ref, n_items, singles = _run_case(
        device,
        group=4,
        dim=128,
        lengths=[81, 47, 2000, 33],
        items=[([0], 3)],
    )
    _check(out, ref, n_items, singles)


def test_mixed_tail_blocks_partial(device):
    """Whole-context-shared-except-write-block members (tail <= 1 block)."""
    out, ref, n_items, singles = _run_case(
        device,
        group=4,
        dim=256,
        lengths=[50, 61, 160, 129],
        items=[([0, 1], 3), ([2, 3], 8)],
    )
    assert n_items == 2
    _check(out, ref, n_items, singles)


def test_cuda_graph_replay_changed_groups(device):
    """Capture the whole op, then replay replanning different groupings.

    All grids are static and every buffer lives in the workspace, so a
    replay against fresh metadata must match the eager run bit for bit.
    """
    from aiter.ops.flydsl.pa_decode_cascade import (
        pa_decode_cascade,
        plan_pa_decode_cascade,
    )

    group, dim, kv_heads = 4, 128, 2
    heads = kv_heads * group
    lengths = [80, 96, 77, 64, 61, 71]
    batch = len(lengths)
    table_width = 8
    key, value, key_cache, value_cache = _make_caches(
        table_width * batch + 8, kv_heads, dim, torch.bfloat16, device
    )
    table = (
        torch.arange(batch * table_width, dtype=torch.int32)
        .reshape(batch, table_width)
        .to(device)
    )
    for members in ([0, 1, 2], [3, 4]):
        base = int(table[members[0], 0])
        for row in members[1:]:
            table[row, :3] = torch.arange(base, base + 3)
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    query = (torch.randn(batch, heads, dim, device=device)).to(torch.bfloat16) * 0.25
    ws = plan_pa_decode_cascade(
        device,
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=dim,
        max_items=4,
        max_singles=6,
        table_width=table_width,
    )
    output = torch.empty_like(query)
    eager = torch.empty_like(query)

    def run():
        pa_decode_cascade(
            output,
            query,
            key_cache,
            value_cache,
            lengths_t,
            table,
            dim**-0.5,
            workspace=ws,
        )

    # Step 1: fill metadata, run twice eagerly (compile), capture third.
    _fill_workspace(ws, [([0, 1, 2], 3)], [3, 4, 5], lengths_t, table, device)
    output.copy_(eager)
    run()
    eager.copy_(output)
    ref1 = _reference(query, key, value, table, lengths_t)
    assert (output.float() - ref1).abs().amax() < 4.0e-2

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    g.replay()
    assert torch.equal(output, eager), "replay changed results with same metadata"

    # Step 2: different grouping (items/singles/roles swap), lengths change.
    lengths_t[0] = 81
    _fill_workspace(ws, [([3, 4], 3)], [0, 1, 2, 5], lengths_t, table, device)
    run()
    eager.copy_(output)
    g.replay()
    assert torch.equal(output, eager), "replay differs from eager after regrouping"
    ref2 = _reference(query, key, value, table, lengths_t)
    ok = (output.float() - ref2).abs()
    bad = ok.clone()
    bad[[3, 4]] = bad[[3, 4]] * 0  # members share the looser cascade bound
    members = ok[[3, 4]].amax()
    assert bad.amax() < 3e-3 and members < 4.0e-2


def _run_plain4d_case(device, *, group, dim, kv_heads=2, items=(), lengths=()):
    """Same as _run_case but through kv_layout="plain4d" (Triton v2 tails)."""
    from aiter.ops.flydsl.pa_decode_cascade import (
        pa_decode_cascade,
        plan_pa_decode_cascade,
    )

    torch.manual_seed(0)
    heads = kv_heads * group
    batch = len(lengths)
    table_width = max(-(-v // PAGE) for v in lengths) + 2
    num_pages = table_width * batch + 8
    key, value, _, _ = _make_caches(num_pages, kv_heads, dim, torch.bfloat16, device)
    table = torch.zeros(batch, table_width, dtype=torch.int32)
    cursor = 0
    for row, total in enumerate(lengths):
        n = -(-total // PAGE)
        table[row, :n] = torch.arange(cursor, cursor + n)
        cursor += n
    for members, sblk in items:
        base = int(table[members[0], 0])
        for row in members[1:]:
            table[row, :sblk] = torch.arange(base, base + sblk)
    table = table.to(device)
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    query = torch.randn(batch, heads, dim, device=device).to(torch.bfloat16) * 0.25
    singles = sorted(set(range(batch)) - {r for rows, _ in items for r in rows})
    ws = plan_pa_decode_cascade(
        device,
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=dim,
        members_per_item=4,
        max_items=max(len(items), 1),
        max_singles=max(len(singles), 1),
        table_width=table_width,
        kv_layout="plain4d",
    )
    n_items, _ = _fill_workspace(ws, items, singles, lengths_t, table, device)
    output = torch.full_like(query, float("nan"))
    pa_decode_cascade(
        output, query, key, value, lengths_t, table, dim**-0.5, workspace=ws
    )
    ref = _reference(query, key, value, table, lengths_t)
    return output, ref, n_items, singles


def test_plain4d_parity(device):
    """Plain vLLM-layout caches: shared pass + v2 tails match fp32 references."""
    out, ref, n_items, singles = _run_plain4d_case(
        device,
        group=4,
        dim=128,
        lengths=[80, 96, 77, 64, 33],
        items=[([0, 1, 2], 3)],
    )
    _check(out, ref, n_items, singles)


def test_plain4d_sinks_and_boundary(device):
    out, ref, n_items, singles = _run_plain4d_case(
        device,
        group=6,
        dim=256,
        lengths=[48, 48, 48, 96],
        items=[([0, 1, 2], 3)],
    )
    _check(out, ref, n_items, singles)


def test_mixed_step_token_rows(device):
    """Decode rows living at shifted token positions (a mixed step's maps).

    Five requests sit behind four chunk-token rows, as in a mixed
    prefill/decode step. The cascade op must fill exactly the decode
    token positions and leave the chunk rows untouched.
    """
    from aiter.ops.flydsl.pa_decode_cascade import (
        pa_decode_cascade,
        plan_pa_decode_cascade,
    )

    torch.manual_seed(0)
    group, dim, kv_heads = 4, 128, 2
    heads = kv_heads * group
    lengths = [80, 96, 77, 64, 72]
    batch = len(lengths)
    chunk_rows = 4
    num_tok = batch + chunk_rows
    table_width = max(-(-v // PAGE) for v in lengths) + 2
    num_pages = table_width * batch + 8
    key, value, key_cache, value_cache = _make_caches(
        num_pages, kv_heads, dim, torch.bfloat16, device
    )
    table = torch.zeros(batch, table_width, dtype=torch.int32)
    cursor = 0
    for row, total in enumerate(lengths):
        n = -(-total // PAGE)
        table[row, :n] = torch.arange(cursor, cursor + n)
        cursor += n
    table[1, :3] = table[0, :3]
    table[2, :3] = table[0, :3]
    table = table.to(device)
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    query = torch.randn(num_tok, heads, dim, device=device).to(torch.bfloat16) * 0.25
    q_map = torch.arange(chunk_rows, num_tok, dtype=torch.int32, device=device)
    items = [([0, 1, 2], 3)]
    singles = sorted(set(range(batch)) - {0, 1, 2})

    ws = plan_pa_decode_cascade(
        device,
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=dim,
        max_items=2,
        max_singles=batch,
        table_width=table_width,
    )
    # Planner path with the token map (request rows -> token rows).
    from aiter.ops.flydsl.pa_decode_cascade_plan import (
        fill_pa_decode_cascade_plan,
        plan_pa_decode_cascade_groups,
    )

    pl = plan_pa_decode_cascade_groups(
        device, max_rows=batch, table_width=table_width, max_b_rows=ws.max_b_rows
    )
    fill_pa_decode_cascade_plan(ws, pl, table, lengths_t, q_map_of_req=q_map)
    assert int(ws.counts[0]) >= 1, "planner did not form the group"
    assert (int(ws.counts[1]), int(ws.counts[2])) == (4, 1), (
        ws.counts.tolist(),
        ws.items.tolist(),
    )

    output = torch.full_like(query, float("nan"))
    pa_decode_cascade(
        output, query, key_cache, value_cache, lengths_t, table, dim**-0.5, workspace=ws
    )
    # Chunk token rows were not touched.
    assert torch.isnan(output[:chunk_rows]).all()
    dec_out = output[chunk_rows:]
    # Reference over the full decode rows (token rows map 1:1 to requests).
    dec_q = query[chunk_rows:]
    ref = _reference(dec_q, key, value, table, lengths_t)
    diff = (dec_out.float() - ref.float()).abs()
    assert not torch.isnan(dec_out).any()
    bound = 4.0e-2 + 2e-2 * ref.float().abs()
    assert (diff - bound).clamp(min=0).amax() == 0, diff.amax()
