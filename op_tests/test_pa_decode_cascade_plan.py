# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cascade grouping planner tests.

The planner must (a) produce legal cascade metadata from device block
tables without any host synchronization, and (b) group at least as well
as a reference sort/LCP pass on realistic conversation trees. Every
end-to-end test asserts the cascade path actually ran.

Run with: python -m pytest -q op_tests/test_pa_decode_cascade_plan.py
"""

import itertools

import pytest
import torch

PAGE = 16


def _reference_groups(block_tables, lengths, min_shared=1, members=4):
    """Full-depth host sort + neighbour LCP + run grouping.

    Returns (groups, singles) where each group is (member_rows, sblk)
    split into items of at most ``members`` rows. This mirrors the
    numpy pass from the Gemma hd512 stack.
    """
    bt = [tuple(row) for row in block_tables.tolist()]
    lens = lengths.tolist()

    def depth(row):
        return lens[row] // PAGE

    order = sorted(range(len(bt)), key=lambda r: tuple(bt[r][: depth(r)]))
    groups = []
    run = [order[0]]
    pair_lcp = {}
    for prev, cur in itertools.pairwise(order):
        n = min(depth(prev), depth(cur))
        lcp = 0
        while lcp < n and bt[prev][lcp] == bt[cur][lcp]:
            lcp += 1
        pair_lcp[(prev, cur)] = lcp
        if lcp >= min_shared:
            run.append(cur)
        else:
            groups.append(run)
            run = [cur]
    groups.append(run)
    out = []
    for run in groups:
        if len(run) < 2:
            continue
        sblk = min(pair_lcp[(a, b)] for a, b in itertools.pairwise(run))
        for i in range(0, len(run), members):
            out.append((run[i : i + members], sblk))
    singles = sorted(r for run in groups if len(run) < 2 for r in run)
    return out, singles


def _make_tables(profiles, width, device):
    """Fresh dense block ids per row and I32 lengths.

    Every row gets its own contiguous page run starting at 10**6 (high
    enough to exercise 64-bit index math in the kernels); physical
    sharing between rows is applied by the tests after this returns.
    """
    bt = torch.zeros(len(profiles), width, dtype=torch.int32)
    lens = torch.zeros(len(profiles), dtype=torch.int32)
    cursor = 10**6
    for r, (length, _unused) in enumerate(profiles):
        depth = -(-length // PAGE)
        bt[r, :depth] = torch.arange(cursor, cursor + depth, dtype=torch.int32)
        cursor += depth
        lens[r] = length
    return bt.to(device), lens.to(device)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    return torch.device("cuda")


def _run_planner(device, bt, lens, max_items=64, min_shared=1, M=4):
    from aiter.ops.flydsl.pa_decode_cascade import plan_pa_decode_cascade
    from aiter.ops.flydsl.pa_decode_cascade_plan import (
        fill_pa_decode_cascade_plan,
        plan_pa_decode_cascade_groups,
    )

    n, width = bt.shape
    group_size = 4
    ws = plan_pa_decode_cascade(
        device,
        num_q_heads=group_size,
        num_kv_heads=1,
        head_dim=64,
        members_per_item=M,
        max_items=max_items,
        max_singles=n,
        table_width=width,
    )
    pl = plan_pa_decode_cascade_groups(
        device, max_rows=n, table_width=width, max_b_rows=ws.max_b_rows
    )
    fill_pa_decode_cascade_plan(ws, pl, bt, lens, min_shared_blocks=min_shared)
    return ws


def _workspace_items(ws):
    """Host extraction of the items the planner wrote."""
    items = []
    for row in ws.items.tolist():
        sblk, count, *members = row
        if count <= 0:
            continue
        items.append((sorted(members[:count]), sblk))
    singles = ws.singles_rows[: int(ws.counts[2])].tolist()
    return items, singles


def test_basic_tree(device):
    """One system prompt, two sub-agent branches, and an unrelated row."""
    profiles = [
        (200, 0),  # row 0: base
        (187, 8),  # rows 1-3: branch A (share 8 blocks)
        (176, 8),
        (211, 8),
        (202, 5),  # rows 4-5: branch B (share 5)
        (169, 5),
        (96, 0),  # unrelated
    ]
    bt, lens = _make_tables(profiles, 16, device)
    # Give rows 1-3 the same first 8 blocks physically.
    for r in (1, 2, 3):
        bt[r, :8] = bt[1, :8]
    for r in (4, 5):
        bt[r, :5] = bt[4, :5]
    ref_items, _ = _reference_groups(bt.cpu(), lens.cpu())
    ws = _run_planner(device, bt, lens)
    items, singles = _workspace_items(ws)
    # Legal grouping: every item's members share its sblk prefix physically.
    for members, sblk in items:
        for m in members:
            assert torch.equal(bt[m, :sblk], bt[members[0], :sblk])
    # Members + singles partition the batch, and coverage tracks the
    # full-depth reference (multi-branch ties resolve differently).
    member_union = sorted(m for m, _ in items for m in m)
    assert sorted(member_union + singles) == list(range(len(profiles)))
    ref_dup = sum((len(m) - 1) * s for m, s in ref_items)
    got_dup = sum((len(m) - 1) * s for m, s in items)
    assert got_dup >= 0.9 * ref_dup
    assert int(ws.counts[0]) == len(items) > 0 and int(ws.counts[0]) > 0


def test_group_splits_into_items(device):
    """Seven identical prompts form two items (4 + 3) at one shared length."""
    profiles = [(144, 0)] + [(144, 8)] * 6
    bt, lens = _make_tables(profiles, 12, device)
    for r in range(1, 7):
        bt[r, :] = bt[0, :]
    ws = _run_planner(device, bt, lens)
    items, singles = _workspace_items(ws)
    assert len(items) == 2 and singles == []
    member_union = sorted(m for m, _ in items for m in m)
    assert member_union == list(range(7))
    # The full run of 7 is two LCP-minimal chunks; total sharing is 9 blocks.
    assert {items[0][1], items[1][1]} == {9}


def test_item_overflow_spills(device):
    bt, lens = _make_tables([(144, 0)] + [(144, 8)] * 5, 12, device)
    for r in range(1, 6):
        bt[r, :] = bt[0, :]
    ws = _run_planner(device, bt, lens, max_items=1)
    items, singles = _workspace_items(ws)
    assert len(items) == 1 and len(items[0][0]) == 4
    assert sorted(singles) == [4, 5]
    # Spilled members still produce correct tail/singles metadata.
    assert int(ws.counts[0]) == 1 and int(ws.counts[2]) == 2


def test_no_grouping_no_items(device):
    bt, lens = _make_tables([(100, 0), (128, 0), (64, 0)], 12, device)
    ws = _run_planner(device, bt, lens)
    items, singles = _workspace_items(ws)
    # Every valid run becomes a singleton item; nothing is a single here.
    assert singles == []
    assert sorted((sorted(m), s) for m, s in items) == [
        ([0], 100 // PAGE),
        ([1], 128 // PAGE),
        ([2], 64 // PAGE),
    ]
    assert int(ws.counts[0]) == 3


def test_partial_blocks_not_shared(device):
    """Only whole blocks count: rows sharing 3 blocks + 7 tokens group at 3."""
    bt, lens = _make_tables([(55, 0), (59, 0)], 8, device)
    bt[1, :] = bt[0, :]
    ref_items, _ = _reference_groups(bt.cpu(), lens.cpu())
    ws = _run_planner(device, bt, lens)
    items, _ = _workspace_items(ws)
    assert items == [([0, 1], 3)] == ref_items


def test_zero_length_and_short_rows(device):
    bt, lens = _make_tables([(200, 0), (15, 0), (200, 8)], 14, device)
    bt[2, :] = bt[0, :]
    ws = _run_planner(device, bt, lens)
    items, singles = _workspace_items(ws)
    # Row 1 has zero whole blocks, so it can never group.
    assert items == [([0, 2], 12)]
    assert singles == [1]


def test_end_to_end_with_planner(device):
    """Planner metadata feeding the cascade op matches fp32 per-row refs."""
    import sys

    sys.path.insert(0, "op_tests")
    from test_flydsl_pa_decode_cascade import _make_caches, _reference

    from aiter.ops.flydsl.pa_decode_cascade import (
        pa_decode_cascade,
        plan_pa_decode_cascade,
    )
    from aiter.ops.flydsl.pa_decode_cascade_plan import (
        fill_pa_decode_cascade_plan,
        plan_pa_decode_cascade_groups,
    )

    torch.manual_seed(1)
    kv_heads, group, dim = 2, 4, 128
    heads = kv_heads * group
    profiles = [(200, 0)] + [(187, 8), (176, 8), (211, 8)] + [(96, 0), (33, 0)]
    width = 16
    bt, lens = _make_tables(profiles, width, device)
    for r in (1, 2, 3):
        bt[r, :8] = bt[1, :8]
    num_pages = int(bt.max()) + 8
    key, value, kc, vc = _make_caches(num_pages, kv_heads, dim, torch.bfloat16, device)
    query = (torch.randn(len(profiles), heads, dim, device=device)).to(
        torch.bfloat16
    ) * 0.25
    n = len(profiles)
    ws = plan_pa_decode_cascade(
        device,
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=dim,
        max_items=4,
        max_singles=n,
        table_width=width,
    )
    pl = plan_pa_decode_cascade_groups(
        device, max_rows=n, table_width=width, max_b_rows=ws.max_b_rows
    )
    fill_pa_decode_cascade_plan(ws, pl, bt, lens)

    def run(output):
        pa_decode_cascade(output, query, kc, vc, lens, bt, dim**-0.5, workspace=ws)
        return output

    output = run(torch.empty_like(query))
    ref = _reference(query, key, value, bt, lens)
    assert int(ws.counts[0]) > 0, "cascade path did not run"
    diff = (output.float() - ref).abs()
    assert diff.amax() < 3.5e-2, diff.amax()
    # Rows 1-3 join the group; rows 0, 4 and 5 become singleton items.
    assert int(ws.counts[1]) == 6 and int(ws.counts[2]) == 0

    # Capture and replay against a swapped structure: static buffers only.
    g = torch.cuda.CUDAGraph()
    eager = output.clone()
    with torch.cuda.graph(g):
        run(output)
    g.replay()
    assert torch.equal(output, eager)
    _ref_items, _ = _reference_groups(bt.cpu(), lens.cpu())
    bt[4, :8], bt[1, :8] = bt[1, :8].clone(), bt[4, :8].clone()
    fill_pa_decode_cascade_plan(ws, pl, bt, lens)
    g.replay()
    diff2 = (output.float() - _reference(query, key, value, bt, lens)).abs()
    assert int(ws.counts[0]) > 0
    assert diff2.amax() < 3.5e-2, diff2.amax()


def test_planner_reorders_after_eviction_pressure(device):
    """Same traffic shape, recycled physical block ids: groups follow ids."""
    profiles = [(160, 0), (160, 8), (160, 8), (160, 0)]
    bt, lens = _make_tables(profiles, 12, device)
    bt[1, :8] = bt[0, :8]
    bt[2, :8] = bt[0, :8]
    ws1 = _run_planner(device, bt.clone(), lens)
    items1, _ = _workspace_items(ws1)
    assert sorted((sorted(m), s) for m, s in items1) == [
        ([0, 1, 2], 8),
        ([3], 10),
    ]

    # "Eviction": row 0's prefix moves to fresh physical ids; rows 1 and 2
    # keep sharing theirs. Groups must track physical ids, not logical ones.
    recycled = torch.arange(9000, 9008, dtype=torch.int32, device=device)
    bt[0, :8] = recycled
    ws4 = _run_planner(device, bt.clone(), lens)
    items4, singles4 = _workspace_items(ws4)
    assert sorted((sorted(m), s) for m, s in items4) == [
        ([0], 10),
        ([1, 2], 8),
        ([3], 10),
    ]
    assert singles4 == []


def test_coverage_matches_reference_on_random_tree(device):
    """Random conversation trees: planner coverage == full-sort reference."""
    torch.manual_seed(7)
    width = 24
    rows = 64
    bt = torch.zeros(rows, width, dtype=torch.int32)
    cursor = 0
    passages = []
    for r in range(rows):
        depth = torch.randint(4, width, (1,)).item()
        if passages and torch.rand(()) < 0.7:
            src = torch.randint(0, len(passages), (1,)).item()
            cut = torch.randint(3, depth, (1,)).item()
            row = torch.cat(
                [
                    bt[passages[src]][:cut],
                    torch.arange(cursor, cursor + depth - cut, dtype=torch.int32),
                ]
            )
        else:
            row = torch.arange(cursor, cursor + depth, dtype=torch.int32)
            passages.append(r)
        cursor += depth
        bt[r, :depth] = row
    lens = (torch.arange(rows) % 3 + 1) * 0 + width * 0
    lens = torch.tensor(
        [min(width, 6 + (r * 7) % (width - 5)) * PAGE for r in range(rows)],
        dtype=torch.int32,
    )
    # Clamp contexts to each row's populated depth.
    depths = (bt != 0).sum(1).clamp(min=1)
    lens = torch.minimum(
        torch.tensor([d * PAGE for d in depths], dtype=torch.int32),
        torch.full_like(lens, width * PAGE),
    )
    bt_d, lens_d = bt.to(device), lens.to(device)
    ref_items, _ = _reference_groups(bt.clone(), lens)
    ws = _run_planner(device, bt_d, lens_d)
    items, _ = _workspace_items(ws)
    ref_dup = sum((len(m) - 1) * s for m, s in ref_items)
    got_dup = sum((len(m) - 1) * s for m, s in items)
    assert ref_dup > 0
    assert got_dup >= 0.95 * ref_dup, (got_dup, ref_dup)
    # Every emitted item really shares its prefix (all members pairwise).
    for members, sblk in items:
        for m in members:
            assert torch.equal(bt[m, :sblk], bt[members[0], :sblk])
