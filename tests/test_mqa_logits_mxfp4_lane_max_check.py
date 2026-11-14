# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The lane maxima check of op_tests/test_pa_mqa_logits_mxfp4_opus.py, checked on CPU.

gfx950 has no instance that writes lane maxima, so the op test's ``lane_max`` check only runs on
gfx1250. Here a CPU model of the kernels' stores (pa_mqa_logits_mxfp4_gfx1250_h32.cuh: a slot
stored at its 16-tile group's last tile or its CTA chunk's last tile) runs on the cuts that
mqa_assign makes, and the check must accept it whichever CTA's store lands, and refuse it with
the flush broken: a group's first or last tile dropped, a flush every 8 tiles, or only the last
tile stored.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "op_tests"))
from mqa_logits_mxfp4_lane_max_ref import (
    LANE_MAX_KEYS,
    LANE_MAX_TILES,
    TILE,
    chunk_cuts,
    slot_errors,
)


def _ceil(a, b):
    return -(-a // b)


# == mqa_assign, slot by slot =======================================================================
def _seq_groups(qsl, ends, qpb):
    """[(sequence, first row, rows, tiles)] a sequence's groups, in order."""
    out = []
    for b in range(len(qsl) - 1):
        r0, n = qsl[b], qsl[b + 1] - qsl[b]
        for j in range(_ceil(n, qpb)):
            last = r0 + min((j + 1) * qpb, n) - 1
            e = ends[last]
            out.append(
                (b, r0 + j * qpb, min(qpb, n - j * qpb), _ceil(e, TILE) if e > 0 else 0)
            )
    return out


def _part_point(part, tiles, parts):
    # the kernel's 32-bit split: part * q + part * r / parts
    q, r = tiles // parts, tiles % parts
    return part * q + part * r // parts


def _assign_free(qsl, ends, qpb, cta_target):
    """[(first row, rows, start, count)] a CTA, as SCHED_FREE assigns slots 0 .. grid - 1."""
    groups = _seq_groups(qsl, ends, qpb)
    total = sum(t for *_, t in groups)
    g = sum(t > 0 for *_, t in groups)
    tgt = cta_target - g if cta_target > 2 * g else cta_target
    safe = _ceil(total, tgt) if total > 0 else 1
    grid = len(groups) + cta_target
    work, slot = [], 0
    for _, r0, n, t in groups:
        for o in range(_ceil(t, safe)):
            start = o * safe
            work.append((r0, n, start, min(safe, t - start)))
            slot += 1
    assert slot <= grid, "the host's grid covers every chunk"
    return work


def _assign_prefill(qsl, ends, qpb, parts, gather):
    """[(first row, rows, start, count)] a CTA, as SCHED_PREFILL assigns them."""
    if gather:
        groups = [
            (None, r, 1, _ceil(ends[r], TILE) if ends[r] > 0 else 0)
            for r in range(qsl[-1])
        ]
    else:
        groups = []
        for b in range(len(qsl) - 1):  # a sequence's groups last first
            groups += _seq_groups(qsl[b : b + 2], ends, qpb)[::-1]
    work = []
    for _, r0, n, t in groups:
        for part in range(parts):
            start = _part_point(part, t, parts)
            count = _part_point(part + 1, t, parts) - start
            if count > 0:
                work.append((r0, n, start, count))
    return work


# == the kernels' stores ============================================================================
BUGS = (
    "drop first tile",
    "drop last tile",
    "flush every 8 tiles",
    "store last tile only",
)


def _stores(tiles, ends, chunk, bug=None):
    """[(row, slot group, 64 values)] a CTA stores for its chunk (first row, rows, start, count):
    a running max a slot, stored at a 16-tile group's last tile or the chunk's, and reset.
    ``tiles[r]`` is row r's logits [tiles, 64], -inf past its window and for NaN (fmax skips it).
    """
    r0, n, start, count = chunk
    every = 8 if bug == "flush every 8 tiles" else LANE_MAX_TILES
    flush = [t for t in range(start, start + count) if t % every == every - 1]
    flush = sorted(set(flush) | {start + count - 1})
    out = []
    for r in range(r0, r0 + n):
        a = start
        for f in flush:
            keep = [
                t
                for t in range(a, f + 1)
                if not (bug == "drop first tile" and t % LANE_MAX_TILES == 0)
                and not (
                    bug == "drop last tile" and t % LANE_MAX_TILES == LANE_MAX_TILES - 1
                )
            ]
            if bug == "store last tile only":
                keep = [f]
            v = tiles[r][keep].amax(0) if keep else torch.full((TILE,), float("-inf"))
            g = f // LANE_MAX_TILES
            if g * LANE_MAX_KEYS < ends[r]:
                out.append((r, g, v))
            a = f + 1
    return out


def _lane_max(logits, ends, work, bug=None, reverse=False):
    """Each row's slots after every CTA's stores, the earlier or the later CTA's landing."""
    lm = [torch.full((TILE * _ceil(e, LANE_MAX_KEYS),), float("nan")) for e in ends]
    keys = torch.arange(logits.shape[1])
    tiles = [
        torch.where((keys < e) & ~torch.isnan(row), row, float("-inf")).view(-1, TILE)
        for row, e in zip(logits, ends)
    ]
    for chunk in reversed(work) if reverse else work:
        for r, g, v in _stores(tiles, ends, chunk, bug):
            lm[r][g * TILE : (g + 1) * TILE] = v
    return lm


# == cases ==========================================================================================
def _decode(ctx, rows_per_seq, ratio):
    """full_layer's rows: a sequence's last rows_per_seq tokens at compression ratio."""
    ends, qsl = [], [0]
    for c, n in zip(ctx, rows_per_seq):
        ends += [(p + 1) // ratio for p in range(c - n, c)]
        qsl.append(qsl[-1] + n)
    return qsl, ends


CASES = {
    # name: (qsl, ends, mode, kwargs)
    "free ragged qpb2": (
        *_decode([40000, 21000, 9000, 1500], [6, 4, 2, 1], 2),
        "free",
        {"rows_per_cta": 2, "cta_target": 96},
    ),
    "free ragged qpb4": (
        *_decode([40000, 21000, 9000, 1500], [6, 4, 2, 1], 2),
        "free",
        {"rows_per_cta": 4, "cta_target": 37},
    ),
    "free long rows": (
        *_decode([70000, 3000], [2, 6], 1),
        "free",
        {"rows_per_cta": 2, "cta_target": 200},
    ),
    "free one cta a group": (
        *_decode([30000, 9000], [2, 3], 1),
        "free",
        {"rows_per_cta": 2, "cta_target": 1},
    ),
    "prefill chunk mid group": (
        *_decode([20300], [300], 1),
        "prefill",
        {"rows_per_cta": 4, "parts": 3},
    ),
    "prefill parts 7": (
        *_decode([33000, 2000], [8, 5], 2),
        "prefill",
        {"rows_per_cta": 2, "parts": 7},
    ),
    "prefill one part": (
        *_decode([33000, 2000], [8, 5], 2),
        "prefill",
        {"rows_per_cta": 2, "parts": 1},
    ),
    "candidate parts 5": (
        [0, 1, 2, 3],
        [40000, 18000, 6100],
        "prefill",
        {"rows_per_cta": 1, "parts": 5, "gather": True},
    ),
}


def _case(name):
    qsl, ends, mode, kw = CASES[name]
    kw = dict(kw)
    gather = kw.pop("gather", False)
    qpb = kw.pop("rows_per_cta")
    if mode == "free":
        work = _assign_free(qsl, ends, qpb, kw["cta_target"])
    else:
        work = _assign_prefill(qsl, ends, qpb, kw["parts"], gather)
    cuts = chunk_cuts(mode, qsl, ends, rows_per_cta=qpb, gather=gather, **kw)
    g = torch.Generator().manual_seed(len(name))
    width = _ceil(max(ends), LANE_MAX_KEYS) * LANE_MAX_KEYS
    logits = torch.randn(len(ends), width, generator=g)
    logits[0, 5] = float("nan")  # skipped by the max
    return qsl, ends, work, cuts, logits


def _row_errors(logits, ends, cuts, lm):
    return [
        (r, slot_errors(lm[r], logits[r, : ends[r]], cuts[r]))
        for r in range(len(ends))
        if ends[r] > 0
    ]


@pytest.mark.parametrize("name", list(CASES))
def test_cuts_match_slot_walk(name):
    """chunk_cuts' cuts are where the slot-by-slot walk starts the chunks."""
    _, ends, work, cuts, _ = _case(name)
    walked = [set() for _ in ends]
    for r0, n, start, _ in work:
        for r in range(r0, r0 + n):
            if start:
                walked[r].add(start)
    assert walked == cuts


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("name", list(CASES))
def test_kernel_stores_pass(name, reverse):
    _, ends, work, cuts, logits = _case(name)
    lm = _lane_max(logits, ends, work, reverse=reverse)
    errs = _row_errors(logits, ends, cuts, lm)
    assert all(not e for _, (e, _) in errs), errs
    split = sum(n for _, (_, n) in errs)
    if all(not c or all(x % LANE_MAX_TILES == 0 for x in c) for c in cuts):
        assert split == 0  # no group cut: every slot exact


def test_cases_cover_split_groups():
    """Some case leaves slots holding a chunk's share, so the partial path is exercised."""
    split = 0
    for name in CASES:
        _, ends, work, cuts, logits = _case(name)
        for _, (_, n) in _row_errors(logits, ends, cuts, _lane_max(logits, ends, work)):
            split += n
    assert split > 0


@pytest.mark.parametrize("bug", BUGS)
@pytest.mark.parametrize("name", list(CASES))
def test_broken_flush_fails(name, bug):
    """Every row with a whole 16-tile slot group no chunk cuts is refused under each bug."""
    _, ends, work, cuts, logits = _case(name)
    for reverse in (False, True):
        lm = _lane_max(logits, ends, work, bug=bug, reverse=reverse)
        for r, (errs, _) in _row_errors(logits, ends, cuts, lm):
            whole = [
                g
                for g in range(ends[r] // LANE_MAX_KEYS)
                if not any(
                    g * LANE_MAX_TILES < c < (g + 1) * LANE_MAX_TILES for c in cuts[r]
                )
            ]
            if whole:
                assert errs, f"row {r} ({bug}) accepted"
