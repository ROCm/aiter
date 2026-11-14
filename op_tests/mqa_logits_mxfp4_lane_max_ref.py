# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What mqa_logits_mxfp4's lane maxima may hold, for op_tests/test_pa_mqa_logits_mxfp4_opus.py.

Torch only and no GPU, so tests/test_mqa_logits_mxfp4_lane_max_check.py can check the check.

Slot 64 g + 32 u + l of a row holds the max of keys (16 g + i) * 64 + 32 u + l, i in [0, 16):
the same position in each of its slot group's 16 64-key tiles (pa_mqa_logits_mxfp4.h). A CTA
stores a slot at its group's last tile or its chunk's last tile, so a slot group is exact unless
a chunk boundary of the schedule-free split falls inside it; then each CTA holding part of it
stores the max of its own part, and either store may land.
"""

import itertools

import torch

TILE = 64  # keys a tile
LANE_MAX_TILES = 16  # tiles a slot spans
LANE_MAX_KEYS = LANE_MAX_TILES * TILE


def _ceil(a, b):
    return -(-a // b)


def chunk_cuts(
    mode, query_start_loc, row_ends, *, rows_per_cta, gather, cta_target=1, parts=1
):
    """Per row, the tiles other than 0 where one of its CTA chunks starts, as mqa_assign in
    pa_mqa_logits_mxfp4_gfx1250_common.cuh cuts them. A group is up to ``rows_per_cta``
    consecutive rows of one sequence (a candidate row on its own with ``gather``) and walks its
    last row's window; its rows share its cuts.

    ``mode`` "free": the call's tiles over ``cta_target`` slots, less one a group when the target
    is more than twice the groups, give ``safe`` tiles a chunk; a group is cut every ``safe``
    tiles. "prefill": every group is cut into ``parts`` pieces at floor(p * tiles / parts).
    """
    qsl = [int(x) for x in query_start_loc]
    ends = [int(x) for x in row_ends]
    if gather:
        # mqa_assign's candidate path: a row is its own group, whatever its sequence
        groups = [([r], ends[r]) for r in range(min(len(ends), qsl[-1]))]
    else:
        groups = []
        for b in range(len(qsl) - 1):
            for r0 in range(qsl[b], qsl[b + 1], rows_per_cta):
                rows = list(range(r0, min(r0 + rows_per_cta, qsl[b + 1])))
                groups.append((rows, ends[rows[-1]]))
    tiles = [_ceil(e, TILE) if e > 0 else 0 for _, e in groups]
    if mode == "free":
        assert not gather, "Free does not take candidate rows"
        total, g = sum(tiles), sum(t > 0 for t in tiles)
        tgt = cta_target - g if cta_target > 2 * g else cta_target
        safe = _ceil(total, tgt) if total > 0 else 1
        cuts = [set(range(safe, t, safe)) for t in tiles]
    elif mode == "prefill":
        cuts = [{p * t // parts for p in range(1, parts)} - {0, t} for t in tiles]
    else:
        raise ValueError(f"no schedule-free mode {mode!r}")
    out = [set() for _ in ends]
    for (rows, _), c in zip(groups, cuts):
        for r in rows:
            out[r] = c
    return out


def slot_maxima(row, cuts):
    """A row's logits [end] and its chunk cuts -> (full [slots], parts {slot group: [n, 64]}):
    each slot's max over its keys, and for a slot group a cut falls inside, the max over each
    chunk's share of it. The kernels' max skips NaN; no keys, -inf."""
    end = row.numel()
    groups = _ceil(end, LANE_MAX_KEYS)
    v = torch.full((groups * LANE_MAX_KEYS,), float("-inf"))
    v[:end] = torch.where(torch.isnan(row), float("-inf"), row.float())
    tiles = v.view(groups, LANE_MAX_TILES, TILE)
    parts = {}
    for g in range(groups):
        lo = g * LANE_MAX_TILES
        inner = sorted(c - lo for c in cuts if lo < c < lo + LANE_MAX_TILES)
        if inner:
            bounds = [0] + inner + [LANE_MAX_TILES]
            parts[g] = torch.stack(
                [tiles[g, a:b].amax(0) for a, b in itertools.pairwise(bounds)]
            )
    return tiles.amax(1).flatten(), parts


def slot_errors(got, row, cuts):
    """Why a row's stored slots ``got`` are not what the kernels may store: each slot its keys'
    max, or, in a slot group a chunk cut falls inside, the max of one chunk's share of it.
    Returns (errors, slots holding a share's max rather than the whole group's)."""
    full, parts = slot_maxima(row, cuts)
    if got.numel() != full.numel():
        return [f"{got.numel()} slots, want {full.numel()}"], 0
    exact = got == full
    ok = exact.clone()
    for g, p in parts.items():
        s = slice(g * TILE, (g + 1) * TILE)
        ok[s] |= (p == got[s]).any(0)
    if bool(ok.all()):
        return [], int((~exact).sum())
    s = int((~ok).nonzero()[0])
    g = s // TILE
    how = (
        f"one of {parts[g][:, s % TILE].tolist()}" if g in parts else "its group's max"
    )
    return [f"slot {s}: {float(got[s])}, want {float(full[s])} ({how})"], 0
