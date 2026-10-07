# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every per-step decision of which attention mailbox cells the score stage
writes and the PV stage polls (AM / AL / AP, a split each), defined once
(DESIGN_v3 4.9, I3), the way ``index_plan`` does the indexer's.

A token's keys are its selection's valid ids then its window rows: ``kv_len``.
The score stage's tasks cover every split of a (token, head tile) and write the
ones ``written_splits`` counts; the PV stage polls ``polled_split`` of each of
its splits. Called by the kernel on traced Int32 and by the CPU property test on
Python ints (``index_plan``'s rules: non-negative ``//`` operands only).
"""

from aiter.ops.flydsl.kernels.dsv41_mono.atomv41 import index_plan as ip

SPLITS = 64  # a (token, head tile)'s splits: the original decode's
BK = 16  # keys a split (a tile)
SCORE_WAVES = 8  # a score task's waves, a split each a round


def kv_len(n_sel, count):
    """A token's keys: its selection's valid ids ``n_sel`` (``index_plan.selected``
    of its scorer's bound) and its window count (0 for a graph's pad row)."""
    return n_sel + count


def written_splits(n):
    """The splits of ``n`` keys the score stage writes: its keys' splits, and
    split 0 for a token with none (a pad row), whose values the PV masks."""
    return ip.imax(ip.cdiv(n, BK), 1)


def last_split(n):
    """The last split the score stage writes for a token of ``n`` keys."""
    return written_splits(n) - 1


def polled_split(i, last):
    """The split the PV stage polls for its split ``i``, ``last`` the token's
    ``last_split``: the last written one past it."""
    return ip.imin(i, last)


def score_splits_a_wave(per):
    """A score wave's splits when a (token, head tile) is ``per`` tasks."""
    return SPLITS // SCORE_WAVES // per


def score_split(sg, per, k, wave):
    """Score task ``sg`` of ``per`` a (token, head tile): wave ``wave``'s k-th
    split."""
    return (sg + per * k) * SCORE_WAVES + wave
