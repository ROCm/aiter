#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fit the FlyDSL all-reduce dispatch tables from a benchmark sweep.

Reads the ``--output-csv`` files produced by ``bench_comm_allreduce.py`` and
emits the two levels of heuristic the dispatch needs, as paste-ready Python
literals:

1. **Which family** -- one-shot (exact), two-shot mesh, or two-shot ring. A
   three-family partition ordered by payload size is exactly two thresholds per
   (link, world size), so that is what is fitted.
2. **Which variant inside a family** -- a ``(min_bytes, ...)`` ladder per family
   per world size, in the same idiom as ``RING_ST_LADDER``.

Both are fitted by **exhaustive search over round threshold values, minimising
the worst-case regret against a per-shape oracle**, rather than by finding where
two curves cross. Those are not the same thing and the difference matters here:
a crossover point says nothing about what it costs to be on the wrong side of
it, and the whole design constraint is "simplest table that stays within ~10% of
optimal". Searching directly on regret answers that question and reports the
answer; solving for a crossing and hoping does not.

Thresholds are drawn from ``2^k`` and ``1.5 * 2^k`` only. A table that reads
``16 MiB`` invites a reviewer to check it; one that reads ``17825792`` invites
them to assume it was measured to that precision, which it was not -- the sweep
ladder is 4 points per octave, so nothing here resolves finer than ~19%.

Two answers come out per world size, not one:

* ``fast`` -- minimise latency, whatever the numerics cost.
* ``exact`` -- keep the bit-exact one-shot as long as it stays within
  ``--exact-slack`` of the fastest option. The one-shot lands at ~55 dB and the
  quantized families at ~19 dB, so the default trades up to 10% of latency to
  leave the model's numerics untouched. This is the shipped default; ``fast``
  is what an explicit opt-out selects.

Usage::

    python3 op_tests/multigpu_tests/fit_allreduce_policy.py \\
        op_tests/dump_data/sweep/*.csv --link pcie

    # what the tables cost against a per-shape oracle, in detail
    python3 op_tests/multigpu_tests/fit_allreduce_policy.py \\
        op_tests/dump_data/sweep/*.csv --link pcie --verbose

Sizes where a family was not measured at all are skipped for that family rather
than treated as infinitely slow -- the one-shot is gated above by
``AITER_BENCH_FLY1S_MAX_KB`` and its absence past that ceiling is a property of
the sweep, not of the kernel.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations, pairwise

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_comm_allreduce import CANDIDATES

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("fit")

# Family a candidate belongs to, for the level-2 fit. Only the flydsl rows have
# a dispatch question open; everything else in the sweep is a reference point.
FAMILIES = ("oneshot", "mesh", "ring")


def family_of(cand) -> str | None:
    if cand.family == "fly1s":
        return "oneshot"
    if cand.family == "fly":
        return cand.algorithm  # "mesh" | "ring"
    return None


# Candidate keys per family, and the subset of each that is a *pinned* variant.
# The auto rows (``fly_int4``, ``fly_int4_ring``) walk a ladder of their own, so
# they are the thing being fitted against, not an input to the fit: including
# them as ladder candidates would let the fit "choose" today's ladder and
# conclude it is optimal.
BY_FAMILY = {f: [c for c in CANDIDATES if family_of(c) == f] for f in FAMILIES}
AUTO_KEYS = {
    "oneshot": ("fly_1stage",),
    "mesh": ("fly_int4",),
    "ring": ("fly_int4_ring",),
}


def round_thresholds(lo: int, hi: int) -> list[int]:
    """``2^k`` and ``1.5 * 2^k`` spanning [lo, hi], plus 0 and infinity.

    0 and infinity are real answers: "this family never wins here" and "this
    family wins everywhere measured" both have to be expressible, or the fit
    will invent a boundary inside the data to avoid saying so.
    """
    out = [0]
    k = max(0, math.floor(math.log2(max(lo, 1))) - 1)
    while 2**k <= hi * 2:
        out.append(2**k)
        out.append(3 * 2 ** (k - 1))
        k += 1
    out.append(1 << 62)
    return sorted(set(out))


@dataclass
class Sample:
    """One measured shape: payload bytes, and each candidate's latency."""

    nbytes: int
    tp: int
    hidden: int
    us: dict  # candidate key -> microseconds
    variant: dict  # candidate key -> JIT symbol that actually ran


def load(paths, metric: str) -> list[Sample]:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    keys = [c.key for c in CANDIDATES if f"{c.key} {metric}" in df.columns]
    if not keys:
        raise SystemExit(
            f"no '<candidate> {metric}' columns in the input; was the sweep run "
            "with --output-csv from a current bench_comm_allreduce.py?"
        )
    # A shape measured in more than one input file (the small and large ladders
    # overlap at 3.5 MiB by construction) keeps its fastest reading per
    # candidate. Averaging would blend two runs' warmup states; the minimum is
    # the one with least contamination.
    #
    # Keyed on hidden size as well as bytes, because two shapes with the same
    # byte count but different K are the *question* in the K-sensitivity pass,
    # not duplicates to collapse.
    best: dict[tuple[int, int, int], dict] = {}
    variants: dict[tuple[int, int, int], dict] = {}
    for _, r in df.iterrows():
        ident = (int(r["TP"]), int(r["_nbytes"]), int(r.get("K", 0)))
        cell = best.setdefault(ident, {})
        vcell = variants.setdefault(ident, {})
        for k in keys:
            v = r.get(f"{k} {metric}")
            if v is not None and pd.notna(v):
                cell[k] = min(v, cell[k]) if k in cell else float(v)
            sym = r.get(f"{k} variant")
            if sym is not None and pd.notna(sym):
                vcell[k] = str(sym)
    return [
        Sample(nbytes=n, tp=tp, hidden=k, us=cell, variant=variants[(tp, n, k)])
        for (tp, n, k), cell in sorted(best.items(), key=lambda kv: kv[0])
    ]


def collapse_aliases(samples, keys) -> tuple[list[str], dict]:
    """Drop candidate keys that compile to a binary another key already covers.

    Two keys can name the same kernel. The ring's ``rs_codec`` defaults to INT6
    at TP8 (``_RS_INT6_MIN_WORLD``), so ``fly_int4_ring_st32`` and
    ``fly_int4_ring_st32_int6`` are *the same symbol* there -- and a fit allowed
    to choose between them will "discover" a rung boundary between two names for
    one kernel, from nothing but measurement noise.

    Keys are aliases when their reported JIT symbols agree at every size where
    both ran. The survivor is the one declared first in ``CANDIDATES``, which is
    the less-qualified name -- ``fly_int4_ring_st32`` over the ``_int6`` variant
    that only spells out what the default already does.

    Returns the surviving keys and ``survivor -> [dropped]`` for reporting; a
    collapse is a finding about the sweep, not an implementation detail to hide.
    """
    order = {c.key: i for i, c in enumerate(CANDIDATES)}
    live = sorted([k for k in keys], key=lambda k: order[k])
    survivors: list[str] = []
    merged: dict[str, list[str]] = {}
    for k in live:
        for s in survivors:
            shared = [sm for sm in samples if k in sm.variant and s in sm.variant]
            if shared and all(sm.variant[k] == sm.variant[s] for sm in shared):
                merged.setdefault(s, []).append(k)
                break
        else:
            survivors.append(k)
    return survivors, merged


def score_policy(samples, one_max: int, mesh_max: int):
    """Per-shape regret of a fixed family policy. ``(nbytes, K, family, ratio)``.

    The holdout check: a policy fitted on one hidden size is scored on others.
    Sizes where the named family was not measured are reported as ``inf`` rather
    than skipped -- on a holdout that is a hole in the evidence, and silently
    dropping it would let a policy look good precisely where it is unsupported.
    """
    out = []
    for s in samples:
        oracle = min(family_best(s, f)[1] for f in FAMILIES)
        if not math.isfinite(oracle):
            continue
        fam = pick_family(s.nbytes, one_max, mesh_max)
        us = family_best(s, fam)[1]
        out.append((s.nbytes, s.hidden, fam, us / oracle))
    return out


def best_in(sample: Sample, keys) -> tuple[str | None, float]:
    """Fastest of *keys* that ran at this shape."""
    live = [(k, sample.us[k]) for k in keys if k in sample.us]
    return min(live, key=lambda kv: kv[1]) if live else (None, float("inf"))


def family_best(sample: Sample, family: str) -> tuple[str | None, float]:
    return best_in(sample, [c.key for c in BY_FAMILY[family]])


# --------------------------------------------------------------------------
# Level 2: which family
# --------------------------------------------------------------------------


def pick_family(nbytes: int, oneshot_max: int, mesh_max: int) -> str:
    if nbytes <= oneshot_max:
        return "oneshot"
    return "mesh" if nbytes <= mesh_max else "ring"


def fit_families(samples, *, exact_slack: float | None):
    """Best (oneshot_max, mesh_max) by worst-case regret, over round values.

    With *exact_slack*, the objective changes from "be fastest" to "stay exact
    for as long as that is nearly free": among every pair whose worst-case
    regret clears the slack, take the one with the largest ``oneshot_max``. That
    is a different question from the speed fit and it is why both are reported
    -- the speed answer is a lower bound on the exact one, never the same
    number by construction.
    """
    sizes = [s.nbytes for s in samples]
    grid = round_thresholds(min(sizes), max(sizes))
    oracle = {s.nbytes: min(family_best(s, f)[1] for f in FAMILIES) for s in samples}
    # A shape where nothing at all ran cannot grade a policy.
    graded = [s for s in samples if math.isfinite(oracle[s.nbytes])]

    def regrets(one_max, mesh_max):
        out = []
        for s in graded:
            fam = pick_family(s.nbytes, one_max, mesh_max)
            us = family_best(s, fam)[1]
            if not math.isfinite(us):
                # The policy named a family that was not measured here. Only
                # legitimate for the one-shot above the sweep's own ceiling;
                # anywhere else it is a hole and must not be scored as free.
                return None
            out.append((s.nbytes, fam, us / oracle[s.nbytes]))
        return out

    scored = []
    for one_max in grid:
        for mesh_max in grid:
            if mesh_max < one_max:
                continue
            r = regrets(one_max, mesh_max)
            if r is None:
                continue
            worst = max(x[2] for x in r)
            scored.append((worst, sum(x[2] for x in r) / len(r), one_max, mesh_max, r))
    if not scored:
        raise SystemExit(
            "no feasible (oneshot_max, mesh_max) pair; is a family absent?"
        )

    if exact_slack is None:
        worst, mean, one_max, mesh_max, r = min(scored, key=lambda t: (t[0], t[1]))
    else:
        ok = [t for t in scored if t[0] <= exact_slack]
        if not ok:
            best = min(scored, key=lambda t: (t[0], t[1]))
            logger.warning(
                "  no policy stays within %.0f%% of the oracle; the tightest is "
                "%.3fx. Falling back to it -- widen --exact-slack or add a rung.",
                (exact_slack - 1) * 100,
                best[0],
            )
            ok = [best]
        # Largest exact window first, then cheapest among the ties.
        worst, mean, one_max, mesh_max, r = max(ok, key=lambda t: (t[2], -t[0], -t[1]))
    return one_max, mesh_max, worst, mean, r


# --------------------------------------------------------------------------
# Level 1: which variant inside a family
# --------------------------------------------------------------------------


def fit_ladder(
    samples, family: str, max_rungs: int = 3, slack: float = 1.10, window=None
):
    """A ``(min_bytes, key)`` ladder for *family*, at most *max_rungs* long.

    Enumerates every set of rung boundaries over the round grid and, for each
    segment, takes the pinned variant with the lowest worst-case regret inside
    it.

    **Rung count is the primary objective, not regret.** Minimising regret alone
    produces ladders like ``g128 -> st1 -> g128``: three compiled engines and
    three IPC inboxes to buy 1.4%, with a non-monotone shape that is the fit
    reading noise rather than structure. So the search takes the *fewest* rungs
    whose worst case clears *slack*, and only falls back to minimising regret
    when nothing does. Returns the cost of every rung count so the choice is
    auditable -- if one rung is within tolerance the ladder should be one rung,
    and the report should say what the extra rungs would have bought.
    """
    pinned = [
        c.key for c in BY_FAMILY[family] if c.key not in AUTO_KEYS.get(family, ())
    ]
    live = [s for s in samples if any(k in s.us for k in pinned)]
    if window is not None:
        # Only the sizes the family policy actually dispatches this family at.
        #
        # Fitting over the whole sweep optimises rungs for payloads the family
        # never sees: at TP8 the one-shot is dispatched only below 48 KiB, yet
        # an unrestricted fit spends its second rung on a 192 KiB boundary that
        # production can never reach. Narrowing the window usually *shortens*
        # the ladder, which is the point -- a rung is a compiled engine.
        lo, hi = window
        live = [s for s in live if lo < s.nbytes <= hi]
    if not live:
        return None
    pinned, aliased = collapse_aliases(live, pinned)
    oracle = {s.nbytes: best_in(s, pinned)[1] for s in live}
    grid = [
        g
        for g in round_thresholds(
            min(o.nbytes for o in live), max(o.nbytes for o in live)
        )
        if g > 0
    ]

    def segment_pick(seg):
        """Variant minimising worst-case regret over *seg*, and that worst."""
        best = None
        for k in pinned:
            if any(k not in s.us for s in seg):
                continue  # must cover the whole segment to be selectable there
            worst = max(s.us[k] / oracle[s.nbytes] for s in seg)
            if best is None or worst < best[1]:
                best = (k, worst)
        return best

    def evaluate(bounds):
        edges = [0, *bounds, 1 << 62]
        rungs, worst = [], 1.0
        for lo, hi in pairwise(edges):
            seg = (
                [s for s in live if lo < s.nbytes <= hi]
                if lo
                else [s for s in live if s.nbytes <= hi]
            )
            if not seg:
                continue
            pick = segment_pick(seg)
            if pick is None:
                return None
            rungs.append((lo, pick[0]))
            worst = max(worst, pick[1])
        return rungs, worst

    # Best achievable at each rung count, so the marginal value of a rung is
    # visible rather than implied.
    by_count: dict[int, tuple] = {}
    for n in range(max_rungs):
        for bounds in combinations(grid, n):
            got = evaluate(list(bounds))
            if got is None:
                continue
            rungs, worst = got
            prev = by_count.get(len(rungs))
            if prev is None or worst < prev[1]:
                by_count[len(rungs)] = (rungs, worst)
    if not by_count:
        return None
    for n in sorted(by_count):
        if by_count[n][1] <= slack:
            return (*by_count[n], by_count, aliased)
    cheapest = min(by_count.values(), key=lambda rw: rw[1])
    return (*cheapest, by_count, aliased)


# --------------------------------------------------------------------------


def human(n: int) -> str:
    if n <= 0:
        return "0"
    if n >= 1 << 62:
        return "inf"
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if n >= 1 << shift:
            v = n / (1 << shift)
            return f"{v:.10g} {unit}"
    return f"{n} B"


def same_work(a: str | None, b: str | None) -> bool:
    """Whether two variant strings describe the same launch.

    A variant reads ``<jit symbol>/g<grid_cap>/x<blocks>``. The symbol is the
    binary and ``x`` is the launch geometry; ``g`` only sizes the IPC inbox and
    is deliberately *not* part of the JIT tag, so two rows agreeing on symbol
    and block count are running byte-identical work into differently-sized
    buffers. Comparing the whole string would call that a difference and let the
    audit attribute a measurement wobble to the dispatch policy.
    """
    if not a or not b:
        return False

    def key(v):
        parts = v.split("/")
        return (parts[0], parts[-1]) if len(parts) >= 2 else (v, "")

    return key(a) == key(b)


def audit_auto(samples, slack: float, verbose: bool) -> int:
    """Grade the shipped dispatcher against the pinned rows. Returns failures.

    ``fly_auto`` routes through ``FlyDSLAllReduce``, i.e. the tables as actually
    shipped -- both levels of heuristic, resolved at launch. The pinned rows are
    the oracle it is held to. This is the acceptance test for the whole
    exercise: everything else in this script *fits* a table, and this is the
    only thing that asks whether the table, once compiled into engines and
    driven through the real dispatch path, still lands where the fit said.

    Graded against the best pinned variant of *any* family, not just the family
    the policy chose, so a wrong family choice is caught as well as a wrong
    rung. Its own SQNR is not checked here -- the benchmark already asserts that
    per row, and the interesting property is latency.
    """
    pinned = [
        c.key
        for f in FAMILIES
        for c in BY_FAMILY[f]
        if c.key not in AUTO_KEYS.get(f, ())
    ]
    tps = sorted({s.tp for s in samples})
    failures = 0
    for tp in tps:
        sub = [s for s in samples if s.tp == tp and "fly_auto" in s.us]
        if not sub:
            logger.warning("## TP%d  fly_auto not measured -- nothing to audit", tp)
            continue
        rows = []
        for s in sub:
            best_key, best_us = best_in(s, pinned)
            if best_key is None:
                continue
            # Did the policy pick the same *work* the winning row ran? The auto
            # variant carries a "<family>:" prefix the pinned rows do not.
            auto_v = s.variant.get("fly_auto", "?").split(":", 1)[-1]
            rows.append(
                (
                    s.nbytes,
                    s.hidden,
                    s.us["fly_auto"] / best_us,
                    best_key,
                    auto_v,
                    same_work(auto_v, s.variant.get(best_key)),
                )
            )
        worst = max(rows, key=lambda r: r[2])
        over = [r for r in rows if r[2] > slack]
        # A row where auto ran the *identical kernel* to the one that "beat" it
        # cannot be a policy error: there is no decision left to get wrong, so
        # the gap is measurement. These collectives are barrier-synchronised and
        # one straggler rank moves the whole reading -- in this sweep the same
        # shapes flip which candidate reads high, and every such row carries a
        # 25-70 us rank spread against ~2 us for its neighbours. Counted and
        # shown separately rather than waved away or scored as a miss.
        bad = [r for r in over if not r[5]]
        noisy = [r for r in over if r[5]]
        failures += len(bad)
        logger.info(
            "## TP%d  fly_auto vs best pinned: worst %.3fx at %s, mean %.3fx, "
            "%d/%d real miss, %d same-kernel noise",
            tp,
            worst[2],
            human(worst[0]),
            sum(r[2] for r in rows) / len(rows),
            len(bad),
            len(rows),
            len(noisy),
        )
        for nbytes, k, ratio, best_key, _v, _ in noisy:
            logger.info(
                "     noise %-10s K=%-5d %.3fx vs %s -- identical kernel",
                human(nbytes),
                k,
                ratio,
                best_key,
            )
        for nbytes, k, ratio, best_key, variant, _ in bad:
            logger.warning(
                "     MISS %-10s K=%-5d %.3fx vs %s   auto ran %s",
                human(nbytes),
                k,
                ratio,
                best_key,
                variant,
            )
        if verbose:
            for nbytes, k, ratio, best_key, variant, same in rows:
                logger.info(
                    "     %-10s K=%-5d %.3fx  best=%-24s same=%d auto=%s",
                    human(nbytes),
                    k,
                    ratio,
                    best_key,
                    int(same),
                    variant,
                )
    return failures


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    ap.add_argument("csv", nargs="+", help="--output-csv files from the sweep")
    ap.add_argument(
        "--link",
        default="pcie",
        choices=["pcie", "xgmi"],
        help="fabric these measurements came from; keys the emitted table.\n"
        "Read it off the report header's `peer links` line -- it is not\n"
        "recoverable from the CSV.",
    )
    ap.add_argument(
        "--metric",
        default="median us",
        choices=["median us", "us"],
        help="'median us' (default) is the median across ranks and is what a\n"
        "threshold should be fitted against; 'us' is the max, which is the\n"
        "right number to *report* but takes the worst of N samples and moves\n"
        "with arrival skew.",
    )
    ap.add_argument(
        "--exact-slack",
        type=float,
        default=1.10,
        help="how much latency the exact-preferring policy may give up to keep\n"
        "the bit-exact one-shot, as a ratio. Default 1.10.",
    )
    ap.add_argument(
        "--max-rungs", type=int, default=3, help="ladder length ceiling per family"
    )
    ap.add_argument(
        "--ladder-slack",
        type=float,
        default=1.10,
        help="worst-case regret a ladder may carry before the fit spends\n"
        "another rung on it. Rung count is the primary objective: a rung is a\n"
        "compiled engine and an IPC inbox, so the fit takes the fewest that\n"
        "clear this and reports what the extra rungs would have bought.",
    )
    ap.add_argument(
        "--holdout",
        nargs="*",
        default=[],
        metavar="CSV",
        help="score the fitted family policy on these files instead of fitting\n"
        "them, and break the result down by hidden size. This is the gate the\n"
        "whole table rests on: the policy is keyed on payload bytes, and if a\n"
        "threshold fitted at one K does not hold at another then bytes is the\n"
        "wrong key and nothing below it is trustworthy. Point it at the\n"
        "K-sensitivity sweep.",
    )
    ap.add_argument(
        "--audit-auto",
        action="store_true",
        help="instead of fitting, grade the `fly_auto` rows -- the shipped\n"
        "dispatcher -- against the best pinned row at every shape, and exit\n"
        "non-zero if any exceeds --ladder-slack. This is the end-to-end\n"
        "acceptance test for the tables; run it on a sweep that includes\n"
        "-c fly_auto alongside the pinned rows.",
    )
    ap.add_argument("--verbose", action="store_true", help="per-shape regret detail")
    args = ap.parse_args()

    samples = load(args.csv, args.metric)
    if args.audit_auto:
        logger.info(
            "# auditing fly_auto over %d shape(s), metric %r, tolerance %.0f%%\n",
            len(samples),
            args.metric,
            (args.ladder_slack - 1) * 100,
        )
        n = audit_auto(samples, args.ladder_slack, args.verbose)
        logger.info(
            "\n%s",
            f"FAILURES: {n}" if n else "PASS: every shape within tolerance",
        )
        raise SystemExit(1 if n else 0)
    holdout = load(args.holdout, args.metric) if args.holdout else []
    tps = sorted({s.tp for s in samples})
    logger.info(
        "# fit from %d file(s), %d shape(s), TP %s, metric %r, link %r\n",
        len(args.csv),
        len(samples),
        tps,
        args.metric,
        args.link,
    )

    fam_rows, ladder_rows = [], []
    for tp in tps:
        sub = [s for s in samples if s.tp == tp]
        logger.info(
            "## TP%d  (%d shapes, %s .. %s)",
            tp,
            len(sub),
            human(min(s.nbytes for s in sub)),
            human(max(s.nbytes for s in sub)),
        )

        for mode, slack in (("fast", None), ("exact", args.exact_slack)):
            one, mesh, worst, mean, detail = fit_families(sub, exact_slack=slack)
            logger.info(
                "  %-5s  oneshot <= %-9s  mesh <= %-9s   worst %.3fx  mean %.3fx",
                mode,
                human(one),
                human(mesh),
                worst,
                mean,
            )
            if args.verbose:
                for nbytes, fam, ratio in detail:
                    if ratio > 1.001:
                        logger.info("      %-10s %-8s %.3fx", human(nbytes), fam, ratio)
            fam_rows.append((args.link, tp, mode, one, mesh, worst))

            # K-sensitivity: the same thresholds, scored on shapes the fit never
            # saw, split by hidden size.
            hold = [h for h in holdout if h.tp == tp]
            if hold and mode == "fast":
                by_k: dict[int, list] = {}
                for nbytes, k, _fam, ratio in score_policy(hold, one, mesh):
                    by_k.setdefault(k, []).append((nbytes, ratio))
                for k in sorted(by_k):
                    rows = by_k[k]
                    bad = max(rows, key=lambda nr: nr[1])
                    logger.info(
                        "    holdout K=%-5d n=%-2d worst %.3fx at %s",
                        k,
                        len(rows),
                        bad[1],
                        human(bad[0]),
                    )

        # Each family's ladder is fitted only over the sizes the family policy
        # actually sends it. The exact-mode one-shot window is the wider of the
        # two, so use it -- a rung the fast mode never reaches is harmless, a
        # missing rung the exact mode does reach is not.
        one_exact = next(r[3] for r in fam_rows if r[:3] == (args.link, tp, "exact"))
        one_fast, mesh_max = next(
            (r[3], r[4]) for r in fam_rows if r[:3] == (args.link, tp, "fast")
        )
        windows = {
            "oneshot": (0, max(one_exact, one_fast)),
            "mesh": (min(one_exact, one_fast), mesh_max),
            "ring": (mesh_max, 1 << 62),
        }
        for family in FAMILIES:
            lo, hi = windows[family]
            got = fit_ladder(
                sub, family, args.max_rungs, args.ladder_slack, windows[family]
            )
            if got is None:
                logger.info("  ladder %-8s -- not dispatched here", family)
                continue
            rungs, worst, by_count, aliased = got
            logger.info(
                "  ladder %-8s (%s .. %s]  %s   worst %.3fx   (by rung count: %s)",
                family,
                human(lo),
                human(hi),
                " ".join(f"[{human(b)}: {k}]" for b, k in rungs),
                worst,
                ", ".join(f"{n}:{w:.3f}x" for n, (_, w) in sorted(by_count.items())),
            )
            for keep, dropped in sorted(aliased.items()):
                logger.info(
                    "      alias: %s compiles the same kernel as %s here",
                    ", ".join(dropped),
                    keep,
                )
            # A variant that is replaced and then comes back is the fit reading
            # noise. The knobs a ladder selects on -- super-tile, atom count,
            # block cap -- all trade the same thing monotonically against
            # payload size, so a real ladder does not revisit a rung. Say so
            # loudly and print what one rung would have cost, because the honest
            # answer to a non-monotone fit is usually a shorter ladder.
            keys_in_order = [k for _, k in rungs]
            if len(set(keys_in_order)) != len(keys_in_order):
                one_worst = by_count.get(1, (None, float("nan")))[1]
                logger.warning(
                    "      NON-MONOTONE: %s revisits a variant -- likely noise. "
                    "One rung (%s) costs %.3fx; prefer it unless the shape is "
                    "reproducible across runs.",
                    family,
                    by_count.get(1, ([(0, "?")], 0))[0][0][1],
                    one_worst,
                )
            ladder_rows.append((tp, family, rungs, worst))

        logger.info("")

    logger.info("## paste-ready\n")
    logger.info("_FAMILY_POLICY = {")
    for link, tp, mode, one, mesh, worst in fam_rows:
        if mode != "fast":
            continue
        ex = next(r for r in fam_rows if r[:3] == (link, tp, "exact"))
        logger.info(
            "    (%r, %d): FamilyPolicy(oneshot_max=%d, oneshot_max_exact=%d, "
            "mesh_max=%d),  # %s / %s / %s, worst %.3fx",
            link,
            tp,
            one,
            ex[3],
            mesh,
            human(one),
            human(ex[3]),
            human(mesh),
            max(worst, ex[5]),
        )
    logger.info("}\n")
    for tp, family, rungs, worst in ladder_rows:
        logger.info(
            "# TP%d %s ladder, worst %.3fx: %s",
            tp,
            family,
            worst,
            ", ".join(f"({lo}, {k!r})" for lo, k in rungs),
        )


if __name__ == "__main__":
    main()
