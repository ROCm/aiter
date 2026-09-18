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

* ``fast`` -- minimise latency, whatever the numerics cost. Chooses among all
  three families (one-shot, mesh, ring).
* ``exact`` -- ``AITER_FLY_AR_ACCURACY=exact`` (the shipped default) never
  builds a quantized engines at all: above its one-shot ceiling,
  ``should_fly_all_reduce`` just declines and the caller falls through to
  whatever it would otherwise dispatch to.

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

``--fusion ar_rmsnorm`` fits the fused tables from a
``bench_comm_allreduce.py --fusion ar_rmsnorm`` sweep instead. Two differences:
a **third** threshold, ``min_bytes``, below which the dispatcher declines and
the caller runs what it would have anyway (priced from each row's ``prod
path``); and an **atoms-keyed** ladder, because a fused ``block`` means a
different ``atoms`` at each width and only ``atoms`` is portable across them.
See :class:`FusedCand`.
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
from bench_comm_allreduce import CANDIDATES, _prod_candidate_key

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
CAND_BY_KEY = {c.key: c for c in CANDIDATES}
AUTO_KEYS = {
    "oneshot": ("fly_1stage",),
    "mesh": ("fly_int4",),
    "ring": ("fly_int4_ring",),
}

# --------------------------------------------------------------------------
# --fusion ar_rmsnorm: a different candidate set, a fourth family, atoms-keyed
# --------------------------------------------------------------------------

FUSION_MODES = ("none", "ar_rmsnorm")
#: Set once by :func:`set_fusion_mode`. Everything above is rebound for the
#: fused sweep rather than parameterised, because every consumer wants the same
#: answer and this is a single-shot CLI.
FUSION = "none"

#: Falling back to what aiter would run here anyway. A fourth "family" so the
#: fit can decide *not* to fuse, which the plain path never has to.
DECLINE = "decline"


def fused_family_of(cand) -> str | None:
    if cand.family == "fused_fly1s":
        return "oneshot"
    if cand.family == "fused_flyqr":
        return cand.algorithm  # "mesh" | "ring"
    return None


@dataclass(frozen=True)
class FusedCand:
    """A fused bench row relabelled by ``atoms``, which is what a rung carries.

    Bench rows pin ``block``, but a block means a different ``atoms`` at each
    width -- b256 is atoms=4 at hidden 8192 and atoms=2 at 4096 -- and
    ``segment_pick`` only selects a key present at *every* sample in a segment.
    Across the fitted widths {4096, 8192, 7168} the common block set is empty,
    so a block-keyed ladder fit returns nothing at all. Atoms is the portable
    name and the one ``FUSED_ONESHOT_LADDER`` stores.
    """

    key: str
    family: str
    atoms: int
    grid_cap: int | None = None
    fanout: str | None = None
    skip_self: bool | None = None

    def fused_rung(self, min_bytes: int) -> tuple:
        """As a ``FUSED_ONESHOT_LADDER`` rung."""
        unpinned = [
            n
            for n in ("atoms", "grid_cap", "fanout", "skip_self")
            if getattr(self, n) is None
        ]
        if unpinned:
            raise ValueError(
                f"{self.key} leaves {', '.join(unpinned)} to the engine default, "
                "so it has no FUSED_ONESHOT_LADDER rung"
            )
        return (int(min_bytes), self.atoms, self.grid_cap, self.fanout, self.skip_self)


def fused_atoms_at(cand, hidden: int, tp: int) -> int | None:
    """*cand*'s atoms-per-row at this width, or None if it has no build here."""
    from aiter.ops.flydsl.kernels.one_shot_allreduce import fused_block_options
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import (
        quick_reduce_row_block_options,
    )

    opts = (
        fused_block_options(hidden)
        if cand.family == "fused_fly1s"
        else quick_reduce_row_block_options(hidden, tp)
    )
    for block, atoms in opts:
        if block == cand.block:
            return atoms
    return None


def fused_atoms_key(cand, atoms: int) -> str:
    """The atoms-keyed name *cand* is relabelled to."""
    if cand.family == "fused_fly1s":
        key = f"fused_fly_1stage_a{atoms}_g{cand.grid_cap}"
        return key + "_ss" if cand.skip_self else key
    return f"fused_fly_{cand.algorithm}_a{atoms}"


def fused_pseudo_candidates():
    """``key -> FusedCand`` for every atoms relabelling the sweep can produce.

    Enumerated over all widths a block-pinned row could be measured at, because
    the same row becomes a different pseudo-candidate at each.
    """
    from aiter.ops.flydsl.kernels.one_shot_allreduce import SUPPORTED_ATOMS

    out: dict[str, FusedCand] = {}
    for cand in CANDIDATES:
        fam = fused_family_of(cand)
        if fam is None or cand.block is None:
            continue
        for atoms in SUPPORTED_ATOMS:
            key = fused_atoms_key(cand, atoms)
            out.setdefault(
                key,
                FusedCand(
                    key=key,
                    family=fam,
                    atoms=atoms,
                    grid_cap=cand.grid_cap,
                    fanout=cand.fanout or "peer",
                    skip_self=bool(cand.skip_self),
                ),
            )
    return out


def set_fusion_mode(mode: str) -> None:
    """Rebind the family tables for *mode*. Call once, before anything else."""
    global FUSION, FAMILIES, BY_FAMILY, CAND_BY_KEY, AUTO_KEYS
    if mode not in FUSION_MODES:
        raise ValueError(f"fusion must be one of {FUSION_MODES}, got {mode!r}")
    FUSION = mode
    if mode == "none":
        return
    pseudo = fused_pseudo_candidates()
    # The unpinned rows are the *auto* rows: what the shipped ladder does today,
    # and so the thing the fit is graded against rather than an input to it.
    autos = {
        "oneshot": ("fused_fly_1stage",),
        "mesh": ("fused_fly_mesh",),
        "ring": ("fused_fly_ring",),
    }
    auto_cands = {
        k: FusedCand(key=k, family=f, atoms=0)
        for f, keys in autos.items()
        for k in keys
    }
    FAMILIES = ("oneshot", "mesh", "ring")
    BY_FAMILY = {
        f: [c for c in pseudo.values() if c.family == f]
        + [c for c in auto_cands.values() if c.family == f]
        for f in FAMILIES
    }
    CAND_BY_KEY = {**pseudo, **auto_cands}
    AUTO_KEYS = autos


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
    db: dict  # candidate key -> SQNR dB, for --min-sqnr
    # --fusion only: what aiter's own dispatch runs here, i.e. the cost of
    # declining to fuse. inf when the row named a path the sweep did not measure.
    decline_us: float = float("inf")


def check_timing_provenance(df, *, require: str | None) -> str:
    """The measurement regime every input row agrees on, or exit.

    A threshold sits where two curves cross, so it is only as good as the metric
    both were measured with. Eager and graph-replay numbers differ most at
    exactly the decode sizes these thresholds live at, and the difference is
    *per candidate family* -- so a fit over a mixture does not merely average
    two regimes, it moves the crossover. Refuse rather than warn.

    A CSV with no ``timing`` column predates the provenance and cannot be
    vouched for. It is reported as ``unknown`` and warned about rather than
    refused -- such files are still worth reading against, and the CSVs the
    current tables were fitted from are all of this kind -- but
    ``--require-timing`` will reject them.
    """
    if "timing" not in df.columns:
        logger.warning(
            "# inputs carry no `timing` column, so the measurement regime "
            "cannot be verified."
        )
        return "unknown"
    seen = sorted({str(v) for v in df["timing"].dropna().unique()})
    if len(seen) > 1:
        raise SystemExit(
            f"inputs mix timing regimes {seen}. A dispatch threshold is a "
            "crossover between two candidates, and the eager/graph gap is both "
            "large at decode sizes and different per candidate family, so a "
            "mixed fit moves the boundary. Re-run the sweep with one --timing."
        )
    got = seen[0] if seen else "unknown"
    if require is not None and got != require:
        raise SystemExit(
            f"inputs were measured with --timing {got}, but --require-timing "
            f"{require} was asked for."
        )
    return got


def load(paths, metric: str, *, require_timing: str | None = None):
    """``(samples, provenance)`` for *paths*.

    The provenance -- which timing regime and which FlyDSL accuracy mode the
    sweep ran under -- decides how the result may be used, so it travels with
    the samples rather than being re-derived by each consumer.
    """
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    timing = check_timing_provenance(df, require=require_timing)
    accuracy = "unknown"
    if "fly accuracy" in df.columns:
        modes = sorted({str(v) for v in df["fly accuracy"].dropna().unique()})
        if len(modes) > 1:
            raise SystemExit(
                f"inputs mix FlyDSL accuracy regimes {modes}: the candidate set "
                "itself differs between them, so they cannot be pooled."
            )
        accuracy = modes[0] if modes else "unknown"
    prov = {"timing": timing, "accuracy": accuracy}
    logger.info("# inputs: timing=%s, fly accuracy=%s", timing, accuracy)
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
    decline: dict[tuple[int, int, int], float] = {}
    sqnr: dict[tuple[int, int, int], dict] = {}
    for _, r in df.iterrows():
        ident = (int(r["TP"]), int(r["_nbytes"]), int(r.get("K", 0)))
        cell = best.setdefault(ident, {})
        vcell = variants.setdefault(ident, {})
        dcell = sqnr.setdefault(ident, {})
        for k in keys:
            v = r.get(f"{k} {metric}")
            if v is not None and pd.notna(v):
                cell[k] = min(v, cell[k]) if k in cell else float(v)
            sym = r.get(f"{k} variant")
            if sym is not None and pd.notna(sym):
                vcell[k] = str(sym)
            q = r.get(f"{k} SQNR dB")
            if q is not None and pd.notna(q):
                dcell[k] = min(float(q), dcell.get(k, float("inf")))
        path = r.get("prod path")
        if FUSION != "none" and path is not None and pd.notna(path):
            pk = _prod_candidate_key(str(path))
            pu = r.get(f"{pk} {metric}") if pk else None
            if pu is not None and pd.notna(pu):
                prev = decline.get(ident, float("inf"))
                decline[ident] = min(float(pu), prev)
    samples = [
        Sample(
            nbytes=n,
            tp=tp,
            hidden=k,
            us=cell,
            variant=variants[(tp, n, k)],
            db=sqnr.get((tp, n, k), {}),
            decline_us=decline.get((tp, n, k), float("inf")),
        )
        for (tp, n, k), cell in sorted(best.items(), key=lambda kv: kv[0])
    ]
    if FUSION != "none":
        samples = [relabel_fused_by_atoms(s) for s in samples]
    return samples, prov


def relabel_fused_by_atoms(s: Sample) -> Sample:
    """Re-key this sample's block-pinned fused rows by atoms. See :class:`FusedCand`."""
    us, variant, db = dict(s.us), dict(s.variant), dict(s.db)
    for cand in CANDIDATES:
        if fused_family_of(cand) is None or cand.block is None:
            continue
        if cand.key not in s.us:
            continue
        atoms = fused_atoms_at(cand, s.hidden, s.tp)
        if atoms is None:
            continue  # measured but no build at this width: nothing to relabel
        key = fused_atoms_key(cand, atoms)
        us[key] = s.us[cand.key]
        if cand.key in s.variant:
            variant[key] = s.variant[cand.key]
        if cand.key in s.db:
            db[key] = s.db[cand.key]
    return Sample(
        nbytes=s.nbytes,
        tp=s.tp,
        hidden=s.hidden,
        us=us,
        variant=variant,
        db=db,
        decline_us=s.decline_us,
    )


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
    # Declaration order of whichever candidate table is active -- the bench rows
    # in plain mode, the atoms-keyed pseudo-candidates under --fusion.
    order = {k: i for i, k in enumerate(CAND_BY_KEY)}
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


def score_policy(samples, one_max: int, mesh_max: int, min_bytes: int = 0):
    """Per-shape regret of a fixed family policy. ``(nbytes, K, family, ratio)``.

    The holdout check: a policy fitted on one hidden size is scored on others.
    Sizes where the named family was not measured are reported as ``inf`` rather
    than skipped -- on a holdout that is a hole in the evidence, and silently
    dropping it would let a policy look good precisely where it is unsupported.
    """
    out = []
    for s in samples:
        oracle = min(family_best(s, f)[1] for f in all_families())
        if not math.isfinite(oracle):
            continue
        fam = pick_family(s.nbytes, one_max, mesh_max, min_bytes)
        us = family_best(s, fam)[1]
        out.append((s.nbytes, s.hidden, fam, us / oracle))
    return out


#: Minimum SQNR a candidate must clear to be selectable, or None for none.
#: Set by --min-sqnr. Applied in :func:`best_in`, so it constrains the
#: oracle as well as every family's pick -- a floor that only filtered the
#: answer would still be graded against an oracle allowed to cheat.
MIN_SQNR: float | None = None


def best_in(sample: Sample, keys) -> tuple[str | None, float]:
    """Fastest of *keys* that ran at this shape and clears MIN_SQNR."""
    live = [(k, sample.us[k]) for k in keys if k in sample.us]
    if MIN_SQNR is not None:
        live = [kv for kv in live if sample.db.get(kv[0], float("inf")) >= MIN_SQNR]
    return min(live, key=lambda kv: kv[1]) if live else (None, float("inf"))


def family_best(sample: Sample, family: str) -> tuple[str | None, float]:
    if family == DECLINE:
        return (DECLINE, sample.decline_us)
    return best_in(sample, [c.key for c in BY_FAMILY[family]])


def all_families() -> tuple[str, ...]:
    """Families the oracle ranks over. Fused mode can also choose not to fuse."""
    return (*FAMILIES, DECLINE) if FUSION != "none" else FAMILIES


# --------------------------------------------------------------------------
# Level 2: which family
# --------------------------------------------------------------------------


def pick_family(
    nbytes: int, oneshot_max: int, mesh_max: int, min_bytes: int = 0
) -> str:
    if nbytes < min_bytes:
        return DECLINE
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
    fams = all_families()
    oracle = {s.nbytes: min(family_best(s, f)[1] for f in fams) for s in samples}
    # A shape where nothing at all ran cannot grade a policy.
    graded = [s for s in samples if math.isfinite(oracle[s.nbytes])]
    # In fused mode the floor is a third threshold: below it the dispatcher
    # declines and the caller runs what it would have anyway. 0 stays in the
    # grid, so "never decline" remains expressible.
    floors = grid if FUSION != "none" else (0,)

    def regrets(min_bytes, one_max, mesh_max):
        out = []
        for s in graded:
            fam = pick_family(s.nbytes, one_max, mesh_max, min_bytes)
            us = family_best(s, fam)[1]
            if not math.isfinite(us):
                # The policy named a family that was not measured here. Only
                # legitimate for the one-shot above the sweep's own ceiling;
                # anywhere else it is a hole and must not be scored as free.
                return None
            out.append((s.nbytes, fam, us / oracle[s.nbytes]))
        return out

    scored = []
    for floor in floors:
        for one_max in grid:
            if one_max < floor:
                continue
            for mesh_max in grid:
                if mesh_max < one_max:
                    continue
                r = regrets(floor, one_max, mesh_max)
                if r is None:
                    continue
                worst = max(x[2] for x in r)
                mean = sum(x[2] for x in r) / len(r)
                scored.append((worst, mean, one_max, mesh_max, r, floor))
    if not scored:
        raise SystemExit(
            "no feasible (oneshot_max, mesh_max) pair; is a family absent?"
        )

    if exact_slack is None:
        # Fewest declines among equally good policies: a floor that buys nothing
        # is a shape the dispatcher gives away for free.
        worst, mean, one_max, mesh_max, r, floor = min(
            scored, key=lambda t: (t[0], t[1], t[5])
        )
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
        worst, mean, one_max, mesh_max, r, floor = max(
            ok, key=lambda t: (t[2], -t[0], -t[1])
        )
    return one_max, mesh_max, worst, mean, r, floor


# --------------------------------------------------------------------------
# Level 2b: the real "exact" mode ceiling -- one-shot vs. what it declines to
# --------------------------------------------------------------------------

# What CudaCommunicator.all_reduce actually falls through to once FlyDSL
# declines a payload in exact mode: `use_new` is hardcoded True with no env
# lever, so `cdr_naive` never runs in production, and rccl is reachable only
# past AITER_CUSTOM_AR_MAX_SIZE (64 MiB) -- far above anything measured here.
# This is the oracle the shipped `oneshot_max_exact` is fitted against.
EXACT_FALLBACK_PRIMARY = ("cdr",)
# cdr's own 1stage/2stage internal dispatch has occasional cliffs (see the
# KB); cdr_naive and rccl are measured and reported alongside as a robustness
# check on the same boundary, not folded into the shipped table.
EXACT_FALLBACK_ROBUST = ("cdr", "cdr_naive", "rccl")

CDR_1STAGE_MAX_BYTES = {4: 160 << 10, 8: 80 << 10}


def cdr_edge_thresholds(tp: int) -> tuple[int, ...]:
    """Ceiling candidates that mirror ``cdr``'s 1stage/2stage edge for *tp*.

    ``edge - 1``: a ceiling is inclusive (``nbytes <= oneshot_max``) while
    ``cdr``'s branch is exclusive (``bytes < edge``), so the payload landing
    exactly on the edge is the first one ``cdr`` handles with its 2stage kernel
    and must fall through rather than be kept.
    """
    edge = CDR_1STAGE_MAX_BYTES.get(int(tp))
    return () if edge is None else (edge - 1,)


def fit_oneshot_exact_ceiling(samples, oneshot_keys, oracle_keys, extra=()):
    """Largest one-shot ceiling by worst-case regret against *oracle_keys*.

    This is exactly ``fit_families``'s plain (``exact_slack=None``) search,
    narrowed to a single threshold and to a caller-supplied oracle instead of
    the mesh/ring families: below the ceiling the policy runs the best pinned
    one-shot variant, above it the best of *oracle_keys*, and the fit picks
    the round threshold minimising worst-case regret against the per-shape
    best of the two. No accuracy slack applies -- every key on both sides is
    exact, so the only question is speed.

    The returned ceiling never exceeds the largest size actually measured.
    ``round_thresholds`` offers infinity as a candidate -- correctly, since "the
    one-shot wins everywhere" has to be expressible -- but shipping it would put
    the boundary somewhere no measurement supports, and it would flow into
    ``windows["oneshot"]`` and re-open the un-windowed ladder fit that
    ``test_ladder_rungs_fall_inside_their_dispatch_window`` exists to prevent.
    That answer is reported as a saturation warning instead: extend the sweep.
    """
    sizes = [s.nbytes for s in samples]
    # *extra* admits non-round candidates that mirror a discontinuity in the
    # fallback itself -- see CDR_1STAGE_MAX_BYTES for why those are exempt from
    # the round-threshold rule.
    grid = sorted(set(round_thresholds(min(sizes), max(sizes))) | set(extra))
    oracle = {s.nbytes: best_in(s, [*oneshot_keys, *oracle_keys])[1] for s in samples}
    graded = [s for s in samples if math.isfinite(oracle[s.nbytes])]

    def regrets(ceiling):
        out = []
        for s in graded:
            keys = oneshot_keys if s.nbytes <= ceiling else oracle_keys
            us = best_in(s, keys)[1]
            if not math.isfinite(us):
                return None
            out.append((s.nbytes, us / oracle[s.nbytes]))
        return out

    scored = []
    for ceiling in grid:
        r = regrets(ceiling)
        if r is None:
            continue
        worst = max(x[1] for x in r)
        scored.append((worst, sum(x[1] for x in r) / len(r), ceiling, r))
    if not scored:
        raise SystemExit(
            "no feasible one-shot exact ceiling; is a fallback candidate absent?"
        )
    # Ties on (worst, mean) cost nothing either way -- prefer the widest such
    # ceiling, so a wash in speed is resolved in favour of running the
    # bit-exact one-shot rather than declining to the fallback.
    worst, mean, ceiling, r = min(scored, key=lambda t: (t[0], t[1], -t[2]))
    # Bounded by where the *one-shot* data stops, not where any data stops. The
    # one-shot rows are gated above by AITER_BENCH_FLY1S_MAX_KB, so a sweep can
    # easily carry cdr out to 114 MiB and the one-shot only to 8 -- and then
    # every ceiling past 8 MiB is infeasible for want of a numerator rather than
    # because the one-shot lost. Landing on that edge is the same "ran out of
    # evidence" answer as landing on infinity and is reported the same way.
    one_max = max(
        (s.nbytes for s in graded if math.isfinite(best_in(s, oneshot_keys)[1])),
        default=0,
    )
    saturated = ceiling >= one_max
    if saturated:
        logger.warning(
            "  exact ceiling saturated at %s: the one-shot never loses inside "
            "the range it was measured over, so this is the edge of the sweep "
            "and not a fitted boundary. Raise AITER_BENCH_FLY1S_MAX_KB (or "
            "extend the shape ladder) to find the real one.",
            human(one_max),
        )
    ceiling = min(ceiling, one_max)
    return ceiling, worst, mean, r, saturated


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


def reconcile_family_row(link, tp, one_fast, one_exact, mesh_max) -> int:
    """``oneshot_max_exact`` adjusted to satisfy ``FamilyPolicy``'s invariants.

    The two ceilings come from independent searches -- ``oneshot_max`` against
    the quantized families, ``oneshot_max_exact`` against ``cdr`` -- so nothing
    in the fit makes them order correctly. ``FamilyPolicy.__post_init__`` raises
    on both ``oneshot_max_exact < oneshot_max`` and ``mesh_max <
    oneshot_max_exact``, which would make the paste-ready block below fail at
    import rather than at review.

    Both violations are meaningful, not bookkeeping, so each is reported:

    * ``exact < fast``: the one-shot beats the *mesh* further up than it beats
      ``cdr``. Widening exact to match is the conservative repair -- the mode
      exists to avoid quantizing, and exact mode already ran the one-shot at
      that size in the fast policy.
    * ``exact > mesh_max``: the fast fit has the one-shot losing to the mesh
      somewhere the exact fit still wants it. Clamping to ``mesh_max`` keeps the
      families partitioning by size; it does not change what exact mode runs,
      since exact mode collapses everything above its ceiling anyway.
    """
    out = one_exact
    if out < one_fast:
        logger.warning(
            "  (%r, %d): oneshot_max_exact (%s) fitted below oneshot_max (%s). "
            "The one-shot outruns the mesh further than it outruns cdr; "
            "widening exact to the fast ceiling.",
            link,
            tp,
            human(out),
            human(one_fast),
        )
        out = one_fast
    if out > mesh_max:
        logger.warning(
            "  (%r, %d): oneshot_max_exact (%s) fitted above mesh_max (%s); "
            "clamping so the families still partition by size.",
            link,
            tp,
            human(out),
            human(mesh_max),
        )
        out = mesh_max
    return out


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


def audit_declines(samples, slack: float, oracle_keys, oneshot_keys, verbose) -> int:
    """Grade the payloads exact mode *refuses*. Returns failures.

    Above ``oneshot_max_exact`` the dispatcher declines and the caller falls
    through to ``cdr``, so ``fly_auto`` reads n/a and drops out of the latency
    audit entirely. That silence is the whole risk of the exact policy: a
    ceiling set too low gives away the one-shot's win at every size above it and
    nothing in the report says so.

    So the decline is audited as its own decision. At each declined size, compare
    the best pinned one-shot against the fallback the decline actually reaches.
    Declining is correct when the fallback is at least as fast, or within
    *slack*; a one-shot that beats it by more than that is a ceiling that should
    have been wider, and is reported with the size so the table can move.
    """
    failures = 0
    for tp in sorted({s.tp for s in samples}):
        declined = [
            s for s in samples if s.tp == tp and "fly_auto" not in s.us and s.us
        ]
        if not declined:
            continue
        rows = []
        for s in declined:
            one_key, one_us = best_in(s, oneshot_keys)
            _fb_key, fb_us = best_in(s, oracle_keys)
            if not (math.isfinite(one_us) and math.isfinite(fb_us)):
                continue
            rows.append((s.nbytes, fb_us / one_us, one_key, one_us, fb_us))
        # A fallback reading that some *larger* payload beats by a wide margin
        # is a straggler, not a cliff -- these collectives are barrier
        # synchronised and one slow rank moves the whole number. Without this,
        # a single transient reads as "CEILING TOO LOW" by 10x and buries the
        # real findings. `audit_auto` filters the same class of noise via
        # same_work; a decline has no variant to compare, so monotonicity in
        # payload size is the available invariant.
        noisy = set()
        for i, (nbytes, _ratio, _k, _one, fb_us) in enumerate(rows):
            cheaper = [r[4] for r in rows if r[0] > nbytes]
            if cheaper and fb_us > 3 * min(cheaper):
                noisy.add(i)
        if noisy:
            for i in sorted(noisy):
                logger.info(
                    "     noise %-10s fallback %.1f us, beaten by a larger "
                    "payload's %.1f us -- straggler, not a cliff",
                    human(rows[i][0]),
                    rows[i][4],
                    min(r[4] for r in rows if r[0] > rows[i][0]),
                )
            rows = [r for i, r in enumerate(rows) if i not in noisy]
        if not rows:
            continue
        if not rows:
            logger.info(
                "## TP%d  %d declined shape(s), none gradeable "
                "(no pinned one-shot measured above the ceiling)",
                tp,
                len(declined),
            )
            continue
        bad = [r for r in rows if r[1] > slack]
        failures += len(bad)
        worst = max(rows, key=lambda r: r[1])
        logger.info(
            "## TP%d  declines: %d shape(s) above the ceiling, worst forgone "
            "speedup %.3fx at %s, %d/%d beyond tolerance",
            tp,
            len(rows),
            worst[1],
            human(worst[0]),
            len(bad),
            len(rows),
        )
        for nbytes, ratio, one_key, one_us, fb_us in bad:
            logger.warning(
                "     CEILING TOO LOW %-10s one-shot %s %.1f us vs fallback "
                "%.1f us -- declining gives up %.3fx",
                human(nbytes),
                one_key,
                one_us,
                fb_us,
                ratio,
            )
        if verbose:
            for nbytes, ratio, one_key, one_us, fb_us in rows:
                logger.info(
                    "     %-10s %.3fx  one-shot %s %.1f us, fallback %.1f us",
                    human(nbytes),
                    ratio,
                    one_key,
                    one_us,
                    fb_us,
                )
    return failures


def audit_auto(
    samples,
    slack: float,
    verbose: bool,
    *,
    accuracy: str = "fast",
    auto_key: str = "fly_auto",
) -> int:
    """Grade the shipped dispatcher against the pinned rows. Returns failures.

    ``fly_auto`` routes through ``FlyDSLAllReduce``, i.e. the tables as actually
    shipped -- both levels of heuristic, resolved at launch. The pinned rows are
    the oracle it is held to. This is the acceptance test for the whole
    exercise: everything else in this script *fits* a table, and this is the
    only thing that asks whether the table, once compiled into engines and
    driven through the real dispatch path, still lands where the fit said.

    *accuracy* is the regime the sweep ran under, and it changes the oracle:

    * ``fast`` -- the full three-family policy. The pinned mesh/ring rows are in
      the sweep and in the oracle, so a wrong *family* choice is caught as well
      as a wrong rung.
    * ``exact`` -- only the one-shot is reachable, so the pinned mesh/ring rows
      were never measured. The alternative to dispatching is not another FlyDSL
      family, it is ``cdr``; grading against the one-shot family alone would
      check the ladder and call the policy green without ever asking whether
      running FlyDSL at all was right. The exact fallback keys join the oracle,
      and ``audit_declines`` covers the sizes this row is absent from.

    Its own SQNR is not checked here -- the benchmark already asserts that per
    row, and the interesting property is latency.
    """
    pinned = [
        c.key
        for f in FAMILIES
        for c in BY_FAMILY[f]
        if c.key not in AUTO_KEYS.get(f, ())
    ]
    if accuracy == "exact":
        pinned = [
            c.key
            for c in BY_FAMILY["oneshot"]
            if c.key not in AUTO_KEYS.get("oneshot", ())
        ] + list(EXACT_FALLBACK_PRIMARY)
    tps = sorted({s.tp for s in samples})
    failures = 0
    for tp in tps:
        sub = [s for s in samples if s.tp == tp and auto_key in s.us]
        if not sub:
            logger.warning("## TP%d  %s not measured -- nothing to audit", tp, auto_key)
            continue
        rows = []
        for s in sub:
            best_key, best_us = best_in(s, pinned)
            # Declining is an option the fused dispatcher has and the plain
            # one does not, so fusing where aiter's own kernel is faster is
            # a miss like any other.
            if FUSION != "none" and s.decline_us < best_us:
                best_key, best_us = DECLINE, s.decline_us
            if best_key is None:
                continue
            # Did the policy pick the same *work* the winning row ran? The auto
            # variant carries a "<family>:" prefix the pinned rows do not.
            auto_v = s.variant.get(auto_key, "?").split(":", 1)[-1]
            rows.append(
                (
                    s.nbytes,
                    s.hidden,
                    s.us[auto_key] / best_us,
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
            "## TP%d  %s vs best pinned: worst %.3fx at %s, mean %.3fx, "
            "%d/%d real miss, %d same-kernel noise",
            tp,
            auto_key,
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


def fit_fused(samples, holdout, args) -> None:
    """``--fusion ar_rmsnorm``: the decline floor, the family boundaries, the ladder.

    A separate driver rather than a branch through the plain one: there is no
    ``exact`` mode here (the fused engines are bf16-only and the quantized ones
    are the accuracy question, not a second policy), no ``cdr`` 1stage/2stage
    edge to reconcile against, and one extra threshold.
    """
    tps = sorted({s.tp for s in samples})
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
        blind = sum(1 for s in sub if not math.isfinite(s.decline_us))
        if blind:
            logger.warning(
                "  %d/%d shape(s) have no resolved `prod path` time, so declining "
                "cannot be priced there and min_bytes is fitted without them.",
                blind,
                len(sub),
            )

        one, mesh, worst, mean, detail, floor = fit_families(sub, exact_slack=None)
        logger.info(
            "  fused   decline < %-9s oneshot <= %-9s  mesh <= %-9s  "
            "worst %.3fx  mean %.3fx",
            human(floor),
            human(one),
            human(mesh),
            worst,
            mean,
        )
        if args.verbose:
            for nbytes, fam, ratio in detail:
                if ratio > 1.001:
                    logger.info("      %-10s %-8s %.3fx", human(nbytes), fam, ratio)
        fam_rows.append((args.link, tp, floor, one, mesh, worst))

        hold = [h for h in holdout if h.tp == tp]
        if hold:
            by_k: dict[int, list] = {}
            for nbytes, k, _fam, ratio in score_policy(hold, one, mesh, floor):
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

        windows = {
            "oneshot": (floor, one),
            "mesh": (one, mesh),
            "ring": (mesh, 1 << 62),
        }
        for family in FAMILIES:
            lo, hi = windows[family]
            if hi <= lo:
                logger.info("  ladder %-8s -- empty window", family)
                continue
            got = fit_ladder(sub, family, args.max_rungs, args.ladder_slack, (lo, hi))
            if got is None:
                logger.info("  ladder %-8s -- not dispatched here", family)
                continue
            rungs, lworst, by_count, aliased = got
            logger.info(
                "  ladder %-8s (%s .. %s]  %s   worst %.3fx   (by rung count: %s)",
                family,
                human(lo),
                human(hi),
                " ".join(f"[{human(b)}: {k}]" for b, k in rungs),
                lworst,
                ", ".join(f"{n}:{w:.3f}x" for n, (_, w) in sorted(by_count.items())),
            )
            ladder_rows.append((tp, family, rungs, lworst))
        logger.info("")

    logger.info("## paste-ready\n")
    logger.info("FUSED_FAMILY_POLICY = {")
    for link, tp, floor, one, mesh, fworst in fam_rows:
        logger.info(
            "    (%r, %d): FamilyPolicy(oneshot_max=%d, oneshot_max_exact=%d, "
            "mesh_max=%d, min_bytes=%d),  # %s / %s / %s, worst %.3fx",
            link,
            tp,
            one,
            one,
            mesh,
            floor,
            human(one),
            human(mesh),
            human(floor),
            fworst,
        )
    logger.info("}\n")

    oneshot = [r for r in ladder_rows if r[1] == "oneshot"]
    if oneshot:
        logger.info("FUSED_ONESHOT_LADDER = {")
        for tp, _family, rungs, lworst in oneshot:
            try:
                tuples = [CAND_BY_KEY[k].fused_rung(lo) for lo, k in rungs]
            except (KeyError, ValueError) as exc:
                logger.warning("    # %d: %s", tp, exc)
                continue
            logger.info(
                "    (%r, %d): (%s),  # worst %.3fx",
                args.link,
                tp,
                "".join(f"{t!r}, " for t in tuples).rstrip(),
                lworst,
            )
        logger.info("}")
    # The quantized schedules already ladder super_tile/grid_cap internally, so
    # the only thing the fused dispatcher has to be told is the row geometry --
    # one atoms_per_row per (link, world, algorithm), not a payload ladder.
    # Where the fit wanted more than one rung, say so rather than silently
    # shipping the first.
    qr = [r for r in ladder_rows if r[1] in ("mesh", "ring")]
    if qr:
        logger.info("\nFUSED_QR_ROW_ATOMS = {")
        for tp, family, rungs, lworst in qr:
            atoms = [CAND_BY_KEY[k].atoms for _lo, k in rungs if k in CAND_BY_KEY]
            if not atoms:
                continue
            note = "" if len(set(atoms)) == 1 else f"  # NOTE: fit wanted {atoms}"
            logger.info(
                "    (%r, %d, %r): %d,  # worst %.3fx%s",
                args.link,
                tp,
                family,
                atoms[0],
                lworst,
                note,
            )
        logger.info("}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    ap.add_argument("csv", nargs="+", help="--output-csv files from the sweep")
    ap.add_argument(
        "--fusion",
        choices=FUSION_MODES,
        default="none",
        help="which sweep this is. 'ar_rmsnorm' fits the fused tables instead:\n"
        "a decline floor as well as the two family boundaries, and an\n"
        "atoms-keyed ladder (a fused block means a different atoms at each\n"
        "hidden, so block is not a portable rung).",
    )
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
        help="legacy: how much latency the fit_families exact(legacy, vs\n"
        "mesh/ring) row may give up to keep the bit-exact one-shot, as a\n"
        "ratio. Default 1.10. Printed for comparison only -- the shipped\n"
        "oneshot_max_exact comes from fit_oneshot_exact_ceiling instead, which\n"
        "has no slack knob (see EXACT_FALLBACK_PRIMARY/ROBUST).",
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
        "dispatcher, forced to accuracy=fast -- against the best pinned row at\n"
        "every shape, and exit non-zero if any exceeds --ladder-slack. This is\n"
        "the end-to-end acceptance test for the tables; run it on a sweep that\n"
        "includes -c fly_auto alongside the pinned rows.",
    )
    ap.add_argument(
        "--require-timing",
        choices=["graph", "eager"],
        default=None,
        help="refuse inputs not measured with this --timing regime. The CSVs\n"
        "carry the regime per row and mixtures are always refused; this is for\n"
        "pinning a fit to one of them on purpose, e.g. in a script that must\n"
        "never accidentally consume an eager sweep.",
    )
    ap.add_argument(
        "--min-sqnr",
        type=float,
        default=None,
        help="exclude candidates below this SQNR from the fit *and* from\n"
        "the oracle. The fused epilogue makes this sharper than it looks:\n"
        "dividing by a row's own norm amplifies whatever quantization\n"
        "variance it had, and the latency-optimal fused choice at TP8 is a\n"
        "~20 dB mesh from M=8 upward. Default: rank on time alone.",
    )
    ap.add_argument("--verbose", action="store_true", help="per-shape regret detail")
    args = ap.parse_args()
    set_fusion_mode(args.fusion)
    global MIN_SQNR
    MIN_SQNR = args.min_sqnr
    if MIN_SQNR is not None:
        logger.info("# accuracy floor: %.1f dB\n", MIN_SQNR)

    samples, prov = load(args.csv, args.metric, require_timing=args.require_timing)
    if args.fusion != "none":
        if args.audit_auto:
            logger.info(
                "# auditing fused_fly_auto over %d shape(s), metric %r, "
                "tolerance %.0f%%\n",
                len(samples),
                args.metric,
                (args.ladder_slack - 1) * 100,
            )
            n = audit_auto(
                samples,
                args.ladder_slack,
                args.verbose,
                auto_key="fused_fly_auto",
            )
            logger.info(
                "\n%s",
                f"FAILURES: {n}" if n else "PASS: every shape within tolerance",
            )
            raise SystemExit(1 if n else 0)
        holdout = load(args.holdout, args.metric)[0] if args.holdout else []
        logger.info(
            "# fused fit from %d file(s), %d shape(s), TP %s, metric %r, link %r\n",
            len(args.csv),
            len(samples),
            sorted({s.tp for s in samples}),
            args.metric,
            args.link,
        )
        fit_fused(samples, holdout, args)
        return
    if args.audit_auto:
        logger.info(
            "# auditing fly_auto over %d shape(s), metric %r, accuracy %s, "
            "tolerance %.0f%%\n",
            len(samples),
            args.metric,
            prov["accuracy"],
            (args.ladder_slack - 1) * 100,
        )
        n = audit_auto(
            samples, args.ladder_slack, args.verbose, accuracy=prov["accuracy"]
        )
        if prov["accuracy"] == "exact":
            # The sizes exact mode refuses are not in the latency audit at all;
            # grading the decline is the other half of the same question.
            n += audit_declines(
                samples,
                args.ladder_slack,
                EXACT_FALLBACK_PRIMARY,
                [c.key for c in BY_FAMILY["oneshot"]],
                args.verbose,
            )
        logger.info(
            "\n%s",
            f"FAILURES: {n}" if n else "PASS: every shape within tolerance",
        )
        raise SystemExit(1 if n else 0)
    if prov["accuracy"] == "exact":
        raise SystemExit(
            "these inputs were swept with --fly-accuracy exact, which measures "
            "no mesh/ring rows at all -- fit_families has no mesh or ring family "
            "to fit and would exit on the first world size. Fit from a "
            "--fly-accuracy fast sweep (it carries the one-shot and cdr rows the "
            "exact ceiling needs too), and use an exact sweep with --audit-auto."
        )
    holdout = load(args.holdout, args.metric)[0] if args.holdout else []
    tps = sorted({s.tp for s in samples})
    logger.info(
        "# fit from %d file(s), %d shape(s), TP %s, metric %r, link %r\n",
        len(args.csv),
        len(samples),
        tps,
        args.metric,
        args.link,
    )

    oneshot_keys = [c.key for c in BY_FAMILY["oneshot"]]
    fam_rows, ladder_rows = [], []
    exact_ceiling: dict[tuple[str, int], int] = {}
    exact_worst: dict[tuple[str, int], float] = {}
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
            one, mesh, worst, mean, detail, _floor = fit_families(
                sub, exact_slack=slack
            )
            label = "fast" if mode == "fast" else "exact(legacy, vs mesh/ring)"
            logger.info(
                "  %-27s oneshot <= %-9s  mesh <= %-9s   worst %.3fx  mean %.3fx",
                label,
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

        # The real "exact" mode ceiling: one-shot vs. the exact fallback its
        # decline path actually reaches (see EXACT_FALLBACK_PRIMARY/ROBUST).
        # This -- not the fit_families exact(legacy) row above -- is what
        # feeds the shipped oneshot_max_exact.
        new_ceiling = {}
        for elabel, oracle_keys in (
            ("primary", EXACT_FALLBACK_PRIMARY),
            ("robust", EXACT_FALLBACK_ROBUST),
        ):
            ceiling, worst, mean, detail, _sat = fit_oneshot_exact_ceiling(
                sub, oneshot_keys, oracle_keys, extra=cdr_edge_thresholds(tp)
            )
            new_ceiling[elabel] = ceiling
            if elabel == "primary":
                exact_worst[(args.link, tp)] = worst
            logger.info(
                "  exact vs %-7s          oneshot <= %-9s               "
                "worst %.3fx  mean %.3fx  (oracle: %s)",
                elabel,
                human(ceiling),
                worst,
                mean,
                "+".join(oracle_keys),
            )
            if args.verbose:
                for nbytes, ratio in detail:
                    if ratio > 1.001:
                        logger.info("      %-10s %.3fx", human(nbytes), ratio)
        if new_ceiling["robust"] != new_ceiling["primary"]:
            logger.warning(
                "  exact ceiling: primary (cdr only, %s) differs from robust "
                "(cdr/cdr_naive/rccl, %s) -- shipping primary, since cdr is\n"
                "  what production actually falls through to.",
                human(new_ceiling["primary"]),
                human(new_ceiling["robust"]),
            )
        exact_ceiling[(args.link, tp)] = new_ceiling["primary"]

        # Each family's ladder is fitted only over the sizes the family policy
        # actually sends it. The exact-mode one-shot window is the wider of the
        # two, so use it -- a rung the fast mode never reaches is harmless, a
        # missing rung the exact mode does reach is not.
        one_exact = new_ceiling["primary"]
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
        one_exact = reconcile_family_row(link, tp, one, exact_ceiling[(link, tp)], mesh)
        logger.info(
            "    (%r, %d): FamilyPolicy(oneshot_max=%d, oneshot_max_exact=%d, "
            "mesh_max=%d),  # %s / %s / %s, worst %.3fx",
            link,
            tp,
            one,
            one_exact,
            mesh,
            human(one),
            human(one_exact),
            human(mesh),
            max(worst, exact_worst[(link, tp)]),
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
    # The one-shot ladder additionally comes out in the form it is pasted in.
    # The key form above stays because the alias and non-monotone warnings name
    # keys, and a rung tuple cannot be traced back to the row that won.
    oneshot_ladders = [r for r in ladder_rows if r[1] == "oneshot"]
    if oneshot_ladders:
        logger.info("\n_ONESHOT_LADDER = {")
        for tp, _family, rungs, worst in oneshot_ladders:
            try:
                tuples = [CAND_BY_KEY[k].fly1s_rung(lo) for lo, k in rungs]
            except (KeyError, ValueError) as exc:
                logger.warning("    # (%r, %d): %s", args.link, tp, exc)
                continue
            # Trailing comma on every rung, or a one-rung ladder pastes as a
            # plain tuple rather than a tuple of one rung and every consumer
            # unpacks an int where it wants a rung.
            logger.info(
                "    (%r, %d): (%s),  # worst %.3fx",
                args.link,
                tp,
                "".join(f"{t!r}, " for t in tuples).rstrip(),
                worst,
            )
        logger.info("}")


if __name__ == "__main__":
    main()
