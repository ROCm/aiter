# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where fusing the RMSNorm pays, and where the FlyDSL kernels beat aiter's.

Two questions, one sweep:

1. **Is fusing beneficial?** Every fused kernel against its *own* two-launch
   baseline -- the same all-reduce followed by ``rmsnorm2d_fwd_with_add``. Pairing
   each kernel with itself is what isolates the fusion from the schedule: a
   fused mesh beating a separate ring says nothing about fusion.
2. **Do the new FlyDSL fused kernels beat the ones aiter ships?** Two answers,
   not one, because they are different questions: best FlyDSL-fused against
   what aiter's own dispatch heuristic (``production_fused_path()``) would
   actually run at that shape (``_prod_dispatch_table`` -- "what changes if
   you swap these kernels in today"), and separately against the fastest of
   *every* aiter-fused candidate measured regardless of whether aiter's
   dispatch would pick it (``_winner_table`` -- an oracle ceiling that isolates
   the kernel comparison from the dispatch one). They can disagree a lot: aiter
   sometimes dispatches to a kernel that is not its own fastest option.

This drives ``bench_comm_allreduce.py`` once per (TP, hidden) and reduces the
per-run CSVs; it measures nothing itself, so the numbers are exactly what the
bench reports and the commands it ran are printed for reproduction.

Why one invocation per (TP, hidden) rather than one big one: each writes its own
CSV, so a failure at TP8 does not lose TP2's data and a subset can be re-run
without redoing the sweep.

Two knobs matter more than the rest:

``--timing`` The bench's default, ``graph``, times a captured replay -- launch
    overhead removed from *both* sides. That is the metric a graph-capturing
    deployment sees, and it understates fusion by exactly the launch and device
    sync that fusing removes. At decode sizes, where a kernel is ~10 us and a
    launch is a few, run ``eager`` as well; ``--timing both`` does both passes.

``--fly1s-max-kb`` ``OneShotAllReduce`` is gated above by a *policy* ceiling
    (wire volume is ``(N-1)x`` the message), not a correctness limit. Left
    alone, its rows are ``n/a`` over most of this sweep and the decode
    comparison comes back empty. The default here lifts it past the largest
    shape so the curve is visible all the way to where it stops winning.

The FlyDSL rows include the pinned ``_b<block>[_ss]`` grid, not just the
ladder-driven rows: block and self-skip move the fused kernels by up to 2.6x, so
a comparison built from the unpinned rows alone understates them badly. Each
distinct config is an engine with its own IPC inbox -- ~470 MiB per rank at
TP2/hidden=8192 with everything on -- so trim with ``-c`` when sweeping a single
family. Which blocks exist is a function of the width, so a different subset of
the ``b*`` rows reports at each hidden; that is the geometry, not a failure.

Usage::

    # the whole sweep, ~30-45 min on 8 GPUs with a warm JIT cache
    python op_tests/multigpu_tests/sweep_fused_allreduce.py --outdir sweep_out

    # just print what it would run
    python op_tests/multigpu_tests/sweep_fused_allreduce.py --dry-run

    # re-reduce CSVs from an earlier run
    python op_tests/multigpu_tests/sweep_fused_allreduce.py --outdir sweep_out \
        --analyze-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE / "bench_comm_allreduce.py"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# Reuse the bench's own prod-path parser rather than re-deriving it: it
# already handles every `production_fused_path()` string, including
# `qr_fused:<regime>` for any regime, not just the ones this sweep exercises.
from bench_comm_allreduce import CANDIDATES as _BENCH_CANDIDATES
from bench_comm_allreduce import _prod_candidate_key

# Decode through prefill, ~4 points per octave at the bottom. The density is
# what lets `fit_allreduce_policy.py` place a threshold: measured points have to
# land near the round values it searches over, and both the fused `min_bytes`
# floor and the one-shot/mesh crossover sit below 512.
DEFAULT_M = (
    1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 48, 64, 96,
    128, 192, 256, 384, 512, 1024, 2048, 4096, 8192,
)  # fmt: skip

# Widths the dispatch table is fitted on. 4096 and 8192 carry the full
# three-wide block axis; 7168 is DeepSeek-V3/V4's and the one width where
# aiter's fused quick-reduce cannot run at all (a 14336 B row does not tile a
# 32 KiB QR tile), so the FlyDSL two-shot kernels are the only quantized fused
# option there.
FIT_HIDDEN = (4096, 8192, 7168)
# Held out to test whether the fitted boundary is a function of bytes alone.
HOLDOUT_HIDDEN = (2048, 3072, 6144)
DEFAULT_HIDDEN = FIT_HIDDEN + HOLDOUT_HIDDEN
DEFAULT_TP = (2, 4, 8)

# Payload ceiling for the one-shot rows, in KiB. 128 MiB clears the largest
# shape here (8192 x 8192 bf16 = 128 MiB).
DEFAULT_FLY1S_MAX_KB = 131072


def _bench_keys(family: str, algorithm: str | None = None) -> tuple[str, ...]:
    """Fused bench rows of *family*, pinned grid included.

    Taken from the bench rather than listed here: which blocks exist is the
    kernel's constraint, and the bench already generates the rows from it.
    """
    return tuple(
        c.key
        for c in _BENCH_CANDIDATES
        if c.family == family and (algorithm is None or c.algorithm == algorithm)
    )


#: Every FlyDSL fused row, per schedule. The pinned `_b<block>[_ss]` rows are
#: what makes these comparisons fair: tuning moves the fused kernels by up to
#: 2.6x, so a table built from the unpinned rows alone understates them.
FLY_1STAGE = _bench_keys("fused_fly1s")
FLY_RING = _bench_keys("fused_flyqr", "ring")
FLY_MESH = _bench_keys("fused_flyqr", "mesh")

#: ``family -> (fused rows, its own two-launch baseline)``. The fused side is a
#: tuple and reduced with :func:`_best`, so a family is represented by its best
#: measured config at each shape.
#:
#: ``separate_cdr`` serves both cdr rows: it is ``cross_device_reduce`` plus a
#: standalone norm, which is the unfused form of either fused schedule.
FUSION_PAIRS = {
    "cdr_1stage": (("fused_cdr_1stage",), "separate_cdr"),
    "cdr_2stage": (("fused_cdr_2stage",), "separate_cdr"),
    "qr_int4": (("fused_qr_int4",), "separate_qr_int4"),
    "fly_1stage": (FLY_1STAGE, "separate_fly1s"),
    "fly_ring": (FLY_RING, "separate_flyring"),
    "fly_mesh": (FLY_MESH, "separate_flymesh"),
    # The shipped policy against itself: same size->family rule on both sides,
    # so this row is the fusion gain production actually gets rather than the
    # gain one hand-pinned schedule would get. Every other entry above pins a
    # schedule, so their fused and separate sides are guaranteed to match; here
    # they are only guaranteed to match *if the two policies agree*, which is
    # itself worth seeing.
    "fly_auto": (("fused_fly_auto",), "separate_fly_auto"),
}

FLYDSL_FUSED = FLY_1STAGE + FLY_RING + FLY_MESH
FLYDSL_SEPARATE = ("separate_fly1s", "separate_flyring", "separate_flymesh")
#: The shipped fused dispatcher. Measured so `fit_allreduce_policy.py
#: --fusion ar_rmsnorm --audit-auto` can grade it, but kept out of
#: `FLYDSL_FUSED`: it is not another kernel to compare, it is the policy that
#: chooses between them, and counting it as a candidate would double-count.
#: Needs AITER_FLY_AR=1 in the environment or its column comes back n/a.
FLYDSL_AUTO = ("fused_fly_auto",)
#: Its two-launch baseline -- `FlyDSLAllReduce` plus a standalone norm. Kept out
#: of `FLYDSL_SEPARATE` for the same reason `FLYDSL_AUTO` is kept out of
#: `FLYDSL_FUSED`: it is the policy, not another schedule to compare, and
#: counting it alongside separate_fly1s/flyring/flymesh would double-count
#: whichever of them the policy chose. Same AITER_FLY_AR=1 requirement.
FLYDSL_AUTO_SEPARATE = ("separate_fly_auto",)
AITER_FUSED = ("fused_cdr_1stage", "fused_cdr_2stage", "fused_qr_int4", "fused_qr_fp8")
AITER_SEPARATE = ("separate_cdr", "separate_rccl", "separate_qr_int4")
#: aiter candidates with no accuracy loss (``Candidate.exact`` in the bench).
#: Quantized aiter candidates (``fused_qr_*``, ``separate_qr_int4``) are
#: deliberately excluded here -- see :func:`_category_table`.
AITER_EXACT = ("fused_cdr_1stage", "fused_cdr_2stage", "separate_cdr", "separate_rccl")
AITER_ALL = tuple(sorted(set(AITER_FUSED) | set(AITER_SEPARATE)))

CANDIDATES = sorted(
    {k for fused, _sep in FUSION_PAIRS.values() for k in fused}
    | {sep for _fused, sep in FUSION_PAIRS.values()}
    | set(FLYDSL_FUSED)
    | set(FLYDSL_AUTO)
    | set(FLYDSL_AUTO_SEPARATE)
    | set(AITER_FUSED)
    | {"separate_rccl"}  # library reference, free to carry
)


def _commands(args) -> list[tuple[tuple, list[str], Path]]:
    """``[((tp, hidden, timing), argv, csv_path)]`` for the whole sweep."""
    out = []
    timings = ("graph", "eager") if args.timing == "both" else (args.timing,)
    for timing in timings:
        for tp in args.tp:
            for hidden in args.hidden:
                csv = Path(args.outdir) / f"fused_tp{tp}_k{hidden}_{timing}.csv"
                argv = [
                    sys.executable,
                    str(_BENCH),
                    "-tp",
                    str(tp),
                    "--fusion",
                    "ar_rmsnorm",
                    "--timing",
                    timing,
                    "--iters",
                    str(args.iters),
                    "-s",
                    *[f"{m},{hidden}" for m in args.m],
                    "-c",
                    *CANDIDATES,
                    "--output-csv",
                    str(csv),
                ]
                out.append(((tp, hidden, timing), argv, csv))
    return out


def _run(args) -> None:
    env = dict(os.environ)
    # Lift the one-shot's policy ceiling; see the module docstring.
    env["AITER_BENCH_FLY1S_MAX_KB"] = str(args.fly1s_max_kb)
    # The fused dispatcher is opt-in and self-disabling; without this its row
    # is n/a everywhere and --audit-auto has nothing to grade.
    env.setdefault("AITER_FLY_AR", "1")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for key, argv, csv in _commands(args):
        print(f"\n=== TP{key[0]} hidden={key[1]} timing={key[2]} -> {csv}", flush=True)
        print("    " + " ".join(argv), flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        proc = subprocess.run(argv, env=env, check=False)
        dt = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"=== {status} in {dt / 60:.1f} min", flush=True)
        if proc.returncode != 0 and not args.keep_going:
            raise SystemExit(proc.returncode)


def _load(outdir: Path) -> pd.DataFrame:
    frames = []
    for csv in sorted(outdir.glob("fused_tp*_k*_*.csv")):
        df = pd.read_csv(csv)
        df["timing"] = csv.stem.rsplit("_", 1)[-1]
        frames.append(df)
    if not frames:
        raise SystemExit(f"no sweep CSVs in {outdir}")
    return pd.concat(frames, ignore_index=True)


def _fusion_table(df: pd.DataFrame, min_gain: float) -> pd.DataFrame:
    """Speedup of each fused kernel over its own two-launch baseline.

    ``> 1`` means fusing won. ``NaN`` means one of the pair was not applicable
    at that shape, which is itself informative -- an ``n/a`` in the fly_ring
    column at hidden=2560 is the row-sized block's width constraint, and one in
    qr_int4 at 7168 is the QR tile's.
    """
    rows = []
    for _, r in df.iterrows():
        row = {
            "timing": r["timing"],
            "TP": r["TP"],
            "K": r["K"],
            "M": r["M"],
            "KiB": r["payload size (KiB)"],
        }
        for name, (fused, sep) in FUSION_PAIRS.items():
            _k, f, _db = _best(r, fused, None)
            s = r.get(f"{sep} us")
            row[name] = (
                (s / f)
                if (pd.notna(s) and f > 0 and f != float("inf"))
                else float("nan")
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["min_gain"] = min_gain
    return out


def _best(r, cands, min_sqnr):
    """Fastest of *cands* applicable at this row, above *min_sqnr* if given."""
    best, best_us, best_db = None, float("inf"), float("nan")
    for c in cands:
        us, db = r.get(f"{c} us"), r.get(f"{c} SQNR dB")
        if pd.isna(us):
            continue
        if min_sqnr is not None and (pd.isna(db) or db < min_sqnr):
            continue
        if us < best_us:
            best, best_us, best_db = c, us, db
    return best, best_us, best_db


def _winner_table(df: pd.DataFrame, min_sqnr: float | None) -> pd.DataFrame:
    """Best FlyDSL-fused against best *of every aiter-fused candidate measured*,
    per shape -- an oracle ceiling, not what aiter's own dispatch would pick.

    Ranked on time alone unless ``--min-sqnr`` is given: the quantized
    candidates buy their speed with accuracy, so a table that ranks a 15 dB
    kernel above a 40 dB one without saying so is a trap. Both winners carry
    their own dB for exactly that reason.

    This is deliberately optimistic about aiter: it picks whichever of
    ``fused_cdr_1stage``/``fused_cdr_2stage``/``fused_qr_*`` measured fastest
    at this exact shape, regardless of which one aiter's own size-based
    heuristic would actually select. See :func:`_prod_dispatch_table` for the
    comparison against what a deployment running today's aiter would get.
    """
    rows = []
    for _, r in df.iterrows():
        fly, fly_us, fly_db = _best(r, FLYDSL_FUSED, min_sqnr)
        ait, ait_us, ait_db = _best(r, AITER_FUSED, min_sqnr)
        rows.append(
            {
                "timing": r["timing"],
                "TP": r["TP"],
                "K": r["K"],
                "M": r["M"],
                "KiB": r["payload size (KiB)"],
                "flydsl": fly,
                "flydsl us": fly_us if fly else float("nan"),
                "flydsl dB": fly_db,
                "aiter": ait,
                "aiter us": ait_us if ait else float("nan"),
                "aiter dB": ait_db,
                "speedup": (ait_us / fly_us)
                if (fly and ait and fly_us > 0)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _prod_dispatch_table(df: pd.DataFrame, min_sqnr: float | None) -> pd.DataFrame:
    """Best FlyDSL-fused against what aiter's *actual dispatch heuristic*
    would run at this shape today -- the fair comparison, not an oracle.

    ``prod path`` (from ``bench_comm_allreduce.py``'s own
    ``production_fused_path()``) says which kernel ``CudaCommunicator``
    would pick, independent of which one happens to measure fastest; this
    looks up *that* candidate's own measured time and SQNR rather than
    :func:`_winner_table`'s best-of-every-fused-candidate. The two can and do
    disagree: at every TP/hidden in this sweep, aiter's heuristic dispatches
    to ``fused_cdr_2stage`` for nearly every mid-to-large shape even though
    ``fused_cdr_1stage`` measures 3-4x faster there -- so the oracle table
    understates today's real gap and this one is what a user actually gets.
    """
    rows = []
    for _, r in df.iterrows():
        fly, fly_us, fly_db = _best(r, FLYDSL_FUSED, min_sqnr)
        path = r.get("prod path")
        cand = _prod_candidate_key(path) if isinstance(path, str) else None
        ait_us = r.get(f"{cand} us") if cand else float("nan")
        ait_db = r.get(f"{cand} SQNR dB") if cand else float("nan")
        rows.append(
            {
                "timing": r["timing"],
                "TP": r["TP"],
                "K": r["K"],
                "M": r["M"],
                "KiB": r["payload size (KiB)"],
                "flydsl": fly,
                "flydsl us": fly_us if fly else float("nan"),
                "flydsl dB": fly_db,
                "prod path": path,
                "aiter us": ait_us,
                "aiter dB": ait_db,
                "speedup": (ait_us / fly_us)
                if (fly and pd.notna(ait_us) and fly_us > 0)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _analyze(args) -> None:
    df = _load(Path(args.outdir))
    sort = ["timing", "TP", "K", "M"]

    fusion = _fusion_table(df, args.min_gain).sort_values(sort)
    winner = _winner_table(df, args.min_sqnr).sort_values(sort)
    prod = _prod_dispatch_table(df, args.min_sqnr).sort_values(sort)

    print("\n## Is fusing beneficial? (separate us / fused us; > 1.00 = fuse)\n")
    print(fusion.to_markdown(index=False, floatfmt=".2f"))

    print(
        "\n## FlyDSL fused vs aiter's actual dispatch (speedup > 1.00 = FlyDSL wins)\n"
    )
    print(
        "The fair comparison: `aiter us`/`aiter dB` is whichever kernel "
        "`production_fused_path()` says a real deployment would run at this "
        "shape today (`prod path`), not the fastest aiter candidate measured.\n"
    )
    print(prod.to_markdown(index=False, floatfmt=".2f"))

    print(
        "\n## FlyDSL fused vs best-of-aiter-fused, oracle ceiling (speedup > "
        "1.00 = FlyDSL wins)\n"
    )
    print(
        "Not what aiter's dispatch would pick -- the fastest of every "
        "*measured* aiter-fused candidate at this shape, regardless of "
        "whether aiter's own heuristic would ever choose it. Compare against "
        "the table above: where they disagree, aiter's dispatch is leaving "
        "its own better kernel on the table.\n"
    )
    print(winner.to_markdown(index=False, floatfmt=".2f"))

    combined = Path(args.outdir) / "summary.csv"
    fusion.to_csv(combined.with_name("fusion_benefit.csv"), index=False)
    winner.to_csv(combined.with_name("flydsl_vs_aiter_oracle.csv"), index=False)
    prod.to_csv(combined.with_name("flydsl_vs_aiter_prod_dispatch.csv"), index=False)
    print(f"\nwrote {combined.with_name('fusion_benefit.csv')}")
    print(f"wrote {combined.with_name('flydsl_vs_aiter_prod_dispatch.csv')}")
    print(f"wrote {combined.with_name('flydsl_vs_aiter_oracle.csv')}")

    # The one-line answers, so the tables do not have to be read to get them.
    wins = fusion.melt(
        id_vars=["timing", "TP", "K", "M", "KiB"], var_name="family", value_name="gain"
    ).dropna()
    print("\n## Where each fusion pays\n")
    for family, grp in wins.groupby("family"):
        good = grp[grp["gain"] >= args.min_gain]
        if good.empty:
            print(
                f"- `{family}`: never, across {len(grp)} shape(s) "
                f"(best {grp['gain'].max():.2f}x)"
            )
            continue
        lo, hi = good["KiB"].min(), good["KiB"].max()
        print(
            f"- `{family}`: {len(good)}/{len(grp)} shape(s), "
            f"{lo:.0f}-{hi:.0f} KiB, up to {good['gain'].max():.2f}x"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tp", type=int, nargs="+", default=list(DEFAULT_TP))
    p.add_argument("--hidden", type=int, nargs="+", default=list(DEFAULT_HIDDEN))
    p.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_M))
    p.add_argument("--outdir", default="sweep_fused_allreduce")
    p.add_argument("--timing", choices=("graph", "eager", "both"), default="graph")
    p.add_argument("--iters", type=int, default=101)
    p.add_argument(
        "--fly1s-max-kb",
        type=int,
        default=DEFAULT_FLY1S_MAX_KB,
        help="lift OneShotAllReduce's policy ceiling for the sweep (KiB)",
    )
    p.add_argument(
        "--min-gain",
        type=float,
        default=1.02,
        help="speedup a fusion must reach to count as beneficial (default 1.02, "
        "i.e. 2%% -- inside that the bench's own spread is the same size)",
    )
    p.add_argument(
        "--min-sqnr",
        type=float,
        default=None,
        help="exclude candidates below this SQNR from the winner table, e.g. 40 "
        "for exact-only. Default: rank on time and report each winner's dB.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    args = p.parse_args()

    if not args.analyze_only:
        _run(args)
    if not args.dry_run:
        _analyze(args)


if __name__ == "__main__":
    main()
