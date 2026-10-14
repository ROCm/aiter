# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Install step: pick the fastest config per swept M and write the family's config file.

Flow:  load_winners    fastest ok record per M; the installed baseline competes too
       assign_buckets  M -> M_LEQ_<smallest family bound >= M>; the largest M wins a shared bucket;
                       an M above the largest bound is not written
       build_table     seeded with this shape's file if it exists, else a new file marked
                       DEFAULT_FALLBACK (the loader serves its missing buckets from DEFAULT.json);
                       "any" is a copy of DEFAULT.json's; DEFAULT.json is never written
       write, confirm  write the file, re-resolve every assigned M through the real loader, and
                       put the previous file back if the loader disagrees

sweep_configs.py runs this when a sweep finishes; run it by hand to re-install from results.
"""

import argparse
import copy
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

from _utils import (
    add_shape_args,
    bucket_for,
    config_key,
    final_results_path,
    plan_path,
    read_records,
    results_files,
    shape_from_args,
    specialized_filename,
)
from kernels import ensure_repo_on_path, get_spec, resolve_installed, seed_table
from space import load_defaults

HERE = os.path.dirname(os.path.abspath(__file__))

Winner = namedtuple(
    "Winner", "M record baseline"
)  # baseline: the installed config's record or None


def load_winners(runs_dir, arch, backend, spec, shape_nk):
    """M -> Winner from every per-M result file of this (kernel, backend, N, K[, B]).

    When several GPUs swept an M, only the final round (<results>.final.jsonl, timed on one
    GPU) counts. When the M's plan file is present only its candidates (and the baseline) can win,
    so records of candidates that are no longer in the search space are ignored.
    """
    winners = {}
    for path in results_files(runs_dir, arch, backend, spec.name, shape_nk):
        records = read_records(path)
        ok = [r for r in records.values() if r["status"] == "ok"]
        if not ok:
            continue
        M = ok[0]["shape"]["M"]
        if os.path.exists(final_results_path(path)):
            records = read_records(final_results_path(path))
            ok = [r for r in records.values() if r["status"] == "ok"]
        elif (
            len({r.get("gpu") for r in ok} - {None}) > 1
        ):  # records without a gpu are from one worker
            print(
                f"  M={M}: timed on several GPUs but the final round is missing; rerun the sweep. Skipped."
            )
            continue
        plan_file = plan_path(runs_dir, arch, backend, spec.name, dict(shape_nk, M=M))
        if os.path.exists(plan_file):
            with open(plan_file) as f:
                current = {config_key(c) for c in json.load(f)["configs"]}
            ok = [
                r for r in ok if r.get("baseline") or config_key(r["config"]) in current
            ]
        else:
            print(f"  M={M}: no plan file, every recorded candidate may win")
        if not ok:
            print(f"  M={M}: no successful record of a current candidate; skipped")
            continue
        baseline = records.get("baseline")
        if baseline is not None and baseline["status"] != "ok":
            baseline = None
        winners[M] = Winner(M, min(ok, key=lambda r: r["us"]), baseline)
    return winners


def assign_buckets(winners, bounds):
    """bucket name -> Winner. Several swept Ms in one bucket: the largest M wins."""
    assignments = {}
    for M in sorted(winners):
        bucket = bucket_for(M, bounds)
        if bucket is None:
            print(
                f"  M={M}: above the largest bound {bounds[-1]}, not written ('any' is DEFAULT.json's)"
            )
            continue
        if bucket in assignments:
            print(
                f"  M={assignments[bucket].M} and M={M} both map to {bucket}; keeping M={M}"
            )
        assignments[bucket] = winners[M]
    return assignments


def build_table(seed, assignments, arch, keys):
    """The seed table with the winners written into their buckets.

    Only the family's DEFAULT.json keys are written (kpack also only on gfx942, persistent never);
    seed buckets that still carry other keys are left as they are and reported.
    """
    table = copy.deepcopy(seed)
    for bucket, entry in table.items():
        extra = (
            sorted(k for k in entry if k not in keys) if isinstance(entry, dict) else []
        )
        if extra:
            print(
                f"  note: untouched bucket {bucket} carries keys not in DEFAULT.json: {extra}"
            )
    for bucket, winner in assignments.items():
        table[bucket] = {
            k: v
            for k, v in winner.record["config"].items()
            if k in keys and k != "persistent" and (k != "kpack" or arch == "gfx942")
        }
    return table


def ordered(table):
    """M_BOUNDS, M_LEQ ascending, M_GEQ descending, any, rest: the order get_gemm_config walks."""

    def rank(key):
        if key in ("DEFAULT_FALLBACK", "M_BOUNDS"):
            return (0, 0)
        if key.startswith("M_LEQ_"):
            return (1, int(key[6:]))
        if key.startswith("M_GEQ_"):
            return (2, -int(key[6:]))
        return (3, 0) if key == "any" else (4, 0)

    return {k: table[k] for k in sorted(table, key=rank)}


def install(spec, backend, shape_nk, runs_dir):
    ensure_repo_on_path()
    from aiter.ops.triton.utils import gemm_config_utils
    from aiter.ops.triton.utils._triton import arch_info
    from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

    arch = arch_info.get_arch()
    backend = backend or spec.default_backend(arch)
    winners = load_winners(runs_dir, arch, backend, spec, shape_nk)
    if not winners:
        sys.exit(f"no results for {spec.name} {backend} {shape_nk} in {runs_dir}")

    seed_path, seed = seed_table(spec, backend, shape_nk)
    print(f"seeding from {seed_path}")
    # the loader walks the kernel's explicit bounds, else the M_BOUNDS of the file it picks
    bounds = spec.bounds or seed.get("M_BOUNDS") or gemm_config_utils.STANDARD_M_BOUNDS
    keys, default_buckets, _ = load_defaults(spec, backend)
    assignments = assign_buckets(winners, bounds)
    if not assignments:
        sys.exit(
            f"no swept M is within the bounds (largest {bounds[-1]}); nothing installed"
        )
    table = build_table(seed, assignments, arch, keys)
    # every file ships DEFAULT.json's "any": a tuned small-M tile must never become the fallback
    table["any"] = copy.deepcopy(default_buckets["any"])
    for bucket, winner in sorted(assignments.items(), key=lambda kv: kv[1].M):
        gain = ""
        if winner.baseline is not None:
            gain = f"{winner.baseline['us'] / winner.record['us']:.3f}x vs installed {winner.baseline['us']:.3f} us"
        print(
            f"  M={winner.M:<6} {bucket:<12} {winner.record['us']:10.3f} us  {gain:<36} {json.dumps(table[bucket])}"
        )

    config_dir = resolve_config_dir("gemm", spec.config_name, backend=backend)
    target = f"{config_dir}/{specialized_filename(spec.config_name, shape_nk)}"
    previous = Path(target).read_text() if os.path.exists(target) else None
    with open(target, "w") as f:
        json.dump(ordered(table), f, indent=4)
        f.write("\n")
    print(f"installed {target}")

    # confirm through the real loader (both caches hold the old file)
    load_config_json.cache_clear()
    gemm_config_utils._get_gemm_config_cached.cache_clear()
    for bucket, winner in assignments.items():
        got, is_tuned = resolve_installed(spec, dict(shape_nk, M=winner.M), backend)
        if got != table[bucket] or not is_tuned:
            if previous is None:  # never leave a bad file in the tree
                os.remove(target)
            else:
                Path(target).write_text(previous)
            sys.exit(
                f"validation failed for M={winner.M} ({bucket}): the loader returned "
                f"is_tuned={is_tuned} {json.dumps(got)}; previous file restored"
            )
    print(f"validated {len(assignments)} bucket(s) through the loader")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kernel", type=get_spec)
    add_shape_args(parser, with_m=False)
    parser.add_argument("--backend", choices=("triton", "gluon"))
    parser.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    args = parser.parse_args(argv)
    install(
        args.kernel,
        args.backend,
        shape_from_args(args, args.kernel.dims),
        args.runs_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
