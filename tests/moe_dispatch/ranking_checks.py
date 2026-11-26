# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Checks of the default ranking table (`aiter/moev2/dispatch/ranking.py`, plan §3.6).

Static: entries and regions are disjoint, no region contradicts itself, every structure
names known, adjacent segments, and the last structure of every list is always feasible.
Over the dispatch snapshot key grid, on every simulated arch: every key some registered
path can rank lands in exactly one entry and one region whose last structure is
feasible, and every region is reached by some key.

    python tests/moe_dispatch/ranking_checks.py --check [--arch gfx950]

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import pathlib
import sys
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parent


def segment_sets():
    from aiter.moev2 import paths

    return [p.segments for p in paths.registered().values() if p.segments]


def grid_problems(arch):
    """`(label, MoeProblem)` for every snapshot key of `arch`; call in its simulation."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.moev2 import api
    from moe_dispatch import snapshot

    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    for key in snapshot.dispatch_keys(gfx, cu_num):
        try:
            problem = api.problem_from_key(key.args, key.call_kwargs())
        except Exception:  # noqa: BLE001, S112 - key_checks.py reports these
            continue
        yield f"{arch} {key.label()}", problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    args = parser.parse_args(argv)

    # `tests` so `moe_dispatch` imports, and the repository root so `aiter` is
    # this checkout rather than an installed copy, even one earlier on the path.
    front = [str(_HERE.parent), str(_HERE.parents[1])]
    sys.path[:] = front + [p for p in sys.path if p not in front]
    from moe_dispatch import snapshot

    snapshot._setup()
    from aiter.moev2.dispatch import ranking
    from moe_dispatch import arch_sim
    from moe_dispatch.differential import captured_logs

    sets = segment_sets()
    errors = ranking.check_table(ranking.RANKING, [s for ss in sets for s in ss])
    print(f"static: {len(ranking.RANKING)} entries, {len(errors)} problems")
    hits = Counter()
    for arch in args.arch or list(arch_sim.ARCHS):
        with arch_sim.simulated_arch(arch), captured_logs():
            grid_errors, arch_hits = ranking.check_problems(grid_problems(arch), sets)
        print(
            f"{arch}: {sum(arch_hits.values())} ranked keys, {len(grid_errors)} problems"
        )
        errors += grid_errors
        hits += arch_hits
    errors += ranking.dead_lists(hits)
    for label, n in sorted(hits.items()):
        print(f"  {label}: {n} keys")
    for line in errors[:40]:
        print("  " + line)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
