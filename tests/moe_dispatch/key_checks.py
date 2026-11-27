# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-key checks of the v2 dispatch inputs against the legacy code.

For every `get_2stage_cfgs` key of the snapshot, on every simulated arch:

- `MoeProblem`: the key builds a problem; `legacy_key()` maps it back to a key that
  builds an equal problem with the same hash and that legacy dispatches identically;
- `tuned.lookup` returns the very row object the legacy lookup returns from legacy's
  tables (or both None), and an equal row from v2's own read of the table.

    python tests/moe_dispatch/key_checks.py --check [--arch gfx950]

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import math
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent


_HITS = []


class _Recording(dict):
    """A tuned-table dict that appends what `get` returns to `_HITS`."""

    def get(self, key, default=None):
        value = dict.get(self, key, default)
        _HITS.append(value)
        return value


def _same_row(a, b):
    def plain(row):
        if row is None:
            return None
        return {
            k: None if isinstance(v, float) and math.isnan(v) else v
            for k, v in row.items()
        }

    return plain(a) == plain(b)


def _legacy_record(fm, record, args, kwargs):
    try:
        return record.normalize(fm.get_2stage_cfgs.__wrapped__(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        return record.error_record(exc)


def check_arch(arch):
    import aiter.fused_moe as fm
    from aiter.moev2 import api
    from aiter.moev2.dispatch import tuned
    from aiter.moev2.problem import MoeProblem
    from moe_dispatch import arch_sim, record, snapshot
    from moe_dispatch.differential import captured_logs

    problems = []
    counts = {"keys": 0, "rows": 0}
    with arch_sim.simulated_arch(arch) as (gfx, cu_num), captured_logs():
        keys = snapshot.dispatch_keys(gfx, cu_num)
        if keys:
            _legacy_record(fm, record, keys[0].args, keys[0].call_kwargs())
        default_tables = fm.cfg_2stages
        fm.cfg_2stages = tuple(_Recording(t) for t in default_tables)
        try:
            for key in keys:
                label, args, kwargs = key.label(), key.args, key.call_kwargs()
                counts["keys"] += 1
                try:
                    problem = api.problem_from_key(args, kwargs)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{label}: no problem: {exc!r}")
                    continue
                again = api.problem_from_key(args, kwargs)
                if again != problem or hash(again) != hash(problem):
                    problems.append(f"{label}: equal keys give unequal problems")
                back_args, back_kwargs = problem.legacy_key()
                back = MoeProblem.from_legacy_key(
                    *back_args, **back_kwargs, arch=gfx, cu_num=cu_num
                )
                if back != problem:
                    problems.append(
                        f"{label}: legacy_key() round trip\n  {problem!r}\n  {back!r}"
                    )
                if _legacy_record(fm, record, back_args, back_kwargs) != _legacy_record(
                    fm, record, args, kwargs
                ):
                    problems.append(
                        f"{label}: legacy dispatches legacy_key() differently"
                    )

                config_file = kwargs.get("config_file")
                if config_file is not None:
                    fm.cfg_2stages_by_file[config_file] = tuple(
                        _Recording(t) for t in fm.cfg_2stages_by_file[config_file]
                    )
                _HITS.clear()
                _legacy_record(fm, record, args, kwargs)
                if not _HITS:
                    continue  # legacy rejected the key before looking it up
                legacy_row = _HITS[-1]
                tables = (
                    fm.cfg_2stages
                    if config_file is None
                    else fm.cfg_2stages_by_file[config_file]
                )
                row = tuned.lookup(problem, tables=tuple(dict(t) for t in tables))
                if row is not legacy_row:
                    problems.append(
                        f"{label}: tuned row differs\n  legacy: {legacy_row}\n  v2:     {row}"
                    )
                # v2's own read of the default table (used without aiter.fused_moe).
                if config_file is None and not _same_row(
                    tuned.lookup(problem, tables=tuned._own_default_table()), legacy_row
                ):
                    problems.append(f"{label}: v2's own table gives a different row")
                counts["rows"] += row is not None
        finally:
            fm.cfg_2stages = default_tables
    return counts, problems


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
    from moe_dispatch import arch_sim

    failed = False
    for arch in args.arch or list(arch_sim.ARCHS):
        counts, problems = check_arch(arch)
        print(
            f"{arch}: {counts['keys']} keys, {counts['rows']} tuned rows, {len(problems)} problems"
        )
        for line in problems[:40]:
            print("  " + line)
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
