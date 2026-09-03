#!/usr/bin/env python3
"""Everything about preparing and accounting for one target run.

WHY THIS IS A TOOL AND THE LAUNCH IS NOT

The entry point still spawns the target itself. It has to: the run needs the assembled
passthrough environment, the private cache roots, and the GPU whose flock is held by the entry
point's own file descriptor, and a child process cannot hold that lock (see SKILL.md). What it
does NOT need to hold is the reasoning around the launch -- what gets written into the probe
module, how a grid cell becomes a Python value, and what a script target's exit code is allowed
to be counted as. Those are decisions, they were spelled out inside bash heredocs where nothing
could reach them, and they are here instead.

The rule these subcommands share with report.py: the model may judge, but may not keep the
books. `script-stats` in particular refuses to publish a case count it did not observe.
"""

import argparse
import json
import os
import pathlib
import re
import sys

SENTINEL = "__VALIDATOR_INVALID_GRID__"

# The target is arbitrary code from an unmerged pull request, and it used to run with the
# reviewer's whole environment attached: `env VAR=... <cmd>` ADDS to the inherited environment,
# it does not replace it. Any GITHUB_TOKEN, GH_TOKEN, API key, SSH agent socket or provider
# credential in the reviewer's shell was readable from `os.environ` inside the code under
# review, and would land in a log the moment a target printed its environment.
#
# So the environment is CONSTRUCTED. Everything a ROCm/PyTorch run legitimately needs is passed
# by name or prefix; everything else is dropped; and anything that looks like a secret is
# dropped even if a prefix would have kept it, because the allowlist is about function and the
# denylist is about consequence.
ALLOW_PREFIXES = (
    "PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "TMPDIR",
    "TZ",
    "LANG",
    "LC_",
    "TERM",
    "USER",
    "LOGNAME",
    "HOSTNAME",
    "ROCM",
    "HIP",
    "HSA",
    "HCC",
    "AMD",
    "GPU_",
    "ROCR",
    "RCCL",
    "NCCL",
    "OMP_",
    "MKL_",
    "OPENBLAS",
    "NUMEXPR",
    "CUDA",
    "TORCH",
    "PYTORCH",
    "TRITON",
    "FLYDSL",
    "AITER",
    "VIRTUAL_ENV",
    "CC",
    "CXX",
    "CMAKE",
    "MAX_JOBS",
)
DENY_RE = re.compile(
    "TOKEN|SECRET|PASSWD|PASSWORD|CREDENTIAL|_KEY|APIKEY|API_KEY|COOKIE|SESSION"
    "|AUTH|PRIVATE|GH_|GITHUB|SSH_|GPG_|NETRC"
)


def passthrough(environ) -> list[str]:
    """The NAME=VALUE pairs the target is allowed to see."""
    return [
        f"{name}={value}"
        for name, value in environ.items()
        if name.startswith(ALLOW_PREFIXES) and not DENY_RE.search(name.upper())
    ]


def environment_summary(pairs) -> dict:
    return {
        "policy": "constructed (env -i + name/prefix allowlist + secret-shaped denylist)",
        "passed_through": sorted(pair.split("=", 1)[0] for pair in pairs),
        "note": (
            "the target is unmerged third-party code; it runs with a built environment rather "
            "than the reviewer's, so a credential in the calling shell is not readable from it "
            "and cannot reach a log"
        ),
    }


def coerce(cell: str):
    """A grid cell as the Python value the target's signature expects.

    Cells arrive as text. Anything that is an integer is passed as one, because a test that
    indexes or allocates with a shape argument needs an int, not "128". Everything else is left
    as the string the caller wrote, which is what a dtype argument wants.
    """
    if cell in ("True", "False"):
        # Before this, "False" reached the target as a non-empty string, which is truthy, and
        # every row of a boolean dimension silently ran the True branch.
        return cell == "True"
    if cell in ("None", "none"):
        return None
    try:
        return int(cell)
    except ValueError:
        pass
    try:
        return float(cell)
    except ValueError:
        return cell


def grid_rows(grid: str, names) -> list[tuple]:
    if grid.strip() == SENTINEL:
        # The invalid-grid probe must reach the TARGET, not crash the plugin. A row of the
        # wrong arity would raise inside pytest's parametrize and the non-zero exit would
        # credit the channel without the target ever having consumed a shape. Keep the arity,
        # poison the values: the target then fails on its own, which is the evidence the probe
        # is for.
        rows = [tuple(SENTINEL for _ in names)]
    else:
        rows = [
            tuple(cell.strip() for cell in row.split(","))
            for row in grid.split(";")
            if row.strip()
        ]
    bad = [row for row in rows if len(row) != len(names)]
    if bad:
        raise SystemExit(
            f"grid rows must have {len(names)} cells to match --shape-argnames "
            f"{','.join(names)}; offending rows: {bad}"
        )
    return [tuple(coerce(cell) for cell in row) for row in rows]


def script_stats(exit_code: int, receipt_path: str, expected_route: str) -> dict:
    """What a script target's run may be counted as.

    A script target publishes no per-case count, so `executed` used to be hard-coded to 1 and
    stood for "the process ran". That is the number a silently-returning target also produces
    -- aiter#4538's own target returns with exit 0 and log output when the arch is unsupported
    or an optional package is missing -- so a run that graded 56 cases and a run that graded
    none were indistinguishable, and both credited runtime architecture coverage.

    `executed` keeps its old meaning for everything that consumes it as a liveness signal (the
    no-GPU requirement probe, the pass/skip decision), because a script that ran is a script
    that ran. What changes is that it no longer PRETENDS to be a case count, and that
    `observed_work` -- the only number here backed by evidence -- is published beside it.
    """
    observed = None
    basis = (
        "script process exit; no route was named, so no executed work could be counted"
    )
    if expected_route:
        try:
            with open(receipt_path) as handle:
                receipt = json.load(handle)
        except (OSError, ValueError):
            receipt = None
        if receipt is None:
            observed = 0
            basis = (
                "script process exit; a route was named and this run wrote no execution "
                "receipt, so no executed work was observed"
            )
        else:
            symbols = len(receipt.get("kernel_symbols") or [])
            shapes = len(receipt.get("executed_shapes") or [])
            observed = max(symbols, shapes)
            basis = (
                "observed route calls in this run's own execution receipt "
                f"({symbols} symbol(s), {shapes} shape record(s))"
            )
    return {
        "tests": 1,
        "failures": int(exit_code != 0),
        "errors": 0,
        "skipped": 0,
        "executed": 1,
        "observed_work": observed,
        "basis": basis,
    }


def preferred_receipt(grid: str, repo: str) -> str:
    """Which of the head phase's two receipts speaks for the run.

    The grid receipt is preferred when it proves the route, because it is the run that
    exercised the injected shapes; otherwise the repository run's receipt stands. A phase that
    observed nothing never speaks over one that observed something.
    """
    try:
        with open(grid) as handle:
            if json.load(handle).get("route"):
                return grid
    except (OSError, ValueError):
        pass
    return repo if os.path.exists(repo) else grid


def _append(source: str, output: str, trailer: str) -> None:
    pathlib.Path(output).write_text(pathlib.Path(source).read_text() + trailer)


def cmd_probe_module(args) -> int:
    _append(
        args.source,
        args.out,
        f"\n_VALIDATION_EXPECTED_ROUTE = {args.route!r}\n"
        f"_VALIDATION_SHAPE_VARS = {args.shape_vars!r}\n"
        f"_VALIDATION_RECEIPT_PATH = {args.receipt!r}\n",
    )
    return 0


def cmd_shape_plugin(args) -> int:
    names = tuple(part.strip() for part in args.argnames.split(",") if part.strip())
    rows = grid_rows(args.grid, names)
    _append(
        args.source,
        args.out,
        f"\n_VALIDATION_SHAPE_ARGNAMES = {names!r}\n"
        f"_VALIDATION_SHAPE_GRID = {rows!r}\n",
    )
    return 0


def cmd_script_stats(args) -> int:
    print(json.dumps(script_stats(args.exit_code, args.receipt, args.route)))
    return 0


def cmd_pick_receipt(args) -> int:
    print(preferred_receipt(args.grid, args.repo))
    return 0


def cmd_env(args) -> int:
    sys.stdout.write("\0".join(passthrough(os.environ)))
    return 0


def cmd_env_summary(args) -> int:
    print(json.dumps(environment_summary(args.pairs)))
    return 0


def cmd_stats_field(args) -> int:
    print(json.loads(args.stats)[args.field])
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe-module", help="write the instrumented probe module")
    probe.add_argument("source")
    probe.add_argument("out")
    probe.add_argument("route")
    probe.add_argument("shape_vars")
    probe.add_argument("receipt")
    probe.set_defaults(func=cmd_probe_module)

    plugin = sub.add_parser("shape-plugin", help="write the parametrize grid plugin")
    plugin.add_argument("source")
    plugin.add_argument("out")
    plugin.add_argument("argnames")
    plugin.add_argument("grid")
    plugin.set_defaults(func=cmd_shape_plugin)

    stats = sub.add_parser("script-stats", help="stats for a script target's run")
    stats.add_argument("exit_code", type=int)
    stats.add_argument("receipt")
    stats.add_argument("route")
    stats.set_defaults(func=cmd_script_stats)

    pick = sub.add_parser("pick-receipt", help="which head receipt speaks for the run")
    pick.add_argument("grid")
    pick.add_argument("repo")
    pick.set_defaults(func=cmd_pick_receipt)

    env = sub.add_parser("env", help="NUL-separated passthrough NAME=VALUE pairs")
    env.set_defaults(func=cmd_env)

    summary = sub.add_parser("env-summary", help="the isolation record for the report")
    summary.add_argument("pairs", nargs="*")
    summary.set_defaults(func=cmd_env_summary)

    field = sub.add_parser("stats-field", help="read one field out of a stats blob")
    field.add_argument("stats")
    field.add_argument("field")
    field.set_defaults(func=cmd_stats_field)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
