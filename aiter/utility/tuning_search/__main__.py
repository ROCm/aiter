# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Command line for the search engine.

::

    python -m aiter.utility.tuning_search strategies
    python -m aiter.utility.tuning_search demo --strategy portfolio --budget 64
    python -m aiter.utility.tuning_search replay --csv sweep.csv.gz \\
        --config-cols tile_m,tile_n,tile_k --value-col tflops --higher-is-better \\
        --problem-cols M,N,K --strategies de,tpe,hill,portfolio,random \\
        --budgets 32,64 --seeds 5 --out replay.json
    python -m aiter.utility.tuning_search tune --factory mypkg.tuning:make \\
        --budget 64 --db tuned.json --trace trace.jsonl --checkpoint ck.json
    python -m aiter.utility.tuning_search show --db tuned.json
    python -m aiter.utility.tuning_search show --trace trace.jsonl

``tune --factory module:function`` calls ``function(args)`` and expects a
dict with ``space`` and ``backend`` (and optionally ``prior``, ``warm``,
``ladder``, ``baseline``, ``record``: a function of the search result
returning the keyword arguments of :func:`.api.record_result` other than
the result, ``db``, ``seed`` and ``trace``; it is called only with
``--db``). ``--arg key=value`` pairs
reach the factory as the dict ``args.factory_args``. This is how a family
plugs its compile/launch/check code into the CLI without the engine
importing torch or any family.

``replay`` prints, per strategy and budget, the mean final fraction of
oracle (the config each search returned) and, separately, ``ideal_mean``
(the best config it measured: an oracle-informed upper bound).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Sequence

from .api import STRATEGY_NAMES, record_result, tune
from .db import TuningDB, read_trace
from .measure import SyntheticBackend
from .replay import ReplayProblem, load_table, replay_problem, replay_summary
from .synthetic import brute_force_optimum, gemm_like, sphere
from .trial import Budget


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _cmd_strategies(_args) -> int:
    for s in STRATEGY_NAMES:
        print(s)
    return 0


def _cmd_demo(args) -> int:
    if args.objective == "sphere":
        space, objective = sphere()
    else:
        space, objective = gemm_like(rugged=args.rugged)
    opt_key, opt, valid = brute_force_optimum(space, objective)
    rows = []
    for seed in range(args.seeds):
        backend = SyntheticBackend(objective, noise=args.noise, seed=seed)
        res = tune(
            space,
            backend,
            strategy=args.strategy,
            budget=Budget(evaluations=args.budget),
            seed=seed,
            confirm=args.confirm,
            batch=args.batch,
        )
        truth = objective(res.best_config) if res.best_key else math.inf
        rows.append(
            {
                "seed": seed,
                "fraction_of_optimum": opt / truth,
                "found_optimum": res.best_key == opt_key,
                "evaluations": res.evaluations,
                "stopped": res.stopped,
            }
        )
    out = {
        "space": space.name,
        "valid_configs": valid,
        "strategy": args.strategy,
        "budget": args.budget,
        "runs": rows,
        "mean_fraction": sum(r["fraction_of_optimum"] for r in rows) / len(rows),
    }
    print(json.dumps(out, indent=1))
    return 0


def _cmd_replay(args) -> int:
    where = dict(kv.split("=", 1) for kv in args.where) if args.where else None
    data = load_table(
        args.csv,
        config_cols=args.config_cols.split(","),
        value_col=args.value_col,
        problem_cols=args.problem_cols.split(","),
        higher_is_better=args.higher_is_better,
        where=where,
        status_col=args.status_col or None,
    )
    names = sorted(data)
    if args.problems:
        wanted = set(args.problems.split(";"))
        names = [n for n in names if n in wanted]
    names = [n for n in names if len(data[n][0]) >= args.min_configs][
        : args.max_problems
    ]
    runs = []
    for name in names:
        configs, vals = data[name][0], data[name][1]
        statuses = data[name][2] if args.status_col else None
        base = ReplayProblem(
            name,
            configs,
            vals,
            statuses=statuses,
            noise=args.noise,
            call_noise=args.call_noise,
            seed=0,
        )
        for seed in range(args.seeds):
            prob = base.with_seed(seed)
            for strat in args.strategies.split(","):
                for b in _ints(args.budgets):
                    runs.append(replay_problem(prob, strat, budget=b, seed=seed))
    summary = {}
    for b in _ints(args.budgets):
        summary[str(b)] = replay_summary([r for r in runs if r["budget"] == b])
    out = {"problems": names, "summary": summary}
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"runs": runs, **out}, f)
    print(json.dumps(out, indent=1))
    return 0


def _cmd_tune(args) -> int:
    args.factory_args = dict(kv.split("=", 1) for kv in args.arg)
    mod, _, fn = args.factory.partition(":")
    made = getattr(importlib.import_module(mod), fn or "make")(args)
    space, backend = made["space"], made["backend"]
    res = tune(
        space,
        backend,
        strategy=args.strategy,
        budget=Budget(evaluations=args.budget, seconds=args.seconds),
        seed=args.seed,
        ladder=made.get("ladder"),
        prior=made.get("prior"),
        warm=made.get("warm", ()),
        batch=args.batch,
        trace=args.trace,
        checkpoint=args.checkpoint,
        resume=args.resume,
        confirm=args.confirm,
        cache_path=args.cache,
        baseline=made.get("baseline"),
    )
    print(json.dumps(res.summary(), indent=1, default=str))
    if args.db and made.get("record"):
        db = TuningDB(args.db)
        rec = record_result(
            res, db=db, seed=args.seed, trace=args.trace or "", **made["record"](res)
        )
        print(json.dumps(None if rec is None else rec.to_json(), indent=1))
    return 0


def _cmd_show(args) -> int:
    if args.db:
        for r in TuningDB(args.db).records():
            print(json.dumps(r.to_json(), sort_keys=True))
    if args.trace:
        rows = read_trace(args.trace)
        obs = [r for r in rows if r.get("type") == "obs"]
        statuses: dict[str, int] = {}
        for r in obs:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        best = min(
            (r for r in obs if r.get("value") is not None and r.get("full")),
            key=lambda r: r["value"],
            default=None,
        )
        end = next((r for r in rows if r.get("type") == "end"), None)
        print(
            json.dumps(
                {
                    "observations": len(obs),
                    "statuses": statuses,
                    "best": best
                    and {
                        "key": best["key"],
                        "value": best["value"],
                        "config": best["config"],
                    },
                    "end": end,
                },
                indent=1,
            )
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m aiter.utility.tuning_search")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("strategies", help="list strategy names")

    d = sub.add_parser("demo", help="run a strategy on a synthetic problem")
    d.add_argument("--objective", choices=("gemm", "sphere"), default="gemm")
    d.add_argument("--strategy", default="portfolio", choices=(*STRATEGY_NAMES, "auto"))
    d.add_argument("--budget", type=int, default=64)
    d.add_argument("--seeds", type=int, default=3)
    d.add_argument("--noise", type=float, default=0.01)
    d.add_argument("--rugged", type=float, default=0.0)
    d.add_argument("--confirm", type=int, default=3)
    d.add_argument("--batch", type=int, default=1)

    r = sub.add_parser("replay", help="score strategies on a recorded sweep CSV")
    r.add_argument("--csv", required=True)
    r.add_argument("--config-cols", required=True)
    r.add_argument("--value-col", required=True)
    r.add_argument("--problem-cols", required=True)
    r.add_argument("--higher-is-better", action="store_true")
    r.add_argument("--where", nargs="*", default=(), help="col=value row filters")
    r.add_argument("--problems", default="", help="';'-separated problem names")
    r.add_argument("--max-problems", type=int, default=20)
    r.add_argument("--min-configs", type=int, default=16)
    r.add_argument("--strategies", default="portfolio,de,tpe,hill,random")
    r.add_argument("--budgets", default="16,32,64")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--noise", type=float, default=0.0, help="log-sigma per rep")
    r.add_argument(
        "--call-noise", type=float, default=0.0, help="log-sigma per measurement"
    )
    r.add_argument(
        "--status-col",
        default="",
        help="status column: failures cost an evaluation, early_stop is a bound",
    )
    r.add_argument("--out", default="")

    t = sub.add_parser("tune", help="tune with a factory-provided space/backend")
    t.add_argument("--factory", required=True, help="module:function")
    t.add_argument("--strategy", default="portfolio", choices=(*STRATEGY_NAMES, "auto"))
    t.add_argument("--budget", type=int, default=64)
    t.add_argument("--seconds", type=float, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--batch", type=int, default=1)
    t.add_argument("--confirm", type=int, default=3)
    t.add_argument("--trace", default=None)
    t.add_argument("--checkpoint", default=None)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--cache", default=None)
    t.add_argument("--db", default=None)
    t.add_argument(
        "--arg",
        nargs="*",
        default=(),
        help="key=value pairs for the factory (args.factory_args)",
    )

    s = sub.add_parser("show", help="print a tuning DB or summarize a trace")
    s.add_argument("--db", default=None)
    s.add_argument("--trace", default=None)

    args = p.parse_args(argv)
    return {
        "strategies": _cmd_strategies,
        "demo": _cmd_demo,
        "replay": _cmd_replay,
        "tune": _cmd_tune,
        "show": _cmd_show,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
