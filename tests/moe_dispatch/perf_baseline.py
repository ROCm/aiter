# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end MoE performance baseline for a fixed shape set.

    python tests/moe_dispatch/perf_baseline.py --record           # golden/perf_<gfx>.json
    python tests/moe_dispatch/perf_baseline.py --check
    python tests/moe_dispatch/perf_baseline.py --record --case dsv3_tp8_pt  # add one case

An arch has a baseline only once a case in `CASES` lists it; without one, both modes
refuse to run rather than record or check an empty set.

Each case runs `op_tests/test_moe_2stage.py` in a subprocess (one per quant and shape,
so an OOM or a crash only loses that case) and reads its result table, so inputs, weight
layouts and timing are exactly those of the existing accuracy suite. Run on an idle GPU:
the numbers are compared across runs, and contention makes them meaningless.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
GOLDEN_DIR = _HERE / "golden"

DECODE_M = (1, 16, 64, 256)
PREFILL_M = (1024, 4096, 16384)

# (name, quant index in test_moe_2stage.py l_quant, activation, (model_dim, inter_dim),
#  (hidden_pad, intermediate_pad), experts, topk, archs). Dims are the padded sizes vLLM
# allocates: GPT-OSS 2880 is rounded up to 3072 on ROCm MXFP4. Qwen3-235B FP8 block
# scale runs at TP4, since its TP8 inter_dim (192) is not a multiple of 128.
CASES = (
    ("dsv3_tp8_pt", 1, "silu", (7168, 256), (0, 0), 256, 8, ("gfx942", "gfx950")),
    ("dsv3_tp8_blockscale", 5, "silu", (7168, 256), (0, 0), 256, 8, ("gfx942", "gfx950")),
    ("dsv3_tp8_a4w4", 4, "silu", (7168, 256), (0, 0), 256, 8, ("gfx950",)),
    ("dsv4_a8w4", 7, "silu", (7168, 384), (0, 0), 384, 6, ("gfx950",)),
    ("dsv4_a4w4", 4, "silu", (7168, 384), (0, 0), 384, 6, ("gfx950",)),
    ("gptoss_tp1_a16w4", 6, "swiglu", (3072, 3072), (192, 192), 128, 4, ("gfx950",)),
    ("gptoss_tp1_a8w4", 7, "swiglu", (3072, 3072), (192, 192), 128, 4, ("gfx950",)),
    ("qwen3_235b_tp8_ptoken", 2, "silu", (4096, 192), (0, 0), 128, 8, ("gfx942", "gfx950")),
    ("qwen3_235b_tp4_blockscale", 5, "silu", (4096, 384), (0, 0), 128, 8, ("gfx942", "gfx950")),
    ("qwen3_235b_tp8_bf16", 0, "silu", (4096, 192), (0, 0), 128, 8, ("gfx942", "gfx950")),
)  # fmt: skip

_RUNNER = r"""
import importlib.util, json, runpy, sys
import pandas as pd
if importlib.util.find_spec("tabulate") is None:  # the script's summary needs it
    pd.DataFrame.to_markdown = lambda self, *a, **k: self.to_string()
sys.argv = ["test_moe_2stage.py"] + json.loads(sys.argv[1])
g = runpy.run_path("op_tests/test_moe_2stage.py", run_name="__main__")
rows = []
for r in g["df"].to_dict("records"):
    rows.append({"token": int(r["token"]), "us": float(r["us"])})
print("@@RESULT@@" + json.dumps(rows))
"""


def _gfx():
    # The baseline names the hardware the numbers were measured on, so this is
    # get_gfx_runtime(): get_gfx() follows GPU_ARCHS and returns its last entry on a
    # multi-target build, which would read or write another GPU's baseline.
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime()


def run_case(case, tokens):
    _, quant, act, (model_dim, inter_dim), (hidden_pad, inter_pad), experts, topk, _ = (
        case
    )
    argv = [
        "-q", str(quant),
        "-a", act,
        "-dim", f"{model_dim},{inter_dim}",
        "-e", str(experts),
        "-k", str(topk),
        "-t", *map(str, tokens),
        "-p", "t",
        "-hip", f"{hidden_pad},{inter_pad}",
        "--stage2-bias", "f",
        "--no-flydsl-csv",
    ]  # fmt: skip
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, json.dumps(argv)],
        cwd=_REPO,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            rows = json.loads(line[len("@@RESULT@@") :])
            per_m = {str(r["token"]): r["us"] for r in rows}
            if len(per_m) != len(rows):
                raise RuntimeError(f"case {case[0]}: several results per token: {rows}")
            requested = {str(token) for token in tokens}
            if set(per_m) != requested:
                missing = sorted(requested - set(per_m), key=int)
                unexpected = sorted(set(per_m) - requested, key=int)
                raise RuntimeError(
                    f"case {case[0]}: incomplete token results; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            return per_m
    raise RuntimeError(f"case {case[0]} failed:\n{proc.stderr[-3000:]}")


def measure(gfx, names=None):
    results = {}
    for case in CASES:
        if gfx not in case[-1] or (names and case[0] not in names):
            continue
        try:
            results[case[0]] = run_case(case, DECODE_M + PREFILL_M)
        except RuntimeError as exc:
            results[case[0]] = {"error": str(exc)[-500:]}
        print(f"{case[0]}: {results[case[0]]}", flush=True)
    return results


def _cases_problem(gfx, names=None):
    """Why `--record` or `--check` cannot measure on `gfx`: no case runs there, or a
    requested case does not; None when it can."""
    archs = {case[0]: case[-1] for case in CASES}
    if not any(gfx in case_archs for case_archs in archs.values()):
        return (
            f"no perf case runs on {gfx}; add {gfx} to the archs of the cases that"
            " run there (CASES in this file), then run --record"
        )
    other_arch = [name for name in names or () if gfx not in archs[name]]
    if other_arch:
        return f"cases {other_arch} do not run on {gfx}"
    return None


def _check_baseline(path, gfx, names=None):
    """The recorded timings of the cases a check measures, and the reasons it cannot
    compare them: no baseline file, or a case with no recorded timings."""
    wanted = names or [case[0] for case in CASES if gfx in case[-1]]
    if not path.exists():
        return {}, [f"no baseline {path}; record one first with --record"]
    recorded = json.loads(path.read_text())["us"]
    unrecorded = [name for name in wanted if name not in recorded]
    if unrecorded:
        cases = " ".join(f"--case {name}" for name in unrecorded)
        message = f"no baseline in {path} for {unrecorded}"
        return {}, [f"{message}; record them first with --record {cases}"]
    return {name: recorded[name] for name in wanted}, []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="allowed slowdown; unchanged code drifts up to ~8%% at small M on gfx950",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case[0] for case in CASES],
        help="only these cases; --record keeps the other recorded cases",
    )
    args = parser.parse_args(argv)

    # The repository root so `aiter` is this checkout rather than an installed copy,
    # even one earlier on the path.
    sys.path[:] = [str(_REPO)] + [p for p in sys.path if p != str(_REPO)]
    gfx = _gfx()
    path = GOLDEN_DIR / f"perf_{gfx}.json"
    problem = _cases_problem(gfx, args.case)
    if problem:
        print(problem)
        return 1
    if args.check:
        baseline, problems = _check_baseline(path, gfx, args.case)
        if problems:
            for line in problems:
                print(line)
            return 1
    results = measure(gfx, args.case)
    errors = [name for name, per_m in results.items() if "error" in per_m]
    if args.record:
        if errors:
            print(f"not writing {path}: cases failed: {errors}")
            return 1
        if args.case and path.exists():
            results = {**json.loads(path.read_text())["us"], **results}
        path.write_text(
            json.dumps({"gfx": gfx, "us": results}, indent=1, sort_keys=True) + "\n"
        )
        print(f"wrote {path}")
        return 0
    problems = []
    for name, per_m in baseline.items():
        for m, us in per_m.items():
            now = results.get(name, {}).get(m)
            if not isinstance(now, float):
                problems.append(f"MISSING {name} M={m}: {results.get(name)}")
            elif now > us * (1 + args.tolerance):
                problems.append(f"SLOWER {name} M={m}: {us:.1f} -> {now:.1f} us")
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
