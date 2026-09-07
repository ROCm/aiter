# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A/B the GDR decode host tiling rule against the tuned table.

``_decode_tiling`` picks the value split and warp shape from the launch shape,
and the tuned table overrides it wherever it has a row. This measures what the
table is still worth: one arm leaves it alone, the other drops this part's
decode rows so the rule decides, and both arms run interleaved in one process
against the same inputs.

Interleaving is the point. The differences here are under a percent on kernels
that take tens of microseconds, and two runs of the standard sweep in separate
processes disagree by ~10% at batch 1 -- more than anything being measured. Any
comparison across processes is noise. So is any comparison that does not report
the cells where both arms chose the same config: those measure nothing but the
harness, and their spread is the floor below which no result here means
anything.

Usage:
    python op_tests/bench_flydsl_gdr_decode_tiling.py --repeats 5 --check

There is nothing to compare on a part whose decode rows have already been
dropped, so reproducing that part's numbers means pointing ``--table`` at a
revision that still has them:

    git show <rev>:aiter/configs/gdr_decode_tuned.csv > /tmp/before.csv
    python op_tests/bench_flydsl_gdr_decode_tiling.py --table /tmp/before.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import types

import torch

import op_tests.test_flydsl_linear_attention as D
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import linear_attention_kernels as LA
from aiter.test_common import checkAllclose, run_perftest

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)

_STATE_DTYPE = torch.float32
_ORIG_CREATE_INPUTS = D.create_inputs


def _create_inputs(args):
    """``create_inputs`` with the state dtype left open.

    The stock benchmark builds the state as f32 unconditionally, which reaches
    only half the table; the bf16-state rows describe launches it never makes.
    """
    out = list(_ORIG_CREATE_INPUTS(args))
    out[-1] = out[-1].to(_STATE_DTYPE)
    return tuple(out)


D.create_inputs = _create_inputs


def _table(path):
    """The tuned table as the lookup builds it, plus this part's decode shapes.

    Built through ``_tuned_config`` rather than parsed here, so the arm that
    keeps the table is reading it exactly the way the launch would.
    """
    saved = LA.AITER_CONFIGS
    LA.AITER_CONFIGS = types.SimpleNamespace(AITER_CONFIG_GDR_DECODE_FILE=path)
    try:
        LA.GDR_GLOBAL_CONFIG_MAP = None
        LA._tuned_config("torch.bfloat16", "torch.float32", 1, 1, 2, 8, 128, 128)
        full = dict(LA.GDR_GLOBAL_CONFIG_MAP)
    finally:
        LA.AITER_CONFIGS = saved
    arch = get_gfx()
    without = {k: v for k, v in full.items() if not (k[2] == arch and k[3] == "decode")}
    shapes = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("variant") or "decode") != "decode" or r["arch"] != arch:
                continue
            shapes.setdefault(
                (
                    r["dtype"],
                    r["state_dtype"],
                    int(r["num_k_heads"]),
                    int(r["num_v_heads"]),
                    int(r["head_k_dim"]),
                    int(r["head_v_dim"]),
                ),
                None,
            )
    return full, without, list(shapes)


def _run(args, tensors, table, reference=None):
    """One arm: install a table, take the config it yields, time the launch."""
    query, key, value, a, beta, dt_bias, A_log, indices, state, out = tensors
    LA.GDR_GLOBAL_CONFIG_MAP = table
    config = LA.get_default_kwargs(
        str(args.dtype),
        str(state.dtype),
        args.b,
        args.sq,
        args.num_k_heads,
        args.num_v_heads,
        args.head_k_dim,
        args.head_v_dim,
    )

    def call(candidate_state, candidate_out):
        return D.func(
            args,
            query,
            key,
            value,
            a,
            beta,
            dt_bias,
            A_log,
            indices,
            candidate_state,
            candidate_out,
        )

    perf_state = state.clone()
    perf_out = torch.zeros_like(out)
    _, us = run_perftest(
        call,
        perf_state,
        perf_out,
        num_rotate_args=D._perf_rotation_count(perf_state, perf_out),
    )

    err = None
    if reference is not None:
        cand_state, cand_out = state.clone(), torch.zeros_like(out)
        call(cand_state, cand_out)
        ref_state, ref_out = reference
        err = max(
            checkAllclose(
                ref_out.to(dtypes.fp32),
                cand_out.to(dtypes.fp32),
                rtol=1e-3,
                atol=1e-3,
                msg="out ",
            ),
            checkAllclose(
                ref_state.to(dtypes.fp32),
                cand_state.to(dtypes.fp32),
                rtol=1e-3,
                atol=1e-3,
                msg="state ",
            ),
        )
    return float(us), config, err


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, nargs="*", default=list(BATCHES))
    parser.add_argument("--check", action="store_true", help="also verify numerics")
    parser.add_argument(
        "--table",
        default=AITER_CONFIGS.AITER_CONFIG_GDR_DECODE_FILE,
        help="tuned csv to use as the table arm (default: the installed one)",
    )
    parser.add_argument("--out", default="")
    opt = parser.parse_args()

    arch = get_gfx()
    full, without, shapes = _table(opt.table)
    if not shapes:
        raise SystemExit(
            f"{opt.table} has no {arch} decode rows, so there is no table to "
            f"compare against. Point --table at a revision that still has them."
        )
    print(
        f"{arch}: {opt.table}\n{len(full)} rows, {len(full) - len(without)} of "
        f"them {arch} decode, over {len(shapes)} shapes"
    )

    global _STATE_DTYPE
    result = {}
    for dtype_str, state_str, nkh, nvh, khd, vhd in shapes:
        _STATE_DTYPE = torch.float32 if state_str == "torch.float32" else torch.bfloat16
        name = f"{state_str.split('.')[-1]}|nkh{nkh}|nvh{nvh}"
        for b in opt.batch:
            args = D.Args(
                dtype=(
                    torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16
                ),
                b=b,
                sq=1,
                num_k_heads=nkh,
                num_v_heads=nvh,
                head_k_dim=khd,
                head_v_dim=vhd,
                use_qk_l2norm=True,
            )
            built = _create_inputs(args)
            tensors = built[1:] + (D.create_outputs(args)[0],)
            reference = None
            if opt.check:
                ref_state = tensors[-2].clone()
                ref_out = torch.zeros_like(tensors[-1])
                D.ref_func(args, *tensors[:-2], ref_state, ref_out)
                reference = (ref_state, ref_out)

            table_us, rule_us, errs = float("inf"), float("inf"), []
            table_cfg = rule_cfg = None
            for _ in range(opt.repeats):
                # Alternate so any drift lands on both arms alike.
                us, table_cfg, err = _run(args, tensors, full, reference)
                table_us = min(table_us, us)
                errs += [err] if err is not None else []
                us, rule_cfg, err = _run(args, tensors, without, reference)
                rule_us = min(rule_us, us)
                errs += [err] if err is not None else []

            ratio = table_us / rule_us
            result[f"{name}|b{b}"] = {
                "table_us": table_us,
                "rule_us": rule_us,
                "ratio": ratio,
                "table_cfg": table_cfg,
                "rule_cfg": rule_cfg,
                "same_cfg": table_cfg == rule_cfg,
                "err": max(errs) if errs else None,
            }
            print(
                f"  {name:<22} b{b:<4} table {table_us:9.2f} "
                f"{tuple(table_cfg.values())!s:<12} rule {rule_us:9.2f} "
                f"{tuple(rule_cfg.values())!s:<12} {ratio:6.3f}x"
                + (f"  err {max(errs):.1e}" if errs else "")
                + ("  [same cfg]" if table_cfg == rule_cfg else ""),
                flush=True,
            )

    same = [r["ratio"] for r in result.values() if r["same_cfg"]]
    diff = [r["ratio"] for r in result.values() if not r["same_cfg"]]
    print(
        f"\nall {len(result)} cells: geomean "
        f"{statistics.geometric_mean([r['ratio'] for r in result.values()]):.4f}x"
    )
    if same:
        print(
            f"  {len(same):3d} cells where both arms chose the same config: "
            f"{min(same):.4f} .. {max(same):.4f}   <- harness noise floor"
        )
    if diff:
        print(
            f"  {len(diff):3d} cells where the configs differ:   geomean "
            f"{statistics.geometric_mean(diff):.4f}x  worst {min(diff):.4f}x  "
            f"best {max(diff):.4f}x"
        )
        behind = sorted(
            (
                (r["ratio"], k, r)
                for k, r in result.items()
                if not r["same_cfg"] and r["ratio"] < 1.0
            )
        )
        print("\nthe rule is behind the table on these, worst first:")
        for ratio, k, r in behind[:12]:
            print(
                f"  {k:<30} {ratio:.3f}x  table "
                f"{tuple(r['table_cfg'].values())} vs rule "
                f"{tuple(r['rule_cfg'].values())}"
            )
        if not behind:
            print("  (none)")
    if opt.out:
        with open(opt.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
