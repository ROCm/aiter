#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A/B the gfx1250 dispatch implementations under the MegaMoE E2E harness.

Runs ``test_mega_moe_gfx1250.py`` once per dispatch implementation and prints one
table comparing them. Each run is a fresh torchrun, so the JIT cache, the cco
arena and the CUDA graph of one implementation cannot colour another's numbers.

Implementations (``--modes``):

  ``flydsl``  FlyDSL vector dispatch -- per-lane vec4 payload copies. The default
              the pipeline ships with, and the natural baseline.
  ``tdm``     FlyDSL TDM dispatch -- block-local route histogram, one remote
              atomic per (block, peer), metadata staged into destTokId-ordered
              runs, payload moved by the Tensor Data Mover.
  ``mori``    mori's HIP/JIT dispatch through ``EpDispatchPlan``. On gfx1250 that
              resolves to mori's own C++ TDM path, so this is the FlyDSL port's
              real reference point rather than just "the other backend".

Two numbers per implementation. ``dispatch us`` is the profiler's per-call self
device time for the dispatch kernel alone, which is what actually changed;
``per_layer us`` is the wall clock of the whole dispatch -> gemm -> combine chain,
which says how much of that lands in the model. Both are means over ranks. Every
run is also gated on the harness's fp32 accuracy check unless ``--acc_verify 0``.

The E2E harness only routes MEGA_DISPATCH through ``MegaMoEGfx1250``, which is
a8w4_mxfp4 + scatter_fused only -- so those two are fixed here and are not knobs.

    # default: all three, one token count
    python test_dispatch_tdm_gfx1250.py

    # crossover sweep, vector vs TDM only
    python test_dispatch_tdm_gfx1250.py --modes flydsl,tdm --sweep 512,2048,8192

    # match a perf run: 61 layers, skip the reference check
    python test_dispatch_tdm_gfx1250.py --layers 61 -tpr 512 --acc_verify 0
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_E2E = os.path.join(_HERE, "test_mega_moe_gfx1250.py")

# MEGA_DISPATCH value + any extra env each implementation needs.
_MODES = {
    "flydsl": {"MEGA_DISPATCH": "flydsl"},
    "tdm": {"MEGA_DISPATCH": "tdm"},
    "mori": {"MEGA_DISPATCH": "mori", "MORI_V2_KERNEL_BACKEND": "hip"},
}

_RE_WALL = re.compile(r"per_layer=([\d.]+) us")
_RE_CHECK = re.compile(r"# MEGA-CHECK .*?: (PASS|FAIL) \(avg logits_diff=([\d.]+)")
_RE_DEV_PER_LAYER = re.compile(r"-> ([\d.]+) us/layer \(device-busy")
# Profile-table row: "<name padded to 52><one cell per rank><avg><calls>". Kernel
# names never contain a space, so the trailing two fields are the last two tokens.
_RE_ROW = re.compile(r"^(\S+)\s+(?:[-\d.]+\s+)+([\d.]+)\s+([\d.]+)\s*$")


def _run_one(mode, args, num_tokens):
    """Run the E2E harness for one dispatch implementation; return parsed stats."""
    env = dict(os.environ)
    env.update(_MODES[mode])
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc}",
        _E2E,
        "-q",
        "a8w4_mxfp4",
        "--combine",
        "scatter_fused",
        "-e",
        str(args.expert),
        "-k",
        str(args.topk),
        "-hd",
        str(args.hidden),
        "-id",
        str(args.inter),
        "--layers",
        str(args.layers),
        "-tpr",
        str(num_tokens),
        "--acc_verify",
        str(args.acc_verify),
        "--profile_table",
        "1",
    ]
    print(f"# [{mode}] tokens/rank={num_tokens} ...", flush=True)
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(f"# [{mode}] FAILED (exit {proc.returncode}); last 25 lines:")
        print("\n".join(out.strip().splitlines()[-25:]), flush=True)
        return None

    stat = {"mode": mode, "tokens": num_tokens}
    wall = _RE_WALL.search(out)
    stat["per_layer_us"] = float(wall.group(1)) if wall else float("nan")
    dev = _RE_DEV_PER_LAYER.search(out)
    stat["device_us"] = float(dev.group(1)) if dev else float("nan")
    check = _RE_CHECK.search(out)
    stat["check"] = check.group(1) if check else ("-" if not args.acc_verify else "?")
    stat["diff"] = float(check.group(2)) if check else float("nan")

    # Sum every dispatch kernel row: a schedule that straddles a bucket boundary
    # can compile more than one variant, and they are separate profiler rows.
    disp_us, disp_name = 0.0, None
    for line in out.splitlines():
        row = _RE_ROW.match(line)
        if not row:
            continue
        name, per_call = row.group(1), float(row.group(2))
        low = name.lower()
        if "dispatch" in low and "combine" not in low:
            disp_us += per_call
            disp_name = name if disp_name is None else disp_name
    stat["dispatch_us"] = disp_us if disp_name else float("nan")
    stat["kernel"] = disp_name or "?"
    return stat


def _table(rows, args, baseline):
    """One block per token count, speedups relative to the baseline mode."""
    # routes x hidden x elem_size: what dispatch would move with no dedup, so the
    # real figure is a little lower wherever a token has two experts on one peer.
    lines = []
    for tokens in sorted({r["tokens"] for r in rows}):
        group = [r for r in rows if r["tokens"] == tokens]
        base = next((r for r in group if r["mode"] == baseline), None)
        bytes_moved = tokens * args.topk * args.hidden * 2
        lines.append(
            f"\n# tokens/rank={tokens}  hidden={args.hidden} topk={args.topk} "
            f"world={args.nproc} layers={args.layers}"
        )
        lines.append(
            f"{'impl':<9}{'dispatch us':>13}{'GB/s':>9}{'vs base':>9}"
            f"{'per_layer us':>14}{'vs base':>9}{'device us':>11}"
            f"{'check':>7}  kernel"
        )
        for r in group:
            gbs = bytes_moved / (r["dispatch_us"] * 1e3) if r["dispatch_us"] else 0.0
            d_rel = (
                f"{base['dispatch_us'] / r['dispatch_us']:.2f}x"
                if base and r["dispatch_us"]
                else "-"
            )
            w_rel = (
                f"{base['per_layer_us'] / r['per_layer_us']:.2f}x"
                if base and r["per_layer_us"]
                else "-"
            )
            lines.append(
                f"{r['mode']:<9}{r['dispatch_us']:>13.1f}{gbs:>9.0f}{d_rel:>9}"
                f"{r['per_layer_us']:>14.1f}{w_rel:>9}{r['device_us']:>11.1f}"
                f"{r['check']:>7}  {r['kernel'][:44]}"
            )
    lines.append(
        "\n# dispatch us = profiler per-call self device time of the dispatch "
        "kernel (mean over ranks)."
    )
    lines.append(
        "# GB/s counts every route's payload; same-peer routes are deduped, so "
        "the true rate is slightly lower."
    )
    return "\n".join(lines)


def main():
    args = _parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in _MODES]
    if unknown:
        raise SystemExit(f"unknown --modes {unknown}; pick from {sorted(_MODES)}")
    sweep = [int(t) for t in args.sweep.split(",")] if args.sweep else [args.tokens]
    baseline = args.baseline if args.baseline in modes else modes[0]

    rows, failed = [], []
    for tokens in sweep:
        for mode in modes:
            stat = _run_one(mode, args, tokens)
            if stat is None:
                failed.append((mode, tokens))
            else:
                rows.append(stat)

    if rows:
        print(_table(rows, args, baseline), flush=True)
    bad_check = [r for r in rows if r["check"] == "FAIL"]
    for mode, tokens in failed:
        print(f"# ERROR {mode} @ {tokens} tokens/rank did not complete")
    for r in bad_check:
        print(f"# ERROR {r['mode']} @ {r['tokens']} tokens/rank failed the fp32 check")
    return 1 if (failed or bad_check) else 0


def _parse_args():
    p = argparse.ArgumentParser(
        description="compare gfx1250 dispatch implementations end to end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--modes",
        type=str,
        default="flydsl,tdm,mori",
        help="comma-separated dispatch implementations: flydsl | tdm | mori",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default="flydsl",
        help="implementation the speedup columns divide by",
    )
    p.add_argument(
        "--sweep",
        type=str,
        default="",
        help="comma-separated tokens/rank to sweep (overrides -tpr)",
    )
    p.add_argument("-tpr", "--tokens", type=int, default=512, help="tokens per rank")
    p.add_argument("-hd", "--hidden", type=int, default=7168, help="model/hidden dim")
    p.add_argument("-id", "--inter", type=int, default=3072, help="intermediate dim")
    p.add_argument("-e", "--expert", type=int, default=384, help="routed experts")
    p.add_argument("-k", "--topk", type=int, default=6, help="top-k")
    p.add_argument("--layers", type=int, default=8, help="MoE layers to chain")
    p.add_argument("--nproc", type=int, default=4, help="ranks / GPUs")
    p.add_argument(
        "--acc_verify",
        type=int,
        default=1,
        help="gate every run on the harness fp32 reference check",
    )
    p.add_argument(
        "--timeout", type=int, default=3600, help="per-run timeout in seconds"
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
