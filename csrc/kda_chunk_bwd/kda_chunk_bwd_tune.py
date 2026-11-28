# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune the FlyDSL KDA chunk backward per shape.

For every row of an untuned CSV (``gfx,cu_num,N,T,H,varlen``: N sequences of T
tokens, packed with ``cu_seqlens`` when ``varlen``), times
``chunk_kda_bwd_flydsl_full`` over the tunables and writes the fastest
configuration to the matching tuned CSV
(``gfx,cu_num,N,T,H,varlen,dhu_bv,fwd_h_bv,prepare,us``):

* ``dhu_bv``: V-block width of the MFMA ``dhu`` stage (16 / 32 / 64);
* ``fwd_h_bv``: V-block width of the K5 recurrence (0 = its own rule);
* ``prepare``: forward-recompute backend, ``triton`` or any registered with
  ``register_fwd_prepare`` (``--prepare-module`` imports a module that does).

Inputs follow Kimi's setting (in-kernel q/k l2norm, ``lower_bound=-5`` gate,
beta sigmoid, safe_gate), bf16, K = V = 128. A candidate whose gradients differ
from the default configuration's by more than ``--max-err`` (relative RMS) is
discarded. Rows are merged into the output: one row per shape, the lower ``us``
wins, so re-tuning never adds a duplicate (see .claude/skills/aiter-config-shape).

    python3 csrc/kda_chunk_bwd/kda_chunk_bwd_tune.py \\
        -i aiter/configs/model_configs/kda_chunk_bwd_untuned_kimi_k3.csv
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import os

import pandas as pd
import torch

from aiter import logger
from aiter.jit.core import AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl import kda_chunk_bwd as fb
from aiter.test_common import run_perftest

KEY = ["gfx", "cu_num", "N", "T", "H", "varlen"]
TUNED_COLUMNS = KEY + ["dhu_bv", "fwd_h_bv", "prepare", "us"]
MODEL_DIR = f"{AITER_ROOT_DIR}/aiter/configs/model_configs"
D = 128


def make_inputs(n, t, h, varlen, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, tokens = (1, n * t) if varlen else (n, t)

    def r(*shape, dtype=torch.bfloat16):
        return torch.randn(
            *shape, device="cuda", dtype=torch.float32, generator=gen
        ).to(dtype)

    x = {
        "do": r(rows, tokens, h, D),
        "q": r(rows, tokens, h, D),
        "k": r(rows, tokens, h, D),
        "v": r(rows, tokens, h, D),
        "g": r(rows, tokens, h, D),
        "beta": r(rows, tokens, h, dtype=torch.float32),
        "A_log": r(h, dtype=torch.float32) * 0.5,
        "dt_bias": r(h * D, dtype=torch.float32) * 0.5,
    }
    cu = (
        torch.arange(0, n * t + 1, t, dtype=torch.int32, device="cuda")
        if varlen
        else None
    )
    return x, cu


def run(x, cu, **cfg):
    return fb.chunk_kda_bwd_flydsl_full(
        x["do"],
        None,
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        D**-0.5,
        cu_seqlens=cu,
        chunk_size=64,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        **cfg,
    )


def rel_err(ref, out):
    worst = 0.0
    for a, b in zip(ref[:5], out[:5]):
        d = (b.float() - a.float()).square().mean().sqrt()
        worst = max(worst, (d / (a.float().square().mean().sqrt() + 1e-8)).item())
    return worst


def tune_row(row, args, prepares):
    n, t, h, vl = (int(row[c]) for c in ("N", "T", "H", "varlen"))
    x, cu = make_inputs(n, t, h, bool(vl))
    ref = run(x, cu, dhu_bv=fb.DEFAULT_DHU_BV, prepare="triton")
    best = None
    for dhu_bv, fwd_h_bv, prep in itertools.product(
        args.dhu_bv, args.fwd_h_bv, prepares
    ):
        cfg = {"dhu_bv": dhu_bv, "fwd_h_bv": fwd_h_bv or None, "prepare": prep}
        try:
            out, us = run_perftest(
                run, x, cu, **cfg, num_iters=args.iters, num_warmup=args.warmup
            )
        except (
            RuntimeError,
            ValueError,
            AssertionError,
        ) as exc:  # build or launch failure
            logger.warning(
                "N=%d T=%d H=%d varlen=%d %s failed: %s", n, t, h, vl, cfg, exc
            )
            continue
        err = rel_err(ref, out)
        logger.info(
            "N=%d T=%d H=%d varlen=%d %s: %.1f us, err %.2e", n, t, h, vl, cfg, us, err
        )
        if err > args.max_err:
            continue
        if best is None or us < best["us"]:
            best = {**cfg, "fwd_h_bv": fwd_h_bv, "us": us}
    if best is None:
        return None
    return {**{c: row[c] for c in KEY}, **best}


def merge_write(path, rows):
    """One row per shape key; the lower ``us`` wins."""
    new = pd.DataFrame(rows, columns=TUNED_COLUMNS)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True) if not old.empty else new
    new = new.sort_values("us").drop_duplicates(KEY, keep="first")
    new = new.sort_values(["H", "varlen", "N", "T"])[TUNED_COLUMNS]
    new.to_csv(path, index=False)
    logger.info("wrote %d rows to %s", len(new), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-i",
        "--untuned",
        nargs="*",
        default=[
            f"{MODEL_DIR}/kda_chunk_bwd_untuned_kimi_k3.csv",
            f"{MODEL_DIR}/kda_chunk_bwd_untuned_kimi_k3.1.csv",
        ],
        help="untuned CSVs; each writes the tuned CSV with 'untuned' -> 'tuned'",
    )
    parser.add_argument("-o", "--tuned", default=None, help="one output for all inputs")
    parser.add_argument("--dhu-bv", type=int, nargs="*", default=[16, 32, 64])
    parser.add_argument("--fwd-h-bv", type=int, nargs="*", default=[0, 16, 32, 64])
    parser.add_argument("--prepare", nargs="*", default=["triton"])
    parser.add_argument(
        "--prepare-module",
        nargs="*",
        default=[],
        help="modules to import first; they call register_fwd_prepare",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-err", type=float, default=1e-2)
    args = parser.parse_args()

    for mod in args.prepare_module:
        importlib.import_module(mod)
    prepares = [p for p in args.prepare if p == "triton" or p in fb._PREPARE_IMPLS]
    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    for path in args.untuned:
        untuned = pd.read_csv(path)
        mine = untuned[(untuned["gfx"] == gfx) & (untuned["cu_num"] == cu_num)]
        if mine.empty:
            logger.warning("%s: no rows for %s / %d CUs", path, gfx, cu_num)
            continue
        rows = [
            r
            for r in (tune_row(row, args, prepares) for _, row in mine.iterrows())
            if r
        ]
        merge_write(args.tuned or path.replace("untuned", "tuned"), rows)


if __name__ == "__main__":
    main()
