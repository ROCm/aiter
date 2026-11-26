# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune the FlyDSL KDA chunk forward (``chunk_kda_fwd_flydsl``) per shape.

Reads untuned shape CSVs (``gfx,cu_num,H,T,N,is_varlen,lower_bound_gate``),
times every config in ``aiter.ops.flydsl.kda_chunk_fwd.TUNABLES`` on the
current GPU, and writes the fastest config of each shape to the matching tuned
CSV (``*_untuned*`` -> ``*_tuned*``). Rows for another gfx / cu_num are kept
as they are.

A config is a candidate only if its output matches the default config's to
bf16 tolerance; a shape whose best config is not at least ``--min-gain``
faster than the default is written with the default config, so a tuned row
never pins noise.

Shapes: ``N`` sequences of ``T / N`` tokens; ``is_varlen`` packs them into one
batch with ``cu_seqlens``, else they are ``N`` batch rows of ``T`` tokens.
``lower_bound_gate`` picks Kimi's call (in-kernel gate with lower_bound=-5,
q/k l2norm, beta sigmoid); otherwise the gate is pre-activated.

Usage (one GPU)::

    python3 csrc/kda_chunk_fwd/kda_chunk_fwd_flydsl_tune.py
    python3 csrc/kda_chunk_fwd/kda_chunk_fwd_flydsl_tune.py \\
        -i aiter/configs/model_configs/kda_chunk_fwd_flydsl_untuned_kimi_k3.csv
"""

from __future__ import annotations

import argparse
import itertools
import os

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import logger
from aiter.jit.core import AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl import kda_chunk_fwd as kf
from aiter.test_common import run_perftest

KEY = ["gfx", "cu_num", "H", "T", "N", "is_varlen", "lower_bound_gate"]
D = 128
_MODEL_DIR = os.path.join(AITER_ROOT_DIR, "aiter", "configs", "model_configs")
DEFAULT_INPUTS = [
    os.path.join(_MODEL_DIR, "kda_chunk_fwd_flydsl_untuned_kimi_k3.csv"),
    os.path.join(_MODEL_DIR, "kda_chunk_fwd_flydsl_untuned_kimi_k3.1.csv"),
]


def _truthy(x) -> bool:
    return str(x).strip().lower() in ("1", "true")


def make_call(H, T, N, is_varlen, lower_bound_gate, seed=0):
    """``fn(config) -> o`` for one shape, inputs built once."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    if is_varlen:
        if T % N:
            raise ValueError(f"varlen T={T} is not a multiple of N={N}")
        B, Tb = 1, T
        cu = torch.arange(0, T + 1, T // N, device="cuda", dtype=torch.int64)
    else:
        B, Tb, cu = N, T, None
    bf = torch.bfloat16
    q, k, v = (r(B, Tb, H, D).to(bf) for _ in range(3))
    kw = {}
    if lower_bound_gate:
        g = r(B, Tb, H, D)
        beta = r(B, Tb, H)
        kw = {
            "use_gate_in_kernel": True,
            "A_log": torch.log(torch.rand(H, generator=gen, device="cuda") * 15 + 1),
            "dt_bias": r(H * D) * 0.5,
            "lower_bound": -5.0,
            "use_qk_l2norm_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
        }
    else:
        q, k = (F.normalize(x.float(), dim=-1).to(bf) for x in (q, k))
        g = F.logsigmoid(r(B, Tb, H, D) + 3)
        beta = torch.sigmoid(r(B, Tb, H))

    def call(config):
        o, _ = kf.chunk_kda_fwd_flydsl(
            q, k, v, g, beta, D**-0.5, cu_seqlens=cu, config=config, **kw
        )
        return o

    return call


def configs(lower_bound_gate):
    names = list(kf.TUNABLES)
    for vals in itertools.product(*kf.TUNABLES.values()):
        cfg = dict(zip(names, vals))
        if cfg["diag_mfma"] and not lower_bound_gate:
            continue  # diag_mfma is only honoured for the lower_bound gate
        yield cfg


def tune_shape(H, T, N, is_varlen, lower_bound_gate, args):
    call = make_call(H, T, N, is_varlen, lower_bound_gate)
    nseq = N
    chunks = kf._Chunks(
        1 if is_varlen else N,
        T,
        torch.device("cuda"),
        (
            torch.arange(0, T + 1, T // N, device="cuda", dtype=torch.int64)
            if is_varlen
            else None
        ),
    )
    default = kf.default_config(
        H=H,
        total_chunks=chunks.NT if is_varlen else nseq * chunks.NT,
        max_seq_chunks=chunks.max_seq_chunks,
        lower_bound_gate=lower_bound_gate,
        device=torch.device("cuda"),
    )
    ref, base_us = run_perftest(
        call, default, num_iters=args.iters, num_warmup=args.warmup
    )
    ref = ref.float()
    best, best_us = default, base_us
    for cfg in configs(lower_bound_gate):
        o, us = run_perftest(call, cfg, num_iters=args.iters, num_warmup=args.warmup)
        err = (o.float() - ref).abs().max().item()
        if not err < args.atol:
            logger.warning(
                "H=%d T=%d N=%d %s: max err %.3g, skipped", H, T, N, cfg, err
            )
            continue
        if us < best_us:
            best, best_us = cfg, us
    if best_us > base_us * (1 - args.min_gain):
        best, best_us = default, base_us
    logger.info(
        "H=%d T=%d N=%d varlen=%s: default %s %.1f us -> %s %.1f us",
        H,
        T,
        N,
        is_varlen,
        default,
        base_us,
        best,
        best_us,
    )
    return best, best_us


def tuned_path(untuned: str) -> str:
    base = os.path.basename(untuned)
    if "untuned" not in base:
        raise ValueError(f"{untuned}: not an untuned CSV")
    return os.path.join(os.path.dirname(untuned), base.replace("untuned", "tuned"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-i", "--untuned", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.02,
        help="fraction a config must beat the default by to be recorded",
    )
    args = parser.parse_args()
    gfx, cu_num = get_gfx_runtime(), get_cu_num()

    for path in args.untuned:
        shapes = pd.read_csv(path)
        out_path = tuned_path(path)
        old = (
            pd.read_csv(out_path)
            if os.path.exists(out_path)
            else pd.DataFrame(columns=KEY + list(kf.TUNABLES) + ["us"])
        )
        rows = []
        for s in shapes.to_dict("records"):
            if str(s["gfx"]).strip() != gfx or int(s["cu_num"]) != cu_num:
                continue
            shape = (
                int(s["H"]),
                int(s["T"]),
                int(s["N"]),
                _truthy(s["is_varlen"]),
                _truthy(s["lower_bound_gate"]),
            )
            cfg, us = tune_shape(*shape, args)
            rows.append(
                dict(zip(KEY, (gfx, cu_num, *shape)))
                | {n: cfg[n] for n in kf.TUNABLES}
                | {"us": us}
            )
        if not rows:
            logger.warning("%s: no shapes for %s / cu_num %d", path, gfx, cu_num)
            continue
        new = pd.DataFrame(rows)
        if len(old):
            for c in ("is_varlen", "lower_bound_gate"):
                old[c] = old[c].map(_truthy)
        merged = (
            pd.concat([old, new])
            .drop_duplicates(subset=KEY, keep="last")
            .sort_values(KEY)
        )
        merged.to_csv(out_path, index=False)
        logger.info("wrote %d rows to %s", len(merged), out_path)
    kf.reload_tuned_table()


if __name__ == "__main__":
    main()
