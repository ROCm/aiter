#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Run after CK scenarios. Reads results/scenario*/{fwd,bwd}.csv and adds jax_unfused(ms).

from __future__ import annotations

import csv
import gc
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from gen_small_attn_lengths import generate_logical_lengths
from jax_unfused_attention import (
    bench_mean_ms,
    gen_dense_batch_uniform,
    gen_dense_self_attn,
    gen_thd_cross_attn,
    gen_thd_varlen,
    jax_attention,
    jax_attention_thd,
    jax_cross_attention_thd,
    sm_scale_ck,
)

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", BENCH_DIR / "results"))
JAX_COL = "jax_unfused(ms)"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _merge_column(path: Path, timings: dict[tuple[str, str, str], float]) -> None:
    fields, rows = _read_csv(path)
    if JAX_COL not in fields:
        fields.append(JAX_COL)
    for row in rows:
        key = (row["batch"], row["s_q"], row["s_kv"])
        ms = timings.get(key)
        row[JAX_COL] = f"{ms:.3f}" if ms is not None else ""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _bench_fwd_dense(q, k, v, seg_q, seg_kv, hdim: int, warmup: int, repeat: int) -> float:
    sm = sm_scale_ck(hdim)
    fn = jax.jit(jax_attention, static_argnames=("softmax_scale", "is_causal"))
    return bench_mean_ms(fn, (q, k, v, seg_q, seg_kv, sm, False), warmup, repeat)


def _bench_bwd_dense(q, k, v, seg_q, seg_kv, hdim: int, warmup: int, repeat: int) -> float:
    sm = sm_scale_ck(hdim)
    do = jax.random.normal(jax.random.PRNGKey(1), q.shape, dtype=q.dtype) / 8

    def fwd(qa, ka, va):
        return jax_attention(qa, ka, va, seg_q, seg_kv, sm, False)

    _, pullback = jax.vjp(fwd, q, k, v)
    return bench_mean_ms(jax.jit(pullback), (do,), warmup, repeat)


def _bench_row(
    scenario: str,
    kind: str,
    row: dict[str, str],
    *,
    dtype,
    nheads: int,
    hdim: int,
    warmup: int,
    repeat: int,
) -> float:
    b = int(row["batch"])
    sq_col = int(row["s_q"])
    sk_col = int(row["s_kv"])
    sm = sm_scale_ck(hdim)

    if scenario in ("1", "2"):
        p = sk_col
        sq_list, sk_list = generate_logical_lengths(int(scenario), b, p, seed=6764)
        if kind == "fwd":
            if scenario == "1":
                q, k, v, cu_q, cu_k = gen_thd_varlen(dtype, sq_list, sk_list, nheads, hdim)
                return bench_mean_ms(
                    jax_attention_thd, (q, k, v, cu_q, cu_k, p, p, sm, False), warmup, repeat
                )
            q, k, v, cu_k = gen_thd_cross_attn(dtype, b, sk_list, nheads, hdim)
            return bench_mean_ms(
                jax_cross_attention_thd, (q, k, v, cu_k, p, sm, False), warmup, repeat
            )
        sq_pad = p if scenario == "1" else 1
        q, k, v, seg_q, seg_kv = gen_dense_batch_uniform(dtype, b, sq_pad, p, nheads, hdim)
        return _bench_bwd_dense(q, k, v, seg_q, seg_kv, hdim, warmup, repeat)

    # scenario 3_4
    s = sq_col
    q, k, v, seg_q, seg_kv = gen_dense_self_attn(dtype, b, s, nheads, hdim)
    if kind == "fwd":
        return _bench_fwd_dense(q, k, v, seg_q, seg_kv, hdim, warmup, repeat)
    return _bench_bwd_dense(q, k, v, seg_q, seg_kv, hdim, warmup, repeat)


def _process_csv(scenario: str, kind: str, csv_path: Path, warmup: int, repeat: int) -> None:
    if not csv_path.is_file():
        print(f"skip missing {csv_path}", file=sys.stderr)
        return

    dtype = jnp.bfloat16
    nheads = _env_int("NHEADS", 32)
    hdim = _env_int("HDIM", 128)
    _, rows = _read_csv(csv_path)
    timings: dict[tuple[str, str, str], float] = {}

    for row in rows:
        key = (row["batch"], row["s_q"], row["s_kv"])
        print(f"JAX scenario {scenario} {kind} batch={key[0]} s_q={key[1]} s_kv={key[2]}", file=sys.stderr, flush=True)
        timings[key] = _bench_row(scenario, kind, row, dtype=dtype, nheads=nheads, hdim=hdim, warmup=warmup, repeat=repeat)
        jax.clear_caches()
        gc.collect()

    _merge_column(csv_path, timings)
    print(f"updated {csv_path}", file=sys.stderr)


def _process_scenario(scenario: str, warmup: int, repeat: int) -> None:
    scen_dir = RESULTS_DIR / f"scenario{scenario}"
    for kind in ("fwd", "bwd"):
        _process_csv(scenario, kind, scen_dir / f"{kind}.csv", warmup, repeat)


def main() -> None:
    warmup = _env_int("WARMUP", 5)
    repeat = _env_int("REPEAT", 25)

    if len(sys.argv) < 2:
        print("usage: run_jax_benchmark.py <1|2|3_4|all>", file=sys.stderr)
        sys.exit(2)

    arg = sys.argv[1]
    scenarios = ["1", "2", "3_4"] if arg == "all" else [arg]
    for s in scenarios:
        if s not in ("1", "2", "3_4"):
            print(f"unknown scenario {s}", file=sys.stderr)
            sys.exit(2)
        _process_scenario(s, warmup, repeat)


if __name__ == "__main__":
    main()
