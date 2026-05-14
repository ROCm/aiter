#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Unfused JAX cross-attn timings (forward + backward), same profiling style as TE `fa_profiling.py`.
# Configuration is **only** via environment variables (set in `run_mha_performance_comparison_cross_attn.sh`).
#
# Required:
#   JAX_UNFUSED_OUT          — output CSV path (batch,s_kv,jax_fwd_ms,jax_bwd_ms)
#
# Optional (defaults match the shell block there):
#   JAX_UNFUSED_BATCH        (2048)
#   JAX_UNFUSED_NHEADS       (32)
#   JAX_UNFUSED_HDIM         (128)
#   JAX_UNFUSED_KV_MIN       (2)
#   JAX_UNFUSED_KV_MAX       (16)
#   JAX_UNFUSED_WARMUP       (5)
#   JAX_UNFUSED_REPEAT       (25)
#   JAX_UNFUSED_SM_SCALE     ck | te   — ck: 1/sqrt(d) like C++ bench; te: 1/d like TE Case
#   JAX_UNFUSED_NR_SEGMENTS  (1)
#   JAX_UNFUSED_LAYOUT       (bshd)    — TE `utils.gen_data` layout; same B/s_q/s_kv/h/d as CK BHSD bench
#
# Requires: jax, jaxlib, einops

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from jax_unfused_attention import gen_data, jax_attention_te_entry


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def block_until_ready_tree(x) -> None:
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), x)


def main() -> None:
    out = os.environ.get("JAX_UNFUSED_OUT")
    if not out:
        print("error: set JAX_UNFUSED_OUT to the output CSV path", file=sys.stderr)
        sys.exit(1)

    batch = _env_int("JAX_UNFUSED_BATCH", 2048)
    nheads = _env_int("JAX_UNFUSED_NHEADS", 32)
    hdim = _env_int("JAX_UNFUSED_HDIM", 128)
    kv_min = _env_int("JAX_UNFUSED_KV_MIN", 2)
    kv_max = _env_int("JAX_UNFUSED_KV_MAX", 16)
    warmup = _env_int("JAX_UNFUSED_WARMUP", 5)
    repeat = _env_int("JAX_UNFUSED_REPEAT", 25)
    sm_mode = _env_str("JAX_UNFUSED_SM_SCALE", "ck").lower()
    nr_seg = _env_int("JAX_UNFUSED_NR_SEGMENTS", 1)
    layout = _env_str("JAX_UNFUSED_LAYOUT", "bshd")

    if sm_mode == "ck":
        sm_scale = float(1.0 / jnp.sqrt(jnp.array(hdim, dtype=jnp.float32)))
    elif sm_mode == "te":
        sm_scale = 1.0 / float(hdim)
    else:
        print("error: JAX_UNFUSED_SM_SCALE must be ck or te", file=sys.stderr)
        sys.exit(1)

    dtype = jnp.bfloat16
    causal = False
    sliding = -1

    rows: list[dict[str, str]] = []

    for p in range(kv_min, kv_max + 1):
        q, k, v, do, seg_q, seg_kv = gen_data(
            dtype,
            batch,
            1,
            p,
            nheads,
            hdim,
            gqa_ratio=1,
            nr_segments=nr_seg,
        )

        def fwd_for_profile(
            q_: jax.Array,
            k_: jax.Array,
            v_: jax.Array,
            seg_q_: jax.Array,
            seg_kv_: jax.Array,
            sm_s: float,
            caus: bool,
            win: int | None,
            lay: str,
            nr: int,
        ) -> jax.Array:
            return jax_attention_te_entry(q_, k_, v_, seg_q_, seg_kv_, sm_s, caus, win, lay, nr)

        fwd_jit = jax.jit(
            fwd_for_profile,
            static_argnames=("sm_s", "caus", "win", "lay", "nr"),
        )

        fwd_args = (q, k, v, seg_q, seg_kv, sm_scale, causal, sliding, layout, nr_seg)

        def f_closure(q_, k_, v_):
            return jax_attention_te_entry(
                q_, k_, v_, seg_q, seg_kv, sm_scale, causal, sliding, layout, nr_seg
            )

        _, pullback = jax.vjp(f_closure, q, k, v)
        bwd_jit = jax.jit(pullback)

        for _ in range(warmup):
            block_until_ready_tree(fwd_jit(*fwd_args))
        for _ in range(warmup):
            block_until_ready_tree(bwd_jit(do))

        fwd_ms: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter_ns()
            o = fwd_jit(*fwd_args)
            block_until_ready_tree(o)
            fwd_ms.append((time.perf_counter_ns() - t0) / 1e6)

        bwd_ms: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter_ns()
            g = bwd_jit(do)
            block_until_ready_tree(g)
            bwd_ms.append((time.perf_counter_ns() - t0) / 1e6)

        rows.append(
            {
                "batch": str(batch),
                "s_kv": str(p),
                "jax_fwd_ms": f"{sum(fwd_ms) / len(fwd_ms):.3f}",
                "jax_bwd_ms": f"{sum(bwd_ms) / len(bwd_ms):.3f}",
            }
        )

    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "s_kv", "jax_fwd_ms", "jax_bwd_ms"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {outp} ({len(rows)} rows), JAX_UNFUSED_SM_SCALE={sm_mode}")


if __name__ == "__main__":
    main()
