# Benchmark shuffled-cache decode split-K choices.
"""A/B split-K planner test for pure decode on the 5D shuffled fp8 KV cache.

A-side forces the production decode kernel even on cells the dispatch gate cedes.
B-side overrides the dispatch module's split planner to sweep candidate counts.
Triton, Gluon and default are timed separately on the same inputs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from unittest import mock

import torch

sys.path.insert(0, os.getcwd())
import aiter.ops.flydsl.unified_attention_kernels as uak
import aiter.ops.unified_attention as ua
from scripts.bench_decode_mixed_shuffled import (
    FLYDSL_DECLINE,
    build_shuffled_full,
    correctness,
    time_us,
)

CANDIDATE_SPLITS = [2, 4, 8, 16]  # capped by _MAX_SEGMENTS=16


def clear_flydsl_cache():
    d = os.path.expanduser("~/.flydsl/cache")
    if os.path.isdir(d):
        shutil.rmtree(d)
        print(f"cleared {d}")
    else:
        print(f"{d} did not exist")


def captured_split(kw):
    """Capture a split only if the decode builder actually launches."""
    real_get = uak._get_decode_kernel
    seen = {}

    def spy(*args, **kwargs):
        seen["split"] = args[5] if len(args) > 5 else kwargs["num_kv_splits"]
        return real_get(*args, **kwargs)

    with mock.patch.object(uak, "_get_decode_kernel", spy):
        try:
            ua.unified_attention(**kw, backend="flydsl")
        except RuntimeError as exc:
            if str(exc) != FLYDSL_DECLINE:
                raise
    return seen.get("split"), "split" in seen


def force_decode_ctx():
    return mock.patch.object(uak, "_decode_dispatch_action", return_value="route")


def force_split_ctx(n):
    return mock.patch.object(uak, "plan_num_kv_splits", return_value=n)


def time_flydsl(kw):
    with force_decode_ctx():
        return time_us(lambda: ua.unified_attention(**kw, backend="flydsl"))


def time_triton(kw, baseline="triton"):
    return time_us(lambda: ua.unified_attention(**kw, backend=baseline))


def bench_shape(b, ctx):
    kw, ref_kw, output = build_shuffled_full([1] * b, [ctx] * b)

    # Distinguish the production gate from the forced decode measurements.
    _, served = captured_split(kw)
    with force_decode_ctx():
        a_split, launched = captured_split(kw)
        assert launched
        err, cos, ok_a = correctness(kw, ref_kw, output)
    fly_a = time_flydsl(kw)
    times = {
        backend: time_triton(kw, None if backend == "default" else backend)
        for backend in ("triton", "gluon", "default")
    }
    tri = times["triton"]

    b_rows = []
    for n in CANDIDATE_SPLITS:
        with force_decode_ctx(), force_split_ctx(n):
            got_split, b_served = captured_split(kw)
            berr, bcos, ok_b = correctness(kw, ref_kw, output)
            t = time_us(lambda: ua.unified_attention(**kw, backend="flydsl"))
        b_rows.append(
            {
                "n": n,
                "got": got_split,
                "served": b_served,
                "err": berr,
                "cos": bcos,
                "ok": ok_b,
                "us": t,
            }
        )

    valid = [r for r in b_rows if r["ok"] and r["served"]]
    best = min(valid, key=lambda r: r["us"]) if valid else None
    return {
        "b": b,
        "ctx": ctx,
        "served": served,
        "a_split": a_split,
        "err": err,
        "cos": cos,
        "ok_a": ok_a,
        "fly_a": fly_a,
        "tri": tri,
        "times": times,
        "b_rows": b_rows,
        "best": best,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cell", nargs=2, type=int, metavar=("B", "CTX"))
    p.add_argument("--baseline", choices=("triton", "gluon"), default="triton")
    a = p.parse_args()
    print("--baseline is deprecated and ignored; timings include all backends.")

    clear_flydsl_cache()
    import aiter

    print(f"aiter: {aiter.__file__}")
    print(
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
        f"device={torch.cuda.get_device_name()}"
    )
    print(
        f"target_num_prgms={uak._target_num_prgms(0)} "
        f"MAX_SEGMENTS={uak._MAX_SEGMENTS}"
    )

    if a.cell:
        batches, ctxs = [a.cell[0]], [a.cell[1]]
    elif a.quick:
        batches = [8, 32, 64]
        ctxs = [4096, 16384]
    else:
        # SILOTIGER-877 A/B decode grid requested for cede-gate boundary pinning.
        batches = [8, 16, 24, 32, 48, 56, 64]
        ctxs = [1024, 2048, 4096, 8192, 16384]

    results = []
    for b in batches:
        for ctx in ctxs:
            r = bench_shape(b, ctx)
            results.append(r)
            best = r["best"]
            best_str = f"n={best['n']} {best['us']:.1f}us" if best else "(none valid)"
            ratio_a = r["tri"] / r["fly_a"]
            ratio_b = (r["tri"] / best["us"]) if best else float("nan")
            print(
                f"b={b:3d} ctx={ctx:6d} served={'yes' if r['served'] else 'ceded':5} "
                f"FlyDSL(split={r['a_split']})={r['fly_a']:6.1f}us "
                f"B[{best_str}] triton={r['tri']:6.1f}us "
                f"gluon={r['times']['gluon']:6.1f}us "
                f"default={r['times']['default']:6.1f}us "
                f"tri/A={ratio_a:.2f} glu/A={r['times']['gluon']/r['fly_a']:.2f} "
                f"tri/B={ratio_b:.2f} "
                f"corrA={'ok' if r['ok_a'] else 'BAD'}"
            )

    print(f"ceded shapes: {sum(not r['served'] for r in results)}")

    # ---- summary table ----
    print("\n=== A/B DECODE SPLIT-K TABLE (5D shuffled fp8, 64Q/4KV, d128) ===")
    hdr = (
        "b",
        "ctx",
        "A_spl",
        "flyA_us",
        "B_spl",
        "flyB_us",
        "tri_us",
        "glu_us",
        "def_us",
        "A_x",
        "B_x",
        "dBvsA",
    )
    print(
        (
            "{:>4} {:>6} {:>5} {:>8} {:>5} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>7}"
        ).format(*hdr)
    )
    for r in results:
        best = r["best"]
        ax = r["tri"] / r["fly_a"]
        if best:
            bx = r["tri"] / best["us"]
            dba = r["fly_a"] / best["us"]  # >1 means B faster than A
            print(
                (
                    "{:>4} {:>6} {:>5} {:>8.1f} {:>5} {:>8.1f} {:>8.1f} "
                    "{:>8.1f} {:>8.1f} {:>6.2f} {:>6.2f} {:>6.2f}x"
                ).format(
                    r["b"],
                    r["ctx"],
                    r["a_split"],
                    r["fly_a"],
                    best["n"],
                    best["us"],
                    r["tri"],
                    r["times"]["gluon"],
                    r["times"]["default"],
                    ax,
                    bx,
                    dba,
                )
            )
        else:
            print(
                (
                    "{:>4} {:>6} {:>5} {:>8.1f} {:>5} {:>8} {:>8.1f} "
                    "{:>8.1f} {:>8.1f} {:>6.2f} {:>6} {:>7}"
                ).format(
                    r["b"],
                    r["ctx"],
                    r["a_split"],
                    r["fly_a"],
                    "-",
                    "-",
                    r["tri"],
                    r["times"]["gluon"],
                    r["times"]["default"],
                    ax,
                    "-",
                    "-",
                )
            )

    # per-shape full B sweep (for choosing best split count)
    print("\n=== FULL B SWEEP (us per forced split; * = best correct) ===")
    print(
        ("{:>4} {:>6} " + " ".join(f"n{n:>5}" for n in CANDIDATE_SPLITS)).format(
            "b", "ctx"
        )
    )
    for r in results:
        cells = []
        best_n = r["best"]["n"] if r["best"] else None
        for row in r["b_rows"]:
            tag = "*" if row["n"] == best_n else (" " if row["ok"] else "x")
            if row["served"]:
                cells.append(f"{row['us']:6.1f}{tag}")
            else:
                cells.append(f"{'NS':>6}{tag}")
        print(
            ("{:>4} {:>6} " + " ".join("{:>7}" for _ in cells)).format(
                r["b"], r["ctx"], *cells
            )
        )

    # verdicts
    print("\n=== VERDICTS ===")
    regressions = []
    recovered = []
    for r in results:
        best = r["best"]
        ax = r["tri"] / r["fly_a"]
        if not best:
            continue
        bx = r["tri"] / best["us"]
        # regression: B slower than A by >3% where A was already >=1.0x
        if best["us"] > r["fly_a"] * 1.03:
            regressions.append((r["b"], r["ctx"], ax, bx, r["fly_a"], best["us"]))
        if r["b"] >= 48 and bx >= 1.0 and ax < 1.0:
            recovered.append((r["b"], r["ctx"], ax, bx))
    print(f"recovered (b>=48 to >=1.0x): {recovered}")
    print(f"B-regressions vs A (>3%): {regressions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
