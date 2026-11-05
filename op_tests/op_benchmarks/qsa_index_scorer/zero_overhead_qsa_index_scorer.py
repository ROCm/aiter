# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023 -- timing lambdas are called inside the loop iteration that defines them
"""GPU check of the API 1.1 zero-overhead path (``prewarm`` + ``launch``), gfx1250.

(a) ``prewarm`` with a fresh ``FLYDSL_RUNTIME_CACHE_DIR`` (``--fresh-cache``):
    per-variant build+compile seconds, then the FIRST real call of every
    prewarmed variant (both position dtypes) must take < ``--first-call-ms`` (5).
(b) host us/call (enqueue only: per-call wall time, sync every 16 calls) of
    ``launch`` vs the 1.0 op (``qsa_paged_mqa_logits`` = decide + launch) vs
    ``decide`` alone vs the kernel wrapper vs ATOM's native scorer (frozen
    ``$QSA_GOLDEN_DIR/atom_impl`` or ``atom`` on PYTHONPATH; skipped if neither).
(c) bitwise: ``launch`` == 1.0 op == kernel wrapper (pre-1.1 launch path) on a few shapes.

    QSA_GOLDEN_DIR=/path/golden_baseline python3 op_tests/op_benchmarks/qsa_index_scorer/\
zero_overhead_qsa_index_scorer.py --fresh-cache [--fp8] [--json out.json]

Run it as the FIRST GPU process of a fresh container (nothing compiled in-process).
GPU work must run under the node's gpu-lock.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def _parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fresh-cache",
        action="store_true",
        help="point FLYDSL_RUNTIME_CACHE_DIR at a new empty directory first",
    )
    ap.add_argument(
        "--fp8", action="store_true", help="also prewarm/check fp8 variants"
    )
    ap.add_argument("--first-call-ms", type=float, default=5.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--json")
    return ap.parse_args(argv)


ARGS = _parse(sys.argv[1:] if __name__ == "__main__" else [])
if ARGS.fresh_cache:  # before FlyDSL is imported (the kernel module imports it lazily)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(
        prefix="qsa_flydsl_cache_"
    )

import torch


def load_api():
    if os.environ.get("QSA_IMPORT_PACKAGE") == "1":
        import aiter.ops.qsa_index_scorer as pkg
        from aiter.ops.qsa_index_scorer import api

        return pkg, api
    d = os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer")
    spec = importlib.util.spec_from_file_location(
        "qsa_isc_zo", os.path.join(d, "__init__.py"), submodule_search_locations=[d]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["qsa_isc_zo"] = pkg
    spec.loader.exec_module(pkg)
    return pkg, sys.modules["qsa_isc_zo.api"]


PKG, API = load_api()
P = API.P
PROFILE = os.environ.get("QSA_TEST_PROFILE") or PKG.DEFAULT_PROFILE
PAGES = 4096


def make_call(rows, requests, width, columns, fmt="bf16", pos=torch.int64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    q = torch.randn((rows, 4, 128), generator=g, device=dev).to(torch.bfloat16)
    k = torch.randn((PAGES, 4, 1, 128), generator=g, device=dev)
    if fmt == "fp8":
        cache = (k * 0.5).to(torch.float8_e4m3fn)
        scale = torch.rand((PAGES, 4, 1), generator=g, device=dev) + 0.5
    else:
        cache, scale = k.to(torch.bfloat16), None
    pt = torch.randint(
        0, PAGES, (requests, width), generator=g, device=dev, dtype=torch.int32
    )
    pt[:, -1] = -1  # an invalid page in every request
    if rows <= requests:
        t2r = torch.arange(rows, device=dev, dtype=torch.int32)
    else:
        t2r = (torch.arange(rows, device=dev, dtype=torch.int32) * requests) // rows
    ctx = torch.randint(
        4 * width // 2, 4 * width * 4 + 1, (requests,), generator=g, device=dev
    )
    cl = ctx.to(torch.int32)
    qp = (ctx[t2r.long()] - 1 - (torch.arange(rows, device=dev) % 7)).to(pos)
    return [q, cache, pt, t2r, qp, cl, 4, None, columns, None, False, scale]


def pick_shapes(cache_dtypes):
    """One representative pass shape per (cache dtype, variant): smallest plane."""
    prof = P.load_profile(PROFILE)
    best = {}
    for r in [*prof.rows, *prof.rules]:
        if not r.verified or r.raw["cache_dtype"] not in cache_dtypes:
            continue
        if isinstance(r, P.ProfileRule):
            shapes = [
                (s["rows"], s["requests"], s["table_width"], s["columns"])
                for s in r.raw["samples"]
            ]
        else:
            shapes = [
                (
                    r.raw["rows"]["min"],
                    r.raw["requests"]["min"],
                    r.raw["table_width"],
                    r.raw["columns"],
                )
            ]
        for s in shapes:
            key = (r.raw["cache_dtype"], r.vid)
            if key not in best or s[0] * s[3] < best[key][0] * best[key][3]:
                best[key] = s
    return best


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


def part_a(cache_dtypes, rec):
    t0 = time.perf_counter()
    tags = API.prewarm(PROFILE, cache_dtypes=cache_dtypes)
    rec["prewarm_total_s"] = round(time.perf_counter() - t0, 3)
    rec["prewarm_variants"] = tags
    rec["prewarm_build_s"] = {k[3]: round(v, 3) for k, v in API._PREWARMED.items()}
    t1 = time.perf_counter()
    again = API.prewarm(PROFILE, cache_dtypes=cache_dtypes)
    rec["prewarm_again_ms"] = round((time.perf_counter() - t1) * 1e3, 3)
    rec["prewarm_idempotent"] = again == tags
    first = []
    for (fmt, vid), (rows, reqs, width, cols) in sorted(
        pick_shapes(cache_dtypes).items()
    ):
        for pos in (torch.int32, torch.int64):
            a = make_call(rows, reqs, width, cols, fmt, pos)
            d = API.decide(*a, profile=PROFILE)
            ms, _ = sync_ms(lambda: API.launch(d, *a))
            ms2, _ = sync_ms(lambda: API.launch(d, *a))
            first.append(
                {
                    "vid": vid,
                    "fmt": fmt,
                    "pos": str(pos),
                    "shape": [rows, reqs, cols],
                    "decision": d.reason.value,
                    "first_ms": round(ms, 3),
                    "second_ms": round(ms2, 3),
                    "ok": d.ok and ms < ARGS.first_call_ms,
                }
            )
            torch.cuda.empty_cache()
    rec["first_calls"] = first
    return bool(tags) and rec["prewarm_idempotent"] and all(f["ok"] for f in first)


def host_us(fn, iters):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if i % 16 == 15:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return round(statistics.median(ts) * 1e6, 2)


def load_atom():
    gd = os.environ.get("QSA_GOLDEN_DIR")
    if gd and os.path.isdir(os.path.join(gd, "atom_impl")):
        sys.path.insert(0, gd)
        try:
            import qsa_baseline as QB

            return QB.load_atom(os.path.join(gd, "atom_impl"))["fn"], "golden atom_impl"
        except Exception as exc:  # noqa: BLE001
            return None, f"golden atom_impl failed: {type(exc).__name__}: {exc}"
    try:
        from atom.model_ops.qwen4_exp.ops.qsa import (
            qsa_paged_mqa_logits as fn,
        )

        return fn, "atom (PYTHONPATH)"
    except Exception as exc:  # noqa: BLE001
        return None, f"skip: {type(exc).__name__}: {exc}"


def part_b(rec):
    atom, how = load_atom()
    rec["atom"] = how
    rows_out = []
    for rows, reqs, cols in ((16, 16, 65536), (64, 16, 65536), (512, 2, 65536)):
        a = make_call(rows, reqs, 16384, cols)
        d = API.decide(*a, profile=PROFILE)
        r = {"shape": [rows, reqs, cols], "vid": d.vid, "decision": d.reason.value}
        if d.ok:
            r["launch_us"] = host_us(lambda: API.launch(d, *a), ARGS.iters)
            r["op_1_0_us"] = host_us(
                lambda: API.qsa_paged_mqa_logits(*a, profile=PROFILE), ARGS.iters
            )
            r["decide_us"] = host_us(
                lambda: API.decide(*a, profile=PROFILE), ARGS.iters
            )
            r["kernel_wrapper_us"] = host_us(
                lambda: API._launch(
                    None,
                    a[0],
                    a[1],
                    a[2],
                    a[3],
                    a[4],
                    a[5],
                    None,
                    cols,
                    None,
                    a[11],
                    None,
                    None,
                ),
                ARGS.iters,
            )
        if atom is not None:
            r["atom_native_us"] = host_us(lambda: atom(*a), ARGS.iters)
        rows_out.append(r)
        print(json.dumps(r), flush=True)
    torch.cuda.empty_cache()
    rec["host"] = rows_out
    return all("launch_us" in r for r in rows_out)


def part_c(cache_dtypes, rec):
    out = []
    shapes = [
        (16, 16, 16384, 65536),
        (64, 16, 16384, 65536),
        (512, 2, 16384, 65536),
        (64, 16, 16384, 32768),
        (5, 1, 64, 256),
    ]
    for fmt in cache_dtypes:
        for rows, reqs, width, cols in shapes:
            for pos in (torch.int64, torch.int32):
                a = make_call(rows, reqs, width, cols, fmt, pos, seed=rows + cols)
                d = API.decide(*a, profile=PROFILE)
                r = {
                    "fmt": fmt,
                    "shape": [rows, reqs, width, cols],
                    "pos": str(pos),
                    "decision": d.reason.value,
                    "vid": d.vid,
                }
                if d.ok:
                    lo, lv = API.launch(d, *a)
                    oo, ov = API.qsa_paged_mqa_logits(*a, profile=PROFILE)
                    wo, wv = API._launch(
                        None,
                        a[0],
                        a[1],
                        a[2],
                        a[3],
                        a[4],
                        a[5],
                        None,
                        cols,
                        None,
                        a[11],
                        None,
                        None,
                    )
                    torch.cuda.synchronize()
                    r["launch_eq_op"] = bool(
                        torch.equal(lo.view(torch.int32), oo.view(torch.int32))
                        and torch.equal(lv, ov)
                    )
                    r["launch_eq_wrapper"] = bool(
                        torch.equal(lo.view(torch.int32), wo.view(torch.int32))
                        and torch.equal(lv, wv)
                    )
                    r["finite_or_neg_inf"] = bool(not torch.isnan(lo).any())
                out.append(r)
                torch.cuda.empty_cache()
    rec["bitwise"] = out
    checked = [r for r in out if "launch_eq_op" in r]
    return bool(checked) and all(
        r["launch_eq_op"] and r["launch_eq_wrapper"] and r["finite_or_neg_inf"]
        for r in checked
    )


def main() -> int:
    props = torch.cuda.get_device_properties(0)
    if not str(props.gcnArchName).startswith("gfx1250"):
        print(f"SKIP: gfx1250 only (device {props.gcnArchName})")
        return 0
    cache_dtypes = ("bf16", "fp8") if ARGS.fp8 else ("bf16",)
    rec = {
        "api_version": PKG.API_VERSION,
        "profile": PROFILE,
        "flydsl_cache_dir": os.environ.get("FLYDSL_RUNTIME_CACHE_DIR"),
        "device": f"{props.gcnArchName} cu={props.multi_processor_count}",
    }
    ok_a = part_a(cache_dtypes, rec)
    print(
        f"[A] prewarm {'PASS' if ok_a else 'FAIL'} {json.dumps(rec['prewarm_build_s'])}",
        flush=True,
    )
    ok_b = part_b(rec)
    ok_c = part_c(cache_dtypes, rec)
    print(f"[C] bitwise {'PASS' if ok_c else 'FAIL'}", flush=True)
    rec.update(pass_a=ok_a, pass_b=ok_b, pass_c=ok_c)
    if ARGS.json:
        with open(ARGS.json, "w") as fh:
            json.dump(rec, fh, indent=1)
    ok = ok_a and ok_b and ok_c
    print(
        f"[SUMMARY] {'PASS' if ok else 'FAIL'} launch_us="
        f"{[r.get('launch_us') for r in rec['host']]} atom_us="
        f"{[r.get('atom_native_us') for r in rec['host']]}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
