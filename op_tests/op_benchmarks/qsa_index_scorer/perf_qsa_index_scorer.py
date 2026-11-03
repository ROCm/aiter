# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Formal performance protocol: aiter API (fused) vs frozen ATOM native, same process + window.

Per bucket, identical golden fixtures (``qsa_golden.build_case``, seed 0,
e2e pool 738,964 pages, table width 16,384) are timed for

* ``fused``  -- ``aiter.ops.qsa_index_scorer.qsa_paged_mqa_logits(profile=...)``
                (eligibility check included: this is what a caller pays);
* ``native`` -- the frozen ATOM ``qsa_paged_mqa_logits`` (``$QSA_GOLDEN_DIR/atom_impl``),

with ``aiter.test_common.run_perftest(testGraph=True, num_rotate_args=ROT)`` and
an explicit graph-replay event timing (ITERS calls captured in one graph, ROT
rotated input copies so K is not L2-resident; median of 5 replays). Legs are
interleaved (fused, native, fused, native, ...) LEGS times in one process; the
reported value is the per-bucket median and ratio fused/native. Prefill fp8:
ATOM's default route aborts (LLVM PHINode assertion), so the native leg is
skipped and only the fused time is reported (no like-for-like ratio exists).

Also reports the host cost of ``decide()`` per call (eager-mode overhead).

    QSA_GOLDEN_DIR=... python3 perf_qsa_index_scorer.py [--cases a,b] [--legs 3] [--json out.json]

Run only under the node's gpu-lock and idle gate (root). Not run by the integration worker.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GD = os.environ.get("QSA_GOLDEN_DIR", "/work/task/MI455_QSA_INDEX_SCORER/golden_baseline")
sys.path.insert(0, os.path.abspath(GD))
import qsa_golden as G  # noqa: E402
import qsa_baseline as QB  # noqa: E402
from qsa_shapes import make_case  # noqa: E402

try:
    from aiter.test_common import run_perftest  # noqa: E402
    TC_SRC = "aiter package"
except Exception as _ex:  # noqa: BLE001 -- repo file with a logger-only aiter stub
    import logging
    import types
    for _m in [m for m in list(sys.modules) if m == "aiter" or m.startswith("aiter.")]:
        del sys.modules[_m]
    _stub = types.ModuleType("aiter")
    _stub.logger = logging.getLogger("aiter")
    _stub.__path__ = []
    sys.modules["aiter"] = _stub
    _p = os.path.join(ROOT, "aiter", "test_common.py")
    _spec = importlib.util.spec_from_file_location("aiter.test_common", _p)
    _tc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tc)
    run_perftest = _tc.run_perftest
    TC_SRC = f"file {_p} (aiter import failed: {type(_ex).__name__})"
    del sys.modules["aiter"]


def load_api():
    d = os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer")
    spec = importlib.util.spec_from_file_location("qsa_isc_perf", os.path.join(d, "__init__.py"),
                                                  submodule_search_locations=[d])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["qsa_isc_perf"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


PKG = load_api()
#: profile name or JSON path (e.g. the GPU-restamped `gen_profile check --out` file)
PROFILE = os.environ.get("QSA_TEST_PROFILE") or PKG.DEFAULT_PROFILE
ITERS = int(os.environ.get("PT_ITERS", "51"))
ROT = int(os.environ.get("PT_ROT", "4"))
CASES = [  # name, regime, rows, requests, max_ctx, fmt  (integration brief section 5)
    ("decode_bf16_c65536", "decode", 64, 16, 262144, "bf16"),
    ("decode_bf16_c32768", "decode", 64, 16, 131072, "bf16"),
    ("draft_bf16_c65536", "draft", 16, 16, 262144, "bf16"),
    ("draft_bf16_c32768", "draft", 16, 16, 131072, "bf16"),
    ("decode_fp8_c65536", "decode", 64, 16, 262144, "fp8"),
    ("decode_fp8_c32768", "decode", 64, 16, 131072, "fp8"),
    ("draft_fp8_c65536", "draft", 16, 16, 262144, "fp8"),
    ("draft_fp8_c32768", "draft", 16, 16, 131072, "fp8"),
    ("prefill512_bf16_c65536", "prefill", 512, 2, 262144, "bf16"),
    ("prefill320_bf16_c65536", "prefill", 320, 2, 262144, "bf16"),
    ("prefill512_fp8_c65536", "prefill", 512, 2, 262144, "fp8"),
    ("prefill320_fp8_c65536", "prefill", 320, 2, 262144, "fp8"),
]


def fused(q, kc, pt, t2r, qp, cl, sc, cols):
    return PKG.qsa_paged_mqa_logits(q, kc, pt, t2r, qp, cl, 4, None, cols, None, False, sc,
                                    profile=PROFILE)


def make_native(atom):
    def native(q, kc, pt, t2r, qp, cl, sc, cols):
        d = {"q": q, "compressed_k_cache": kc, "page_table": pt, "token_to_request": t2r,
             "query_positions": qp, "context_lens": cl, "compressed_k_scale": sc}
        return QB.call_scorer(atom["fn"], d, cols, False, None)
    return native


def graph_replay_us(fn, args, iters=ITERS):
    rot = [copy.deepcopy(args) for _ in range(ROT - 1)] + [args]
    for a in rot:
        fn(*a)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for i in range(iters):
                fn(*rot[i % ROT])
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / iters)
    del rot, g
    return statistics.median(ts)


def decide_host_us(args, n=2000):
    q, kc, pt, t2r, qp, cl, sc, cols = args
    for _ in range(50):
        PKG.decide(q, kc, pt, t2r, qp, cl, 4, None, cols, None, False, sc,
                   profile=PROFILE)
    t0 = time.perf_counter()
    for _ in range(n):
        PKG.decide(q, kc, pt, t2r, qp, cl, 4, None, cols, None, False, sc,
                   profile=PROFILE)
    return (time.perf_counter() - t0) * 1e6 / n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--legs", type=int, default=3)
    ap.add_argument("--json")
    ap.add_argument("--profile", help="profile name or JSON path (default $QSA_TEST_PROFILE or the default)")
    a = ap.parse_args(argv)
    global PROFILE
    PROFILE = a.profile or PROFILE
    atom = QB.load_atom(os.path.join(GD, "atom_impl"))
    native = make_native(atom)
    props = torch.cuda.get_device_properties(0)
    meta = {"iters": ITERS, "rotate": ROT, "legs": a.legs, "test_common": TC_SRC,
            "device": f"{props.gcnArchName} cu={props.multi_processor_count}",
            "atom_sha256": atom["sha256"], "profile": PROFILE, "kernel_sha256": PKG.api._kernel_sha256(),
            "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("QSA_", "ATOM_"))}}
    print(json.dumps(meta), flush=True)
    sel = [s for s in a.cases.split(",") if s]
    out = []
    for name, regime, rows, reqs, ctx, fmt in CASES:
        if sel and name not in sel:
            continue
        case = make_case(name, regime, rows, reqs, ctx, index_dtype=fmt, table_width=16384,
                         cache_pages=738964, allow_flydsl=False)
        d = G.build_case(case, device="cuda", seed=0)
        cols = int(case.columns)
        args = (d["q"], d["compressed_k_cache"], d["page_table"], d["token_to_request"],
                d["query_positions"], d["context_lens"], d["compressed_k_scale"], cols)
        dec = PKG.decide(*args[:6], 4, None, cols, None, False, args[6],
                         profile=PROFILE)
        rec = {"case": name, "decision": dec.reason.value, "cell": dec.cell, "vid": dec.vid,
               "fused_replay_us": [], "native_replay_us": [], "fused_perftest_us": [],
               "native_perftest_us": []}
        native_note = "ATOM default route" if not (regime == "prefill" and fmt == "fp8") else \
            "ATOM prefill fp8 default route aborts: native leg skipped (see known limitations)"
        rec["native_note"] = native_note
        runs = [("fused", fused)] if dec.ok else []
        if not (regime == "prefill" and fmt == "fp8"):
            runs.append(("native", native))
        for _leg in range(a.legs):
            for who, fn in runs:
                try:
                    _, us = run_perftest(fn, *args, num_iters=ITERS, num_warmup=3,
                                         num_rotate_args=ROT, testGraph=True)
                except Exception as ex:  # noqa: BLE001
                    us = float("nan")
                    rec.setdefault("errors", []).append(f"{who} perftest {type(ex).__name__}: {ex}")
                rec[f"{who}_perftest_us"].append(us)
                rec[f"{who}_replay_us"].append(graph_replay_us(fn, args))
                torch.cuda.empty_cache()
        for key in ("fused_replay_us", "native_replay_us", "fused_perftest_us", "native_perftest_us"):
            vals = [v for v in rec[key] if v == v]
            rec[key.replace("_us", "_median_us")] = statistics.median(vals) if vals else None
        f, n = rec["fused_replay_median_us"], rec["native_replay_median_us"]
        rec["ratio_replay"] = (f / n) if (f and n) else None
        if dec.ok:
            rec["decide_host_us"] = decide_host_us(args)
        print(json.dumps({k: rec[k] for k in ("case", "decision", "fused_replay_median_us",
                                              "native_replay_median_us", "ratio_replay")}),
              flush=True)
        out.append(rec)
        del d, args
        torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"meta": meta, "cases": out}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
