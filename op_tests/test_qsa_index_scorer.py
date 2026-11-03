# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU golden test: Phase-1 golden cases through the public aiter API (gfx1250 only).

Every case runs through ``aiter.ops.qsa_index_scorer.qsa_paged_mqa_logits``
with the default profile and is compared against the **frozen golden**
(``$QSA_GOLDEN_DIR/qsa_golden.py``) with the frozen comparator and tolerance
(``compare_logits(rtol=1e-4, atol=1e-4)`` over ``valid_page & c < visible``),
plus: ``visible_groups`` exact, no NaN, every element outside the mask
bit-identical to the golden ``-inf``, ``row_starts`` zero-filled, caller
outputs used in place, inputs untouched, int32 vs int64 metadata bit-identical,
CUDA-graph replay bit-identical to eager, profile rows re-checked, and fused vs
the frozen ATOM native scorer on identical inputs (same comparator).

    QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline \
        python3 op_tests/test_qsa_index_scorer.py [-k case_substring] [--json out.json]

The package is loaded standalone by default (``import aiter`` builds/touches the
GPU at import time); ``QSA_IMPORT_PACKAGE=1`` uses ``aiter.ops.qsa_index_scorer``.
GPU work must run under the node's gpu-lock (root runs this file).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GOLDEN_DIR = os.environ.get("QSA_GOLDEN_DIR", "/work/task/MI455_QSA_INDEX_SCORER/golden_baseline")
sys.path.insert(0, os.path.abspath(GOLDEN_DIR))
import qsa_golden as G  # noqa: E402
import qsa_shapes as S  # noqa: E402

RTOL = ATOL = 1e-4      # frozen Phase-1 tolerance
W, PAGES, CTX = 16384, 738964, 262144


def load_api():
    if os.environ.get("QSA_IMPORT_PACKAGE") == "1":
        import aiter.ops.qsa_index_scorer as pkg  # noqa: E402
        from aiter.ops.qsa_index_scorer import api, gen_profile  # noqa: E402
        return pkg, api, gen_profile
    d = os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer")
    spec = importlib.util.spec_from_file_location("qsa_isc_gpu", os.path.join(d, "__init__.py"),
                                                  submodule_search_locations=[d])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["qsa_isc_gpu"] = pkg
    spec.loader.exec_module(pkg)
    import qsa_isc_gpu.gen_profile as gen  # noqa: E402
    return pkg, sys.modules["qsa_isc_gpu.api"], gen


PKG, API, GEN = load_api()
PROFILE = os.environ.get("QSA_TEST_PROFILE") or PKG.DEFAULT_PROFILE
#: --allow-unverified: also serve `pending` rows (gen_profile sweep path). Needed to
#: run the suite on a kernel whose rows are not restamped yet (82c4dec5: fp8 rows).
ALLOW_UNVERIFIED = False


def scorer(*args, **kw):
    """Public raise-API; with --allow-unverified the try-API sweep path (pending rows)."""
    if not ALLOW_UNVERIFIED:
        return API.qsa_paged_mqa_logits(*args, **kw)
    out, d = API.try_qsa_paged_mqa_logits(*args, allow_unverified=True, **kw)
    if out is None:
        raise API.QSAIndexScorerUnsupported(d)
    return out

# Phase-1 GPU golden suite (_g2/run_gpu.py --stage correctness), same specs.
CASES = [
    ("I5_min_decode_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16", table_width=16, cache_pages=64)),
    ("I5_min_decode_fp8", dict(regime="decode", rows=4, requests=1, fmt="fp8", table_width=16, cache_pages=64)),
    ("I6_2req_bf16", dict(regime="prefill", rows=16, requests=2, fmt="bf16", table_width=32, cache_pages=128)),
    ("I6_2req_fp8", dict(regime="prefill", rows=16, requests=2, fmt="fp8", table_width=32, cache_pages=128)),
    ("I7_64rows_bf16", dict(regime="prefill", rows=64, requests=2, fmt="bf16", table_width=64, cache_pages=256)),
    ("I8_272rows_bf16", dict(regime="prefill", rows=272, requests=1, fmt="bf16", table_width=64, cache_pages=64)),
    ("tail_columns_1000_bf16", dict(regime="prefill", rows=8, requests=2, fmt="bf16", table_width=256,
                                    cache_pages=1024, max_columns=1000)),
    ("tail_columns_1000_fp8", dict(regime="prefill", rows=8, requests=2, fmt="fp8", table_width=256,
                                   cache_pages=1024, max_columns=1000)),
    ("edge_visible_zero_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16", table_width=16,
                                    cache_pages=64, patch="visible_zero")),
    ("edge_t2r_invalid_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16", table_width=16,
                                   cache_pages=64, patch="t2r_invalid")),
    ("edge_bad_page_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16", table_width=16,
                                cache_pages=64, patch="bad_page")),
    ("edge_bad_page_fp8", dict(regime="decode", rows=4, requests=1, fmt="fp8", table_width=16,
                               cache_pages=64, patch="bad_page")),
    ("edge_page_oob_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16", table_width=16,
                                cache_pages=64, patch="bad_page_oob")),
    ("edge_page_oob_fp8", dict(regime="decode", rows=4, requests=1, fmt="fp8", table_width=16,
                               cache_pages=64, patch="bad_page_oob")),
    ("edge_short_ctx_fp8", dict(regime="decode", rows=4, requests=1, fmt="fp8", table_width=16,
                                cache_pages=64, patch="short_ctx")),
    ("I1_e2e_decode_bf16", dict(regime="decode", rows=64, requests=16, fmt="bf16", table_width=W,
                                cache_pages=PAGES, max_ctx=CTX)),
    ("I2_e2e_decode_fp8", dict(regime="decode", rows=64, requests=16, fmt="fp8", table_width=W,
                               cache_pages=PAGES, max_ctx=CTX)),
    ("I3_e2e_draft_bf16", dict(regime="draft", rows=16, requests=16, fmt="bf16", table_width=W,
                               cache_pages=PAGES, max_ctx=CTX)),
    ("I3_e2e_draft_fp8", dict(regime="draft", rows=16, requests=16, fmt="fp8", table_width=W,
                              cache_pages=PAGES, max_ctx=CTX)),
    ("prefill_chunk512_bf16", dict(regime="prefill", rows=512, requests=2, fmt="bf16", table_width=W,
                                   cache_pages=PAGES, max_ctx=CTX)),
    ("prefill_chunk512_fp8", dict(regime="prefill", rows=512, requests=2, fmt="fp8", table_width=W,
                                  cache_pages=PAGES, max_ctx=CTX)),
    ("I14_decode_c32768_bf16", dict(regime="decode", rows=64, requests=16, fmt="bf16", table_width=W,
                                    cache_pages=PAGES, max_ctx=CTX, max_columns=32768)),
]


def fixture(spec, seed=0):
    max_ctx = spec.get("max_ctx", 16 * spec["table_width"])
    case = S.make_case("t", spec["regime"], rows=spec["rows"], requests=spec["requests"],
                       max_ctx=max_ctx, index_dtype=spec["fmt"], table_width=spec["table_width"],
                       cache_pages=spec["cache_pages"])
    d = G.build_case(case, device="cuda", seed=seed)
    d["max_columns"] = spec.get("max_columns", case.columns)
    what = spec.get("patch")
    if what == "visible_zero":
        d["query_positions"][0] = 2
        d["query_positions"][1] = 0
    elif what == "t2r_invalid":
        d["token_to_request"][0] = -1
        d["token_to_request"][1] = 999
    elif what == "bad_page":
        d["page_table"][0, 0] = -1
    elif what == "bad_page_oob":
        d["page_table"][0, 0] = 1 << 24
    elif what == "short_ctx":
        d["context_lens"][0] = 8
    return d


def args_of(d):
    return (d["q"], d["compressed_k_cache"], d["page_table"], d["token_to_request"],
            d["query_positions"], d["context_lens"])


def compare(out, vis, d, cols):
    gold, gvis, valid = G.reference_logits(*args_of(d), compress_ratio=4, max_columns=cols,
                                           compressed_k_scale=d["compressed_k_scale"])
    res = G.compare_logits(out, vis, gold, gvis, valid, rtol=RTOL, atol=ATOL)
    colix = torch.arange(cols, device=gold.device)
    inside = valid & (colix[None, :] < gvis.to(torch.int64)[:, None])
    oa, ob = out[~inside], gold[~inside]
    res["outside_diff"] = int((~((oa == ob) | (torch.isnan(oa) & torch.isnan(ob)))).sum())
    res["nan_kernel"] = int(torch.isnan(out).sum())
    res["pass"] = bool(res["pass"] and res["outside_diff"] == 0 and res["nan_kernel"] == 0)
    return res


def run_case(name, spec):
    t0 = time.time()
    d = fixture(spec)
    cols = int(d["max_columns"])
    rows = spec["rows"]
    snap = [t.clone() for t in args_of(d)] + (
        [d["compressed_k_scale"].clone()] if d["compressed_k_scale"] is not None else [])
    rs = torch.full((rows,), 7, dtype=torch.int32, device="cuda")
    lg = torch.full((rows, cols), 12345.0, dtype=torch.float32, device="cuda")
    vg = torch.full((rows,), -7, dtype=torch.int32, device="cuda")
    dec = API.decide(*args_of(d), 4, None, cols, rs, False, d["compressed_k_scale"],
                     logits=lg, visible_groups=vg, profile=PROFILE,
                     allow_unverified=ALLOW_UNVERIFIED)
    rec = {"name": name, "decision": dec.reason.value, "cell": dec.cell, "vid": dec.vid}
    if not dec.ok:
        rec.update(passed=False, error=f"not eligible: {dec.reason.value}: {dec.detail}")
        return rec
    out, vis = scorer(*args_of(d), 4, None, cols, rs, False,
                                        d["compressed_k_scale"], logits=lg, visible_groups=vg,
                                        profile=PROFILE)
    torch.cuda.synchronize()
    checks = {"outputs_in_place": out.data_ptr() == lg.data_ptr() and vis.data_ptr() == vg.data_ptr(),
              "row_starts_zero": bool((rs == 0).all()),
              "inputs_untouched": all(torch.equal(a, b) for a, b in zip(
                  snap, list(args_of(d)) + ([d["compressed_k_scale"]]
                                            if d["compressed_k_scale"] is not None else [])))}
    if spec.get("patch") == "bad_page_oob":
        seg = out[d["token_to_request"] == 0][:, :4]
        res = {"pass": bool(torch.isinf(seg).all() and (seg < 0).all() and not torch.isnan(out).any()),
               "contract": "page >= num_pages -> -inf in its 4 columns, no NaN"}
    else:
        res = compare(out, vis, d, cols)
    # int32 metadata (zero-copy path) must be bit-identical to int64 positions
    d32 = dict(d)
    for k in ("token_to_request", "query_positions", "context_lens", "page_table"):
        d32[k] = d[k].to(torch.int32)
    o32, v32 = scorer(*args_of(d32), 4, None, cols, None, False,
                                        d["compressed_k_scale"], profile=PROFILE)
    checks["int32_bitwise"] = bool(torch.equal(o32.view(torch.int32), out.view(torch.int32))
                                   and torch.equal(v32, vis))
    # CUDA-graph capture/replay == eager, bitwise
    gl = torch.empty_like(out)
    gv = torch.empty_like(vis)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        scorer(*args_of(d), 4, None, cols, None, False, d["compressed_k_scale"],
                                 logits=gl, visible_groups=gv, profile=PROFILE)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    gl.fill_(0)
    with torch.cuda.graph(g):
        scorer(*args_of(d), 4, None, cols, None, False, d["compressed_k_scale"],
                                 logits=gl, visible_groups=gv, profile=PROFILE)
    gl.fill_(0)
    g.replay()
    torch.cuda.synchronize()
    checks["graph_bitwise"] = bool(torch.equal(gl.view(torch.int32), out.view(torch.int32))
                                   and torch.equal(gv, vis))
    rec.update(golden=res, checks=checks, seconds=round(time.time() - t0, 2),
               passed=bool(res["pass"] and all(checks.values())))
    return rec


def run_profile_rows():
    out = []
    for row in API.P.load_profile(PROFILE).data["rows"]:
        status = row["golden"]["status"]
        if status != "pass" and not ALLOW_UNVERIFIED:
            continue
        rec = GEN.check_row(row, API, G, S, profile=PROFILE)
        out.append({"name": "profile:" + row["cell"], "passed": bool(rec.pop("pass_")),
                    "profile_status": status, **rec})
        torch.cuda.empty_cache()
    return out


def run_vs_atom():
    """Fused vs frozen ATOM native on identical inputs (prefill fp8: ATOM default aborts)."""
    import qsa_baseline as QB
    atom = QB.load_atom(os.path.join(GOLDEN_DIR, "atom_impl"))
    out = []
    for name, regime, rows, reqs, fmt in (("decode_bf16", "decode", 64, 16, "bf16"),
                                          ("decode_fp8", "decode", 64, 16, "fp8"),
                                          ("draft_bf16", "draft", 16, 16, "bf16"),
                                          ("draft_fp8", "draft", 16, 16, "fp8"),
                                          ("prefill512_bf16", "prefill", 512, 2, "bf16")):
        d = fixture(dict(regime=regime, rows=rows, requests=reqs, fmt=fmt, table_width=W,
                         cache_pages=PAGES, max_ctx=CTX))
        cols = int(d["max_columns"])
        ours, ov = scorer(*args_of(d), 4, None, cols, None, False,
                                            d["compressed_k_scale"], profile=PROFILE)
        ref, rv = QB.call_scorer(atom["fn"], d, cols, False, None)
        gold, gvis, valid = G.reference_logits(*args_of(d), compress_ratio=4, max_columns=cols,
                                               compressed_k_scale=d["compressed_k_scale"])
        res = G.compare_logits(ours, ov, ref, rv, valid, rtol=RTOL, atol=ATOL)
        out.append({"name": "vs_atom:" + name, "passed": bool(res["pass"]), "golden": res})
        torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("--json")
    ap.add_argument("--skip-atom", action="store_true")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="serve + check pending profile rows too (before `gen_profile check` restamps)")
    a = ap.parse_args(argv)
    global ALLOW_UNVERIFIED
    ALLOW_UNVERIFIED = bool(a.allow_unverified)
    props = torch.cuda.get_device_properties(0)
    if not str(props.gcnArchName).startswith("gfx1250"):
        print(f"SKIP: gfx1250 only (device {props.gcnArchName})")
        return 0
    recs = []
    for name, spec in CASES:
        if a.k and a.k not in name:
            continue
        print(f"[CASE] {name}", flush=True)
        try:
            rec = run_case(name, spec)
        except Exception as exc:  # noqa: BLE001
            rec = {"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        print(f"[{'PASS' if rec['passed'] else 'FAIL'}] {name} {json.dumps(rec)[:500]}", flush=True)
        recs.append(rec)
        torch.cuda.empty_cache()
    if not a.k:
        recs += run_profile_rows()
        if not a.skip_atom:
            recs += run_vs_atom()
    bad = [r["name"] for r in recs if not r["passed"]]
    summary = {"pass": not bad, "n": len(recs), "failed": bad, "profile": PROFILE, "allow_unverified": ALLOW_UNVERIFIED,
               "kernel_sha256": API._kernel_sha256(),
               "device": f"{props.gcnArchName} cu={props.multi_processor_count}", "records": recs}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(summary, fh, indent=1)
    print(f"[SUMMARY] {'PASS' if not bad else 'FAIL'} n={len(recs)} failed={bad}")
    return 0 if not bad else 1


def test_qsa_index_scorer_gpu():   # pytest entry
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main())
