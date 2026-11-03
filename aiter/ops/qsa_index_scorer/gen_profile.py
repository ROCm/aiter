# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate / verify / golden-check a QSA index scorer profile.

Modes::

    # CPU: (re)write the profile from the sweep plan (+ Phase-1 evidence)
    python -m aiter.ops.qsa_index_scorer.gen_profile emit  [--plan-file captured.json] [--out X.json]
    # CPU: assert the committed profile equals what `emit` would write (drift guard)
    python -m aiter.ops.qsa_index_scorer.gen_profile verify [--profile X.json]
    # GPU: run every row through the public API vs the frozen golden; write status back
    QSA_GOLDEN_DIR=/path/golden_baseline \
    python -m aiter.ops.qsa_index_scorer.gen_profile check [--profile X.json] [--cells a,b] [--out Y.json]

The golden (``qsa_golden.py`` / ``qsa_shapes.py``) is read from ``$QSA_GOLDEN_DIR``
and is never modified. The comparator and tolerance are the frozen Phase-1 ones:
``qsa_golden.compare_logits(rtol=1e-4, atol=1e-4)`` over ``valid_page & c < visible``,
``visible_groups`` exact, ``row_starts`` all zero, no NaN, and every element
outside the mask bit-identical to the golden's ``-inf``.

A ``--plan-file`` is a JSON list of captured call classes
``{"cache_dtype", "rows", "requests", "columns", "table_width"[, "cache_pages", "max_ctx"]}``
(e.g. from the ATOM capture hook); new rows start as ``pending`` until ``check``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from typing import Any

if __package__ in (None, ""):  # executed as a file: make the sibling module importable
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import profiles as P  # type: ignore  # noqa: E402
else:
    from . import profiles as P

TOOL = "aiter.ops.qsa_index_scorer.gen_profile"
TOOL_VERSION = 1
GOLDEN_REVISION = "4b8888907e2bd3530bfde10eb081bf8cf60d49749aea4cac62ec486211b12ff8"
GOLDEN_RTOL = 1e-4
GOLDEN_ATOL = 1e-4
KERNEL_GIT_REV = "a78c285c"
E2E_WIDTH = 16384
E2E_PAGES = 738964
ARCH, CU = "gfx1250", 256

#: The builtin sweep plan: target buckets (integration brief section 5) + the
#: Phase-1 GPU golden suite shapes. (cache_dtype, rows, requests, columns,
#: table_width, cache_pages, max_ctx, cell prefix)
def builtin_plan() -> list[dict]:
    plan = []
    for dt in P.CACHE_DTYPES:
        for cols in (65536, 32768):
            plan.append(dict(cache_dtype=dt, rows=64, requests=16, columns=cols))
            plan.append(dict(cache_dtype=dt, rows=16, requests=16, columns=cols))
        # prefill 2880 rows / 2 requests at c65536 = 5 x 512-row chunks + one 320-row chunk
        plan.append(dict(cache_dtype=dt, rows=512, requests=2, columns=65536))
        plan.append(dict(cache_dtype=dt, rows=320, requests=2, columns=65536))
    for p in plan:
        p.update(table_width=E2E_WIDTH, cache_pages=E2E_PAGES, max_ctx=16 * E2E_WIDTH)
    small = [  # Phase-1 run_gpu.py correctness shapes (I5/I6/I7/I8/tail)
        dict(cache_dtype="bf16", rows=4, requests=1, columns=64, table_width=16, cache_pages=64),
        dict(cache_dtype="fp8", rows=4, requests=1, columns=64, table_width=16, cache_pages=64),
        dict(cache_dtype="bf16", rows=16, requests=2, columns=128, table_width=32, cache_pages=128),
        dict(cache_dtype="fp8", rows=16, requests=2, columns=128, table_width=32, cache_pages=128),
        dict(cache_dtype="bf16", rows=64, requests=2, columns=256, table_width=64, cache_pages=256),
        dict(cache_dtype="bf16", rows=272, requests=1, columns=256, table_width=64, cache_pages=64),
        dict(cache_dtype="bf16", rows=8, requests=2, columns=1000, table_width=256, cache_pages=1024),
        dict(cache_dtype="fp8", rows=8, requests=2, columns=1000, table_width=256, cache_pages=1024),
    ]
    for s in small:
        s.update(max_ctx=16 * s["table_width"])
    return plan + small


def cell_name(e: dict) -> str:
    reg = P.regime_of(e["rows"], e["requests"])
    return (f"{reg}_{e['cache_dtype']}_r{e['rows']}q{e['requests']}"
            f"_c{e['columns']}_w{e['table_width']}")


#: Phase-1 GPU golden evidence for the champion kernel (a78c285c commit body:
#: "GPU (a08-1 GPU3, gpu-lock + idle gate): correctness 22/22 default").
#: cell -> run_gpu.py correctness case names that exercised exactly that row.
PHASE1_EVIDENCE = {
    "decode_bf16_r64q16_c65536_w16384": ["I1_e2e_decode_bf16"],
    "decode_fp8_r64q16_c65536_w16384": ["I2_e2e_decode_fp8"],
    "draft_bf16_r16q16_c65536_w16384": ["I3_e2e_draft_bf16"],
    "draft_fp8_r16q16_c65536_w16384": ["I3_e2e_draft_fp8"],
    "prefill_bf16_r512q2_c65536_w16384": ["prefill_chunk512_bf16"],
    "prefill_fp8_r512q2_c65536_w16384": ["prefill_chunk512_fp8"],
    "decode_bf16_r64q16_c32768_w16384": ["I14_decode_c32768_bf16"],
    "decode_bf16_r4q1_c64_w16": ["I5_min_decode_bf16", "edge_visible_zero_bf16",
                                 "edge_t2r_invalid_bf16", "edge_bad_page_bf16",
                                 "edge_page_oob_bf16"],
    "decode_fp8_r4q1_c64_w16": ["I5_min_decode_fp8", "edge_bad_page_fp8", "edge_page_oob_fp8",
                                "edge_short_ctx_fp8"],
    "decode_bf16_r16q2_c128_w32": ["I6_2req_bf16"],
    "decode_fp8_r16q2_c128_w32": ["I6_2req_fp8"],
    "prefill_bf16_r64q2_c256_w64": ["I7_64rows_bf16"],
    "prefill_bf16_r272q1_c256_w64": ["I8_272rows_bf16"],
    "decode_bf16_r8q2_c1000_w256": ["tail_columns_1000_bf16"],
    "decode_fp8_r8q2_c1000_w256": ["tail_columns_1000_fp8"],
}
PHASE1_SOURCE = ("phase1: _g2/run_gpu.py --stage correctness on kernel a78c285c "
                 "(a08-1 GPU3, root, 2026-11-03, 22/22 PASS default env)")


def make_row(e: dict, evidence: dict | None = None) -> dict:
    regime = P.regime_of(e["rows"], e["requests"])
    if regime is None:
        raise ValueError(f"{e}: no profiled regime for rows={e['rows']} requests={e['requests']}")
    vid, params = P.expected_variant(e["cache_dtype"], regime, e["columns"])
    cell = cell_name(e)
    row: dict[str, Any] = {
        "sig": "",
        "arch": e.get("arch", ARCH), "cu": int(e.get("cu", CU)),
        "cache_dtype": e["cache_dtype"], "regime": regime,
        "rows": {"min": int(e["rows"]), "max": int(e["rows"])},
        "requests": {"min": int(e["requests"]), "max": int(e["requests"])},
        "columns": int(e["columns"]), "table_width": int(e["table_width"]),
        "vid": vid, "params": params, "cell": cell,
        "fixture": {"cache_pages": int(e.get("cache_pages", E2E_PAGES)),
                    "max_ctx": int(e.get("max_ctx", 16 * e["table_width"]))},
        "golden": {"status": "pending"},
        "generated_by": f"{TOOL}@{TOOL_VERSION}",
        "notes": e.get("notes", ""),
    }
    if evidence and cell in evidence:
        row["golden"] = {"status": "pass", "source": PHASE1_SOURCE, "cases": evidence[cell],
                         "comparator": "qsa_golden.compare_logits",
                         "rtol": GOLDEN_RTOL, "atol": GOLDEN_ATOL}
    elif not row["notes"]:
        row["notes"] = "not yet golden-checked on GPU for this kernel: run `gen_profile check`"
    row["sig"] = P.row_signature(row)
    return row


def _kernel_sha() -> str:
    with open(P.KERNEL_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build_profile(plan: list[dict], name: str = P.DEFAULT_PROFILE,
                  evidence: dict | None = PHASE1_EVIDENCE) -> dict:
    rows, seen = [], set()
    for e in plan:
        r = make_row(e, evidence)
        if r["sig"] in seen:
            continue
        seen.add(r["sig"])
        rows.append(r)
    return {
        "schema_version": P.SCHEMA_VERSION,
        "feature": P.FEATURE,
        "profile": name,
        "api_version": P.API_VERSION,
        "backend": P.BACKEND,
        "kernel": {"module": P.KERNEL_MODULE, "path": P.KERNEL_REL, "sha256": _kernel_sha(),
                   "git_rev": KERNEL_GIT_REV, "toolchain": "flydsl 0.3.3 (gfx1250-atom-20261015)"},
        "kernel_env": dict(P.KERNEL_ENV_DEFAULTS),
        "contract": {"index_heads": P.INDEX_HEADS, "head_dim": P.HEAD_DIM,
                     "page_size": P.PAGE_SIZE, "compress_ratio": P.COMPRESS_RATIO,
                     "q_dtype": "bf16", "score_divisor": "sqrt(head_dim)",
                     "cache_align_bytes": P.CACHE_ALIGN_BYTES,
                     "metadata_dtypes": ["int32", "int64"]},
        "golden": {"revision": GOLDEN_REVISION, "module": "qsa_golden.py",
                   "comparator": "compare_logits", "rtol": GOLDEN_RTOL, "atol": GOLDEN_ATOL,
                   "extra_checks": ["visible_groups exact", "row_starts == 0", "no NaN",
                                    "outside-mask bit-identical (-inf)"]},
        "generated_by": {"tool": TOOL, "version": TOOL_VERSION, "mode": "emit",
                         "plan": "builtin (target buckets + Phase-1 GPU suite shapes)",
                         "evidence": PHASE1_SOURCE,
                         "note": "exact rows only; pending rows are never selected at runtime"},
        "rows": rows,
    }


def _dump(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _strip_dynamic(data: dict) -> dict:
    """Fields `check` may legitimately update (golden status/stats, generated_by)."""
    out = json.loads(json.dumps(data))
    out.pop("generated_by", None)
    for r in out["rows"]:
        r.pop("golden", None)
        r.pop("notes", None)
    return out


def verify(path: str, plan: list[dict] | None = None) -> list[str]:
    """Differences between the profile file and a fresh `emit` (empty = in sync)."""
    with open(path, encoding="utf-8") as fh:
        have = json.load(fh)
    want = build_profile(plan or builtin_plan(), have.get("profile", P.DEFAULT_PROFILE))
    a, b = _strip_dynamic(have), _strip_dynamic(want)
    errs = []
    for key in sorted(set(a) | set(b)):
        if key != "rows" and a.get(key) != b.get(key):
            errs.append(f"top-level {key}: {a.get(key)!r} != {b.get(key)!r}")
    ra = {r["sig"]: r for r in a.get("rows", [])}
    rb = {r["sig"]: r for r in b.get("rows", [])}
    for sig in sorted(set(ra) - set(rb)):
        errs.append(f"row {ra[sig]['cell']} not produced by the plan")
    for sig in sorted(set(rb) - set(ra)):
        errs.append(f"row {rb[sig]['cell']} missing from the file")
    for sig in sorted(set(ra) & set(rb)):
        if ra[sig] != rb[sig]:
            errs.append(f"row {ra[sig]['cell']} differs")
    for r in have.get("rows", []):
        st = r["golden"]["status"]
        if st == "pass" and not (r["golden"].get("cases") or r["golden"].get("checked")):
            errs.append(f"row {r['cell']} is 'pass' without evidence")
    return errs


# ---------------------------------------------------------------------------
# GPU golden check (root runs this; never on a no-device host)
# ---------------------------------------------------------------------------
def _golden_modules():
    gdir = os.environ.get("QSA_GOLDEN_DIR")
    if not gdir:
        raise SystemExit("set QSA_GOLDEN_DIR to the frozen golden_baseline directory")
    sys.path.insert(0, os.path.abspath(gdir))
    import qsa_golden as G  # noqa: E402
    import qsa_shapes as S  # noqa: E402
    return G, S


def golden_regime(rows: int, requests: int) -> str:
    """Fixture layout of qsa_golden.build_case for a row shape."""
    if rows == requests:
        return "draft"
    if rows == 4 * requests:
        return "decode"
    return "prefill"   # two-way split of the rows (qsa_golden._row_layout)


def check_row(row: dict, api, G, S, seed: int = 0, profile: str | None = None) -> dict:
    import torch

    profile = profile or os.environ.get("QSA_PROFILE_PATH") or P.DEFAULT_PROFILE

    rows, reqs = row["rows"]["min"], row["requests"]["min"]
    fx = row.get("fixture", {})
    width, cols = row["table_width"], row["columns"]
    case = S.make_case(row["cell"], golden_regime(rows, reqs), rows=rows, requests=reqs,
                       max_ctx=int(fx.get("max_ctx", 16 * width)),
                       index_dtype=row["cache_dtype"], table_width=width,
                       cache_pages=int(fx.get("cache_pages", E2E_PAGES)))
    d = G.build_case(case, device="cuda", seed=seed)
    rs = torch.full((rows,), 7, dtype=torch.int32, device="cuda")
    args = (d["q"], d["compressed_k_cache"], d["page_table"], d["token_to_request"],
            d["query_positions"], d["context_lens"])
    dec = api.decide(*args, 4, None, cols, rs, False, d["compressed_k_scale"],
                     profile=profile, allow_unverified=True)
    rec: dict[str, Any] = {"decision": dec.reason.value, "vid": dec.vid}
    if not dec.ok:
        rec.update(pass_=False, error=f"not eligible: {dec.reason.value}: {dec.detail}")
        return rec
    out, d2 = api.try_qsa_paged_mqa_logits(*args, 4, None, cols, rs, False,
                                           d["compressed_k_scale"],
                                           profile=profile, allow_unverified=True)
    logits, vis = out
    torch.cuda.synchronize()
    gold, gvis, valid = G.reference_logits(*args, compress_ratio=4, max_columns=cols,
                                           compressed_k_scale=d["compressed_k_scale"])
    res = G.compare_logits(logits, vis, gold, gvis, valid, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL)
    colix = torch.arange(cols, device=gold.device)
    inside = valid & (colix[None, :] < gvis.to(torch.int64)[:, None])
    oa, ob = logits[~inside], gold[~inside]
    outside_diff = int((~((oa == ob) | (torch.isnan(oa) & torch.isnan(ob)))).sum())
    nan = int(torch.isnan(logits).sum())
    rs_zero = bool((rs == 0).all())
    ok = bool(res.get("pass")) and outside_diff == 0 and nan == 0 and rs_zero
    rec.update(pass_=ok, golden=res, outside_diff=outside_diff, nan=nan, row_starts_zero=rs_zero)
    return rec


def check(path: str, out: str | None, cells: set | None) -> int:
    import torch
    if __package__ in (None, ""):
        raise SystemExit("run `check` as a module: python -m aiter.ops.qsa_index_scorer.gen_profile")
    from . import api

    G, S = _golden_modules()
    os.environ["QSA_PROFILE_PATH"] = os.path.abspath(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    dev = torch.cuda.get_device_properties(0)
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    fails = 0
    for row in data["rows"]:
        if cells and row["cell"] not in cells:
            continue
        print(f"[CHECK] {row['cell']} ...", flush=True)
        try:
            rec = check_row(row, api, G, S)
        except Exception as exc:  # noqa: BLE001
            rec = {"pass_": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        ok = rec.pop("pass_")
        fails += not ok
        prev = row["golden"]
        row["golden"] = {"status": "pass" if ok else "fail", "checked": rec,
                         "checked_at": stamp, "device": f"{dev.gcnArchName} cu={dev.multi_processor_count}",
                         "kernel_sha256": data["kernel"]["sha256"], "golden_revision": GOLDEN_REVISION,
                         "comparator": "qsa_golden.compare_logits",
                         "rtol": GOLDEN_RTOL, "atol": GOLDEN_ATOL}
        if prev.get("cases"):
            row["golden"]["phase1_cases"] = prev["cases"]
        print(f"[{'PASS' if ok else 'FAIL'}] {row['cell']} {json.dumps(rec)[:400]}", flush=True)
        torch.cuda.empty_cache()
    data["generated_by"] = dict(data["generated_by"], mode="check", checked_at=stamp)
    _dump(data, out or path)
    print(f"[CHECK] fails={fails} -> {out or path}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("emit", "verify", "check"))
    ap.add_argument("--profile", default=os.path.join(P.CONFIG_DIR, f"{P.DEFAULT_PROFILE}.json"))
    ap.add_argument("--out")
    ap.add_argument("--plan-file", help="JSON list of captured call classes to add as rows")
    ap.add_argument("--cells", help="comma-separated cells (check mode)")
    a = ap.parse_args(argv)
    plan = builtin_plan()
    if a.plan_file:
        with open(a.plan_file, encoding="utf-8") as fh:
            plan += json.load(fh)
    if a.mode == "emit":
        out = a.out or a.profile
        data = build_profile(plan, os.path.splitext(os.path.basename(out))[0])
        _dump(data, out)
        print(f"wrote {out}: {len(data['rows'])} rows "
              f"({sum(r['golden']['status'] == 'pass' for r in data['rows'])} pass)")
        return 0
    if a.mode == "verify":
        errs = verify(a.profile, plan)
        for e in errs:
            print("DRIFT", e)
        print("VERIFY", "PASS" if not errs else "FAIL")
        return 1 if errs else 0
    return check(a.profile, a.out, set(a.cells.split(",")) if a.cells else None)


if __name__ == "__main__":
    sys.exit(main())
