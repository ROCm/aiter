#!/usr/bin/env python3
"""ROUND-PAD GPU check: padded/non-bucket shapes through the bucket kernel (run under gpu-lock).

Reuses run_gpu.run_case (fixtures + golden compare + visible check) with true-but-non-bucket shapes
(golden constraints: decode rows=requests*4, draft rows=requests, prefill rows arbitrary/2 requests).
Run:  QSA_PAD_BUCKET=1 python3 pad_gpu_check.py --tag pad_gpu
Prints one line per case and exits non-zero if any case fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QSA_PAD_BUCKET", "1")
import run_gpu as R  # noqa: E402

CASES = [
    # decode: rows = requests*4 (not a 16-row bucket), requests not a multiple of 4, odd column tails
    ("pad_dec_r12_q3_bf16_c1000", dict(regime="decode", rows=12, requests=3, fmt="bf16",
                                       table_width=256, cache_pages=2048, max_ctx=1024,
                                       max_columns=1000)),
    ("pad_dec_r12_q3_fp8_c1000", dict(regime="decode", rows=12, requests=3, fmt="fp8",
                                      table_width=256, cache_pages=2048, max_ctx=1024,
                                      max_columns=1000)),
    ("pad_dec_r20_q5_bf16_c257", dict(regime="decode", rows=20, requests=5, fmt="bf16",
                                      table_width=128, cache_pages=512, max_ctx=1024,
                                      max_columns=257)),
    # draft: rows = requests (1..5 rows)
    ("pad_draft_r3_q3_bf16_c67", dict(regime="draft", rows=3, requests=3, fmt="bf16",
                                      table_width=32, cache_pages=2048, max_ctx=512,
                                      max_columns=67)),
    ("pad_draft_r5_q5_fp8_c31", dict(regime="draft", rows=5, requests=5, fmt="fp8",
                                     table_width=32, cache_pages=2048, max_ctx=512,
                                     max_columns=31)),
    ("pad_draft_r1_q1_fp8_c5", dict(regime="draft", rows=1, requests=1, fmt="fp8",
                                    table_width=2, cache_pages=16, max_ctx=32, max_columns=5)),
    # prefill: arbitrary rows (split between 2 requests), odd columns
    ("pad_pref_r37_q2_bf16_c257", dict(regime="prefill", rows=37, requests=2, fmt="bf16",
                                       table_width=32, cache_pages=2048, max_ctx=512,
                                       max_columns=257)),
    ("pad_pref_r13_q2_fp8_c65", dict(regime="prefill", rows=13, requests=2, fmt="fp8",
                                     table_width=32, cache_pages=2048, max_ctx=512,
                                     max_columns=65)),
    ("pad_pref_r1_q2_bf16_c31", dict(regime="prefill", rows=1, requests=2, fmt="bf16",
                                     table_width=8, cache_pages=32, max_ctx=128, max_columns=31)),
    # single column / single row extremes
    ("pad_dec_r4_q1_bf16_c1", dict(regime="decode", rows=4, requests=1, fmt="bf16",
                                   table_width=1, cache_pages=16, max_ctx=16, max_columns=1)),
    # exact buckets through the padded dispatcher (must be overhead-only, byte identical)
    ("pad_exact_dec64_bf16", dict(regime="decode", rows=64, requests=16, fmt="bf16",
                                  table_width=16384, cache_pages=738964, max_ctx=262144)),
    ("pad_exact_dec64_fp8", dict(regime="decode", rows=64, requests=16, fmt="fp8",
                                 table_width=16384, cache_pages=738964, max_ctx=262144)),
    ("pad_exact_draft16_bf16", dict(regime="draft", rows=16, requests=16, fmt="bf16",
                                    table_width=16384, cache_pages=738964, max_ctx=262144)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pad_gpu")
    args = ap.parse_args()
    print(f"[START] pad_gpu_check tag={args.tag} pad_bucket={os.environ.get('QSA_PAD_BUCKET')}")
    print("[VERSIONS] " + json.dumps(R.versions(), default=str))
    records = []
    rc = 0
    for name, spec in CASES:
        rec, _ = R.run_case(name, spec)
        rec["pad_case"] = name
        records.append(rec)
        passed = bool(rec.get("pass", rec.get("passed")))
        if not passed:
            rc = 1
        print(f"[{'PASS' if passed else 'FAIL'}] {name:30s} "
              f"rows={rec.get('rows')} cols={rec.get('cols')} max_abs={rec.get('max_abs_err')} "
              f"bad={rec.get('bad')} visible_equal={rec.get('visible_equal')}", flush=True)
    import torch
    torch.cuda.empty_cache()
    out = os.path.join(os.environ.get("QSA_RESULTS", "/work/results/g2/cand"), f"pad_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"tag": args.tag, "pad_bucket": os.environ.get("QSA_PAD_BUCKET"),
                   "cases": records, "all_pass": rc == 0}, fh, indent=1, default=str)
    print(f"wrote {out} all_pass={rc == 0}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
