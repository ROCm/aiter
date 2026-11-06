# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023 -- timing lambdas are called inside the loop iteration that defines them
"""R2 GPU check of the op's token-dimension buckets (gfx1250, under gpu-lock).

For MTP verify bench3 (rows = 3C, 3 rows per request) and tail-band rows (rows % 4 != 0):
bucket path (``decide(buckets=True)`` + ``launch``) vs the unpadded op path:

* bitwise: real rows of the bucket path == the unpadded call (reported per case; the
  unpadded call may route to another variant, so a mismatch is reported, not fatal) and
  golden (frozen ``qsa_golden.reference_logits`` / ``compare_logits``, rtol=atol=1e-4);
* caller buffers: guard rows around caller-provided outputs stay untouched;
* device time (CUDA-graph replay, 50 replays, 4 rotated inputs not needed: same inputs)
  of both paths.

Pending buckets/rules are allowed (``allow_unverified=True``): this is the check run
before ``gen_profile check --only buckets`` restamps them.

    QSA_GOLDEN_DIR=/path/golden_baseline gpu-lock run -- python3 \
        op_tests/op_benchmarks/qsa_index_scorer/bucket_qsa_index_scorer.py [--page 4] --json out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def load_api():
    d = os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer")
    spec = importlib.util.spec_from_file_location(
        "qsa_isc_bk", os.path.join(d, "__init__.py"), submodule_search_locations=[d]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["qsa_isc_bk"] = pkg
    spec.loader.exec_module(pkg)
    return pkg, sys.modules["qsa_isc_bk.api"]


def graph_us(fn, reps=50):
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / reps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    import torch

    sys.path.insert(0, os.environ["QSA_GOLDEN_DIR"])
    import qsa_golden as G
    import qsa_shapes as S

    pkg, api = load_api()
    gen_path = os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer", "gen_profile.py")
    sys.path.insert(0, os.path.dirname(gen_path))
    spec = importlib.util.spec_from_file_location("qsa_bk_gen", gen_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    prof = pkg.DEFAULT_PROFILE
    page = a.page
    cases = []
    for ctx in (4096, 16384, 65536):
        cols, w = ctx // 4, ctx // (4 * page)
        for c in (1, 8, 16, 32, 64, 128, 192, 256, 384, 448):
            cases.append(("verify3", 3 * c, c, cols, w))
        for rows, reqs in ((87, 16), (501, 32), (6, 1)):
            cases.append(("tail", rows, reqs, cols, w))
    out = []
    for kind, rows, reqs, cols, w in cases:
        fx = gen.sample_fixture(
            {
                "rows": rows,
                "requests": reqs,
                "columns": cols,
                "table_width": w,
                "page_size": page,
            }
        )
        case = S.make_case(
            f"bk_{kind}_{rows}_{reqs}_{cols}",
            "prefill",
            rows=rows,
            requests=reqs,
            max_ctx=fx["max_ctx"],
            index_dtype="bf16",
            table_width=w,
            cache_pages=fx["cache_pages"],
            page_size=page,
        )
        d = G.build_case(case, device="cuda", seed=0)
        t2r, pos = gen._even_layout(rows, reqs, fx["max_ctx"])
        d["token_to_request"] = t2r.to(device="cuda", dtype=d["token_to_request"].dtype)
        d["query_positions"] = pos.to(device="cuda", dtype=d["query_positions"].dtype)
        args = [
            d["q"],
            d["compressed_k_cache"],
            d["page_table"],
            d["token_to_request"],
            d["query_positions"],
            d["context_lens"],
            4,
            None,
            cols,
            None,
            False,
            d["compressed_k_scale"],
        ]
        rec = {
            "kind": kind,
            "rows": rows,
            "requests": reqs,
            "columns": cols,
            "table_width": w,
            "page": page,
        }
        db = api.decide(*args, profile=prof, allow_unverified=True, buckets=True)
        du = api.decide(*args, profile=prof, allow_unverified=True, buckets=False)
        rec.update(
            bucket_cell=db.cell,
            bucket_ok=db.ok,
            unpadded_cell=du.cell,
            unpadded_ok=du.ok,
        )
        if not (db.ok and du.ok and str(db.cell).startswith("bucket:")):
            rec["skip"] = f"{db.reason.value}/{du.reason.value}"
            out.append(rec)
            print(json.dumps(rec), flush=True)
            continue
        big = torch.full((rows + 2, cols), 12345.0, device="cuda")
        bigv = torch.full((rows + 2,), -7, dtype=torch.int32, device="cuda")
        lg, vg = big[1 : rows + 1], bigv[1 : rows + 1]
        lb, vb = api.launch(db, *args, logits=lg, visible_groups=vg)
        lu, vu = api.launch(du, *args)
        torch.cuda.synchronize()
        gold, gvis, valid = G.reference_logits(
            *args[:6], compress_ratio=4, max_columns=cols, compressed_k_scale=None
        )
        res = G.compare_logits(lb, vb, gold, gvis, valid, rtol=1e-4, atol=1e-4)
        rec.update(
            golden_pass=bool(res.get("pass")),
            bitwise_unpadded=bool(
                torch.equal(vb, vu)
                and ((lb == lu) | (torch.isnan(lb) & torch.isnan(lu))).all()
            ),
            guards_intact=bool(
                (big[0] == 12345.0).all()
                and (big[-1] == 12345.0).all()
                and (bigv[0] == -7)
                and (bigv[-1] == -7)
            ),
        )
        rec["bucket_us"] = round(
            graph_us(lambda: api.launch(db, *args, logits=lg, visible_groups=vg)), 2
        )
        lu2 = torch.empty_like(lu)
        vu2 = torch.empty_like(vu)
        rec["unpadded_us"] = round(
            graph_us(lambda: api.launch(du, *args, logits=lu2, visible_groups=vu2)), 2
        )
        rec["ratio"] = round(rec["bucket_us"] / rec["unpadded_us"], 3)
        out.append(rec)
        print(json.dumps(rec), flush=True)
        del d, big, lb, lu, gold
        torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    bad = [
        r
        for r in out
        if "skip" not in r and not (r["golden_pass"] and r["guards_intact"])
    ]
    print(f"[BUCKET] cases={len(out)} bad={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
