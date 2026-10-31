# SPDX-License-Identifier: MIT
"""GPU correctness + first-look latency driver for the page-4 QSA index scorer v0.

Compares `qsa_paged_mqa_logits_page4_gfx1250` against the **public golden**
(`golden_baseline/qsa_golden.py`), which is staged read-only on the node, then
runs an *exploratory, unprofiled* latency sweep.

    python run_gpu.py --stage correctness
    python run_gpu.py --stage timing
    python run_gpu.py --stage all

Writes JSON to $QSA_RESULTS (default /work/results/g2/builder).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.environ.get("QSA_GOLDEN_DIR", "/work/task/MI455_QSA_INDEX_SCORER/golden_baseline")
RESULTS = os.environ.get("QSA_RESULTS", "/work/results/g2/builder")
sys.path.insert(0, os.path.abspath(GOLDEN_DIR))

import qsa_golden as G  # noqa: E402
import qsa_shapes as S  # noqa: E402

sys.path.insert(0, _HERE)
from _load import load_kernel_module  # noqa: E402

K, K_IMPORT = load_kernel_module()

RTOL = 1e-4
ATOL = 1e-4
PAGE = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_fixture(regime, rows, requests, fmt, table_width, cache_pages, max_ctx=None,
                 seed=0, max_columns=None):
    if max_ctx is None:
        max_ctx = 16 * table_width
    case = S.make_case(f"{regime}_{rows}x{requests}_{fmt}_w{table_width}",
                       regime, rows=rows, requests=requests, max_ctx=max_ctx,
                       index_dtype=fmt, table_width=table_width, cache_pages=cache_pages)
    data = G.build_case(case, device="cuda", seed=seed)
    data["max_columns"] = case.columns if max_columns is None else max_columns
    return case, data


def patch(data, what):
    """Apply one edge-case mutation to a fixture's metadata."""
    d = dict(data)
    d["token_to_request"] = data["token_to_request"].clone()
    d["query_positions"] = data["query_positions"].clone()
    d["context_lens"] = data["context_lens"].clone()
    d["page_table"] = data["page_table"].clone()
    if what == "visible_zero":
        d["query_positions"][0] = 2
        d["query_positions"][1] = 0
    elif what == "t2r_invalid":
        d["token_to_request"][0] = -1
        d["token_to_request"][1] = 999
    elif what == "bad_page":
        d["page_table"][0, 0] = -1
    elif what == "bad_page_oob":
        # a high out-of-range page index: the public golden crashes on it
        # (clamp_min(0) only), so this case is checked against the contract
        # (invalid page -> -inf) instead of against the golden.
        d["page_table"][0, 0] = 1 << 24
    elif what == "short_ctx":
        d["context_lens"][0] = 8
    else:
        raise ValueError(what)
    return d


def golden_for(data, max_columns):
    return G.reference_logits(
        data["q"], data["compressed_k_cache"], data["page_table"],
        data["token_to_request"], data["query_positions"], data["context_lens"],
        compress_ratio=S.COMPRESS_RATIO, max_columns=max_columns,
        compressed_k_scale=data["compressed_k_scale"],
    )


def outside_stats(out, gold, valid_page, visible):
    """Counters outside the `valid_page & c < visible` mask and NaN/Inf mismatches."""
    cols = torch.arange(gold.shape[1], device=gold.device)
    inside = valid_page & (cols[None, :] < visible.to(torch.int64)[:, None])
    outside = ~inside
    a = out.float()
    b = gold.float()
    nan_out = int(torch.isnan(a).sum().item())
    nan_ref = int(torch.isnan(b).sum().item())
    oa, ob = a[outside], b[outside]
    same_bits = (oa == ob) | (torch.isnan(oa) & torch.isnan(ob))
    outside_diff = int((~same_bits).sum().item())
    inf_mismatch = int(((torch.isinf(oa) != torch.isinf(ob))).sum().item())
    return {
        "outside_count": int(outside.sum().item()),
        "outside_diff": outside_diff,
        "nan_kernel": nan_out,
        "nan_golden": nan_ref,
        "inf_mismatch_outside": inf_mismatch,
    }


def run_case(name, spec, do_compare=True):
    """spec: dict(regime, rows, requests, fmt, table_width, cache_pages, max_ctx, max_columns, patch)"""
    t0 = time.time()
    # Progress line BEFORE the first GPU touch of the case: the d07-2 smoke run
    # produced no output for 40 min because the first print came after a whole
    # case. Never again: print, flush, then work.
    print(f"[CASE] {name} start rows={spec['rows']} fmt={spec['fmt']} "
          f"width={spec['table_width']} pages={spec['cache_pages']}", flush=True)
    case, data = make_fixture(**{k: v for k, v in spec.items()
                                 if k in ("regime", "rows", "requests", "fmt", "table_width",
                                          "cache_pages", "max_ctx", "max_columns")})
    print(f"[CASE] {name} fixture+golden ready ({time.time() - t0:.1f}s)", flush=True)
    what = spec.get("patch")
    if what:
        data = patch(data, what)
    cols = data["max_columns"]
    rec = {"name": name, "spec": {**{k: v for k, v in spec.items()}, "columns": cols},
           "dtype": spec["fmt"], "rows": spec["rows"], "columns": int(cols)}
    try:
        print(f"[CASE] {name} dispatch", flush=True)
        out, out_vis = K.qsa_paged_mqa_logits_page4(
            data["q"], data["compressed_k_cache"], data["page_table"],
            data["token_to_request"], data["query_positions"], data["context_lens"],
            compress_ratio=PAGE, max_columns=cols,
            compressed_k_scale=data["compressed_k_scale"],
        )
        print(f"[CASE] {name} dispatched -> synchronize", flush=True)
        torch.cuda.synchronize()
        rec["ran"] = True
    except Exception as exc:  # noqa: BLE001
        rec["ran"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        rec["seconds"] = round(time.time() - t0, 2)
        print(f"[ERROR] {name}: {rec['error'][:600]}")
        return rec, None

    if not do_compare:
        rec["seconds"] = round(time.time() - t0, 2)
        return rec, data

    if what == "bad_page_oob":
        # Contract check only: page index past the pool -> those 4 columns are
        # -inf. (The public golden raises IndexError on such an entry.)
        seg = out[data["token_to_request"] == 0][:, :4]
        ok = bool(torch.isinf(seg).all() and (seg < 0).all()) and not bool(torch.isnan(out).any())
        rec["contract_check"] = "page>=num_pages -> -inf in its 4 columns, no NaN"
        rec["pass"] = ok
        rec["seconds"] = round(time.time() - t0, 2)
        print("[%s] %-28s rows=%4d cols=%6d %-4s contract=%s"
              % ("PASS" if ok else "FAIL", name, rec["rows"], cols, spec["fmt"],
                 rec["contract_check"]))
        return rec, data

    gold, gold_vis, valid = golden_for(data, cols)
    res = G.compare_logits(out, out_vis, gold, gold_vis, valid, rtol=RTOL, atol=ATOL)
    rec["golden"] = {k: v for k, v in res.items()}
    rec["outside"] = outside_stats(out, gold, valid, gold_vis)
    rec["pass"] = bool(res.get("pass"))
    rec["seconds"] = round(time.time() - t0, 2)
    print("[%s] %-28s rows=%4d cols=%6d %-4s checked=%9d max_abs=%.3e max_rel=%.3e "
          "bad=%d outside_diff=%d nan=%d/%d (%.1fs)"
          % ("PASS" if rec["pass"] else "FAIL", name, rec["rows"], cols, spec["fmt"],
             res.get("checked", 0), res.get("max_abs_err", float("nan")),
             res.get("max_rel_err", float("nan")), res.get("bad", -1),
             rec["outside"]["outside_diff"], rec["outside"]["nan_kernel"],
             rec["outside"]["nan_golden"], rec["seconds"]))
    if not res.get("visible_equal", True):
        print("       visible mismatch! kernel:", out_vis[:16].tolist(),
              "golden:", gold_vis[:16].tolist())
    if res.get("bad", 0):
        cols_ix = torch.arange(cols, device=out.device)
        m = valid & (cols_ix[None, :] < gold_vis.to(torch.int64)[:, None])
        diff = (out.float() - gold.float()).abs()
        diff = torch.where(m, diff, torch.zeros_like(diff))
        idx = int(diff.argmax().item())
        r, c = idx // cols, idx % cols
        print(f"       worst at row={r} col={c}: kernel={out[r, c].item():.6f} "
              f"golden={gold[r, c].item():.6f} vis={int(gold_vis[r])} "
              f"page={int(data['page_table'][int(data['token_to_request'][r]), c // 4])} off={c % 4}")
    del gold, gold_vis, valid
    return rec, data


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def correctness():
    cases = [
        ("I5_min_decode_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16",
                                    table_width=16, cache_pages=64)),
        ("I5_min_decode_fp8", dict(regime="decode", rows=4, requests=1, fmt="fp8",
                                   table_width=16, cache_pages=64)),
        ("I6_2req_bf16", dict(regime="prefill", rows=16, requests=2, fmt="bf16",
                              table_width=32, cache_pages=128)),
        ("I6_2req_fp8", dict(regime="prefill", rows=16, requests=2, fmt="fp8",
                             table_width=32, cache_pages=128)),
        ("I7_64rows_bf16", dict(regime="prefill", rows=64, requests=2, fmt="bf16",
                                table_width=64, cache_pages=256)),
        ("I8_272rows_bf16", dict(regime="prefill", rows=272, requests=1, fmt="bf16",
                                 table_width=64, cache_pages=64)),
        ("tail_columns_1000_bf16", dict(regime="prefill", rows=8, requests=2, fmt="bf16",
                                        table_width=256, cache_pages=1024, max_columns=1000)),
        ("tail_columns_1000_fp8", dict(regime="prefill", rows=8, requests=2, fmt="fp8",
                                       table_width=256, cache_pages=1024, max_columns=1000)),
        ("edge_visible_zero_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16",
                                        table_width=16, cache_pages=64, patch="visible_zero")),
        ("edge_t2r_invalid_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16",
                                       table_width=16, cache_pages=64, patch="t2r_invalid")),
        ("edge_bad_page_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16",
                                    table_width=16, cache_pages=64, patch="bad_page")),
        ("edge_bad_page_fp8", dict(regime="decode", rows=4, requests=1, fmt="fp8",
                                   table_width=16, cache_pages=64, patch="bad_page")),
        ("edge_page_oob_bf16", dict(regime="decode", rows=4, requests=1, fmt="bf16",
                                    table_width=16, cache_pages=64, patch="bad_page_oob")),
        ("edge_page_oob_fp8", dict(regime="decode", rows=4, requests=1, fmt="fp8",
                                   table_width=16, cache_pages=64, patch="bad_page_oob")),
        ("edge_short_ctx_fp8", dict(regime="decode", rows=4, requests=1, fmt="fp8",
                                    table_width=16, cache_pages=64, patch="short_ctx")),
        # e2e shapes (I1/I2/I3) — the full 738964-page pool and 16384-wide table
        ("I1_e2e_decode_bf16", dict(regime="decode", rows=64, requests=16, fmt="bf16",
                                    table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("I2_e2e_decode_fp8", dict(regime="decode", rows=64, requests=16, fmt="fp8",
                                   table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("I3_e2e_draft_bf16", dict(regime="draft", rows=16, requests=16, fmt="bf16",
                                   table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("I3_e2e_draft_fp8", dict(regime="draft", rows=16, requests=16, fmt="fp8",
                                  table_width=16384, cache_pages=738964, max_ctx=262144)),
        # prefill chunk: 512 rows x 65536
        ("prefill_chunk512_bf16", dict(regime="prefill", rows=512, requests=2, fmt="bf16",
                                       table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("prefill_chunk512_fp8", dict(regime="prefill", rows=512, requests=2, fmt="fp8",
                                      table_width=16384, cache_pages=738964, max_ctx=262144)),
        # columns truncation to the 32768 bucket (I14)
        ("I14_decode_c32768_bf16", dict(regime="decode", rows=64, requests=16, fmt="bf16",
                                        table_width=16384, cache_pages=738964,
                                        max_ctx=262144, max_columns=32768)),
    ]
    out = []
    for name, spec in cases:
        rec, _ = run_case(name, spec)
        out.append(rec)
        del rec
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# First look (exploratory, unprofiled)
# ---------------------------------------------------------------------------
def bench(fn, warmup=5, rep=20, flush_bytes=256 << 20):
    flush = torch.empty(flush_bytes // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = []
    for _ in range(rep):
        flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        evs.append((s, e))
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in evs)   # ms
    del flush
    return {"median_ns": ts[len(ts) // 2] * 1e6, "min_ns": ts[0] * 1e6,
            "max_ns": ts[-1] * 1e6, "rep": rep, "warmup": warmup,
            "l2_flush_bytes": flush_bytes}


def timing():
    buckets = [
        ("decode_bf16_c65536", dict(regime="decode", rows=64, requests=16, fmt="bf16",
                                    table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("decode_fp8_c65536", dict(regime="decode", rows=64, requests=16, fmt="fp8",
                                   table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("draft_bf16_c65536", dict(regime="draft", rows=16, requests=16, fmt="bf16",
                                   table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("draft_fp8_c65536", dict(regime="draft", rows=16, requests=16, fmt="fp8",
                                  table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("prefill512_bf16_c65536", dict(regime="prefill", rows=512, requests=2, fmt="bf16",
                                        table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("prefill512_fp8_c65536", dict(regime="prefill", rows=512, requests=2, fmt="fp8",
                                       table_width=16384, cache_pages=738964, max_ctx=262144)),
        ("decode_bf16_c32768", dict(regime="decode", rows=64, requests=16, fmt="bf16",
                                    table_width=16384, cache_pages=738964, max_ctx=262144,
                                    max_columns=32768)),
        ("decode_fp8_c32768", dict(regime="decode", rows=64, requests=16, fmt="fp8",
                                   table_width=16384, cache_pages=738964, max_ctx=262144,
                                   max_columns=32768)),
    ]
    # full prefill (2880 rows split into 512-row chunks) = sum of chunk calls, one layer
    out = []
    for name, spec in buckets:
        rec, data = run_case(name, spec, do_compare=False)
        if not rec.get("ran"):
            rec["bucket"] = name
            out.append(rec)
            torch.cuda.empty_cache()
            continue
        cols = data["max_columns"]
        fn = lambda: K.qsa_paged_mqa_logits_page4(  # noqa: B023, E731
            data["q"], data["compressed_k_cache"], data["page_table"],
            data["token_to_request"], data["query_positions"], data["context_lens"],
            compress_ratio=PAGE, max_columns=cols,
            compressed_k_scale=data["compressed_k_scale"],
        )
        stats = bench(fn)
        rec.update(stats)
        rec["bucket"] = name
        nbytes = modeled_route_bytes(data, cols)
        rec["modeled_route_bytes"] = nbytes
        rec["bw_eff_GBps"] = nbytes / rec["median_ns"] if rec["median_ns"] else None
        print("[first-look] %-24s median=%9.1f us  min=%9.1f us  %.2f TB/s (exploratory)"
              % (name, rec["median_ns"] / 1e3, rec["min_ns"] / 1e3,
                 (rec["bw_eff_GBps"] or 0) / 1e3))
        out.append(rec)
        del data
        torch.cuda.empty_cache()
    return out


def modeled_route_bytes(data, columns):
    """Q + K (visible window) + logits write, per the shape-extraction model."""
    q = data["q"].numel() * data["q"].element_size()
    kb = data["compressed_k_cache"].numel() * data["compressed_k_cache"].element_size()
    sb = data["compressed_k_scale"].numel() * 4 if data["compressed_k_scale"] is not None else 0
    t2r = data["token_to_request"].to(torch.int64)
    cl = data["context_lens"].to(torch.int64)
    pos = data["query_positions"].to(torch.int64)
    ok = (t2r >= 0) & (t2r < cl.numel())
    vis = torch.minimum((pos + 1) // S.COMPRESS_RATIO, cl[t2r.clamp(0, cl.numel() - 1)] // S.COMPRESS_RATIO).clamp_min(0)
    vis = torch.where(ok, vis, torch.zeros_like(vis))
    vis = torch.minimum(vis, torch.tensor(columns, dtype=torch.int64))
    pages = data["page_table"].shape[0]
    unique_pages = torch.unique(data["page_table"].to(torch.int64).clamp_min(0)).numel()
    key_pages = min(int(unique_pages), pages)
    # each request reads vis/4 pages; pages are shared across rows of one request
    reqs = torch.zeros(pages, dtype=torch.bool)
    # approximate: one read per (request, visible page) with 4 rows/req sharing
    n_req = int(cl.numel())
    tot_pages = 0
    for r in range(n_req):
        rows_r = (t2r == r).nonzero().flatten()
        if rows_r.numel() == 0:
            continue
        v = int(vis[rows_r].max().item())
        tot_pages += (v + 3) // 4
    kbytes = tot_pages * (kb // max(1, data["compressed_k_cache"].shape[0])) + 0
    return int(q + kbytes + columns * data["q"].shape[0] * 4 + 0)


# ---------------------------------------------------------------------------
def versions():
    import flydsl
    info = {
        "flydsl": getattr(flydsl, "__version__", "?"),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    try:
        info["hip"] = torch.version.hip
    except Exception:  # noqa: BLE001
        pass
    try:
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["gpu_arch"] = getattr(p, "gcnArchName", None)
        info["cu_count"] = p.multi_processor_count
        info["total_mem_GB"] = round(p.total_memory / 1e9, 1)
    except Exception as exc:  # noqa: BLE001
        info["gpu_error"] = str(exc)
    return info


def hsaco_hashes(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith((".hsaco", ".co", ".s")):
                p = os.path.join(dirpath, f)
                with open(p, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                out.append({"path": p, "bytes": os.path.getsize(p), "sha256": h})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["first_touch", "smoke", "correctness", "timing", "all"])
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.stage == "first_touch":
        # The ladder is its own driver: every step in its own process with a hard
        # timeout (see FIRST_TOUCH.md).  Do not fold it into this script.
        import first_touch
        return first_touch.main(["--timeout", "60",
                                 "--out", os.path.join(RESULTS, "first_touch_" + args.tag)])
    print(f"[START] run_gpu stage={args.stage} tag={args.tag} results={RESULTS}", flush=True)
    res = {"tag": args.tag, "stage": args.stage, "versions": versions(), "rtol": RTOL,
           "atol": ATOL, "kernel_import": K_IMPORT}
    print("[VERSIONS] " + json.dumps(res["versions"], default=str), flush=True)
    print("[KERNEL]   " + str(K_IMPORT), flush=True)
    if args.stage == "smoke":
        res["correctness"] = [
            run_case("I5_min_decode_bf16", dict(regime="decode", rows=4, requests=1,
                                                fmt="bf16", table_width=16, cache_pages=64))[0],
            run_case("I5_min_decode_fp8", dict(regime="decode", rows=4, requests=1,
                                               fmt="fp8", table_width=16, cache_pages=64))[0],
        ]
        res["correctness_pass"] = all(c.get("pass") for c in res["correctness"])
    if args.stage in ("correctness", "all"):
        res["correctness"] = correctness()
        res["correctness_pass"] = all(c.get("pass") for c in res["correctness"])
    if args.stage in ("timing", "all"):
        res["first_look"] = timing()
    res["hsaco"] = hsaco_hashes(os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "/nonexistent"))
    path = os.path.join(RESULTS, f"gpu_{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print("wrote", path)
    return 0 if res.get("correctness_pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
