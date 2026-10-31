# SPDX-License-Identifier: MIT
"""CPU-only contract test for the page-4 QSA index-scorer kernel.

No GPU, no flydsl: this mirrors the kernel's index / mask / dequant math in
plain torch (one thread per output column, same clamps) and compares it with the
**public golden** (`golden_baseline/qsa_golden.py`) on small deterministic cases.
It also exercises the wrapper's host-side contract checks (the rejection paths
run before any CUDA call).

The case list is the *fault-hypothesis* list from the d07-2 post-mortem
(`records/g2/builder2-log.md`): column tails (`columns % 4 != 0`, non-multiples
of the 256-thread block), `page = -1` / `page >= num_pages`, `token_to_request =
-1` / out of range, `visible = 0`, single-page tables, and the fp8 scale plane
(scale = 0, 1, >1).  Address bounds themselves are audited separately by
`audit_addressing_cpu.py`.

    python test_contract_cpu.py            # -> exit 0 on PASS
"""

from __future__ import annotations

import math
import os
import struct
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.environ.get("QSA_GOLDEN_DIR", os.path.join(_HERE, "..", "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.abspath(_GOLDEN))

import qsa_golden as G  # noqa: E402
import qsa_shapes as S  # noqa: E402

PAGE = 4
HEADS = 4
HDIM = 128


# ---------------------------------------------------------------------------
# Emulation of the kernel (same clamps, same order of operations)
# ---------------------------------------------------------------------------
def emulate_page4(q, cache, scale, table, t2r, qpos, ctx, max_columns=None):
    """logits [rows, cols] fp32 and visible [rows] int32, exactly as the kernel."""
    rows = q.shape[0]
    num_pages, P, _, D = cache.shape
    assert P == PAGE and D == HDIM
    num_requests, width = table.shape
    columns = width * PAGE if max_columns is None else max(1, min(width * PAGE, int(max_columns)))
    is_fp8 = cache.dtype == torch.float8_e4m3fn

    keys = cache.to(torch.float32)      # NOT pre-scaled: the kernel scales the dot
    out = torch.full((rows, columns), float("-inf"), dtype=torch.float32)
    vis_out = torch.zeros(rows, dtype=torch.int32)
    for row in range(rows):
        req = int(t2r[row])
        req_ok = 0 <= req < num_requests
        safe_req = req if req_ok else 0
        cl = int(ctx[safe_req])
        pos = int(qpos[row])
        head = max(min(pos + 1, cl), 0)          # min/max commute with //4
        vis = head // PAGE
        vis = vis if req_ok else 0
        vis_out[row] = vis
        for col in range(columns):
            logical = col // PAGE
            off = col % PAGE
            logical_ok = logical < width
            phys = int(table[safe_req, logical if logical_ok else 0])
            page_ok = req_ok and logical_ok and 0 <= phys < num_pages
            phys = phys if page_ok else 0
            acc = torch.zeros(HEADS, dtype=torch.float32)
            for h in range(HEADS):
                acc[h] = torch.dot(q[row, h, :].to(torch.float32), keys[phys, off, 0, :])
            if is_fp8:
                acc = acc * float(scale[phys, off, 0])
            total = torch.clamp_min(acc, 0.0).sum() / math.sqrt(HDIM)
            keep = page_ok and col < vis
            out[row, col] = total if keep else float("-inf")
    return out, vis_out


def _run_case(name, regime, rows, requests, fmt, table_width, cache_pages,
              max_columns=None, patch=None, seed=0, verbose=True):
    # max_ctx = 16 * table_width fills the page table (needed_per_req == width) and
    # makes ctx//4 == columns, so the visible window covers every column.
    max_ctx = 16 * table_width
    case = S.make_case(name, regime, rows=rows, requests=requests, max_ctx=max_ctx,
                       index_dtype=fmt, table_width=table_width, cache_pages=cache_pages)
    data = G.build_case(case, device="cpu", seed=seed)
    t2r, qpos, ctx = data["token_to_request"], data["query_positions"], data["context_lens"]
    table = data["page_table"]
    if patch is not None:
        t2r, qpos, ctx, table = patch(t2r.clone(), qpos.clone(), ctx.clone(), table.clone())
    cols = case.columns if max_columns is None else max_columns
    gold, gold_vis, valid = G.reference_logits(
        data["q"], data["compressed_k_cache"], table, t2r, qpos, ctx,
        compress_ratio=S.COMPRESS_RATIO, max_columns=cols,
        compressed_k_scale=data["compressed_k_scale"],
    )
    out, out_vis = emulate_page4(
        data["q"], data["compressed_k_cache"], data["compressed_k_scale"], table, t2r, qpos, ctx,
        max_columns=cols,
    )
    res = G.compare_logits(out, out_vis, gold, gold_vis, valid)
    ok = bool(res.get("pass"))
    if verbose:
        mae = res.get("max_abs_err", float("nan"))
        mre = res.get("max_rel_err", float("nan"))
        print("[%s] %-26s rows=%3d cols=%5d %-4s vis_sum=%6d checked=%6d max_abs=%.2e max_rel=%.2e bad=%d"
              % ("PASS" if ok else "FAIL", name, rows, out.shape[1], fmt,
                 int(gold_vis.sum()), res.get("checked", 0), mae, mre, res.get("bad", -1)))
    return ok, res


def _e4m3_byte_scalar(b):
    """Pure-Python mirror of the kernel's `_e4m3_byte_to_f32` integer formula."""
    sign = (b & 0x80) << 24
    normal_bits = (((b & 0x7F) << 20) + (120 << 23) | sign) & 0xFFFFFFFF
    normal = struct.unpack("<f", struct.pack("<I", normal_bits))[0]
    sub_mag = struct.unpack("<f", struct.pack("<I", 0x46800000 | (b & 7)))[0] - 16384.0
    sub = -sub_mag if b >= 0x80 else sub_mag
    return sub if ((b >> 3) & 0xF) == 0 else normal


def _check_e4m3_formula():
    """Every E4M3 code must decode to exactly torch's value (NaN codes excepted)."""
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
    ref = codes.to(torch.float32).tolist()
    bad = []
    nan_codes = {0x7F, 0xFF}
    for b in range(256):
        if b in nan_codes:
            continue
        got = _e4m3_byte_scalar(b)
        want = ref[b]
        same = (got == want) or (got != got and want != want)
        if not same:
            bad.append((hex(b), got, want))
    if bad:
        print("[FAIL] e4m3 formula: %d/%d codes wrong, first: %s"
              % (len(bad), 256, bad[:6]))
        return 1
    print("[PASS] e4m3 formula: 254/254 non-NaN codes exact "
          "(0x7F/0xFF are NaN in e4m3fn and decode as +-480; the ATOM fp8 "
          "writer clamps to +-448 so they never appear)")
    return 0


def _case_fixture(regime, rows, requests, fmt, table_width, cache_pages, max_columns,
                  patch, seed):
    """Build one fixture and return (data, patched metadata, columns)."""
    # max_ctx = 16 * table_width fills the page table (needed_per_req == width) and
    # makes ctx//4 == columns, so the visible window covers every column.
    max_ctx = 16 * table_width
    case = S.make_case("fx", regime, rows=rows, requests=requests, max_ctx=max_ctx,
                       index_dtype=fmt, table_width=table_width, cache_pages=cache_pages)
    data = G.build_case(case, device="cpu", seed=seed)
    t2r, qpos, ctx = data["token_to_request"], data["query_positions"], data["context_lens"]
    table = data["page_table"]
    if patch is not None:
        t2r, qpos, ctx, table = patch(t2r.clone(), qpos.clone(), ctx.clone(), table.clone())
    cols = case.columns if max_columns is None else int(max_columns)
    return data, (t2r, qpos, ctx, table), cols


def _contract_only(name, regime, rows, requests, fmt, table_width, cache_pages,
                   max_columns=None, patch=None, seed=0, verbose=True):
    """Check the kernel's contract where the *public golden cannot run*.

    ``qsa_golden.reference_logits`` clamps page indices with ``clamp_min(0)`` only
    and then indexes ``keys[phys]``, so a page-table entry ``>= num_pages`` makes
    the golden raise ``IndexError`` instead of being excluded by its own
    ``valid_page`` mask.  For those entries the contract is checked directly:
    the four columns of such a page must be ``-inf``, everything outside
    ``valid_page & c < visible`` must be ``-inf``, and nothing may be NaN.
    """
    data, (t2r, qpos, ctx, table), cols = _case_fixture(
        regime, rows, requests, fmt, table_width, cache_pages, max_columns, patch, seed)
    num_pages = int(data["compressed_k_cache"].shape[0])
    width = int(table.shape[1])
    out, out_vis = emulate_page4(
        data["q"], data["compressed_k_cache"], data["compressed_k_scale"], table,
        t2r, qpos, ctx, max_columns=cols)
    vis_ref = G.visible_groups(t2r, qpos, ctx, compress_ratio=S.COMPRESS_RATIO)
    ar = torch.arange(cols, dtype=torch.int64)
    logical = ar // PAGE
    safe_logical = logical.clamp(max=max(0, width - 1))
    t2r64 = t2r.to(torch.int64)
    req_ok = (t2r64 >= 0) & (t2r64 < requests)
    safe_req = t2r64.clamp(0, max(0, requests - 1))
    entry = table.to(torch.int64)[safe_req][:, safe_logical]          # [rows, cols]
    page_ok = req_ok[:, None] & (logical[None, :] < width) & (entry >= 0) & (entry < num_pages)
    inside = page_ok & (ar[None, :] < vis_ref.to(torch.int64)[:, None])
    outside = ~inside
    checks = {
        "visible_equal": bool(torch.equal(out_vis.to(torch.int32), vis_ref.to(torch.int32))),
        "invalid_page_neg_inf": bool(torch.isinf(out[~page_ok]).all()
                                     and (out[~page_ok] < 0).all()),
        "outside_window_neg_inf": bool(torch.isinf(out[outside]).all()
                                       and (out[outside] < 0).all()),
        "no_nan": not bool(torch.isnan(out).any()),
        "inside_finite": bool(torch.isfinite(out[inside]).all()) if bool(inside.any()) else True,
    }
    ok = all(checks.values())
    if verbose:
        print("[%s] %-26s rows=%3d cols=%5d %-4s pages=%4d checked=%s %s"
              % ("PASS" if ok else "FAIL", name, rows, cols, fmt, num_pages,
                 int(inside.sum()), "" if ok else checks))
    return ok, checks


def _scale_case(name, value, rows=4, requests=1, table_width=16, cache_pages=32):
    """fp8 with a uniform per-token scale: the scale must be applied exactly once."""
    case = S.make_case(name, "decode", rows=rows, requests=requests, max_ctx=256,
                       index_dtype="fp8", table_width=table_width,
                       cache_pages=cache_pages)
    data = G.build_case(case, device="cpu", seed=3)
    data["compressed_k_scale"].fill_(value)
    gold, gold_vis, valid = G.reference_logits(
        data["q"], data["compressed_k_cache"], data["page_table"], data["token_to_request"],
        data["query_positions"], data["context_lens"], compress_ratio=S.COMPRESS_RATIO,
        compressed_k_scale=data["compressed_k_scale"])
    out, out_vis = emulate_page4(
        data["q"], data["compressed_k_cache"], data["compressed_k_scale"],
        data["page_table"], data["token_to_request"], data["query_positions"],
        data["context_lens"])
    res = G.compare_logits(out, out_vis, gold, gold_vis, valid)
    ok = bool(res.get("pass"))
    print("[%s] %-26s scale=%-8g checked=%6d max_abs=%.2e bad=%d"
          % ("PASS" if ok else "FAIL", name, value, res.get("checked", 0),
             res.get("max_abs_err", float("nan")), res.get("bad", -1)))
    return ok, res


def main() -> int:
    fails = 0
    print("== emulated page-4 kernel vs public golden (CPU, fp32) ==")
    specs = [
        ("I5_min_decode", "decode", 4, 1, "bf16", 16, 32, None),
        ("I5_min_decode_fp8", "decode", 4, 1, "fp8", 16, 32, None),
        ("I3_style_draft", "draft", 8, 8, "bf16", 16, 256, None),
        ("I3_style_draft_fp8", "draft", 8, 8, "fp8", 16, 256, None),
        ("I6_two_req", "prefill", 16, 2, "bf16", 32, 128, None),
        ("I6_two_req_fp8", "prefill", 16, 2, "fp8", 32, 128, None),
        ("I7_rows64", "prefill", 64, 2, "bf16", 64, 256, None),
        ("I7_rows64_fp8", "prefill", 64, 2, "fp8", 64, 256, None),
        ("I8_272rows_prefill", "prefill", 272, 1, "bf16", 64, 64, None),
        ("tail_columns_1000", "prefill", 8, 2, "bf16", 256, 1024, 1000),
        ("tail_columns_1000_fp8", "prefill", 8, 2, "fp8", 256, 1024, 1000),
        ("tail_columns_1", "decode", 4, 1, "bf16", 16, 32, 1),
    ]
    for spec in specs:
        ok, _ = _run_case(*spec)
        fails += 0 if ok else 1

    print("== edge cases ==")
    def patch_vis0(t2r, qpos, ctx, table):          # visible == 0 rows
        qpos[0] = 2
        qpos[1] = 0
        return t2r, qpos, ctx, table
    ok, _ = _run_case("edge_visible_zero", "decode", 4, 1, "bf16", 16, 32, None, patch_vis0)
    fails += 0 if ok else 1
    ok, _ = _run_case("edge_visible_zero_fp8", "decode", 4, 1, "fp8", 16, 32, None, patch_vis0)
    fails += 0 if ok else 1

    def patch_t2r(t2r, qpos, ctx, table):           # padding row + out-of-range req
        t2r[0] = -1
        t2r[1] = 999
        return t2r, qpos, ctx, table
    ok, _ = _run_case("edge_t2r_invalid", "decode", 4, 1, "bf16", 16, 32, None, patch_t2r)
    fails += 0 if ok else 1

    def patch_page(t2r, qpos, ctx, table):          # -1 page entry
        # NB: a *high* out-of-range page index is NOT testable against the public
        # golden: qsa_golden.reference_logits clamps phys with clamp_min(0) only and
        # then indexes keys[phys], so phys >= num_pages raises IndexError (its
        # valid_page mask would have excluded those columns).
        table[0, 0] = -1
        return t2r, qpos, ctx, table
    for fmt in ("bf16", "fp8"):
        ok, _ = _run_case("edge_bad_page_" + fmt, "decode", 4, 1, fmt, 16, 32, None, patch_page)
        fails += 0 if ok else 1

    def patch_ctx(t2r, qpos, ctx, table):           # visible clamps at ctx//4
        ctx[0] = 8
        return t2r, qpos, ctx, table
    ok, _ = _run_case("edge_short_ctx", "decode", 4, 1, "bf16", 16, 32, None, patch_ctx)
    fails += 0 if ok else 1

    # --- fault-hypothesis cases: page / column tails ------------------------
    print("== fault hypotheses (tail columns, page range, single page) ==")
    def patch_page_high(t2r, qpos, ctx, table):
        table[0, 0] = 1 << 24                       # far past num_pages
        return t2r, qpos, ctx, table
    for fmt in ("bf16", "fp8"):
        ok, _ = _contract_only("page_oob_high_" + fmt, "decode", 4, 1, fmt, 16, 32, None,
                               patch_page_high)
        fails += 0 if ok else 1
    def patch_page_eq(t2r, qpos, ctx, table):
        table[0, 1] = 32                            # exactly num_pages == 32
        return t2r, qpos, ctx, table
    for fmt in ("bf16", "fp8"):
        ok, _ = _contract_only("page_eq_numpages_" + fmt, "decode", 4, 1, fmt, 16, 32, None,
                               patch_page_eq)
        fails += 0 if ok else 1
    def patch_page_last(t2r, qpos, ctx, table):
        table[0, 0] = 31                            # last valid page
        table[0, 1] = 0                             # first valid page
        return t2r, qpos, ctx, table
    for fmt in ("bf16", "fp8"):
        ok, _ = _run_case("page_edges_" + fmt, "decode", 4, 1, fmt, 16, 32, None, patch_page_last)
        fails += 0 if ok else 1

    # a single logical page (width == 1, 4 columns) and one row
    for fmt in ("bf16", "fp8"):
        ok, _ = _run_case("width1_" + fmt, "decode", 4, 1, fmt, 1, 4, None)
        fails += 0 if ok else 1

    # column tails: %4 != 0 and non-multiples of the 256-thread block
    for cols in (1, 3, 5, 255, 257):
        ok, _ = _run_case("tail_cols_%d" % cols, "prefill", 4, 1, "bf16", 65, 256, cols)
        fails += 0 if ok else 1
    for spec in (("tail_cols_2", 2), ("tail_cols_4", 4), ("tail_cols_256", 256),
                 ("tail_cols_258", 258), ("tail_cols_511", 511), ("tail_cols_512", 512)):
        ok, _ = _run_case(spec[0], "prefill", 4, 1, "bf16", 129, 512, spec[1])
        fails += 0 if ok else 1

    # fp8 scale plane: 0 (already covered below), 1.0 and > 1 must all match the
    # golden, i.e. the per-token fp32 scale is applied exactly once, at the
    # (page, off) index of the *physical* page.
    for tag, value in (("scale_one", 1.0), ("scale_gt1", 2.5), ("scale_small", 1.0e-3)):
        ok, _ = _scale_case("fp8_" + tag, value)
        fails += 0 if ok else 1

    # fp8 with an all-zero scale plane: k == 0 -> logits 0 inside the window
    case = S.make_case("scale0", "decode", rows=4, requests=1, max_ctx=256,
                       index_dtype="fp8", table_width=16, cache_pages=32)
    data = G.build_case(case, device="cpu", seed=1)
    data["compressed_k_scale"][:] = 0.0
    gold, gold_vis, valid = G.reference_logits(
        data["q"], data["compressed_k_cache"], data["page_table"], data["token_to_request"],
        data["query_positions"], data["context_lens"], compress_ratio=S.COMPRESS_RATIO,
        compressed_k_scale=data["compressed_k_scale"])
    out, out_vis = emulate_page4(
        data["q"], data["compressed_k_cache"], data["compressed_k_scale"], data["page_table"],
        data["token_to_request"], data["query_positions"], data["context_lens"])
    res = G.compare_logits(out, out_vis, gold, gold_vis, valid)
    print("[%s] edge_scale_zero_fp8            checked=%6d bad=%d"
          % ("PASS" if res.get("pass") else "FAIL", res.get("checked", 0), res.get("bad", -1)))
    fails += 0 if res.get("pass") else 1

    print("== E4M3 bit-formula check (all 256 codes, mirrors _e4m3_byte_to_f32) ==")
    fails += _check_e4m3_formula()

    print("== wrapper host-side contract checks (no CUDA reached) ==")
    sys.path.insert(0, _HERE)
    from _load import load_kernel_module

    try:
        K, how = load_kernel_module(allow_stub=True)
    except Exception as exc:  # noqa: BLE001
        print("[SKIP] kernel module not importable here (%s: %s) — needs flydsl"
              % (type(exc).__name__, exc))
        print("\nCPU contract test: %s (wrapper checks skipped)"
              % ("PASS" if fails == 0 else "%d FAILURE(S)" % fails))
        return 1 if fails else 0
    print("     kernel module: " + how)

    def expect_raise(tag, exc, fn):
        nonlocal fails
        try:
            fn()
        except exc:
            print("[PASS] " + tag)
            return
        except Exception as other:  # noqa: BLE001
            print("[FAIL] %s: raised %s: %s" % (tag, type(other).__name__, other))
            fails += 1
            return
        print("[FAIL] %s: no exception" % tag)
        fails += 1

    q = torch.zeros(4, HEADS, HDIM, dtype=torch.bfloat16)
    cache = torch.zeros(8, PAGE, 1, HDIM, dtype=torch.bfloat16)
    table = torch.zeros(1, 16, dtype=torch.int32)
    t2r = torch.zeros(4, dtype=torch.int32)
    qpos = torch.zeros(4, dtype=torch.int64)
    ctx = torch.zeros(1, dtype=torch.int32)
    base = dict(q=q, compressed_k_cache=cache, page_table=table, token_to_request=t2r,
                query_positions=qpos, context_lens=ctx)
    expect_raise("reject compress_ratio != 4", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(**base, compress_ratio=16))
    expect_raise("reject fp32 q", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(**{**base, "q": q.float()}))
    expect_raise("reject page != 4", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "compressed_k_cache": torch.zeros(8, 16, 1, HDIM,
                                                                 dtype=torch.bfloat16)}))
    expect_raise("reject fp8 cache without scale", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "compressed_k_cache": torch.zeros(8, PAGE, 1, HDIM,
                                                                 dtype=torch.float8_e4m3fn)}))
    expect_raise("reject bf16 cache with scale", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "compressed_k_scale": torch.zeros(8, PAGE, 1)}))
    expect_raise("reject wrong scale shape", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base,
                        "compressed_k_cache": torch.zeros(8, PAGE, 1, HDIM,
                                                          dtype=torch.float8_e4m3fn),
                        "compressed_k_scale": torch.zeros(8, HDIM)}))
    expect_raise("reject non-contiguous q", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "q": torch.zeros(8, HEADS, HDIM, dtype=torch.bfloat16)[::2]}))
    expect_raise("reject int8 metadata", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "token_to_request": torch.zeros(4, dtype=torch.int8)}))

    # --- GPU-safety guard rails added by the builder2 fix --------------------
    print("== GPU-safety guard rails (zero-size planes, i32 cap, alignment, divisor) ==")
    expect_raise("reject width == 0 (zero-length page table V#)", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "page_table": torch.zeros(1, 0, dtype=torch.int32)}))
    expect_raise("reject num_pages == 0 (zero-length cache V#)", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "compressed_k_cache": torch.zeros(0, PAGE, 1, HDIM,
                                                                 dtype=torch.bfloat16)}))
    expect_raise("reject max_columns == 0", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(**base, max_columns=0))
    expect_raise("reject divisor == 0", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(**base, divisor=0.0))
    expect_raise("reject divisor == inf", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(**base, divisor=float("inf")))
    # a contiguous but 2-byte-offset q view: 16-byte dwordx4 Q loads from it would
    # be misaligned
    misaligned = torch.zeros(4 * HEADS * HDIM + 1, dtype=torch.bfloat16)[1:].view(
        4, HEADS, HDIM)
    expect_raise("reject misaligned q (2-byte offset view)", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(**{**base, "q": misaligned}))
    expect_raise("reject metadata that wraps int32", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "context_lens": torch.tensor([2**40], dtype=torch.int64)}))
    expect_raise("reject page_table values that wrap int32", ValueError,
                 lambda: K.qsa_paged_mqa_logits_page4(
                     **{**base, "page_table": torch.full((1, 16), 2**33, dtype=torch.int64)}))

    # the guard helpers are also unit-tested directly (they carry the invariants)
    for tag, fn, exc in (
        ("_check_plane rejects 0 B", lambda: K._check_plane("p", 0), ValueError),
        ("_check_plane rejects > 2 GiB", lambda: K._check_plane("p", 2**31), ValueError),
        ("_check_plane accepts 1 B", lambda: K._check_plane("p", 1), None),
        ("_to_i32_checked rejects 2**31", lambda: K._to_i32_checked(
            torch.tensor([2**31], dtype=torch.int64), "m"), ValueError),
        ("_to_i32_checked passes int32 through", lambda: K._to_i32_checked(
            torch.zeros(3, dtype=torch.int32), "m"), None),
    ):
        try:
            fn()
        except Exception as other:  # noqa: BLE001
            got = other
        else:
            got = None
        if exc is None:
            ok = got is None
        else:
            ok = isinstance(got, exc)
        print("[%s] %s" % ("PASS" if ok else "FAIL", tag))
        fails += 0 if ok else 1

    print("\nCPU contract test: %s" % ("PASS" if fails == 0 else "%d FAILURE(S)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
