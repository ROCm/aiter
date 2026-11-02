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


# ---------------------------------------------------------------------------
# Emulation of the bf16 WMMA kernel (round C), lane by lane
# ---------------------------------------------------------------------------
# Mirrors `_build_wmma_kernel` exactly: grid (ceil(cols/64), ceil(rows/16)),
# 4 waves x 32 lanes, wave rows t0..t0+3, request-dedup passes, the early -inf
# branch, per-lane byte offsets of every Q / K / page-table / metadata load
# (bounds-checked against the plane), the gfx1250 WMMA operand layouts
#   A/B: lane l, elem v<16 -> row l%16, k = (v//8)*16 + (l//16)*8 + v%8
#   C/D: lane l, elem v<8  -> m = (l//16)*8 + v, n = l%16
# and every predicated store (bounds-checked, counted: each logit and each
# visible/row_starts entry must be written exactly once).
WM_WAVE, WM_WAVES, WM_ROWS, WM_N, WM_K, WM_NT = 32, 4, 4, 16, 32, 4
WM_COLS = WM_N * WM_NT
WM_KS = HDIM // WM_K
_LANES = torch.arange(WM_WAVE)
_VALS = torch.arange(16)
_A_ROW = (_LANES[:, None] % 16).expand(WM_WAVE, 16)
_A_K = (_VALS[None, :] // 8) * 16 + (_LANES[:, None] // 16) * 8 + _VALS[None, :] % 8


class _Plane:
    def __init__(self, name, nbytes):
        self.name, self.nbytes = name, nbytes

    def check(self, offs, width):
        offs = torch.as_tensor(offs, dtype=torch.int64).reshape(-1)
        if offs.numel() == 0:
            return
        lo, hi = int(offs.min()), int(offs.max()) + width
        if lo < 0 or hi > self.nbytes:
            raise AssertionError(f"{self.name}: access [{lo}, {hi}) outside plane of {self.nbytes} B")
        if hi > 2**31 - 1:
            raise AssertionError(f"{self.name}: byte offset {hi} overflows int32")


def _ld16_bf16(u8, plane, offs):
    """[L] byte offsets -> [L, 8] f32 (one 16-byte buffer_load per lane)."""
    plane.check(offs, 16)
    idx = offs[:, None] + torch.arange(16)[None, :]
    b = u8[idx].to(torch.int32)
    bits = (b[:, 0::2] | (b[:, 1::2] << 8)) << 16
    return bits.to(torch.int32).view(torch.float32)


def _wm_frag(u8, plane, base):
    """Operand fragment: 16 bytes at base and at base + 32 -> [32, 16] f32."""
    return torch.cat([_ld16_bf16(u8, plane, base), _ld16_bf16(u8, plane, base + 32)], dim=1)


def _wm_matrix(frag):
    m = torch.zeros(16, WM_K, dtype=torch.float32)
    m[_A_ROW, _A_K] = frag
    return m


def emulate_wmma_bf16(q, cache, table, t2r, qpos, ctx, max_columns=None):
    """logits / visible / row_starts exactly as the WMMA kernel writes them."""
    rows = q.shape[0]
    num_pages = cache.shape[0]
    num_requests, width = table.shape
    columns = width * PAGE if max_columns is None else max(1, min(width * PAGE, int(max_columns)))
    q_u8 = q.contiguous().view(torch.uint8).reshape(-1)
    kv_u8 = cache.contiguous().view(torch.uint8).reshape(-1)
    pq = _Plane("q", rows * HEADS * HDIM * 2)
    pkv = _Plane("compressed_k_cache", num_pages * PAGE * HDIM * 2)
    ppt = _Plane("page_table", num_requests * width * 4)
    pmeta = _Plane("rows_meta", rows * 4)
    pcl = _Plane("context_lens", num_requests * 4)
    plog = _Plane("logits", rows * columns * 4)
    tbl = table.to(torch.int64).reshape(-1)
    t2r_l, qpos_l, ctx_l = t2r.tolist(), qpos.tolist(), ctx.tolist()
    inv = 1.0 / math.sqrt(HDIM)

    out = torch.full((rows * columns,), float("nan"), dtype=torch.float32)
    count = torch.zeros(rows * columns, dtype=torch.int32)
    vis_out = torch.full((rows,), -7, dtype=torch.int32)
    vis_cnt = torch.zeros(rows, dtype=torch.int32)
    rst_cnt = torch.zeros(rows, dtype=torch.int32)

    def store(ok, off, val):
        ok = torch.as_tensor(ok, dtype=torch.bool)
        off = torch.as_tensor(off, dtype=torch.int64)
        plog.check(off, 4)                      # the clamped address, taken or not
        if not bool(ok.any()):
            return
        idx = off[ok] // 4
        val = torch.as_tensor(val, dtype=torch.float32).expand(off.shape)[ok]
        out[idx] = val
        count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))

    grid_x = (columns + WM_COLS - 1) // WM_COLS
    grid_y = (rows + WM_ROWS * WM_WAVES - 1) // (WM_ROWS * WM_WAVES)
    n_in = _LANES % WM_N
    kgrp = _LANES // WM_N
    for by in range(grid_y):
        for wave in range(WM_WAVES):
            t0 = (by * WM_WAVES + wave) * WM_ROWS
            meta = []
            for i in range(WM_ROWS):
                row = t0 + i
                safe_row = min(row, rows - 1)
                pmeta.check([safe_row * 4], 4)
                req = int(t2r_l[safe_row])
                req_ok = 0 <= req < num_requests
                safe_req = req if req_ok else 0
                pcl.check([safe_req * 4], 4)
                head = max(min(int(qpos_l[safe_row]) + 1, int(ctx_l[safe_req])), 0)
                vis = head // PAGE if req_ok else 0
                meta.append(dict(row=row, row_ok=row < rows, req=req, req_ok=req_ok,
                                 safe_req=safe_req, vis=vis))
            # metadata writer (chunk 0, lanes 0..3)
            for lane in range(WM_ROWS):
                my_row = t0 + min(lane, WM_ROWS - 1)
                pmeta.check([min(my_row, rows - 1) * 4], 4)
                if my_row < rows:
                    vis_out[my_row] = meta[lane]["vis"]
                    vis_cnt[my_row] += 1
                    rst_cnt[my_row] += 1
            # per-pass plan (uniform), lane rows a = t0 + 2*kgrp, b = a + 1
            plans = []
            for p in range(WM_ROWS):
                mp = meta[p]
                active = mp["row_ok"] and all(
                    (meta[qq]["row"] >= rows) or (meta[qq]["req"] != mp["req"]) for qq in range(p))
                own = [meta[i]["row_ok"] and meta[i]["req"] == mp["req"] for i in range(WM_ROWS)]
                vmax = max([meta[i]["vis"] if own[i] else 0 for i in range(WM_ROWS)] + [0])
                own_a = torch.tensor([own[0], own[2]])[kgrp]
                own_b = torch.tensor([own[1], own[3]])[kgrp]
                vis_a = torch.tensor([meta[0]["vis"], meta[2]["vis"]])[kgrp]
                vis_b = torch.tensor([meta[1]["vis"], meta[3]["vis"]])[kgrp]
                plans.append((active, vmax, mp["safe_req"], mp["req_ok"], own_a, own_b, vis_a, vis_b))
            if t0 >= rows:
                continue
            q_row = torch.clamp(t0 + n_in // HEADS, max=rows - 1)
            q_base = q_row * (HEADS * HDIM * 2) + (n_in % HEADS) * (HDIM * 2) + kgrp * 16
            qa = [_wm_matrix(_wm_frag(q_u8, pq, q_base + s * WM_K * 2)) for s in range(WM_KS)]
            row_a = torch.clamp(t0 + 2 * kgrp, max=rows - 1)
            row_b = torch.clamp(t0 + 2 * kgrp + 1, max=rows - 1)
            last = rows * columns - 1
            for bx in range(grid_x):
                col0 = bx * WM_COLS
                cols = [col0 + nt * WM_N + n_in for nt in range(WM_NT)]
                offs = []
                for c in cols:
                    offs.append(torch.clamp(row_a * columns + c, max=last) * 4)
                    offs.append(torch.clamp(row_b * columns + c, max=last) * 4)
                for (active, vmax, safe_req, req_ok, own_a, own_b, vis_a, vis_b) in plans:
                    if not active:
                        continue
                    if col0 < vmax:
                        for nt, c in enumerate(cols):
                            logical = c // PAGE
                            off = c % PAGE
                            logical_ok = logical < width
                            safe_logical = torch.where(logical_ok, logical, torch.zeros_like(logical))
                            pt_off = (safe_req * width + safe_logical) * 4
                            ppt.check(pt_off, 4)
                            phys = tbl[pt_off // 4]
                            page_ok = req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
                            phys = torch.where(page_ok, phys, torch.zeros_like(phys))
                            k_base = phys * (PAGE * HDIM * 2) + off * (HDIM * 2) + kgrp * 16
                            d = torch.zeros(16, WM_N, dtype=torch.float32)
                            for s in range(WM_KS):
                                kb = _wm_matrix(_wm_frag(kv_u8, pkv, k_base + s * WM_K * 2))
                                d = d + qa[s] @ kb.T
                            acc = d[(kgrp[:, None] * 8 + torch.arange(8)[None, :]), n_in[:, None]]
                            relu = acc.clamp_min(0.0)
                            sa = ((relu[:, 0] + relu[:, 1]) + (relu[:, 2] + relu[:, 3])) * inv
                            sb = ((relu[:, 4] + relu[:, 5]) + (relu[:, 6] + relu[:, 7])) * inv
                            col_ok = c < columns
                            va = torch.where(page_ok & (c < vis_a), sa, torch.full_like(sa, float("-inf")))
                            vb = torch.where(page_ok & (c < vis_b), sb, torch.full_like(sb, float("-inf")))
                            store(own_a & col_ok, offs[2 * nt], va)
                            store(own_b & col_ok, offs[2 * nt + 1], vb)
                    else:
                        for nt, c in enumerate(cols):
                            col_ok = c < columns
                            store(own_a & col_ok, offs[2 * nt], float("-inf"))
                            store(own_b & col_ok, offs[2 * nt + 1], float("-inf"))
    coverage = {
        "logits_written_once": bool((count == 1).all()),
        "logits_min_max_writes": (int(count.min()), int(count.max())),
        "visible_written_once": bool((vis_cnt == 1).all()),
        "row_starts_written_once": bool((rst_cnt == 1).all()),
    }
    return out.view(rows, columns), vis_out, coverage


def _wmma_check(name, data_q, cache, table, t2r, qpos, ctx, cols, gold=None, gold_vis=None,
                valid=None, verbose=True):
    """Run the WMMA emulator and check coverage (+ golden when given)."""
    out, out_vis, cov = emulate_wmma_bf16(data_q, cache, table, t2r, qpos, ctx, max_columns=cols)
    cov_ok = (cov["logits_written_once"] and cov["visible_written_once"]
              and cov["row_starts_written_once"])
    res = {"pass": True}
    if gold is not None:
        res = G.compare_logits(out, out_vis, gold, gold_vis, valid)
        cols_ix = torch.arange(gold.shape[1])
        inside = valid & (cols_ix[None, :] < gold_vis.to(torch.int64)[:, None])
        outside = out[~inside]
        res["outside_not_neg_inf"] = int((~(torch.isinf(outside) & (outside < 0))).sum())
        res["pass"] = bool(res.get("pass")) and res["outside_not_neg_inf"] == 0
    ok = cov_ok and bool(res.get("pass")) and not bool(torch.isnan(out).any())
    if verbose:
        print("       [%s] wmma-emu %-22s writes(min,max)=%s vis1=%s max_abs=%.2e outside!=-inf=%s"
              % ("PASS" if ok else "FAIL", name, cov["logits_min_max_writes"],
                 cov["visible_written_once"], res.get("max_abs_err", 0.0),
                 res.get("outside_not_neg_inf", "-")))
    return ok, out, out_vis


def _run_case(name, regime, rows, requests, fmt, table_width, cache_pages,
              max_columns=None, patch=None, seed=0, verbose=True):
    ok, res = _run_case_scalar(name, regime, rows, requests, fmt, table_width, cache_pages,
                               max_columns, patch, seed, verbose)
    if fmt != "bf16":
        return ok, res
    data, (t2r, qpos, ctx, table), cols = _case_fixture(
        regime, rows, requests, fmt, table_width, cache_pages, max_columns, patch, seed)
    gold, gold_vis, valid = G.reference_logits(
        data["q"], data["compressed_k_cache"], table, t2r, qpos, ctx,
        compress_ratio=S.COMPRESS_RATIO, max_columns=cols)
    wok, _, _ = _wmma_check(name, data["q"], data["compressed_k_cache"], table, t2r, qpos, ctx,
                            cols, gold, gold_vis, valid, verbose)
    return ok and wok, res


def _run_case_scalar(name, regime, rows, requests, fmt, table_width, cache_pages,
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


def _expect(cond):
    if not cond:
        raise AssertionError("expectation failed")


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

    def _checks(o, ov):
        return {
            "visible_equal": bool(torch.equal(ov.to(torch.int32), vis_ref.to(torch.int32))),
            "invalid_page_neg_inf": bool(torch.isinf(o[~page_ok]).all()
                                         and (o[~page_ok] < 0).all()),
            "outside_window_neg_inf": bool(torch.isinf(o[outside]).all()
                                           and (o[outside] < 0).all()),
            "no_nan": not bool(torch.isnan(o).any()),
            "inside_finite": bool(torch.isfinite(o[inside]).all()) if bool(inside.any()) else True,
        }

    checks = _checks(out, out_vis)
    if fmt == "bf16":
        # the WMMA path must satisfy the same contract and agree with the scalar
        # emulation inside the window (both are fp32 dot products of bf16 data)
        wout, wvis, cov = emulate_wmma_bf16(
            data["q"], data["compressed_k_cache"], table, t2r, qpos, ctx, max_columns=cols)
        for k, v in _checks(wout, wvis).items():
            checks["wmma_" + k] = v
        checks["wmma_written_once"] = (cov["logits_written_once"] and cov["visible_written_once"]
                                       and cov["row_starts_written_once"])
        if bool(inside.any()):
            checks["wmma_matches_scalar"] = bool(torch.allclose(
                wout[inside], out[inside], rtol=1e-4, atol=1e-4))
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

    # --- bf16 WMMA path (round C): every bf16 case above already runs the WMMA
    # lane emulator too; these add its own geometry / dedup / early-exit edges --
    print("== bf16 WMMA path: row/column geometry, request dedup, early exit ==")
    wm_specs = [
        # rows not a multiple of 4 / 16 / 64 (partial wave, partial workgroup)
        ("wm_rows7", "prefill", 7, 2, "bf16", 32, 128, None, None),
        ("wm_rows13", "prefill", 13, 2, "bf16", 32, 128, None, None),
        ("wm_rows70", "prefill", 70, 2, "bf16", 16, 64, None, None),
        # decode shape class (4 rows / request) and draft (1 row / request:
        # four request passes in every wave)
        ("wm_decode_64x16", "decode", 64, 16, "bf16", 16, 512, None, None),
        ("wm_draft_16x16", "draft", 16, 16, "bf16", 16, 512, None, None),
        # columns not a multiple of 64 / 16 / 4
        ("wm_tail_cols_1000", "prefill", 8, 2, "bf16", 256, 1024, 1000, None),
        ("wm_tail_cols_70", "prefill", 13, 2, "bf16", 32, 128, 70, None),
    ]
    for spec in wm_specs:
        ok, _ = _run_case(*spec)
        fails += 0 if ok else 1

    def patch_mixed(t2r, qpos, ctx, table):        # non-contiguous requests in a wave
        n = t2r.numel()
        pattern = [0, 1, 1, 0, 1, 0, 0, 1]
        for i in range(n):
            t2r[i] = pattern[i % len(pattern)]
        t2r[5] = -1                                 # invalid request inside a wave
        if n > 9:
            t2r[9] = 7                              # out of range request
        return t2r, qpos, ctx, table
    ok, _ = _run_case("wm_mixed_requests", "prefill", 16, 2, "bf16", 32, 128, None, patch_mixed)
    fails += 0 if ok else 1

    def patch_short(t2r, qpos, ctx, table):        # early-exit branch on most blocks
        ctx[0] = 40                                 # visible <= 10 for request 0
        qpos[-1] = 70                               # visible 17 on the last row (partial block)
        return t2r, qpos, ctx, table
    ok, _ = _run_case("wm_early_exit", "prefill", 13, 2, "bf16", 64, 256, None, patch_short)
    fails += 0 if ok else 1

    def patch_holes(t2r, qpos, ctx, table):        # -1 / OOB pages inside a 64-col block
        table[0, 3] = -1
        table[0, 17] = -1
        table[1, 15] = 10**6
        return t2r, qpos, ctx, table
    ok, _ = _contract_only("wm_page_holes", "prefill", 13, 2, "bf16", 32, 128, None, patch_holes)
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
    # ROUND-NOSYNC: the historical raise-on-wrap contract is opt-in
    # (QSA_STRICT_I32=1, read at import -> patched module flag here). The
    # default saturates on the device with no host sync; its exactness is
    # covered by test_nosync_i32_cpu.py and checked directly below.
    _strict = K._QSA_STRICT_I32
    K._QSA_STRICT_I32 = True
    try:
        expect_raise("[strict] reject metadata that wraps int32", ValueError,
                     lambda: K.qsa_paged_mqa_logits_page4(
                         **{**base, "context_lens": torch.tensor([2**40], dtype=torch.int64)}))
        expect_raise("[strict] reject page_table values that wrap int32", ValueError,
                     lambda: K.qsa_paged_mqa_logits_page4(
                         **{**base, "page_table": torch.full((1, 16), 2**33, dtype=torch.int64)}))
    finally:
        K._QSA_STRICT_I32 = _strict
    for tag, value, hi, want in (
        ("default saturates context 2**40 -> INT32_MAX", 2**40, 2**31 - 1, 2**31 - 1),
        ("default saturates page 2**33 -> INT32_MAX", 2**33, 2**31 - 1, 2**31 - 1),
        ("default saturates -2**40 -> INT32_MIN", -2**40, 2**31 - 1, -2**31),
        ("default position INT32_MAX -> INT32_MAX-1", 2**31 - 1, 2**31 - 2, 2**31 - 2),
    ):
        got = K._to_i32_saturated(torch.tensor([value], dtype=torch.int64), hi)
        ok = got.dtype == torch.int32 and int(got[0]) == want
        print("[%s] %s" % ("PASS" if ok else "FAIL", tag))
        fails += 0 if ok else 1
    i32 = torch.zeros(3, dtype=torch.int32)
    ok = K._to_i32_saturated(i32) is i32
    print("[%s] default int32 metadata is zero-copy" % ("PASS" if ok else "FAIL"))
    fails += 0 if ok else 1

    # the guard helpers are also unit-tested directly (they carry the invariants)
    for tag, fn, exc in (
        ("_check_plane rejects 0 B", lambda: K._check_plane("p", 0), ValueError),
        ("_check_plane rejects > 2 GiB", lambda: K._check_plane("p", 2**31), ValueError),
        ("_check_plane accepts 1 B", lambda: K._check_plane("p", 1), None),
        ("_to_i32_checked rejects 2**31", lambda: K._to_i32_checked(
            torch.tensor([2**31], dtype=torch.int64), "m"), ValueError),
        ("_to_i32_checked passes int32 through", lambda: K._to_i32_checked(
            torch.zeros(3, dtype=torch.int32), "m"), None),
        # round C dispatch: bf16 -> WMMA kernel by default, grid covers every
        # (row, column) with 16 x 64 workgroups
        ("bf16 path defaults to wmma", lambda: _expect(K._BF16_PATH == "wmma"), None),
        ("wmma_grid(7 rows, 1000 cols) == (16, 1)",
         lambda: _expect(K.wmma_grid(7, 1000) == (16, 1)), None),
        ("wmma_grid(512, 65536) == (1024, 32)",
         lambda: _expect(K.wmma_grid(512, 65536) == (1024, 32)), None),
        ("wmma_grid(65535, 4) == (1, 4096)",
         lambda: _expect(K.wmma_grid(65535, 4) == (1, 4096)), None),
        ("WMMA geometry 4 waves x 32 lanes, 16 rows x 64 cols",
         lambda: _expect((K.WMMA_THREADS, K.WMMA_ROWS_PER_BLOCK, K.WMMA_COLS)
                         == (128, 16, 64)), None),
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
