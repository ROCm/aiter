# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ROUND-R2B kernel CPU gates (no GPU, no FlyDSL): kstat prefill, persist fallback loop, route table.

1. kstat (K-stationary / Q-reuse r32 prefill), lane-level emulation with the ROUND-PAGE16
   harness (``test_qsa_page16_cpu``): the *real* trace-time helpers of the kernel source
   (``_kstat_slab_req``, ``_kstat_tile_ok`` -> ``_r32_guard``, ``_persist_block_vis``, the TDM
   producer ``_persist_issue_tdm`` into buffer ``ct``, the consumer ``_pair_base_tdm`` and
   ``_pair_base_raw`` of the exact fallback) are executed against the fake ``fx`` and the
   recording / bounds-checked TDM gather, workgroup by workgroup over ``kstat_grid``; every
   (row, column) is resolved to the K row its WMMA B fragment reads (a consumer read of an
   LDS row that was never gathered fails), logits vs the frozen golden ``reference_logits``
   on ``qsa_shapes.make_case(page_size=P)`` + ``qsa_golden.build_case`` fixtures for
   P = 4 / 8 / 16 and every (row_tiles, col_tiles), with bad / out-of-range pages, invalid
   token_to_request, multi-request spans (other-request blocks -> fallback), rows not a
   multiple of 32 / 64 / 32R, columns not a multiple of the slab, visible < columns.
2. Persist fallback loop: ``kernel_r32_persist_fbl`` is ``kernel_r32_persist`` verbatim
   except the fallback chunk loop header (runtime scf.for, unroll disabled) and its column
   origin; the kstat kernel body has no wait/barrier under a guard.
3. Route table: ``legacy`` == the d438d2b3 condition for every shape; rule parsing / first
   match / page filters / persist veto; ``kstat_grid`` covers every element exactly once.

    QSA_GOLDEN_DIR=/path/golden_baseline python3 -B op_tests/test_qsa_r2b_kernel_cpu.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import textwrap
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("qsa_page16_cpu_harness_r2b",
                                               os.path.join(HERE, "test_qsa_page16_cpu.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
G, S = T.G, T.S
FI, FB, _i, Plane, Env = T.FI, T.FB, T._i, T.Plane, T.Env
ROW = T.ROW
TILE = 64 * ROW


def _nested(name):
    return T._nested("_build_wmma_kernel", name)


def _route_ns(**env):
    """Top-level route / grid helpers of the kernel source in a fresh namespace."""
    ns = {"re": re, "os": os, "PAGE_SIZE": 4, "SUPPORTED_PAGE_SIZES": (4, 8, 16), "WMMA_COLS": 64,
          "TDM_PERSIST": env.get("persist", True), "TDM_PERSIST_CHUNKS": env.get("chunks", 8)}
    for c in ("KSTAT_ROW_TILES", "KSTAT_COL_TILES", "PREFILL_ROUTE_PRESETS"):
        ns[c] = T._const(c)
    for f in ("kstat_grid", "_parse_route_variant", "parse_prefill_route", "prefill_bf16_route",
              "wmma_grid"):
        exec(T._top(f), ns)
    ns["PREFILL_ROUTE"] = ns["parse_prefill_route"](env.get("spec", "legacy"))
    return ns


class KstatEmu:
    """One kstat specialization (R, C) at page P over a golden fixture."""

    def __init__(self, d, P, R, C, columns):
        self.d, self.P, self.R, self.C, self.cols = d, P, R, C, columns
        self.pt = d["page_table"].cpu().numpy().astype(np.int64)
        self.pages = int(d["compressed_k_cache"].shape[0])
        self.width = int(self.pt.shape[1])
        self.t2r, self.ok, self.sreq, self.vis = T._meta(d)
        self.rows = self.t2r.size
        self.nreq = int(d["context_lens"].numel())
        self.ptp = Plane(self.pt)
        self.t2rp = Plane(self.t2r, "t2r")
        self.clp = Plane(d["context_lens"].cpu().numpy().astype(np.int64), "cl")
        env = Env()
        self.env = env
        imin = lambda a, b: FI(min(_i(a), _i(b)))      # noqa: E731
        imax = lambda a, b: FI(max(_i(a), _i(b)))      # noqa: E731
        ns = dict(fx=env.fx, range_constexpr=range, _ld_uniform_i32=env.ld_uniform, _imin=imin,
                  _imax=imax, COMPRESS_RATIO=4, guard_load=env.ld_uniform)
        exec(_nested("_persist_block_vis"), ns)
        exec(_nested("_r32_guard"), ns)
        exec(_nested("_kstat_slab_req"), ns)
        exec(_nested("_kstat_tile_ok"), ns)
        self.slab_req, self.tile_ok = ns["_kstat_slab_req"], ns["_kstat_tile_ok"]
        self.issue = T._exec(env, P, _nested("_persist_issue_tdm"), "_persist_issue_tdm",
                             _tdm_k_gather=env.gather(ROW))
        self.base_tdm = T._exec(env, P, _nested("_pair_base_tdm"), "_pair_base_tdm", tdm_cols=64)
        self.base_raw = T._exec(env, P, _nested("_pair_base_raw"), "_pair_base_raw",
                                page_bytes=P * 256, key_row_bytes=256)
        self.stats = {"wg": 0, "tiles_lds": 0, "tiles_fallback": 0, "issues": 0}

    def run(self):
        P, R, C, cols, rows = self.P, self.R, self.C, self.cols, self.rows
        ns = _route_ns()
        gx, gy = ns["kstat_grid"](rows, cols, R, C)
        assert gx * 64 * C >= cols and (gx - 1) * 64 * C < cols
        assert gy * 32 * R >= rows and (gy - 1) * 32 * R < rows
        src = np.full((rows, cols), -2, dtype=np.int64)        # -2 = never written
        width, pages, nreq, rows_i = FI(self.width), FI(self.pages), FI(self.nreq), FI(rows)
        for y in range(gy):
            for x in range(gx):
                self.stats["wg"] += 1
                slab0 = x * 64 * C
                kreq, kreq_ok, ksafe, kvis = self.slab_req(self.t2rp, self.clp, FI(y * 32 * R),
                                                           rows_i, nreq)
                self.env.lds = {}
                for ct in range(C):
                    c0 = slab0 + ct * 64
                    if (kreq_ok & (FI(c0) < kvis) & (FI(c0) < FI(cols))).v:
                        self.stats["issues"] += 1
                        for w in range(4):
                            self.env.wave = w
                            self.issue(None, None, self.ptp, ksafe, FI(c0), width, pages, FI(ct))
                lds = dict(self.env.lds)
                for it in range(R):
                    rb = y * R + it
                    ok = self.tile_ok(self.t2rp, FI(rb * 32), rows_i, nreq, kreq).v
                    if rb * 32 < rows:
                        self.stats["tiles_lds" if ok else "tiles_fallback"] += 1
                    for r in range(rb * 32, min(rb * 32 + 32, rows)):
                        for c in range(slab0, min(slab0 + 64 * C, cols)):
                            assert src[r, c] == -2, ("written twice", r, c)
                            src[r, c] = self._elem(r, c, ok, lds, slab0)
        assert (src != -2).all(), "element never written"
        return src

    def _elem(self, r, c, lds_path, lds, slab0):
        if not self.ok[r] or c >= int(self.vis[r]):
            return -1
        req = int(self.sreq[r])
        page_ok, phys = T._consumer_page(self.P, self.pt, req, True, c, self.width, self.pages)
        if not page_ok:
            return -1
        if lds_path:
            ct = (c - slab0) // 64
            off = _i(self.base_tdm(None, None, FI(c), 0))
            assert off % ROW == 0
            return lds[ct * TILE + off]          # KeyError = consumer of a never-gathered row
        addr = _i(self.base_raw(FI(phys), FI(c % self.P), None, 0))
        assert addr % 256 == 0
        return addr // 256


def _edits_multi(d):
    """Make request 1's rows start mid-block: spans/blocks straddle requests."""
    t = d["token_to_request"]
    n = t.numel()
    t[: n // 2 + 7] = 0
    t[n // 2 + 7:] = 1


def _edits_one_req(d):
    """Single-request prefill call (the prefill_bench shape): every row is request 0."""
    d["token_to_request"][:] = 0


def _edits_all(d):
    T._edits_bad_pages(d)
    T._edits_invalid_t2r(d)


class KstatEmulation(unittest.TestCase):
    KNOBS = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (4, 2), (8, 1), (1, 4), (2, 4)]
    # (name, rows, requests, max_ctx, table_width @P=4, pages @P=4, edits, columns)
    CASES = [
        ("prefill", 128, 1, 2300, 160, 480, _edits_one_req, None),
        ("bad_pages", 96, 2, 2300, 160, 480, T._edits_bad_pages, None),
        ("invalid_t2r", 100, 2, 1900, 160, 480, T._edits_invalid_t2r, None),
        ("multi_req_tail", 150, 2, 2100, 160, 480, _edits_multi, 600),
        ("all_edges", 70, 2, 1500, 160, 480, _edits_all, 333),
    ]

    def _check(self, P, name, rows, reqs, ctx, width, pages, edits, columns, knobs):
        scale = P // 4
        d = T._fixture(P, "prefill", rows, reqs, ctx, max(width // scale, 4),
                       max(pages // scale, 8), edits=edits)
        cols_full = int(d["page_table"].shape[1]) * P
        cols = cols_full if columns is None else min(columns, cols_full)
        gold, gvis, valid = T._golden(d, cols)
        inside = valid & (torch.arange(cols)[None, :] < gvis.to(torch.int64)[:, None])
        ins = inside.numpy()
        rr, cc = np.nonzero(ins)
        for R, C in knobs:
            emu = KstatEmu(d, P, R, C, cols)
            src = emu.run()
            out = torch.from_numpy(T._scores(d, src)).to(torch.float32)
            vis = torch.from_numpy(emu.vis.astype(np.int32))
            res = G.compare_logits(out, vis, gold.to(torch.float32), gvis, valid, rtol=1e-5,
                                   atol=1e-5)
            self.assertTrue(res["pass"], (P, name, R, C, res))
            self.assertTrue(bool(torch.isneginf(out[~inside]).all()), (P, name, R, C))
            want = d["page_table"].numpy().astype(np.int64)[emu.sreq[rr], cc // P] * P + cc % P
            self.assertTrue(np.array_equal(src[rr, cc], want), (P, name, R, C))
            self.assertTrue((src[~ins] == -1).all(), (P, name, R, C))
            if name == "prefill":
                self.assertEqual(emu.stats["tiles_fallback"], 0, (P, R, C, emu.stats))

    def test_kstat_all_pages_all_knobs(self):
        for P in (4, 8, 16):
            for case in self.CASES:
                with self.subTest(P=P, case=case[0]):
                    self._check(P, *case, knobs=self.KNOBS)

    def test_kstat_fallback_is_exercised(self):
        d = T._fixture(4, "prefill", 150, 2, 2100, 160, 480, edits=_edits_multi)
        emu = KstatEmu(d, 4, 2, 2, 600)
        emu.run()
        self.assertGreater(emu.stats["tiles_fallback"], 0)
        self.assertGreater(emu.stats["tiles_lds"], 0)

    def test_kstat_gathers_once_per_slab(self):
        """K is gathered once per (slab, span): R row blocks share it."""
        d = T._fixture(4, "prefill", 256, 1, 4096, 256, 600, edits=_edits_one_req)
        for R, C in ((1, 1), (4, 1), (8, 2)):
            emu = KstatEmu(d, 4, R, C, 1024)
            emu.run()
            gx, gy = _route_ns()["kstat_grid"](256, 1024, R, C)
            kvis = max(int(d["context_lens"][0]), 0) // 4
            want = sum(1 for x in range(gx) for ct in range(C)
                       if x * 64 * C + ct * 64 < min(kvis, 1024)) * gy
            self.assertEqual(emu.stats["issues"], want)
            self.assertEqual(emu.stats["tiles_fallback"], 0)


class PersistFallbackLoop(unittest.TestCase):
    def _kernel_src(self, name):
        node = next(n for n in ast.walk(T._builder("_build_wmma_kernel"))
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        return [ln.strip() for ln in textwrap.dedent(ast.get_source_segment(T.SRC, node)).splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    def test_fbl_is_persist_except_the_fallback_loop(self):
        a = self._kernel_src("kernel_r32_persist")
        b = self._kernel_src("kernel_r32_persist_fbl")
        a, b = a[1:], b[1:]               # def lines differ by name only
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(a), len(b))
        self.assertEqual(diff, [
            ("for kf in range_constexpr(persist_chunks):",
             "for kf in range(fx.Int32(0), fx.Int32(persist_chunks), fx.Int32(1), unroll=1):"),
            ("fcol0 = span0 + kf * WMMA_COLS", "fcol0 = span0 + fx.Int32(kf) * WMMA_COLS"),
        ])

    def test_no_wait_or_barrier_under_a_guard_in_kstat(self):
        lines = self._kernel_src("kernel_r32_kstat")
        depth_ok = True
        src = textwrap.dedent(ast.get_source_segment(
            T.SRC, next(n for n in ast.walk(T._builder("_build_wmma_kernel"))
                        if isinstance(n, ast.FunctionDef) and n.name == "kernel_r32_kstat")))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.If):
                seg = ast.unparse(node)
                if "barrier" in seg or "tensor_wait" in seg:
                    depth_ok = False
        self.assertTrue(depth_ok and lines)

    def test_defaults(self):
        self.assertEqual(T._const("PERSIST_FALLBACKS"), ("loop", "unroll"))
        self.assertIn('os.environ.get("QSA_PERSIST_FALLBACK", "loop")', T.SRC)
        # ROUND-R2E: default preset r2e (legacy stays a preset)
        self.assertIn('os.environ.get("QSA_PREFILL_ROUTE", PREFILL_ROUTE_DEFAULT)', T.SRC)
        self.assertEqual(T._const("PREFILL_ROUTE_DEFAULT"), "r2e")


class RouteTable(unittest.TestCase):
    def test_legacy_equals_d438d2b3_condition(self):
        for persist in (True, False):
            for chunks in (2, 4, 8):
                ns = _route_ns(persist=persist, chunks=chunks)
                for cols in (64, 128, 256, 448, 512, 1024, 4096, 65536):
                    for rows in (32, 1024, 16384):
                        for P in (4, 8, 16):
                            want = "persist" if persist and cols >= 64 * chunks else "tdm"
                            self.assertEqual(ns["prefill_bf16_route"](rows, 1, cols, P),
                                             (want, {}), (persist, chunks, cols, rows, P))

    def test_rules(self):
        ns = _route_ns(spec="16,4096,*,*,*,persist;*,*,2048,2048,*,kstat4x2;*,*,*,*,*,kstat1x2")
        r = ns["prefill_bf16_route"]
        self.assertEqual(r(1024, 1, 8192, 16), ("persist", {}))
        self.assertEqual(r(1024, 1, 8192, 4), ("kstat", {"row_tiles": 1, "col_tiles": 2}))
        self.assertEqual(r(4096, 1, 1024, 4), ("kstat", {"row_tiles": 4, "col_tiles": 2}))
        self.assertEqual(r(4096, 1, 4096, 8), ("kstat", {"row_tiles": 1, "col_tiles": 2}))
        veto = _route_ns(persist=False, spec="*,*,*,*,*,persist")
        self.assertEqual(veto["prefill_bf16_route"](64, 1, 1024, 4), ("tdm", {}))
        nomatch = _route_ns(spec="8,*,*,*,*,kstat2x2")
        self.assertEqual(nomatch["prefill_bf16_route"](64, 1, 1024, 4), ("tdm", {}))
        r2b = _route_ns(spec="r2b")["prefill_bf16_route"]
        self.assertEqual(r2b(4096, 1, 65536, 4), ("persist", {}))
        self.assertEqual(r2b(4096, 1, 1024, 4)[0], "kstat")

    def test_bad_specs(self):
        ns = _route_ns()
        for bad in ("*,1,2,3,4", "3,*,*,*,*,tdm", "*,*,*,*,*,kstat3x1", "*,*,*,*,*,kstat1x8",
                    "*,*,*,*,*,r64"):
            with self.assertRaises(ValueError, msg=bad):
                ns["parse_prefill_route"](bad)

    def test_kstat_grid_bounds(self):
        g = _route_ns()["kstat_grid"]
        self.assertEqual(g(1024, 1024, 1, 1), (16, 32))
        self.assertEqual(g(4096, 1024, 4, 2), (8, 32))
        self.assertEqual(g(33, 65, 8, 4), (1, 1))
        with self.assertRaises(ValueError):
            g(32, 64, 3, 1)


if __name__ == "__main__":
    unittest.main()
