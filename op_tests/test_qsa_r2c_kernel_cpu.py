# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: S102, RUF012 -- exec of kernel source fragments for CPU lane emulation; read-only class tables
"""ROUND-R2C kernel CPU gates (no GPU, no FlyDSL): the opt-in 'early' prefetch variants.

1. bf16 kstat<R>x<C>e: the R2B lane emulation (``test_qsa_r2b_kernel_cpu.KstatEmu``) with the
   LDS-path page validity taken from the *real* ``_kstat_page_pre`` helper (executed against
   the fake ``fx`` per workgroup and per lane column n_in = 0..15, loaded once per slab with the
   slab request) instead of the per-row consumer lookup; every LDS-path element additionally
   asserts the prefetched page_ok equals the per-row ``_tile_pair`` value. Logits vs the frozen
   golden for P = 4 / 8 / 16, every (R, C), bad / out-of-range pages, invalid token_to_request,
   multi-request spans, row / column tails.
2. fp8 kernel_r32_early: the real ``_page_scale_pre`` (page-table + per-column scale loads) vs
   the golden fixture's page table / scale plane for every lane column of every r32 block;
   source shape: Q fragments + page/scale prefetch precede the TDM stage inside the uniform
   full-block branch, the fallback branch is kernel_r32's verbatim, no wait/barrier under a
   non-uniform guard.
3. Route table: ``kstat<R>x<C>e`` parses to early; the 'legacy' / 'r2b' presets are unchanged.

    QSA_GOLDEN_DIR=/path/golden_baseline python3 -B op_tests/test_qsa_r2c_kernel_cpu.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import os
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "qsa_r2b_kernel_cpu_for_r2c", os.path.join(HERE, "test_qsa_r2b_kernel_cpu.py")
)
R2B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R2B)
T = R2B.T
G = T.G
FI, FB, _i, Plane, Env = T.FI, T.FB, T._i, T.Plane, T.Env


class FPlane:
    """fp32 plane (per-token scale), bounds-checked like Plane."""

    def __init__(self, vals):
        self.vals = np.asarray(vals, dtype=np.float64).reshape(-1)

    def load(self, byte_off):
        off = _i(byte_off)
        assert off % 4 == 0 and 0 <= off < 4 * self.vals.size, f"scale load OOB {off}"
        return float(self.vals[off // 4])


def _lane_ns(env, P, **extra):
    ns = {
        "fx": env.fx,
        "range_constexpr": range,
        "P": P,
        "WMMA_COLS": 64,
        "WMMA_N": 16,
        "WMMA_N_TILES": 4,
        "_ld_i32": lambda rsrc, off: rsrc.load(off),
        "_ld_f32": lambda rsrc, off: rsrc.load(off),
    }
    ns.update(extra)
    return ns


class KstatEarlyEmu(R2B.KstatEmu):
    """KstatEmu whose LDS-path page_ok comes from the real ``_kstat_page_pre``."""

    def __init__(self, d, P, R, C, columns):
        super().__init__(d, P, R, C, columns)
        ns = _lane_ns(self.env, P, col_tiles=C)
        exec(R2B._nested("_lane_cols"), ns)
        exec(R2B._nested("_kstat_page_pre"), ns)
        self.page_pre = ns["_kstat_page_pre"]
        self.kpre = None
        self.checked = 0

    def run(self):
        orig = self.slab_req

        def slab_req(*a):
            out = orig(*a)
            _kreq, kreq_ok, ksafe, _kvis = out
            self._slab = a[2]
            # per lane column n_in: kpre[n_in][ct][nt]
            self.kpre = [
                self.page_pre(
                    self.ptp,
                    ksafe,
                    kreq_ok,
                    FI(self._x0),
                    FI(n),
                    FI(self.width),
                    FI(self.pages),
                )
                for n in range(16)
            ]
            return out

        self.slab_req = slab_req
        self._x0 = 0
        _P, C, cols = self.P, self.C, self.cols
        gx = (cols + 64 * C - 1) // (64 * C)
        # KstatEmu.run walks y outer, x inner and calls slab_req once per WG: track x.
        seq = [
            x * 64 * C
            for _ in range((self.rows + 32 * self.R - 1) // (32 * self.R))
            for x in range(gx)
        ]
        it = iter(seq)

        def slab_req_x(*a):
            self._x0 = next(it)
            return slab_req(*a)

        self.slab_req = slab_req_x
        return super().run()

    def _elem(self, r, c, lds_path, lds, slab0):
        if lds_path:
            assert slab0 == self._x0
            ct, rem = divmod(c - slab0, 64)
            nt, n_in = divmod(rem, 16)
            page_ok = T._b(self.kpre[n_in][ct][nt])
            want, _ = T._consumer_page(
                self.P, self.pt, int(self.sreq[r]), True, c, self.width, self.pages
            )
            assert page_ok == bool(want), ("prefetched page_ok differs", r, c)
            self.checked += 1
            if not self.ok[r] or c >= int(self.vis[r]) or not page_ok:
                return -1
            off = _i(self.base_tdm(None, None, FI(c), 0))
            return lds[ct * R2B.TILE + off]
        return super()._elem(r, c, lds_path, lds, slab0)


class KstatEarlyEmulation(unittest.TestCase):
    KNOBS = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4)]

    def _check(self, P, name, rows, reqs, ctx, width, pages, edits, columns):
        scale = P // 4
        d = T._fixture(
            P,
            "prefill",
            rows,
            reqs,
            ctx,
            max(width // scale, 4),
            max(pages // scale, 8),
            edits=edits,
        )
        cols_full = int(d["page_table"].shape[1]) * P
        cols = cols_full if columns is None else min(columns, cols_full)
        gold, gvis, valid = T._golden(d, cols)
        inside = valid & (torch.arange(cols)[None, :] < gvis.to(torch.int64)[:, None])
        ins = inside.numpy()
        rr, cc = np.nonzero(ins)
        for R, C in self.KNOBS:
            emu = KstatEarlyEmu(d, P, R, C, cols)
            src = emu.run()
            out = torch.from_numpy(T._scores(d, src)).to(torch.float32)
            vis = torch.from_numpy(emu.vis.astype(np.int32))
            res = G.compare_logits(
                out, vis, gold.to(torch.float32), gvis, valid, rtol=1e-5, atol=1e-5
            )
            self.assertTrue(res["pass"], (P, name, R, C, res))
            self.assertTrue(bool(torch.isneginf(out[~inside]).all()), (P, name, R, C))
            want = (
                d["page_table"].numpy().astype(np.int64)[emu.sreq[rr], cc // P] * P
                + cc % P
            )
            self.assertTrue(np.array_equal(src[rr, cc], want), (P, name, R, C))
            if emu.stats["tiles_lds"]:
                self.assertGreater(emu.checked, 0)

    def test_kstat_early_all_pages(self):
        for P in (4, 8, 16):
            for case in R2B.KstatEmulation.CASES:
                with self.subTest(P=P, case=case[0]):
                    self._check(P, *case[:7], case[7])


def _kt_route_ns():
    ns = R2B._route_ns()
    for c in ("KT_ROWS", "KT_COL_TILES"):
        ns[c] = T._const(c)
    exec(T._top("kt_grid"), ns)
    exec(T._top("_parse_route_variant"), ns)
    exec(T._top("parse_prefill_route"), ns)
    exec(T._top("prefill_bf16_route"), ns)
    return ns


class KtEmu(R2B.KstatEmu):
    """ktile (RB rows x C 64-col tiles, one wave per 4-row M tile) over a golden fixture:
    the real ``_kt_guard``, ``_persist_block_vis``, TDM producers (``_persist_issue_tdm`` for
    4 waves, ``_kt_issue_w8`` for 8), ``_kstat_page_pre`` and the consumer ``_pair_base_tdm``.
    """

    def __init__(self, d, P, RB, C, columns):
        super().__init__(d, P, 1, C, columns)
        self.RB, self.W = RB, RB // 4
        env = self.env
        imin = lambda a, b: FI(min(_i(a), _i(b)))
        ns = {
            "fx": env.fx,
            "range_constexpr": range,
            "_ld_uniform_i32": env.ld_uniform,
            "_imin": imin,
            "kt_rows": RB,
            "COMPRESS_RATIO": 4,
            "_imax": lambda a, b: FI(max(_i(a), _i(b))),
        }
        exec(R2B._nested("_kt_guard"), ns)
        exec(R2B._nested("_persist_block_vis"), ns)
        self.guard, self.bvis = ns["_kt_guard"], ns["_persist_block_vis"]
        name = "_kt_issue_w8" if self.W == 8 else "_persist_issue_tdm"
        self.kt_issue = T._exec(
            env, P, R2B._nested(name), name, _tdm_k_gather=env.gather(R2B.ROW)
        )
        pns = _lane_ns(env, P, col_tiles=C)
        exec(R2B._nested("_lane_cols"), pns)
        exec(R2B._nested("_kstat_page_pre"), pns)
        self.page_pre = pns["_kstat_page_pre"]
        self.stats = {"wg": 0, "full": 0, "fallback": 0, "issues": 0, "lds_elems": 0}

    def run(self):
        _P, RB, C, W, cols, rows = self.P, self.RB, self.C, self.W, self.cols, self.rows
        gx, gy = _kt_route_ns()["kt_grid"](rows, cols, RB, C)
        assert gx * 64 * C >= cols and (gx - 1) * 64 * C < cols
        assert gy * RB >= rows and (gy - 1) * RB < rows
        src = np.full((rows, cols), -2, dtype=np.int64)
        width, pages, nreq, rows_i = (
            FI(self.width),
            FI(self.pages),
            FI(self.nreq),
            FI(rows),
        )
        for y in range(gy):
            for x in range(gx):
                self.stats["wg"] += 1
                slab0, base = x * 64 * C, y * RB
                kreq, full = self.guard(self.t2rp, FI(base), rows_i, nreq)
                full = T._b(full)
                self.env.lds = {}
                kpre = None
                if full:
                    self.stats["full"] += 1
                    kvis = self.bvis(self.clp, kreq)
                    for ct in range(C):
                        c0 = slab0 + ct * 64
                        if T._b((FI(c0) < kvis) & (FI(c0) < FI(cols))):
                            self.stats["issues"] += 1
                            for w in range(W):
                                self.env.wave = w
                                self.kt_issue(
                                    None,
                                    None,
                                    self.ptp,
                                    kreq,
                                    FI(c0),
                                    width,
                                    pages,
                                    FI(ct),
                                )
                    kpre = [
                        self.page_pre(
                            self.ptp, kreq, FB(True), FI(slab0), FI(n), width, pages
                        )
                        for n in range(16)
                    ]
                else:
                    self.stats["fallback"] += 1
                lds = dict(self.env.lds)
                for r in range(base, min(base + RB, rows)):
                    w = (r - base) // 4
                    wrows = range(base + 4 * w, min(base + 4 * w + 4, rows))
                    vmax = max(int(self.vis[q]) for q in wrows)
                    for c in range(slab0, min(slab0 + 64 * C, cols)):
                        assert src[r, c] == -2, ("written twice", r, c)
                        src[r, c] = (
                            self._kt_elem(r, c, slab0, lds, kpre, vmax)
                            if full
                            else R2B.KstatEmu._elem(self, r, c, False, lds, slab0)
                        )
        assert (src != -2).all(), "element never written"
        return src

    def _kt_elem(self, r, c, slab0, lds, kpre, vmax):
        ct, rem = divmod(c - slab0, 64)
        nt, n_in = divmod(rem, 16)
        col0 = slab0 + ct * 64
        if col0 >= vmax:  # mask-only tile of this wave
            return -1
        off = _i(self.base_tdm(None, None, FI(c), 0))
        key = ct * R2B.TILE + off
        assert key in lds, ("LDS row of a scored tile never gathered", r, c)
        page_ok = T._b(kpre[n_in][ct][nt])
        want, _ = T._consumer_page(
            self.P, self.pt, int(self.sreq[r]), True, c, self.width, self.pages
        )
        assert page_ok == bool(want), ("prefetched page_ok differs", r, c)
        if not self.ok[r] or c >= int(self.vis[r]) or not page_ok:
            return -1
        self.stats["lds_elems"] += 1
        return lds[key]


class KtEmulation(unittest.TestCase):
    KNOBS = [(16, 1), (16, 2), (16, 4), (32, 1), (32, 2), (32, 4)]

    def _check(self, P, name, rows, reqs, ctx, width, pages, edits, columns):
        scale = P // 4
        d = T._fixture(
            P,
            "prefill",
            rows,
            reqs,
            ctx,
            max(width // scale, 4),
            max(pages // scale, 8),
            edits=edits,
        )
        cols_full = int(d["page_table"].shape[1]) * P
        cols = cols_full if columns is None else min(columns, cols_full)
        gold, gvis, valid = T._golden(d, cols)
        inside = valid & (torch.arange(cols)[None, :] < gvis.to(torch.int64)[:, None])
        ins = inside.numpy()
        rr, cc = np.nonzero(ins)
        for RB, C in self.KNOBS:
            emu = KtEmu(d, P, RB, C, cols)
            src = emu.run()
            out = torch.from_numpy(T._scores(d, src)).to(torch.float32)
            vis = torch.from_numpy(emu.vis.astype(np.int32))
            res = G.compare_logits(
                out, vis, gold.to(torch.float32), gvis, valid, rtol=1e-5, atol=1e-5
            )
            self.assertTrue(res["pass"], (P, name, RB, C, res))
            self.assertTrue(bool(torch.isneginf(out[~inside]).all()), (P, name, RB, C))
            want = (
                d["page_table"].numpy().astype(np.int64)[emu.sreq[rr], cc // P] * P
                + cc % P
            )
            self.assertTrue(np.array_equal(src[rr, cc], want), (P, name, RB, C))
            if name == "prefill":
                self.assertEqual(
                    emu.stats["fallback"], 0 if rows % RB == 0 else -(-cols // (64 * C))
                )
                self.assertGreater(emu.stats["lds_elems"], 0)

    def test_kt_all_pages_all_knobs(self):
        for P in (4, 8, 16):
            for case in R2B.KstatEmulation.CASES:
                with self.subTest(P=P, case=case[0]):
                    self._check(P, *case[:7], case[7])

    def test_kt_w8_gathers_cover_the_tile_once(self):
        for P in (4, 8, 16):
            d = T._fixture(
                P,
                "prefill",
                64,
                1,
                4096,
                max(256 // (P // 4), 4),
                600,
                edits=R2B._edits_one_req,
            )
            emu = KtEmu(d, P, 32, 2, 512)
            emu.env.gathers = 0
            emu.run()
            self.assertEqual(
                emu.env.gathers, emu.stats["issues"] * 8
            )  # 8 waves x 1 gather
            self.assertEqual(emu.stats["fallback"], 0)

    def test_kt_route_and_grid(self):
        ns = _kt_route_ns()
        self.assertEqual(
            ns["_parse_route_variant"]("kt16x2"),
            ("ktile", {"kt_rows": 16, "col_tiles": 2}),
        )
        for bad in ("kt8x2", "kt16x3", "kt16x2e"):
            with self.assertRaises(ValueError):
                ns["_parse_route_variant"](bad)
        self.assertEqual(
            ns["kt_grid"](1024, 1024, 16, 2), (8, 64)
        )  # native BMT16 x BN128
        self.assertEqual(ns["kt_grid"](1024, 1024, 32, 1), (16, 32))
        rules = ns["parse_prefill_route"]("*,*,*,*,2048,kt16x2;*,*,*,*,*,kstat1x2e")
        self.assertEqual(
            ns["prefill_bf16_route"](1024, 1, 1024, 8, table=rules),
            ("ktile", {"kt_rows": 16, "col_tiles": 2}),
        )


def _fp8_nested(name):
    return T._nested("_build_wmma_fp8_kernel", name)


ROW8 = T.ROW8
TILE8 = 64 * ROW8


class Kt8Emu(KtEmu):
    """fp8 ktile: the real fp8 ``_kt8_guard``, ``_kt8_issue`` (E4M3 TDM, 144 B pitch),
    ``_kt8_pre`` (page validity + per-column scale) and fp8 ``_pair_base_tdm``; the block
    visibility bound is the kernel's one-line max(ctx, 0) // 4. The fallback is the
    unchanged fp8 per-wave `_pass_values` path (address-mapped like KstatEmu)."""

    def __init__(self, d, P, RB, C, columns):
        R2B.KstatEmu.__init__(self, d, P, 1, C, columns)
        self.RB, self.W = RB, RB // 4
        env = self.env
        imin = lambda a, b: FI(min(_i(a), _i(b)))
        ns = {
            "fx": env.fx,
            "range_constexpr": range,
            "_ld_uniform_i32": env.ld_uniform,
            "_imin": imin,
            "kt_rows": RB,
        }
        exec(_fp8_nested("_kt8_guard"), ns)
        self.guard = ns["_kt8_guard"]
        self.bvis = lambda clp, kreq: FI(max(_i(clp.load(_i(kreq) * 4)), 0) // 4)
        self.kt_issue = T._exec(
            env,
            P,
            _fp8_nested("_kt8_issue"),
            "_kt8_issue",
            _tdm_fp8_k_gather=env.gather(ROW8),
            kt_waves=self.W,
            WMMA_COLS=64,
            TDM_FP8_K_TILE_BYTES=TILE8,
        )
        self.ks = d["compressed_k_scale"].reshape(-1).to(torch.float64).numpy()
        ksp = FPlane(self.ks)
        pns = _lane_ns(env, P, col_tiles=C)
        exec(_fp8_nested("_lane_cols"), pns)
        exec(_fp8_nested("_kt8_pre"), pns)
        pre = pns["_kt8_pre"]
        self.page_pre = lambda pt, req, ok, s0, n, w, pg: pre(
            pt, ksp, req, ok, s0, n, w, pg
        )
        bns = {"WMMA_COLS": 64, "TDM_FP8_K_ROW_BYTES": ROW8}
        exec(_fp8_nested("_pair_base_tdm"), bns)
        self.base_tdm8 = bns["_pair_base_tdm"]
        self.stats = {"wg": 0, "full": 0, "fallback": 0, "issues": 0, "lds_elems": 0}

    def _kt_elem(self, r, c, slab0, lds, kpre, vmax):
        ct, rem = divmod(c - slab0, 64)
        nt, n_in = divmod(rem, 16)
        if slab0 + ct * 64 >= vmax:
            return -1
        off = _i(self.base_tdm8(None, None, FI(c), 0))
        assert off % ROW8 == 0
        key = ct * TILE8 + off
        assert key in lds, ("LDS row of a scored tile never gathered", r, c)
        pok, scale = kpre[n_in][ct][nt]
        page_ok = T._b(pok)
        want, phys = T._consumer_page(
            self.P, self.pt, int(self.sreq[r]), True, c, self.width, self.pages
        )
        assert page_ok == bool(want), ("prefetched page_ok differs", r, c)
        sphys = int(_i(phys)) if want else 0
        assert scale == self.ks[sphys * self.P + c % self.P], (
            "prefetched scale differs",
            r,
            c,
        )
        if not self.ok[r] or c >= int(self.vis[r]) or not page_ok:
            return -1
        assert lds[key] == sphys * self.P + c % self.P
        self.stats["lds_elems"] += 1
        return lds[key]


class Kt8Emulation(unittest.TestCase):
    KNOBS = [(16, 1), (16, 2), (32, 1), (32, 2), (16, 4)]

    def test_fp8_kt_all_pages(self):
        for P in (4, 8, 16):
            for (
                name,
                rows,
                reqs,
                ctx,
                width,
                pages,
                edits,
                columns,
            ) in R2B.KstatEmulation.CASES:
                with self.subTest(P=P, case=name):
                    scale = P // 4
                    d = T._fixture(
                        P,
                        "prefill",
                        rows,
                        reqs,
                        ctx,
                        max(width // scale, 4),
                        max(pages // scale, 8),
                        dtype="fp8",
                        edits=edits,
                    )
                    cols_full = int(d["page_table"].shape[1]) * P
                    cols = cols_full if columns is None else min(columns, cols_full)
                    gold, gvis, valid = T._golden(d, cols)
                    inside = valid & (
                        torch.arange(cols)[None, :] < gvis.to(torch.int64)[:, None]
                    )
                    for RB, C in self.KNOBS:
                        emu = Kt8Emu(d, P, RB, C, cols)
                        src = emu.run()
                        out = torch.from_numpy(T._scores(d, src)).to(torch.float32)
                        vis = torch.from_numpy(emu.vis.astype(np.int32))
                        res = G.compare_logits(
                            out,
                            vis,
                            gold.to(torch.float32),
                            gvis,
                            valid,
                            rtol=1e-4,
                            atol=1e-4,
                        )
                        self.assertTrue(res["pass"], (P, name, RB, C, res))
                        self.assertTrue(
                            bool(torch.isneginf(out[~inside]).all()), (P, name, RB, C)
                        )
                        if name == "prefill":
                            self.assertGreater(emu.stats["lds_elems"], 0)


class Fp8EarlyPrefetch(unittest.TestCase):
    def _pp(self, req, ok):
        import types

        return types.SimpleNamespace(safe_req=FI(req if ok else 0), req_ok=FB(ok))

    def test_page_scale_pre_matches_tile_pair(self):
        for P in (4, 8, 16):
            scale = P // 4
            d = T._fixture(
                P,
                "prefill",
                96,
                2,
                2300,
                max(160 // scale, 4),
                max(480 // scale, 8),
                dtype="fp8",
                edits=T._edits_bad_pages,
            )
            pt = d["page_table"].numpy().astype(np.int64)
            pages, width = int(d["compressed_k_cache"].shape[0]), pt.shape[1]
            ks = d["compressed_k_scale"].reshape(-1).to(torch.float64).numpy()
            env = Env()
            ns = _lane_ns(env, P)
            exec(_fp8_nested("_page_scale_pre"), ns)
            fn = ns["_page_scale_pre"]
            ptp, ksp = Plane(pt), FPlane(ks)
            n_checked = 0
            for req, ok in ((0, True), (1, True), (5, False)):
                for col0 in range(0, width * P + 64, 64):
                    for n_in in range(16):
                        cols = [FI(col0 + nt * 16 + n_in) for nt in range(4)]
                        out = fn(
                            ptp, ksp, self._pp(req, ok), cols, FI(width), FI(pages)
                        )
                        for nt, (pok, sc) in enumerate(out):
                            c = col0 + nt * 16 + n_in
                            lg = c // P
                            phys = pt[req if ok else 0, lg if lg < width else 0]
                            want_ok = ok and lg < width and 0 <= phys < pages
                            self.assertEqual(T._b(pok), want_ok, (P, req, c))
                            want_phys = phys if want_ok else 0
                            self.assertEqual(sc, ks[want_phys * P + c % P], (P, req, c))
                            n_checked += 1
            self.assertGreater(n_checked, 1000)

    def _kernel(self, name):
        node = next(
            n
            for n in ast.walk(T._builder("_build_wmma_fp8_kernel"))
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        return node

    def test_prefetch_precedes_stage_in_uniform_branch(self):
        k = self._kernel("kernel_r32_early")
        reuse_if = next(
            n
            for n in ast.walk(k)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "reuse"
        )
        names = []
        for st in reuse_if.body:
            for c in ast.walk(st):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                    names.append(c.func.id)
        i_stage = names.index("pair_stage")
        self.assertLess(names.index("_q_frags"), i_stage)
        self.assertLess(names.index("_page_scale_pre"), i_stage)
        self.assertGreater(names.index("_pair_values_pre"), i_stage)
        # wait / barrier only via pair_stage, directly in the uniform `if reuse:` body
        for n in ast.walk(k):
            if isinstance(n, ast.If) and n is not reuse_if:
                seg = ast.unparse(n)
                self.assertNotIn(
                    "pair_stage", seg if ast.unparse(n.test) != "reuse" else ""
                )
                self.assertNotIn("barrier", seg)
                self.assertNotIn("tensor_wait", seg)

    def test_fallback_branch_is_kernel_r32_verbatim(self):
        def fb(name):
            k = self._kernel(name)
            node = next(
                n
                for n in ast.walk(k)
                if isinstance(n, ast.If) and ast.unparse(n.test) == "reuse == 0"
            )
            return ast.unparse(node)

        self.assertEqual(fb("kernel_r32"), fb("kernel_r32_early"))

    def test_default_off(self):
        self.assertIn('os.environ.get("QSA_FP8_EARLY", "0")', T.SRC)


class RouteTableR2C(unittest.TestCase):
    def test_early_variant(self):
        ns = R2B._route_ns()
        self.assertEqual(
            ns["_parse_route_variant"]("kstat1x1e"),
            ("kstat", {"row_tiles": 1, "col_tiles": 1, "early": True}),
        )
        self.assertEqual(
            ns["_parse_route_variant"]("kstat1x2"),
            ("kstat", {"row_tiles": 1, "col_tiles": 2}),
        )
        for bad in ("kstat1x1ee", "kstat3x1e", "early"):
            with self.assertRaises(ValueError):
                ns["_parse_route_variant"](bad)
        rules = ns["parse_prefill_route"]("*,*,*,*,2048,kstat1x1e;*,*,*,*,*,kstat1x2e")
        self.assertEqual(
            ns["prefill_bf16_route"](1024, 1, 1024, 4, table=rules),
            ("kstat", {"row_tiles": 1, "col_tiles": 1, "early": True}),
        )
        self.assertEqual(
            ns["prefill_bf16_route"](4096, 1, 1024, 16, table=rules),
            ("kstat", {"row_tiles": 1, "col_tiles": 2, "early": True}),
        )

    def test_presets_unchanged(self):
        p = T._const("PREFILL_ROUTE_PRESETS")
        self.assertIsNone(p["legacy"])
        self.assertEqual(
            p["r2b"], "*,4096,*,*,*,persist;*,*,*,2048,*,kstat2x2;*,*,*,*,*,kstat1x2"
        )


if __name__ == "__main__":
    unittest.main()
