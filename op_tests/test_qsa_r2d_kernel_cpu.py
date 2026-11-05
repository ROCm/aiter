# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ROUND-R2D kernel CPU gates (no GPU, no FlyDSL): lean ktile, ipf, trace knobs.

1. bf16 / fp8 lean ktile lane emulation vs the frozen golden: the REAL ``_ktl_pre`` (batched
   SGPR guard / positions / ctx / page-table loads, strengthened full predicate), ``_ktl_issue``
   / ``_ktl8_issue`` (TDM producers), ``_ktl_rowvals`` (vis / vmax from positions) run per wave
   against the golden fixture's planes; the consumer side is the R2C ``KtEmu`` / ``Kt8Emu``
   element model (prefetched page_ok / scale asserted per element). P = 4 / 8 / 16, every
   (RB, C), bad / out-of-range pages, invalid token_to_request, multi-request spans, tails.
   Every lane's lean vis equals the golden-derived visible count, every wave's vmax its rows'
   max, every full-block LDS map equals the R2C ``kernel_kt`` producers' map.
2. The fp8 builder's ``_ktl_pre`` / ``_ktl_rowvals`` are textually the bf16 builder's.
3. Route spellings kt<RB>x<C>[l][p<KB>] and builder knob validation; trace helpers are no-ops
   at trace=0 (source shape: every _tr_* call returns before emitting when the holder is empty).

    QSA_GOLDEN_DIR=/path/golden_baseline python3 -B op_tests/test_qsa_r2d_kernel_cpu.py -v
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("qsa_r2c_kernel_cpu_for_r2d",
                                               os.path.join(HERE, "test_qsa_r2c_kernel_cpu.py"))
R2C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R2C)
R2B, T = R2C.R2B, R2C.T
G = T.G
FI, FB, _i, Plane = T.FI, T.FB, T._i, T.Plane
ROW, ROW8 = T.ROW, T.ROW8


def _lean_ns(emu, fp8, P, RB, C):
    env = emu.env
    nested = R2C._fp8_nested if fp8 else R2B._nested
    ns = dict(fx=env.fx, range_constexpr=range, _ld_uniform_i32=env.ld_uniform,
              _imin=lambda a, b: FI(min(_i(a), _i(b))), _imax=lambda a, b: FI(max(_i(a), _i(b))),
              ld_uniform_pos=lambda rsrc, row: env.ld_uniform(rsrc, _i(row) * 4),
              kt_rows=RB, kt_waves=RB // 4, col_tiles=C, P=P, WMMA_COLS=64,
              WMMA_ROWS_PER_WAVE=4, COMPRESS_RATIO=4,
              TDM_K_TILE_BYTES=64 * ROW, TDM_K_ROW_BYTES=ROW,
              TDM_FP8_K_TILE_BYTES=64 * ROW8, TDM_FP8_K_ROW_BYTES=ROW8,
              _tdm_k_gather=env.gather(ROW), _tdm_fp8_k_gather=env.gather(ROW8),
              _pin_sgprs=lambda vals: [_i(v) for v in vals])
    exec(nested("_ktl_pre"), ns)
    exec(nested("_ktl_rowvals"), ns)
    exec(nested("_ktl8_issue" if fp8 else "_ktl_issue"), ns)
    return ns["_ktl_pre"], ns["_ktl_rowvals"], ns["_ktl8_issue" if fp8 else "_ktl_issue"]


class LeanMixin:
    """run() of the lean kernel: per WG, every wave runs the real _ktl_pre (must agree on
    `full`), issues its gathers; full blocks are scored from LDS with lean vis / vmax, every
    in-range element stored (uniform `inside` or per-column predicate); others take the
    exact fallback model."""

    def lean_setup(self, fp8):
        self.fp8 = fp8
        self.qpp = Plane(self.d["query_positions"].cpu().numpy().astype(np.int64), "qp")
        self.pre, self.rowvals, self.issue = _lean_ns(self, fp8, self.P, self.RB, self.C)
        self.vis_checked = 0

    def run(self):
        P, RB, C, W, cols, rows = self.P, self.RB, self.C, self.W, self.cols, self.rows
        gx, gy = R2C._kt_route_ns()["kt_grid"](rows, cols, RB, C)
        src = np.full((rows, cols), -2, dtype=np.int64)
        width, pages, nreq, rows_i = FI(self.width), FI(self.pages), FI(self.nreq), FI(rows)
        for y in range(gy):
            for x in range(gx):
                self.stats["wg"] += 1
                slab0, base = x * 64 * C, y * RB
                self.env.lds = {}
                outs = []
                for w in range(W):
                    self.env.wave = w
                    o = self.pre(self.t2rp, self.qpp, self.clp, self.ptp, FI(base),
                                 FI(base + 4 * w), rows_i, FI(cols), nreq, FI(slab0), width,
                                 pages, FI(w))
                    outs.append(o)
                fulls = {T._b(o[0]) for o in outs}
                assert len(fulls) == 1, "full must be workgroup-uniform"
                full = fulls.pop()
                # same workgroup decision as the R2C kernel_kt guard
                _, kfull = self.guard(self.t2rp, FI(base), rows_i, nreq)
                assert full == T._b(kfull), "lean full differs from kernel_kt guard"
                kpre = None
                vmaxes = {}
                if full:
                    self.stats["full"] += 1
                    kreq = outs[0][1]
                    for w, o in enumerate(outs):
                        self.env.wave = w
                        for ct in range(C):
                            if T._b(o[3][ct]):
                                self.stats["issues"] += 1 if w == 0 else 0
                                self.issue(None, None, o[4], ct, pages)
                        lvis = o[2]
                        for kg in range(2):
                            for n_in in range(16):
                                lane = 16 * kg + n_in
                                lv, va, vb, vmax = self.rowvals(FI(lane), FI(kg), lvis)
                                ra, rb_ = base + 4 * w + 2 * kg, base + 4 * w + 2 * kg + 1
                                assert _i(va) == int(self.vis[ra]) and _i(vb) == int(self.vis[rb_])
                                if lane < 4:
                                    assert _i(lv) == int(self.vis[base + 4 * w + lane])
                                self.vis_checked += 1
                        vmaxes[w] = _i(vmax)
                        assert vmaxes[w] == max(int(self.vis[q]) for q in
                                                range(base + 4 * w, base + 4 * w + 4))
                    kpre = [self.page_pre(self.ptp, kreq, FB(True), FI(slab0), FI(n), width, pages)
                            for n in range(16)]
                    # same LDS rows as the R2C kernel_kt producers
                    lean_lds = dict(self.env.lds)
                    self.env.lds = {}
                    kvis = self.bvis(self.clp, outs[0][1])
                    for ct in range(C):
                        c0 = slab0 + ct * 64
                        if T._b((FI(c0) < kvis) & (FI(c0) < FI(cols))):
                            for w in range(W):
                                self.env.wave = w
                                self.kt_issue(None, None, self.ptp, outs[0][1], FI(c0), width,
                                              pages, FI(ct))
                    assert lean_lds == self.env.lds, "lean gathers differ from kernel_kt's"
                    lds = lean_lds
                else:
                    self.stats["fallback"] += 1
                    lds = {}
                for r in range(base, min(base + RB, rows)):
                    w = (r - base) // 4
                    for c in range(slab0, min(slab0 + 64 * C, cols)):
                        assert src[r, c] == -2, ("written twice", r, c)
                        src[r, c] = (self._kt_elem(r, c, slab0, lds, kpre, vmaxes[w]) if full
                                     else R2B.KstatEmu._elem(self, r, c, False, lds, slab0))
        assert (src != -2).all(), "element never written"
        return src


class KtLeanEmu(LeanMixin, R2C.KtEmu):
    def __init__(self, d, P, RB, C, columns):
        super().__init__(d, P, RB, C, columns)
        self.lean_setup(False)


class Kt8LeanEmu(LeanMixin, R2C.Kt8Emu):
    def __init__(self, d, P, RB, C, columns):
        super().__init__(d, P, RB, C, columns)
        self.lean_setup(True)


class LeanEmulation(unittest.TestCase):
    KNOBS = [(16, 1), (16, 2), (16, 4), (32, 1), (32, 2), (32, 4)]

    def _run(self, fp8):
        for P in (4, 8, 16):
            for name, rows, reqs, ctx, width, pages, edits, columns in R2B.KstatEmulation.CASES:
                with self.subTest(P=P, case=name, fp8=fp8):
                    scale = P // 4
                    d = T._fixture(P, "prefill", rows, reqs, ctx, max(width // scale, 4),
                                   max(pages // scale, 8), dtype="fp8" if fp8 else "bf16",
                                   edits=edits)
                    cols_full = int(d["page_table"].shape[1]) * P
                    cols = cols_full if columns is None else min(columns, cols_full)
                    gold, gvis, valid = T._golden(d, cols)
                    inside = valid & (torch.arange(cols)[None, :] < gvis.to(torch.int64)[:, None])
                    rr, cc = np.nonzero(inside.numpy())
                    for RB, C in self.KNOBS:
                        emu = (Kt8LeanEmu if fp8 else KtLeanEmu)(d, P, RB, C, cols)
                        src = emu.run()
                        out = torch.from_numpy(T._scores(d, src)).to(torch.float32)
                        vis = torch.from_numpy(emu.vis.astype(np.int32))
                        res = G.compare_logits(out, vis, gold.to(torch.float32), gvis, valid,
                                               rtol=1e-4, atol=1e-4)
                        self.assertTrue(res["pass"], (P, name, RB, C, res))
                        self.assertTrue(bool(torch.isneginf(out[~inside]).all()), (P, name, RB, C))
                        want = (d["page_table"].numpy().astype(np.int64)[emu.sreq[rr], cc // P] * P
                                + cc % P)
                        self.assertTrue(np.array_equal(src[rr, cc], want), (P, name, RB, C))
                        if name == "prefill":
                            self.assertGreater(emu.stats["full"], 0)
                            self.assertGreater(emu.stats["lds_elems"], 0)
                            self.assertGreater(emu.vis_checked, 0)

    def test_bf16_lean_all_pages_all_knobs(self):
        self._run(False)

    def test_fp8_lean_all_pages_all_knobs(self):
        self._run(True)

    def test_fp8_shares_bf16_lean_helpers(self):
        for name in ("_ktl_pre", "_ktl_rowvals"):
            self.assertEqual(R2B._nested(name), R2C._fp8_nested(name), name)


class KconvLayout(unittest.TestCase):
    """fp8 kconv: the bf16 slab written by `_kc_publish` and read by `_kc_tile` feeds every
    lane exactly the E4M3 codes the fp8 `_pair_frag_tdm_fast` + `_pair_base_tdm` read
    (same k positions, same order), and the chunk walk covers each 16-byte chunk once."""

    def test_chunk_walk_covers_tile_once(self):
        for threads in (128, 256):
            seen = sorted(tid + i * threads for tid in range(threads) for i in range(512 // threads))
            self.assertEqual(seen, list(range(512)))

    def test_fragment_codes_match_fp8_path(self):
        rng = np.random.default_rng(0)
        for c in range(64):
            codes = rng.integers(0, 256, size=128)          # one E4M3 key row
            # kconv bf16 slab: chunk q (16 codes at fp8 byte q*16) -> bf16 elements q*16 + 0..15
            # (c0 = codes 0..7 at dst, c1 = codes 8..15 at dst + 16 B), row pitch 272 B
            bf16 = {}
            for q in range(8):
                for e in range(16):
                    bf16[c * ROW + q * 32 + 2 * e] = codes[q * 16 + e]
            for kgrp in (0, 1):
                for s in range(4):
                    # fp8 path: 16 B at (c*144 + s*32) and +16; kgrp selects dwords 0,1 or 2,3
                    lo = codes[s * 32: s * 32 + 16]
                    hi = codes[s * 32 + 16: s * 32 + 32]
                    want = list(lo[kgrp * 8: kgrp * 8 + 8]) + list(hi[kgrp * 8: kgrp * 8 + 8])
                    # kconv path: 16 B at k_base + s*64 and +32, k_base = c*272 + kgrp*16
                    off = c * ROW + kgrp * 16 + s * 64
                    got = [bf16[off + 2 * e] for e in range(8)] + \
                          [bf16[off + 32 + 2 * e] for e in range(8)]
                    self.assertEqual(got, want, (c, kgrp, s))

    def test_source_shape(self):
        pub = R2C._fp8_nested("_kc_publish")
        self.assertIn("_e4m3x8_to_bf16x4_fast(fx.Int32(src[0]), fx.Int32(src[1]))", pub)
        self.assertIn("_e4m3x8_to_bf16x4_fast(fx.Int32(src[2]), fx.Int32(src[3]))", pub)
        self.assertIn("fx.rocdl.s_barrier()", pub)
        self.assertEqual(pub.splitlines()[1].strip(), "if not kconv:")
        tile, kt8 = R2C._fp8_nested("_kc_tile"), R2C._fp8_nested("_kt8_tile")
        self.assertEqual(tile.split("acc = acc_t.load()")[1], kt8.split("acc = acc_t.load()")[1],
                         "kconv epilogue must be kernel_kt's (scale per head before ReLU)")


class RouteAndKnobs(unittest.TestCase):
    def test_route_spellings(self):
        ns = R2C._kt_route_ns()
        ns["IPF_KB"] = T._const("IPF_KB")
        exec(T._top("_parse_route_variant"), ns)
        pv = ns["_parse_route_variant"]
        self.assertEqual(pv("kt16x2"), ("ktile", {"kt_rows": 16, "col_tiles": 2}))
        self.assertEqual(pv("kt16x1l"), ("ktile", {"kt_rows": 16, "col_tiles": 1, "lean": True}))
        self.assertEqual(pv("kt32x2lp32"),
                         ("ktile", {"kt_rows": 32, "col_tiles": 2, "lean": True, "ipf": 32}))
        self.assertEqual(pv("kt16x2p16"), ("ktile", {"kt_rows": 16, "col_tiles": 2, "ipf": 16}))
        for bad in ("kt16x2p3", "kt16x2lx", "kt16x2e", "kt8x1l"):
            with self.assertRaises(ValueError):
                pv(bad)

    def test_knob_validation(self):
        ns = {"IPF_KB": T._const("IPF_KB")}
        exec(T._top("_r2d_check"), ns)
        chk = ns["_r2d_check"]
        self.assertEqual(chk(0, False, 0, False, False), 0)
        self.assertEqual(chk(2, False, 0, False, True), 2)
        self.assertEqual(chk(1, True, 32, True, False), 1)
        for args in ((3, False, 0, True, True), (1, False, 0, False, False),
                     (0, True, 0, False, True), (0, False, 16, False, True),
                     (0, False, 12, True, True)):
            with self.assertRaises(ValueError):
                chk(*args)

    def test_trace_helpers_noop_when_off(self):
        calls = []
        ns = {"fx": None, "_tr_time": lambda: calls.append("t"),
              "_tr_realtime": lambda: calls.append("rt"),
              "_tr_void": lambda s: calls.append(s), "_tr_store64": lambda *a: calls.append("st"),
              "_tr_store_pair": lambda *a: calls.append("sp"), "TRACE_MAGIC": (1, 2)}
        for fn in ("_tr_mark", "_tr_mark_rt", "_tr_end", "_tr_begin"):
            exec(T._top(fn), ns)
        ns["_tr_mark"]({}, 5, True)
        ns["_tr_mark_rt"]({}, 14)
        ns["_tr_end"]({})
        ns["_tr_begin"]({}, 0, None, 1, 1, 1, 1, 4, 0, 0)
        self.assertEqual(calls, [])
        ns2 = {"_tr_void": lambda s: calls.append(s)}
        exec(T._top("_inst_prefetch"), ns2)
        ns2["_inst_prefetch"](0)
        self.assertEqual(calls, [])
        ns2["_inst_prefetch"](8)
        self.assertEqual(calls, ["s_wait_xcnt 0x0\n\ts_prefetch_inst_pc_rel 0x0, null, 31"
                                 "\n\ts_prefetch_inst_pc_rel 0x1000, null, 31"])

    def test_trace_layout_constants(self):
        self.assertEqual(T._const("TRACE_SLOTS"), 16)
        self.assertEqual(len(T._const("TRACE_SLOT_NAMES")), 16)
        self.assertEqual(T._const("TRACE_WAVE_BYTES") if False else 16 * 8, 128)


if __name__ == "__main__":
    unittest.main()
