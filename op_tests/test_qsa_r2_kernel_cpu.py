# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: S102 -- exec of kernel source fragments for CPU lane emulation
"""ROUND-R2 kernel CPU gates (no GPU, no FlyDSL): P = 8 and the decode-group row modes.

1. P = 8 (KV block 32): the ROUND-PAGE16 lane-level emulator (``test_qsa_page16_cpu``:
   real TDM producer helpers + consumer address helpers extracted with ``ast``, fake
   ``fx``, recording / bounds-checked TDM gathers) on every TDM regime (draft, decode
   group, r16 tail, r32, fp8 r32, r32 persist, raw) vs the frozen golden
   ``reference_logits`` on ``qsa_shapes.make_case(..., page_size=8)`` +
   ``qsa_golden.build_case`` fixtures (unmodified), with bad / out-of-range pages,
   invalid token_to_request, tails and visible < columns; one page-table lookup per
   8-row gather.
2. Decode-group kernel, lane level: the *real* ``_group_meta(_x)``, ``_group_plans(_x)``,
   ``_pick_vis(_x)``, ``_writer_x``, ``_store_offsets``, ``_q_frags`` and ``_issue`` of
   ``_build_decode_group_kernel`` are executed per (group, block, wave, lane) and every
   predicated logits / visible_groups / row_starts store is recorded: MTP verify with
   3 rows per request (group_rows 3, and 4-row dedup groups straddling requests),
   op-level 3 -> 4 token padding (row_mode 'pad': zero stores to padded rows, real rows
   bit-identical to the unpadded 3-row call), base mode (harness sanity), every element
   stored exactly once, values vs the golden.
3. Source shape: P = 4 and P = 16 TDM branches are verbatim vs 5ba88331 (536667d5),
   defaults pass no new keyword, wg_waves column coverage.

    QSA_GOLDEN_DIR=/path/golden_baseline python3 -B op_tests/test_qsa_r2_kernel_cpu.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import subprocess
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "qsa_page16_cpu_harness", os.path.join(HERE, "test_qsa_page16_cpu.py")
)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
G, S = T.G, T.S
FI, FB, _i, Plane, Env = T.FI, T.FB, T._i, T.Plane, T.Env
ROW = T.ROW
BASE_R2 = os.environ.get("QSA_R2_BASE_REV", "5ba88331")
BASE_R2_SHA = "536667d5997670a8b6d8fad54db438b2d958617d2ff5bd9b9edd359c5ea528a1"


# ---------------------------------------------------------------------------
# 1. P = 8 address emulation (reuses the ROUND-PAGE16 harness unchanged)
# ---------------------------------------------------------------------------
class Page8Emulation(unittest.TestCase):
    def _check(self, *case, columns=None):
        T.AddressEmulation._check(self, 8, *case, columns=columns)

    def test_page8_all_regimes(self):
        for (
            name,
            regime,
            rows,
            reqs,
            ctx,
            width,
            pages,
            variants,
            edits,
        ) in T.AddressEmulation.CASES:
            with self.subTest(case=name):
                # same token extent as the P = 16 cases: twice the table width / pages
                self._check(
                    name,
                    regime,
                    rows,
                    reqs,
                    ctx,
                    width * 2,
                    pages * 2,
                    variants
                    + (
                        ("persist",)
                        if regime == "prefill" and "persist" not in variants
                        else ()
                    ),
                    edits,
                )

    def test_page8_tails_and_visible_below_columns(self):
        self._check(
            "tail_cols",
            "prefill",
            40,
            1,
            700,
            128,
            128,
            ("r32", "fp8r32", "r16", "raw"),
            None,
            columns=1000,
        )
        self._check(
            "short_ctx", "decode", 4, 1, 37, 16, 16, ("decg", "r16", "raw"), None
        )
        self._check(
            "odd_cols",
            "draft",
            5,
            5,
            3000,
            100,
            600,
            ("draft", "raw"),
            None,
            columns=777,
        )
        self._check(
            "invalid_t2r_dec",
            "decode",
            8,
            2,
            1200,
            64,
            160,
            ("decg", "r16"),
            T._edits_invalid_t2r,
        )

    def test_page8_one_lookup_per_gather(self):
        d = T._fixture(8, "draft", 1, 1, 256, 32, 64)
        for kind in ("draft", "decg", "r32", "fp8r32", "persist", "r16"):
            emu = T.Emu(d, 8, kind, 64)
            emu._stage(kind, 0, 0, 0)
            self.assertGreater(emu.stats["gathers"], 0)
            self.assertEqual(
                emu.stats["lookups"], emu.stats["gathers"], (kind, emu.stats)
            )

    def test_page8_row_ids_clamps(self):
        f = T._exec(Env(), 8, "", "_p16_row_ids")
        pt = Plane([[5, -1, 99, 3]])
        rows = [_i(x) for x in f(pt, FI(0), FI(4), FI(10), FI(0), 0, None, 8)]
        self.assertEqual(rows, list(range(40, 48)))
        for logical in (1, 2, 4, 7):
            rows = [_i(x) for x in f(pt, FI(0), FI(4), FI(10), FI(logical), 0, None, 8)]
            self.assertEqual(rows, list(range(8)), logical)
        rows = [_i(x) for x in f(pt, FI(0), FI(4), FI(10), FI(3), 0, FB(False), 8)]
        self.assertEqual(rows, list(range(8)))

    def test_tdm_group_plan_p8(self):
        ns = {}
        exec(T._top("_check_page_size"), dict(ns, SUPPORTED_PAGE_SIZES=(4, 8, 16)), ns)
        ns["SUPPORTED_PAGE_SIZES"] = (4, 8, 16)
        exec(T._top("tdm_group_plan"), ns)
        for g in range(0, 64, 8):
            self.assertEqual(ns["tdm_group_plan"](8, g), [(g // 8, list(range(8)))])
            self.assertEqual(
                ns["tdm_group_plan"](16, g), [(g // 16, [g % 16 + r for r in range(8)])]
            )


# ---------------------------------------------------------------------------
# 2. Decode-group lane-level emulation (real trace-time helpers)
# ---------------------------------------------------------------------------
NUM_HEADS, HEAD_DIM = 4, 128


def _decg_ns(P, group_rows, row_mode, qp, env):
    ns = {
        "fx": env.fx,
        "range_constexpr": range,
        "_ld_uniform_i32": env.ld_uniform,
        "_imin": lambda a, b: FI(min(_i(a), _i(b))),
        "_imax": lambda a, b: FI(max(_i(a), _i(b))),
        "COMPRESS_RATIO": 4,
        "DECG_ROWS": 4,
        "DECG_WAVES": 4,
        "group_rows": group_rows,
        "row_mode": row_mode,
        "P": P,
        "NUM_HEADS": NUM_HEADS,
        "q_row_bytes": 1024,
        "head_bytes": 256,
        "kstep_q": 64,
        "WMMA_K_STEPS": 4,
        "_frag": lambda rsrc, off: _i(off),
        "ld_uniform_pos": lambda rsrc, row: FI(int(qp[_i(row)])),
    }
    for top in ("_RowMeta", "_Pass"):
        node = next(
            n for n in T.TREE.body if isinstance(n, ast.ClassDef) and n.name == top
        )
        exec(ast.get_source_segment(T.SRC, node), ns)
    for name in (
        "_i01",
        "_group_meta",
        "_group_plans",
        "_pick_vis",
        "_group_meta_x",
        "_group_plans_x",
        "_writer_x",
        "_pick_vis_x",
        "_head_base",
        "_head_x",
        "_store_offsets",
        "_q_frags",
    ):
        exec(T._nested("_build_decode_group_kernel", name), ns)
    for top in ("decode_group_grid", "decode_group_passes_x"):
        exec(T._top(top), ns)
    return ns


def _verify3(d, requests):
    """Drop row 0 of every 4-row decode request: 3 verify rows / request (positions
    ctx-3..ctx-1), with per-request context lengths so the visible bounds differ."""
    keep = torch.tensor([r for r in range(4 * requests) if r % 4], dtype=torch.int64)
    out = dict(d)
    for k in ("q", "token_to_request", "query_positions"):
        out[k] = d[k][keep].clone()
    cl = d["context_lens"].clone()
    cl -= torch.arange(cl.numel(), dtype=cl.dtype) * 37
    out["context_lens"] = cl
    return out


def _pad3to4(d, requests):
    """Op-level token padding 3 -> 4: a pad row (t2r = -1) after each request's rows."""
    rows = []
    for r in range(requests):
        rows += [3 * r, 3 * r + 1, 3 * r + 2, -1]
    idx = torch.tensor([max(x, 0) for x in rows])
    pad = torch.tensor([x < 0 for x in rows])
    out = dict(d)
    q = d["q"][idx].clone()
    q[pad] = 7.0  # garbage Q on pad rows
    t2r = d["token_to_request"][idx].clone()
    t2r[pad] = -1
    qp = d["query_positions"][idx].clone()
    qp[pad] = 123456
    out.update(q=q, token_to_request=t2r, query_positions=qp)
    return out, [i for i, x in enumerate(rows) if x >= 0]


class DecgEmu:
    """Every store of the decode-group kernel for one launch (bf16, wave_cols 32)."""

    def __init__(self, d, P, group_rows=4, row_mode="base", wg_waves=4, columns=None):
        self.d, self.P = d, P
        kn = {
            "DECG_ROW_MODES": ("base", "dedup", "pad"),
            "_WG_WAVES": (4, 2, 1),
            "DECG_ROWS": 4,
        }
        exec(T._top("_check_decg_knobs"), kn)  # the builder's own normalization
        group_rows, row_mode, wg_waves = kn["_check_decg_knobs"](
            group_rows, row_mode, wg_waves
        )
        self.G, self.mode, self.waves = group_rows, row_mode, wg_waves
        self.pt = d["page_table"].numpy().astype(np.int64)
        self.width = self.pt.shape[1]
        self.columns = (
            self.width * P if columns is None else min(columns, self.width * P)
        )
        self.pages = int(d["compressed_k_cache"].shape[0])
        self.t2r = d["token_to_request"].numpy().astype(np.int64)
        self.qp = d["query_positions"].numpy().astype(np.int64)
        self.nreq = int(d["context_lens"].numel())
        self.rows = self.t2r.size
        self.q = d["q"].double().numpy()
        k = d["compressed_k_cache"].double().numpy()
        if d.get("compressed_k_scale") is not None:
            k = k * d["compressed_k_scale"].double().numpy()[..., None]
        self.k = k.reshape(-1, HEAD_DIM)
        self.emu = T.Emu(d, P, "decg", self.columns)  # real `_issue` producer
        self.env = Env()
        self.ns = _decg_ns(P, group_rows, row_mode, self.qp, self.env)
        self.t2rp = Plane(np.clip(self.t2r, -(2**31), 2**31 - 1), "t2r")
        self.clp = Plane(d["context_lens"].numpy(), "cl")
        self.passes = 0
        self._stages = {}

    def stage(self, req, col0, wave):
        key = (req, col0, wave)
        if key not in self._stages:
            self._stages[key] = self.emu._stage("decg", req, col0, wave)
        return self._stages[key]

    def run(self):
        ns, rows, cols, G = self.ns, self.rows, self.columns, self.G
        base = self.mode == "base"
        logits = np.full((rows, cols), np.nan)
        writes = np.zeros((rows, cols), dtype=np.int64)
        vis_out = np.full(rows, -999, dtype=np.int64)
        meta_writes = np.zeros(rows, dtype=np.int64)
        gx, gy = ns["decode_group_grid"](rows, cols, 32, G, self.waves)
        span = self.waves * 32
        for grp in range(gy):
            t0 = FI(grp * G)
            head = ns["_head_base"] if base else ns["_head_x"]
            n_pass = 4 if base else G
            for bx in range(gx):
                for wave in range(self.waves):
                    col0 = bx * span + wave * 32
                    for lane in range(32):
                        n_in, kgrp = lane % 16, lane // 16
                        plans, my_vis, meta_off, writer = head(
                            self.t2rp,
                            None,
                            self.clp,
                            t0,
                            FI(rows),
                            FI(self.nreq),
                            FI(kgrp),
                            FI(lane),
                            FI(bx),
                            FI(wave),
                        )
                        if writer.v:
                            r = _i(meta_off) // 4
                            vis_out[r] = _i(my_vis)
                            meta_writes[r] += 1
                        tile_cols = [FI(col0 + nt * 16 + n_in) for nt in range(2)]
                        offs = ns["_store_offsets"](
                            t0, FI(kgrp), tile_cols, FI(rows), FI(cols)
                        )
                        qoffs = [
                            ns["_q_frags"](None, t0, FI(rows), FI(m), FI(kgrp))[0]
                            - kgrp * 16
                            for m in range(8 * kgrp, 8 * kgrp + 8)
                        ]
                        qsrc = [
                            (o // 1024, (o % 1024) // 256) for o in qoffs
                        ]  # (row, head)
                        for p in range(n_pass):
                            pp = plans[p]
                            if not pp.active.v:
                                continue
                            if lane == 0:
                                self.passes += 1
                            live = col0 < _i(pp.vmax)
                            if live:
                                lds, lbase = self.stage(
                                    _i(pp.safe_req) if pp.req_ok.v else -1, col0, wave
                                )
                            for nt in range(2):
                                c = col0 + nt * 16 + n_in
                                for half, own, vis, off in (
                                    (0, pp.own_a, pp.vis_a, offs[2 * nt]),
                                    (1, pp.own_b, pp.vis_b, offs[2 * nt + 1]),
                                ):
                                    if not (own.v and c < cols):
                                        continue
                                    e = _i(off) // 4
                                    r, cc = divmod(e, cols)
                                    self.assertEq(cc, c)
                                    self.assertEq(r, _i(t0) + 2 * kgrp + half)
                                    writes[r, cc] += 1
                                    if not live:
                                        logits[r, cc] = -np.inf
                                        continue
                                    ok, _phys = T._consumer_page(
                                        self.P,
                                        self.pt,
                                        _i(pp.safe_req),
                                        pp.req_ok.v,
                                        c,
                                        self.width,
                                        self.pages,
                                    )
                                    if not (ok and c < _i(vis)):
                                        logits[r, cc] = -np.inf
                                        continue
                                    krow = lds[lbase + (nt * 16 + n_in) * ROW]
                                    s = 0.0
                                    for h in range(4):
                                        qr, qh = qsrc[4 * half + h]
                                        self.assertEq(qr, r)
                                        self.assertEq(qh, h)
                                        s += max(
                                            float(self.q[qr, qh] @ self.k[krow]), 0.0
                                        )
                                    logits[r, cc] = s / math.sqrt(128)
        return logits, writes, vis_out, meta_writes

    @staticmethod
    def assertEq(a, b):
        assert a == b, (a, b)


def _gold(d, cols):
    return T._golden(d, cols)


class DecodeGroupRowModes(unittest.TestCase):
    def _compare(self, d, logits, vis_out, cols, rows_idx=None):
        gold, gvis, valid = _gold(d, cols)
        out = torch.from_numpy(logits).to(torch.float32)
        vis = torch.from_numpy(vis_out.astype(np.int32))
        res = G.compare_logits(
            out, vis, gold.to(torch.float32), gvis, valid, rtol=1e-5, atol=1e-5
        )
        self.assertTrue(res["pass"], res)
        inside = valid & (torch.arange(cols)[None, :] < gvis.to(torch.int64)[:, None])
        self.assertTrue(bool(torch.isneginf(out[~inside]).all()))

    def _run(self, d, P, **kw):
        e = DecgEmu(d, P, **kw)
        lg, wr, vis, mw = e.run()
        return e, lg, wr, vis, mw

    def test_verify3_group3_one_pass_per_request(self):
        for P in (4, 8, 16):
            R = 5
            d = _verify3(
                T._fixture(P, "decode", 4 * R, R, 900 + P, 512 // P, R * 512 // P + 8),
                R,
            )
            e, lg, wr, vis, mw = self._run(d, P, group_rows=3)
            self.assertTrue((wr == 1).all(), P)  # every element exactly once
            self.assertTrue((mw == 1).all(), P)
            self._compare(d, lg, vis, e.columns)
            gx, gy = e.ns["decode_group_grid"](e.rows, e.columns, 32, 3, 4)
            self.assertEqual(e.passes, gx * 4 * gy, P)  # ONE pass per (group, wave)
            self.assertEqual(gy, R)

    def test_verify3_dedup4_straddling_groups(self):
        for P in (8, 16):
            R = 4  # rows = 12: groups straddle requests
            d = _verify3(
                T._fixture(
                    P,
                    "decode",
                    4 * R,
                    R,
                    1300,
                    512 // P,
                    R * 512 // P + 8,
                    edits=T._edits_bad_pages,
                ),
                R,
            )
            e, lg, wr, vis, mw = self._run(d, P, row_mode="dedup")
            self.assertTrue((wr == 1).all() and (mw == 1).all())
            self._compare(d, lg, vis, e.columns)
            eb, lgb, _wrb, visb, _ = self._run(
                d, P
            )  # base: 4 single-row passes per mixed group
            self._compare(d, lgb, visb, eb.columns)
            np.testing.assert_array_equal(lg, lgb)
            self.assertLess(e.passes, eb.passes)  # 3 groups: 6 passes vs 12 per slab

    def test_pad_mode_zero_writes_and_bit_identity(self):
        for P in (4, 8, 16):
            R = 3
            d3 = _verify3(
                T._fixture(P, "decode", 4 * R, R, 777, 256 // P, R * 256 // P + 8), R
            )
            _, lg3, _wr3, vis3, _ = self._run(d3, P, group_rows=3)
            d4, real = _pad3to4(d3, R)
            e, lg4, wr4, vis4, mw4 = self._run(d4, P, row_mode="pad")
            pad = [r for r in range(e.rows) if r not in real]
            self.assertTrue(
                (wr4[pad] == 0).all() and (mw4[pad] == 0).all(), P
            )  # zero pad stores
            self.assertTrue((wr4[real] == 1).all() and (mw4[real] == 1).all(), P)
            np.testing.assert_array_equal(lg4[real], lg3)  # bit-identical
            np.testing.assert_array_equal(vis4[real], vis3)
            self._compare(d3, lg3, vis3, e.columns)

    def test_invalid_t2r_and_tails_dedup(self):
        P = 8
        d = T._fixture(P, "decode", 12, 3, 1100, 40, 130, edits=T._edits_invalid_t2r)
        for kw in ({"row_mode": "dedup"}, {"group_rows": 3}, {}):
            e, lg, wr, vis, mw = self._run(d, P, columns=300, **kw)
            self.assertTrue((wr == 1).all() and (mw == 1).all(), kw)
            self._compare(d, lg, vis, e.columns)

    def test_rows_not_multiple_of_group(self):
        P = 8
        R = 4
        d = _verify3(T._fixture(P, "decode", 4 * R, R, 600, 32, 140), R)
        for k in ("q", "token_to_request", "query_positions"):
            d[k] = d[k][:10].clone()  # 10 rows: tail group of 1 row (G=3)
        for kw in ({"group_rows": 3}, {"row_mode": "dedup"}):
            e, lg, wr, vis, mw = self._run(d, P, **kw)
            self.assertTrue((wr == 1).all() and (mw == 1).all(), kw)
            self._compare(d, lg, vis, e.columns)

    def test_wg_waves_coverage(self):
        P = 8
        d = T._fixture(P, "decode", 8, 2, 700, 24, 64)
        ref = None
        for w in (4, 2, 1):
            _e, lg, wr, _vis, _mw = self._run(d, P, wg_waves=w, row_mode="dedup")
            self.assertTrue((wr == 1).all(), w)
            if ref is None:
                ref = lg
            np.testing.assert_array_equal(lg, ref)

    def test_passes_mirror(self):
        ns = _decg_ns(8, 4, "dedup", np.zeros(1), Env())
        f = ns["decode_group_passes_x"]
        self.assertEqual(
            f([0, 0, 0, 1], [1, 1, 1, 1]),
            [(True, [0, 1, 2]), (False, [0, 1, 2]), (False, [0, 1, 2]), (True, [3])],
        )
        self.assertEqual(
            [a for a, _ in f([2, 2, -1, -1], [1, 1, 0, 0])], [True, False, False, False]
        )


# ---------------------------------------------------------------------------
# 3. Source shape vs 5ba88331 (kernel 536667d5)
# ---------------------------------------------------------------------------
class SourceShapeR2(unittest.TestCase):
    def _base(self):
        try:
            return subprocess.check_output(
                ["git", "show", f"{BASE_R2}:{T.KREL}"], cwd=T.ROOT
            ).decode()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"git base {BASE_R2} unavailable: {exc}")

    def test_base_sha(self):
        import hashlib

        self.assertEqual(hashlib.sha256(self._base().encode()).hexdigest(), BASE_R2_SHA)

    def test_p4_p16_branches_verbatim(self):
        base_tree = ast.parse(self._base())

        def sites(tree):
            out = []
            for fn in (
                "_build_wmma_kernel",
                "_build_wmma_fp8_kernel",
                "_build_draft_kernel",
                "_build_decode_group_kernel",
            ):
                node = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == fn
                )
                # ROUND-R2C: the new ktile 8-wave producer is a new site (own lane emulation)
                skip = {
                    id(x)
                    for f in ast.walk(node)
                    if isinstance(f, ast.FunctionDef)
                    and f.name
                    in ("_kt_issue_w8", "_kt8_issue", "_ktl_pre")  # + R2D lean
                    for x in ast.walk(f)
                }
                out += [
                    n
                    for n in ast.walk(node)
                    if id(n) not in skip
                    and isinstance(n, ast.If)
                    and ast.unparse(n.test) == "P == 4"
                ]
            return out

        new, old = sites(T.TREE), sites(base_tree)
        self.assertEqual(len(new), 6)
        self.assertEqual(len(old), 6)
        for a, b in zip(new, old):
            body = lambda xs: ast.unparse(ast.Module(body=xs, type_ignores=[]))
            self.assertEqual(body(a.body), body(b.body))  # P = 4 verbatim
            self.assertEqual(len(a.orelse), 1)
            p16 = a.orelse[0]
            self.assertEqual(ast.unparse(p16.test), "P == 16")
            self.assertEqual(body(p16.body), body(b.orelse))  # P = 16 verbatim
            self.assertIn("8)", body(p16.orelse))  # P = 8 branch

    def test_defaults_pass_no_new_keyword(self):
        # AITER-LINT: whitespace-insensitive (the kernel source is black-formatted)
        flat = lambda s: "".join(s.split())
        src = flat(T.SRC)
        self.assertIn(
            flat(
                'dg_kw = (_NO_KW if (dg_rows, dg_mode, dg_waves) == (DECG_ROWS, "base", '
                "DECG_WAVES)"
            ),
            src,
        )
        # ROUND-R2E: per-dtype draft waves ('auto': bf16 4 = no keyword, fp8 2)
        self.assertIn(
            flat('dr_kw = _NO_KW if dr_w == DRAFT_WAVES else {"wg_waves": dr_w}'), src
        )
        self.assertEqual(T._const("DRAFT_AUTO_WG_WAVES"), {False: 4, True: 2})
        self.assertIn(
            flat('_PAGE_KW = {8: {"page_size": 8}, 16: {"page_size": 16}}'), src
        )
        for env, default in (
            ("QSA_DECG_ROWS3", '"0"'),
            ("QSA_DECG_MODE", '"base"'),
            ("QSA_DECG_WG_WAVES", '"4"'),
            ("QSA_DRAFT_WG_WAVES", '"auto"'),
        ):
            self.assertIn(flat(f'os.environ.get("{env}", {default})'), src, env)

    def test_draft_wg_waves_coverage(self):
        ns = {"DRAFT_WAVES": 4}
        exec(T._top("draft_grid"), ns)
        for w in (4, 2, 1):
            for cols in (1, 31, 1000, 1024, 4096):
                gx, _gy = ns["draft_grid"](3, cols, 32, w)
                seen = [
                    c
                    for bx in range(gx)
                    for wave in range(w)
                    for j in range(1)
                    for lane in range(32)
                    for c in [bx * w * 32 + wave * 32 + 32 * j + lane]
                    if c < cols
                ]
                self.assertEqual(sorted(seen), list(range(cols)), (w, cols))


if __name__ == "__main__":
    unittest.main()
