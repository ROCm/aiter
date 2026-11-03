# SPDX-License-Identifier: MIT
"""Round DRAFT CPU gates: routing, store/visibility plan, lane-mapping emulation.

No device and no FlyDSL needed (stub loader). Each environment variant loads a
fresh module copy because every switch is read once at import time.

    python3 -B test_draft_cpu.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
import unittest
from collections import Counter
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import _load  # noqa: E402

KEYS = ("QSA_DRAFT_PATH", "QSA_DRAFT_K_LOAD", "QSA_DRAFT_COLS", "QSA_G2_ACCESS",
        "QSA_FP8_PATH", "QSA_BF16_PATH", "QSA_R16_NOBAR")
_N = [0]


def load(**env):
    clean = {k: v for k, v in os.environ.items() if k not in KEYS}
    clean.update(env)
    with mock.patch.dict(os.environ, clean, clear=True):
        _N[0] += 1
        spec = importlib.util.spec_from_file_location(f"qsa_draft_{_N[0]}", _load.KERNEL_PATH)
        mod = importlib.util.module_from_spec(spec)
        try:
            import flydsl  # noqa: F401
            spec.loader.exec_module(mod)
        except ModuleNotFoundError:
            installed = _load._make_stub()
            sys.modules.update(installed)
            try:
                spec.loader.exec_module(mod)
            finally:
                for name in installed:
                    sys.modules.pop(name, None)
        return mod


class Routing(unittest.TestCase):
    def test_default_on(self):
        # ROUND-MERGE1: QSA_DRAFT_PATH defaults to 1 (TDM K, 32-column wave slabs).
        k = load()
        self.assertTrue(k._DRAFT_PATH)
        self.assertEqual(k._DRAFT_K_LOAD, "tdm")
        self.assertEqual(k.draft_wave_cols("tdm"), 32)
        for rows, reqs in ((16, 16), (1, 1), (5, 7)):
            for fp8 in (False, True):
                self.assertEqual(k.draft_route(rows, reqs, fp8, 0x1000), ("tdm", 32))
        for fp8 in (False, True):
            self.assertIsNone(k.draft_route(64, 16, fp8, 0x1000))

    def test_opt_out(self):
        k = load(QSA_DRAFT_PATH="0")
        self.assertFalse(k._DRAFT_PATH)
        for rows, reqs in ((16, 16), (1, 1), (64, 16), (5, 7)):
            for fp8 in (False, True):
                self.assertIsNone(k.draft_route(rows, reqs, fp8, 0x1000))

    def test_opt_in_routes(self):
        k = load(QSA_DRAFT_PATH="1")
        self.assertEqual(k.draft_route(16, 16, False, 0x1000), ("tdm", 32))
        self.assertEqual(k.draft_route(16, 16, True, 0x1000), ("tdm", 32))
        self.assertEqual(k.draft_route(3, 9, True, 0x1000), ("tdm", 32))
        # decode / prefill keep the dedup kernels
        self.assertIsNone(k.draft_route(64, 16, False, 0x1000))
        self.assertIsNone(k.draft_route(17, 16, True, 0x1000))
        # TDM needs a 128 B aligned cache base, else the raw draft variant
        self.assertEqual(k.draft_route(16, 16, False, 0x1010), ("raw", 64))

    def test_knobs(self):
        self.assertEqual(load(QSA_DRAFT_PATH="1", QSA_DRAFT_K_LOAD="raw")
                         .draft_route(16, 16, True, 0), ("raw", 64))
        self.assertEqual(load(QSA_DRAFT_PATH="1", QSA_DRAFT_COLS="64")
                         .draft_route(16, 16, False, 0), ("tdm", 64))
        self.assertEqual(load(QSA_DRAFT_PATH="1", QSA_DRAFT_K_LOAD="raw", QSA_DRAFT_COLS="32")
                         .draft_route(16, 16, False, 0), ("raw", 32))
        self.assertIsNone(load(QSA_DRAFT_PATH="1", QSA_FP8_PATH="scalar")
                          .draft_route(16, 16, True, 0))
        self.assertIsNone(load(QSA_DRAFT_PATH="1", QSA_BF16_PATH="scalar")
                          .draft_route(16, 16, False, 0))
        self.assertIsNone(load(QSA_DRAFT_PATH="1", QSA_G2_ACCESS="cop")
                          .draft_route(16, 16, False, 0))
        for key, bad in (("QSA_DRAFT_PATH", "2"), ("QSA_DRAFT_K_LOAD", "lds"),
                         ("QSA_DRAFT_COLS", "48")):
            with self.assertRaises(ValueError):
                load(**{key: bad})

    def test_wrapper_block_is_flag_gated_and_before_original_branches(self):
        tree = ast.parse(open(_load.KERNEL_PATH, encoding="utf-8").read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "qsa_paged_mqa_logits_page4")
        tests = [ast.unparse(n.test) for n in fn.body if isinstance(n, ast.If)]
        i_draft = tests.index("_DRAFT_PATH")
        self.assertLess(i_draft, tests.index("not is_fp8 and _BF16_PATH == 'wmma'"))
        self.assertLess(i_draft, tests.index("is_fp8 and _FP8_PATH == 'wmma'"))
        block = next(n for n in fn.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "_DRAFT_PATH")
        calls = {ast.unparse(c.func) for c in ast.walk(block) if isinstance(c, ast.Call)}
        self.assertIn("_build_draft_kernel", calls)
        self.assertIn("draft_route", calls)


def _builder_fn(name):
    tree = ast.parse(open(_load.KERNEL_PATH, encoding="utf-8").read())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_build_wmma_kernel")
    return {n.name: n for n in ast.walk(builder) if isinstance(n, ast.FunctionDef)}[name]


class NoBarrier(unittest.TestCase):
    def test_default_on_and_knob(self):
        # ROUND-MERGE1: QSA_R16_NOBAR defaults to 1; 0 restores the barriered kernel.
        self.assertTrue(load().TDM_R16_NOBAR)
        self.assertTrue(load(QSA_R16_NOBAR="1").TDM_R16_NOBAR)
        self.assertFalse(load(QSA_R16_NOBAR="0").TDM_R16_NOBAR)
        with self.assertRaises(ValueError):
            load(QSA_R16_NOBAR="yes")

    def test_nobar_kernel_is_champion_minus_two_barriers(self):
        champ = _builder_fn("kernel_tdm_r16")
        nobar = _builder_fn("kernel_tdm_r16_nobar")
        a = ast.unparse(ast.Module(body=champ.body, type_ignores=[]))
        nb = [s for s in nobar.body if not (isinstance(s, ast.Expr)
                                            and isinstance(s.value, ast.Constant))]
        b = ast.unparse(ast.Module(body=nb, type_ignores=[]))
        self.assertEqual(a.count("fx.gpu.barrier()"), 2)
        self.assertEqual(b.count("fx.gpu.barrier()"), 0)
        self.assertEqual(b.count("r16_publish()"), 1)
        self.assertEqual(b.count("r16_reuse()"), 1)
        # identical except the two hooks (comments are not in the AST)
        a2 = a.replace("fx.gpu.barrier()", "HOOK", 2)
        b2 = b.replace("r16_publish()", "HOOK").replace("r16_reuse()", "HOOK")
        self.assertEqual(a2, b2)
        self.assertEqual([ast.unparse(x) for x in champ.args.args],
                         [ast.unparse(x) for x in nobar.args.args])

    def test_tiles_are_wave_private(self):
        stage = ast.unparse(_builder_fn("_r16_stage_tdm"))
        self.assertIn("wave_id = fx.Int32(fx.rocdl.wave_id())", stage)
        self.assertIn("byte_off = wave_id * tdm_tile_bytes + local * TDM_K_ROW_BYTES", stage)
        for name in ("kernel_tdm_r16", "kernel_tdm_r16_nobar"):
            body = ast.unparse(_builder_fn(name))
            self.assertIn("wave = fx.Int32(fx.rocdl.wave_id())", body)
            self.assertIn("kptr = storage + wave * (tdm_tile_bytes // 4)", body)
        hooks = {n: ast.unparse(_builder_fn(n)) for n in ("_r16_nop", "_r16_dscnt0", "_r16_barrier")}
        self.assertIn("fx.rocdl.s_wait_dscnt(0)", hooks["_r16_dscnt0"])
        self.assertIn("fx.gpu.barrier()", hooks["_r16_barrier"])

    def test_selection_and_name(self):
        src = open(_load.KERNEL_PATH, encoding="utf-8").read()
        self.assertIn("r16_nobar = (globals().get(\"TDM_R16_NOBAR\", False) and use_tdm and rows_per_block == 16\n                 and not persist)", src)
        self.assertIn("    if r16_nobar:\n        selected_kernel = kernel_tdm_r16_nobar\n", src)
        self.assertIn("    if r16_nobar:\n        kernel_name = kernel_name + \"_nobar\"\n", src)
        # ROUND-MERGE1: the nobar kernel also gets the pos64 suffix (..._nobar_qp64).
        self.assertIn("        kernel_name = kernel_name + \"_nobar\"\n    kernel_name = kernel_name + qp_tag\n", src)


class DraftPos64(unittest.TestCase):
    """ROUND-MERGE1: the draft builder reads positions through ``_pos_loaders``."""

    def _draft_src(self):
        tree = ast.parse(open(_load.KERNEL_PATH, encoding="utf-8").read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_build_draft_kernel")
        return fn, ast.unparse(fn)

    def test_builder_uses_seam(self):
        fn, src = self._draft_src()
        self.assertEqual([a.arg for a in fn.args.args],
                         ["is_fp8", "divisor", "k_load", "wave_cols", "pos64"])
        self.assertEqual(ast.unparse(fn.args.defaults[-1]), "False")
        self.assertIn("_, ld_uniform_pos, qp_bytes, qp_tag", src)
        self.assertIn(" = _pos_loaders(pos64)", src)
        self.assertIn("qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))", src)
        self.assertIn("pos = ld_uniform_pos(qp_rsrc, safe_row)", src)
        self.assertIn("+ qp_tag", src)
        # every QP access goes through the seam (no raw dword position load left)
        self.assertNotIn("_ld_uniform_i32(qp_rsrc", src)
        self.assertNotIn("_ld_i32(qp_rsrc", src)
        self.assertEqual(src.count("qp_rsrc"), 2)

    def test_int32_binding_is_the_draft_round_ir(self):
        # pos64=False: the seam binds exactly the DRAFT-round calls.
        k = load()
        ld, ldu, nbytes, tag = k._pos_loaders(False)
        self.assertIs(ldu, k._ld_uniform_pos_i32)
        self.assertEqual((nbytes, tag), (4, ""))
        self.assertEqual(k._pos_loaders(True)[2:], (8, "_qp64"))
        body = ast.unparse(next(n for n in ast.parse(open(_load.KERNEL_PATH, encoding="utf-8").read()).body
                                if isinstance(n, ast.FunctionDef) and n.name == "_ld_uniform_pos_i32"))
        self.assertIn("return _ld_uniform_i32(rsrc, row * 4)", body)

    def test_module_wide_position_seam_census(self):
        # ROUND-MERGE1 regression: every QP descriptor (incl. kernel_tdm_r16_nobar,
        # whose DRAFT-round body predated the seam) is sized with qp_bytes, and
        # no position is loaded with a raw dword offset.
        src = open(_load.KERNEL_PATH, encoding="utf-8").read()
        self.assertEqual(src.count("_plane(QP, _plane_bytes(rows, qp_bytes))"), 10)
        self.assertEqual(src.count("_plane(QP, "), 10)
        self.assertNotIn("(qp_rsrc, safe_row * 4)", src)
        self.assertNotIn("(qp_rsrc, row * 4)", src)
        self.assertNotIn("(qp_rsrc, (t0 + i) * 4)", src)

    def test_wrapper_passes_qp_kw(self):
        tree = ast.parse(open(_load.KERNEL_PATH, encoding="utf-8").read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "qsa_paged_mqa_logits_page4")
        block = next(n for n in fn.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "_DRAFT_PATH")
        calls = [ast.unparse(c) for c in ast.walk(block) if isinstance(c, ast.Call)
                 and ast.unparse(c.func) == "_build_draft_kernel"]
        self.assertEqual(calls, ["_build_draft_kernel(is_fp8, float(divisor), draft[0], draft[1], **qp_kw)"])


class StorePlan(unittest.TestCase):
    def test_exactly_once(self):
        k = load(QSA_DRAFT_PATH="1")
        for wc in (32, 64):
            for rows in (1, 3, 16):
                for cols in (1, 3, 31, 33, 127, 128, 129, 255, 257, 1000, 4096):
                    got = Counter(k.draft_store_plan(rows, cols, wc))
                    want = {(r, c): 1 for r in range(rows) for c in range(cols)}
                    self.assertEqual(dict(got), want, (wc, rows, cols))
                    gx, gy = k.draft_grid(rows, cols, wc)
                    self.assertEqual(gy, rows)
                    self.assertEqual(gx, math.ceil(cols / (4 * wc)))

    def test_grid_sizes(self):
        k = load(QSA_DRAFT_PATH="1")
        self.assertEqual(k.draft_grid(16, 32768, 32), (256, 16))
        self.assertEqual(k.draft_grid(16, 65536, 32), (512, 16))
        self.assertEqual(k.draft_grid(16, 32768, 64), (128, 16))


# ---- lane-level emulation of the draft kernel (layouts of the WMMA block) ----
def _emulate(q, kc, ks, pt, t2r, qp, cl, columns, wave_cols, is_fp8, divisor):
    """Execute the draft kernel's per-lane index math on CPU (float64 dots).

    A: lane l row m = l % 16 -> head m % 4 (all of one query row);
    B: lane l column n = l % 16 of tile nt; C: lane l element v -> m = (l//16)*8 + v.
    """
    rows, reqs, width = q.shape[0], pt.shape[0], pt.shape[1]
    pages = kc.shape[0]
    out = torch.full((rows, columns), float("nan"), dtype=torch.float64)
    vis_out = torch.full((rows,), -7, dtype=torch.int64)
    span = 4 * wave_cols
    gx = (columns + span - 1) // span
    for row in range(rows):
        req = int(t2r[row]); ok = 0 <= req < reqs; sreq = req if ok else 0
        vis = max(min(int(qp[row]) + 1, int(cl[sreq])), 0) // 4 if ok else 0
        vis_out[row] = vis
        for bx in range(gx):
            for wave in range(4):
                col0 = bx * span + wave * wave_cols
                for lane in range(32):
                    n_in, kgrp = lane % 16, lane // 16
                    for j in range(wave_cols // 32):
                        c = col0 + 32 * j + lane
                        if c >= columns:
                            continue
                        if not col0 < vis:
                            out[row, c] = float("-inf"); continue
                        nt = 2 * j + kgrp              # tile whose column is c
                        cc = col0 + nt * 16 + n_in
                        assert cc == c
                        logical, off = cc // 4, cc % 4
                        lok = logical < width
                        phys = int(pt[sreq, logical if lok else 0])
                        pok = ok and lok and 0 <= phys < pages
                        phys = phys if pok else 0
                        krow = kc[phys, off].double()
                        # element v of this lane: slot m = kgrp*8 + v -> head m % 4
                        acc = [float(q[row, (kgrp * 8 + v) % 4].double() @ krow) for v in range(4)]
                        if is_fp8:
                            s = float(ks[phys, off])
                            score = sum(max(a * s, 0.0) for a in acc)
                        else:
                            score = (max(acc[0], 0.0) + max(acc[1], 0.0)) + (
                                max(acc[2], 0.0) + max(acc[3], 0.0))
                        score /= divisor
                        out[row, c] = score if (pok and c < vis) else float("-inf")
    return out, vis_out


def _reference(q, kc, ks, pt, t2r, qp, cl, columns, is_fp8, divisor):
    rows, reqs, width = q.shape[0], pt.shape[0], pt.shape[1]
    out = torch.full((rows, columns), float("-inf"), dtype=torch.float64)
    vis_out = torch.zeros(rows, dtype=torch.int64)
    for t in range(rows):
        req = int(t2r[t])
        if not 0 <= req < reqs:
            continue
        vis = max(min((int(qp[t]) + 1), int(cl[req])), 0) // 4
        vis_out[t] = vis
        for c in range(min(columns, vis)):
            if c // 4 >= width:
                continue
            p = int(pt[req, c // 4])
            if not 0 <= p < kc.shape[0]:
                continue
            k = kc[p, c % 4].double() * (float(ks[p, c % 4]) if is_fp8 else 1.0)
            out[t, c] = sum(max(float(q[t, h].double() @ k), 0.0) for h in range(4)) / divisor
    return out, vis_out


class LaneEmulation(unittest.TestCase):
    def _case(self, is_fp8, wave_cols, seed):
        g = torch.Generator().manual_seed(seed)
        rows, reqs, width, pages = 6, 7, 70, 90
        columns = width * 4 - 3                       # tail inside the last page
        q = torch.randn(rows, 4, 128, generator=g).to(torch.bfloat16)
        kc = torch.randn(pages, 4, 128, generator=g).to(torch.bfloat16)
        ks = torch.rand(pages, 4, generator=g) + 0.5
        pt = torch.randint(-2, pages + 3, (reqs, width), generator=g)   # invalid pages
        t2r = torch.tensor([0, 3, -1, 6, 7, 2])                         # invalid requests
        qp = torch.tensor([279, 100, 50, 3, 200, 10_000])
        cl = torch.randint(0, 300, (reqs,), generator=g)
        cl[0], cl[2] = 281, 1_000_000
        got, gv = _emulate(q, kc, ks, pt, t2r, qp, cl, columns, wave_cols, is_fp8, math.sqrt(128))
        want, wv = _reference(q, kc, ks, pt, t2r, qp, cl, columns, is_fp8, math.sqrt(128))
        self.assertFalse(torch.isnan(got).any())
        self.assertTrue(torch.equal(gv, wv))
        fin = torch.isfinite(want)
        self.assertTrue(torch.equal(fin, torch.isfinite(got)))
        self.assertTrue(torch.allclose(got[fin], want[fin], rtol=1e-12, atol=1e-12))
        self.assertTrue(fin.any() and (~fin).any())

    def test_bf16_w32(self):
        self._case(False, 32, 1)

    def test_bf16_w64(self):
        self._case(False, 64, 2)

    def test_fp8_w32(self):
        self._case(True, 32, 3)

    def test_fp8_w64(self):
        self._case(True, 64, 4)


if __name__ == "__main__":
    unittest.main()
