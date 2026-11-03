# SPDX-License-Identifier: MIT
"""Round DECG CPU gates: opt-in decode row-group kernel (QSA_DECODE_GROUP).

Routing/knobs (ROUND-DECG2: default 'all', '0' opts out, fp8 never w64),
wrapper placement (flag-gated block), pass plan incl.
mixed-request groups, exactly-once stores incl. tails and partial groups, and
a lane-level emulation of the WMMA A/B/C layouts against the contract formula
(and against an emulation of the champion r16 dedup passes: bitwise-equal
inputs per WMMA slot). No device, no FlyDSL needed (stub loader).

    python3 -B test_decg_cpu.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import random
import sys
import unittest
from collections import Counter
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import _load  # noqa: E402

KEYS = ("QSA_DECODE_GROUP", "QSA_DECODE_GROUP_COLS", "QSA_DRAFT_PATH", "QSA_G2_ACCESS",
        "QSA_FP8_PATH", "QSA_BF16_PATH", "QSA_R16_NOBAR", "QSA_BF16_R64", "QSA_BF16_K_LOAD")
_N = [0]


def load(**env):
    clean = {k: v for k, v in os.environ.items() if k not in KEYS}
    clean.update(env)
    with mock.patch.dict(os.environ, clean, clear=True):
        _N[0] += 1
        spec = importlib.util.spec_from_file_location(f"qsa_decg_{_N[0]}", _load.KERNEL_PATH)
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


def _src():
    with open(_load.KERNEL_PATH, encoding="utf-8") as fh:
        return fh.read()


def _top(name):
    tree = ast.parse(_src())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _inner(builder, name):
    return {n.name: n for n in ast.walk(_top(builder)) if isinstance(n, ast.FunctionDef)}[name]


class Routing(unittest.TestCase):
    def test_default_on_all(self):
        # ROUND-DECG2: default flipped to 'all' (was '0'), COLS default stays 32.
        k = load()
        self.assertEqual(k._DECODE_GROUP, "all")
        self.assertTrue(k._DECODE_GROUP_BF16 and k._DECODE_GROUP_FP8)
        self.assertEqual(k._DECODE_GROUP_COLS, 32)
        for fp8 in (False, True):
            self.assertEqual(k.decode_group_route(64, 16, fp8, 0x1000), 32)
            self.assertEqual(k.decode_group_route(16, 4, fp8, 0x1000), 32)
            self.assertIsNone(k.decode_group_route(64, 64, fp8, 0x1000))   # draft
            self.assertIsNone(k.decode_group_route(10, 2, fp8, 0x1000))    # rows % 4

    def test_opt_out_zero(self):
        # ROUND-DECG2: QSA_DECODE_GROUP=0 restores the previous (c45b24e0) routing.
        k = load(QSA_DECODE_GROUP="0")
        self.assertEqual(k._DECODE_GROUP, "0")
        self.assertFalse(k._DECODE_GROUP_BF16 or k._DECODE_GROUP_FP8)
        for rows, reqs in ((64, 16), (16, 4), (8, 2), (64, 64)):
            for fp8 in (False, True):
                self.assertIsNone(k.decode_group_route(rows, reqs, fp8, 0x1000))

    def test_fp8_never_w64(self):
        # ROUND-DECG2: fp8 w64 GPU-incorrect (root run 2026-11-03), disabled.
        for env in ({"QSA_DECODE_GROUP_COLS": "64"},
                    {"QSA_DECODE_GROUP": "all", "QSA_DECODE_GROUP_COLS": "64"},
                    {"QSA_DECODE_GROUP": "fp8", "QSA_DECODE_GROUP_COLS": "64"}):
            k = load(**env)
            self.assertEqual(k._DECODE_GROUP_FP8_COLS, 32)
            self.assertEqual(k.decode_group_route(64, 16, True, 0x1000), 32)
        self.assertEqual(load(QSA_DECODE_GROUP_COLS="64").decode_group_route(64, 16, False, 0), 64)
        src = ast.unparse(_top("_build_decode_group_kernel"))
        self.assertIn("if is_fp8 and wave_cols != _DECODE_GROUP_FP8_COLS:", src)
        self.assertIn("return _DECODE_GROUP_FP8_COLS if is_fp8 else _DECODE_GROUP_COLS",
                      ast.unparse(_top("decode_group_route")))

    def test_opt_in_routes(self):
        k = load(QSA_DECODE_GROUP="1")
        self.assertEqual(k.decode_group_route(64, 16, False, 0x1000), 32)
        self.assertEqual(k.decode_group_route(16, 4, False, 0x1000), 32)
        self.assertIsNone(k.decode_group_route(64, 16, True, 0x1000))      # "1" = bf16 only
        self.assertIsNone(k.decode_group_route(16, 16, False, 0x1000))     # draft regime
        self.assertIsNone(k.decode_group_route(10, 2, False, 0x1000))      # rows % 4
        self.assertIsNone(k.decode_group_route(512, 1, False, 0x1000))     # prefill (r32)
        self.assertIsNone(k.decode_group_route(64, 2, False, 0x1000))      # rows >= 32 * reqs
        self.assertEqual(k.decode_group_route(60, 2, False, 0x1000), 32)   # still r16
        self.assertIsNone(k.decode_group_route(64, 16, False, 0x1010))     # TDM alignment

    def test_knobs(self):
        self.assertEqual(load(QSA_DECODE_GROUP="fp8").decode_group_route(64, 16, True, 0), 32)
        self.assertIsNone(load(QSA_DECODE_GROUP="fp8").decode_group_route(64, 16, False, 0))
        both = load(QSA_DECODE_GROUP="all", QSA_DECODE_GROUP_COLS="64")
        self.assertEqual(both.decode_group_route(64, 16, True, 0), 32)   # DECG2: fp8 never w64
        self.assertEqual(both.decode_group_route(64, 16, False, 0), 64)
        self.assertEqual(load(QSA_DECODE_GROUP="bf16").decode_group_route(64, 16, False, 0), 32)
        self.assertIsNone(load(QSA_DECODE_GROUP="all", QSA_FP8_PATH="scalar")
                          .decode_group_route(64, 16, True, 0))
        self.assertIsNone(load(QSA_DECODE_GROUP="all", QSA_BF16_PATH="scalar")
                          .decode_group_route(64, 16, False, 0))
        self.assertIsNone(load(QSA_DECODE_GROUP="all", QSA_G2_ACCESS="cop")
                          .decode_group_route(64, 16, False, 0))
        for key, bad in (("QSA_DECODE_GROUP", "2"), ("QSA_DECODE_GROUP", "yes"),
                         ("QSA_DECODE_GROUP_COLS", "48")):
            with self.assertRaises(ValueError):
                load(**{key: bad})

    def test_wrapper_block_flag_gated_after_draft_before_original(self):
        fn = _top("qsa_paged_mqa_logits_page4")
        tests = [ast.unparse(n.test) for n in fn.body if isinstance(n, ast.If)]
        i = tests.index("_DECODE_GROUP != '0'")
        self.assertLess(tests.index("_DRAFT_PATH"), i)
        self.assertLess(i, tests.index("not is_fp8 and _BF16_PATH == 'wmma'"))
        self.assertLess(i, tests.index("is_fp8 and _FP8_PATH == 'wmma'"))
        block = next(n for n in fn.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "_DECODE_GROUP != '0'")
        calls = [ast.unparse(c) for c in ast.walk(block) if isinstance(c, ast.Call)]
        self.assertIn("_build_decode_group_kernel(is_fp8, float(divisor), decg, **qp_kw)", calls)
        self.assertIn("decode_group_route(rows, num_requests, is_fp8, kv_u8.data_ptr())", calls)
        self.assertIn("decode_group_grid(rows, columns, decg)", calls)

    def test_wrapper_launch_default_on_and_opt_out(self):
        """Real wrapper on CPU tensors, _run captured: which builder runs.

        ROUND-DECG2: the default env now launches decg; QSA_DECODE_GROUP=0 the
        previous nobar route."""
        for env, want in (({}, "decg"), ({"QSA_DECODE_GROUP": "0"}, "nobar"),
                          ({"QSA_DECODE_GROUP": "1"}, "decg")):
            k = load(**env)
            calls = []
            rows, reqs, cols = 64, 16, 256
            q = torch.zeros(rows, 4, 128, dtype=torch.bfloat16)
            back = torch.zeros(8 * 4 * 128 * 2 + 128, dtype=torch.uint8)
            off = (-back.data_ptr()) % 128
            cache = back[off:off + 8 * 4 * 128 * 2].view(torch.bfloat16).view(8, 4, 1, 128)
            built = []
            with mock.patch.object(k, "_run", lambda launch, *a: calls.append((launch, a))), \
                    mock.patch.object(k, "_build_decode_group_kernel",
                                      lambda *a, **kw: built.append(("decg", a, kw)) or "decg"), \
                    mock.patch.object(k, "_build_wmma_kernel",
                                      lambda *a, **kw: built.append(("wmma", a, kw)) or
                                      type("L", (), {"cols_per_block": 32})()), \
                    mock.patch.object(k, "_launch_stream", lambda dev: None):
                k.qsa_paged_mqa_logits_page4(
                    q, cache, torch.zeros(reqs, cols // 4, dtype=torch.int32),
                    torch.arange(rows, dtype=torch.int32) // 4,
                    torch.full((rows,), cols * 4 - 1, dtype=torch.int64),
                    torch.full((reqs,), cols * 4, dtype=torch.int32), max_columns=cols)
            self.assertEqual(len(calls), 1)
            if want == "decg":
                self.assertEqual(built[0][0], "decg")
                self.assertEqual(built[0][1][2], 32)
                self.assertEqual(built[0][2], {"pos64": True})
                self.assertEqual(calls[0][1][10:12], (cols // 128, rows // 4))
            else:
                self.assertEqual(built[0][0], "wmma")


class Source(unittest.TestCase):
    def test_builder_signature_and_seam(self):
        fn = _top("_build_decode_group_kernel")
        self.assertEqual([a.arg for a in fn.args.args], ["is_fp8", "divisor", "wave_cols", "pos64"])
        src = ast.unparse(fn)
        self.assertIn("_, ld_uniform_pos, qp_bytes, qp_tag = _pos_loaders(pos64)", src)
        self.assertIn("qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))", src)
        self.assertIn("pos = ld_uniform_pos(qp_rsrc, safe_row)", src)
        self.assertNotIn("_ld_uniform_i32(qp_rsrc", src)
        self.assertNotIn("_ld_i32(qp_rsrc", src)
        self.assertNotIn("barrier", src.replace("no WG barrier", ""))

    def test_metadata_is_sgpr_and_descriptors_uniform(self):
        meta = ast.unparse(_inner("_build_decode_group_kernel", "_group_meta"))
        self.assertEqual(meta.count("_ld_uniform_i32("), 2)
        self.assertNotIn("_ld_i32(", meta)
        issue = ast.unparse(_inner("_build_decode_group_kernel", "_issue"))
        self.assertIn("_ld_uniform_i32(pt_rsrc, (pp.safe_req * width + safe_logical) * 4)", issue)
        self.assertIn("wave * tile_bytes + group * 8 * pitch", issue)

    def test_champion_identical_helpers(self):
        # clamped store offsets: textually the r16 helper
        a = ast.unparse(_inner("_build_wmma_kernel", "_store_offsets").body[1:])
        b = ast.unparse(_inner("_build_decode_group_kernel", "_store_offsets").body[1:])
        self.assertEqual(a, b)
        q = ast.unparse(_inner("_build_decode_group_kernel", "_q_frags"))
        self.assertIn("q_row = _imin(t0 + n_in // NUM_HEADS, rows - 1)", q)
        self.assertIn("q_row * q_row_bytes + n_in % NUM_HEADS * head_bytes + kgrp * 16", q)
        # post-processing order: draft (== champion) finishers
        def s_assign(fn):
            return next(ast.unparse(n) for n in fn.body if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == "s")
        for name in ("_finish_bf16", "_finish_fp8"):
            d = s_assign(_inner("_build_draft_kernel", name))
            g = s_assign(_inner("_build_decode_group_kernel", name))
            self.assertEqual(d.replace("acc[", "a["), g)
        self.assertIn("_ld_f32(ks_rsrc, (phys * PAGE_SIZE + off) * 4)",
                      ast.unparse(_inner("_build_decode_group_kernel", "_scale_fp8")))
        # fp8 K fragment converter == the GPU-verified draft TDM converter
        d = ast.unparse(_inner("_build_draft_kernel", "_kfrag_fp8_tdm").body)
        g = ast.unparse(_inner("_build_decode_group_kernel", "_kfrag_fp8").body)
        self.assertEqual(d, g)

    def test_kernel_war_guard_and_tensor_wait_in_live_pass(self):
        body = ast.unparse(_inner("_build_decode_group_kernel", "kernel"))
        i_issue = body.index("_issue(")
        i_wait = body.index("_tensor_done()")
        i_vals = body.index("_pass_values(")
        i_guard = body.index("_war_guard()")
        self.assertLess(i_issue, i_wait)
        self.assertLess(i_wait, i_vals)
        self.assertLess(i_vals, i_guard)
        self.assertIn("fx.rocdl.s_wait_dscnt(0)",
                      ast.unparse(_inner("_build_decode_group_kernel", "_war_guard")))
        self.assertIn("fx.rocdl.tdm_ops.tensor_wait(0)",
                      ast.unparse(_inner("_build_decode_group_kernel", "_tensor_done")))


class PassPlan(unittest.TestCase):
    def test_decode_single_pass(self):
        k = load()
        plan = k.decode_group_passes([5, 5, 5, 5], [True] * 4)
        self.assertEqual(plan[0], (True, [0, 1, 2, 3]))
        self.assertTrue(all(not a for a, _ in plan[1:]))

    def test_mixed_groups_per_row(self):
        k = load()
        for reqs in ([0, 1, 1, 1], [2, 2, -1, 2], [0, 0, 0, 1], [3, 4, 5, 6], [-1, -1, -1, 0]):
            plan = k.decode_group_passes(reqs, [True] * 4)
            self.assertEqual(plan, [(True, [p]) for p in range(4)], reqs)

    def test_partial_group(self):
        k = load()
        # rows = t0 + 2: clamped rows 2, 3 repeat row 1's request
        self.assertEqual(k.decode_group_passes([7, 7, 7, 7], [True, True, False, False])[0],
                         (True, [0, 1]))
        plan = k.decode_group_passes([6, 7, 7, 7], [True, True, False, False])
        self.assertEqual([p for p in plan if p[0]], [(True, [0]), (True, [1])])


class StorePlan(unittest.TestCase):
    def test_exactly_once(self):
        k = load()
        rng = random.Random(7)
        for wc in (32, 64):
            for rows in (4, 8, 10, 64):
                for cols in (1, 31, 33, 127, 128, 129, 257, 1000):
                    for mode in ("decode", "mixed", "random"):
                        if mode == "decode":
                            t2r = [r // 4 for r in range(rows)]
                        elif mode == "mixed":
                            t2r = [(r + 1) // 4 for r in range(rows)]
                        else:
                            t2r = [rng.randint(-1, 3) for _ in range(rows)]
                        got = Counter(k.decode_group_store_plan(rows, cols, wc, t2r))
                        want = {(r, c): 1 for r in range(rows) for c in range(cols)}
                        self.assertEqual(dict(got), want, (wc, rows, cols, mode))

    def test_grid_sizes(self):
        k = load()
        self.assertEqual(k.decode_group_grid(64, 32768, 32), (256, 16))
        self.assertEqual(k.decode_group_grid(64, 65536, 32), (512, 16))
        self.assertEqual(k.decode_group_grid(64, 32768, 64), (128, 16))
        self.assertEqual(k.decode_group_grid(10, 129, 32), (2, 3))


# ---- lane-level emulation of the decode-group kernel ----------------------
def _kblock(kc, ks, pt, sreq, ok, c, width, pages, is_fp8):
    logical, off = c // 4, c % 4
    lok = logical < width
    phys = int(pt[sreq, logical if lok else 0])
    pok = ok and lok and 0 <= phys < pages
    phys = phys if pok else 0
    return kc[phys, off].double(), (float(ks[phys, off]) if is_fp8 else None), pok


def _finish(acc4, scale, is_fp8, divisor):
    if is_fp8:
        s = 0.0
        for a in acc4:
            s = s + max(a * scale, 0.0)
    else:
        s = (max(acc4[0], 0.0) + max(acc4[1], 0.0)) + (max(acc4[2], 0.0) + max(acc4[3], 0.0))
    return s * (1.0 / divisor)


def _meta(t2r, qp, cl, reqs, row):
    req = int(t2r[row]); ok = 0 <= req < reqs; sreq = req if ok else 0
    vis = max(min(int(qp[row]) + 1, int(cl[sreq])), 0) // 4 if ok else 0
    return req, ok, sreq, vis


def _emulate_decg(k, q, kc, ks, pt, t2r, qp, cl, columns, wc, is_fp8, divisor):
    """Per-lane index math of the decg kernel; A built from the lane Q loads."""
    rows, reqs, width, pages = q.shape[0], pt.shape[0], pt.shape[1], kc.shape[0]
    out = torch.full((rows, columns), float("nan"), dtype=torch.float64)
    vis_out = torch.full((rows,), -7, dtype=torch.int64)
    gx, gy = k.decode_group_grid(rows, columns, wc)
    for g in range(gy):
        t0 = 4 * g
        safe = [min(t0 + r, rows - 1) for r in range(4)]
        m = [_meta(t2r, qp, cl, reqs, s) for s in safe]
        rok = [t0 + r < rows for r in range(4)]
        for lane in range(4):                       # writer: bx 0, wave 0, lanes 0..3
            if t0 + lane < rows:
                vis_out[t0 + lane] = m[lane][3]
        plan = k.decode_group_passes([x[0] for x in m], rok)
        same = all(x[0] == m[0][0] for x in m) and rok[0]
        # A[m_slot, :] from lane (n_in = m_slot) loads: row min(t0 + m//4, rows-1), head m%4
        A = torch.stack([q[min(t0 + ms // 4, rows - 1), ms % 4].double() for ms in range(16)])
        for bx in range(gx):
            for wave in range(4):
                col0 = bx * 4 * wc + wave * wc
                for p, (active, owned) in enumerate(plan):
                    if not active:
                        continue
                    vmax = max(x[3] for x in m) if (p == 0 and same) else m[p][3]
                    _, pok_req, sreq, _ = m[p]
                    for lane in range(32):
                        n_in, kgrp = lane % 16, lane // 16
                        for nt in range(wc // 16):
                            c = col0 + nt * 16 + n_in
                            krow, scale, pok = _kblock(kc, ks, pt, sreq, pok_req, c, width,
                                                       pages, is_fp8)
                            C = [float(A[ms] @ krow) for ms in range(16)]  # D[:, n_in]
                            for half, r in ((0, 2 * kgrp), (1, 2 * kgrp + 1)):
                                if r not in owned or c >= columns:
                                    continue
                                if not col0 < vmax:
                                    out[t0 + r, c] = float("-inf")
                                    continue
                                v0 = kgrp * 8 + 4 * half           # C element of this lane
                                acc4 = [float(C[v0 + h]) for h in range(4)]
                                # slot -> (row, head): the r16 contract
                                assert all((v0 + h) // 4 == r and (v0 + h) % 4 == h for h in range(4))
                                s = _finish(acc4, scale, is_fp8, divisor)
                                out[t0 + r, c] = s if (pok and c < m[r][3]) else float("-inf")
    return out, vis_out


def _emulate_r16(q, kc, ks, pt, t2r, qp, cl, columns, is_fp8, divisor):
    """The champion r16 dedup passes (per row the request-pass that owns it)."""
    rows, reqs, width, pages = q.shape[0], pt.shape[0], pt.shape[1], kc.shape[0]
    out = torch.full((rows, columns), float("-inf"), dtype=torch.float64)
    for row in range(rows):
        req, ok, sreq, vis = _meta(t2r, qp, cl, reqs, row)
        for c in range(columns):
            krow, scale, pok = _kblock(kc, ks, pt, sreq, ok, c, width, pages, is_fp8)
            acc4 = [float(q[row, h].double() @ krow) for h in range(4)]
            s = _finish(acc4, scale, is_fp8, divisor)
            out[row, c] = s if (pok and c < vis) else float("-inf")
    return out


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
            kk = kc[p, c % 4].double() * (float(ks[p, c % 4]) if is_fp8 else 1.0)
            out[t, c] = sum(max(float(q[t, h].double() @ kk), 0.0) for h in range(4)) / divisor
    return out, vis_out


class LaneEmulation(unittest.TestCase):
    def _case(self, is_fp8, wc, seed, t2r, rows):
        k = load()
        g = torch.Generator().manual_seed(seed)
        reqs, width, pages = 6, 40, 60
        columns = width * 4 - 3                       # tail inside the last page
        q = torch.randn(rows, 4, 128, generator=g).to(torch.bfloat16)
        kc = torch.randn(pages, 4, 128, generator=g).to(torch.bfloat16)
        ks = torch.rand(pages, 4, generator=g) + 0.5
        pt = torch.randint(-2, pages + 3, (reqs, width), generator=g)   # invalid pages
        t2r = torch.tensor(t2r)
        qp = torch.randint(-3, 200, (rows,), generator=g)
        qp[0] = 10_000
        cl = torch.randint(0, 170, (reqs,), generator=g)
        cl[0] = 161
        div = math.sqrt(128)
        got, gv = _emulate_decg(k, q, kc, ks, pt, t2r, qp, cl, columns, wc, is_fp8, div)
        want, wv = _reference(q, kc, ks, pt, t2r, qp, cl, columns, is_fp8, div)
        champ = _emulate_r16(q, kc, ks, pt, t2r, qp, cl, columns, is_fp8, div)
        self.assertFalse(torch.isnan(got).any())
        self.assertTrue(torch.equal(gv, wv))
        fin = torch.isfinite(want)
        self.assertTrue(torch.equal(fin, torch.isfinite(got)))
        self.assertTrue(torch.allclose(got[fin], want[fin], rtol=1e-12, atol=1e-12))
        # same per-slot inputs and post-processing order as the champion: equal
        self.assertTrue(torch.equal(got, champ))
        self.assertTrue(fin.any() and (~fin).any())

    DECODE = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    MIXED = [0, 1, 1, 1, 2, 2, -1, 2, 3, 3, 3, 3, 5, 4, 7, 4, -1, -1, -1, -1]
    PARTIAL = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]

    def test_bf16_w32_decode(self):
        self._case(False, 32, 1, self.DECODE, 12)

    def test_bf16_w64_mixed(self):
        self._case(False, 64, 2, self.MIXED, 20)

    def test_bf16_w32_partial(self):
        self._case(False, 32, 3, self.PARTIAL, 10)

    def test_fp8_w32_mixed(self):
        self._case(True, 32, 4, self.MIXED, 20)

    def test_fp8_w64_decode(self):
        self._case(True, 64, 5, self.DECODE, 12)


if __name__ == "__main__":
    unittest.main()
