# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests of the ``qsa_paged_mqa_logits`` op contract v1 and its native path.

No GPU, no device, no ``aiter`` import (the package is loaded standalone):

* contract: the public op ``aiter.qsa_paged_mqa_logits`` has exactly ATOM's 12-parameter
  signature (``inspect.signature``), a backend-neutral name, and is exported from
  ``aiter/__init__.py``;
* default off: without a profile every call is served by the native path
  (``Decision.path == "native"``, reason ``disabled``, vid ``native.torch.v1``) and the
  kernel module is never imported;
* routing: fused vs native decisions at index page 4 / 8 / 16 for bf16 and fp8 caches,
  with the typed reason of every native decision; contract violations are ``invalid``;
* native numerics vs an independent float64 torch reference written here (and vs the
  frozen golden when ``$QSA_GOLDEN_DIR`` is set): invalid pages, invalid requests,
  ``max_columns``, ``score_divisor``, negative positions, int32/int64 metadata, multi-tile;
* graph safety: ``launch`` of a prepared native decision with caller outputs creates no
  tensor storage and issues no host-synchronizing op (TorchDispatchMode audit), reads
  no environment;
* ipf (``s_prefetch_inst``) kernel variants are not routable by a profile.

    QSA_GOLDEN_DIR=/path/golden_baseline python3 -B op_tests/test_qsa_index_scorer_native_cpu.py -v
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import os
import sys
import unittest
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_qsa_index_scorer_cpu as base

API, PR, K, PKG = base.API, base.PR, base.K, base.PKG
NAT = sys.modules[PKG.__name__ + ".native"]
OP = sys.modules[PKG.__name__ + ".op"]
FP8 = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Independent reference (float64, per row; shares no code with the op)
# ---------------------------------------------------------------------------
def reference(
    q, cache, pt, t2r, qp, cl, ratio=4, divisor=None, max_columns=None, scale=None
):
    rows, _, dim = q.shape
    page, npages = cache.shape[1], cache.shape[0]
    nreq, width = pt.shape
    cols = (
        width * page
        if max_columns is None
        else max(1, min(width * page, int(max_columns)))
    )
    div = math.sqrt(dim) if divisor is None else float(divisor)
    keys = cache[:, :, 0, :].to(torch.float64)
    if scale is not None:
        keys = keys * scale[:, :, 0, None].to(torch.float64)
    out = torch.full((rows, cols), float("-inf"), dtype=torch.float64)
    vis = torch.zeros(rows, dtype=torch.int32)
    for t in range(rows):
        r = int(t2r[t])
        if not 0 <= r < nreq:
            continue
        v = max(0, min((int(qp[t]) + 1) // ratio, int(cl[r]) // ratio))
        vis[t] = v
        c = torch.arange(min(v, cols))
        if c.numel() == 0:
            continue
        lg = c // page
        okw = lg < width  # logical page inside the table
        pg = (
            pt[r, lg.clamp(max=max(width - 1, 0))].to(torch.int64)
            if width
            else torch.full_like(c, -1)
        )
        ok = okw & (pg >= 0) & (pg < npages)
        k = keys[pg.clamp(0, npages - 1), c % page]
        s = (
            torch.relu(torch.einsum("hd,nd->hn", q[t].to(torch.float64), k)).sum(0)
            / div
        )
        out[t, c[ok]] = s[ok]
    return out, vis


def make_call(
    rows=13,
    requests=3,
    width=9,
    page=4,
    pages=40,
    fmt="bf16",
    seed=0,
    md=torch.int64,
    heads=4,
    dim=128,
    bad_pages=True,
    bad_requests=True,
):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(rows, heads, dim, generator=g).to(torch.bfloat16)
    vals = torch.randn(pages, page, 1, dim, generator=g)
    if fmt == "bf16":
        cache, scale = vals.to(torch.bfloat16), None
    else:
        amax = vals.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        cache = (vals * (448.0 / amax)).clamp(-448, 448).to(FP8)
        scale = (amax / 448.0)[..., 0].to(torch.float32)  # [pages, page, 1]
    lo = -1 if bad_pages else 0
    hi = pages + 3 if bad_pages else pages
    pt = torch.randint(lo, hi, (requests, width), generator=g).to(md)
    t2r = torch.randint(0, requests, (rows,), generator=g).to(md)
    if bad_requests and rows >= 3:
        t2r[1], t2r[2] = -1, requests  # invalid requests
    qp = torch.randint(-3, width * page * 4 + 8, (rows,), generator=g).to(md)
    cl = torch.randint(0, width * page * 4 + 8, (requests,), generator=g).to(md)
    return [q, cache, pt, t2r, qp, cl, 4, None, None, None, False, scale]


def assert_matches(tc, out, ref, msg=""):
    lg, vg = out
    rl, rv = ref
    tc.assertTrue(torch.equal(vg, rv), msg)
    tc.assertEqual(lg.dtype, torch.float32)
    tc.assertTrue(
        torch.equal(torch.isneginf(lg), torch.isneginf(rl)), msg
    )  # mask exact
    fin = ~torch.isneginf(rl)
    tc.assertFalse(torch.isnan(lg).any(), msg)
    if fin.any():
        torch.testing.assert_close(
            lg[fin].double(), rl[fin], rtol=1e-4, atol=1e-4, msg=msg
        )


# ---------------------------------------------------------------------------
class Contract(unittest.TestCase):
    def test_public_signature_is_atoms(self):
        sig = inspect.signature(OP.qsa_paged_mqa_logits)
        params = list(sig.parameters.values())
        self.assertEqual(
            tuple((p.name, p.default) for p in params), API.ATOM_SCORER_SIGNATURE
        )
        self.assertTrue(
            all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)
        )
        self.assertEqual(
            (OP.OP_NAME, OP.OP_CONTRACT_VERSION), ("qsa_paged_mqa_logits", 1)
        )
        self.assertEqual(OP.qsa_paged_mqa_logits.__name__, OP.OP_NAME)
        for suffix in (
            "_flydsl",
            "_gfx1250",
            "_triton",
            "_hip",
            "_asm",
            "_ck",
            "_opus",
            "_torch",
        ):
            self.assertFalse(OP.OP_NAME.endswith(suffix))
        # module API keeps the 12 leading parameters; extras are keyword-only
        mod = list(inspect.signature(API.qsa_paged_mqa_logits).parameters.values())
        self.assertEqual(
            tuple((p.name, p.default) for p in mod[:12]), API.ATOM_SCORER_SIGNATURE
        )
        self.assertTrue(all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in mod[12:]))

    def test_exported_from_aiter(self):
        with open(
            os.path.join(base.ROOT, "aiter", "__init__.py"), encoding="utf-8"
        ) as fh:
            tree = ast.parse(fh.read())
        hits = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and n.module == "ops.qsa_index_scorer.op"
            and n.level == 1
            and [a.name for a in n.names] == ["qsa_paged_mqa_logits"]
        ]
        self.assertEqual(len(hits), 1)

    def test_native_module_is_torch_only(self):
        with open(NAT.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.level == 0:
                mods.add(n.module.split(".")[0])
        self.assertEqual(mods, {"__future__", "math", "torch"})

    def test_contract_violations_raise_value_error(self):
        args = make_call()
        cases = []
        a = list(args)
        a[0] = a[0][:, :, :64]
        cases.append(("dim", a))
        a = list(args)
        a[1] = a[1].view(torch.uint8)
        a[11] = torch.zeros(40, 4, 1, 4, dtype=torch.uint8)
        cases.append(("fp4", a))
        a = make_call(fmt="fp8")
        a[11] = a[11].to(torch.uint8)
        cases.append(("mxfp8", a))
        a = make_call(fmt="fp8")
        a[11] = None
        cases.append(("noscale", a))
        a = list(args)
        a[3] = a[3][:-1]
        cases.append(("t2r len", a))
        a = list(args)
        a[7] = 0.0
        cases.append(("divisor", a))
        a = list(args)
        a[6] = 0
        cases.append(("ratio", a))
        a = list(args)
        a[2] = a[2].float()
        cases.append(("pt dtype", a))
        a = list(args)
        a[0] = a[0].float()
        cases.append(("q dtype vs bf16 cache", a))
        for name, a in cases:
            with mock.patch.dict(os.environ, base._CLEAN, clear=True):
                d = API.decide(*a)
                self.assertEqual(
                    (d.ok, d.path, d.vid), (False, API.PATH_INVALID, None), name
                )
                with self.assertRaises(ValueError, msg=name):
                    API.launch(d, *a)
                with self.assertRaises(ValueError, msg=name):
                    OP.qsa_paged_mqa_logits(*a)


class DefaultOff(unittest.TestCase):
    def test_native_without_profile(self):
        args = make_call()
        with mock.patch.dict(os.environ, base._CLEAN, clear=True), mock.patch.object(
            API, "_KERNEL", None
        ), mock.patch.object(
            API, "_load_kernel", side_effect=AssertionError("kernel import")
        ), mock.patch.object(
            API, "_device_info", side_effect=AssertionError("device query")
        ):
            d = API.decide(*args)
            self.assertEqual(
                (d.ok, d.reason, d.path, d.vid),
                (False, PR.Reason.DISABLED, API.PATH_NATIVE, API.NATIVE_VID),
            )
            self.assertIsInstance(d._plan, NAT.NativePlan)
            out = OP.qsa_paged_mqa_logits(*args)
            ref = reference(*args[:6], max_columns=args[8], scale=args[11])
            assert_matches(self, out, ref)
            out2 = API.launch(d, *args)
            self.assertTrue(
                torch.equal(out[0], out2[0]) and torch.equal(out[1], out2[1])
            )
            # strict / try_ keep "None instead of native" semantics, no workspace allocated
            res, d2 = API.try_qsa_paged_mqa_logits(*args)
            self.assertIsNone(res)
            self.assertIsNone(d2._plan)
            with self.assertRaises(API.QSAIndexScorerUnsupported):
                API.qsa_paged_mqa_logits(*args, strict=True)


class Routing(base.Env):
    """Fused vs native decisions (kernel module injected, device faked as gfx1250/256)."""

    def _call(self, page, fmt, rows=64, requests=16, columns=4096, **kw):
        width = columns // page
        return base.call_args(
            rows=rows,
            requests=requests,
            width=width,
            columns=columns,
            fmt=fmt,
            page=page,
            **kw,
        )

    def test_fused_for_covered_calls(self):
        for page in (4, 8, 16):
            for fmt in ("bf16", "fp8"):
                pre = PR.variant_prefix(page)
                for rows, requests, kind in ((64, 16, "decg"), (16, 16, "draft")):
                    d = self.decide(self._call(page, fmt, rows=rows, requests=requests))
                    self.assertEqual(
                        (d.ok, d.path), (True, API.PATH_FUSED), (page, fmt, d)
                    )
                    self.assertEqual(d.vid, f"{pre}.{kind}.v1", (page, fmt))
                    self.assertTrue(d.cell.startswith("rule:"), d.cell)

    def test_native_with_reason_for_uncovered_calls(self):
        for page in (4, 8, 16):
            for fmt in ("bf16", "fp8"):
                cases = {
                    PR.Reason.DIVISOR: {
                        "args": self._call(page, fmt),
                        "patch": {7: 8.0},
                    },
                    PR.Reason.COLUMNS: {
                        "args": self._call(page, fmt),
                        "patch": {8: 1000},
                    },
                    PR.Reason.CACHE_ALIGNMENT: {
                        "args": self._call(page, fmt, cache_offset=16)
                    },
                }
                for reason, c in cases.items():
                    a = list(c["args"])
                    for i, v in c.get("patch", {}).items():
                        a[i] = v
                    d = self.decide(a)
                    self.assertEqual(
                        (d.ok, d.reason, d.path, d.vid),
                        (False, reason, API.PATH_NATIVE, API.NATIVE_VID),
                        (page, fmt, d),
                    )
                    self.assertIsInstance(d._plan, NAT.NativePlan)
        with mock.patch.object(API, "_device_info", lambda dev: ("gfx950", 256)):
            d = self.decide(self._call(4, "bf16"))
            self.assertEqual(
                (d.reason, d.path), (PR.Reason.ARCH_UNSUPPORTED, API.PATH_NATIVE)
            )
        d = self.decide(self._call(4, "bf16"), profile="nope_v0")
        self.assertEqual(
            (d.reason, d.path), (PR.Reason.PROFILE_NOT_FOUND, API.PATH_NATIVE)
        )

    def test_page_32_is_native(self):
        a = base.call_args(rows=64, requests=16, width=128, columns=4096, page=32)
        d = self.decide(a)
        self.assertEqual((d.reason, d.path), (PR.Reason.CACHE_SHAPE, API.PATH_NATIVE))

    def test_native_launch_guards(self):
        a = self._call(4, "bf16")
        a[7] = 8.0
        d = self.decide(a)
        self.assertEqual(d.path, API.PATH_NATIVE)
        b = self._call(4, "bf16", rows=32, requests=16)
        b[7] = 8.0
        with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
            API.launch(d, *b)
        self.assertEqual(cm.exception.reason, PR.Reason.ROUTE_MISMATCH)
        c = self._call(4, "fp8")
        c[7] = 8.0
        with self.assertRaises(API.QSAIndexScorerUnsupported):
            API.launch(d, *c)


class NativeNumerics(unittest.TestCase):
    def _check(self, args, tile=None, msg=""):
        geo = NAT.validate(*args[:10], args[11])
        plan = NAT.NativePlan(geo, **({} if tile is None else {"tile_elems": tile}))
        rs = torch.full((args[0].shape[0],), 7, dtype=torch.int32)
        out = plan.run(*args[:6], rs, args[11])
        ref = reference(
            *args[:6],
            ratio=args[6],
            divisor=args[7],
            max_columns=args[8],
            scale=args[11],
        )
        assert_matches(self, out, ref, msg)
        self.assertEqual(int(rs.abs().sum()), 0, msg)  # row_starts zeroed
        return out

    def test_pages_dtypes_metadata(self):
        for page in (4, 8, 16, 2, 64):
            for fmt in ("bf16", "fp8"):
                for md in (torch.int32, torch.int64):
                    self._check(
                        make_call(page=page, fmt=fmt, md=md, seed=page), msg=(page, fmt)
                    )

    def test_max_columns_divisor_ratio(self):
        for mc in (1, 17, 36, 1000, 0, -5):
            a = make_call(seed=3)
            a[8] = mc
            self._check(a, msg=mc)
        a = make_call(seed=4)
        a[7], a[6] = 3.5, 2
        self._check(a)

    def test_multi_tile_equals_single_tile(self):
        a = make_call(rows=37, requests=5, width=40, page=4, fmt="fp8", seed=7)
        one = self._check(a)
        many = self._check(a, tile=96)  # 96 cols x 1 row tiles
        self.assertTrue(torch.equal(one[0], many[0]) and torch.equal(one[1], many[1]))
        plan = NAT.NativePlan(NAT.validate(*a[:10], a[11]), tile_elems=96)
        self.assertEqual(
            (plan.tile_rows, plan.tile_cols), (1, 96)
        )  # 160 columns: 96 + 64

    def test_other_geometry(self):
        self._check(make_call(heads=2, dim=64, seed=9))
        self._check(make_call(heads=8, dim=32, fmt="fp8", seed=10))

    def test_empty_and_degenerate(self):
        a = make_call(rows=0, bad_requests=False)
        lg, vg = OP.qsa_paged_mqa_logits(*a)
        self.assertEqual((tuple(lg.shape), tuple(vg.shape)), ((0, 36), (0,)))
        a = make_call(width=0, bad_pages=False)  # columns 0 -> 1 (ATOM)
        a[8] = 5
        self._check(a)

    def test_vs_frozen_golden(self):
        try:
            gdir = base._golden_dir()
        except RuntimeError:
            self.skipTest("frozen golden not found (set QSA_GOLDEN_DIR)")
        sys.path.insert(0, gdir)
        try:
            import qsa_golden as G
        finally:
            sys.path.remove(gdir)
        n = 0
        for page in (4, 8, 16):
            for fmt in ("bf16", "fp8"):
                for mc in (None, 23):
                    a = make_call(page=page, fmt=fmt, seed=page + 1, bad_pages=False)
                    a[2][0, 1] = -1  # golden: -1 invalid
                    a[8] = mc
                    out = OP.qsa_paged_mqa_logits(*a)
                    gl, gv, valid = G.reference_logits(*a[:6], 4, mc, a[11])
                    res = G.compare_logits(out[0], out[1], gl, gv, valid)
                    self.assertTrue(res["pass"], (page, fmt, mc, res))
                    self.assertTrue(
                        torch.equal(torch.isneginf(out[0]), torch.isneginf(gl))
                    )
                    n += 1
        self.assertEqual(n, 12)


class _Audit(TorchDispatchMode):
    SYNC = (
        "aten._local_scalar_dense",
        "aten.item",
        "aten.nonzero",
        "aten.is_nonzero",
        "aten.equal",
        "aten.allclose",
        "aten.unique",
    )

    def __init__(self, known):
        super().__init__()
        self.known, self.new, self.ops = set(known), [], []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = (
            str(func.overloadpacket) if hasattr(func, "overloadpacket") else str(func)
        )
        self.ops.append(name)
        for t in tree_flatten(out)[0]:
            if (
                isinstance(t, torch.Tensor)
                and t.untyped_storage().nbytes() > 0
                and t.untyped_storage().data_ptr() not in self.known
            ):
                self.new.append((name, tuple(t.shape)))
        return out


def _storages(*objs):
    out = set()
    for o in objs:
        for t in tree_flatten(o)[0]:
            if isinstance(t, torch.Tensor):
                out.add(t.untyped_storage().data_ptr())
    return out


class GraphSafety(unittest.TestCase):
    def test_launch_allocates_nothing_and_never_syncs(self):
        for fmt in ("bf16", "fp8"):
            a = make_call(rows=21, requests=4, width=30, fmt=fmt, seed=11)
            rows = a[0].shape[0]
            a[9] = torch.full((rows,), 3, dtype=torch.int32)
            with mock.patch.dict(os.environ, base._CLEAN, clear=True):
                d = API.decide(*a)  # prepare: workspace allocated here
            self.assertEqual(d.path, API.PATH_NATIVE)
            plan = d._plan
            plan_small = NAT.NativePlan(plan.geo, tile_elems=64)  # many tiles
            for p in (plan, plan_small):
                lg = torch.empty((rows, p.columns), dtype=torch.float32)
                vg = torch.empty(rows, dtype=torch.int32)
                dd = API.replace(d, _plan=p)
                known = _storages(
                    a,
                    lg,
                    vg,
                    [v for v in vars(p).values() if isinstance(v, torch.Tensor)],
                )

                class _NoEnv(dict):
                    def get(self, *x, **k):
                        raise AssertionError("env read")

                    __getitem__ = get

                audit = _Audit(known)
                with mock.patch.object(os, "environ", _NoEnv()), audit:
                    out = API.launch(dd, *a, logits=lg, visible_groups=vg)
                self.assertEqual(audit.new, [], (fmt, p.tile_rows, p.tile_cols))
                bad = [o for o in audit.ops if o.startswith(_Audit.SYNC)]
                self.assertEqual(bad, [], fmt)
                self.assertIs(out[0], lg)
                ref = reference(*a[:6], scale=a[11])
                assert_matches(self, (lg, vg), ref, fmt)
                self.assertEqual(int(a[9].abs().sum()), 0)

    def test_audit_bites(self):
        """Negative control: the audit flags an allocating launch (outputs not supplied)."""
        a = make_call(rows=5, fmt="bf16", seed=12)
        with mock.patch.dict(os.environ, base._CLEAN, clear=True):
            d = API.decide(*a)
        known = _storages(
            a, [v for v in vars(d._plan).values() if isinstance(v, torch.Tensor)]
        )
        audit = _Audit(known)
        with audit:
            API.launch(d, *a)
        self.assertTrue(audit.new)

    def test_workspace_is_bounded(self):
        a = make_call(rows=4096, requests=2, width=4096, pages=8, seed=13)
        geo = NAT.validate(*a[:10], a[11])
        plan = NAT.NativePlan(geo)
        self.assertEqual(plan.tile_rows * plan.tile_cols, NAT.TILE_ELEMS)
        self.assertLess(plan.workspace_bytes(), 64 << 20)


class IpfNotRoutable(unittest.TestCase):
    def test_profile_rejects_ipf(self):
        with open(base.PROFILE_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        bad = copy.deepcopy(data)
        rule = next(r for r in bad["rules"] if r["vid"].endswith(".r32_kt.v1"))
        rule["params"]["ipf"] = 32
        with self.assertRaises(PR.ProfileError) as cm:
            PR.validate_profile(bad)
        self.assertIn("ipf", str(cm.exception))
        PR.validate_profile(data)

    def test_plan_refuses_ipf(self):
        with self.assertRaises(ValueError):
            API._Plan(
                K,
                PR.load_profile(PR.DEFAULT_PROFILE),
                "qsa.page4.r32_kt.v1",
                {
                    "k_load": "tdm",
                    "rows_per_block": 32,
                    "kt_rows": 32,
                    "col_tiles": 2,
                    "lean": True,
                    "ipf": 32,
                },
                False,
                torch.bfloat16,
                32,
                1,
                16,
                64,
                4,
            )


if __name__ == "__main__":
    unittest.main()
