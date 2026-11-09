# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests of the neutral op ``aiter.ops.qsa.qsa_paged_mqa_logits`` (op contract v1).

No GPU, no device, no ``aiter`` import: ``aiter/ops/qsa.py`` is loaded from its file
with aiter's Triton / top-k modules stubbed (the stub Triton scorer records its calls
and computes with the native implementation), and the private implementation package
``aiter/ops/qsa_index_scorer`` loaded standalone (as the other QSA CPU tests do).

* contract: ``inspect.signature`` = ATOM's 12 leading parameters + main's
  ``stacked_decode`` + keyword-only ``logits`` / ``visible_groups``; the existing
  11-argument (and keyword) calls keep working; no top-level ``aiter`` export of the
  old 12-argument op; the implementation package is private;
* default off: the existing native dispatch serves every call it served before (no
  decision cache, no kernel import, no device query); CPU calls are served by the
  pure-torch path instead of raising;
* dispatch decisions at index page 4 / 8 / 16 for bf16 and fp8 caches (fused /
  native-triton / native-torch / invalid) with the profile configured;
* native numerics vs an independent float64 reference;
* per-class decision cache: one host decision per class, O(1) later calls, no tensor
  allocation and no host sync on the hot path when outputs are supplied;
* ``configure`` / ``$AITER_QSA_PROFILE`` / ``$AITER_QSA_FUSED=0`` precedence, idempotence,
  ``explain`` / ``status``; ``prewarm``;
* route-ready shim (``@route(op="qsa_paged_mqa_logits", version=1)`` when
  ``aiter.plugin`` exists);
* AOT job enumeration of ``aiter/aot/flydsl/qsa.py`` (every verified profile class);
* decode-group (any variant) FlyDSL cache-key determinism: no module state read at
  trace time (static) and identical keys in three compile orders (needs FlyDSL; skipped
  without it).

    python3 -B -m pytest -q op_tests/test_qsa_neutral_op_cpu.py
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_qsa_index_scorer_cpu as base
import test_qsa_index_scorer_native_cpu as nat

API, PR, PKG = base.API, base.PR, base.PKG
D = importlib.import_module(PKG.__name__ + ".dispatch")
NAT = nat.NAT
ROOT = base.ROOT
QSA_PY = os.path.join(ROOT, "aiter", "ops", "qsa.py")
AOT_PY = os.path.join(ROOT, "aiter", "aot", "flydsl", "qsa.py")
KERNEL_PY = os.path.join(
    ROOT,
    "aiter",
    "ops",
    "flydsl",
    "kernels",
    "mqa_logits",
    "qsa_paged_mqa_logits_page4_gfx1250.py",
)
_ENV_KEYS = (D.PROFILE_ENV, D.FUSED_ENV, D.CACHE_DIR_ENV)
CLEAN = {k: v for k, v in base._CLEAN.items() if k not in _ENV_KEYS}


# ---------------------------------------------------------------------------
# Loading aiter/ops/qsa.py without aiter (stubbed Triton / top-k)
# ---------------------------------------------------------------------------
class TritonStub(types.ModuleType):
    """``aiter.ops.triton.attention.qsa`` stand-in: records scorer calls and computes
    them with the native implementation (validated like Triton: ValueError)."""

    FP4_GROUP = 32
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = 448.0

    def __init__(self):
        super().__init__("aiter.ops.triton.attention.qsa")
        self.calls = []
        for n in (
            "qsa_cache_row_layout",
            "qsa_expand_block_indices",
            "qsa_sparse_paged_gqa",
            "qsa_store_quantized_rows",
        ):
            setattr(self, n, lambda *a, **k: None)
        self.FP4_GROUP, self.FP8_DTYPE, self.FP8_MAX = 32, torch.float8_e4m3fn, 448.0

    def qsa_paged_mqa_logits(
        self_,
        q,
        cache,
        pt,
        t2r,
        pos,
        ctx,
        ratio=4,
        divisor=None,
        max_columns=None,
        row_starts=None,
        compressed_k_scale=None,
        stacked_decode=False,
    ):
        self_.calls.append({"rows": q.shape[0], "stacked_decode": stacked_decode})
        return NAT.qsa_paged_mqa_logits_native(
            q,
            cache,
            pt,
            t2r,
            pos,
            ctx,
            ratio,
            divisor,
            max_columns,
            row_starts,
            False,
            compressed_k_scale,
        )


_N = [0]


def load_qsa(plugin=None):
    """``aiter/ops/qsa.py`` as a fresh module bound to this test's implementation
    package; ``plugin`` (a module) stands in for ``aiter.plugin`` (None: absent)."""
    tri = TritonStub()
    tri.qsa_paged_mqa_logits = types.MethodType(TritonStub.qsa_paged_mqa_logits, tri)
    topk = types.ModuleType("aiter.ops.topk")
    topk.top_k_per_row_prefill = lambda *a, **k: None
    stubs = {
        "aiter.ops.topk": topk,
        "aiter.ops.triton": types.ModuleType("aiter.ops.triton"),
        "aiter.ops.triton.attention": types.ModuleType("aiter.ops.triton.attention"),
        "aiter.ops.triton.attention.qsa": tri,
        "aiter.ops.qsa_index_scorer": PKG,
        "aiter.ops.qsa_index_scorer.dispatch": D,
        "aiter.plugin": plugin,
        "aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout": None,
        "aiter.ops.flydsl.qsa_sparse_prefill": None,
    }
    for name in ("aiter", "aiter.ops"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            stubs[name] = m
    _N[0] += 1
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(f"qsa_neutral_{_N[0]}", QSA_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod, tri


Q, TRI = load_qsa()
ATOM12 = API.ATOM_SCORER_SIGNATURE
#: aiter main a0c4b69c ``aiter.ops.qsa.qsa_paged_mqa_logits`` (13 positional parameters)
MAIN13 = (*ATOM12, ("stacked_decode", False))


def _clean_env(**extra):
    env = dict(CLEAN)
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


class _Base(unittest.TestCase):
    """Fresh policy (clean env, nothing configured); ``Q._on_gpu`` controls whether a
    CPU tensor counts as a GPU tensor for the existing-dispatch route."""

    gpu = False

    def setUp(self):
        self._p = [
            _clean_env(),
            mock.patch.object(Q, "_on_gpu", lambda t: self.gpu),
            mock.patch.object(D, "_API", API),
        ]
        for p in self._p:
            p.start()
        D._reset_for_testing()
        TRI.calls.clear()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()
        D._reset_for_testing()


# ---------------------------------------------------------------------------
class Contract(unittest.TestCase):
    def test_signature(self):
        params = list(inspect.signature(Q.qsa_paged_mqa_logits).parameters.values())
        self.assertEqual(tuple((p.name, p.default) for p in params[:12]), ATOM12)
        self.assertEqual(tuple((p.name, p.default) for p in params[:13]), MAIN13)
        self.assertTrue(
            all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params[:13])
        )
        self.assertEqual(
            [(p.name, p.default, p.kind) for p in params[13:]],
            [
                ("logits", None, inspect.Parameter.KEYWORD_ONLY),
                ("visible_groups", None, inspect.Parameter.KEYWORD_ONLY),
            ],
        )
        # explain takes exactly the op's arguments
        ex = list(inspect.signature(Q.explain).parameters.values())
        self.assertEqual(
            [(p.name, p.default, p.kind) for p in ex],
            [(p.name, p.default, p.kind) for p in params],
        )

    def test_names_versions_exports(self):
        self.assertEqual(
            (Q.OP_NAME, Q.OP_CONTRACT_VERSION, Q.API_VERSION),
            ("qsa_paged_mqa_logits", 1, 1),
        )
        self.assertEqual(Q.qsa_paged_mqa_logits.__name__, Q.OP_NAME)
        for suffix in ("_flydsl", "_gfx1250", "_triton", "_torch", "_fused"):
            self.assertFalse(Q.OP_NAME.endswith(suffix))
        for n in ("configure", "prewarm", "explain", "status", "Decision"):
            self.assertIn(n, Q.__all__)
        sig = inspect.signature(Q.configure)
        self.assertEqual(list(sig.parameters), ["profile", "source"])
        self.assertTrue(
            all(
                p.kind is inspect.Parameter.KEYWORD_ONLY
                for p in sig.parameters.values()
            )
        )
        sig = inspect.signature(Q.prewarm)
        self.assertEqual(
            [(n, p.default) for n, p in sig.parameters.items()],
            [
                ("page_sizes", inspect.Parameter.empty),
                ("cache_dtypes", inspect.Parameter.empty),
                ("pos_dtypes", (torch.int64,)),
                ("compressed_k_cache", None),
                ("compressed_k_scale", None),
                ("device", None),
            ],
        )

    def test_no_top_level_export_and_private_package(self):
        with open(os.path.join(ROOT, "aiter", "__init__.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = [
            n.module or ""
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level == 1
        ]
        self.assertFalse([m for m in mods if "qsa_index_scorer" in m], mods)
        self.assertFalse(
            os.path.exists(os.path.join(base.PKG_DIR, "op.py")), "public op module"
        )
        self.assertIn("not a public API", PKG.__doc__)
        self.assertFalse(hasattr(PKG, "qsa_paged_mqa_logits"))

    def test_eleven_and_keyword_calls(self):
        """Existing callers: 11 positional args (no scale), keywords, 13 positional."""
        with _clean_env():
            D._reset_for_testing()
            a = nat.make_call(rows=9, requests=3, width=7, seed=3)
            ref = nat.reference(*a[:6])
            nat.assert_matches(self, Q.qsa_paged_mqa_logits(*a[:11]), ref, "11 args")
            out = Q.qsa_paged_mqa_logits(
                *a[:6],
                compress_ratio=4,
                max_columns=None,
                row_starts=None,
                allow_flydsl=False,
                compressed_k_scale=None,
                stacked_decode=True,
            )
            nat.assert_matches(self, out, ref, "keywords")
            nat.assert_matches(self, Q.qsa_paged_mqa_logits(*a, True), ref, "13 args")

    def test_invalid_calls_raise_value_error(self):
        with _clean_env():
            D._reset_for_testing()
            a = nat.make_call()
            for name, i, v in (
                ("dim", 0, a[0][:, :, :64]),
                ("divisor", 7, 0.0),
                ("ratio", 6, 0),
                ("t2r", 3, a[3][:-1]),
            ):
                b = list(a)
                b[i] = v
                self.assertEqual(Q.explain(*b).path, D.PATH_INVALID, name)
                with self.assertRaises(ValueError, msg=name):
                    Q.qsa_paged_mqa_logits(*b)


class DefaultOff(_Base):
    gpu = True

    def test_existing_dispatch_unchanged(self):
        a = nat.make_call(rows=12, requests=3, width=5, seed=5)
        with mock.patch.object(
            API, "_load_kernel", side_effect=AssertionError("kernel import")
        ), mock.patch.object(
            API, "_device_info", side_effect=AssertionError("device query")
        ), mock.patch.object(
            API, "decide", side_effect=AssertionError("fused decide")
        ):
            out = Q.qsa_paged_mqa_logits(*a, True)
            st = Q.status()
        self.assertEqual(TRI.calls, [{"rows": 12, "stacked_decode": True}])
        nat.assert_matches(self, out, nat.reference(*a[:6], scale=a[11]))
        self.assertEqual(len(D._CLASSES), 0)  # no decision cache when off
        self.assertFalse(st["enabled"])
        self.assertIsNone(st["profile"])
        self.assertEqual(st["classes"], 0)
        d = Q.explain(*a)
        self.assertEqual(
            (d.path, d.reason, d.vid), (D.PATH_NATIVE_TRITON, "disabled", D.VID_TRITON)
        )

    def test_output_buffers_filled(self):
        a = nat.make_call(rows=6, requests=2, width=4, seed=6)
        lg = torch.empty((6, 16), dtype=torch.float32)
        vg = torch.empty(6, dtype=torch.int32)
        out = Q.qsa_paged_mqa_logits(*a, logits=lg, visible_groups=vg)
        self.assertIs(out[0], lg)
        self.assertIs(out[1], vg)
        nat.assert_matches(self, out, nat.reference(*a[:6], scale=a[11]))

    def test_cpu_call_served_by_torch(self):
        self.gpu = False
        a = nat.make_call(rows=10, requests=2, width=6, fmt="fp8", seed=7)
        out = Q.qsa_paged_mqa_logits(*a)
        self.assertEqual(TRI.calls, [])
        nat.assert_matches(self, out, nat.reference(*a[:6], scale=a[11]))
        d = Q.explain(*a)
        self.assertEqual((d.path, d.vid), (D.PATH_NATIVE_TORCH, D.VID_TORCH))

    def test_fp8_without_scale_keeps_existing_error(self):
        a = nat.make_call(fmt="fp8")
        a[11] = None
        self.assertEqual(Q._existing_route(a[0], a[1], a[2], False, None), None)
        self.assertEqual(Q.explain(*a).path, D.PATH_INVALID)
        with self.assertRaises(ValueError):
            Q.qsa_paged_mqa_logits(*a)


class Decisions(base.Env):
    """Profile configured; kernel injected; device faked as gfx1250 / 256 CUs."""

    gpu = True

    def setUp(self):
        super().setUp()
        self._q = [
            mock.patch.dict(
                os.environ,
                {k: v for k, v in os.environ.items() if k not in _ENV_KEYS},
                clear=True,
            ),
            mock.patch.object(Q, "_on_gpu", lambda t: self.gpu),
            mock.patch.object(D, "_API", API),
        ]
        for p in self._q:
            p.start()
        D._reset_for_testing()
        Q.configure(profile=PR.DEFAULT_PROFILE, source="test")
        TRI.calls.clear()

    def tearDown(self):
        for p in reversed(self._q):
            p.stop()
        D._reset_for_testing()
        super().tearDown()

    def _call(self, page, fmt, rows=64, requests=16, columns=4096, **kw):
        return base.call_args(
            rows=rows,
            requests=requests,
            width=columns // page,
            columns=columns,
            fmt=fmt,
            page=page,
            **kw,
        )

    def test_fused_page_4_8_16_bf16_fp8(self):
        for page in (4, 8, 16):
            for fmt in ("bf16", "fp8"):
                pre = PR.variant_prefix(page)
                for rows, reqs, kind in ((64, 16, "decg"), (16, 16, "draft")):
                    d = Q.explain(*self._call(page, fmt, rows=rows, requests=reqs))
                    self.assertEqual(
                        (d.path, d.reason, d.vid),
                        (D.PATH_FUSED, "ok", f"{pre}.{kind}.v1"),
                        (page, fmt, d),
                    )
                    self.assertTrue(d.cell.startswith("rule:"), d.cell)
                    self.assertEqual(d.profile, PR.DEFAULT_PROFILE)

    def test_uncovered_goes_native_with_reason(self):
        for page in (4, 8, 16):
            for fmt in ("bf16", "fp8"):
                a = self._call(page, fmt)
                a[7] = 8.0  # divisor the golden does not cover
                self.gpu = True
                d = Q.explain(*a)
                self.assertEqual(
                    (d.path, d.reason, d.vid),
                    (D.PATH_NATIVE_TRITON, "divisor_uncovered", D.VID_TRITON),
                )
                self.gpu = False
                D._CLASSES.clear()
                d = Q.explain(*a)
                self.assertEqual(
                    (d.path, d.reason, d.vid),
                    (D.PATH_NATIVE_TORCH, "divisor_uncovered", D.VID_TORCH),
                )
        a = base.call_args(rows=64, requests=16, width=128, columns=4096, page=32)
        self.gpu = True
        d = Q.explain(*a)
        self.assertEqual(
            (d.path, d.reason), (D.PATH_NATIVE_TRITON, "cache_shape_unsupported")
        )

    def test_fused_runs_through_launch_and_decides_once(self):
        a = self._call(8, "fp8")
        sentinel = (torch.zeros(1), torch.zeros(1))
        with mock.patch.object(
            API, "launch", return_value=sentinel
        ) as launch, mock.patch.object(API, "decide", wraps=API.decide) as dec:
            for _ in range(5):
                self.assertIs(Q.qsa_paged_mqa_logits(*a), sentinel)
        self.assertEqual(dec.call_count, 1)
        self.assertEqual(launch.call_count, 5)
        d0 = launch.call_args_list[0].args[0]
        self.assertTrue(all(c.args[0] is d0 for c in launch.call_args_list))
        self.assertEqual(TRI.calls, [])

    def test_first_launch_failure_falls_back_once(self):
        a = self._call(4, "bf16")
        with mock.patch.object(API, "launch", side_effect=RuntimeError("no hsaco")):
            out = Q.qsa_paged_mqa_logits(*a)
        self.assertEqual(len(TRI.calls), 1)
        self.assertEqual(out[0].shape, (64, 4096))
        d = Q.explain(*a)
        self.assertEqual(
            (d.path, d.reason), (D.PATH_NATIVE_TRITON, "fused_launch_error")
        )
        with mock.patch.object(D, "FUSED_FALLBACK_ON_ERROR", False):
            D._CLASSES.clear()
            with mock.patch.object(
                API, "launch", side_effect=RuntimeError("x")
            ), self.assertRaises(RuntimeError):
                Q.qsa_paged_mqa_logits(*a)

    def test_allow_flydsl_page16_route_is_existing_native(self):
        a = base.call_args(rows=512, requests=1, width=16, columns=4096, page=16)
        a[7] = 8.0  # fused declines
        a[10] = True
        with mock.patch.object(Q, "_HAS_FLYDSL", True), mock.patch.object(
            Q, "_FLYDSL_PAGE_SIZE", 16, create=True
        ):
            d = Q.explain(*a)
        self.assertEqual((d.path, d.vid), (D.PATH_NATIVE_TRITON, D.VID_FLYDSL_PAGE16))


class NativeNumerics(_Base):
    def test_torch_path_vs_reference(self):
        for page in (4, 8, 16):
            for fmt in ("bf16", "fp8"):
                for md in (torch.int32, torch.int64):
                    a = nat.make_call(
                        rows=11,
                        requests=3,
                        width=5,
                        page=page,
                        fmt=fmt,
                        md=md,
                        seed=page,
                    )
                    for mc in (None, 7):
                        b = list(a)
                        b[8] = mc
                        out = Q.qsa_paged_mqa_logits(*b)
                        ref = nat.reference(*b[:6], max_columns=mc, scale=b[11])
                        nat.assert_matches(self, out, ref, (page, fmt, md, mc))

    def test_fp16_and_fp32_caches(self):
        for dt in (torch.float16, torch.float32):
            a = nat.make_call(rows=7, requests=2, width=4, seed=9)
            a[0], a[1] = a[0].to(dt), a[1].to(dt)
            out = Q.qsa_paged_mqa_logits(*a)
            nat.assert_matches(self, out, nat.reference(*a[:6]), dt)


class ClassCache(_Base):
    def test_one_decision_per_class(self):
        a = nat.make_call(rows=8, requests=2, width=4, seed=1)
        b = nat.make_call(rows=9, requests=2, width=4, seed=1)
        with mock.patch.object(D, "_decide", wraps=D._decide) as dec:
            for _ in range(20):
                Q.qsa_paged_mqa_logits(*a)
            for _ in range(3):
                Q.qsa_paged_mqa_logits(*b)
        self.assertEqual(dec.call_count, 2)
        self.assertEqual(Q.status()["classes"], 2)
        self.assertEqual(Q.status()["paths"], {D.PATH_NATIVE_TORCH: 2})

    def test_key_covers_semantics(self):
        a = nat.make_call(rows=8, requests=2, width=4, seed=1)
        k = D.class_key(*a, False, None, None)
        for i, v in ((7, 3.0), (8, 5), (10, True), (6, 2)):
            b = list(a)
            b[i] = v
            self.assertNotEqual(D.class_key(*b, False, None, None), k, i)
        self.assertNotEqual(D.class_key(*a, True, None, None), k)
        b = list(a)
        b[4] = b[4].to(torch.int32)
        self.assertNotEqual(D.class_key(*b, False, None, None), k)

    def test_hot_path_allocates_nothing_and_never_syncs(self):
        for fmt in ("bf16", "fp8"):
            a = nat.make_call(rows=21, requests=4, width=30, fmt=fmt, seed=11)
            rows = a[0].shape[0]
            a[9] = torch.full((rows,), 3, dtype=torch.int32)
            lg = torch.empty((rows, 120), dtype=torch.float32)
            vg = torch.empty(rows, dtype=torch.int32)
            Q.qsa_paged_mqa_logits(*a, logits=lg, visible_groups=vg)  # decides
            e = next(iter(D._CLASSES.values()))
            plan = e.run.__closure__[0].cell_contents
            known = nat._storages(
                a,
                lg,
                vg,
                [v for v in vars(plan).values() if isinstance(v, torch.Tensor)],
            )
            audit = nat._Audit(known)
            with mock.patch.object(
                D, "_decide", side_effect=AssertionError("re-decide")
            ), audit:
                out = Q.qsa_paged_mqa_logits(*a, logits=lg, visible_groups=vg)
            self.assertEqual(audit.new, [], fmt)
            self.assertEqual(
                [o for o in audit.ops if o.startswith(nat._Audit.SYNC)], [], fmt
            )
            self.assertIs(out[0], lg)
            nat.assert_matches(self, (lg, vg), nat.reference(*a[:6], scale=a[11]))
            D._CLASSES.clear()

    def test_hot_path_cost_below_one_signature_bind(self):
        """Machine-independent ratio (absolute µs printed, never asserted): a decided
        class costs less host time than one ``inspect.Signature.bind`` of the op."""
        import time

        a = nat.make_call(rows=64, requests=16, width=16, seed=0)
        args = (*a, False, None, None)
        D._entry(Q._existing_native, Q._existing_route, args)
        sig = inspect.signature(Q.qsa_paged_mqa_logits)

        def best(fn, n=5000):
            out = []
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                out.append((time.perf_counter() - t0) / n)
            return min(out)

        hit = best(lambda: D._entry(Q._existing_native, Q._existing_route, args))
        bind = best(lambda: sig.bind(*a))
        print(
            f"decided-class lookup {hit * 1e6:.2f} us, Signature.bind {bind * 1e6:.2f} us"
        )
        self.assertLess(hit, bind)

    def test_bounded(self):
        with mock.patch.object(D, "MAX_CLASSES", 3):
            for r in range(1, 7):
                Q.qsa_paged_mqa_logits(*nat.make_call(rows=r, requests=1, width=2))
            self.assertEqual(len(D._CLASSES), 3)


class Policy(_Base):
    def _st(self):
        return Q.status()

    def test_precedence(self):
        prof = PR.DEFAULT_PROFILE
        Q.configure(profile=prof, source="atom:test")
        st = self._st()
        self.assertEqual(
            (st["enabled"], st["profile"], st["origin"]), (True, prof, "atom:test")
        )
        with _clean_env(**{D.PROFILE_ENV: "nope_v0"}):
            Q.configure(profile=prof, source="atom:test")  # env re-read here
            st = self._st()
            self.assertEqual(
                (st["profile"], st["origin"]), ("nope_v0", "env:" + D.PROFILE_ENV)
            )
            self.assertFalse(st["enabled"])  # unusable profile: reported, not raised
            self.assertIn("nope_v0", st["profile_error"] or "nope_v0")
        with _clean_env(**{D.PROFILE_ENV: "off"}):
            Q.configure(profile=prof, source="atom:test")
            self.assertIsNone(self._st()["profile"])
        for v in ("0", "off", "false", "no"):
            with _clean_env(**{D.FUSED_ENV: v, D.PROFILE_ENV: prof}):
                Q.configure(profile=prof, source="atom:test")
                st = self._st()
                self.assertEqual((st["enabled"], st["profile"]), (False, None), v)
                self.assertTrue(st["origin"].startswith("kill-switch"), st["origin"])
                self.gpu = True
                d = Q.explain(*nat.make_call())
                self.assertEqual(
                    (d.path, d.reason), (D.PATH_NATIVE_TRITON, "kill_switch")
                )
                self.gpu = False
        with _clean_env(**{D.FUSED_ENV: "1"}):
            Q.configure(profile=None, source="atom:test")
            self.assertFalse(self._st()["enabled"])

    def test_env_alone_enables_and_is_snapshotted(self):
        with _clean_env(**{D.PROFILE_ENV: PR.DEFAULT_PROFILE}):
            D._reset_for_testing()
            self.assertTrue(self._st()["enabled"])
        # snapshot: the env change above is not seen until configure / reset
        self.assertTrue(self._st()["enabled"])
        Q.configure(profile=None, source="t")
        self.assertFalse(self._st()["enabled"])

    def test_idempotent_and_reconfigure_drops_classes(self):
        Q.configure(profile=PR.DEFAULT_PROFILE, source="a")
        a = nat.make_call(rows=8, requests=2, width=4)
        Q.qsa_paged_mqa_logits(*a)
        self.assertEqual(self._st()["classes"], 1)
        Q.configure(profile=PR.DEFAULT_PROFILE, source="a")
        self.assertEqual(self._st()["classes"], 1)
        Q.configure(profile=None, source="a")
        self.assertEqual(self._st()["classes"], 0)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            Q.configure(profile=PR.DEFAULT_PROFILE, source="")
        with self.assertRaises(ValueError):
            Q.configure(profile="", source="x")
        with self.assertRaises(TypeError):
            Q.configure(PR.DEFAULT_PROFILE, "x")  # keyword-only

    def test_status_fields(self):
        st = self._st()
        for k in (
            "op",
            "op_version",
            "api_version",
            "enabled",
            "profile",
            "origin",
            "kernel_sha256",
            "classes",
            "paths",
            "routed",
        ):
            self.assertIn(k, st)
        Q.configure(profile=PR.DEFAULT_PROFILE, source="t")
        st = self._st()
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        self.assertEqual(st["kernel_sha256"], prof.kernel_sha256)  # profile pin holds
        self.assertEqual(st["profile_kernel_sha256"], prof.kernel_sha256)
        self.assertFalse(st["kernel_loaded"])  # status imports nothing

    def test_prewarm(self):
        self.assertEqual(Q.prewarm(page_sizes=(4,), cache_dtypes=("bf16",)), [])
        Q.configure(profile=PR.DEFAULT_PROFILE, source="t")
        with mock.patch.object(API, "prewarm", return_value=["x"]) as pw:
            got = Q.prewarm(
                page_sizes=(4, 16),
                cache_dtypes=(torch.bfloat16, torch.float8_e4m3fn, "fp8"),
            )
        self.assertEqual(got, ["x"])
        kw = pw.call_args.kwargs
        self.assertEqual(pw.call_args.args, (PR.DEFAULT_PROFILE,))
        self.assertEqual(kw["cache_dtypes"], ("bf16", "fp8", "fp8"))
        self.assertEqual(kw["page_sizes"], (4, 16))
        self.assertEqual(kw["pos_dtypes"], (torch.int64,))
        self.assertEqual(kw["cache_dir"], "")  # FlyDSL's own cache (the AOT cache)
        with self.assertRaises(TypeError):
            Q.prewarm(cache_dtypes=("bf16",))
        with self.assertRaises(ValueError):
            Q.prewarm(page_sizes=(4,), cache_dtypes=(torch.float16,))


class RouteReady(unittest.TestCase):
    def test_shim_without_plugin(self):
        self.assertFalse(Q.ROUTED)
        self.assertEqual(
            Q.qsa_paged_mqa_logits.__aiter_route__, ("qsa_paged_mqa_logits", 1)
        )

    def test_routed_when_plugin_exists(self):
        seen = []
        plugin = types.ModuleType("aiter.plugin")

        def route(*, op, version=1, supersedes=None):
            def wrap(f):
                seen.append((op, version, list(inspect.signature(f).parameters)))
                return f

            return wrap

        plugin.route = route
        mod, _ = load_qsa(plugin)
        self.assertTrue(mod.ROUTED)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][:2], ("qsa_paged_mqa_logits", 1))
        self.assertEqual(seen[0][2][:12], [n for n, _ in ATOM12])


class AotJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("aot_qsa_under_test", AOT_PY)
        cls.M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.M)
        with open(base.PROFILE_JSON, encoding="utf-8") as fh:
            cls.data = json.load(fh)

    def test_every_verified_class(self):
        jobs = self.M.collect_jobs([], None)
        want = set()
        for r in self.data["rows"]:
            if r["golden"]["status"] == "pass":
                want.add(
                    (
                        r["cache_dtype"],
                        r.get("page_size", 4),
                        r["rows"]["min"],
                        r["requests"]["min"],
                        r["columns"],
                        r["table_width"],
                        None,
                    )
                )
        for k in ("rules", "buckets"):
            for r in self.data[k]:
                if r["golden"]["status"] == "pass":
                    for s in r["samples"]:
                        want.add(
                            (
                                r["cache_dtype"],
                                r.get("page_size", 4),
                                s["rows"],
                                s["requests"],
                                s["columns"],
                                s["table_width"],
                                s.get("layout"),
                            )
                        )
        got = {
            (
                j["cache_dtype"],
                j["page"],
                j["rows"],
                j["requests"],
                j["columns"],
                j["table_width"],
                j["layout"],
            )
            for j in jobs
        }
        self.assertEqual(got, want)
        self.assertEqual(len(jobs), len(got))  # deduplicated
        self.assertEqual({j["page"] for j in jobs}, {4, 8, 16})
        self.assertEqual({j["cache_dtype"] for j in jobs}, {"bf16", "fp8"})
        self.assertEqual({j["pos_dtype"] for j in jobs}, {"int64"})
        self.assertEqual({(j["gfx"], j["cu_num"]) for j in jobs}, {("gfx1250", 256)})
        self.assertEqual(
            {j["kind"] for j in jobs}, {"row", "rule", "bucket"}
        )  # pending records are never replayed
        self.assertGreater(len(jobs), 2000)

    def test_registered_with_run_aot(self):
        with open(
            os.path.join(ROOT, "aiter", "aot", "flydsl", "common.py"), encoding="utf-8"
        ) as fh:
            tree = ast.parse(fh.read())
        mods = next(
            ast.literal_eval(n.value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "AOT_MODULES" for t in n.targets)
        )
        self.assertIn("aiter.aot.flydsl.qsa", mods)
        self.assertEqual(self.M.DEFAULT_CSVS, [])
        for n in ("replay", "describe", "collect_jobs", "configure", "LABEL"):
            self.assertTrue(hasattr(self.M, n), n)

    def test_profile_option(self):
        ns = types.SimpleNamespace(profile=base.PROFILE_JSON)
        self.assertEqual(
            len(self.M.collect_jobs([], ns)), len(self.M.collect_jobs([], None))
        )


# ---------------------------------------------------------------------------
# FlyDSL disk-cache key determinism (decode-group bf16 / fp8 and every variant)
# ---------------------------------------------------------------------------
_KEYS_SCRIPT = r"""
import hashlib, importlib.util, json, os, sys
root, order, out = sys.argv[1], sys.argv[2].split(";"), sys.argv[3]
import torch
torch.cuda.current_stream = lambda *a, **k: None
here = os.path.join(root, "aiter", "ops", "qsa_index_scorer")
spec = importlib.util.spec_from_file_location(
    "qsa_keys_probe", os.path.join(here, "__init__.py"), submodule_search_locations=[here])
pkg = importlib.util.module_from_spec(spec); sys.modules["qsa_keys_probe"] = pkg
spec.loader.exec_module(pkg)
api, P = sys.modules["qsa_keys_probe.api"], sys.modules["qsa_keys_probe.profiles"]
plan = {t + "|" + repr(items): (t, dt, items, pd) for t, dt, items, pd in api.prewarm_plan(
    P.DEFAULT_PROFILE, cache_dtypes=("bf16", "fp8"), pos_dtypes=(torch.int64,), page_sizes=(4,))}
from flydsl.compiler.jit_function import JitFunction
orig, seen = JitFunction._build_full_cache_key, []
def hook(self, *a, **k):
    key = orig(self, *a, **k); seen.append(hashlib.sha256(str(key).encode()).hexdigest()); return key
JitFunction._build_full_cache_key = hook
prof, K, res = P.load_profile(P.DEFAULT_PROFILE), api._load_kernel(), {}
names = sorted(plan)
for i in order:
    t, dt, items, pd = plan[names[int(i)]]
    n0 = len(seen)
    api._prewarm_one(K, prof, torch.device("cpu"), dt, t.split("/")[0], dict(items), pd, True)
    res[names[int(i)]] = seen[n0:]
json.dump(res, open(out, "w"))
"""


def _has_flydsl() -> bool:
    try:
        return importlib.util.find_spec("flydsl") is not None
    except (ImportError, ValueError):
        return False


class CacheKeyDeterminism(unittest.TestCase):
    def test_kernel_reads_no_lazily_filled_module_state(self):
        """Static: FlyDSL folds the value of every module global a kernel's code reads
        into its disk-cache key, so no trace-time helper may read a module-level
        container that is filled at run time (the R3 ``_RAW`` cause)."""
        with open(KERNEL_PY, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mutable = set()
        for n in tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                v = n.value
                if isinstance(v, (ast.Dict, ast.List, ast.Set)) and not getattr(
                    v, "keys", getattr(v, "elts", None)
                ):
                    for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                        mutable.add(t.id)
        # wrapper-side host caches (never read by traced code) are the only ones left
        self.assertEqual(mutable, {"_DUMMY_SCALE", "_I32_BOUNDS", "_NO_KW"})
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        raw_names = {x.id for x in ast.walk(fns["_raw"]) if isinstance(x, ast.Name)}
        self.assertFalse(raw_names & mutable)
        self.assertNotIn("_RAW", {t for t in mutable})
        # the wrapper caches are only touched by host-side helpers, never by builders
        for name, fn in fns.items():
            used = {x.id for x in ast.walk(fn) if isinstance(x, ast.Name)} & {
                "_DUMMY_SCALE",
                "_I32_BOUNDS",
            }
            if used:
                self.assertFalse(name.startswith("_build"), name)

    @unittest.skipUnless(_has_flydsl(), "FlyDSL not installed (no-device container)")
    def test_same_key_in_three_compile_orders(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("QSA_")}
        env.update(COMPILE_ONLY="1", FLYDSL_GPU_ARCH="gfx1250")
        with tempfile.TemporaryDirectory() as tmp:
            n = 13  # page-4 bf16 + fp8 variants of the profile
            orders = {
                "fwd": ";".join(str(i) for i in range(n)),
                "rev": ";".join(str(i) for i in reversed(range(n))),
                "odd": ";".join(
                    str(i) for i in (5, 0, 12, 3, 9, 1, 7, 11, 2, 8, 4, 10, 6)
                ),
            }
            res = {}
            for name, order in orders.items():
                out = os.path.join(tmp, name + ".json")
                env["FLYDSL_RUNTIME_CACHE_DIR"] = os.path.join(tmp, "c_" + name)
                subprocess.run(
                    [sys.executable, "-B", "-c", _KEYS_SCRIPT, ROOT, order, out],
                    env=env,
                    check=True,
                    timeout=3600,
                )
                with open(out, encoding="utf-8") as fh:
                    res[name] = json.load(fh)
            self.assertEqual(res["fwd"], res["rev"])
            self.assertEqual(res["fwd"], res["odd"])
            decg = [k for k in res["fwd"] if ".decg." in k]
            self.assertEqual(len(decg), 4)  # bf16 / fp8 x default / 3-row group


if __name__ == "__main__":
    unittest.main()
