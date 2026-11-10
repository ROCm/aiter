# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests of the routed op ``aiter.ops.qsa.qsa_paged_mqa_logits`` (contract v1).

No GPU, no built aiter: ``aiter/plugin`` (standard library only) and
``aiter/ops/qsa_index_scorer/native.py`` load standalone; ``aiter/ops/qsa.py`` loads
from its file with aiter's Triton / top-k modules stubbed (the stub Triton scorer
records its calls and computes with the native implementation).

* contract: ``inspect.signature`` (ATOM's 12 leading parameters + ``stacked_decode`` +
  keyword-only ``logits`` / ``visible_groups``), the ``tests/plugin_contracts.json``
  baseline entry equals what the static gate reads from the source;
* routing through the real ``aiter.plugin``: native by default (nothing recorded),
  a fake provider served when enabled (canonical 13 + 2 arguments; no native kernel
  reached), decline -> native, a ``run()`` error propagates (no native retry),
  ``AITER_PLUGINS=0``;
* native implementation: torch numerics vs an independent float64 reference (bf16 /
  fp8 + scale, invalid pages / requests, ``max_columns``), contract errors,
  ``stacked_decode=None`` = aiter decides (ATOM's predicate), the FlyDSL page-16 arch
  gate (gfx9 only), ``native_route`` labels.

    python3 -B -m pytest -q op_tests/test_qsa_paged_mqa_logits_route_cpu.py
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OP = "qsa_paged_mqa_logits"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if "aiter.plugin" in sys.modules:  # a built aiter in this process
    plugin = sys.modules["aiter.plugin"]
else:
    plugin = _load("aiter.plugin", os.path.join(ROOT, "aiter", "plugin", "__init__.py"))
NAT = _load(
    "qsa_route_cpu_native",
    os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer", "native.py"),
)


class TritonStub(types.ModuleType):
    """``aiter.ops.triton.attention.qsa`` stand-in: records scorer calls and computes
    them with the native implementation."""

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

        def scorer(
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
            self.calls.append({"rows": q.shape[0], "stacked_decode": stacked_decode})
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

        self.qsa_paged_mqa_logits = scorer


def load_qsa():
    tri = TritonStub()
    topk = types.ModuleType("aiter.ops.topk")
    topk.top_k_per_row_prefill = lambda *a, **k: None
    pkg = types.ModuleType("aiter.ops.qsa_index_scorer")
    pkg.native = NAT
    stubs = {
        "aiter.ops.topk": topk,
        "aiter.ops.triton": types.ModuleType("aiter.ops.triton"),
        "aiter.ops.triton.attention": types.ModuleType("aiter.ops.triton.attention"),
        "aiter.ops.triton.attention.qsa": tri,
        "aiter.ops.qsa_index_scorer": pkg,
        "aiter.ops.qsa_index_scorer.native": NAT,
        "aiter.plugin": plugin,
        "aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout": None,
        "aiter.ops.flydsl.qsa_sparse_prefill": None,
    }
    for name in ("aiter", "aiter.ops"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            stubs[name] = m
    with mock.patch.dict(sys.modules, stubs):
        mod = _load("qsa_route_cpu_op", os.path.join(ROOT, "aiter", "ops", "qsa.py"))
    return mod, tri


Q, TRI = load_qsa()


# ---------------------------------------------------------------------------
# calls and an independent float64 reference
# ---------------------------------------------------------------------------
def make_call(
    rows=12, requests=3, width=5, page=4, heads=4, dim=128, fmt="bf16", seed=0
):
    g = torch.Generator().manual_seed(seed)
    pages = requests * width + 2
    q = torch.randn(rows, heads, dim, generator=g).to(torch.bfloat16)
    kv = torch.randn(pages, page, 1, dim, generator=g)
    scale = None
    if fmt == "fp8":
        scale = torch.rand(pages, page, 1, generator=g) + 0.5
        cache = kv.to(torch.float8_e4m3fn)
    else:
        cache = kv.to(torch.bfloat16)
    pt = torch.randperm(pages, generator=g)[: requests * width].view(requests, width)
    pt = pt.to(torch.int32)
    pt[0, -1] = -1  # an invalid page
    t2r = torch.randint(0, requests, (rows,), generator=g, dtype=torch.int32)
    t2r[-1] = requests + 3  # an invalid request
    pos = torch.randint(0, width * page * 4, (rows,), generator=g, dtype=torch.int64)
    ctx = torch.randint(
        1, width * page * 4, (requests,), generator=g, dtype=torch.int32
    )
    return [q, cache, pt, t2r, pos, ctx, 4, None, None, None, False, scale]


def reference(
    q, cache, pt, t2r, pos, ctx, ratio=4, divisor=None, max_columns=None, scale=None
):
    rows, _heads, dim = q.shape
    page = cache.shape[1]
    columns = pt.shape[1] * page
    if max_columns is not None:
        columns = max(1, min(columns, int(max_columns)))
    div = math.sqrt(dim) if divisor is None else float(divisor)
    k = cache.to(torch.float64)
    if scale is not None:
        k = k * scale.to(torch.float64)[..., None]
    out = torch.full((rows, columns), float("-inf"), dtype=torch.float64)
    vis = torch.zeros(rows, dtype=torch.int32)
    for t in range(rows):
        r = int(t2r[t])
        if not 0 <= r < pt.shape[0]:
            continue
        v = max(min((int(pos[t]) + 1) // ratio, int(ctx[r]) // ratio), 0)
        vis[t] = v
        for c in range(min(v, columns)):
            phys = int(pt[r, c // page])
            if not 0 <= phys < cache.shape[0]:
                continue
            dots = q[t].to(torch.float64) @ k[phys, c % page, 0]
            out[t, c] = torch.relu(dots).sum() / div
    return out, vis


def assert_matches(tc, out, ref, msg=""):
    logits, vis = out
    tc.assertTrue(torch.equal(vis.view(-1).to(torch.int32), ref[1]), msg)
    fin = torch.isfinite(ref[0])
    tc.assertTrue(torch.equal(torch.isfinite(logits), fin), msg)
    torch.testing.assert_close(
        logits.to(torch.float64)[fin], ref[0][fin], rtol=1e-4, atol=1e-4, msg=msg
    )


# ---------------------------------------------------------------------------
class _Base(unittest.TestCase):
    gpu = False

    def setUp(self):
        self._p = [
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(Q, "_on_gpu", lambda t: self.gpu),
        ]
        for p in self._p:
            p.start()
        os.environ.pop("AITER_PLUGINS", None)
        plugin._reset_for_testing()
        TRI.calls.clear()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()
        plugin._reset_for_testing()


class Contract(unittest.TestCase):
    def test_signature(self):
        params = list(inspect.signature(Q.qsa_paged_mqa_logits).parameters.values())
        self.assertEqual(
            [(p.name, p.default) for p in params[6:]],
            [
                ("compress_ratio", 4),
                ("score_divisor", None),
                ("max_columns", None),
                ("row_starts", None),
                ("allow_flydsl", False),
                ("compressed_k_scale", None),
                ("stacked_decode", None),
                ("logits", None),
                ("visible_groups", None),
            ],
        )
        kinds = [p.kind for p in params]
        self.assertEqual(kinds.count(inspect.Parameter.KEYWORD_ONLY), 2)
        self.assertEqual(Q.OP_NAME, OP)
        self.assertEqual(Q.qsa_paged_mqa_logits.__aiter_plugin_op__, plugin.Op(OP, 1))
        self.assertIn(OP, plugin.status()["routed_ops"])

    def test_baseline_entry_is_what_the_gate_reads(self):
        gate = _load(
            "qsa_route_cpu_gate",
            os.path.join(ROOT, "tests", "test_plugin_contracts.py"),
        )
        found = gate.routed_contracts(gate._REPO)[OP]
        with open(os.path.join(ROOT, "tests", "plugin_contracts.json")) as fh:
            recorded = json.load(fh)[OP]
        self.assertEqual(recorded["version"], 1)
        self.assertEqual(recorded["fingerprint"], found["fingerprint"])
        self.assertEqual(recorded["signature"], found["signature"])
        self.assertEqual(recorded["source"].replace("\\", "/"), "aiter/ops/qsa.py")


@unittest.skipIf(
    not hasattr(plugin, "_entry_points"), "aiter.plugin without _entry_points"
)
class Routing(_Base):
    class _EP:
        def __init__(self, name, target):
            self.name, self.target, self.value = name, target, "fake:provider"

        def load(self):
            return self.target

    class _Provider:
        api_versions = frozenset({1})

        def __init__(self, name, serve=True, error=None):
            self.name, self.serve, self.error = name, serve, error
            self.op_versions = {OP: {1}}
            self.seen, self.runs = [], 0

        def _run(
            self,
            q,
            compressed_k_cache,
            page_table,
            token_to_request,
            query_positions,
            context_lens,
            compress_ratio=4,
            score_divisor=None,
            max_columns=None,
            row_starts=None,
            allow_flydsl=False,
            compressed_k_scale=None,
            stacked_decode=None,
            *,
            logits=None,
            visible_groups=None,
        ):
            self.runs += 1
            if self.error is not None:
                raise self.error
            return "provider-logits", "provider-visible"

        def select(self, op, args, kwargs):
            self.seen.append((op, len(args), sorted(kwargs)))
            if not self.serve:
                return None
            return plugin.KernelCandidate(
                provider=self.name,
                kernel_id="fake.qsa",
                artifact_digest="sha256:0",
                run=self.run_fn,
            )

    def _install(self, prov):
        prov.run_fn = prov._run  # one stable callable
        p = mock.patch.object(
            plugin, "_entry_points", lambda: (self._EP(prov.name, prov),)
        )
        p.start()
        self.addCleanup(p.stop)
        plugin._reset_for_testing()

    def test_native_by_default_records_nothing(self):
        prov = self._Provider("fake-qsa")
        self._install(prov)
        a = make_call()
        assert_matches(self, Q.qsa_paged_mqa_logits(*a), reference(*a[:6]))
        self.assertEqual(prov.seen, [])
        self.assertIsNone(plugin.last_decision())
        d = plugin.explain(OP, *a).decision
        self.assertEqual((d.route, d.reason), ("native", "provider-not-enabled"))

    def test_enabled_provider_serves_canonical_call(self):
        prov = self._Provider("fake-qsa")
        self._install(prov)
        plugin.configure(source="test", enable=["fake-qsa"])
        a = make_call()
        self.assertEqual(
            Q.qsa_paged_mqa_logits(*a[:6]), ("provider-logits", "provider-visible")
        )
        self.assertEqual(prov.seen[-1][1:], (13, ["logits", "visible_groups"]))
        self.assertEqual(TRI.calls, [])
        d = plugin.last_decision()
        self.assertEqual(
            (d.route, d.provider, d.kernel_id), ("provider", "fake-qsa", "fake.qsa")
        )

    def test_decline_runs_native(self):
        prov = self._Provider("fake-qsa", serve=False)
        self._install(prov)
        plugin.configure(source="test", enable=["fake-qsa"])
        a = make_call(fmt="fp8", seed=3)
        out = Q.qsa_paged_mqa_logits(*a)
        assert_matches(self, out, reference(*a[:6], scale=a[11]))
        d = plugin.last_decision()
        self.assertEqual((d.route, d.reason), ("native", "provider-declined"))

    def test_run_error_propagates_without_native_retry(self):
        prov = self._Provider("fake-qsa", error=RuntimeError("kernel failed"))
        self._install(prov)
        plugin.configure(source="test", enable=["fake-qsa"])
        self.gpu = True
        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            Q.qsa_paged_mqa_logits(*make_call())
        self.assertEqual((prov.runs, TRI.calls), (1, []))

    def test_kill_switch(self):
        prov = self._Provider("fake-qsa")
        self._install(prov)
        plugin.configure(source="test", enable=["fake-qsa"])
        os.environ["AITER_PLUGINS"] = "0"
        plugin.reload_config()
        a = make_call()
        assert_matches(self, Q.qsa_paged_mqa_logits(*a), reference(*a[:6]))
        self.assertEqual(prov.seen, [])
        self.assertIsNone(plugin.last_decision())


class Native(_Base):
    def test_cpu_numerics_bf16_fp8_max_columns(self):
        for fmt in ("bf16", "fp8"):
            for page in (4, 16):
                a = make_call(fmt=fmt, page=page, seed=7)
                assert_matches(
                    self,
                    Q.qsa_paged_mqa_logits(*a),
                    reference(*a[:6], scale=a[11]),
                    fmt,
                )
                b = list(a)
                b[8] = 9
                assert_matches(
                    self,
                    Q.qsa_paged_mqa_logits(*b),
                    reference(*a[:6], max_columns=9, scale=a[11]),
                    f"{fmt} max_columns",
                )
        self.assertEqual(TRI.calls, [])
        self.assertEqual(Q.native_route(a[0], a[1], a[2]), Q.NATIVE_TORCH)

    def test_output_buffers_filled(self):
        a = make_call(seed=8)
        logits = torch.empty(12, 20, dtype=torch.float32)
        vis = torch.empty(12, dtype=torch.int32)
        out = Q.qsa_paged_mqa_logits(*a, logits=logits, visible_groups=vis)
        self.assertEqual(out[0].data_ptr(), logits.data_ptr())
        self.assertEqual(out[1].data_ptr(), vis.data_ptr())
        assert_matches(self, (logits, vis), reference(*a[:6]))

    def test_contract_violation_raises_value_error(self):
        a = make_call(fmt="fp8")
        a[11] = None  # fp8 without its scale
        with self.assertRaises(ValueError):
            Q.qsa_paged_mqa_logits(*a)


class StackedAuto(_Base):
    gpu = True

    def _hint(self, a, *extra, **kw):
        TRI.calls.clear()
        out = Q.qsa_paged_mqa_logits(*a, *extra, **kw)
        assert_matches(self, out, reference(*a[:6], scale=a[11]))
        self.assertEqual(len(TRI.calls), 1)
        return TRI.calls[0]["stacked_decode"]

    def test_default_is_auto_and_explicit_values_pass_through(self):
        a = make_call(seed=21)
        self.assertEqual(Q.native_route(a[0], a[1], a[2]), Q.NATIVE_TRITON)
        self.assertIs(self._hint(a), True)
        self.assertIs(self._hint(a, None), True)
        self.assertIs(self._hint(a, False), False)
        self.assertIs(self._hint(a, stacked_decode=True), True)
        self.assertIs(self._hint(make_call(rows=257, seed=22)), False)
        self.assertIs(self._hint(make_call(heads=3, seed=22)), False)


class FlydslPage16ArchGate(_Base):
    gpu = True

    def test_route_by_arch(self):
        a = make_call(rows=300, requests=1, width=4, page=16, seed=31)
        for arch, label in (
            ("gfx942", Q.NATIVE_FLYDSL_PAGE16),
            ("gfx950", Q.NATIVE_FLYDSL_PAGE16),
            ("gfx1250", Q.NATIVE_TRITON),
        ):
            with mock.patch.object(Q, "_HAS_FLYDSL", True), mock.patch.object(
                Q, "_FLYDSL_PAGE_SIZE", 16, create=True
            ), mock.patch.object(Q, "_device_arch", lambda dev, arch=arch: arch):
                self.assertEqual(Q.native_route(a[0], a[1], a[2], True), label, arch)


if __name__ == "__main__":
    unittest.main()
