# SPDX-License-Identifier: MIT
"""CPU-only bucket planning, real host dispatch and opaque K-byte checks.

No aiter package import, compiler, GPU runtime or device launch. The host
wrapper is real; _run is replaced with a recording scalar contract model.
Real-source WMMA store/descriptor traces live in test_pad_bucket_trace_cpu.py.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from _load import load_kernel_module
from test_k_tdm_cpu import fallback_fragments, gather_padded_tile, padded_tile_fragment
from test_fp8_k_tdm_cpu import raw_fragment as fp8_fragment, stage_tile as fp8_tile, lds_fragment as fp8_lds_fragment

K, HOW = load_kernel_module(allow_stub=True)


def plan(rows, cols, requests, width, dtype="bf16", pages=8):
    return K.qsa_pad_plan((rows, 4, cols, requests, width, pages), dtype)


def model_run(launch, *args):
    fp8 = launch.dtype_tag in ("fp8_wmma", "fp8_draft", "fp8_decg")
    # ROUND-MERGE1: the default draft kernel (rows <= requests) takes the fp8
    # argument layout for both dtypes (bf16 passes a dummy scale plane).
    # ROUND-DECG2: so does the decode-group kernel (default QSA_DECODE_GROUP=all).
    if fp8 or launch.dtype_tag in ("bf16_draft", "bf16_decg"):
        q, cache, scale, table, t2r, qp, ctx, logits, vis, rst, gx, gy, rows, cols, pages, reqs, width, stream = args
    else:
        q, cache, table, t2r, qp, ctx, logits, vis, rst, gx, gy, rows, cols, pages, reqs, width, stream = args
    q = q.view(torch.bfloat16).view(rows, 4, 128).float()
    cache = cache.view(torch.float8_e4m3fn if fp8 else torch.bfloat16).view(pages, 4, 128).float()
    logits.fill_(float("-inf"))
    rst.zero_()
    for r in range(rows):
        req = int(t2r[r])
        valid = 0 <= req < reqs
        visible = max(0, min((int(qp[r]) + 1) // 4, int(ctx[req]) // 4)) if valid else 0
        vis[r] = visible
        for c in range(cols):
            logical = c // 4
            physical = int(table[req if valid else 0, logical if logical < width else 0])
            good = valid and logical < width and 0 <= physical < pages and c < visible
            if good:
                key = cache[physical, c % 4]
                dots = (q[r] * key).sum(-1)
                if fp8:
                    dots = dots * scale[physical, c % 4, 0]
                logits[r, c] = dots.relu().sum() / math.sqrt(128)


class PadPlanCPU(unittest.TestCase):
    def test_table_and_pinned_true_routing(self):
        cases = [
            (1, 1, 1, 1, 16, 32, 16, 32, 4, 8),
            (7, 5, 3, 17, 16, 32, 16, 32, 4, 24),
            (37, 67, 3, 17, 16, 32, 48, 96, 4, 24),
            (37, 67, 1, 17, 32, 64, 64, 128, 4, 32),
            (65, 257, 5, 65, 16, 32, 80, 288, 8, 72),
            (512, 1000, 2, 256, 32, 512, 512, 1024, 4, 256),
        ]
        for r, c, n, w, br, bc, pr, pc, pn, pw in cases:
            with self.subTest(rows=r, columns=c, requests=n):
                p = plan(r, c, n, w)
                self.assertEqual((p["rows_per_block"], p["tile_cols"]), (br, bc))
                self.assertEqual(tuple(p["padded_shape"][k] for k in
                                       ("rows", "columns", "requests", "page_table_width")),
                                 (pr, pc, pn, pw))
                self.assertEqual(p["copy_back"], (r, c))
                self.assertGreaterEqual(pc, c)
                self.assertGreaterEqual(pr, r)
                self.assertLessEqual(pc, pw * 4)
        # Padding requests first would incorrectly select r16 for this prefill.
        self.assertEqual(plan(37, 67, 1, 17)["rows_per_block"], 32)
        self.assertEqual(K.wmma_rows_per_block(64, 4), 16)

    def test_overrides_keep_candidate_and_actual_kernel_family(self):
        with patch.object(K, "_ACCESS", "cop"):
            p = plan(37, 67, 1, 17)
            self.assertEqual((p["rows_per_block"], p["tile_cols"], p["k_load"]), (32, 64, "raw"))
        with patch.object(K, "_BF16_R64", True):
            p = plan(65, 67, 1, 17)
            self.assertEqual((p["rows_per_block"], p["tile_cols"]), (64, 64))
            self.assertEqual(p["padded_shape"]["rows"], 128)

    def test_fp8_and_exact_identity_plan(self):
        p = plan(37, 67, 3, 17, "fp8")
        self.assertEqual((p["rows_per_block"], p["tile_cols"], p["k_load"]), (16, 64, "raw"))
        self.assertEqual(p["padded_shape"]["columns"], 128)
        p = plan(16, 32, 4, 8)
        self.assertFalse(p["needs_padding"])
        self.assertEqual(p["true_shape"], p["padded_shape"])
        p = K.qsa_pad_plan(dict(rows=7, heads=4, columns=5, requests=3,
                               page_table_width=17, cache_pages=8, regime="draft"), torch.bfloat16)
        self.assertEqual(p["regime"], "draft")

    def test_import_time_native_units_match_champion_overrides(self):
        defaults = {"QSA_G2_ACCESS": "raw", "QSA_BF16_K_LOAD": "auto",
                    "QSA_BF16_R16_TDM_COLS": "32", "QSA_BF16_R64": "0",
                    "QSA_BF16_PERSIST": "1", "QSA_PERSIST_COLS": "8"}
        overrides = ({}, {"QSA_BF16_K_LOAD": "raw"}, {"QSA_G2_ACCESS": "cop"},
                     {"QSA_BF16_R16_TDM_COLS": "64"}, {"QSA_BF16_R64": "1"},
                     {"QSA_BF16_R64": "1", "QSA_BF16_K_LOAD": "raw"},
                     {"QSA_BF16_PERSIST": "0"}, {"QSA_PERSIST_COLS": "2"},
                     {"QSA_PERSIST_COLS": "4"})
        for settings in overrides:
            with self.subTest(settings=settings), patch.dict(os.environ, dict(defaults, **settings)):
                fresh, _ = load_kernel_module(allow_stub=True)
                r16 = fresh.qsa_pad_plan((16, 4, 64, 4, 16, 8), "bf16")
                r32 = fresh.qsa_pad_plan((128, 4, 32768, 4, 8192, 8), "bf16")
                self.assertEqual(fresh._QSA_PAD_BF16_R16_COLS, r16["tile_cols"])
                self.assertEqual(fresh._QSA_PAD_BF16_R32_COLS, r32["tile_cols"])
                self.assertEqual(fresh._QSA_PAD_BF16_R64,
                                 fresh.wmma_bf16_rows_per_block(64, 1) == 64)
                # Below the persistent activation threshold the native unit is64.
                short = fresh.qsa_pad_plan((32, 4, 64, 1, 16, 8), "bf16")
                self.assertEqual(short["tile_cols"], 64)

    def test_reject_empty_overflow_and_never_truncate(self):
        for values in ((0, 4, 1, 1, 1, 1), (1, 4, 0, 1, 1, 1),
                       (1, 4, 1, 0, 1, 1), (1, 4, 5, 1, 1, 1),
                       (1, 3, 1, 1, 1, 1), (65536, 4, 1, 1, 1, 1),
                       (8191, 4, 65536, 1, 16384, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                K.qsa_pad_plan(values, "bf16")
        with self.assertRaises(ValueError):
            K.qsa_pad_plan((1, 4, 1, 1, 1, 1), "fp16")

    def test_guard_product_is_python_i64_safe(self):
        with self.assertRaises(ValueError):
            plan(16, 1, 2**30, 1)
        with self.assertRaises(ValueError):
            plan(16, 1, 1, 2**30)


class PadDispatchCPU(unittest.TestCase):
    def setUp(self):
        self.records = []
        def run(launcher, *args):
            self.records.append((launcher, args))
            model_run(launcher, *args)
        def bf16(divisor, geometry, mode):
            cols = K.WMMA_COLS * K.TDM_PERSIST_CHUNKS if mode == "persist" else K.R16_TDM_COLS if geometry == 16 and mode == "tdm" else 64
            return type("Launch", (), {"dtype_tag": "bf16_wmma", "cols_per_block": cols,
                                        "geometry": geometry, "mode": mode})()
        def fp8(divisor, geometry, mode):
            return type("Launch", (), {"dtype_tag": "fp8_wmma", "geometry": geometry, "mode": mode})()
        def decg(is_fp8, divisor, wave_cols=32, pos64=False):
            # ROUND-DECG2: default decode route; stubbed like the other builders
            # (a cold real build reads FLYDSL_* env and would break the getenv census).
            return type("Launch", (), {"dtype_tag": ("fp8" if is_fp8 else "bf16") + "_decg",
                                        "wave_cols": wave_cols})()
        self.stack = [patch.object(K, "_run", run), patch.object(K, "_build_wmma_kernel", bf16),
                      patch.object(K, "_build_wmma_fp8_kernel", fp8),
                      patch.object(K, "_build_decode_group_kernel", decg),
                      patch.object(torch.cuda, "current_stream", lambda dev: None)]
        for p in self.stack:
            p.start()

    def tearDown(self):
        for p in reversed(self.stack):
            p.stop()

    def fixture(self, rows, cols, requests, width, dtype):
        g = torch.Generator().manual_seed(41)
        q = torch.randn(rows, 4, 128, generator=g).to(torch.bfloat16)
        cache = torch.randn(8, 4, 1, 128, generator=g).to(torch.bfloat16 if dtype == "bf16" else torch.float8_e4m3fn)
        table = torch.arange(requests * width, dtype=torch.int32).view(requests, width) % 8
        table[:, 0] = -1
        if width > 1:
            table[:, 1] = 999
        t2r = torch.arange(rows, dtype=torch.int32) % requests
        if rows > 1:
            t2r[-1] = requests  # Must not become valid when requests pad to 4/8.
        qp = torch.arange(rows, dtype=torch.int32) * 4 + 3
        ctx = torch.arange(requests, dtype=torch.int32) * 8 + cols * 4
        scale = torch.ones(8, 4, 1, dtype=torch.float32) if dtype == "fp8" else None
        return q, cache, table, t2r, qp, ctx, scale

    def call(self, data, columns, env):
        q, cache, table, t2r, qp, ctx, scale = data
        # Sentinel backing tests ensure copy-back never touches a caller tail.
        backing = torch.full((q.shape[0] * columns + 7,), 12345., dtype=torch.float32)
        logits = backing[:-7].view(q.shape[0], columns)
        vb = torch.full((q.shape[0] + 3,), 98765, dtype=torch.int32)
        rb = vb.clone()
        with patch.dict(os.environ, {"QSA_PAD_BUCKET": env}):
            out, vis = K.qsa_paged_mqa_logits_page4(q, cache, table, t2r, qp, ctx,
                max_columns=columns, compressed_k_scale=scale, logits=logits,
                visible_groups=vb[:-3], row_starts=rb[:-3])
        self.assertIs(out, logits)
        self.assertTrue(torch.equal(backing[-7:], torch.full((7,), 12345.)))
        self.assertTrue(torch.equal(vb[-3:], torch.full((3,), 98765, dtype=torch.int32)))
        self.assertTrue(torch.equal(rb[-3:], torch.full((3,), 98765, dtype=torch.int32)))
        self.assertTrue(bool((rb[:-3] == 0).all()))
        return out, vis

    def test_default_off_true_extents_and_padded_on_bytes(self):
        cases = ((1, 1, 1, 1), (7, 5, 3, 17), (37, 67, 3, 17),
                 (37, 67, 1, 17), (15, 63, 5, 17), (16, 32, 4, 8))
        for dtype in ("bf16", "fp8"):
            for rows, cols, reqs, width in cases:
                with self.subTest(dtype=dtype, rows=rows, cols=cols):
                    data = self.fixture(rows, cols, reqs, width, dtype)
                    off = self.call(data, cols, "0")
                    off_record = self.records[-1]
                    # ROUND-MERGE1: the exact (QSA_PAD_BUCKET=0) path of a draft
                    # shape is the default draft kernel unless QSA_DRAFT_PATH=0.
                    draft = K._DRAFT_PATH and rows <= reqs
                    self.assertEqual(off_record[0].dtype_tag.endswith("_draft"), draft)
                    on = self.call(data, cols, "1")
                    self.assertTrue(torch.equal(off[0].view(torch.int32), on[0].view(torch.int32)))
                    self.assertTrue(torch.equal(off[1], on[1]))
                    args = self.records[-1][1]
                    # ROUND-DECG2: the decode-group launch (default for rows >
                    # requests, r16 regime) always carries the KS slot.
                    idx = 3 if dtype == "fp8" or self.records[-1][0].dtype_tag.endswith("_decg") else 2
                    pt, t2r, qp, ctx = args[idx:idx + 4]
                    p = plan(rows, cols, reqs, width, dtype)
                    if p["needs_padding"]:
                        self.assertEqual(pt.shape, (p["padded_shape"]["requests"], p["padded_shape"]["page_table_width"]))
                        self.assertTrue(bool((t2r[rows:] == -1).all()))
                        self.assertTrue(bool((qp[rows:] == 0).all()))
                        self.assertTrue(bool((ctx[reqs:] == 0).all()))
                        self.assertTrue(bool((pt[reqs:] == -1).all()))
                        self.assertTrue(bool((pt[:reqs, width:] == -1).all()))
                        if rows > 1:
                            self.assertEqual(int(t2r[rows - 1]), -1)
                        plogits = args[idx + 4]
                        self.assertTrue(bool(torch.isneginf(plogits[:, cols:]).all()))
                        self.assertTrue(bool(torch.isneginf(plogits[rows:]).all()))
                    elif not draft:
                        # ROUND-DECG2: same kernel family (dtype_tag; decg launches
                        # have no ``geometry``) and geometry on both paths.
                        def kind(launch):
                            return launch.dtype_tag, getattr(launch, "geometry", None)
                        self.assertEqual(kind(off_record[0]), kind(self.records[-1][0]))

    def test_exact_bucket_reuses_outputs_and_scalar_override_stays_exact(self):
        data = self.fixture(16, 32, 4, 8, "bf16")
        out, vis = self.call(data, 32, "1")
        args = self.records[-1][1]
        # ROUND-DECG2: decg (default for this 16x4 decode bucket) has the KS slot.
        o = 1 if self.records[-1][0].dtype_tag.endswith("_decg") else 0
        self.assertIs(args[6 + o], out)
        self.assertEqual(args[0].data_ptr(), data[0].data_ptr())
        self.assertEqual(args[2 + o].data_ptr(), data[2].data_ptr())
        self.assertEqual(args[3 + o].data_ptr(), data[3].data_ptr())
        self.assertEqual(args[4 + o].data_ptr(), data[4].data_ptr())

    def test_native_buckets_zero_allocations_syncs_and_cold_calls(self):
        class ViewOnlyCounter(TorchDispatchMode):
            def __init__(self):
                self.views = 0
                self.allocations = 0
                self.synchronizations = 0
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = func._schema.name
                if name in ("aten::view", "aten::_unsafe_view", "aten::alias", "aten::detach"):
                    self.views += 1  # Same alias-only views as the original wrapper.
                elif name == "aten::_local_scalar_dense":
                    self.synchronizations += 1
                    raise AssertionError(f"device scalar synchronization: {func}")
                else:
                    self.allocations += 1
                    raise AssertionError(f"non-view tensor work/allocation: {func}")
                return func(*args, **(kwargs or {}))
        # A superset of the nine approved buckets, both column capacities and
        # prefill req1/2. Allocate fixtures/outputs BEFORE instrumented launches.
        for dtype in ("bf16", "fp8"):
            for columns in (32768, 65536):
                for rows, requests in ((64, 16), (16, 4), (512, 1), (512, 2), (512, 4)):
                    with self.subTest(dtype=dtype, columns=columns, rows=rows, requests=requests):
                        data = self.fixture(rows, columns, requests, columns // 4, dtype)
                        q, cache, table, t2r, qp, ctx, scale = data
                        # Warm PyTorch's lazy dtype-view/Dynamo setup; identical
                        # one-time initialization exists on the PAD=0 path.
                        with ViewOnlyCounter():
                            cache.view(torch.uint8)
                        # ROUND-DECG2: the default bf16 decode route (decg) passes the
                        # per-device cached dummy KS plane; warm its one-time allocation.
                        K._dummy_scale(q.device)
                        out = torch.empty((rows, columns), dtype=torch.float32)
                        vis = torch.empty(rows, dtype=torch.int32)
                        rst = torch.empty(rows, dtype=torch.int32)
                        captured = []
                        counter = ViewOnlyCounter()
                        with patch.dict(os.environ, {"QSA_PAD_BUCKET": "1"}), \
                             patch.object(K, "_QSA_PAD_INT32_GUARD", False), \
                             patch.object(K, "_run", lambda launch, *args: captured.append((launch, args))), \
                             patch.object(K, "qsa_pad_plan", side_effect=AssertionError("planner on native bucket")), \
                             patch.object(K, "_qsa_pad_validate_buffers", side_effect=AssertionError("alias validation on native bucket")), \
                             patch.object(K, "_qsa_pad_checked_dispatch", side_effect=AssertionError("cold dispatch on native bucket")), \
                             patch.object(K, "_qsa_pad_dispatch", side_effect=AssertionError("padding on native bucket")), \
                             patch.object(torch.cuda, "synchronize", side_effect=AssertionError("explicit synchronization")) as sync, \
                             patch.object(os.environ, "get", wraps=os.environ.get) as getenv, counter:
                            result = K.qsa_paged_mqa_logits_page4(q, cache, table, t2r, qp, ctx,
                                max_columns=columns, compressed_k_scale=scale,
                                logits=out, visible_groups=vis, row_starts=rst)
                        self.assertEqual((counter.allocations, counter.synchronizations), (0, 0))
                        self.assertGreater(counter.views, 0)
                        sync.assert_not_called()
                        self.assertEqual(getenv.call_args_list, [(("QSA_PAD_BUCKET", "0"), {})])
                        self.assertIs(result[0], out)
                        launch, args = captured[0]
                        # ROUND-DECG2: default decode routes (64x16, 16x4) launch the
                        # decode-group kernel, whose layout always has the KS slot.
                        idx = 3 if dtype == "fp8" or launch.dtype_tag.endswith("_decg") else 2
                        self.assertEqual(tuple(args[-6:-1]), (rows, columns, 8, requests, columns // 4))
                        for actual, original in zip(args[idx:idx + 4], (table, t2r, qp, ctx)):
                            self.assertEqual(actual.data_ptr(), original.data_ptr())
                        self.assertIs(args[idx + 4], out)
                        self.assertEqual(result[1].data_ptr(), vis.data_ptr())

    def test_metadata_only_req1_req2_bypass_preserves_invalid_requests(self):
        for dtype in ("bf16", "fp8"):
            for requests in (1, 2):
                data = self.fixture(32, 512, requests, 128, dtype)
                self.assertTrue(plan(32, 512, requests, 128, dtype)["needs_padding"])
                off = self.call(data, 512, "0")
                with patch.object(K, "qsa_pad_plan", side_effect=AssertionError("metadata-only native plan")):
                    on = self.call(data, 512, "1")
                self.assertTrue(torch.equal(off[0].view(torch.int32), on[0].view(torch.int32)))
                self.assertTrue(torch.equal(off[1], on[1]))
                self.assertEqual(int(on[1][-1]), 0)
                self.assertEqual(self.records[-1][1][-3], requests)

    def test_endpoint_env_is_read_once_before_import(self):
        for env, enabled in (("0", False), ("1", True), ("yes", False)):
            with patch.dict(os.environ, {"QSA_PAD_INT32_GUARD": env}):
                fresh, _ = load_kernel_module(allow_stub=True)
            self.assertIs(fresh._QSA_PAD_INT32_GUARD, enabled)
        # Diagnostic must not affect PAD=0; padding-needed always saturates,
        # independently of the diagnostic, and never performs a host scalar scan.
        exact = list(self.fixture(16, 32, 4, 8, "bf16"))
        exact[4][0] = 2**31 - 1
        with patch.object(K, "_QSA_PAD_INT32_GUARD", True):
            self.call(exact, 32, "0")
        for enabled in (False, True):
            padded = list(self.fixture(1, 1, 1, 1, "bf16"))
            padded[4][0] = 2**31 - 1
            with patch.object(K, "_QSA_PAD_INT32_GUARD", enabled):
                self.call(padded, 1, "1")
            self.assertEqual(int(self.records[-1][1][4][0]), 2**31 - 2)
        scalar = list(self.fixture(7, 5, 3, 17, "bf16"))
        scalar[4][0] = 2**31 - 1
        with patch.object(K, "_BF16_PATH", "scalar"), patch.object(K, "_QSA_PAD_INT32_GUARD", True), \
             self.assertRaisesRegex(ValueError, "INT32_MAX"):
            self.call(scalar, 5, "1")

    def test_overlapping_outputs_and_inputs_rejected_before_launch(self):
        q, cache, table, t2r, qp, ctx, scale = self.fixture(7, 5, 3, 17, "bf16")
        with patch.dict(os.environ, {"QSA_PAD_BUCKET": "1"}):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                K.qsa_paged_mqa_logits_page4(q, cache, table, t2r, qp, ctx,
                    max_columns=5, visible_groups=t2r)
            output = torch.empty(7, dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                K.qsa_paged_mqa_logits_page4(q, cache, table, t2r, qp, ctx,
                    max_columns=5, visible_groups=output, row_starts=output)
        self.assertEqual(len(self.records), 0)

    def test_scalar_override_stays_exact_and_endpoint_rejects(self):
        data = self.fixture(7, 5, 3, 17, "bf16")
        captured = []
        def capture(launch, *args):
            captured.append(args)
            args[7].fill_(float("-inf")); args[8].zero_(); args[9].zero_()
        with patch.object(K, "_BF16_PATH", "scalar"), patch.object(K, "_run", capture), patch.object(K, "_build_kernel", lambda *args: object()):
            self.call(data, 5, "0")
            self.call(data, 5, "1")
        # Scalar launch keeps true rows/columns/requests/width, never padding.
        for args in captured:
            self.assertEqual(tuple(args[-6:-1]), (7, 5, 8, 3, 17))
            self.assertEqual(args[0].data_ptr(), data[0].data_ptr())
        exact_data = list(self.fixture(16, 32, 4, 8, "bf16"))
        exact_data[4][0] = 2**31 - 1
        # Default-off diagnostic preserves the frozen endpoint behavior without
        # an endpoint allocation/scan. Explicit diagnostic rejects on bypass.
        with patch.object(K, "_QSA_PAD_INT32_GUARD", False):
            self.call(exact_data, 32, "1")
        with patch.object(K, "_QSA_PAD_INT32_GUARD", True), self.assertRaisesRegex(ValueError, "INT32_MAX"):
            self.call(exact_data, 32, "1")

    def test_fp8_tdm_alignment_falls_back_to_raw_not_misread(self):
        # Contiguous fp8 subview allowed at +8B, but not aligned for TDM.
        data = list(self.fixture(37, 67, 1, 17, "fp8"))
        raw = torch.zeros(data[1].numel() + 8, dtype=torch.uint8)
        cache = raw[8:].view(torch.float8_e4m3fn).view_as(data[1])
        cache.copy_(data[1])
        data[1] = cache
        self.call(data, 67, "1")
        launch = self.records[-1][0]
        self.assertEqual((launch.geometry, launch.mode), (16, "raw"))

    def test_position_endpoint_saturates_without_visible_change(self):
        data = list(self.fixture(1, 1, 1, 1, "bf16"))
        data[4][0] = 2**31 - 1
        data[5][0] = 2**31 - 1
        _, vis = self.call(data, 1, "1")
        self.assertEqual(int(vis[0]), (2**31 - 1) // 4)
        self.assertEqual(int(self.records[-1][1][4][0]), 2**31 - 2)

    def test_unsafe_copy_back_and_empty_shapes_rejected(self):
        data = self.fixture(7, 5, 3, 17, "bf16")
        q, cache, table, t2r, qp, ctx, scale = data
        with patch.dict(os.environ, {"QSA_PAD_BUCKET": "1"}), self.assertRaises(ValueError):
            K.qsa_paged_mqa_logits_page4(q, cache, table, t2r, qp, ctx, max_columns=5,
                                        visible_groups=torch.empty(14, dtype=torch.int32)[::2])
        for zero in ("rows", "columns"):
            with patch.dict(os.environ, {"QSA_PAD_BUCKET": "1"}), self.assertRaises(ValueError):
                K.qsa_paged_mqa_logits_page4(q[:0] if zero == "rows" else q,
                    cache, table, t2r[:0] if zero == "rows" else t2r,
                    qp[:0] if zero == "rows" else qp, ctx, max_columns=0 if zero == "columns" else 5)


class PadKFragmentCPU(unittest.TestCase):
    def test_fp8_all_codes_true_and_padded_fragment_bytes(self):
        source = bytes(range(256)) * 16  # all E4M3 encodings, eight pages
        for width, columns in ((1, 1), (17, 67), (65, 257)):
            p = plan(37, columns, 3, width, "fp8", pages=8)
            table = [(i * 7) % 8 for i in range(width)]
            table[0] = -1
            if width > 1:
                table[1] = 2**31 - 1
            padded = table + [-1] * (p["padded_shape"]["page_table_width"] - width)
            for origin in range(0, p["padded_shape"]["columns"], 64):
                tile, issues, masks, writes = fp8_tile(source, padded, origin,
                    p["padded_shape"]["columns"], 3)
                for nt in range(4):
                    for chunk in range(4):
                        for lane in range(32):
                            exact = fp8_fragment(source, table, origin, nt, chunk, lane, columns, 3)
                            pad = fp8_fragment(source, padded, origin, nt, chunk, lane,
                                               p["padded_shape"]["columns"], 3)
                            lds = fp8_lds_fragment(tile, nt, chunk, lane)
                            self.assertEqual(exact[:2], pad[:2])
                            self.assertEqual(exact[:2], lds[:2])
                            c = origin + nt * 16 + lane % 16
                            if c < columns:
                                self.assertEqual(exact[2], pad[2])
                self.assertTrue(all(0 <= row < 8 * 4 for issue in issues for row in issue.row_ids))
                self.assertLessEqual(max(writes), 1)

    def test_bf16_all_fragment_bytes_true_and_padded_table(self):
        # Includes all opaque payload kinds; no numeric conversion.
        encodings = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 0xFFFF]
        encodings += [(i * 40503) & 0xFFFF for i in range(4096 - len(encodings))]
        source = struct.pack("<4096H", *encodings)
        for width, columns in ((1, 1), (17, 67), (65, 257)):
            p = plan(37, columns, 3, width, pages=8)
            table = [(i * 7) % 8 for i in range(width)]
            table[0] = -1
            if width > 1:
                table[1] = 2**31 - 1
            padded = table + [-1] * (p["padded_shape"]["page_table_width"] - width)
            for origin in range(0, p["padded_shape"]["columns"], 64):
                exact, em = fallback_fragments(source, table, origin, columns, 3)
                frag, pm = fallback_fragments(source, padded, origin, p["padded_shape"]["columns"], 3)
                tile, groups, masks, starts = gather_padded_tile(source, padded, origin,
                    p["padded_shape"]["columns"], 3)
                for key in exact:
                    self.assertEqual(exact[key], frag[key])
                    self.assertEqual(padded_tile_fragment(tile, *key), exact[key])
                for key in em:
                    c = origin + key[0] * 16 + key[1] % 16
                    if c < columns:
                        self.assertEqual(em[key], pm[key])
                self.assertTrue(all(0 <= row < 8 * 4 for group in groups for row in group))


if __name__ == "__main__":
    unittest.main()
