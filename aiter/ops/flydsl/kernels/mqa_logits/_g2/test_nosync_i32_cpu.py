# SPDX-License-Identifier: MIT
"""ROUND-NOSYNC CPU gate: sync-free int64 metadata narrowing.

No aiter import, compiler, GPU runtime or launch. The real host wrapper runs on
CPU tensors; ``_run``/builders are replaced with recorders plus a scalar model
of the frozen kernel's i32 metadata arithmetic (``head = max(min(pos + 1, ctx),
0) // 4`` with *wrapping* i32 ``pos + 1``, request/page validity guards).

Checks
* boundary values (INT32_MIN/MAX, 2**31, -1, +-2**32+k low-word aliases,
  +-2**40, +-2**62, int64 extremes) for each of the 4 metadata tensors:
  saturated i32 kernel semantics == golden int64 formula (visible groups,
  request validity, page validity, logits mask) -- exact whenever the golden
  visible count is < 2**29, mask-exact always;
* end-to-end wrapper + model vs ``qsa_golden.reference_logits`` with int64
  boundary metadata (bf16 and fp8);
* per-call dispatch census: int32 metadata -> 0 tensor ops; each int64 tensor
  -> exactly one ``aten::clamp.Tensor_out`` (+ its output allocation); never
  ``_local_scalar_dense`` (host sync), min/max reductions or casts;
* QSA_STRICT_I32=1 restores the historical raise; read once at import.

ROUND-NATIVE64: int64 ``query_positions`` are no longer narrowed by the
wrapper. The launch receives the zero-copy int32 view [2*rows] of the caller's
tensor and a ``pos64=True`` specialization whose in-kernel saturation
(``kernel_sat_pos64`` below, a line-by-line model of ``_sat_pos64``; the real
source is executed in ``test_native64_cpu.py``) equals ``sat(v, INT32_MAX-1)``
for all 36 boundary values. The census for int64 positions is 0 ops.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.environ.get("QSA_GOLDEN_DIR", os.path.join(_HERE, "..", "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.abspath(_GOLDEN))
sys.path.insert(0, _HERE)

import qsa_golden as G  # noqa: E402
from _load import load_kernel_module  # noqa: E402

K, HOW = load_kernel_module(allow_stub=True)

I32_MIN, I32_MAX = -(2**31), 2**31 - 1
I64_MIN, I64_MAX = -(2**63), 2**63 - 1
BOUNDARY = sorted({
    0, 1, -1, 2, 3, 4, 5, 7, 8, 4095, 2**29 - 1, 2**29, 2**29 + 1,
    I32_MAX - 4, I32_MAX - 3, I32_MAX - 2, I32_MAX - 1, I32_MAX, 2**31, 2**31 + 1,
    2**32 - 1, 2**32, 2**32 + 1, 2**32 + 5, 2**33 + 3, 2**40, 2**62, I64_MAX,
    I32_MIN, I32_MIN + 1, I32_MIN - 1, -(2**32), -(2**32) + 5, -(2**40), -(2**62), I64_MIN,
})


def wrap32(value):
    value &= 0xFFFFFFFF
    return value - 2**32 if value >= 2**31 else value


def kernel_visible(req_i32, pos_i32, ctx_list_i32, requests):
    """The frozen kernel's per-row visible-group arithmetic, exactly in i32."""
    if not 0 <= req_i32 < requests:
        return 0
    head = max(min(wrap32(pos_i32 + 1), ctx_list_i32[req_i32]), 0)
    return head // 4


def golden_visible(req, pos, ctx_list, requests):
    """Golden int64 formula without the golden's final wrapping int32 cast."""
    if not 0 <= req < requests:
        return 0
    return max(0, min((pos + 1) // 4, ctx_list[req] // 4))


def sat(value, hi=I32_MAX):
    return max(I32_MIN, min(hi, value))


def kernel_sat_pos64(lo, hi):
    """Model of the kernel's ``_sat_pos64`` on the two i32 words of an int64."""
    ext = -1 if lo < 0 else 0
    fits = hi == ext
    outside = I32_MAX - 1 if hi >= 0 else I32_MIN
    return min(lo, I32_MAX - 1) if fits else outside


def split_words(values):
    words = torch.tensor(values, dtype=torch.int64).view(torch.int32).tolist()
    return list(zip(words[0::2], words[1::2]))


def launch_positions(launch, qp):
    """Per-row i32 position as the launched specialization computes it."""
    if getattr(launch, "pos64", False):
        assert qp.dtype == torch.int32 and qp.numel() % 2 == 0
        words = qp.tolist()
        return [kernel_sat_pos64(lo, hi) for lo, hi in zip(words[0::2], words[1::2])]
    return qp.tolist()


class SaturationSemanticsCPU(unittest.TestCase):
    def test_converter_values_dtype_and_zero_copy(self):
        src = torch.tensor(BOUNDARY, dtype=torch.int64)
        for hi in (I32_MAX, I32_MAX - 1):
            got = K._to_i32_saturated(src, hi)
            self.assertEqual(got.dtype, torch.int32)
            self.assertEqual(got.tolist(), [sat(v, hi) for v in BOUNDARY])
        # 2-D page table and a strided 1-D view keep shape/values.
        table = src[: 30].view(5, 6)
        self.assertEqual(K._to_i32_saturated(table).tolist(),
                         [[sat(v) for v in row] for row in table.tolist()])
        strided = src[::3]
        self.assertEqual(K._to_i32_saturated(strided).tolist(), [sat(v) for v in strided.tolist()])
        i32 = torch.tensor([I32_MIN, -1, 0, I32_MAX], dtype=torch.int32)
        self.assertIs(K._to_i32_saturated(i32), i32)
        self.assertIs(K._to_i32_saturated(i32, I32_MAX - 1), i32)
        # Historical helper is untouched: still rejects 2**31.
        with self.assertRaises(ValueError):
            K._to_i32_checked(torch.tensor([2**31], dtype=torch.int64), "m")

    def test_request_and_page_validity_preserved_for_every_boundary(self):
        for count in (1, 3, 4, 2**20, 2**29 - 1):
            for value in BOUNDARY:
                with self.subTest(count=count, value=value):
                    s = sat(value)
                    self.assertEqual(0 <= s < count, 0 <= value < count)
                    if 0 <= value < count:
                        self.assertEqual(s, value)

    def test_visible_groups_all_boundary_pairs(self):
        # Every (position, context) boundary pair, with request ids from the
        # boundary set too. columns*4 <= INT32_MAX => every column < 2**29 - 1.
        max_column = (I32_MAX // 4) - 1
        requests = 3
        for pos in BOUNDARY:
            for ctx in BOUNDARY:
                ctx_list = [ctx, 7, 0]
                ctx_i32 = [sat(c) for c in ctx_list]
                for req in (0, 1, 2, 3, -1, 2**31, 2**32, I64_MIN, I64_MAX, -(2**32)):
                    gold = golden_visible(req, pos, ctx_list, requests)
                    got = kernel_visible(sat(req), sat(pos, I32_MAX - 1), ctx_i32, requests)
                    with self.subTest(pos=pos, ctx=ctx, req=req):
                        if gold < 2**29:
                            self.assertEqual(got, gold)
                        else:
                            self.assertEqual(got, 2**29 - 1)
                        for col in {0, 1, max(gold - 1, 0), gold, min(got, max_column),
                                    max(got - 1, 0), max_column}:
                            if 0 <= col <= max_column:
                                self.assertEqual(col < got, col < gold, col)

    def test_in_kernel_saturation_model_all_36_boundaries(self):
        # ROUND-NATIVE64: lo/hi dwords exactly as the kernel loads them from the
        # zero-copy view; equals the NOSYNC wrapper clamp and the golden formula.
        self.assertEqual(len(BOUNDARY), 36)
        for value, (lo, hi) in zip(BOUNDARY, split_words(BOUNDARY)):
            with self.subTest(value=value):
                self.assertEqual((lo & 0xFFFFFFFF) + hi * 2**32, value)  # signed lo, hi
                self.assertEqual(kernel_sat_pos64(lo, hi), sat(value, I32_MAX - 1))
                for ctx in BOUNDARY:
                    ctx_list = [ctx]
                    gold = golden_visible(0, value, ctx_list, 1)
                    got = kernel_visible(0, kernel_sat_pos64(lo, hi), [sat(ctx)], 1)
                    self.assertEqual(got, gold if gold < 2**29 else 2**29 - 1)
        got = K._to_i32_saturated(torch.tensor(BOUNDARY, dtype=torch.int64), I32_MAX - 1)
        self.assertEqual(got.tolist(), [kernel_sat_pos64(lo, hi) for lo, hi in split_words(BOUNDARY)])

    def test_historical_endpoint_is_fixed_for_int64_positions(self):
        # Old path: int64 INT32_MAX passed the probe, became int32 INT32_MAX and
        # the kernel's i32 pos+1 wrapped to INT32_MIN => visible 0 (golden: ctx//4).
        ctx = [4096]
        old = kernel_visible(0, I32_MAX, ctx, 1)
        new = kernel_visible(0, sat(I32_MAX, I32_MAX - 1), ctx, 1)
        self.assertEqual(old, 0)
        self.assertEqual(new, golden_visible(0, I32_MAX, ctx, 1))
        self.assertEqual(new, 1024)


def model_run(records, launch, *args):
    fp8 = launch.dtype_tag in ("fp8_wmma", "fp8_decg")
    # ROUND-DECG2: the decode-group kernel (default QSA_DECODE_GROUP=all) always
    # carries the KS slot (bf16: dummy scale plane).
    if fp8 or launch.dtype_tag == "bf16_decg":
        q, cache, scale, table, t2r, qp, ctx, logits, vis, rst, gx, gy, rows, cols, pages, reqs, width, stream = args
        scale = scale if fp8 else None
    else:
        q, cache, table, t2r, qp, ctx, logits, vis, rst, gx, gy, rows, cols, pages, reqs, width, stream = args
        scale = None
    records.append((launch, args))
    for name, tensor in (("table", table), ("t2r", t2r), ("qp", qp), ("ctx", ctx)):
        assert tensor.dtype == torch.int32, name
    positions = launch_positions(launch, qp)
    q = q.view(torch.bfloat16).view(rows, 4, 128).float()
    keys = cache.view(torch.float8_e4m3fn if fp8 else torch.bfloat16).view(pages, 4, 128).float()
    ctx_list = ctx.tolist()
    logits.fill_(float("-inf"))
    rst.zero_()
    for r in range(rows):
        req = int(t2r[r])
        visible = kernel_visible(req, positions[r], ctx_list, reqs)
        vis[r] = visible
        valid = 0 <= req < reqs
        for c in range(cols):
            logical = c // 4
            physical = int(table[req if valid else 0, logical]) if logical < width else -1
            if valid and 0 <= physical < pages and c < visible:
                k = keys[physical, c % 4]
                if fp8:
                    k = k * scale[physical, c % 4, 0]
                logits[r, c] = torch.relu(q[r] @ k).sum() / math.sqrt(128)


class WrapperCPU(unittest.TestCase):
    def setUp(self):
        self.records = []

        def bf16(divisor, geometry, mode, pos64=False):
            cols = (K.WMMA_COLS * K.TDM_PERSIST_CHUNKS if mode == "persist" else
                    K.R16_TDM_COLS if geometry == 16 and mode == "tdm" else 64)
            return type("Launch", (), {"dtype_tag": "bf16_wmma", "cols_per_block": cols,
                                       "pos64": pos64})()

        def fp8(divisor, geometry, mode, pos64=False):
            return type("Launch", (), {"dtype_tag": "fp8_wmma", "pos64": pos64})()

        def decg(is_fp8, divisor, wave_cols=32, pos64=False):
            # ROUND-DECG2: default decode route (rows > requests, r16 regime).
            return type("Launch", (), {"dtype_tag": ("fp8" if is_fp8 else "bf16") + "_decg",
                                       "pos64": pos64, "wave_cols": wave_cols})()

        self.stack = [
            patch.object(K, "_run", lambda launch, *a: model_run(self.records, launch, *a)),
            patch.object(K, "_build_wmma_kernel", bf16),
            patch.object(K, "_build_wmma_fp8_kernel", fp8),
            patch.object(K, "_build_decode_group_kernel", decg),
            patch.object(torch.cuda, "current_stream", lambda dev=None: None),
            patch.dict(os.environ, {"QSA_PAD_BUCKET": "0"}),
        ]
        for p in self.stack:
            p.start()

    def tearDown(self):
        for p in reversed(self.stack):
            p.stop()

    @staticmethod
    def fixture(fmt, table, t2r, qp, ctx, rows):
        g = torch.Generator().manual_seed(7)
        q = torch.randn(rows, 4, 128, generator=g).to(torch.bfloat16)
        values = torch.randn(8, 4, 1, 128, generator=g) * 3.0
        if fmt == "bf16":
            cache, scale = values.to(torch.bfloat16), None
        else:
            cache, scale = G.quantize_fp8_rows(values)
        return dict(q=q, compressed_k_cache=cache, compressed_k_scale=scale,
                    page_table=torch.tensor(table, dtype=torch.int64),
                    token_to_request=torch.tensor(t2r, dtype=torch.int64),
                    query_positions=torch.tensor(qp, dtype=torch.int64),
                    context_lens=torch.tensor(ctx, dtype=torch.int64))

    def check_against_golden(self, data):
        out, vis = K.qsa_paged_mqa_logits_page4(**data)
        # The golden indexes keys before masking, so it cannot take page ids
        # >= cache_pages; every out-of-range id means "invalid" to it, so map
        # them to -1 in int64 (no narrowing) for the golden input only.
        pages = data["compressed_k_cache"].shape[0]
        table = data["page_table"].to(torch.int64)
        table = torch.where((table >= 0) & (table < pages), table, torch.full_like(table, -1))
        gold, gold_vis, valid = G.reference_logits(
            data["q"], data["compressed_k_cache"], table, data["token_to_request"],
            data["query_positions"], data["context_lens"],
            compressed_k_scale=data["compressed_k_scale"])
        self.assertTrue(torch.equal(vis, gold_vis), (vis, gold_vis))
        self.assertTrue(torch.equal(torch.isneginf(out), torch.isneginf(gold)))
        finite = ~torch.isneginf(gold)
        self.assertTrue(torch.allclose(out[finite], gold[finite], rtol=1e-4, atol=1e-4))
        return out, vis

    def test_end_to_end_int64_boundaries_match_golden(self):
        # Rows: [valid normal, pos INT32_MAX (historical wrap), pos 2**40,
        # pos -2**40, req 2**32 (low word 0!), req -2**32+1, req INT32_MAX,
        # pos 2**32+3 (low word 3), huge-ctx request, low-word-12 ctx request].
        # Golden visible counts stay < 2**29 so the golden's own int32 cast is exact.
        t2r = [0, 0, 0, 1, 2**32, -(2**32) + 1, I32_MAX, 0, 1, 2]
        qp = [6, I32_MAX, 2**40, -(2**40), 5, 5, 5, 2**32 + 3, 27, 47]
        rows = len(t2r)
        # ctx: req0 small, req1 2**40 (huge, saturates), req2 2**33+12 (low word 12).
        ctx = [20, 2**40, 2**33 + 12]
        # Page table: req0 normal; req1 has 2**32+1 (low word 1!), -2**33, 2**31;
        # req2 has I64_MAX / I64_MIN.
        table = [[1, 2, 3, 4, 5, 6],
                 [2**32 + 1, 3, -(2**33), 2**31, 4, 5],
                 [I64_MAX, 6, I64_MIN, 7, 0, 1]]
        for fmt in ("bf16", "fp8"):
            with self.subTest(fmt=fmt):
                data = self.fixture(fmt, table, t2r, qp, ctx, rows)
                out, vis = self.check_against_golden(data)
                launch, args = self.records[-1]
                idx = 3 if fmt == "fp8" or launch.dtype_tag.endswith("_decg") else 2
                pt, rt2r, rqp, rctx = args[idx:idx + 4]
                # ROUND-NATIVE64: raw int64 words, zero-copy, pos64 specialization;
                # the in-kernel saturation yields the former wrapper values.
                self.assertTrue(launch.pos64)
                self.assertEqual(rqp.data_ptr(), data["query_positions"].data_ptr())
                self.assertEqual(rqp.view(torch.int64).tolist(), qp)
                self.assertEqual(launch_positions(launch, rqp), [sat(v, I32_MAX - 1) for v in qp])
                self.assertEqual(rt2r.tolist(), [sat(v) for v in t2r])
                self.assertEqual(rctx.tolist(), [sat(v) for v in ctx])
                self.assertEqual(pt.tolist(), [[sat(v) for v in row] for row in table])
                # Historical wrap row now visible: min(2**29, 20//4) = 5.
                self.assertEqual(vis.tolist(), [1, 5, 5, 0, 0, 0, 0, 5, 7, 12])
                # Low-word aliases (2**32+1 -> page 1) must stay invalid.
                self.assertTrue(bool(torch.isneginf(out[8, 0:4]).all()))
                self.assertFalse(bool(torch.isneginf(out[8, 4:7]).any()))
                self.assertTrue(bool(torch.isneginf(out[9, 0:4]).all()))
                self.assertFalse(bool(torch.isneginf(out[9, 4:8]).any()))
                self.assertTrue(bool(torch.isneginf(out[9, 8:]).all()))
                self.assertTrue(bool(torch.isneginf(out[8, 7:]).all()))

    def test_int32_metadata_is_zero_copy_and_unchanged(self):
        data = self.fixture("bf16", [[1, 2, 3, 4]], [0, 0, 0, 0], [3, 7, 11, 15], [16], 4)
        for key in ("page_table", "token_to_request", "query_positions", "context_lens"):
            data[key] = data[key].to(torch.int32)
        self.check_against_golden(data)
        launch, args = self.records[-1]
        self.assertFalse(launch.pos64)   # int32 callers: the unchanged specialization
        # ROUND-DECG2: 4 rows / 1 request now default-routes to decg (KS slot -> +1).
        base = 3 if launch.dtype_tag.endswith("_decg") else 2
        for i, key in zip(range(base, base + 4), ("page_table", "token_to_request", "query_positions", "context_lens")):
            self.assertEqual(args[i].data_ptr(), data[key].data_ptr(), key)

    def test_dispatch_census_no_sync_one_op_per_int64_tensor(self):
        class Census(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.ops = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                overload = func._schema.overload_name
                name = func._schema.name + ("." + overload if overload else "")
                self.ops.append(name)
                if func._schema.name in ("aten::_local_scalar_dense", "aten::item"):
                    raise AssertionError(f"host synchronization: {name}")
                return func(*args, **(kwargs or {}))

        views = {"aten::view", "aten::_unsafe_view", "aten::alias", "aten::detach"}
        launches = []
        base = self.fixture("bf16", [[1, 2, 3, 4]], [0, 0, 0, 0], [3, 7, 11, 15], [16], 4)
        keys = ("page_table", "token_to_request", "query_positions", "context_lens")
        for int64_keys in ((), ("query_positions",), ("token_to_request", "query_positions"), keys):
            data = dict(base)
            for key in keys:
                data[key] = base[key].to(torch.int64 if key in int64_keys else torch.int32)
            out = torch.empty(4, 16, dtype=torch.float32)
            vis = torch.empty(4, dtype=torch.int32)
            rst = torch.empty(4, dtype=torch.int32)
            data["q"].view(torch.uint8)
            data["compressed_k_cache"].view(torch.uint8)
            # ROUND-DECG2: 4 rows / 1 request default-routes (when 128 B aligned) to
            # decg, which passes the per-device cached dummy KS plane (one-time
            # aten::zeros); warm it like the other one-time caches.
            K._dummy_scale(data["q"].device)
            K._to_i32_saturated(torch.zeros(1, dtype=torch.int64))  # warm bound cache
            K._to_i32_saturated(torch.zeros(1, dtype=torch.int64), I32_MAX - 1)
            census = Census()
            with patch.object(K, "_run", lambda launch, *a: launches.append(a)), census:
                K.qsa_paged_mqa_logits_page4(**data, logits=out, visible_groups=vis, row_starts=rst)
            work = [op for op in census.ops if op.split(".")[0] not in views]
            with self.subTest(int64=int64_keys):
                # ROUND-NATIVE64: int64 query_positions cost no op (view only).
                narrowed = [k for k in int64_keys if k != "query_positions"]
                self.assertEqual(sorted(work), sorted(
                    ["aten::empty.memory_format", "aten::clamp.Tensor_out"] * len(narrowed)), census.ops)
                self.assertFalse(any(("min" in op and "clamp" not in op) or "max" in op
                                     or "_to_copy" in op or "copy_" in op for op in work))

    def test_strict_mode_restores_raise_and_env_read_once(self):
        data = self.fixture("bf16", [[1, 2, 3, 4]], [0, 0, 0, 0], [3, 7, 11, 15], [16], 4)
        with patch.object(K, "_QSA_STRICT_I32", True):
            for key, bad in (("page_table", [[2**31, 1, 2, 3]]), ("token_to_request", [0, 0, 0, -(2**31) - 1]),
                             ("query_positions", [3, 7, 11, 2**31]), ("context_lens", [2**40])):
                bad_data = dict(data, **{key: torch.tensor(bad, dtype=torch.int64)})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "do not fit in int32"):
                    K.qsa_paged_mqa_logits_page4(**bad_data)
            self.check_against_golden(data)  # in-range int64 still exact in strict mode
        for env, enabled in (("0", False), ("1", True), (" 1 ", True), ("yes", False)):
            with patch.dict(os.environ, {"QSA_STRICT_I32": env}):
                fresh, _ = load_kernel_module(allow_stub=True)
            self.assertIs(fresh._QSA_STRICT_I32, enabled)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QSA_STRICT_I32", None)
            fresh, _ = load_kernel_module(allow_stub=True)
        self.assertIs(fresh._QSA_STRICT_I32, False)


if __name__ == "__main__":
    print("kernel module:", HOW)
    unittest.main()
