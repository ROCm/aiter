# SPDX-License-Identifier: MIT
"""ROUND-LAUNCH CPU gate: lower per-call host cost, identical launch arguments.

No device. The real wrapper runs on CPU tensors with ``_run`` replaced by a
recorder. Checks
* launch arguments (pointer, shape, stride, dtype of every tensor; every scalar)
  are identical to the historical ``.contiguous().view(-1)`` / ``.view(-1)``
  forms for: wrapper-allocated outputs, caller 1-D outputs, caller 2-D [rows,1]
  outputs, strided 1-D metadata (copied exactly as before), size-1 metadata
  with a non-unit stride, int64 positions;
* returned visible_groups equals the launch's visible plane, 1-D int32 [rows];
* the stream fast path returns ``torch._C._cuda_getCurrentRawStream(index)`` for
  CUDA devices only when ``torch.cuda.current_stream`` is the stock function;
  a patched ``current_stream`` is honoured (and CPU devices never take it);
* no cache was introduced: every guard still runs on every call (alignment,
  plane, shape, dtype, non-viewable caller output).
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _load import load_kernel_module  # noqa: E402

K, HOW = load_kernel_module(allow_stub=True)


class _Launch:
    cols_per_block = 64
    dtype_tag = "stub"


def meta(t):
    if isinstance(t, torch.Tensor):
        return ("T", t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
    return ("S", t)


def fixture(rows=16, requests=4, width=32, fmt="bf16", qp64=True):
    q = torch.randn(rows, K.NUM_HEADS, K.HEAD_DIM).to(torch.bfloat16)
    pages = 64
    if fmt == "bf16":
        cache = torch.zeros((pages, 4, 1, K.HEAD_DIM), dtype=torch.bfloat16)
        scale = None
    else:
        cache = torch.zeros((pages, 4, 1, K.HEAD_DIM), dtype=torch.float8_e4m3fn)
        scale = torch.ones((pages, 4, 1), dtype=torch.float32)
    pt = torch.arange(requests * width, dtype=torch.int32).view(requests, width) % pages
    t2r = torch.arange(requests, dtype=torch.int32).repeat_interleave(rows // requests)
    qp = torch.arange(rows, dtype=torch.int64 if qp64 else torch.int32) + 100
    cl = torch.full((requests,), 4 * width, dtype=torch.int32)
    return [q, cache, pt, t2r, qp, cl], dict(compress_ratio=4, compressed_k_scale=scale)


def historical_args(args, kw):
    """Launch-argument metadata the pre-ROUND-LAUNCH wrapper produced."""
    q, cache, pt, t2r, qp, cl = args
    t2r = t2r.contiguous().view(-1)
    qp = qp.contiguous().view(-1)
    cl = cl.contiguous().view(-1)
    return t2r, qp, cl


class LaunchPathCPU(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.build_kw = []

        def build(*a, **kw):
            self.build_kw.append(kw)
            return _Launch()

        self.stack = [patch.object(K, "_run", lambda launch, *a: self.calls.append(a)),
                      patch.object(K, "_build_wmma_kernel", build),
                      patch.object(K, "_build_wmma_fp8_kernel", build),
                      patch.object(torch.cuda, "current_stream", lambda dev=None: "STREAM")]
        for s in self.stack:
            s.start()

    def tearDown(self):
        for s in reversed(self.stack):
            s.stop()

    def _check(self, args, kw, fmt):
        self.calls.clear()
        logits, vis = K.qsa_paged_mqa_logits_page4(*args, **kw)
        (a,) = self.calls
        off = 1 if fmt == "fp8" else 0
        t2r, qp, cl = a[3 + off], a[4 + off], a[5 + off]
        exp_t2r, _, exp_cl = historical_args(args, kw)
        for got, exp, src in ((t2r, exp_t2r, args[3]), (cl, exp_cl, args[5])):
            if src.stride(0) == 1:  # identity: the very same pointer/shape/stride
                self.assertEqual(meta(got), meta(exp))
            else:  # non-unit stride: the historical .contiguous().view(-1) path
                self.assertEqual(meta(got)[2:], meta(exp)[2:])
                self.assertEqual(got.data_ptr() == src.data_ptr(), src.is_contiguous())
            self.assertEqual(got.tolist(), src.tolist())
        self.assertEqual(qp.dtype, torch.int32)
        self.assertEqual(tuple(qp.stride()), (1,))
        src = args[4]
        if src.dtype == torch.int64:
            # ROUND-NATIVE64: zero-copy int32 view [2*rows] + pos64 specialization.
            self.assertEqual(self.build_kw[-1], {"pos64": True})
            self.assertEqual(tuple(qp.shape), (2 * src.shape[0],))
            self.assertEqual(qp.view(torch.int64).tolist(), src.tolist())
            if src.stride(0) == 1:
                self.assertEqual(qp.data_ptr(), src.data_ptr())
        else:
            self.assertEqual(self.build_kw[-1], {})   # unchanged int32 call
            self.assertEqual(qp.tolist(), src.tolist())
        vg, rs = a[7 + off], a[8 + off]
        rows = args[0].shape[0]
        for t in (vg, rs):
            self.assertEqual((tuple(t.shape), tuple(t.stride()), t.dtype), ((rows,), (1,), torch.int32))
        self.assertEqual(meta(vis), meta(vg))
        self.assertIs(logits, a[6 + off])
        self.assertEqual(a[-1], "STREAM")  # patched current_stream honoured
        return a

    def test_allocated_outputs_and_metadata(self):
        for fmt in ("bf16", "fp8"):
            for qp64 in (True, False):
                with self.subTest(fmt=fmt, qp64=qp64):
                    args, kw = fixture(fmt=fmt, qp64=qp64)
                    self._check(args, kw, fmt)

    def test_caller_outputs_1d_and_2d(self):
        for shape in ((16,), (16, 1)):
            args, kw = fixture()
            vg = torch.empty(shape, dtype=torch.int32)
            rs = torch.empty(shape, dtype=torch.int32)
            a = self._check(args, dict(kw, visible_groups=vg, row_starts=rs), "bf16")
            self.assertEqual(meta(a[7]), meta(vg.view(-1)))
            self.assertEqual(meta(a[8]), meta(rs.view(-1)))

    def test_strided_and_size1_metadata(self):
        args, kw = fixture()
        args[3] = torch.zeros(32, dtype=torch.int32)[::2]  # strided 1-D t2r
        args[4] = (torch.arange(32, dtype=torch.int64) + 7)[::2]  # strided int64 qp
        a = self._check(args, kw, "bf16")
        self.assertNotEqual(a[3].data_ptr(), args[3].data_ptr())  # copied, like before
        # size-1 request with a non-unit stride: historical view(-1) resets the stride
        args, kw = fixture(rows=4, requests=1)
        args[5] = torch.full((1, 3), 128, dtype=torch.int32)[:, 0]
        self.assertEqual(tuple(args[5].stride()), (3,))
        self._check(args, kw, "bf16")

    def test_guards_still_raise_every_call(self):
        args, kw = fixture()
        K.qsa_paged_mqa_logits_page4(*args, **kw)
        bad = torch.empty(16 * K.NUM_HEADS * K.HEAD_DIM + 8, dtype=torch.bfloat16)[1:]
        q_mis = bad[: 16 * K.NUM_HEADS * K.HEAD_DIM].view(16, K.NUM_HEADS, K.HEAD_DIM)
        with self.assertRaisesRegex(ValueError, "aligned"):
            K.qsa_paged_mqa_logits_page4(q_mis, *args[1:], **kw)
        with self.assertRaisesRegex(ValueError, "page_table"):
            K.qsa_paged_mqa_logits_page4(*args[:2], args[2].to(torch.float32), *args[3:], **kw)
        with self.assertRaisesRegex(ValueError, "visible_groups"):
            K.qsa_paged_mqa_logits_page4(*args, **dict(kw, visible_groups=torch.empty(15, dtype=torch.int32)))
        nonview = torch.empty((8, 2), dtype=torch.int32).t()  # [2, 8], stride (1, 2)
        with self.assertRaises(RuntimeError):  # historical: view(-1) refuses
            K.qsa_paged_mqa_logits_page4(*args, **dict(kw, visible_groups=nonview))
        # and the good call still works afterwards (no stale state)
        self._check(args, kw, "bf16")


class StreamFastPathCPU(unittest.TestCase):
    def test_raw_pointer_only_for_stock_current_stream(self):
        seen = []

        def raw(index):
            seen.append(index)
            return 0xABC0 + index

        cuda3 = torch.device("cuda", 3)
        with patch.object(K, "_RAW_CURRENT_STREAM", raw):
            self.assertIs(torch.cuda.current_stream, K._TORCH_CURRENT_STREAM)
            self.assertEqual(K._launch_stream(cuda3), 0xABC3)
            self.assertEqual(K._launch_stream_ptr(cuda3), 0xABC3)
            with patch.object(torch.cuda, "current_stream", lambda dev=None: "PATCHED"):
                self.assertEqual(K._launch_stream(cuda3), "PATCHED")
            with patch.object(torch.cuda, "current_stream", lambda dev=None: None):
                self.assertIsNone(K._launch_stream(torch.device("cpu")))
        self.assertEqual(seen, [3, 3])

    def test_bounds_key_uses_same_stream_identity(self):
        with patch.object(K, "_RAW_CURRENT_STREAM", lambda index: 0x77), \
                patch.object(K, "_I32_BOUNDS", {}), \
                patch.object(torch, "full", lambda *a, **k: ("full", a)):
            K._i32_bounds(torch.device("cuda", 0), 5)
            self.assertEqual(list(K._I32_BOUNDS), [(torch.device("cuda", 0), 0x77, 5)])


if __name__ == "__main__":
    unittest.main()
