# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""R2 CPU tests (no GPU, no device): token-dimension buckets, prewarm scoping /
persistent compile cache, page-8 plumbing.

* buckets: profile schema + matcher; the op's bucket path (real kernel wrapper geometry,
  ``_run`` replaced by a per-row deterministic stub "kernel") returns real rows
  bit-identical to the unpadded call, never touches caller memory outside the real rows,
  and launches with no allocation / no host sync per call once the class is decided;
* prewarm: scoped to a bound cache (page size, dtype) and position dtypes, idempotent,
  per-variant stats, persistent cache directory scoped around our compiles only,
  compile-only mode, thread pool;
* page 8: matcher / layout / capability probe before and after the page-8 kernel.

    QSA_GOLDEN_DIR=/path/golden_baseline python3 -B op_tests/test_qsa_index_scorer_r2_cpu.py -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_qsa_index_scorer_cpu as base  # noqa: E402

API, PR, K = base.API, base.PR, base.K
call_args, aligned, _tmpdir = base.call_args, base.aligned, base._tmpdir


def _gen():
    return base._gen()


def _pass(entry, ksha):
    """Mark a rule/bucket ``pass`` with a passing record per sample (matcher tests only)."""
    entry["golden"] = {"status": "pass", "kernel_sha256": ksha, "samples_n": len(entry["samples"]),
                       "checked": [dict(s, **{"pass": True}) for s in entry["samples"]]}


def _profile_copy(td, *, rules=(), buckets=(), pending=(), name="qsa_r2test_v1"):
    """Committed profile with the given rule / bucket ids marked pass (and the ``pending``
    rule / bucket ids reset to pending, e.g. to step over a committed ``pass`` bucket),
    written to ``td``."""
    with open(base.PROFILE_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    ksha = data["kernel"]["sha256"]
    for e in [*data["rules"], *data["buckets"]]:
        if e["id"] in pending:
            e["golden"] = {"status": "pending"}
    for r in data["rules"]:
        if r["id"] in rules:
            _pass(r, ksha)
    for b in data["buckets"]:
        if b["id"] in buckets:
            _pass(b, ksha)
    data["profile"] = name
    path = os.path.join(td, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    PR.validate_profile(data)
    return path


# ---------------------------------------------------------------------------
# Deterministic per-row stub kernel: row r depends on its own q / t2r / position only
# ---------------------------------------------------------------------------
class RowKernel:
    """Replacement for ``K._run``: writes exactly rows [0, rows) of logits / visible /
    row_starts as a pure function of the row's own inputs (like the real kernels)."""

    def __init__(self):
        self.calls = []
        self.mode = None

    def __call__(self, launch, *a):
        mode, self.mode = self.mode, None          # stub work is not "op work"
        try:
            rows, columns = int(a[-6]), int(a[-5])
            logits, vg, rs = a[-11], a[-10], a[-9]
            cl, qp, t2r = a[-12], a[-13], a[-14]
            q = a[0]
            self.calls.append({"kind": launch.kind, "rows": rows, "gx": a[-8], "gy": a[-7],
                               "logits_rows": int(logits.shape[0]), "q_rows": int(q.shape[0])})
            pos = qp.view(-1)[0:2 * rows:2] if qp.numel() == 2 * rows else qp.view(-1)[:rows]
            base_v = q[:rows].to(torch.float64).sum(dim=1)
            val = (base_v + 7.0 * pos.to(torch.float64) + 1000.0 * t2r[:rows].to(torch.float64))
            cols = torch.arange(columns, dtype=torch.float64) * 0.5
            logits.view(-1, columns)[:rows] = (val[:, None] + cols[None, :]).to(torch.float32)
            vg.view(-1)[:rows] = (pos + 1).to(torch.int32)
            rs.view(-1)[:rows] = 0
        finally:
            self.mode = mode


class CountOps(TorchDispatchMode):
    """Counts factory (allocating) ops and host syncs dispatched by the op itself."""

    FACTORY = ("empty", "zeros", "ones", "full", "arange", "randn", "rand", "new_empty",
               "new_zeros", "new_full", "empty_like", "zeros_like", "clone", "_to_copy",
               "empty_strided")
    SYNC = ("_local_scalar_dense", "item", "is_nonzero")

    def __init__(self, stub):
        super().__init__()
        self.stub, self.alloc, self.sync, self.ops = stub, [], [], []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if self.stub.mode is self:
            self.ops.append(name)
            if name in self.FACTORY:
                self.alloc.append(name)
            if name in self.SYNC:
                self.sync.append(name)
        return func(*args, **(kwargs or {}))


class _R2Env(base._Stubs):
    def setUp(self):
        super().setUp()
        self.stub = RowKernel()
        self._r = [mock.patch.object(K, "_run", self.stub),
                   mock.patch.dict(API._STAGES, clear=True)]
        for p in self._r:
            p.start()
        self._td = _tmpdir()
        self.td = self._td.__enter__()

    def tearDown(self):
        self._td.__exit__(None, None, None)
        for p in reversed(self._r):
            p.stop()
        super().tearDown()


# ---------------------------------------------------------------------------
# Buckets: profile
# ---------------------------------------------------------------------------
class BucketProfile(unittest.TestCase):
    def setUp(self):
        with open(base.PROFILE_JSON, encoding="utf-8") as fh:
            self.data = json.load(fh)

    def test_committed_buckets(self):
        ids = {b["id"] for b in self.data["buckets"]}
        want = set()
        for sfx in ("", "_p8", "_p16"):
            want |= {f"verify3_g3_bf16{sfx}", f"verify3_pad4_bf16{sfx}", f"tail_round4_bf16{sfx}",
                     f"verify3_g3_fp8{sfx}"}     # R2B-FP8: kernel-native fp8 verify3 only
        self.assertEqual(ids, want)
        # R2-MERGE: the kernel-native 3-row bucket precedes the staging bucket per page
        order = [b["id"] for b in self.data["buckets"]]
        for sfx in ("", "_p8", "_p16"):
            self.assertLess(order.index(f"verify3_g3_bf16{sfx}"),
                            order.index(f"verify3_pad4_bf16{sfx}"))
            g3 = self.data["buckets"][order.index(f"verify3_g3_bf16{sfx}")]
            self.assertEqual(g3["params"]["group_rows"], 3)
            self.assertEqual(g3["group"], {"real": 3, "padded": 4, "kernel_group_rows": 3})
            pad = self.data["buckets"][order.index(f"verify3_pad4_bf16{sfx}")]
            self.assertNotIn("group_rows", pad["params"])
            self.assertNotEqual(g3["sig"], pad["sig"])
        # GPU restamp on kernel 58051973: the bf16 and fp8 g3 buckets pass; the staging /
        # row-rounding buckets are routing-only 'fail' (never selected)
        want_status = {}
        for sfx in ("", "_p8", "_p16"):
            want_status.update({f"verify3_g3_bf16{sfx}": "pass", f"verify3_pad4_bf16{sfx}": "fail",
                                f"tail_round4_bf16{sfx}": "fail", f"verify3_g3_fp8{sfx}": "pass"})
        for b in self.data["buckets"]:
            PR.validate_bucket(b, self.data["kernel"]["sha256"])
            self.assertEqual(b["golden"]["status"], want_status[b["id"]], b["id"])
            self.assertEqual(PR.ProfileBucket(b).verified, want_status[b["id"]] == "pass", b["id"])
            self.assertEqual(b["target_regime"], "decode")
            self.assertTrue(b["vid"].endswith(".decg.v1"))
            self.assertGreater(len(b["samples"]), 10, b["id"])

    def test_targets(self):
        g = next(b for b in self.data["buckets"] if b["id"] == "verify3_pad4_bf16")
        self.assertEqual(PR.bucket_target(g, 24, 8), (32, (24, 24)))
        self.assertIsNone(PR.bucket_target(g, 25, 8))
        r = next(b for b in self.data["buckets"] if b["id"] == "tail_round4_bf16")
        self.assertEqual(PR.bucket_target(r, 87, 16), (88, (85, 88)))
        self.assertIsNone(PR.bucket_target(r, 31, 1))       # 32 rows / 1 request = prefill
        self.assertFalse(PR.ProfileBucket(r).admits(88, 16, 65536, 16384))   # already decode
        for (rows, reqs, cols, w, grp) in _gen().r2_matrix_shapes(4):
            if grp == "verify3":
                self.assertTrue(PR.ProfileBucket(g).admits(rows, reqs, cols, w), (rows, reqs))

    def test_validation_rejects(self):
        b = json.loads(json.dumps(next(x for x in self.data["buckets"]
                                       if x["id"] == "verify3_pad4_bf16")))
        for mut in (lambda d: d.update(kind="rows"), lambda d: d["group"].update(padded=8),
                    lambda d: d.update(target_regime="tail"),
                    lambda d: d.update(vid="qsa.page4.r16.v1"), lambda d: d.update(extra=1),
                    lambda d: d["samples"].append(dict(d["samples"][0], rows=5))):
            bad = json.loads(json.dumps(b))
            mut(bad)
            with self.assertRaises(PR.ProfileError):
                PR.validate_bucket(bad)

    def test_match_bucket_needs_verified_bucket_and_target(self):
        kw = dict(arch="gfx1250", cu=256, cache_dtype="bf16", rows=24, requests=8,
                  columns=1024, table_width=256)
        prof = PR.profile_from_data(self.data)
        # committed: the kernel-native g3 bucket and its decode target both pass
        bm = PR.match_bucket(prof, **kw)
        self.assertEqual((bm.bucket.id, bm.target_rows, bm.call_rows), ("verify3_g3_bf16", 32,
                                                                        (24, 24)))
        self.assertEqual(bm.target.row.id, "decode_bf16")
        # a kernel without native 3-row groups -> the staging bucket, which is 'fail':
        # never selected, not even with allow_unverified
        self.assertIsNone(PR.match_bucket(prof, native_group_rows=(4,), **kw))
        self.assertIsNone(PR.match_bucket(prof, allow_unverified=True, native_group_rows=(4,),
                                          **kw))
        d = json.loads(json.dumps(self.data))
        next(b for b in d["buckets"] if b["id"] == "verify3_pad4_bf16")["golden"] = \
            {"status": "pending"}
        prof = PR.profile_from_data(d)
        self.assertIsNone(PR.match_bucket(prof, native_group_rows=(4,), **kw))   # pending
        bm = PR.match_bucket(prof, allow_unverified=True, native_group_rows=(4,), **kw)
        self.assertEqual((bm.bucket.id, bm.target_rows), ("verify3_pad4_bf16", 32))
        _pass(next(b for b in d["buckets"] if b["id"] == "verify3_pad4_bf16"),
              d["kernel"]["sha256"])
        prof = PR.profile_from_data(d)
        bm = PR.match_bucket(prof, native_group_rows=(4,), **kw)
        self.assertEqual((bm.bucket.id, bm.target.row.id), ("verify3_pad4_bf16", "decode_bf16"))
        # a pending bucket is never selected even with a passing target
        next(b for b in d["buckets"] if b["id"] == "verify3_g3_bf16")["golden"] = \
            {"status": "pending"}
        prof = PR.profile_from_data(d)
        self.assertEqual(PR.match_bucket(prof, **kw).bucket.id, "verify3_pad4_bf16")
        # page 8: g3 bucket pass + page-8 decode rule pass -> bucket; decode rule pending
        # -> no bucket (the target must be verified too)
        kw8 = dict(kw, table_width=128, page_size=8)
        prof = PR.profile_from_data(d)
        self.assertEqual(PR.match_bucket(prof, **kw8).bucket.id, "verify3_g3_bf16_p8")
        next(r for r in d["rules"] if r["id"] == "decode_bf16_p8")["golden"] = \
            {"status": "pending"}
        prof = PR.profile_from_data(d)
        self.assertIsNone(PR.match_bucket(prof, **kw8))

    def test_launch_knobs_schema(self):
        """R2-MERGE: wg_waves is pinnable per row/rule/bucket (new sig, no inherited
        evidence); group_rows is bucket-only; row_mode never pinnable."""
        rule = next(r for r in self.data["rules"] if r["id"] == "draft_bf16")
        pinned = json.loads(json.dumps(rule))
        pinned["params"]["wg_waves"] = 2
        pinned["golden"] = {"status": "pending"}
        pinned["sig"] = PR.rule_signature(pinned)
        self.assertNotEqual(pinned["sig"], rule["sig"])
        PR.validate_rule(pinned)
        self.assertEqual(PR.split_knobs(pinned["vid"], pinned["params"]),
                         (dict(rule["params"]), {"wg_waves": 2}))
        for k, v in (("wg_waves", 3), ("wg_waves", 4), ("group_rows", 3), ("row_mode", "pad")):
            bad = json.loads(json.dumps(rule))
            bad["params"][k] = v
            bad["sig"] = PR.rule_signature(bad)
            with self.assertRaises(PR.ProfileError, msg=(k, v)):
                PR.validate_rule(bad)
        g3 = next(b for b in self.data["buckets"] if b["id"] == "verify3_g3_bf16")
        for mut in (lambda d: d["params"].pop("group_rows"),
                    lambda d: d["group"].pop("kernel_group_rows"),
                    lambda d: d["params"].update(row_mode="pad")):
            bad = json.loads(json.dumps(g3))
            mut(bad)
            bad["sig"] = PR.bucket_signature(bad)
            with self.assertRaises(PR.ProfileError):
                PR.validate_bucket(bad)

    def test_gen_profile_verify_green(self):
        self.assertEqual(_gen().verify(base.PROFILE_JSON), [])


# ---------------------------------------------------------------------------
# Buckets: op path
# ---------------------------------------------------------------------------
def _verify3_args(C, columns=1024, width=256, qp64=True, seed=0):
    rows = 3 * C
    args = call_args(rows=rows, requests=C, width=width, columns=columns, qp64=qp64)
    g = torch.Generator().manual_seed(seed)
    args[0].copy_(torch.randn(args[0].shape, generator=g).to(torch.bfloat16))
    args[3].copy_(torch.arange(rows, dtype=torch.int32) // 3)        # 3 rows per request
    args[4].copy_(torch.randint(0, 4 * width, (rows,), generator=g).to(args[4].dtype))
    return args


class BucketOp(_R2Env):
    def _prof(self):
        """Staging / row buckets marked pass; the committed kernel-native g3 bucket
        (pass) reset to pending so the staging path is the one exercised."""
        return _profile_copy(self.td, buckets=("verify3_pad4_bf16", "tail_round4_bf16"),
                             pending=("verify3_g3_bf16",))

    def _unpadded(self, args, prof):
        d = API.decide(*args, profile=prof, buckets=False)
        self.assertTrue(d.ok, d)
        self.assertFalse(str(d.cell).startswith("bucket:"))
        return API.launch(d, *args)

    def test_on_by_default(self):
        """ROUND-R2E: buckets are ON by default; ``buckets=False`` or
        ``$AITER_QSA_INDEX_SCORER_BUCKETS=0`` disables them (an explicit argument wins)."""
        self.assertTrue(API.BUCKETS_DEFAULT)
        prof = self._prof()
        args = _verify3_args(8)
        with mock.patch.dict(os.environ):
            os.environ.pop(API.BUCKETS_ENV, None)
            d = API.decide(*args, profile=prof)
        self.assertTrue(d.ok and str(d.cell).startswith("bucket:verify3_pad4_bf16>"), d)
        for v in ("1", "on", "true"):
            with mock.patch.dict(os.environ, {API.BUCKETS_ENV: v}):
                d = API.decide(*args, profile=prof)
            self.assertTrue(str(d.cell).startswith("bucket:verify3_pad4_bf16>"), (v, d))
        d = API.decide(*args, profile=prof, buckets=False)
        self.assertTrue(d.ok and not str(d.cell).startswith("bucket:"), d)
        for v in ("0", "off", "false"):
            with mock.patch.dict(os.environ, {API.BUCKETS_ENV: v}):
                d = API.decide(*args, profile=prof)
                self.assertTrue(d.ok and not str(d.cell).startswith("bucket:"), (v, d))
                d = API.decide(*args, profile=prof, buckets=True)       # explicit wins
                self.assertTrue(str(d.cell).startswith("bucket:verify3_pad4_bf16>"), (v, d))
        # committed profile: the verified kernel-native g3 bucket is selected by default
        d = API.decide(*args, profile=self.profile)
        self.assertTrue(d.ok and str(d.cell).startswith("bucket:verify3_g3_bf16>"), d)
        # ... a 'fail' bucket never is (kernel without native 3-row groups -> staging
        # bucket verify3_pad4_bf16, committed 'fail' -> the call is matched unpadded)
        with mock.patch.object(API, "kernel_real_rows_per_group", return_value=()):
            d = API.decide(*args, profile=self.profile, buckets=True)
        self.assertTrue(d.ok and not str(d.cell).startswith("bucket:"), d)

    def test_group_bit_identical_and_no_overrun(self):
        prof = self._prof()
        for C in (1, 5, 8, 17):
            for qp64 in (True, False):
                args = _verify3_args(C, qp64=qp64, seed=C)
                rows, cols = 3 * C, args[8]
                ref_l, ref_v = (t.clone() for t in self._unpadded(args, prof))
                d = API.decide(*args, profile=prof, buckets=True)
                self.assertTrue(d.ok and d.cell.startswith("bucket:verify3_pad4_bf16>"), d)
                self.assertEqual(d._plan.rows, 4 * C)
                # caller buffers inside guarded allocations
                big_l = torch.full((rows + 2, cols), 12345.0)
                big_v = torch.full((rows + 2,), -77, dtype=torch.int32)
                big_r = torch.full((rows + 2,), -55, dtype=torch.int32)
                lg, vg, rs = big_l[1:rows + 1], big_v[1:rows + 1], big_r[1:rows + 1]
                a = list(args)
                a[9] = rs
                self.stub.calls.clear()
                out_l, out_v = API.launch(d, *a, logits=lg, visible_groups=vg)
                self.assertEqual(out_l.data_ptr(), lg.data_ptr())
                self.assertEqual(out_v.data_ptr(), vg.data_ptr())
                self.assertTrue(torch.equal(out_l, ref_l), (C, qp64))       # bit-identical
                self.assertTrue(torch.equal(out_v, ref_v), (C, qp64))
                self.assertTrue(bool((rs == 0).all()))
                for big, fill in ((big_l, 12345.0), (big_v, -77), (big_r, -55)):
                    self.assertTrue(bool((big[0] == fill).all()) and bool((big[-1] == fill).all()))
                call = self.stub.calls[-1]
                self.assertEqual((call["kind"], call["rows"], call["q_rows"]),
                                 ("decg", 4 * C, 4 * C))
                self.assertEqual((call["gx"], call["gy"]), K.decode_group_grid(4 * C, cols, 32))
                # 1.0 op path (buckets on by default) gives the same bits, allocating outputs
                l2, v2 = API.qsa_paged_mqa_logits(*args, profile=prof)
                self.assertTrue(torch.equal(l2, ref_l) and torch.equal(v2, ref_v))
                self.assertEqual(tuple(l2.shape), (rows, cols))

    def test_group_pad_slot_keeps_group_uniform(self):
        prof = self._prof()
        args = _verify3_args(4)
        d = API.decide(*args, profile=prof, buckets=True)
        API.launch(d, *args)
        st = next(iter(API._STAGES.values()))
        t2r = st.t2r.view(-1, 4)
        self.assertTrue(bool((t2r == t2r[:, :1]).all()))    # one request per 4-row group
        self.assertEqual(st.gather.tolist()[:8], [0, 1, 2, 2, 3, 4, 5, 5])
        self.assertEqual(st.scatter.tolist()[:6], [0, 1, 2, 4, 5, 6])

    def test_group_graph_safety(self):
        """After decide: launches allocate nothing, never sync, staging planes reused."""
        prof = self._prof()
        args = _verify3_args(8)
        rows, cols = 24, args[8]
        d = API.decide(*args, profile=prof, buckets=True)
        self.assertEqual(len(API._STAGES), 1)                 # allocated at decide time
        st = next(iter(API._STAGES.values()))
        ptrs = [t.data_ptr() for t in (st.q, st.t2r, st.pos, st.logits, st.vis, st.rs)]
        lg, vg = torch.empty(rows, cols), torch.empty(rows, dtype=torch.int32)
        a = list(args)
        a[9] = torch.empty(rows, dtype=torch.int32)
        API.launch(d, *a, logits=lg, visible_groups=vg)      # first launch (cache scope)
        for _ in range(3):
            with CountOps(self.stub) as m:
                self.stub.mode = m
                API.launch(d, *a, logits=lg, visible_groups=vg)
                self.stub.mode = None
            self.assertEqual(m.alloc, [], m.ops)
            self.assertEqual(m.sync, [], m.ops)
            self.assertIn("index_select", m.ops)
        self.assertEqual(len(API._STAGES), 1)
        self.assertEqual(ptrs, [t.data_ptr() for t in (st.q, st.t2r, st.pos, st.logits, st.vis,
                                                       st.rs)])
        # a second decide of the same class reuses the staging planes
        API.decide(*args, profile=prof, buckets=True)
        self.assertEqual(len(API._STAGES), 1)

    def test_group_guards(self):
        prof = self._prof()
        d = API.decide(*_verify3_args(8), profile=prof, buckets=True)
        with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
            API.launch(d, *_verify3_args(8)[:0], *call_args(rows=32, requests=8, width=256,
                                                           columns=1024))
        self.assertEqual(cm.exception.reason, PR.Reason.ROUTE_MISMATCH)

    def test_rows_bucket_kernel_row_bound(self):
        prof = self._prof()
        for rows in (85, 86, 87):
            args = call_args(rows=rows, requests=16, width=256, columns=1024)
            args[0].copy_(torch.randn(args[0].shape).to(torch.bfloat16))
            ref_l, ref_v = (t.clone() for t in self._unpadded(args, prof))
            d = API.decide(*args, profile=prof, buckets=True)
            self.assertTrue(d.ok and d.cell.startswith("bucket:tail_round4_bf16>"), d)
            self.assertEqual((d._plan.rows, d._plan.call_lo, d._plan.call_hi), (88, 85, 88))
            big = torch.full((rows + 1, 1024), 9.0)
            lg = big[:rows]
            self.stub.calls.clear()
            out_l, out_v = API.launch(d, *args, logits=lg)
            call = self.stub.calls[-1]
            self.assertEqual((call["kind"], call["rows"], call["logits_rows"]), ("decg", rows, rows))
            self.assertEqual((call["gx"], call["gy"]), K.decode_group_grid(rows, 1024, 32))
            self.assertTrue(torch.equal(out_l, ref_l) and torch.equal(out_v, ref_v))
            self.assertTrue(bool((big[-1] == 9.0).all()))
            self.assertEqual(API._STAGES, {})                  # no staging for row buckets
        # one decision serves every real row count of its bucket
        d = API.decide(*call_args(rows=87, requests=16, width=256, columns=1024), profile=prof,
                       buckets=True)
        for rows in (85, 86, 87, 88):
            out_l, _ = API.launch(d, *call_args(rows=rows, requests=16, width=256, columns=1024))
            self.assertEqual(tuple(out_l.shape), (rows, 1024))
        with self.assertRaises(API.QSAIndexScorerUnsupported):
            API.launch(d, *call_args(rows=84, requests=16, width=256, columns=1024))

    def test_kernel_mirror_row_bound_and_uniform_groups(self):
        """Kernel's own trace-free mirrors: the decode-group kernel with a real row count
        that is not a multiple of 4 stores only rows < rows (rows bucket), and a padded
        verify3 group (pad slot = last real row) runs the single-pass uniform plan."""
        for rows in (85, 86, 87, 5, 6, 7):
            t2r = [r // 6 for r in range(rows)]
            got = {r for r, _c in K.decode_group_store_plan(rows, 96, 32, t2r)}
            self.assertEqual(got, set(range(rows)), rows)
        self.assertEqual([p[0] for p in K.decode_group_passes([3, 3, 3, 3], [True] * 4)],
                         [True, False, False, False])
        # unpadded verify3 rows 0..3 straddle two requests -> 4 single-row passes
        self.assertEqual([p[0] for p in K.decode_group_passes([0, 0, 0, 1], [True] * 4)],
                         [True, True, True, True])

    def test_real_rows_kernel_capability_routes_to_adapter(self):
        # R2-MERGE kernel: the decode-group builder's group_rows keyword is the capability
        self.assertEqual(API.kernel_real_rows_per_group(K), (3, 4))
        with mock.patch.object(K, "DECG_REAL_ROWS_PER_GROUP", (4,), create=True):
            self.assertEqual(API.kernel_real_rows_per_group(K), (4,))
        prof = _profile_copy(self.td, buckets=("verify3_g3_bf16", "verify3_pad4_bf16"))
        args = _verify3_args(4)
        d = API.decide(*args, profile=prof, buckets=True)
        self.assertTrue(d.cell.startswith("bucket:verify3_g3_bf16>"), d)
        self.assertTrue(d._plan.bucket.real_rows_kernel)
        self.assertEqual(API._STAGES, {})                  # no staging needed
        with mock.patch.object(API, "_group_kernel_launch",
                               return_value=("L", "V")) as gk:
            self.assertEqual(API.launch(d, *args), ("L", "V"))
        gk.assert_called_once()
        # kernel without native 3-row groups: the staging bucket of the class applies
        with mock.patch.object(API, "kernel_real_rows_per_group", return_value=()):
            d = API.decide(*args, profile=prof, buckets=True)
        self.assertTrue(d.cell.startswith("bucket:verify3_pad4_bf16>"), d)
        self.assertFalse(d._plan.bucket.real_rows_kernel)

    def test_g3_bucket_zero_copy_bit_identical(self):
        """Kernel-native verify3 bucket: ONE group_rows=3 launch on the caller's own
        tensors (the wrapper's QSA_DECG_ROWS3=1 launcher), real rows only, no staging,
        no allocation / sync once decided, bit-identical to the unpadded call."""
        prof = _profile_copy(self.td, buckets=("verify3_g3_bf16", "verify3_pad4_bf16"))
        for C in (1, 5, 8, 17):
            for qp64 in (True, False):
                args = _verify3_args(C, qp64=qp64, seed=C)
                rows, cols = 3 * C, args[8]
                ref_l, ref_v = (t.clone() for t in self._unpadded(args, prof))
                d = API.decide(*args, profile=prof, buckets=True)
                self.assertTrue(d.ok and d.cell.startswith("bucket:verify3_g3_bf16>"), d)
                p = d._plan
                self.assertEqual((p.rows, p.call_lo, p.call_hi), (rows, rows, rows))
                self.assertEqual(p.build_kw, {"group_rows": 3, "row_mode": "dedup",
                                              "wg_waves": 4})
                big_l = torch.full((rows + 2, cols), 12345.0)
                big_v = torch.full((rows + 2,), -77, dtype=torch.int32)
                big_r = torch.full((rows + 2,), -55, dtype=torch.int32)
                lg, vg, rs = big_l[1:rows + 1], big_v[1:rows + 1], big_r[1:rows + 1]
                a = list(args)
                a[9] = rs
                self.stub.calls.clear()
                out_l, out_v = API.launch(d, *a, logits=lg, visible_groups=vg)
                self.assertEqual(len(self.stub.calls), 1)
                call = self.stub.calls[-1]
                self.assertEqual((call["kind"], call["rows"], call["q_rows"], call["logits_rows"]),
                                 ("decg", rows, rows, rows))
                self.assertEqual(call["gy"], C)                       # one group per request
                self.assertEqual((call["gx"], call["gy"]),
                                 K.decode_group_grid(rows, cols, 32, 3, 4))
                self.assertEqual(out_l.data_ptr(), lg.data_ptr())
                self.assertTrue(torch.equal(out_l, ref_l) and torch.equal(out_v, ref_v))
                for big, fill in ((big_l, 12345.0), (big_v, -77), (big_r, -55)):
                    self.assertTrue(bool((big[0] == fill).all()) and bool((big[-1] == fill).all()))
                self.assertEqual(API._STAGES, {})
                # same launcher object the kernel wrapper uses under QSA_DECG_ROWS3=1
                # (wrapper: _build_decode_group_kernel(is_fp8, div, wc, **qp_kw, **pg_kw, **dg_kw))
                kw = dict(group_rows=3, row_mode="dedup", wg_waves=4)
                if qp64:
                    kw["pos64"] = True
                self.assertEqual(p.exes[1 if qp64 else 0][0].build_args,
                                 ((False, float(API._DEFAULT_DIVISOR), 32), kw))
                self.assertEqual(K._check_decg_knobs(3, K._DECG_MODE, K._DECG_WG_WAVES),
                                 (3, "dedup", 4))
                with CountOps(self.stub) as m:
                    self.stub.mode = m
                    API.launch(d, *a, logits=lg, visible_groups=vg)
                    self.stub.mode = None
                self.assertEqual((m.alloc, m.sync), ([], []), m.ops)
                self.assertNotIn("index_select", m.ops)


# ---------------------------------------------------------------------------
# Prewarm (R2)
# ---------------------------------------------------------------------------
class PrewarmR2(base._Stubs):
    def setUp(self):
        super().setUp()
        self._w = [mock.patch.dict(API._PREWARMED, clear=True),
                   mock.patch.dict(API._PREWARM_STATS, clear=True),
                   mock.patch.object(torch.cuda, "synchronize",
                                     side_effect=AssertionError("cuda sync on cpu"))]
        for p in self._w:
            p.start()
        self._td = _tmpdir()
        self.td = self._td.__enter__()

    def tearDown(self):
        self._td.__exit__(None, None, None)
        for p in reversed(self._w):
            p.stop()
        super().tearDown()

    def test_scoped_to_bound_cache_and_pos_dtype(self):
        cache = aligned((8, 4, 1, 128), torch.bfloat16)
        work = API.prewarm_plan(self.profile, compressed_k_cache=cache,
                                pos_dtypes=(torch.int64,), kernel=K)
        # page-4 bf16 pass variants (draft / decode-group / r16 / lean ktile / persist) + the
        # verified kernel-native g3 bucket's own group_rows=3 decode-group specialization
        self.assertEqual(sorted((w[0], dict(w[2]).get("group_rows", 4)) for w in work),
                         [("qsa.page4.decg.v1/bf16/i64", 3), ("qsa.page4.decg.v1/bf16/i64", 4),
                          ("qsa.page4.draft.v1/bf16/i64", 4),
                          ("qsa.page4.r16.v1/bf16/i64", 4),
                          ("qsa.page4.r32_kt.v1/bf16/i64", 4),
                          ("qsa.page4.r32_persist.v1/bf16/i64", 4)])
        got = API.prewarm(self.profile, compressed_k_cache=cache, pos_dtypes=(torch.int64,),
                          device="cpu", cache_dir=self.td)
        self.assertEqual(got, sorted({w[0] for w in work}))   # tags: i64 only, page 4 only
        self.assertEqual(len(self.runs), 6)                   # one launch per specialization
        self.assertEqual(sorted((sorted(kw.items()) for _, _, kw in self.kinds)),
                         sorted([[("pos64", True)]] * 4
                                + [sorted({"pos64": True, "group_rows": 3, "row_mode": "dedup",
                                           "wg_waves": 4}.items())]
                                + [sorted({"pos64": True, "kt_rows": 32, "col_tiles": 2,
                                           "lean": True}.items())]))
        # a page-16 cache: only page-16 variants
        cache16 = aligned((8, 16, 1, 128), torch.bfloat16)
        got16 = API.prewarm(self.profile, compressed_k_cache=cache16, pos_dtypes=(torch.int64,),
                            device="cpu", cache_dir=self.td)
        self.assertEqual(got16, ["qsa.page16.decg.v1/bf16/i64", "qsa.page16.draft.v1/bf16/i64",
                                 "qsa.page16.r16.v1/bf16/i64", "qsa.page16.r32_kt.v1/bf16/i64",
                                 "qsa.page16.r32_persist.v1/bf16/i64"])
        # fp8 cache: only the fp8 pass variants
        c8 = aligned((8, 4, 1, 128), torch.float8_e4m3fn)
        got8 = API.prewarm(self.profile, compressed_k_cache=c8,
                           compressed_k_scale=torch.ones(8, 4, 1), pos_dtypes=(torch.int64,),
                           device="cpu", cache_dir=self.td)
        self.assertEqual(got8, ["qsa.page4.decg.v1/fp8/i64", "qsa.page4.draft.v1/fp8/i64",
                                "qsa.page4.r16_fp8.v1/fp8/i64", "qsa.page4.r32_fp8.v1/fp8/i64",
                                "qsa.page4.r32_kt_fp8.v1/fp8/i64"])

    def test_idempotent_and_stats(self):
        first = API.prewarm(self.profile, pos_dtypes=(torch.int64,), device="cpu",
                            cache_dir=self.td)
        stats = API.prewarm_stats()
        self.assertEqual(sorted(stats), first)
        self.assertTrue(all(not v["cached"] and v["seconds"] >= 0 for v in stats.values()))
        n = len(self.runs)
        self.assertEqual(API.prewarm(self.profile, pos_dtypes=(torch.int64,), device="cpu",
                                     cache_dir=self.td), first)
        self.assertEqual(len(self.runs), n)
        self.assertTrue(all(v["cached"] for v in API.prewarm_stats().values()))

    def test_persistent_cache_dir_scoped_to_our_compiles(self):
        seen = []
        orig = K._run

        def run(launch, *a):
            seen.append(os.environ.get(API.FLYDSL_CACHE_ENV))
            return orig(launch, *a)

        with mock.patch.object(K, "_run", run), \
                mock.patch.dict(os.environ, {API.FLYDSL_CACHE_ENV: "/user/set"}):
            # explicit cache_dir wins inside prewarm, the user's value is restored after
            API.prewarm(self.profile, pos_dtypes=(torch.int64,), device="cpu", cache_dir=self.td)
            self.assertEqual(set(seen), {self.td})
            self.assertEqual(os.environ[API.FLYDSL_CACHE_ENV], "/user/set")
        self.assertTrue(os.path.isdir(self.td))

    def test_compile_cache_dir_precedence(self):
        with mock.patch.dict(os.environ, {"USERPROFILE": self.td, "HOME": self.td}):
            os.environ.pop(API.CACHE_DIR_ENV, None)
            os.environ.pop(API.FLYDSL_CACHE_ENV, None)
            d = API.compile_cache_dir("ab" * 32)
            self.assertEqual(d, os.path.join(self.td, ".cache", "aiter", "qsa_index_scorer",
                                             "ab" * 8))
            os.environ[API.FLYDSL_CACHE_ENV] = "/flydsl"
            self.assertEqual(API.compile_cache_dir("ab" * 32), "/flydsl")
            os.environ[API.CACHE_DIR_ENV] = self.td
            self.assertEqual(API.compile_cache_dir("ab" * 32), os.path.abspath(self.td))
            for off in ("off", "0", "none"):
                os.environ[API.CACHE_DIR_ENV] = off
                self.assertIsNone(API.compile_cache_dir("ab" * 32))

    def test_unwritable_cache_dir_is_ignored(self):
        with mock.patch.object(os, "makedirs", side_effect=PermissionError("ro")):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(API.FLYDSL_CACHE_ENV, None)
                with API._scoped_cache_dir("/ro/dir"):
                    self.assertIsNone(os.environ.get(API.FLYDSL_CACHE_ENV))

    def test_jobs_use_worker_processes_then_in_process_pass(self):
        """FlyDSL tracing is not thread-safe (measured: 'Only one SharedAllocator is allowed
        per kernel' under a thread pool): jobs > 1 compiles in no-device worker processes
        into the shared cache, then the normal in-process pass (disk hits) runs."""
        calls = []
        with mock.patch.object(API, "_compile_in_workers",
                               lambda path, work, cdir, jobs: calls.append((work, cdir, jobs))):
            a = API.prewarm(self.profile, device="cpu", cache_dir=self.td, jobs=4,
                            cache_dtypes=("bf16", "fp8"))
        self.assertEqual(len(calls), 1)
        work = calls[0][0]
        self.assertEqual(sorted({w[0] for w in work}), a)
        self.assertEqual(len(work), len(API.prewarm_plan(self.profile, kernel=K,
                                                         cache_dtypes=("bf16", "fp8"))))
        self.assertEqual(calls[0][1:], (self.td, 4))
        self.assertEqual(len(self.runs), len(work))          # in-process pass still launches
        API._PREWARMED.clear()
        b = API.prewarm(self.profile, device="cpu", cache_dir=self.td,
                        cache_dtypes=("bf16", "fp8"))
        self.assertEqual(a, b)

    def test_workers_one_variant_per_process_longest_first(self):
        prof = _profile_copy(self.td, rules=("prefill_bf16_kt", "prefill_bf16_persist"))
        work = API.prewarm_plan(prof, pos_dtypes=(torch.int64,), kernel=K)
        bins = []

        class _P:
            def __init__(self, cmd, **kw):
                bins.append(json.loads(cmd[-1])["work"])

            def wait(self):
                return 0

        import subprocess
        with mock.patch.object(subprocess, "Popen", _P):
            self.assertEqual(API._compile_in_workers("p.json", work, self.td, 2),
                             [0] * len(work))
        # one variant per fresh process (cache-key history dependence), longest first
        self.assertEqual(len(bins), len(work))
        self.assertTrue(all(len(b) == 1 for b in bins))
        kinds = [b[0][0].split("/")[0].split(".")[2] for b in bins]
        n_persist = kinds.count("r32_persist")
        self.assertGreaterEqual(n_persist, 1)
        self.assertEqual(kinds[:n_persist], ["r32_persist"] * n_persist)
        self.assertEqual(kinds[n_persist], "r32_kt")      # ROUND-R2E lean ktile next
        cost = [API._COMPILE_COST[b[0][0].split("/")[0].split(".")[2]] for b in bins]
        self.assertEqual(cost, sorted(cost, reverse=True))

    def test_compile_only_needs_no_device(self):
        seen = []
        with mock.patch.object(API, "_compile_only_run",
                               lambda K_, plan, exe, args: seen.append((plan.vid, args))), \
                mock.patch.object(API, "_device_info", side_effect=AssertionError("device")), \
                mock.patch.object(API, "_static_decision", side_effect=AssertionError("dev")):
            got = API.prewarm(self.profile, pos_dtypes=(torch.int64,), compile_only=True,
                              cache_dir=self.td)
        work = API.prewarm_plan(self.profile, pos_dtypes=(torch.int64,), kernel=K)
        # pages 4/8/16 x (draft, decode-group, decode-group g3 bucket, r16, ktile, persist)
        self.assertEqual(len(work), 18)
        self.assertEqual(got, sorted({w[0] for w in work}))
        self.assertEqual(self.runs, [])                     # nothing launched
        self.assertEqual(len(seen), len(work))
        for _vid, args in seen:
            self.assertTrue(all(t.device.type == "cpu" for t in args
                                if isinstance(t, torch.Tensor)))

    def test_page8_gated_on_kernel_capability(self):
        prof = _profile_copy(self.td, rules=tuple(r + "_p8" for r in (
            "draft_bf16", "decode_bf16", "tail_bf16", "prefill_bf16_kt", "prefill_bf16_persist",
            "prefill_bf16_persist_wide")))
        with mock.patch.object(K, "SUPPORTED_PAGE_SIZES", (4, 16)):    # pre-R2 kernel
            got = API.prewarm(prof, page_sizes=(8,), device="cpu", cache_dir=self.td)
        self.assertEqual(got, [])                           # kernel without page 8
        self.assertEqual(tuple(K.SUPPORTED_PAGE_SIZES), (4, 8, 16))     # R2-MERGE kernel
        got = API.prewarm(prof, page_sizes=(8,), pos_dtypes=(torch.int64,), device="cpu",
                          cache_dir=self.td)
        self.assertEqual(got, ["qsa.page8.decg.v1/bf16/i64", "qsa.page8.draft.v1/bf16/i64",
                               "qsa.page8.r16.v1/bf16/i64", "qsa.page8.r32_kt.v1/bf16/i64",
                               "qsa.page8.r32_persist.v1/bf16/i64"])
        self.assertEqual(len(self.runs), 6)                 # + the g3 bucket's decg build
        for _, _, kw in self.kinds:
            self.assertEqual((kw.get("pos64"), kw.get("page_size")), (True, 8), kw)


# ---------------------------------------------------------------------------
# Page 8
# ---------------------------------------------------------------------------
P8_RULES = tuple(r + "_p8" for r in ("draft_bf16", "decode_bf16", "tail_bf16", "prefill_bf16_kt",
                                     "prefill_bf16_persist", "prefill_bf16_persist_wide"))
#: page-8 bf16 rules with a GPU pass (draft / decode / tail); the prefill rules are pending
P8_PASS = P8_RULES          # GPU restamp on kernel 58051973: every page-8 rule passes


class Page8(_R2Env):
    def test_rules_mirror_page4(self):
        g = _gen()
        p4 = {s["id"]: s for s in g.RULE_SPEC}
        for s in g.RULE_SPEC_P8:
            b = p4[s["id"][:-3]]
            self.assertEqual(s["page_size"], 8)
            for k in ("relation", "rows", "requests", "columns", "cache_dtype"):
                self.assertEqual(s[k], b[k])
            self.assertLessEqual(s["table_width"]["max"], 8192)
            self.assertLessEqual(PR.column_values(s["columns"])[0], s["table_width"]["max"] * 8)
        with open(base.PROFILE_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        rules = [PR.ProfileRule(r) for r in data["rules"]
                 if PR.page_of(r) == 8 and r["cache_dtype"] == "bf16"]
        self.assertEqual({r.id for r in rules}, set(P8_RULES))
        self.assertTrue(all(r.vid.startswith("qsa.page8.") for r in rules))
        self.assertEqual({r.id for r in rules if r.verified}, set(P8_PASS))
        for r in rules:          # vid/params = the router's default pick for every column
            for c in PR.column_values(r.raw["columns"]):
                self.assertEqual((r.vid, PR.split_knobs(r.vid, r.params)[0]),
                                 PR.expected_variant("bf16", r.raw["regime"], c, 8,
                                                     rows=r.raw["rows"]["min"]), (r.id, c))
        # every R2 intake shape at page 8 is admitted by exactly one page-8 rule
        for (rows, reqs, cols, w, grp) in g.r2_matrix_shapes(8):
            hits = [r.id for r in rules if r.admits(rows, reqs, cols, w)]
            self.assertEqual(len(hits), 1, (rows, reqs, cols, w, grp, hits))

    def test_match_and_reasons(self):
        prof = PR.load_profile(_profile_copy(self.td, rules=P8_RULES), use_cache=False)
        kw = dict(arch="gfx1250", cu=256, cache_dtype="bf16")
        m = PR.match(prof, rows=64, requests=16, columns=16384, table_width=2048, page_size=8, **kw)
        self.assertTrue(m.ok and m.row.id == "decode_bf16_p8", m)
        m = PR.match(prof, rows=64, requests=16, columns=16384, table_width=4096, page_size=4, **kw)
        self.assertTrue(m.ok and m.row.id == "decode_bf16", m)          # page isolation
        m = PR.match(prof, rows=64, requests=16, columns=16384, table_width=2048, page_size=4, **kw)
        self.assertEqual(m.reason, PR.Reason.COLUMNS)    # 2048 pages x 4 < 16384 columns
        for p in (2, 32):
            m = PR.match(prof, rows=64, requests=16, columns=1024, table_width=2048, page_size=p,
                         **kw)
            self.assertEqual(m.reason, PR.Reason.CACHE_SHAPE)
        committed = PR.load_profile(base.PROFILE_JSON, use_cache=False)
        m = PR.match(committed, rows=64, requests=16, columns=16384, table_width=2048,
                     page_size=8, **kw)
        self.assertTrue(m.ok and m.row.id == "decode_bf16_p8", m)       # GPU pass committed
        m = PR.match(committed, rows=64, requests=2, columns=1024, table_width=2048,
                     page_size=8, **kw)
        self.assertTrue(m.ok and m.row.id == "prefill_bf16_kt_p8", m)   # prefill: GPU pass
        m = PR.match(committed, rows=64, requests=2, columns=1024, table_width=2048,
                     page_size=8, allow_unverified=True, **kw)
        self.assertTrue(m.ok and m.row.id == "prefill_bf16_kt_p8", m)

    def test_capability_probe_before_and_after_kernel(self):
        self.assertEqual(API._source_page_sizes(), tuple(K.SUPPORTED_PAGE_SIZES))
        path = os.path.join(self.td, "k.py")
        with open(path, "w") as fh:
            fh.write("X = 1\nSUPPORTED_PAGE_SIZES = (4, 8, 16)\n")
        self.assertEqual(API._source_page_sizes(path), (4, 8, 16))
        prof = _profile_copy(self.td, rules=P8_RULES)
        args = call_args(rows=64, requests=16, width=2048, columns=16384, page=8)
        with mock.patch.object(K, "SUPPORTED_PAGE_SIZES", (4, 16)):     # pre-R2 kernel
            d = API.decide(*args, profile=prof)
            self.assertEqual(d.reason, PR.Reason.CACHE_SHAPE)
            lay = API.layout_eligibility(args[1], profile=prof)
            self.assertEqual(lay.reason, PR.Reason.CACHE_SHAPE)
        # R2-MERGE kernel (d438d2b3): page 8 implemented
        d = API.decide(*args, profile=prof)
        self.assertTrue(d.ok and d.vid == "qsa.page8.decg.v1", d)
        self.assertEqual(d._plan.build_kw, {"page_size": 8})
        API.launch(d, *args)
        self.assertEqual(self.stub.calls[-1]["kind"], "decg")
        lay = API.layout_eligibility(args[1], profile=prof)
        self.assertTrue(lay.ok, lay)

    def test_verify3_bucket_at_page8(self):
        args = _verify3_args(8, columns=4096, width=512)
        args[1] = aligned((8, 8, 1, 128), torch.bfloat16)
        args[5].fill_(8 * 512)
        # committed profile: the page-8 g3 bucket (pass) on the decode rule (pass), default on
        ref = [t.clone() for t in API.launch(API.decide(*args, profile=self.profile,
                                                        buckets=False), *args)]
        d = API.decide(*args, profile=self.profile)
        self.assertTrue(d.ok and d.cell == "bucket:verify3_g3_bf16_p8>rule:decode_bf16_p8", d)
        self.assertEqual(d.vid, "qsa.page8.decg.v1")
        self.assertEqual(d._plan.build_kw, {"page_size": 8, "group_rows": 3, "row_mode": "dedup",
                                            "wg_waves": 4})
        out = API.launch(d, *args)
        self.assertTrue(torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]))
        # staging bucket at page 8 (g3 reset to pending, pad4 marked pass)
        prof = _profile_copy(self.td, rules=P8_RULES, buckets=("verify3_pad4_bf16_p8",),
                             pending=("verify3_g3_bf16_p8",))
        with mock.patch.object(K, "SUPPORTED_PAGE_SIZES", (4, 8, 16)):
            ref = [t.clone() for t in API.launch(API.decide(*args, profile=prof, buckets=False),
                                                 *args)]
            d = API.decide(*args, profile=prof, buckets=True)
            self.assertTrue(d.ok and d.cell.startswith("bucket:verify3_pad4_bf16_p8>"), d)
            out = API.launch(d, *args)
        self.assertTrue(torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]))


# ---------------------------------------------------------------------------
# R2B-FP8: fp8 range rules + kernel-native fp8 verify3 bucket
# ---------------------------------------------------------------------------
FP8_RULES = ("draft_fp8", "decode_fp8", "tail_fp8", "prefill_fp8_kt", "prefill_fp8_kt_tall",
             "prefill_fp8_r32_mid", "prefill_fp8_r32_wide")
#: fp8 rule -> the bf16 rule of the same slot (ROUND-R2E: the fp8 lean ktile band is split
#: by rows into x C2 (rows <= 2048) and x C4 (2049..16384) column tiles)
FP8_BF16_SLOT = {"draft_fp8": "draft_bf16", "decode_fp8": "decode_bf16", "tail_fp8": "tail_bf16",
                 "prefill_fp8_kt": "prefill_bf16_kt", "prefill_fp8_kt_tall": "prefill_bf16_kt",
                 "prefill_fp8_r32_mid": "prefill_bf16_persist",
                 "prefill_fp8_r32_wide": "prefill_bf16_persist_wide"}
#: fp8 rules with a GPU pass under the R2E router; the others await the GPU check
FP8_PASS = FP8_RULES          # GPU restamp on kernel 58051973: every fp8 rule passes
_SFX = {4: "", 8: "_p8", 16: "_p16"}


#: read at import: the test Env clears QSA_* variables inside every test
_R2_MANIFEST_ENV = os.environ.get("QSA_R2_MANIFEST", "")


def _r2_manifest_path():
    cands = [_R2_MANIFEST_ENV,
             os.path.join(base.ROOT, "..", "..", "..", "..", "scripts", "r2",
                          "golden_r2_manifest.json")]
    return next((os.path.abspath(c) for c in cands if c and os.path.isfile(c)), None)


def _fp8_verify3_args(C, columns=1024, width=256, seed=0, page=4):
    rows = 3 * C
    args = call_args(rows=rows, requests=C, width=width, columns=columns, fmt="fp8", page=page)
    g = torch.Generator().manual_seed(seed)
    args[0].copy_(torch.randn(args[0].shape, generator=g).to(torch.bfloat16))
    args[3].copy_(torch.arange(rows, dtype=torch.int32) // 3)        # 3 rows per request
    args[4].copy_(torch.randint(0, 4 * width, (rows,), generator=g).to(args[4].dtype))
    return args


class Fp8Rules(_R2Env):
    def setUp(self):
        super().setUp()
        with open(base.PROFILE_JSON, encoding="utf-8") as fh:
            self.data = json.load(fh)
        self.prof = PR.profile_from_data(self.data)

    def _rules(self, page):
        return [PR.ProfileRule(r) for r in self.data["rules"]
                if r["cache_dtype"] == "fp8" and PR.page_of(r) == page]

    def test_rules_mirror_bf16_and_router(self):
        """Same relations / rows / requests / columns / table_width (per page) as the bf16
        rule of the same slot; variant = the kernel router's fp8 pick; all pending."""
        g = _gen()
        bf = {4: g.RULE_SPEC, 8: g.RULE_SPEC_P8, 16: g.RULE_SPEC_P16}
        fp = {4: g.RULE_SPEC_FP8, 8: g.RULE_SPEC_FP8_P8, 16: g.RULE_SPEC_FP8_P16}
        committed = {r["id"]: r for r in self.data["rules"]}
        for page in (4, 8, 16):
            sfx = _SFX[page]
            self.assertEqual([s["id"] for s in fp[page]], [i + sfx for i in FP8_RULES])
            bfs = {s["id"]: s for s in bf[page]}
            for f in fp[page]:
                b = bfs[FP8_BF16_SLOT[f["id"][:len(f["id"]) - len(sfx)]] + sfx]
                self.assertEqual(f["cache_dtype"], "fp8")
                for k in ("relation", "requests", "columns", "table_width", "rows_x_columns_max"):
                    self.assertEqual(f.get(k), b.get(k), (f["id"], k))
                if "_kt" not in f["id"]:
                    self.assertEqual(f["rows"], b["rows"], f["id"])
                self.assertEqual(PR.page_of(f), page)
                r = committed[f["id"]]
                base_id = f["id"][:len(f["id"]) - len(sfx)]
                self.assertEqual(r["golden"]["status"],
                                 "pass" if base_id in FP8_PASS else "pending", f["id"])
                # variant = the router's default pick at both row bounds, every column;
                # pinned knobs = exactly the default-env wrapper knobs (fp8 draft wg_waves 2)
                vbase, knobs = PR.split_knobs(r["vid"], r["params"])
                self.assertEqual(knobs, PR.default_knobs("fp8", r["regime"]), f["id"])
                for c in PR.column_values(r["columns"]):
                    for rows in (r["rows"]["min"], r["rows"]["max"]):
                        self.assertEqual((r["vid"], vbase),
                                         PR.expected_variant("fp8", r["regime"], c, page,
                                                             rows=rows), (f["id"], c, rows))
            # the fp8 ktile pair splits the bf16 ktile rows band exactly at 2048 / 2049
            kt, tall = (committed["prefill_fp8_kt" + sfx]["rows"],
                        committed["prefill_fp8_kt_tall" + sfx]["rows"])
            bkt = bfs["prefill_bf16_kt" + sfx]["rows"]
            self.assertEqual((kt["min"], kt["max"] + 1, tall["min"], tall["max"]),
                             (bkt["min"], tall["min"], 2049, bkt["max"]))
        pre = PR.variant_prefix
        kt_base = {"k_load": "tdm", "rows_per_block": 32, "kt_rows": 32, "cvt": "fast",
                   "pk8": "intrin", "lean": True, "kconv": True}
        for page in (4, 8, 16):
            ids = {r.id[:-len(_SFX[page])] if _SFX[page] else r.id: r for r in self._rules(page)}
            self.assertEqual(set(ids), set(FP8_RULES))
            self.assertEqual(ids["draft_fp8"].vid, pre(page) + ".draft.v1")
            self.assertEqual(dict(ids["draft_fp8"].params),
                             {"k_load": "tdm", "wave_cols": 32, "sched": "qpre", "pk8": "intrin",
                              "wg_waves": 2})                             # R2E fp8 draft auto
            self.assertEqual(ids["decode_fp8"].vid, pre(page) + ".decg.v1")
            self.assertEqual(ids["decode_fp8"].params["wave_cols"], 64)      # fp8 w64 default
            self.assertEqual(ids["tail_fp8"].vid, pre(page) + ".r16_fp8.v1")
            self.assertEqual(ids["tail_fp8"].params["k_load"], "raw")        # fp8 r16: never TDM
            for k, ct in (("prefill_fp8_kt", 2), ("prefill_fp8_kt_tall", 4)):
                self.assertEqual(ids[k].vid, pre(page) + ".r32_kt_fp8.v1")
                self.assertEqual(dict(ids[k].params), dict(kt_base, col_tiles=ct))
                self.assertEqual(PR.column_values(ids[k].raw["columns"])[-1], 2048)
            for k in ("prefill_fp8_r32_mid", "prefill_fp8_r32_wide"):
                self.assertEqual(ids[k].vid, pre(page) + ".r32_fp8.v1")
                self.assertEqual(ids[k].params["k_load"], "tdm")
                self.assertGreaterEqual(PR.column_values(ids[k].raw["columns"])[0], 4096)
            self.assertEqual(ids["prefill_fp8_r32_wide"].raw["rows_x_columns_max"], 2 ** 25)

    def test_route_guard_holds_for_every_sample(self):
        """route_mismatch guard: the kernel's own router (predict_route, default env,
        128 B cache base) picks each fp8 rule's vid/params for every one of its samples,
        and the decode-group router gives the fp8 g3 bucket's slab for its samples."""
        n = 0
        for r in self.data["rules"]:
            if r["cache_dtype"] != "fp8":
                continue
            page = PR.page_of(r)
            for s in r["samples"]:
                got = API.predict_route(K, "fp8", s["rows"], s["requests"], s["columns"],
                                        0x1000, page)
                self.assertTrue(API._route_ok(*got, r["vid"], r["params"]), (r["id"], s, got))
                n += 1
        self.assertGreater(n, 700)
        for b in self.data["buckets"]:
            if b["cache_dtype"] != "fp8":
                continue
            self.assertEqual(b["params"]["group_rows"], 3)
            for s in b["samples"]:
                rows, reqs = s["rows"], s["requests"]
                w3 = K.decode_group_route(rows, reqs, True, 0x1000, 3)
                self.assertEqual(w3, b["params"]["wave_cols"], (b["id"], s))
                # target (4-row) route = the fp8 decode rule's w64 decode-group variant; the
                # g3 launch takes the router's group_rows=3 slab instead (_decide_bucket)
                got = API.predict_route(K, "fp8", 4 * reqs, reqs, s["columns"], 0x1000,
                                        PR.page_of(b))
                self.assertEqual(got[1]["wave_cols"], 64, (b["id"], s))
                self.assertFalse(API._route_ok(*got, b["vid"], b["params"]))
                got = (got[0], dict(got[1], wave_cols=w3))
                self.assertTrue(API._route_ok(*got, b["vid"], b["params"]), (b["id"], s, got))

    def test_fp8_g3_bucket_capability(self):
        """The fp8 decode-group kernel takes group_rows=3 (dtype-independent knob check)."""
        self.assertEqual(K._check_decg_knobs(3, "base", 4), (3, "dedup", 4))
        ids = [b["id"] for b in self.data["buckets"] if b["cache_dtype"] == "fp8"]
        self.assertEqual(ids, ["verify3_g3_fp8", "verify3_g3_fp8_p8", "verify3_g3_fp8_p16"])
        for b in self.data["buckets"]:
            if b["cache_dtype"] == "fp8":
                PR.validate_bucket(b, self.data["kernel"]["sha256"])
                self.assertEqual(b["golden"]["status"], "pass")
                self.assertEqual(b["group"], {"real": 3, "padded": 4, "kernel_group_rows": 3})
                self.assertEqual(b["vid"], PR.variant_prefix(PR.page_of(b)) + ".decg.v1")
                # ROUND-R2E: decode_group_route(..., group_rows=3) 'auto' slab = 32 for fp8
                self.assertEqual(b["params"]["wave_cols"], 32)
                self.assertEqual(K.decode_group_route(3, 1, True, 0x1000, 3), 32)
                vid, vp = PR.expected_variant("fp8", "decode", 1024, PR.page_of(b), group_rows=3)
                self.assertEqual((b["vid"], PR.split_knobs(b["vid"], b["params"])[0]), (vid, vp))
                self.assertGreater(len(b["samples"]), 10)

    def _resolve(self, prof, page, rows, reqs, cols, w, allow=True):
        kw = dict(arch="gfx1250", cu=256, cache_dtype="fp8", rows=rows, requests=reqs,
                  columns=cols, table_width=w, page_size=page, allow_unverified=allow)
        return PR.match(prof, **kw), PR.match_bucket(prof, **kw)

    def test_r2_matrix_resolves_to_exactly_one_fp8_rule(self):
        """Every R2 intake shape (decode C, verify3 3C, verify4 4C, ctx 4096/16384/65536,
        prefill 1024/4096) at P = 4/8/16 is admitted by exactly ONE fp8 rule, which the
        matcher selects (no exact row intercepts); verify3 also resolves to the page's fp8
        g3 bucket whose target is the fp8 decode rule."""
        g = _gen()
        n = 0
        for page in (4, 8, 16):
            rules = self._rules(page)
            for (rows, reqs, cols, w, grp) in g.r2_matrix_shapes(page):
                hits = [r.id for r in rules if r.admits(rows, reqs, cols, w)]
                self.assertEqual(len(hits), 1, (page, rows, reqs, cols, w, grp, hits))
                m, bm = self._resolve(self.prof, page, rows, reqs, cols, w)
                self.assertTrue(m.ok and m.row.cell == f"rule:{hits[0]}", (page, grp, m))
                if grp == "verify3":
                    self.assertIsNotNone(bm, (page, rows, reqs))
                    self.assertEqual(bm.bucket.id, "verify3_g3_fp8" + _SFX[page])
                    self.assertEqual(bm.target.row.id, "decode_fp8" + _SFX[page])
                else:
                    self.assertIsNone(bm, (page, grp, rows, reqs))
                # committed: a pass rule serves the call; a pending one leaves it uncovered
                # with the typed reason (never a bf16 rule)
                m2, bm2 = self._resolve(self.prof, page, rows, reqs, cols, w, allow=False)
                if hits[0][:len(hits[0]) - len(_SFX[page])] in FP8_PASS:
                    self.assertTrue(m2.ok and m2.row.cell == f"rule:{hits[0]}", (page, grp, m2))
                else:
                    self.assertEqual(m2.reason, PR.Reason.ROW_UNVERIFIED, (page, grp, m2))
                if grp == "verify3":                          # fp8 g3 bucket: GPU pass
                    self.assertEqual(bm2.bucket.id, "verify3_g3_fp8" + _SFX[page], (page, grp))
                else:
                    self.assertIsNone(bm2, (page, grp))
                n += 1
        self.assertEqual(n, 3 * len(g.r2_matrix_shapes(4)))
        # once the GPU check passes every fp8 rule, all of them are served (no allow flag)
        ids = [i + s for i in FP8_RULES for s in _SFX.values()]
        prof = PR.load_profile(_profile_copy(self.td, rules=ids), use_cache=False)
        for page in (4, 8, 16):
            for (rows, reqs, cols, w, grp) in g.r2_matrix_shapes(page):
                m, _ = self._resolve(prof, page, rows, reqs, cols, w, allow=False)
                self.assertTrue(m.ok and m.row.raw["cache_dtype"] == "fp8", (page, grp, m))

    def test_r2_manifest_fp8_cases(self):
        """The frozen R2 golden manifest itself (if present): every fp8 matrix case resolves to
        exactly one fp8 rule; only r1_edge non-pow2 / sub-tile column cases are not
        rule-covered (exact rows or native, as for bf16)."""
        path = _r2_manifest_path()
        if path is None:
            self.skipTest("golden_r2_manifest.json not found (set QSA_R2_MANIFEST)")
        with open(path, encoding="utf-8") as fh:
            cases = [c for c in json.load(fh)["cases"] if "fp8" in c["key_dtypes"]]
        matrix = [c for c in cases if c["group"] != "r1_edge"]
        self.assertEqual(len(matrix), 258)
        self.assertEqual(sum(c["benchmark_exact"] for c in matrix), 69)   # the G2 fp8 cells
        for c in cases:
            k = (c["rows"], c["requests"], c["columns"], c["table_width"])
            hits = [r.id for r in self._rules(c["page_size"]) if r.admits(*k)]
            if c["group"] == "r1_edge" and len(hits) == 0:
                self.assertFalse(PR.columns_admitted(_POW2, c["columns"]), c["name"])
                continue
            self.assertEqual(len(hits), 1, (c["name"], hits))
            m, bm = self._resolve(self.prof, c["page_size"], *k)
            self.assertTrue(m.ok, (c["name"], m))
            if c["group"] == "r1_edge":
                continue
            self.assertEqual(m.row.cell, f"rule:{hits[0]}", c["name"])
            self.assertEqual(bm is not None, c["group"].startswith("mtp_verify_bench3"), c["name"])

    def test_decide_fp8_rule_and_g3_bucket(self):
        """Op path (real wrapper geometry, stub _run): an fp8 decode / verify3 call is served
        by the fp8 rule (decode-group w64) and, with buckets, by ONE group_rows=3 launch."""
        prof = _profile_copy(self.td, rules=("decode_fp8", "tail_fp8", "draft_fp8"),
                             buckets=("verify3_g3_fp8",))
        args = call_args(rows=32, requests=8, width=256, columns=1024, fmt="fp8")
        d = API.decide(*args, profile=prof)
        self.assertTrue(d.ok and d.cell == "rule:decode_fp8" and d.vid == "qsa.page4.decg.v1", d)
        args = call_args(rows=8, requests=8, width=256, columns=1024, fmt="fp8")
        d = API.decide(*args, profile=prof)
        self.assertTrue(d.ok and d.cell == "rule:draft_fp8", d)
        for C in (1, 4, 8):
            args = _fp8_verify3_args(C, seed=C)
            rows, cols = 3 * C, args[8]
            d0 = API.decide(*args, profile=prof, buckets=False)
            self.assertTrue(d0.ok and d0.cell == ("rule:tail_fp8" if C == 1 else "rule:decode_fp8"),
                            d0)
            ref_l, ref_v = (t.clone() for t in API.launch(d0, *args))
            d = API.decide(*args, profile=prof)                 # buckets on by default
            self.assertTrue(d.ok and d.cell == "bucket:verify3_g3_fp8>rule:decode_fp8", d)
            self.assertEqual(d.vid, "qsa.page4.decg.v1")
            p = d._plan
            self.assertEqual((p.rows, p.call_lo, p.call_hi), (rows, rows, rows))
            self.assertEqual(p.build_kw, {"group_rows": 3, "row_mode": "dedup", "wg_waves": 4})
            self.stub.calls.clear()
            out_l, out_v = API.launch(d, *args)
            self.assertEqual(len(self.stub.calls), 1)
            call = self.stub.calls[-1]
            self.assertEqual((call["kind"], call["rows"]), ("decg", rows))
            # ROUND-R2E: the g3 launch uses the router's group_rows=3 slab (32), not the
            # decode rule's w64
            self.assertEqual((call["gx"], call["gy"]), K.decode_group_grid(rows, cols, 32, 3, 4))
            self.assertEqual(call["gy"], C)                           # one group per request
            self.assertTrue(torch.equal(out_l, ref_l) and torch.equal(out_v, ref_v))
            self.assertEqual(API._STAGES, {})
            self.assertEqual(p.exes[1][0].build_args,
                             ((True, float(API._DEFAULT_DIVISOR), 32),
                              dict(group_rows=3, row_mode="dedup", wg_waves=4, pos64=True)))
        # the committed profile: fp8 g3 bucket GPU pass -> selected
        args = _fp8_verify3_args(8)
        d = API.decide(*args, profile=self.profile)
        self.assertTrue(d.ok and str(d.cell).startswith("bucket:verify3_g3_fp8>"), d)


_POW2 = {"pow2": {"min": 64, "max": 65536}}


class Scripts(unittest.TestCase):
    def test_r2_scripts_compile_and_exports(self):
        for rel in ("op_tests/op_benchmarks/qsa_index_scorer/prewarm_time_qsa_index_scorer.py",
                    "op_tests/op_benchmarks/qsa_index_scorer/bucket_qsa_index_scorer.py",
                    "aiter/ops/qsa_index_scorer/_prewarm_worker.py"):
            with open(os.path.join(base.ROOT, rel), encoding="utf-8") as fh:
                compile(fh.read(), rel, "exec")
        for name in ("prewarm_plan", "prewarm_stats", "compile_cache_dir", "kernel_page_sizes"):
            self.assertIs(getattr(base.PKG, name), getattr(API, name))


if __name__ == "__main__":
    unittest.main()
