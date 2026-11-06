# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""RULES-REQUESTS (CPU only): prefill rules cover mixed prefill+decode steps with many
requests.

The ATOM e2e (a08-1, KV block 16 / 32, decode concurrency 8..448 at ctx 4096) issues
prefill-regime scorer calls with many requests (``rows >= 32 * requests``; e.g. rows 16384
/ requests 53 at page 4, rows 13600 / requests 63 at page 8) that the prefill rules
(requests 1..32) did not admit -> ``rows_rule_uncovered`` -> native torch path. The
prefill kernels have no request-count limit (grid = column tiles x row blocks, LDS fixed
per specialization, per-block SGPR request guard with the exact per-wave fallback), so the
prefill rules now admit requests 1..512 (= what rows <= 16384 and rows >= 32 * requests
allow); new samples cover many requests and mixed per-request row counts.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_qsa_index_scorer_cpu as base

API, PR, K = base.API, base.PR, base.K
SFX = {4: "", 8: "_p8", 16: "_p16"}
WIDENED = (
    "prefill_bf16_kt",
    "prefill_bf16_persist",
    "prefill_bf16_persist_wide",
    "prefill_fp8_kt",
    "prefill_fp8_kt_tall",
    "prefill_fp8_r32_mid",
    "prefill_fp8_r32_wide",
)
E2E_W = {4: 16384, 8: 8192, 16: 4096}


def _profile():
    with open(base.PROFILE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


GEN = base._gen()


class RuleBounds(unittest.TestCase):
    def setUp(self):
        self.data = _profile()
        self.rules = {r["id"]: r for r in self.data["rules"]}

    def test_prefill_rules_admit_512_requests(self):
        self.assertEqual(GEN.PREFILL_MAX_REQUESTS, 512)
        ids = {s["id"] for s in GEN.all_rule_specs()}
        for rid in WIDENED:
            for sfx in SFX.values():
                r = self.rules[rid + sfx]
                self.assertEqual(r["relation"], "rows>=32*requests")
                self.assertEqual(r["requests"], {"min": 1, "max": 512}, rid + sfx)
                self.assertIn(rid + sfx, ids)
                # rows <= rows.max and rows >= 32 * requests: 512 is the tight bound
                self.assertLessEqual(int(r["rows"]["max"]) // 32, 512)
        # every prefill rule is one of the widened ones
        pre = {r["id"] for r in self.data["rules"] if r["regime"] == "prefill"}
        self.assertEqual(pre, {w + s for w in WIDENED for s in SFX.values()})

    def test_observed_e2e_calls_are_rule_covered(self):
        prof = PR.profile_from_data(self.data)
        cases = []
        for page in (4, 8, 16):
            for rows, reqs, cols in GEN.E2E_MIXED_STEP_CALLS:
                for w in (E2E_W[page], 1024 // page):
                    cases += [
                        ("bf16", page, rows, reqs, cols, w, "prefill_bf16_kt"),
                        ("fp8", page, rows, reqs, cols, w, "prefill_fp8_kt_tall"),
                    ]
        for dt, page, rows, reqs, cols, w, rid in cases:
            self.assertEqual(PR.regime_of(rows, reqs), "prefill")
            m = PR.match(
                prof,
                arch="gfx1250",
                cu=256,
                cache_dtype=dt,
                rows=rows,
                requests=reqs,
                columns=cols,
                table_width=w,
                allow_unverified=True,
                page_size=page,
            )
            self.assertTrue(m.ok, (dt, page, rows, reqs, w, m))
            self.assertEqual(m.row.id, rid + SFX[page])

    def test_many_request_prefill_coverage(self):
        """Every prefill-regime call with up to 448 requests (the e2e max-num-seqs) and rows
        up to 16384 at columns 64..2048 resolves to a prefill rule (pending allowed)."""
        prof = PR.profile_from_data(self.data)
        for page in (4, 8, 16):
            for q in (33, 48, 53, 63, 64, 100, 128, 256, 300, 448, 512):
                for rows in sorted({32 * q, 32 * q + 7, 16384}):
                    if rows > 16384:
                        continue
                    for dt, cols in (("bf16", 1024), ("fp8", 1024), ("bf16", 4096)):
                        m = PR.match(
                            prof,
                            arch="gfx1250",
                            cu=256,
                            cache_dtype=dt,
                            rows=rows,
                            requests=q,
                            columns=cols,
                            table_width=E2E_W[page],
                            allow_unverified=True,
                            page_size=page,
                        )
                        self.assertTrue(m.ok, (page, q, rows, dt, cols, m))
                        self.assertEqual(m.row.raw["regime"], "prefill")

    def test_non_prefill_rules_unchanged(self):
        """Only the prefill rules changed: draft / decode / tail rules and every row and
        bucket keep their committed golden 'pass'."""
        for r in self.data["rules"]:
            if r["regime"] != "prefill":
                self.assertEqual(r["golden"]["status"], "pass", r["id"])
                self.assertFalse(any(s.get("layout") for s in r["samples"]), r["id"])
        for r in self.data["rows"]:
            self.assertEqual(r["golden"]["status"], "pass", r["cell"])


class Samples(unittest.TestCase):
    def setUp(self):
        self.rules = {r["id"]: r for r in _profile()["rules"]}

    def test_many_request_and_mixed_samples(self):
        for rid in WIDENED:
            for page, sfx in SFX.items():
                r = self.rules[rid + sfx]
                many = [s for s in r["samples"] if s["requests"] > 32]
                mixed = [s for s in r["samples"] if s.get("layout") == "mixed"]
                self.assertTrue(many, rid + sfx)
                self.assertTrue(mixed, rid + sfx)
                for s in mixed:
                    self.assertEqual(
                        set(s),
                        {
                            "rows",
                            "requests",
                            "columns",
                            "table_width",
                            "source",
                            "layout",
                        },
                    )
                    fx = GEN.sample_fixture(dict(s, page_size=page))
                    self.assertEqual(fx["layout"], "mixed")
                    GEN.mixed_counts(s["rows"], s["requests"])  # feasible
        # the observed calls are mixed samples of the ktile rules (bf16 kt, fp8 kt_tall)
        for page, sfx in SFX.items():
            for rid in ("prefill_bf16_kt", "prefill_fp8_kt_tall"):
                keys = {
                    PR.sample_key(s)
                    for s in self.rules[rid + sfx]["samples"]
                    if s.get("layout") == "mixed"
                }
                for rows, q, c in GEN.E2E_MIXED_STEP_CALLS:
                    self.assertIn((rows, q, c, E2E_W[page]), keys, rid + sfx)
                    self.assertIn((rows, q, c, 1024 // page), keys, rid + sfx)
            reqs = {
                s["requests"] for s in self.rules["prefill_bf16_kt" + sfx]["samples"]
            }
            self.assertTrue({48, 53, 63, 64, 128, 256, 448, 512} <= reqs, sfx)

    def test_pre_widening_samples_kept(self):
        """The bounds sampler keeps the requests 16 / 32 corners of the old rules."""
        for rid in WIDENED:
            for sfx in SFX.values():
                qs = {s["requests"] for s in self.rules[rid + sfx]["samples"]}
                self.assertTrue({1, 16, 32} <= qs, (rid + sfx, sorted(qs)))

    def test_samples_route_to_the_rule_variant(self):
        n = 0
        for rid in WIDENED:
            for page, sfx in SFX.items():
                r = self.rules[rid + sfx]
                for s in r["samples"]:
                    got = API.predict_route(
                        K,
                        r["cache_dtype"],
                        s["rows"],
                        s["requests"],
                        s["columns"],
                        0x1000,
                        page,
                    )
                    self.assertTrue(
                        API._route_ok(*got, r["vid"], r["params"]), (rid + sfx, s, got)
                    )
                    n += 1
        self.assertGreater(n, 1000)


class MixedLayout(unittest.TestCase):
    def test_counts(self):
        for rows, q in (
            (16384, 53),
            (13600, 63),
            (16384, 448),
            (2047, 63),
            (2000, 60),
            (32, 1),
            (64, 2),
        ):
            c = GEN.mixed_counts(rows, q)
            self.assertEqual((len(c), sum(c)), (q, rows))
            n_long = max(1, q // 8)
            longs = [n for n in c if n > 4]
            self.assertEqual(len(longs), n_long, (rows, q))
            self.assertTrue(all(n >= 1 for n in c))
            short = [n for i, n in enumerate(c) if n <= 4]
            self.assertTrue(set(short) <= {1, 2, 3, 4})
            if q >= 8:
                self.assertIn(1, short)
                self.assertIn(4, short)
            self.assertEqual(PR.regime_of(rows, q), "prefill")
        with self.assertRaises(ValueError):
            GEN.mixed_counts(16, 8)  # 7 decode requests take 16 rows: no long row left

    def test_layout_tensors(self):
        ctx = 4096
        t2r, pos = GEN._mixed_layout(16384, 53, ctx)
        c = GEN.mixed_counts(16384, 53)
        self.assertEqual(t2r.numel(), 16384)
        self.assertTrue(bool((t2r[1:] >= t2r[:-1]).all()))
        self.assertEqual(sorted(set(t2r.tolist())), list(range(53)))
        self.assertTrue(bool(((pos >= 0) & (pos < ctx)).all()))
        start = 0
        for i, n in enumerate(c):
            self.assertEqual(int(pos[start + n - 1]), ctx - 1, i)
            start += n
        # mixed blocks: some 32-row blocks hold several requests (per-wave fallback),
        # others one request (reuse path)
        blocks = [set(t2r[b : b + 32].tolist()) for b in range(0, 16384, 32)]
        self.assertTrue(any(len(b) > 1 for b in blocks))
        self.assertTrue(any(len(b) == 1 for b in blocks))
        # the even layout is unchanged
        t2e, pe = GEN._even_layout(16384, 512, ctx)
        self.assertEqual(t2e.tolist(), [i // 32 for i in range(16384)])
        self.assertEqual(int(pe[31]), ctx - 1)


class KernelRouter(unittest.TestCase):
    """The router's prefill decision and variant choice depend on rows / columns only:
    the request count enters only through the regime predicate rows >= 32 * requests."""

    def test_regime_predicate(self):
        for rows, q in ((16384, 53), (13600, 63), (16384, 512), (2047, 63)):
            self.assertEqual(K.wmma_rows_per_block(rows, q), 32)
            self.assertEqual(K.wmma_bf16_rows_per_block(rows, q), 32)
            self.assertIsNone(K.draft_route(rows, q, False, 0x1000))
            self.assertIsNone(K.decode_group_route(rows, q, False, 0x1000))
            self.assertIsNone(K.decode_group_route(rows, q, True, 0x1000))
        # one row short of 32 * requests is not prefill (decode / tail kernels)
        self.assertEqual(K.wmma_rows_per_block(32 * 53 - 1, 53), 16)

    def test_prefill_routes_ignore_requests(self):
        for page in (4, 8, 16):
            for rows in (1696, 2048, 4096, 16384):
                for cols in (64, 1024, 2048, 4096, 8192, 16384):
                    bf = K.prefill_bf16_route(rows, 1, cols, page)
                    f8 = K.prefill_fp8_route(rows, 1, cols, page)
                    for q in (2, 32, 53, 448, 512):
                        if 32 * q > rows:
                            continue
                        self.assertEqual(K.prefill_bf16_route(rows, q, cols, page), bf)
                        self.assertEqual(K.prefill_fp8_route(rows, q, cols, page), f8)


if __name__ == "__main__":
    unittest.main()
