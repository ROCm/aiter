# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ROUND-R2E (CPU only): the op's DEFAULT path = the GPU-measured winners.

* kernel router defaults: bf16 prefill route preset ``r2e`` (lean ktile RB32 x C2 for
  columns <= 2048, persist above), fp8 prefill route preset ``r2e`` (lean+kconv ktile
  RB32 x C2 for rows <= 2048 / x C4 above, columns <= 2048; r32 fp8 above), fp8 draft
  ``wg_waves`` 2 (``QSA_DRAFT_WG_WAVES=auto``), decode-group ``group_rows=3`` slab 32 for both
  dtypes; never an ipf (``s_prefetch_inst``) route;
* profile: rules / rows / buckets pin exactly those variants + knobs (``decide`` needs no
  env), route_mismatch guard consistent for every sample, evidence carried only where the
  variant is unchanged, buckets ON by default;
* prewarm: the new default variants have a routed private geometry.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_qsa_index_scorer_cpu as base  # noqa: E402

API, PR, K = base.API, base.PR, base.K
SFX = {4: "", 8: "_p8", 16: "_p16"}


def _gen():
    return base._gen()


def _profile():
    with open(base.PROFILE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


class KernelDefaults(unittest.TestCase):
    def test_bf16_prefill_presets(self):
        self.assertEqual(K.PREFILL_ROUTE_DEFAULT, "r2e")
        self.assertEqual(K.PREFILL_ROUTE_SPEC, "r2e")
        t = K.parse_prefill_route("r2e")
        kt = ("ktile", {"kt_rows": 32, "col_tiles": 2, "lean": True})
        for page in (4, 8, 16):
            for cols in (64, 256, 1024, 2048):
                for rows in (32, 1024, 4096, 16384):
                    self.assertEqual(K.prefill_bf16_route(rows, 1, cols, page, t), kt)
            for cols in (4096, 16384, 65536):
                self.assertEqual(K.prefill_bf16_route(1024, 1, cols, page, t), ("persist", {}))
        alt = K.parse_prefill_route("kt_all")
        self.assertEqual(K.prefill_bf16_route(512, 1, 65536, 4, alt), kt)
        self.assertIsNone(K.parse_prefill_route("legacy"))
        self.assertIsNone(K.parse_fp8_prefill_route("legacy"))
        self.assertEqual(K.prefill_bf16_route(1024, 1, 1024, 4, "legacy"), ("persist", {}))

    def test_fp8_prefill_presets(self):
        self.assertEqual(K.FP8_PREFILL_ROUTE_DEFAULT, "r2e")
        self.assertEqual(K.FP8_PREFILL_ROUTE_SPEC, "r2e")
        self.assertIsNone(K.FP8_KT)
        self.assertFalse(K.FP8_R32_EARLY)
        t = K.parse_fp8_prefill_route("r2e")
        c2 = ("ktile", {"kt_rows": 32, "col_tiles": 2, "lean": True, "kconv": True})
        c4 = ("ktile", {"kt_rows": 32, "col_tiles": 4, "lean": True, "kconv": True})
        for page in (4, 8, 16):
            for cols in (64, 1024, 2048):
                self.assertEqual(K.prefill_fp8_route(1024, 1, cols, page, t), c2)
                self.assertEqual(K.prefill_fp8_route(2048, 1, cols, page, t), c2)
                self.assertEqual(K.prefill_fp8_route(2049, 1, cols, page, t), c4)
                self.assertEqual(K.prefill_fp8_route(16384, 1, cols, page, t), c4)
            for cols in (4096, 16384, 65536):
                self.assertEqual(K.prefill_fp8_route(1024, 1, cols, page, t), ("tdm", {}))
        alt = K.parse_fp8_prefill_route("kt_all")
        self.assertEqual(K.prefill_fp8_route(512, 1, 65536, 4, alt), c2)
        self.assertEqual(K.prefill_fp8_route(4096, 1, 4096, 4, alt), c4)
        self.assertEqual(K.prefill_fp8_route(1024, 1, 1024, 4, "legacy"),   # table None = module default
                         ("tdm", {}))
        # spelling: k (kconv) needs l (lean); early / tdm
        self.assertEqual(K._parse_fp8_route_variant("early"), ("early", {}))
        with self.assertRaises(ValueError):
            K._parse_fp8_route_variant("kt32x2k")

    def test_no_default_preset_routes_to_ipf(self):
        """s_prefetch_inst (ipf) variants fault on GPU (bf16): never in a preset."""
        for name, spec in K.PREFILL_ROUTE_PRESETS.items():
            for r in K.parse_prefill_route(name) or ():
                self.assertNotIn("ipf", r[6], (name, spec))
        for name, spec in K.FP8_PREFILL_ROUTE_PRESETS.items():
            for r in K.parse_fp8_prefill_route(name) or ():
                self.assertNotIn("ipf", r[6], (name, spec))

    def test_draft_and_decg_auto_knobs(self):
        self.assertEqual((K.draft_wg_waves(False), K.draft_wg_waves(True)), (4, 2))
        self.assertIsNone(K._DRAFT_WG_WAVES)
        for fp8 in (False, True):
            self.assertEqual(K.decode_group_route(48, 16, fp8, 0x1000, 3), 32)
        self.assertEqual(K.decode_group_route(64, 16, False, 0x1000), 32)
        self.assertEqual(K.decode_group_route(64, 16, True, 0x1000), 64)   # 4-row fp8 unchanged

    def test_profiles_mirror_kernel(self):
        self.assertEqual(PR.KERNEL_ENV_DEFAULTS["QSA_PREFILL_ROUTE"], K.PREFILL_ROUTE_DEFAULT)
        self.assertEqual(PR.KERNEL_ENV_DEFAULTS["QSA_FP8_PREFILL_ROUTE"], K.FP8_PREFILL_ROUTE_DEFAULT)
        self.assertEqual(PR.KERNEL_ENV_DEFAULTS["QSA_DRAFT_WG_WAVES"], "auto")
        self.assertEqual(PR.DECG_G3_WAVE_COLS, {"bf16": K._DECODE_GROUP_AUTO_COLS_G3[False],
                                                "fp8": K._DECODE_GROUP_AUTO_COLS_G3[True]})
        self.assertEqual(PR.default_knobs("fp8", "draft"), {"wg_waves": K.draft_wg_waves(True)})
        self.assertEqual(PR.default_knobs("bf16", "draft"), {})


class RouterDefaults(unittest.TestCase):
    """predict_route (the kernel's own router, default env) == the GPU winners."""

    def _r(self, dt, rows, reqs, cols, page=4):
        return API.predict_route(K, dt, rows, reqs, cols, 0x1000, page)

    def test_r2_targets(self):
        for page in (4, 8, 16):
            pre = PR.variant_prefix(page)
            v, p = self._r("bf16", 1024, 1, 1024, page)
            self.assertEqual((v, p["kt_rows"], p["col_tiles"], p.get("lean")),
                             (pre + ".r32_kt.v1", 32, 2, True))
            self.assertNotIn("ipf", p)
            v, p = self._r("fp8", 1024, 1, 1024, page)
            self.assertEqual((v, p["col_tiles"], p.get("lean"), p.get("kconv")),
                             (pre + ".r32_kt_fp8.v1", 2, True, True))
            v, p = self._r("fp8", 4096, 1, 1024, page)
            self.assertEqual((v, p["col_tiles"], p.get("lean"), p.get("kconv")),
                             (pre + ".r32_kt_fp8.v1", 4, True, True))
            self.assertEqual(self._r("bf16", 1024, 1, 4096, page)[0], pre + ".r32_persist.v1")
            self.assertEqual(self._r("fp8", 1024, 1, 4096, page)[0], pre + ".r32_fp8.v1")
            v, p = self._r("fp8", 16, 16, 4096, page)
            self.assertEqual((v, p.get("wg_waves")), (pre + ".draft.v1", 2))
            v, p = self._r("bf16", 16, 16, 4096, page)
            self.assertEqual((v, p.get("wg_waves")), (pre + ".draft.v1", None))

    def test_expected_variant_matches_router_on_every_rule_sample(self):
        data = _profile()
        n = 0
        for r in data["rules"] + data["rows"]:
            samples = r.get("samples") or [{"rows": r["rows"]["min"], "requests": r["requests"]["min"],
                                            "columns": r["columns"]}]
            for s in samples:
                got = API.predict_route(K, r["cache_dtype"], s["rows"], s["requests"], s["columns"],
                                        0x1000, PR.page_of(r))
                self.assertTrue(API._route_ok(*got, r["vid"], r["params"]),
                                (r.get("id") or r.get("cell"), s, got))
                n += 1
        self.assertGreater(n, 2000)


class ProfileDefaults(unittest.TestCase):
    def setUp(self):
        self.data = _profile()
        self.rules = {r["id"]: r for r in self.data["rules"]}
        self.buckets = {b["id"]: b for b in self.data["buckets"]}

    def test_kernel_and_chain(self):
        g = _gen()
        self.assertEqual(self.data["kernel"]["sha256"], g.KERNEL_SHA256)
        self.assertEqual(g.ISA_IDENTITY_CHAIN[-2:], (g.R2D_KERNEL_SHA256, g.KERNEL_SHA256))
        self.assertEqual(self.data["kernel_env"]["QSA_PREFILL_ROUTE"], "r2e")

    def test_rules_pin_the_winners(self):
        for page, sfx in SFX.items():
            pre = PR.variant_prefix(page)
            kt = self.rules["prefill_bf16_kt" + sfx]
            self.assertEqual((kt["vid"], kt["params"]["kt_rows"], kt["params"]["col_tiles"],
                              kt["params"]["lean"]), (pre + ".r32_kt.v1", 32, 2, True))
            self.assertEqual(PR.column_values(kt["columns"])[-1], 2048)
            self.assertEqual(PR.column_values(self.rules["prefill_bf16_persist" + sfx]["columns"])[0],
                             4096)
            f2, f4 = self.rules["prefill_fp8_kt" + sfx], self.rules["prefill_fp8_kt_tall" + sfx]
            for r, ct, rr in ((f2, 2, (32, 2048)), (f4, 4, (2049, 16384))):
                self.assertEqual(r["vid"], pre + ".r32_kt_fp8.v1")
                self.assertEqual((r["params"]["col_tiles"], r["params"]["lean"],
                                  r["params"]["kconv"]), (ct, True, True))
                self.assertEqual((r["rows"]["min"], r["rows"]["max"]), rr)
            self.assertEqual(self.rules["draft_fp8" + sfx]["params"]["wg_waves"], 2)
            self.assertNotIn("wg_waves", self.rules["draft_bf16" + sfx]["params"])
            g3 = self.buckets["verify3_g3_fp8" + sfx]
            self.assertEqual((g3["params"]["wave_cols"], g3["params"]["group_rows"]), (32, 3))
            b3 = self.buckets["verify3_g3_bf16" + sfx]
            self.assertEqual((b3["params"]["wave_cols"], b3["params"]["group_rows"]), (32, 3))
            self.assertNotIn("wg_waves", b3["params"])
        for e in self.data["rules"] + self.data["rows"] + self.data["buckets"]:
            self.assertNotIn("ipf", e["params"])

    def test_evidence_state(self):
        """pass = unchanged variants (GPU evidence carried by ISA identity); pending = every
        re-routed / re-pinned entry (GPU gen_profile check)."""
        st = {k: v["golden"]["status"] for k, v in {**self.rules, **self.buckets}.items()}
        for sfx in SFX.values():
            for rid in ("prefill_bf16_kt", "prefill_bf16_persist", "prefill_bf16_persist_wide",
                        "draft_fp8", "prefill_fp8_kt", "prefill_fp8_kt_tall",
                        "prefill_fp8_r32_mid", "verify3_g3_fp8"):
                self.assertEqual(st[rid + sfx], "pending", rid + sfx)
            for rid in ("draft_bf16", "decode_bf16", "tail_bf16", "decode_fp8", "tail_fp8",
                        "prefill_fp8_r32_wide", "verify3_g3_bf16"):
                self.assertEqual(st[rid + sfx], "pass", rid + sfx)
            for rid in ("verify3_pad4_bf16", "tail_round4_bf16"):   # routing-only GPU 'fail'
                self.assertEqual(st[rid + sfx], "fail", rid + sfx)
        ksha = self.data["kernel"]["sha256"]
        for e in self.data["rules"] + self.data["buckets"] + self.data["rows"]:
            g = e["golden"]
            if g["status"] == "pass" and g["kernel_sha256"] != ksha:
                self.assertIn("ROUND-R2E no-device gate", g["isa_identity"], e.get("id") or e["cell"])
        for r in self.data["rows"]:
            if r["cache_dtype"] == "fp8" and r["regime"] == "draft":
                self.assertEqual((r["params"].get("wg_waves"), r["golden"]["status"]), (2, "pending"))


class BucketsDefaultOn(unittest.TestCase):
    def test_env_and_arg(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(API.BUCKETS_ENV, None)
            self.assertTrue(API._buckets_enabled(None))
            self.assertFalse(API._buckets_enabled(False))
            for off in ("0", "off", "false", "no"):
                os.environ[API.BUCKETS_ENV] = off
                self.assertFalse(API._buckets_enabled(None), off)
            os.environ[API.BUCKETS_ENV] = "1"
            self.assertTrue(API._buckets_enabled(None))

    def test_verify3_matches_g3_bucket_both_dtypes(self):
        prof = PR.profile_from_data(_profile())
        for dt in ("bf16", "fp8"):
            for page, sfx in SFX.items():
                for C in (1, 8, 64, 448):
                    bm = PR.match_bucket(prof, arch="gfx1250", cu=256, cache_dtype=dt, rows=3 * C,
                                         requests=C, columns=1024, table_width=1024 // page,
                                         allow_unverified=True, page_size=page,
                                         native_group_rows=API.kernel_real_rows_per_group(K))
                    self.assertIsNotNone(bm, (dt, page, C))
                    self.assertEqual(bm.bucket.id, f"verify3_g3_{dt}{sfx}")


class PrewarmGeometry(unittest.TestCase):
    def test_default_variants_have_routed_geometry(self):
        data = _profile()
        seen = set()
        for e in data["rules"] + data["buckets"]:
            if e.get("kind") not in (None, "group") or (e.get("kind") == "group"
                                                        and "group_rows" not in e["params"]):
                continue
            vid, params = e["vid"], dict(e["params"])
            page = PR.page_of(e)
            v4 = "qsa.page4." + vid.split(".", 2)[2]
            rows, reqs, width = API._PREWARM_GEOMETRY[v4]
            for (gv, (pk, pv)), geo in API._PREWARM_GEOMETRY_PARAMS.items():
                if gv == v4 and params.get(pk) == pv:
                    rows, reqs, width = geo
            knobs = PR.split_knobs(vid, params)[1]
            if "group_rows" in knobs:
                rows = 3 * reqs
            if page != 4:
                width = max(1, width * 4 // page)
            cols = width * page
            got = API.predict_route(K, e["cache_dtype"], rows if "group_rows" not in knobs
                                    else 4 * reqs, reqs, cols, 0x1000, page)
            if "group_rows" in knobs:
                got = (got[0], dict(got[1], wave_cols=K.decode_group_route(
                    rows, reqs, e["cache_dtype"] == "fp8", 0x1000, 3)))
            self.assertTrue(API._route_ok(*got, vid, params), (e["id"], got))
            seen.add(vid.split(".")[2])
        self.assertTrue({"r32_kt", "r32_kt_fp8", "r32_persist", "r32_fp8", "draft", "decg"} <= seen)


if __name__ == "__main__":
    unittest.main()
