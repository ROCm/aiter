# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023 -- stub closures are called inside the loop iteration that defines them
"""CPU contract tests for ``aiter.ops.qsa_index_scorer`` (no GPU, no device, no aiter import).

Covers: profile format/loader/matcher and every typed reason, the off-by-default
guard (no kernel/FlyDSL import, no device query), ABI/signature drift against the
frozen ATOM scorer (read from ``$QSA_GOLDEN_DIR/atom_impl/...``, never copied),
the kernel's public API, the profile generator (``verify`` = no drift), the
regime/variant rules against the kernel's own router, and the dispatch path
(real kernel wrapper with ``_run``/builders intercepted) for every profiled row.

    QSA_GOLDEN_DIR=/path/golden_baseline python3 -B op_tests/test_qsa_index_scorer_cpu.py -v
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import uuid
from unittest import mock

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PKG_DIR = os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer")
LOADER_DIR = os.path.join(ROOT, "op_tests", "op_benchmarks", "qsa_index_scorer")
PROFILE_JSON = os.path.join(
    ROOT, "aiter", "configs", "qsa_index_scorer", "qsa_page4_gfx1250_v1.json"
)
ATOM_REL = os.path.join("atom_impl", "atom", "model_ops", "qwen4_exp", "ops", "qsa.py")


def _golden_dir() -> str:
    cands = [
        os.environ.get("QSA_GOLDEN_DIR", ""),
        os.path.join(ROOT, "..", "..", "..", "..", "golden_baseline"),
        "/work/task/MI455_QSA_INDEX_SCORER/golden_baseline",
    ]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, ATOM_REL)):
            return os.path.abspath(c)
    raise RuntimeError(
        "frozen golden not found: set QSA_GOLDEN_DIR (drift guards must run)"
    )


@contextlib.contextmanager
def _tmpdir():
    """Plain makedirs temp dir (mkdtemp's 0o700 breaks some sandboxed Windows ACLs)."""
    path = os.path.join(
        tempfile.gettempdir(), f"qsa_isc_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


_PKG_N = [0]


def load_pkg():
    """Load aiter/ops/qsa_index_scorer standalone (no `aiter` package import)."""
    _PKG_N[0] += 1
    name = f"qsa_isc_test{_PKG_N[0]}"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PKG_DIR, "__init__.py"), submodule_search_locations=[PKG_DIR]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, sys.modules[name + ".api"], sys.modules[name + ".profiles"]


def load_kernel():
    sys.path.insert(0, LOADER_DIR)
    try:
        import qsa_kernel_loader
    finally:
        sys.path.remove(LOADER_DIR)
    K, how = qsa_kernel_loader.load_kernel_module(allow_stub=True)
    return K, how


PKG, API, PR = load_pkg()
_CLEAN = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith("QSA_") or k in ("QSA_GOLDEN_DIR", "QSA_RESULTS")
}
_CLEAN.pop(PR.PROFILE_ENV, None)
with mock.patch.dict(os.environ, _CLEAN, clear=True):
    K, K_HOW = load_kernel()
FREEZE_REASONS = (
    "ok",
    "disabled",
    "profile_not_found",
    "profile_invalid",
    "profile_kernel_mismatch",
    "kernel_env_mismatch",
    "kernel_unavailable",
    "device_not_gpu",
    "device_mismatch",
    "arch_unsupported",
    "cu_unsupported",
    "compress_ratio_unsupported",
    "q_dtype_unsupported",
    "q_shape_unsupported",
    "q_layout_unsupported",
    "cache_format_unsupported",
    "cache_shape_unsupported",
    "cache_layout_unsupported",
    "cache_alignment_unsupported",
    "scale_layout_unsupported",
    "metadata_unsupported",
    "output_layout_unsupported",
    "empty_call",
    "divisor_uncovered",
    "max_columns_invalid",
    "plane_too_large",
    "rows_rule_uncovered",
    "columns_uncovered",
    "table_width_uncovered",
    "row_unverified",
    "route_mismatch",
)


def aligned(shape, dtype, align=128, offset=0):
    n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    back = torch.zeros(n + 2 * align, dtype=torch.uint8)
    start = (-back.data_ptr()) % align + offset
    return back[start : start + n].view(dtype).view(*shape)


def call_args(
    rows=64,
    requests=16,
    width=16384,
    columns=65536,
    fmt="bf16",
    pages=8,
    qp64=True,
    cache_offset=0,
    page=4,
):
    q = aligned((rows, 4, 128), torch.bfloat16)
    if fmt == "bf16":
        cache, scale = (
            aligned((pages, page, 1, 128), torch.bfloat16, offset=cache_offset),
            None,
        )
    else:
        cache = aligned((pages, page, 1, 128), torch.float8_e4m3fn, offset=cache_offset)
        scale = torch.ones((pages, page, 1), dtype=torch.float32)
    pt = torch.zeros((requests, width), dtype=torch.int32)
    if rows <= requests:
        t2r = torch.arange(rows, dtype=torch.int32)
    else:
        t2r = (torch.arange(rows, dtype=torch.int32) * requests) // rows
    qp = torch.arange(rows, dtype=torch.int64 if qp64 else torch.int32)
    cl = torch.full((requests,), page * width, dtype=torch.int32)
    return [q, cache, pt, t2r, qp, cl, 4, None, columns, None, False, scale]


class _Launch:
    def __init__(self, kind, cols=64):
        self.kind, self.cols_per_block = kind, cols


class Env(unittest.TestCase):
    """Clean QSA_* env, CPU tensors admitted, device info faked, kernel injected."""

    profile = PR.DEFAULT_PROFILE

    def setUp(self):
        self._p = [
            mock.patch.dict(os.environ, _CLEAN, clear=True),
            mock.patch.object(API, "_ALLOWED_DEVICE_TYPES", ("cuda", "cpu")),
            mock.patch.object(API, "_device_info", lambda dev: ("gfx1250", 256)),
            mock.patch.object(API, "_KERNEL", K),
            mock.patch.object(API, "_KERNEL_ENV", PR.effective_kernel_env(_CLEAN)),
        ]
        for p in self._p:
            p.start()
        PR.clear_cache()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()

    def decide(self, args, **kw):
        kw.setdefault("profile", self.profile)
        return API.decide(*args, **kw)

    def assertReason(self, args, reason, **kw):
        d = self.decide(args, **kw)
        self.assertFalse(d.ok, d)
        self.assertEqual(d.reason.value, reason, d.detail)
        return d


# ---------------------------------------------------------------------------
class ProfileFormat(unittest.TestCase):
    def setUp(self):
        PR.clear_cache()
        with open(PROFILE_JSON, encoding="utf-8") as fh:
            self.data = json.load(fh)

    def test_loads_and_validates(self):
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        self.assertEqual(prof.name, "qsa_page4_gfx1250_v1")
        self.assertEqual(os.path.normcase(prof.path), os.path.normcase(PROFILE_JSON))
        self.assertGreaterEqual(len(prof.rows), 15)
        self.assertEqual(prof.archs(), {"gfx1250"})

    def test_kernel_sha_matches_source(self):
        with open(PR.KERNEL_PATH, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(
            self.data["kernel"]["sha256"],
            sha,
            "kernel changed: re-run gen_profile emit + GPU check",
        )
        gen = _gen()
        self.assertEqual(
            sha, gen.KERNEL_SHA256, "kernel source differs from the profiled kernel"
        )
        self.assertEqual(self.data["kernel"]["git_rev"], gen.KERNEL_GIT_REV)
        # ROUND-PAGE16: page-4 evidence recorded on bf492262 carries over only by ISA identity
        self.assertEqual(
            gen.PAGE4_EVIDENCE_KERNEL_SHA256,
            "bf49226206b999f948e8b1eb0605935157b3d291b0a0b243581897d8ca661b93",
        )

    def test_rows(self):
        ksha = self.data["kernel"]["sha256"]
        for r in self.data["rows"]:
            self.assertEqual(r["sig"], PR.row_signature(r))
            # R2E: prefill variants depend on rows (ktile bands); rows pin the default-env
            # wrapper knobs (fp8 draft wg_waves 2) like the rules
            vid, params = PR.expected_variant(
                r["cache_dtype"],
                r["regime"],
                r["columns"],
                PR.page_of(r),
                rows=r["rows"]["min"],
            )
            self.assertEqual(
                (r["vid"], r["params"]),
                (vid, dict(params, **PR.default_knobs(r["cache_dtype"], r["regime"]))),
            )
            self.assertEqual(
                PR.regime_of(r["rows"]["min"], r["requests"]["min"]), r["regime"]
            )
            self.assertIn(r["golden"]["status"], ("pass", "pending"))
            g = r["golden"]
            if g["status"] == "pass":
                self.assertTrue(g.get("cases") or g.get("checked"))
                # evidence names the kernel it was verified on; older kernel => ISA identity
                self.assertTrue(g.get("kernel_sha256"), r["cell"])
                if g["kernel_sha256"] != ksha:
                    self.assertTrue(g.get("isa_identity"), r["cell"])
                if g["kernel_sha256"] in _gpu_stamp_kernels(ksha):
                    _assert_lint_link(self, g, ksha, r["cell"])
                elif r["cache_dtype"] != "bf16":  # fp8: GPU-restamped on bf492262 only
                    self.assertEqual(
                        g["kernel_sha256"], _gen().PAGE4_EVIDENCE_KERNEL_SHA256
                    )
                    self.assertIn("ROUND-PAGE16", g["isa_identity"], r["cell"])
            for key in (
                "arch",
                "cu",
                "cache_dtype",
                "columns",
                "table_width",
                "vid",
                "params",
                "cell",
                "generated_by",
                "notes",
            ):
                self.assertIn(key, r)

    def test_kernel_change_evidence_policy(self):
        """82c4dec5: bf16 Phase-1 rows carried by ISA identity; every fp8 row pending until
        a GPU `gen_profile check` on this kernel restamps it (then: pass on this sha).
        """
        old = "30c6213d492ad50a9e063456628737ca993060d73dfa4ead248558ba5abb121c"
        ksha = self.data["kernel"]["sha256"]
        n_bf16 = 0
        p4sha = _gen().PAGE4_EVIDENCE_KERNEL_SHA256
        pending_bf16 = {}
        for r in self.data["rows"]:
            g = r["golden"]
            if g["status"] == "pending" and "stale" in g:
                # ROUND-R2E: evidence that no longer covers the routed build is kept as
                # `stale` (never selected) with the reason the row went back to pending
                self.assertEqual(g["stale"]["status"], "pass", r["cell"])
                self.assertIn("ROUND-R2E", g["reason"], r["cell"])
                if r["cache_dtype"] == "bf16":
                    pending_bf16[r["cell"]] = g
                continue
            gpu = _gpu_stamp_kernels(ksha)
            if g["status"] == "pass" and g["kernel_sha256"] in (*gpu, p4sha):
                self.assertTrue(g.get("checked"), r["cell"])  # restamped on GPU
                if g["kernel_sha256"] in gpu:  # AITER-LINT identity
                    _assert_lint_link(self, g, ksha, r["cell"])
                else:  # ROUND-PAGE16 identity
                    self.assertIn("ROUND-PAGE16", g["isa_identity"], r["cell"])
                n_bf16 += r["cache_dtype"] == "bf16"
            elif r["cache_dtype"] == "fp8":
                self.assertEqual(
                    g["status"], "pending", r["cell"]
                )  # never inherits old evidence
                if g.get("stale_cases"):
                    self.assertEqual(g["stale_kernel_sha256"], old)
            elif g["status"] == "pass":
                self.assertEqual(
                    (g["kernel_sha256"], g["kernel_git_rev"]), (old, "a78c285c")
                )
                self.assertIn("byte-identical", g["isa_identity"])
                n_bf16 += 1
        # R2E: the 2 bf16 c256 prefill rows re-route to lean ktile (their Phase-1 evidence is
        # from kernel 30c6, off the ISA-identity chain); the 2 bf16 c65536 prefill rows stay
        # r32_persist whose ISA changed on the R2C link -> all 4 pending until a GPU check
        self.assertGreaterEqual(n_bf16, 7)
        if pending_bf16:
            self.assertLessEqual(
                set(pending_bf16),
                {
                    "prefill_bf16_r512q2_c65536_w16384",
                    "prefill_bf16_r320q2_c65536_w16384",
                    "prefill_bf16_r64q2_c256_w64",
                    "prefill_bf16_r272q1_c256_w64",
                },
            )
        for cell in ("prefill_bf16_r64q2_c256_w64", "prefill_bf16_r272q1_c256_w64"):
            if cell in pending_bf16:
                st = pending_bf16[cell]["stale"]
                self.assertEqual(
                    (st["kernel_sha256"], st["kernel_git_rev"]), (old, "a78c285c")
                )
                self.assertIn("30c6213d492a", pending_bf16[cell]["reason"])
        for cell in (
            "prefill_bf16_r512q2_c65536_w16384",
            "prefill_bf16_r320q2_c65536_w16384",
        ):
            if cell in pending_bf16:
                self.assertEqual(pending_bf16[cell]["stale"]["kernel_sha256"], p4sha)
                self.assertIn("ISA changed", pending_bf16[cell]["reason"])
        bf16 = {r["cell"]: r for r in self.data["rows"] if r["cache_dtype"] == "bf16"}
        self.assertEqual(
            (
                bf16["prefill_bf16_r64q2_c256_w64"]["vid"],
                bf16["prefill_bf16_r64q2_c256_w64"]["params"],
            ),
            (
                "qsa.page4.r32_kt.v1",
                {
                    "k_load": "tdm",
                    "rows_per_block": 32,
                    "kt_rows": 32,
                    "col_tiles": 2,
                    "lean": True,
                },
            ),
        )
        self.assertEqual(
            bf16["prefill_bf16_r512q2_c65536_w16384"]["vid"], "qsa.page4.r32_persist.v1"
        )
        fp8 = {r["cell"]: r for r in self.data["rows"] if r["cache_dtype"] == "fp8"}
        self.assertEqual(
            fp8["decode_fp8_r64q16_c65536_w16384"]["params"],
            {
                "k_load": "tdm",
                "wave_cols": 64,
                "barrier": False,
                "sched": "base",
                "pk8": "intrin",
            },
        )
        # R2E: the default-env wrapper launches fp8 draft with wg_waves 2 (QSA_DRAFT_WG_WAVES
        # auto) -> the row pins that knob (P.default_knobs), like the draft_fp8 rule
        self.assertEqual(PR.default_knobs("fp8", "draft"), {"wg_waves": 2})
        self.assertEqual(
            fp8["draft_fp8_r16q16_c65536_w16384"]["params"],
            {
                "k_load": "tdm",
                "wave_cols": 32,
                "sched": "qpre",
                "pk8": "intrin",
                "wg_waves": 2,
            },
        )
        self.assertEqual(
            fp8["prefill_fp8_r512q2_c65536_w16384"]["vid"], "qsa.page4.r32_fp8.v1"
        )
        # a pass on an older kernel without an ISA-identity record is rejected
        with _tmpdir() as td:
            d = json.loads(json.dumps(self.data))
            d["profile"] = "qsa_bad_v1"
            r0 = next(r for r in d["rows"] if r["golden"]["status"] == "pass")
            # simulate a pass verified on an older kernel that lacks an ISA-identity record
            r0["golden"]["kernel_sha256"] = "0" * 64
            r0["golden"].pop("isa_identity", None)
            with self.assertRaises(PR.ProfileError) as cm:
                PR.load_profile(self._write(td, d), use_cache=False)
            self.assertIn("isa_identity", str(cm.exception))
            d = json.loads(json.dumps(self.data))
            d["profile"] = "qsa_bad_v1"
            r0 = next(r for r in d["rows"] if r["cache_dtype"] == "fp8")
            r0["golden"] = {"status": "pass", "cases": ["x"]}  # no kernel sha
            with self.assertRaises(PR.ProfileError):
                PR.load_profile(self._write(td, d), use_cache=False)

    def test_target_buckets_present(self):
        cells = {r["cell"] for r in self.data["rows"]}
        for dt in ("bf16", "fp8"):
            for c in (65536, 32768):
                self.assertIn(f"decode_{dt}_r64q16_c{c}_w16384", cells)
                self.assertIn(f"draft_{dt}_r16q16_c{c}_w16384", cells)
            self.assertIn(f"prefill_{dt}_r512q2_c65536_w16384", cells)
            self.assertIn(f"prefill_{dt}_r320q2_c65536_w16384", cells)

    def test_kernel_env_and_golden_meta(self):
        self.assertEqual(self.data["kernel_env"], PR.KERNEL_ENV_DEFAULTS)
        g = self.data["golden"]
        self.assertEqual((g["rtol"], g["atol"]), (1e-4, 1e-4))
        self.assertEqual(
            g["revision"],
            "4b8888907e2bd3530bfde10eb081bf8cf60d49749aea4cac62ec486211b12ff8",
        )

    def test_generator_no_drift(self):
        spec = importlib.util.spec_from_file_location(
            "qsa_gen_profile_test", os.path.join(PKG_DIR, "gen_profile.py")
        )
        gen = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [PKG_DIR] + sys.path):
            spec.loader.exec_module(gen)
        self.assertEqual(gen.verify(PROFILE_JSON), [])
        with _tmpdir() as td:
            out = os.path.join(td, "qsa_page4_gfx1250_v1.json")
            self.assertEqual(gen.main(["emit", "--out", out]), 0)
            with open(out, encoding="utf-8") as fh:
                emitted = json.load(fh)
            # identical modulo the fields a GPU `check` restamps (golden / notes / generated_by)
            self.assertEqual(gen._strip_dynamic(emitted), gen._strip_dynamic(self.data))
            if self.data["generated_by"].get("mode") == "emit":
                self.assertEqual(emitted, self.data)
            cap = os.path.join(td, "cap.json")
            inside = {
                "cache_dtype": "bf16",
                "rows": 128,
                "requests": 32,
                "columns": 16384,
                "table_width": 16384,
            }
            outside = {
                "cache_dtype": "bf16",
                "rows": 16,
                "requests": 1,
                "columns": 4,
                "table_width": 16384,
            }
            with open(cap, "w") as fh:
                json.dump({"plan": [inside, outside]}, fh)
            out2 = os.path.join(td, "qsa_extra_v1.json")
            # default: a rule-admitted capture entry becomes a sample of that rule, no row
            self.assertEqual(gen.main(["emit", "--out", out2, "--plan-file", cap]), 0)
            prof = PR.load_profile(out2, use_cache=False)
            self.assertEqual(len(prof.rows), len(self.data["rows"]))
            dec = next(r for r in prof.rules if r.id == "decode_bf16")
            self.assertIn(
                (128, 32, 16384, 16384), {PR.sample_key(s) for s in dec.raw["samples"]}
            )
            self.assertFalse(
                dec.verified
            )  # new samples -> rule pending until GPU check
            self.assertEqual(gen.verify(out2, capture_plan=[inside, outside]), [])
            self.assertTrue(gen.verify(out2))  # drift without the plan file
            # --plan-rows: entries no rule admits become (pending) exact rows
            out3 = os.path.join(td, "qsa_extra2_v1.json")
            self.assertEqual(
                gen.main(["emit", "--out", out3, "--plan-file", cap, "--plan-rows"]), 0
            )
            prof = PR.load_profile(out3, use_cache=False)
            new = [r for r in prof.rows if r.cell == "decode_bf16_r16q1_c4_w16384"]
            self.assertEqual(len(new), 1)
            self.assertFalse(new[0].verified)  # captured rows start pending
            self.assertFalse(
                any(r.cell.startswith("decode_bf16_r128q32") for r in prof.rows)
            )
        with self.assertRaises(ValueError):
            gen.make_row(
                {
                    "cache_dtype": "bf16",
                    "rows": 0,
                    "requests": 1,
                    "columns": 64,
                    "table_width": 16,
                }
            )

    def test_check_restamp_cpu(self):
        """`gen_profile check` restamp logic (GPU row check mocked): file-mode standalone API
        load, every row -> pass on the profiled kernel sha, result loads and verifies.
        """
        spec = importlib.util.spec_from_file_location(
            "qsa_gen_profile_check", os.path.join(PKG_DIR, "gen_profile.py")
        )
        gen = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [PKG_DIR] + sys.path):
            spec.loader.exec_module(gen)
        seen = []

        def fake_row(row, api, G, S, seed=0, profile=None):
            self.assertTrue(hasattr(api, "decide"))
            seen.append(row["cell"])
            return {"pass_": True, "decision": "ok", "vid": row["vid"]}

        def fake_sample(rule, s, api, G, S, seed=0, profile=None):
            seen.append(rule["id"])
            out = {k: s[k] for k in ("rows", "requests", "columns", "table_width")}
            return dict(out, pass_=True, decision="ok", vid=rule["vid"])

        def fake_bucket(b, s, api, G, S, seed=0, profile=None):
            seen.append(b["id"])
            out = {k: s[k] for k in ("rows", "requests", "columns", "table_width")}
            return dict(
                out, pass_=True, decision="ok", vid=b["vid"], bitwise_unpadded=True
            )

        props = types.SimpleNamespace(gcnArchName="gfx1250", multi_processor_count=256)
        with _tmpdir() as td:
            out = os.path.join(td, "qsa_page4_gfx1250_v1.json")
            with mock.patch.object(gen, "check_row", fake_row), mock.patch.object(
                gen, "check_sample", fake_sample
            ), mock.patch.object(
                gen, "check_bucket_sample", fake_bucket
            ), mock.patch.object(
                gen, "_golden_modules", lambda: (None, None)
            ), mock.patch.object(
                torch.cuda, "get_device_properties", lambda i: props
            ), mock.patch.dict(
                os.environ, {}
            ), mock.patch.object(
                torch.cuda, "empty_cache", lambda: None
            ):
                self.assertEqual(gen.check(PROFILE_JSON, out, None), 0)
            n_samples = sum(len(r["samples"]) for r in self.data["rules"])
            n_samples += sum(len(b["samples"]) for b in self.data.get("buckets") or [])
            self.assertEqual(len(seen), len(self.data["rows"]) + n_samples)
            prof = PR.load_profile(out, use_cache=False)
            self.assertTrue(prof.buckets and all(b.verified for b in prof.buckets))
            self.assertTrue(all(r.verified for r in prof.rows))
            self.assertTrue(prof.rules and all(r.verified for r in prof.rules))
            for r in prof.rules:
                g = r.raw["golden"]
                self.assertEqual(
                    (g["samples_n"], g["samples_failed"]), (len(r.raw["samples"]), 0)
                )
                self.assertEqual(g["kernel_sha256"], self.data["kernel"]["sha256"])
            for r in prof.rows:
                g = r.raw["golden"]
                self.assertEqual(g["kernel_sha256"], self.data["kernel"]["sha256"])
                self.assertNotIn("isa_identity", g)
            self.assertEqual(gen.verify(out), [])
            fp8 = next(
                r.raw for r in prof.rows if r.cell == "decode_fp8_r64q16_c65536_w16384"
            )
            # phase-1 provenance is kept when the source row carried it (pre-restamp profile)
            if "phase1_kernel_sha256" in fp8["golden"]:
                self.assertEqual(
                    fp8["golden"]["phase1_kernel_sha256"],
                    "30c6213d492ad50a9e063456628737ca993060d73dfa4ead248558ba5abb121c",
                )
            self.assertEqual(fp8["notes"], "")

    def _write(self, td, data, name="qsa_bad_v1"):
        path = os.path.join(td, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        return path

    def test_invalid_profiles(self):
        with _tmpdir() as td:
            cases = []
            d = json.loads(json.dumps(self.data))
            d["profile"] = "qsa_bad_v1"
            del d["kernel_env"]
            cases.append(d)
            d = json.loads(json.dumps(self.data))
            d["profile"] = "qsa_bad_v1"
            d["schema_version"] = 99
            cases.append(d)
            d = json.loads(json.dumps(self.data))
            d["profile"] = "qsa_bad_v1"
            d["rows"][0]["columns"] = 1234
            cases.append(d)  # sig no longer matches
            d = json.loads(json.dumps(self.data))
            d["profile"] = "qsa_bad_v1"
            d["rows"].append(d["rows"][0])
            cases.append(d)  # duplicate sig
            d = json.loads(json.dumps(self.data))
            d["profile"] = "qsa_bad_v1"
            d["rows"][0]["vid"] = "qsa.page4.decg.v2"
            cases.append(d)
            d = json.loads(json.dumps(self.data))
            d["profile"] = "qsa_bad_v1"
            d["rows"][0]["params"] = dict(d["rows"][0]["params"], wave_cols=64)
            cases.append(d)
            d = json.loads(json.dumps(self.data))  # name != file stem
            cases.append(d)
            for i, bad in enumerate(cases):
                path = self._write(td, bad)
                with self.assertRaises(PR.ProfileError, msg=str(i)) as cm:
                    PR.load_profile(path, use_cache=False)
                self.assertEqual(cm.exception.reason, PR.Reason.PROFILE_INVALID)
            with self.assertRaises(PR.ProfileError) as cm:
                PR.load_profile("qsa_no_such_profile_v9")
            self.assertEqual(cm.exception.reason, PR.Reason.PROFILE_NOT_FOUND)

    def test_kernel_env_defaults_mirror_kernel_source(self):
        with open(PR.KERNEL_PATH, encoding="utf-8") as fh:
            src = fh.read()
        found = {}
        for node in ast.walk(ast.parse(src)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and ast.unparse(node.func.value) == "os.environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("QSA_")
            ):
                key = node.args[0].value
                dflt = ast.unparse(node.args[1]) if len(node.args) > 1 else None
                found.setdefault(key, set()).add(dflt)
        # every import-time switch that changes code selection is pinned by the profile
        ignore = {"QSA_TRACE_ROWS_PER_BLOCK"}
        self.assertEqual(set(found) - ignore, set(PR.KERNEL_ENV_DEFAULTS))
        for key, val in PR.KERNEL_ENV_DEFAULTS.items():
            sym = PR.KERNEL_ENV_SYMBOLIC_DEFAULTS.get(key)
            want = sym if sym else repr(val)
            self.assertEqual(found[key], {want}, key)
            if sym and key != "QSA_FP8_K_LOAD":
                self.assertEqual(
                    getattr(K, sym), val, key
                )  # the constant's value is pinned
        # the env snapshot of the injected (clean-env) kernel matches its resolved switches
        self.assertEqual(
            (
                K._DECODE_GROUP,
                K._DECODE_GROUP_COLS,
                K._DRAFT_PATH,
                K._ACCESS,
                K._BF16_K_LOAD,
                K._FP8_K_LOAD,
                K.TDM_PERSIST,
                K.TDM_PERSIST_CHUNKS,
                K.TDM_R16_NOBAR,
                K._BF16_R64,
                K._FP8_CVT,
                K.R16_TDM_COLS,
                K._FP8_CVT_ASM,
                K.draft_fp8_sched(),
                K.decg_fp8_sched(),
                K._DECODE_GROUP_AUTO_COLS,
            ),
            (
                "all",
                None,
                True,
                "raw",
                "auto",
                "auto",
                True,
                8,
                True,
                False,
                "fast",
                32,
                False,
                "qpre",
                "base",
                {False: 32, True: 64},
            ),
        )
        self.assertEqual(
            PR.DECG_WAVE_COLS,
            {
                "bf16": K._DECODE_GROUP_AUTO_COLS[False],
                "fp8": K._DECODE_GROUP_AUTO_COLS[True],
            },
        )

    def test_effective_env_inheritance(self):
        e = PR.effective_kernel_env({"QSA_BF16_K_LOAD": "RAW"})
        self.assertEqual((e["QSA_BF16_K_LOAD"], e["QSA_FP8_K_LOAD"]), ("raw", "raw"))
        e = PR.effective_kernel_env(
            {
                "QSA_DECODE_GROUP_COLS": " AUTO ",
                "QSA_DRAFT_FP8_SCHED": "QPRE",
                "QSA_DECG_FP8_SCHED": "Base",
            }
        )
        self.assertEqual(e, PR.KERNEL_ENV_DEFAULTS)  # kernel lower()s these: same build
        for key, val in (
            ("QSA_FP8_CVT_ASM", "1"),
            ("QSA_DECODE_GROUP_COLS", "32"),
            ("QSA_DRAFT_FP8_SCHED", "kpipe"),
            ("QSA_DECG_FP8_SCHED", "kpipe"),
        ):
            self.assertNotEqual(
                PR.effective_kernel_env({key: val})[key], PR.KERNEL_ENV_DEFAULTS[key]
            )
        self.assertEqual(PR.effective_kernel_env({}), PR.KERNEL_ENV_DEFAULTS)


# ---------------------------------------------------------------------------
class OffByDefault(unittest.TestCase):
    def test_disabled_without_profile(self):
        with mock.patch.dict(os.environ, _CLEAN, clear=True), mock.patch.object(
            API, "_KERNEL", None
        ), mock.patch.object(
            API, "_device_info", side_effect=AssertionError("device query")
        ), mock.patch.object(
            API, "_load_kernel", side_effect=AssertionError("kernel import")
        ):
            args = call_args()
            d = API.decide(*args)
            self.assertEqual((d.ok, d.reason.value), (False, "disabled"))
            # API 1.2: no profile => aiter's native path serves the call, reason visible
            self.assertEqual((d.path, d.vid), (API.PATH_NATIVE, API.NATIVE_VID))
            self.assertEqual(API.eligibility(*args), (False, "disabled"))
            with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
                API.qsa_paged_mqa_logits(*args, strict=True)
            self.assertEqual(cm.exception.reason, PR.Reason.DISABLED)
            out, d = API.try_qsa_paged_mqa_logits(*args)
            self.assertIsNone(out)
            self.assertEqual(API.layout_eligibility(args[1]).reason, PR.Reason.DISABLED)
            self.assertIsNone(API.selected_profile())

    def test_env_enables(self):
        env = dict(_CLEAN, **{PR.PROFILE_ENV: PR.DEFAULT_PROFILE})
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(API.selected_profile(), PR.DEFAULT_PROFILE)
        with mock.patch.dict(
            os.environ, dict(_CLEAN, **{PR.PROFILE_ENV: "nope_v0"}), clear=True
        ):
            self.assertEqual(
                API.decide(*call_args()).reason, PR.Reason.PROFILE_NOT_FOUND
            )

    def test_import_has_no_side_effects(self):
        code = (
            "import importlib.util, sys\n"
            f"d = {PKG_DIR!r}\n"
            "spec = importlib.util.spec_from_file_location('qsa_isc_sub', d + '/__init__.py',"
            " submodule_search_locations=[d])\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['qsa_isc_sub'] = m\n"
            "spec.loader.exec_module(m)\n"
            "bad = [k for k in sys.modules if k == 'flydsl' or k.startswith('flydsl.')"
            " or k == 'aiter' or 'qsa_paged_mqa_logits_page4' in k or k == 'triton']\n"
            "assert not bad, bad\n"
            "import torch; assert not torch.cuda.is_initialized()\n"
            "print('NO_SIDE_EFFECTS')\n"
        )
        env = dict(_CLEAN)
        r = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=300,
        )
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("NO_SIDE_EFFECTS", r.stdout)


# ---------------------------------------------------------------------------
class AbiDrift(unittest.TestCase):
    def _atom_def(self):
        path = os.path.join(_golden_dir(), ATOM_REL)
        with open(path, "rb") as fh:
            raw = fh.read()
        tree = ast.parse(raw.decode("utf-8"))
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "qsa_paged_mqa_logits"
        )
        return raw, fn

    def test_atom_source_frozen(self):
        raw, _ = self._atom_def()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            API.ATOM_SCORER_SOURCE_SHA256,
            "ATOM scorer source changed: re-review ATOM_SCORER_SIGNATURE",
        )

    def test_atom_signature_matches_frozen(self):
        _, fn = self._atom_def()
        a = fn.args
        self.assertEqual(
            (a.posonlyargs, a.vararg, a.kwarg, a.kwonlyargs), ([], None, None, [])
        )
        names = [x.arg for x in a.args]
        defaults = [inspect.Parameter.empty] * (len(names) - len(a.defaults)) + [
            ast.literal_eval(d) for d in a.defaults
        ]
        self.assertEqual(tuple(zip(names, defaults)), API.ATOM_SCORER_SIGNATURE)
        ret = ast.unparse(fn.returns)
        self.assertEqual(ret, "tuple[torch.Tensor, torch.Tensor]")

    def _check_ours(self, fn, extra_kwonly):
        params = list(inspect.signature(fn).parameters.values())
        lead = params[: len(API.ATOM_SCORER_SIGNATURE)]
        self.assertEqual(
            tuple((p.name, p.default) for p in lead), API.ATOM_SCORER_SIGNATURE
        )
        self.assertTrue(
            all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in lead)
        )
        rest = params[len(lead) :]
        self.assertTrue(all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in rest))
        self.assertEqual([p.name for p in rest], extra_kwonly)
        # API 1.2: decide(native=True) prepares aiter's native path by default
        self.assertTrue(
            all(
                p.default is None
                or p.default is False
                or (p.name, p.default) == ("native", True)
                for p in rest
            )
        )

    def test_our_signatures(self):
        self._check_ours(
            API.qsa_paged_mqa_logits, ["logits", "visible_groups", "profile", "strict"]
        )
        self._check_ours(API.eligibility, ["logits", "visible_groups", "profile"])
        self._check_ours(
            API.decide,
            [
                "logits",
                "visible_groups",
                "profile",
                "allow_unverified",
                "buckets",
                "native",
            ],
        )
        # AITER-R3: the package is private (no re-export); the public op is aiter.ops.qsa
        self.assertFalse(hasattr(PKG, "qsa_paged_mqa_logits"))

    def test_api_1_1_signatures(self):
        """1.1 is additive: launch = (decision, <ATOM 12>, *, logits, visible_groups)."""
        params = list(inspect.signature(API.launch).parameters.values())
        self.assertEqual(
            (params[0].name, params[0].kind),
            ("decision", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        )
        self.assertIs(params[0].default, inspect.Parameter.empty)
        lead = params[1 : 1 + len(API.ATOM_SCORER_SIGNATURE)]
        self.assertEqual(
            tuple((p.name, p.default) for p in lead), API.ATOM_SCORER_SIGNATURE
        )
        self.assertTrue(
            all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in lead)
        )
        rest = params[1 + len(lead) :]
        self.assertEqual(
            [(p.name, p.kind, p.default) for p in rest],
            [
                ("logits", inspect.Parameter.KEYWORD_ONLY, None),
                ("visible_groups", inspect.Parameter.KEYWORD_ONLY, None),
            ],
        )
        sig = inspect.signature(API.prewarm)
        self.assertEqual(
            [(p.name, p.kind) for p in sig.parameters.values()],
            [
                ("profile", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                ("cache_dtypes", inspect.Parameter.KEYWORD_ONLY),
                ("pos_dtypes", inspect.Parameter.KEYWORD_ONLY),
                ("device", inspect.Parameter.KEYWORD_ONLY),
                # R2 (additive, keyword-only): scoping, persistent cache, jobs
                ("page_sizes", inspect.Parameter.KEYWORD_ONLY),
                ("compressed_k_cache", inspect.Parameter.KEYWORD_ONLY),
                ("compressed_k_scale", inspect.Parameter.KEYWORD_ONLY),
                ("cache_dir", inspect.Parameter.KEYWORD_ONLY),
                ("jobs", inspect.Parameter.KEYWORD_ONLY),
                ("compile_only", inspect.Parameter.KEYWORD_ONLY),
            ],
        )
        self.assertEqual(
            {k: v.default for k, v in sig.parameters.items()},
            {
                "profile": None,
                "cache_dtypes": ("bf16",),
                "pos_dtypes": (torch.int32, torch.int64),
                "device": None,
                "page_sizes": None,
                "compressed_k_cache": None,
                "compressed_k_scale": None,
                "cache_dir": None,
                "jobs": 1,
                "compile_only": False,
            },
        )
        self.assertTrue(callable(API.launch) and callable(API.prewarm))
        # Decision keeps its 1.0 public fields (the plan is private, not compared); API 1.2
        # adds ``path`` (fused / native / invalid)
        import dataclasses

        self.assertEqual(
            [f.name for f in dataclasses.fields(API.Decision)],
            [
                "ok",
                "reason",
                "detail",
                "profile",
                "cell",
                "vid",
                "columns",
                "path",
                "_plan",
            ],
        )

    def test_reason_enum_frozen(self):
        self.assertEqual(tuple(PR.iter_reasons()), FREEZE_REASONS)
        self.assertEqual(
            (PR.API_VERSION, PR.PROFILE_ENV, PR.DEFAULT_PROFILE),
            ("1.2", "AITER_QSA_INDEX_SCORER_PROFILE", "qsa_page4_gfx1250_v1"),
        )
        # profiles stay compatible: major match, generated profiles record the 1.0 floor
        self.assertEqual(PR.PROFILE_API_VERSION, "1.0")
        with open(PROFILE_JSON, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["api_version"], "1.0")

    def test_kernel_module_api_kept(self):
        with open(PR.KERNEL_PATH, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "qsa_paged_mqa_logits_page4"
        )
        self.assertEqual(
            [x.arg for x in fn.args.args],
            [
                "q",
                "compressed_k_cache",
                "page_table",
                "token_to_request",
                "query_positions",
                "context_lens",
                "compress_ratio",
                "divisor",
                "max_columns",
                "compressed_k_scale",
                "logits",
                "visible_groups",
                "row_starts",
            ],
        )
        self.assertTrue(callable(K.qsa_paged_mqa_logits_page4))
        # the op passes the kernel exactly the knobs it understands
        src = inspect.getsource(API._launch)
        for kw in (
            "compress_ratio=",
            "divisor=score_divisor",
            "max_columns=max_columns",
            "compressed_k_scale=",
            "logits=logits",
            "visible_groups=",
            "row_starts=",
        ):
            self.assertIn(kw, src)

    def test_default_divisor_equals_atom(self):
        self.assertEqual(API._DEFAULT_DIVISOR, 128**0.5)


# ---------------------------------------------------------------------------
class Reasons(Env):
    def test_ok_every_verified_row(self):
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        for row in prof.rows:
            r = row.raw
            args = call_args(
                rows=r["rows"]["min"],
                requests=r["requests"]["min"],
                width=r["table_width"],
                columns=r["columns"],
                fmt=r["cache_dtype"],
            )
            d = self.decide(args)
            if row.verified:
                self.assertTrue(d.ok, (row.cell, d))
                self.assertEqual(
                    (d.cell, d.vid, d.columns), (row.cell, row.vid, r["columns"])
                )
                ok, reason = d
                self.assertEqual((ok, reason), (True, "ok"))
                self.assertEqual(
                    API.eligibility(*args, profile=self.profile), (True, "ok")
                )
                # int32 positions and full-width (max_columns=None) variants of the call
                a32 = call_args(
                    rows=r["rows"]["min"],
                    requests=r["requests"]["min"],
                    width=r["table_width"],
                    columns=r["columns"],
                    fmt=r["cache_dtype"],
                    qp64=False,
                )
                self.assertTrue(self.decide(a32).ok)
            else:
                self.assertEqual(d.reason, PR.Reason.ROW_UNVERIFIED, row.cell)
                self.assertTrue(self.decide(args, allow_unverified=True).ok)

    def test_tensor_reasons(self):
        base = call_args()

        def mut(i, v):
            a = list(base)
            a[i] = v
            return a

        self.assertReason(mut(6, 16), "compress_ratio_unsupported")
        self.assertReason(mut(0, base[0].float()), "q_dtype_unsupported")
        self.assertReason(
            mut(0, torch.zeros(64, 8, 128, dtype=torch.bfloat16)), "q_shape_unsupported"
        )
        self.assertReason(
            mut(0, torch.zeros(64, 4, 64, dtype=torch.bfloat16)), "q_shape_unsupported"
        )
        self.assertReason(
            mut(0, aligned((4, 64, 128), torch.bfloat16).transpose(0, 1)),
            "q_layout_unsupported",
        )
        self.assertReason(
            mut(0, torch.zeros(0, 4, 128, dtype=torch.bfloat16)), "empty_call"
        )
        # cache formats: mxfp8 / fp4 / bf16+scale / fp8 without scale / fp16
        fp8 = aligned((8, 4, 1, 128), torch.float8_e4m3fn)
        a = mut(1, fp8)
        a[11] = torch.zeros(8, 4, 1, 4, dtype=torch.uint8)
        self.assertIn("mxfp8", self.assertReason(a, "cache_format_unsupported").detail)
        a = mut(1, aligned((8, 4, 1, 64), torch.uint8))
        a[11] = torch.zeros(8, 4, 1, 4, dtype=torch.uint8)
        self.assertIn("fp4", self.assertReason(a, "cache_format_unsupported").detail)
        a = list(base)
        a[11] = torch.ones(8, 4, 1)
        self.assertReason(a, "cache_format_unsupported")
        self.assertReason(mut(1, fp8), "cache_format_unsupported")
        self.assertReason(
            mut(1, aligned((8, 4, 1, 128), torch.float16)), "cache_format_unsupported"
        )
        # ROUND-PAGE16 / R2: P = 8, 16 are supported geometries (page-8/16 rules); other P are not
        self.assertReason(
            mut(1, aligned((8, 2, 1, 128), torch.bfloat16)), "cache_shape_unsupported"
        )
        self.assertReason(
            mut(1, aligned((8, 32, 1, 128), torch.bfloat16)), "cache_shape_unsupported"
        )
        self.assertReason(
            mut(1, aligned((8, 4, 2, 128), torch.bfloat16)[:, :, :1]),
            "cache_layout_unsupported",
        )
        for off in (16, 64):
            self.assertReason(
                mut(1, aligned((8, 4, 1, 128), torch.bfloat16, offset=off)),
                "cache_alignment_unsupported",
            )
        a = call_args(fmt="fp8")
        a[11] = torch.ones(8, 4, 2)[:, :, :1]
        self.assertReason(a, "scale_layout_unsupported")
        a = call_args(fmt="fp8")
        a[11] = torch.ones(7, 4, 1)
        self.assertReason(a, "scale_layout_unsupported")
        # metadata
        self.assertReason(mut(2, base[2].to(torch.int16)), "metadata_unsupported")
        self.assertReason(mut(2, base[2].t().contiguous().t()), "metadata_unsupported")
        self.assertReason(
            mut(2, torch.zeros(16, 0, dtype=torch.int32)), "metadata_unsupported"
        )
        self.assertReason(mut(3, base[3][:-1]), "metadata_unsupported")
        self.assertReason(mut(3, base[3].to(torch.int16)), "metadata_unsupported")
        self.assertReason(mut(4, base[4].view(-1, 1)), "metadata_unsupported")
        self.assertReason(
            mut(4, torch.zeros(128, dtype=torch.int64)[::2]), "metadata_unsupported"
        )
        self.assertReason(mut(5, base[5][:-1]), "metadata_unsupported")
        self.assertReason(
            mut(3, torch.zeros(64, dtype=torch.int32, device="meta")), "device_mismatch"
        )
        # divisor / max_columns
        self.assertReason(mut(7, 2.0), "divisor_uncovered")
        self.assertTrue(self.decide(mut(7, 128**0.5)).ok)
        self.assertReason(mut(8, 0), "max_columns_invalid")
        self.assertReason(mut(8, -5), "max_columns_invalid")
        self.assertTrue(self.decide(mut(10, True)).ok)  # allow_flydsl is ignored
        # outputs
        self.assertReason(
            base,
            "output_layout_unsupported",
            logits=torch.empty(64, 1024, dtype=torch.float32),
        )
        self.assertReason(
            base,
            "output_layout_unsupported",
            visible_groups=torch.empty(64, dtype=torch.int64),
        )
        self.assertReason(
            mut(9, torch.empty(63, dtype=torch.int32)), "output_layout_unsupported"
        )
        self.assertTrue(
            self.decide(
                mut(9, torch.empty(64, dtype=torch.int32)),
                logits=torch.empty(64, 65536),
                visible_groups=torch.empty(64, dtype=torch.int32),
            ).ok
        )
        # plane caps (logits rows*columns*4 > INT32_MAX)
        self.assertReason(call_args(rows=8192, requests=1), "plane_too_large")

    def test_device_reasons(self):
        with mock.patch.object(API, "_ALLOWED_DEVICE_TYPES", ("cuda",)):
            self.assertReason(call_args(), "device_not_gpu")
            d = API.layout_eligibility(call_args()[1], profile=self.profile)
            self.assertEqual(d.reason, PR.Reason.DEVICE_NOT_GPU)
        with mock.patch.object(API, "_device_info", lambda dev: ("gfx950", 256)):
            self.assertReason(call_args(), "arch_unsupported")
        with mock.patch.object(API, "_device_info", lambda dev: ("gfx1250", 128)):
            d = self.assertReason(call_args(), "cu_unsupported")
            self.assertTrue(d.static)

    def test_build_reasons(self):
        env = dict(PR.effective_kernel_env(_CLEAN), QSA_DECODE_GROUP="0")
        with mock.patch.object(API, "_KERNEL_ENV", env):
            d = self.assertReason(call_args(), "kernel_env_mismatch")
            self.assertIn("QSA_DECODE_GROUP", d.detail)
        with mock.patch.object(API, "_KERNEL", None), mock.patch.object(
            API, "_KERNEL_ENV", None
        ), mock.patch.dict(os.environ, {"QSA_DECODE_GROUP_COLS": "64"}):
            self.assertReason(call_args(), "kernel_env_mismatch")
        with mock.patch.dict(API._KERNEL_SHA, {}, clear=True), mock.patch.object(
            API, "_kernel_sha256", lambda: "0" * 64
        ):
            self.assertReason(call_args(), "profile_kernel_mismatch")
        with mock.patch.object(API, "_KERNEL", None), mock.patch.object(
            API, "_load_kernel", side_effect=ImportError("no flydsl")
        ):
            self.assertReason(call_args(), "kernel_unavailable")
        for key, val in (
            ("QSA_FP8_CVT_ASM", "1"),
            ("QSA_DRAFT_FP8_SCHED", "kpipe"),
            ("QSA_DECG_FP8_SCHED", "kpipe"),
            ("QSA_DECODE_GROUP_COLS", "32"),
        ):
            env = dict(PR.effective_kernel_env(_CLEAN), **{key: val})
            with mock.patch.object(API, "_KERNEL_ENV", env):
                d = self.assertReason(call_args(fmt="fp8"), "kernel_env_mismatch")
                self.assertIn(key, d.detail)
        with mock.patch.object(K, "decode_group_route", lambda *a: 64):
            self.assertReason(call_args(), "route_mismatch")  # bf16 profiled w32
        # fp8 rows: w64 + sched base + intrinsic pk8 are part of the row identity
        fp8 = call_args(fmt="fp8")
        self.assertTrue(self.decide(fp8, allow_unverified=True).ok)
        for patch in (
            mock.patch.object(K, "decode_group_route", lambda *a: 32),
            mock.patch.object(K, "decg_fp8_sched", lambda: "kpipe"),
            mock.patch.object(K, "_FP8_CVT_ASM", True),
        ):
            with patch:
                self.assertReason(fp8, "route_mismatch", allow_unverified=True)
        draft8 = call_args(rows=16, requests=16, fmt="fp8")
        self.assertTrue(self.decide(draft8, allow_unverified=True).ok)
        with mock.patch.object(K, "draft_fp8_sched", lambda: "base"):
            self.assertReason(draft8, "route_mismatch", allow_unverified=True)

    def test_coverage_reasons(self):
        self.assertReason(
            call_args(rows=600, requests=600), "rows_rule_uncovered"
        )  # > draft rule
        self.assertReason(
            call_args(rows=2052, requests=513), "rows_rule_uncovered"
        )  # > decode rule
        self.assertReason(
            call_args(rows=64, requests=16, width=32768, columns=32768),
            "table_width_uncovered",
        )  # > rule width
        self.assertReason(call_args(columns=1000), "columns_uncovered")  # not pow2
        self.assertReason(call_args(columns=32), "columns_uncovered")  # < 64
        self.assertTrue(
            self.decide(call_args(columns=None)).ok
        )  # full width = 65536 bucket
        # pending rows -> row_unverified, pass rows -> ok (holds before and after the
        # root's GPU `gen_profile check` restamps the profile)
        status = {
            r.cell: r.raw["golden"]["status"]
            for r in PR.load_profile(self.profile).rows
        }
        n_pending = 0
        for kw, cell in (
            (
                {"rows": 16, "requests": 16, "columns": 32768},
                "draft_bf16_r16q16_c32768_w16384",
            ),
            ({"fmt": "fp8"}, "decode_fp8_r64q16_c65536_w16384"),
            (
                {"fmt": "fp8", "rows": 16, "requests": 16},
                "draft_fp8_r16q16_c65536_w16384",
            ),
            (
                {"fmt": "fp8", "rows": 512, "requests": 2},
                "prefill_fp8_r512q2_c65536_w16384",
            ),
        ):
            if status[cell] == "pending":
                d = self.assertReason(call_args(**kw), "row_unverified")
                n_pending += 1
            else:
                d = self.decide(call_args(**kw))
                self.assertTrue(d.ok, d)
            self.assertEqual(d.cell, cell)
            self.assertTrue(self.decide(call_args(**kw), allow_unverified=True).ok)
        if n_pending == 0 and any(s == "pending" for s in status.values()):
            # still exercise the reason on whatever row is pending
            row = next(
                r for r in PR.load_profile(self.profile).rows if not r.verified
            ).raw
            self.assertReason(
                call_args(
                    rows=row["rows"]["min"],
                    requests=row["requests"]["min"],
                    width=row["table_width"],
                    columns=row["columns"],
                    fmt=row["cache_dtype"],
                ),
                "row_unverified",
            )

    def test_static_reason_set(self):
        self.assertIn(PR.Reason.CACHE_ALIGNMENT, PR.STATIC_REASONS)
        self.assertNotIn(PR.Reason.COLUMNS, PR.STATIC_REASONS)
        d = API.layout_eligibility(call_args()[1], profile=self.profile)
        self.assertTrue(d.ok, d)
        d = API.layout_eligibility(call_args(cache_offset=16)[1], profile=self.profile)
        self.assertEqual(d.reason, PR.Reason.CACHE_ALIGNMENT)


# ---------------------------------------------------------------------------
def _rules_requests_pending() -> bool:
    """RULES-REQUESTS: the 21 prefill rules (requests 1..32 -> 1..512) are ``pending`` in
    the committed profile until the GPU ``gen_profile check`` restamps them."""
    with open(PROFILE_JSON, encoding="utf-8") as fh:
        rules = json.load(fh)["rules"]
    return any(
        r["regime"] == "prefill" and r["golden"]["status"] == "pending" for r in rules
    )


RR_PENDING = _rules_requests_pending()


def _gen():
    spec = importlib.util.spec_from_file_location(
        f"qsa_gen_profile_rules{uuid.uuid4().hex[:6]}",
        os.path.join(PKG_DIR, "gen_profile.py"),
    )
    gen = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "path", [PKG_DIR] + sys.path):
        spec.loader.exec_module(gen)
    return gen


def _gpu_stamp_kernels(ksha):
    """Kernels whose GPU stamps count as on the profiled kernel ``ksha``: ``ksha`` itself
    and (AITER-LINT, black/ruff only) the R2E kernel the GPU check ran on, provided the
    record carries the AITER-LINT ISA-identity link (see :func:`_assert_lint_link`);
    AITER-R3: also the lint kernel c01cf365 the 2026-11-06 GPU check ran on (carried by the
    AITER-R3 cache-key link)."""
    gen = _gen()
    return (ksha, gen.R2E_KERNEL_SHA256, gen.LINT_KERNEL_SHA256)


def _assert_lint_link(case, g, ksha, label):
    """A pass stamped on an earlier chain kernel (not ``ksha``) carries the link(s) to it."""
    if g["kernel_sha256"] != ksha:
        case.assertIn("AITER-R3", g.get("isa_identity", ""), label)
        if g["kernel_sha256"] != _gen().LINT_KERNEL_SHA256:
            case.assertIn("AITER-LINT", g.get("isa_identity", ""), label)


def _verified_rules(data):
    """Copy of a profile with every rule stamped `pass` (as a GPU check would)."""
    d = json.loads(json.dumps(data))
    for r in d["rules"]:
        r["golden"] = {
            "status": "pass",
            "kernel_sha256": d["kernel"]["sha256"],
            "samples_n": len(r["samples"]),
            "checked": [dict(s, **{"pass": True}) for s in r["samples"]],
        }
    return d


class Rules(Env):
    """Schema-2 range rules: validation, precedence, bounds, reasons, route guard and the
    Stage-3 captured plan (rules stamped pass in a temp copy = a mocked GPU check)."""

    def setUp(self):
        super().setUp()
        with open(PROFILE_JSON, encoding="utf-8") as fh:
            self.data = json.load(fh)
        self._td = _tmpdir()
        self.td = self._td.__enter__()
        d = _verified_rules(self.data)
        d["profile"] = "qsa_rules_v1"
        self.vpath = os.path.join(self.td, "qsa_rules_v1.json")
        with open(self.vpath, "w") as fh:
            json.dump(d, fh)
        self.vdata = d

    def tearDown(self):
        self._td.__exit__(None, None, None)
        super().tearDown()

    def write(self, d, name="qsa_bad_v1"):
        d = json.loads(json.dumps(d))
        d["profile"] = name
        path = os.path.join(self.td, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(d, fh)
        return path

    def test_committed_rules(self):
        ids = {r["id"]: r for r in self.data["rules"]}
        self.assertEqual(self.data["schema_version"], 2)
        # ROUND-R2E route presets: bf16 lean ktile RB32 x C2 for columns <= 2048, persist
        # above (4096..8192, wide 16384..65536 under the ATOM chunk bound)
        p4 = {
            "draft_bf16",
            "decode_bf16",
            "tail_bf16",
            "prefill_bf16_kt",
            "prefill_bf16_persist",
            "prefill_bf16_persist_wide",
        }
        # fp8: lean+kconv ktile x C2 (rows <= 2048) / x C4 (rows 2049..16384) for columns <=
        # 2048, r32 fp8 above (no persistent fp8 kernel)
        f4 = {
            "draft_fp8",
            "decode_fp8",
            "tail_fp8",
            "prefill_fp8_kt",
            "prefill_fp8_kt_tall",
            "prefill_fp8_r32_mid",
            "prefill_fp8_r32_wide",
        }
        self.assertEqual(
            set(ids), {i + s for i in p4 | f4 for s in ("", "_p16", "_p8")}
        )
        gen = _gen()
        ksha = self.data["kernel"]["sha256"]
        # evidence carried to this kernel by the ISA-identity chain (GPU restamps on bf49 for
        # page 4 bf16, on d438 for page 8/16 and fp8; AITER-LINT: GPU check on R2E 58051973)
        gpu = _gpu_stamp_kernels(ksha)
        chain = (*gpu, gen.PAGE4_EVIDENCE_KERNEL_SHA256, gen.R2_KERNEL_SHA256)
        caps = {
            4: gen.capture_shapes(),
            16: gen.capture_shapes_p16(),
            8: gen.capture_shapes_p8(),
        }
        # RULES-REQUESTS: + the many-request / mixed-step prefill sources (7-tuples)
        rr_src = {4: gen.manyreq_shapes(4), 16: gen.manyreq_shapes(16)}
        for page, want in (
            (4, gen.builtin_sample_shapes()),
            (16, gen.builtin_sample_shapes_p16()),
        ):
            got = [k for k in caps[page] if k[0] == "bf16"]
            rr = [k for k in rr_src[page] if k[0] == "bf16"]
            self.assertEqual([k[1:] for k in got if k not in rr], want)
            self.assertTrue(all(k in got for k in rr))
        for r in self.data["rules"]:
            page = PR.page_of(r)
            shapes = [k[1:] for k in caps[page] if k[0] == r["cache_dtype"]]
            g = r["golden"]
            self.assertIn(g["status"], ("pass", "pending"))
            base = (
                r["id"].rsplit("_p", 1)[0]
                if r["id"].endswith(("_p8", "_p16"))
                else r["id"]
            )
            if g["status"] == "pass":
                self.assertIn(g["kernel_sha256"], chain, r["id"])
                if g["kernel_sha256"] in gpu:
                    _assert_lint_link(self, g, ksha, r["id"])
                else:
                    self.assertTrue(g.get("isa_identity"), r["id"])
                    # R2E re-routed / ISA-changed builds and the new fp8 draft wg_waves 2
                    # launch carry no older evidence: pass only on a fresh GPU check
                    self.assertNotIn(
                        base,
                        {
                            "prefill_bf16_kt",
                            "prefill_bf16_persist",
                            "prefill_bf16_persist_wide",
                            "prefill_fp8_kt",
                            "prefill_fp8_kt_tall",
                            "draft_fp8",
                        },
                        r["id"],
                    )
            elif "stale" in g:
                self.assertIn("ROUND-R2E", g["reason"], r["id"])
            # P8/P16 + fp8 non-prefill rules pass via the d438 GPU restamp chain
            if base in {
                "draft_bf16",
                "decode_bf16",
                "tail_bf16",
                "decode_fp8",
                "tail_fp8",
                "prefill_fp8_r32_wide",
            } and not (RR_PENDING and r["regime"] == "prefill"):
                self.assertEqual(g["status"], "pass", r["id"])
            # one variant per rule: the router at every column and at both row extremes;
            # launch knobs = the default-env wrapper's (fp8 draft wg_waves 2)
            R = PR.ProfileRule(r)
            vb, kn = PR.split_knobs(r["vid"], r["params"])
            self.assertEqual(
                kn, PR.default_knobs(r["cache_dtype"], r["regime"]), r["id"]
            )
            for c in PR.column_values(r["columns"]):
                for rr in sorted({int(r["rows"]["min"]), R.rows_max_at(c)}):
                    self.assertEqual(
                        (r["vid"], vb),
                        PR.expected_variant(
                            r["cache_dtype"], r["regime"], c, page, rows=rr
                        ),
                        (r["id"], c, rr),
                    )
            # every bounds sample and every captured / exact-row / AgentX class the rule
            # admits is a sample
            keys = {PR.sample_key(s) for s in r["samples"]}
            R = PR.ProfileRule(r)
            for k in shapes:
                self.assertEqual(R.admits(*k[:4]), tuple(k[:4]) in keys, (r["id"], k))
            self.assertEqual(r["samples"], gen.rule_samples(r, shapes))
        self.assertEqual(ids["draft_bf16"]["vid"], "qsa.page4.draft.v1")
        self.assertEqual(ids["tail_bf16"]["vid"], "qsa.page4.r16.v1")
        self.assertEqual(ids["prefill_bf16_persist"]["vid"], "qsa.page4.r32_persist.v1")
        # ROUND-R2E prefill column / row bands
        kt = ids["prefill_bf16_kt"]
        self.assertEqual(
            (kt["vid"], kt["params"], kt["columns"], kt["rows"]),
            (
                "qsa.page4.r32_kt.v1",
                {
                    "k_load": "tdm",
                    "rows_per_block": 32,
                    "kt_rows": 32,
                    "col_tiles": 2,
                    "lean": True,
                },
                {"pow2": {"min": 64, "max": PR.PREFILL_KT_MAX_COLUMNS}},
                {"min": 32, "max": PR.PREFILL_KT_MAX_ROWS},
            ),
        )
        self.assertEqual(
            ids["prefill_bf16_persist"]["columns"], {"pow2": {"min": 4096, "max": 8192}}
        )
        fkt = {
            "k_load": "tdm",
            "rows_per_block": 32,
            "kt_rows": 32,
            "cvt": "fast",
            "pk8": "intrin",
            "lean": True,
            "kconv": True,
        }
        for rid_, tiles, rows in (
            ("prefill_fp8_kt", 2, {"min": 32, "max": 2048}),
            ("prefill_fp8_kt_tall", 4, {"min": 2049, "max": 16384}),
        ):
            r = ids[rid_]
            self.assertEqual(
                (r["vid"], r["params"], r["rows"], r["columns"]),
                (
                    "qsa.page4.r32_kt_fp8.v1",
                    dict(fkt, col_tiles=tiles),
                    rows,
                    {"pow2": {"min": 64, "max": 2048}},
                ),
                rid_,
            )
        for rid_, cols in (
            ("prefill_fp8_r32_mid", {"min": 4096, "max": 8192}),
            ("prefill_fp8_r32_wide", {"min": 16384, "max": 65536}),
        ):
            self.assertEqual(
                (ids[rid_]["vid"], ids[rid_]["columns"]),
                ("qsa.page4.r32_fp8.v1", {"pow2": cols}),
                rid_,
            )
        self.assertEqual(ids["prefill_fp8_r32_wide"]["rows_x_columns_max"], 2**25)
        self.assertEqual(
            ids["draft_fp8"]["params"],
            {
                "k_load": "tdm",
                "wave_cols": 32,
                "sched": "qpre",
                "pk8": "intrin",
                "wg_waves": 2,
            },
        )
        self.assertEqual(
            ids["draft_bf16"]["params"], {"k_load": "tdm", "wave_cols": 32}
        )
        w = ids["prefill_bf16_persist_wide"]
        self.assertEqual(w["vid"], "qsa.page4.r32_persist.v1")
        self.assertEqual(w["rows_x_columns_max"], 2**25)
        # the rows bound is ATOM's select-loop chunk: rows_per_chunk = 128 MiB // (columns * 4)
        R = PR.ProfileRule(w)
        for c in PR.column_values(w["columns"]):
            self.assertEqual(R.rows_max_at(c), (128 * 1024 * 1024) // (c * 4))
        # every sample is inside the chunk bound; each column has its chunk-cap corner
        for s in w["samples"]:
            self.assertLessEqual(s["rows"] * s["columns"], 2**25)
        caps = {(s["rows"], s["columns"]) for s in w["samples"]}
        self.assertTrue({(2048, 16384), (1024, 32768), (512, 65536)} <= caps)
        # the profile-v1 exact bf16 prefill rows at 65536 columns are samples
        keys = {PR.sample_key(s) for s in w["samples"]}
        self.assertTrue({(512, 2, 65536, 16384), (320, 2, 65536, 16384)} <= keys)

    def test_agentx_select_loop_coverage(self):
        """Every scorer call ATOM's select loop issues for the AgentX recipe (bf16, table
        width 16384, <= 16384 batched tokens, requests 1..32, rows chunked to 2^25 / columns,
        pow2 columns 64..65536) resolves to a rule, and MTP3 decode / draft at CUDA-graph
        sizes 1..32 (+ eager batches up to 512 requests) stays inside draft/decode."""
        prof = PR.load_profile(self.vpath, use_cache=False)

        def rid(rows, reqs, cols, width=16384):
            m = PR.match(
                prof,
                arch="gfx1250",
                cu=256,
                cache_dtype="bf16",
                rows=rows,
                requests=reqs,
                columns=cols,
                table_width=width,
            )
            return (
                m.row.id
                if m.ok and isinstance(m.row, PR.ProfileRule)
                else (m.row.cell if m.ok else m.reason.value)
            )

        want = {"draft": "draft_bf16", "decode": "decode_bf16", "tail": "tail_bf16"}
        for c in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
            rpc = min(16384, (128 * 1024 * 1024) // (c * 4))
            for q in (1, 2, 3, 7, 15, 16, 17, 31, 32):
                rows_set = {
                    1,
                    2,
                    3,
                    4,
                    5,
                    31,
                    32,
                    33,
                    32 * q - 1,
                    32 * q,
                    32 * q + 1,
                    rpc // 2 - 1,
                    rpc // 2,
                    rpc - 1,
                    rpc,
                }
                for r in sorted(x for x in rows_set if 1 <= x <= rpc):
                    reg = PR.regime_of(r, q)
                    got = rid(r, q, c)
                    if reg == "prefill":
                        exp = (
                            "prefill_bf16_kt"
                            if c <= 2048
                            else (
                                "prefill_bf16_persist"
                                if c <= 8192
                                else "prefill_bf16_persist_wide"
                            )
                        )
                    else:
                        exp = want[reg]
                    if got != exp:  # an exact row (precedence) is fine
                        self.assertNotIn(
                            got, [x.value for x in PR.Reason], (r, q, c, got)
                        )
        # a call above the chunk cap is not admitted by the wide rule
        self.assertEqual(rid(1024, 1, 65536), "columns_uncovered")
        self.assertEqual(rid(4096, 1, 16384), "columns_uncovered")
        # MTP3 verify (rows == 4*bs) and draft (rows == bs) at 65536 columns
        for bs in list(range(1, 33)) + [64, 128, 256, 512]:
            rows = 4 * bs
            chunks = [min(512, rows - s) for s in range(0, rows, 512)]
            for r in chunks:
                self.assertIn(
                    rid(r, bs, 65536),
                    ("decode_bf16", "decode_bf16_r64q16_c65536_w16384", "draft_bf16"),
                    (bs, r),
                )
            self.assertIn(
                rid(bs, bs, 65536), ("draft_bf16", "draft_bf16_r16q16_c65536_w16384")
            )

    def test_pending_rules_are_never_selected(self):
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        if any(r.verified for r in prof.rules):
            self.skipTest("rules already GPU-verified")
        d = self.assertReason(call_args(rows=8, requests=8), "row_unverified")
        self.assertEqual(d.cell, "rule:draft_bf16")
        d = self.decide(call_args(rows=8, requests=8), allow_unverified=True)
        self.assertTrue(d.ok, d)
        self.assertEqual((d.cell, d.vid), ("rule:draft_bf16", "qsa.page4.draft.v1"))

    def test_exact_rows_take_precedence(self):
        d = self.decide(call_args(rows=16, requests=16), profile=self.vpath)
        self.assertTrue(d.ok, d)
        self.assertEqual(d.cell, "draft_bf16_r16q16_c65536_w16384")
        d = self.decide(call_args(rows=64, requests=16), profile=self.vpath)
        self.assertEqual(d.cell, "decode_bf16_r64q16_c65536_w16384")
        # a pending exact row does not block a verified rule
        dd = json.loads(json.dumps(self.vdata))
        row = next(
            r for r in dd["rows"] if r["cell"] == "draft_bf16_r16q16_c65536_w16384"
        )
        row["golden"] = {"status": "pending"}
        d = self.decide(
            call_args(rows=16, requests=16), profile=self.write(dd, "qsa_prec_v1")
        )
        self.assertTrue(d.ok, d)
        self.assertEqual(d.cell, "rule:draft_bf16")

    def test_rule_bounds_and_reasons(self):
        ok = [  # (kw, rule)
            ({"rows": 1, "requests": 1, "columns": 64, "width": 16}, "draft_bf16"),
            ({"rows": 512, "requests": 512}, "draft_bf16"),
            (
                {"rows": 3, "requests": 500, "columns": 2048, "width": 1024},
                "draft_bf16",
            ),
            ({"rows": 4, "requests": 1, "columns": 512}, "decode_bf16"),
            (
                {"rows": 2048, "requests": 512, "columns": 64, "width": 16},
                "decode_bf16",
            ),
            ({"rows": 124, "requests": 4, "columns": 256}, "decode_bf16"),
            ({"rows": 2, "requests": 1}, "tail_bf16"),
            (
                {"rows": 4095, "requests": 512, "columns": 1024, "width": 256},
                "tail_bf16",
            ),
            (
                {"rows": 32, "requests": 1, "columns": 64, "width": 16},
                "prefill_bf16_kt",
            ),
            ({"rows": 16384, "requests": 32, "columns": 256}, "prefill_bf16_kt"),
            # RULES-REQUESTS: prefill requests 1..512 (mixed prefill+decode steps)
            ({"rows": 2112, "requests": 33, "columns": 512}, "prefill_bf16_kt"),
            ({"rows": 16384, "requests": 512, "columns": 1024}, "prefill_bf16_kt"),
            (
                {"rows": 4096, "requests": 2, "columns": 2048, "width": 512},
                "prefill_bf16_kt",
            ),
            (
                {"rows": 64, "requests": 1, "columns": 4096, "width": 1024},
                "prefill_bf16_persist",
            ),
            (
                {"rows": 8192, "requests": 1, "columns": 4096, "width": 1024},
                "prefill_bf16_persist",
            ),
            ({"rows": 16384, "requests": 32, "columns": 8192}, "prefill_bf16_persist"),
            (
                {"rows": 2048, "requests": 32, "columns": 16384},
                "prefill_bf16_persist_wide",
            ),
            (
                {"rows": 1024, "requests": 1, "columns": 16384, "width": 4096},
                "prefill_bf16_persist_wide",
            ),
            (
                {"rows": 1024, "requests": 32, "columns": 32768},
                "prefill_bf16_persist_wide",
            ),
            (
                {"rows": 512, "requests": 16, "columns": 65536},
                "prefill_bf16_persist_wide",
            ),
            (
                {"rows": 32, "requests": 1, "columns": 65536},
                "prefill_bf16_persist_wide",
            ),
            # fp8 (R2E): draft wg_waves 2; ktile C2 rows <= 2048, C4 above; r32 fp8 > 2048 cols
            ({"rows": 8, "requests": 8, "fmt": "fp8"}, "draft_fp8"),
            (
                {"rows": 32, "requests": 1, "columns": 64, "width": 16, "fmt": "fp8"},
                "prefill_fp8_kt",
            ),
            (
                {
                    "rows": 2048,
                    "requests": 1,
                    "columns": 2048,
                    "width": 512,
                    "fmt": "fp8",
                },
                "prefill_fp8_kt",
            ),
            (
                {
                    "rows": 2080,
                    "requests": 1,
                    "columns": 256,
                    "width": 64,
                    "fmt": "fp8",
                },
                "prefill_fp8_kt_tall",
            ),
            (
                {
                    "rows": 16384,
                    "requests": 32,
                    "columns": 2048,
                    "width": 512,
                    "fmt": "fp8",
                },
                "prefill_fp8_kt_tall",
            ),
            (
                {
                    "rows": 64,
                    "requests": 1,
                    "columns": 4096,
                    "width": 1024,
                    "fmt": "fp8",
                },
                "prefill_fp8_r32_mid",
            ),
            (
                {"rows": 512, "requests": 16, "columns": 65536, "fmt": "fp8"},
                "prefill_fp8_r32_wide",
            ),
        ]
        for kw, rid in ok:
            d = self.decide(call_args(**kw), profile=self.vpath)
            self.assertTrue(d.ok, (kw, d))
            self.assertEqual(d.cell, f"rule:{rid}", kw)
        bad = [
            (
                {"rows": 513, "requests": 513},
                "rows_rule_uncovered",
            ),  # draft requests > 512
            (
                {"rows": 2052, "requests": 513},
                "rows_rule_uncovered",
            ),  # decode rows > 2048
            (
                {"rows": 4097, "requests": 512},
                "rows_rule_uncovered",
            ),  # tail rows > 4095
            (
                {"rows": 16416, "requests": 1, "columns": 512},
                "rows_rule_uncovered",
            ),  # prefill > 16384
            (
                {"rows": 16384, "requests": 513, "columns": 512},
                "rows_rule_uncovered",
            ),  # rows < 32*requests (not prefill) and above the decode / tail bands
            (
                {"rows": 8, "requests": 8, "width": 32768, "columns": 65536},
                "table_width_uncovered",
            ),
            (
                {"rows": 8, "requests": 8, "width": 8, "columns": 32},
                "table_width_uncovered",
            ),
            (
                {"rows": 64, "requests": 1, "width": 64, "columns": 256},
                "ok",
            ),  # kt rule, tw 64
            (
                {"rows": 64, "requests": 1, "width": 127, "columns": 512},
                "columns_uncovered",
            ),  # 508 cols
            (
                {"rows": 8, "requests": 8, "columns": 1000},
                "columns_uncovered",
            ),  # not a power of two
            ({"rows": 8, "requests": 8, "columns": 32}, "columns_uncovered"),  # < 64
            (
                {"rows": 4096, "requests": 1, "columns": 16384},
                "columns_uncovered",
            ),  # > 2^25 chunk
            (
                {"rows": 1028, "requests": 1, "columns": 32768},
                "columns_uncovered",
            ),  # > 2^25 chunk
            (
                {"rows": 516, "requests": 1, "columns": 65536},
                "columns_uncovered",
            ),  # > 2^25 chunk
            (
                {"rows": 1024, "requests": 1, "columns": 131072, "width": 32768},
                "table_width_uncovered",
            ),
            (
                {"rows": 16, "requests": 1, "columns": 4},
                "columns_uncovered",
            ),  # captured class
            (
                {"rows": 11, "requests": 1, "columns": 8},
                "columns_uncovered",
            ),  # captured class
        ]
        for kw, reason in bad:
            d = self.decide(call_args(**kw), profile=self.vpath)
            if reason == "ok":
                self.assertTrue(d.ok, (kw, d))
            else:
                self.assertFalse(d.ok, (kw, d))
                self.assertEqual(d.reason.value, reason, (kw, d.detail))

    def test_route_mismatch_guard(self):
        args = call_args(rows=8, requests=8)
        self.assertTrue(self.decide(args, profile=self.vpath).ok)
        with mock.patch.object(K, "draft_route", lambda *a: ("raw", 64)):
            d = self.assertReason(args, "route_mismatch", profile=self.vpath)
            self.assertEqual(d.cell, "rule:draft_bf16")
        # R2E: columns <= 2048 -> lean ktile rule; the legacy route table (persist >= 512
        # columns) or a non-lean ktile is a different build
        pre = call_args(rows=2048, requests=2, columns=512)
        d = self.decide(pre, profile=self.vpath)
        self.assertTrue(d.ok, d)
        self.assertEqual(
            (d.cell, d.vid), ("rule:prefill_bf16_kt", "qsa.page4.r32_kt.v1")
        )
        with mock.patch.object(K, "PREFILL_ROUTE", None):
            self.assertReason(pre, "route_mismatch", profile=self.vpath)
        with mock.patch.object(
            K, "PREFILL_ROUTE", K.parse_prefill_route("*,*,*,*,*,kt32x2")
        ):
            self.assertReason(pre, "route_mismatch", profile=self.vpath)
        per = call_args(rows=2048, requests=2, columns=4096, width=1024)
        d = self.decide(per, profile=self.vpath)
        self.assertEqual((d.ok, d.cell), (True, "rule:prefill_bf16_persist"), d)
        with mock.patch.object(K, "TDM_PERSIST", False):
            self.assertReason(per, "route_mismatch", profile=self.vpath)
        # fp8 prefill: lean+kconv ktile x C2 (rows <= 2048); legacy table -> kernel_r32
        pre8 = call_args(rows=2048, requests=2, columns=512, fmt="fp8")
        d = self.decide(pre8, profile=self.vpath)
        self.assertEqual((d.ok, d.cell), (True, "rule:prefill_fp8_kt"), d)
        with mock.patch.object(K, "FP8_PREFILL_ROUTE", None):
            self.assertReason(pre8, "route_mismatch", profile=self.vpath)
        with mock.patch.object(
            K, "FP8_PREFILL_ROUTE", K.parse_fp8_prefill_route("*,*,*,*,*,kt32x4lk")
        ):
            self.assertReason(pre8, "route_mismatch", profile=self.vpath)
        # fp8 draft: the rule pins wg_waves 2 (the plan launches the pinned knob, so an env
        # back at the default 4 still serves it); a different non-default env knob mismatches
        dr8 = call_args(rows=8, requests=8, fmt="fp8")
        d = self.decide(dr8, profile=self.vpath)
        self.assertEqual((d.ok, d.cell), (True, "rule:draft_fp8"), d)
        self.assertEqual(
            API.predict_route(K, "fp8", 8, 8, 65536, 0x1000)[1]["wg_waves"], 2
        )
        with mock.patch.object(K, "draft_wg_waves", lambda is_fp8: 4):
            self.assertTrue(self.decide(dr8, profile=self.vpath).ok)
        with mock.patch.object(K, "draft_wg_waves", lambda is_fp8: 1):
            self.assertReason(dr8, "route_mismatch", profile=self.vpath)
        tail = call_args(rows=13, requests=1, columns=512)
        self.assertTrue(self.decide(tail, profile=self.vpath).ok)
        with mock.patch.object(K, "R16_TDM_COLS", 64), mock.patch.object(
            K, "_resolve_bf16_k_load", lambda rpb, k=None: "raw"
        ):
            self.assertReason(tail, "route_mismatch", profile=self.vpath)

    def test_stage3_captured_plan_coverage(self):
        """Every Stage-3 captured class resolves to an exact row or a rule except the two
        sub-tile column classes (columns 4 / 8 -> columns_uncovered)."""
        gen = _gen()
        by, unc = {}, []
        for rows, reqs, cols, width in gen.STAGE3_CAPTURE:
            d = self.decide(
                call_args(rows=rows, requests=reqs, columns=cols, width=width),
                profile=self.vpath,
            )
            if d.ok:
                by[d.cell] = by.get(d.cell, 0) + 1
            else:
                unc.append(((rows, reqs, cols, width), d.reason.value))
        self.assertEqual(len(gen.STAGE3_CAPTURE), 75)
        self.assertEqual(
            sorted(unc),
            [
                ((11, 1, 8, 16384), "columns_uncovered"),
                ((16, 1, 4, 16384), "columns_uncovered"),
            ],
        )
        self.assertEqual(
            by,
            {
                "rule:draft_bf16": 15,
                "draft_bf16_r16q16_c65536_w16384": 1,
                "rule:decode_bf16": 4,
                "rule:tail_bf16": 10,
                # R2E: every captured prefill class with columns <= 2048 is
                # lean ktile; the one wider class stays persist
                "rule:prefill_bf16_kt": 42,
                "rule:prefill_bf16_persist": 1,
            },
        )
        # the generator's coverage mode agrees (rules pending -> allow_unverified view)
        cov = gen.coverage(
            self.data,
            [
                {
                    "cache_dtype": "bf16",
                    "rows": r,
                    "requests": q,
                    "columns": c,
                    "table_width": w,
                }
                for r, q, c, w in gen.STAGE3_CAPTURE
            ],
        )
        self.assertEqual(cov["by"], dict(by, **{"uncovered:columns_uncovered": 2}))

    def test_rule_validation(self):
        PR.load_profile(self.vpath, use_cache=False)
        base = self.vdata
        muts = []

        def mut(fn):
            d = json.loads(json.dumps(base))
            fn(d)
            muts.append(d)

        rid = {r["id"]: i for i, r in enumerate(base["rules"])}
        P_ = rid["prefill_bf16_persist"]
        D = rid["draft_bf16"]
        mut(lambda d: d["rules"][D].update(vid="qsa.page4.decg.v1"))  # wrong variant
        mut(lambda d: d["rules"][D]["params"].update(wave_cols=64))
        mut(
            lambda d: d["rules"][P_]["columns"]["pow2"].update(min=256)
        )  # splits variant
        mut(lambda d: d["rules"][D].update(regime="decode"))  # != relation
        mut(lambda d: d["rules"][D].update(relation="sideways"))
        mut(lambda d: d["rules"][D]["rows"].update(max=600))  # sig mismatch
        mut(lambda d: d["rules"][D].update(page_size=16))
        mut(lambda d: d["rules"][D].update(divisor="1.0"))
        mut(
            lambda d: d["rules"][D]["samples"].append(
                {"rows": 600, "requests": 600, "columns": 64, "table_width": 16}
            )
        )  # outside
        mut(
            lambda d: d["rules"][D]["samples"].append(dict(d["rules"][D]["samples"][0]))
        )  # dup
        mut(lambda d: d["rules"][D].update(samples=[]))
        mut(lambda d: d["rules"][D]["golden"].pop("kernel_sha256"))  # pass, no sha
        mut(lambda d: d["rules"][D]["golden"]["checked"].pop())  # sample unchecked
        mut(lambda d: d["rules"][D]["golden"]["checked"][0].update(**{"pass": False}))
        mut(
            lambda d: d["rules"][D]["golden"].update(kernel_sha256="0" * 64)
        )  # old kernel
        mut(
            lambda d: d["rules"].append(json.loads(json.dumps(d["rules"][D])))
        )  # duplicate id
        mut(lambda d: d.update(schema_version=1))  # rules need v2

        def too_big(d):  # 16384 x 65536 fp32 logits exceed int32 byte addressing
            r = d["rules"][P_]
            r["columns"] = {"pow2": {"min": 512, "max": 65536}}
            r["sig"] = PR.rule_signature(r)

        mut(too_big)
        W = rid["prefill_bf16_persist_wide"]

        def resig(fn):
            def f(d):
                fn(d["rules"][W])
                d["rules"][W]["sig"] = PR.rule_signature(d["rules"][W])

            return f

        # wide rule spanning the r32 / persist split (columns 256 -> r32) is rejected
        mut(
            resig(
                lambda r: r.update(
                    columns={"pow2": {"min": 256, "max": 65536}},
                    table_width={"min": 64, "max": 16384},
                )
            )
        )
        mut(resig(lambda r: r.update(rows_x_columns_max=0)))
        mut(resig(lambda r: r.update(rows_x_columns_max="33554432")))
        mut(
            resig(lambda r: r.update(rows_x_columns_max=31 * 65536))
        )  # no rows at 65536
        # chunk bound lifted: 16384 rows x 65536 columns exceed int32 byte addressing
        mut(
            resig(
                lambda r: (
                    r.update(rows_x_columns_max=2**30),
                    r["rows"].update(max=16384),
                )
            )
        )
        mut(lambda d: d["rules"][W].pop("rows_x_columns_max"))  # sig mismatch
        mut(lambda d: d["rules"][W].update(bogus_key=1))  # unknown key
        mut(
            lambda d: d["rules"][W]["samples"].append(
                {"rows": 1024, "requests": 1, "columns": 65536, "table_width": 16384}
            )
        )  # above chunk
        for i, d in enumerate(muts):
            with self.assertRaises(PR.ProfileError, msg=str(i)) as cm:
                PR.load_profile(self.write(d), use_cache=False)
            self.assertEqual(cm.exception.reason, PR.Reason.PROFILE_INVALID)
        # schema 1 (exact rows only) still loads
        d = json.loads(json.dumps(base))
        d["schema_version"] = 1
        d.pop("rules")
        prof = PR.load_profile(self.write(d, "qsa_v1schema_v1"), use_cache=False)
        self.assertEqual(prof.rules, [])
        d2 = self.decide(call_args(rows=8, requests=8), profile=prof.path)
        self.assertEqual(d2.reason.value, "rows_rule_uncovered")

    def test_check_plan_samples_and_merge(self):
        """`check --plan-file` appends admitted plan shapes as rule samples (GPU mocked); a
        rule passes only if every sample passes; merge_golden keeps verified evidence.
        """
        gen = _gen()
        extra = [
            {
                "cache_dtype": "bf16",
                "rows": 96,
                "requests": 24,
                "columns": 1024,
                "table_width": 4096,
            },
            {
                "cache_dtype": "bf16",
                "rows": 16,
                "requests": 1,
                "columns": 4,
                "table_width": 16384,
            },
        ]
        props = types.SimpleNamespace(gcnArchName="gfx1250", multi_processor_count=256)

        def fake_sample(rule, s, api, G, S, seed=0, profile=None):
            out = {k: s[k] for k in ("rows", "requests", "columns", "table_width")}
            bad = rule["id"] == "tail_bf16" and s["rows"] == 4095
            return dict(out, pass_=not bad, vid=rule["vid"])

        out = os.path.join(self.td, "qsa_page4_gfx1250_v1.json")
        with mock.patch.object(gen, "check_sample", fake_sample), mock.patch.object(
            gen, "_golden_modules", lambda: (None, None)
        ), mock.patch.object(
            torch.cuda, "get_device_properties", lambda i: props
        ), mock.patch.dict(
            os.environ, {}
        ), mock.patch.object(
            torch.cuda, "empty_cache", lambda: None
        ):
            self.assertEqual(gen.check(PROFILE_JSON, out, None, extra, "rules"), 1)
        with open(out, encoding="utf-8") as fh:
            got = json.load(fh)
        rules = {r["id"]: r for r in got["rules"]}
        self.assertIn(
            (96, 24, 1024, 4096),
            {PR.sample_key(s) for s in rules["decode_bf16"]["samples"]},
        )
        self.assertEqual(rules["decode_bf16"]["golden"]["status"], "pass")
        self.assertEqual(rules["tail_bf16"]["golden"]["status"], "fail")
        self.assertGreater(rules["tail_bf16"]["golden"]["samples_failed"], 0)
        # rows untouched by --only rules
        self.assertEqual(got["rows"], self.data["rows"])
        prof = PR.load_profile(out, use_cache=False)
        self.assertFalse(next(r for r in prof.rules if r.id == "tail_bf16").verified)
        self.assertEqual(gen.verify(out, capture_plan=extra), [])
        # re-emit with the same plan keeps the decode pass, nothing invents a pass
        merged = gen.merge_golden(
            gen.build_profile(gen.builtin_plan(), capture_plan=extra), got
        )
        m = {r["id"]: r for r in merged["rules"]}
        self.assertEqual(m["decode_bf16"]["golden"]["status"], "pass")
        self.assertEqual(m["tail_bf16"]["golden"]["status"], "fail")
        # without the plan entry, the decode pass still covers the (smaller) sample set
        merged = gen.merge_golden(gen.build_profile(gen.builtin_plan()), got)
        PR.validate_profile(merged)
        self.assertEqual(
            {r["id"]: r for r in merged["rules"]}["decode_bf16"]["golden"]["status"],
            "pass",
        )

    def test_sample_fixture(self):
        gen = _gen()
        fx = gen.sample_fixture(
            {"rows": 512, "requests": 512, "columns": 65536, "table_width": 16384}
        )
        self.assertEqual(
            fx, {"max_ctx": 16 * 4095, "cache_pages": 512 * 4095, "layout": "draft"}
        )
        # every fixture's bf16 cache plane stays below the op's int32 byte-addressing guard
        for reqs in (1, 16, 127, 128, 256, 512):
            fx = gen.sample_fixture(
                {"rows": reqs, "requests": reqs, "columns": 65536, "table_width": 16384}
            )
            self.assertLess(fx["cache_pages"] * 4 * 128 * 2, 2**31)
        self.assertEqual(
            gen.sample_fixture(
                {"rows": 64, "requests": 16, "columns": 65536, "table_width": 16384}
            )["layout"],
            "decode",
        )
        fx = gen.sample_fixture(
            {"rows": 16383, "requests": 32, "columns": 512, "table_width": 16384}
        )
        self.assertEqual((fx["layout"], fx["max_ctx"]), ("even", 16 * 16384))
        t2r, pos = gen._even_layout(16383, 32, 256)
        self.assertEqual(t2r.numel(), 16383)
        self.assertEqual(int(t2r.max()), 31)
        self.assertTrue(bool((pos >= 0).all()) and bool((pos < 256).all()))
        t2r, pos = gen._even_layout(3, 500, 1024)  # draft rows < requests: one row each
        self.assertEqual(t2r.tolist(), [0, 1, 2])
        self.assertEqual(pos.tolist(), [1023, 1023, 1023])


class Router(Env):
    """Profile regimes / variants vs the kernel's own router (no duplicated truth)."""

    def test_regime_of_matches_kernel(self):
        for reqs in (1, 2, 3, 4, 16, 64):
            for rows in list(range(1, 80)) + [127, 128, 129, 320, 512, 2048]:
                reg = PR.regime_of(rows, reqs)
                for fp8 in (False, True):
                    draft = K.draft_route(rows, reqs, fp8, 0x1000) is not None
                    decg = K.decode_group_route(rows, reqs, fp8, 0x1000) is not None
                    r32 = K.wmma_rows_per_block(rows, reqs) == 32
                    self.assertEqual(reg == "draft", draft, (rows, reqs))
                    self.assertEqual(reg == "decode", decg, (rows, reqs))
                    self.assertEqual(reg == "prefill", r32, (rows, reqs))
                    # the r16 tail class is exactly "none of the three specialised kernels"
                    self.assertEqual(
                        reg == "tail", not (draft or decg or r32), (rows, reqs)
                    )

    def test_predict_route_matches_profile_variants(self):
        for dt in ("bf16", "fp8"):
            for rows, reqs in (
                (16, 16),
                (64, 16),
                (512, 2),
                (320, 2),
                (4, 1),
                (64, 2),
                (6, 1),
                (50, 5),
                (23, 2),
                (4095, 512),
                (2048, 1),
                (2049, 1),
                (16384, 32),
            ):
                for cols in (64, 256, 1000, 2048, 4096, 8192, 32768, 65536):
                    reg = PR.regime_of(rows, reqs)
                    vid, params = PR.expected_variant(dt, reg, cols, rows=rows)
                    # + the default-env wrapper knobs (R2E: fp8 draft wg_waves 2)
                    self.assertEqual(
                        API.predict_route(K, dt, rows, reqs, cols, 0x1000),
                        (vid, dict(params, **PR.default_knobs(dt, reg))),
                        (dt, rows, reqs, cols),
                    )
        # R2E route presets, spelled out
        kt = {"k_load": "tdm", "rows_per_block": 32, "kt_rows": 32, "lean": True}
        self.assertEqual(
            API.predict_route(K, "bf16", 512, 2, 2048, 0x1000),
            ("qsa.page4.r32_kt.v1", dict(kt, col_tiles=2)),
        )
        self.assertEqual(
            API.predict_route(K, "bf16", 512, 2, 4096, 0x1000)[0],
            "qsa.page4.r32_persist.v1",
        )
        f8 = dict(kt, cvt="fast", pk8="intrin", kconv=True)
        self.assertEqual(
            API.predict_route(K, "fp8", 2048, 1, 1024, 0x1000),
            ("qsa.page4.r32_kt_fp8.v1", dict(f8, col_tiles=2)),
        )
        self.assertEqual(
            API.predict_route(K, "fp8", 2049, 1, 1024, 0x1000),
            ("qsa.page4.r32_kt_fp8.v1", dict(f8, col_tiles=4)),
        )
        self.assertEqual(
            API.predict_route(K, "fp8", 2048, 1, 4096, 0x1000),
            (
                "qsa.page4.r32_fp8.v1",
                {"k_load": "tdm", "rows_per_block": 32, "cvt": "fast", "pk8": "intrin"},
            ),
        )
        self.assertEqual(
            API.predict_route(K, "fp8", 16, 16, 64, 0x1000)[1]["wg_waves"], 2
        )
        self.assertNotIn(
            "wg_waves", API.predict_route(K, "bf16", 16, 16, 64, 0x1000)[1]
        )
        # kernel-native 3-row decode groups: 'auto' slab 32 for both dtypes (4-row fp8: 64)
        self.assertEqual(K.decode_group_route(48, 16, True, 0x1000, 3), 32)
        self.assertEqual(K.decode_group_route(48, 16, False, 0x1000, 3), 32)
        self.assertEqual(K.decode_group_route(64, 16, True, 0x1000), 64)
        self.assertEqual(
            PR.expected_variant("fp8", "decode", 65536, group_rows=3)[1]["wave_cols"],
            32,
        )


class _Stubs(Env):
    """Real kernel wrapper; only _run / builders / stream lookup are intercepted."""

    def setUp(self):
        super().setUp()
        self.kinds, self.runs = [], []

        def mk(kind, cols_fn):
            def build(*a, **kw):
                lk = _Launch(kind, cols_fn(*a, **kw))
                # ROUND-R2B: real WMMA launchers carry rows_per_launch (= rows_per_block)
                lk.rows_per_launch = (
                    a[1] if len(a) > 1 else kw.get("rows_per_block", 16)
                )
                lk.build_args = (a, kw)
                self.kinds.append((kind, a, kw))
                return lk

            return build

        def wmma_cols(divisor, rpb=16, k_load=None, **kw):
            if k_load == "persist":
                return K.WMMA_COLS * K.TDM_PERSIST_CHUNKS
            if k_load == "ktile":  # R2E default prefill (<= 2048 columns)
                return K.WMMA_COLS * kw["col_tiles"]
            return K.R16_TDM_COLS if (rpb == 16 and k_load == "tdm") else K.WMMA_COLS

        self._d = [
            mock.patch.object(
                K, "_run", lambda launch, *a: self.runs.append((launch, a))
            ),
            mock.patch.object(K, "_build_wmma_kernel", mk("wmma", wmma_cols)),
            mock.patch.object(
                K,
                "_build_wmma_fp8_kernel",
                mk(
                    "wmma_fp8",
                    lambda d, rpb=16, k_load=None, **k: (
                        K.WMMA_COLS * k["col_tiles"] if k_load == "ktile" else 64
                    ),
                ),
            ),
            mock.patch.object(
                K, "_build_draft_kernel", mk("draft", lambda *a, **k: 32)
            ),
            mock.patch.object(
                K, "_build_decode_group_kernel", mk("decg", lambda *a, **k: 32)
            ),
            mock.patch.object(K, "_build_kernel", mk("scalar", lambda *a, **k: 256)),
            mock.patch.object(torch.cuda, "current_stream", lambda dev=None: "STREAM"),
        ]
        for p in self._d:
            p.start()

    def tearDown(self):
        for p in reversed(self._d):
            p.stop()
        super().tearDown()


class Dispatch(_Stubs):
    """Every profiled row / rule sample launches its variant through the op."""

    @staticmethod
    def _vid(kind, a, kw):
        vid, params = Dispatch._vid4(kind, a, kw)
        page = kw.get("page_size", 4)
        return vid.replace("qsa.page4.", f"qsa.page{page}."), params

    @staticmethod
    def _vid4(kind, a, kw):
        pk8 = "asm" if K._FP8_CVT_ASM else "intrin"

        def ktile(p):  # R2E lean ktile builder keywords -> profile params
            p.update(kt_rows=kw["kt_rows"], col_tiles=kw["col_tiles"])
            for flag in ("lean", "kconv"):
                if kw.get(flag):
                    p[flag] = True
            if kw.get("ipf"):
                p["ipf"] = kw["ipf"]
            return p

        if kind == "draft":
            p = {"k_load": a[2], "wave_cols": a[3]}
            if a[0]:
                p.update(sched=K.draft_fp8_sched(), pk8=pk8)
            if "wg_waves" in kw:  # launch knob (R2E: fp8 'auto' = 2)
                p["wg_waves"] = kw["wg_waves"]
            return "qsa.page4.draft.v1", p
        if kind == "decg":
            p = {"k_load": "tdm", "wave_cols": a[2], "barrier": False}
            if a[0]:
                p.update(sched=K.decg_fp8_sched(), pk8=pk8)
            for knob, default in (("group_rows", 4), ("wg_waves", 4)):
                if kw.get(knob, default) != default:
                    p[knob] = kw[knob]
            return "qsa.page4.decg.v1", p
        if kind == "wmma":
            rpb, k_load = a[1], a[2]
            if k_load == "persist":
                return "qsa.page4.r32_persist.v1", {
                    "k_load": "tdm",
                    "rows_per_block": 32,
                    "persist_cols": K.TDM_PERSIST_CHUNKS,
                    "buffers": K.TDM_PERSIST_BUFFERS,
                }
            if k_load == "ktile":
                return "qsa.page4.r32_kt.v1", ktile(
                    {"k_load": "tdm", "rows_per_block": 32}
                )
            return f"qsa.page4.r{rpb}.v1", {"k_load": k_load, "rows_per_block": rpb}
        if kind == "wmma_fp8":
            rpb, k_load = a[1], a[2]
            if k_load == "ktile":
                return "qsa.page4.r32_kt_fp8.v1", ktile(
                    {
                        "k_load": "tdm",
                        "rows_per_block": 32,
                        "cvt": K._FP8_CVT,
                        "pk8": pk8,
                    }
                )
            if rpb == 32:
                return "qsa.page4.r32_fp8.v1", {
                    "k_load": k_load,
                    "rows_per_block": 32,
                    "cvt": K._FP8_CVT,
                    "pk8": pk8,
                }
            return "qsa.page4.r16_fp8.v1", {
                "k_load": k_load,
                "rows_per_block": 16,
                "pk8": pk8,
            }
        return "qsa.page4.scalar.v0", {}

    def test_every_row_dispatches_its_variant(self):
        """Every row (verified via the public raise-API, pending via gen_profile's
        allow_unverified path) launches exactly its profiled variant."""
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        n = n_ver = 0
        for row in prof.rows:
            r = row.raw
            args = call_args(
                rows=r["rows"]["min"],
                requests=r["requests"]["min"],
                width=r["table_width"],
                columns=r["columns"],
                fmt=r["cache_dtype"],
            )
            rs = torch.full((r["rows"]["min"],), 7, dtype=torch.int32)
            args[9] = rs
            self.kinds.clear()
            self.runs.clear()
            if row.verified:
                logits, vis = API.qsa_paged_mqa_logits(*args, profile=self.profile)
                n_ver += 1
            else:
                with self.assertRaises(API.QSAIndexScorerUnsupported):
                    API.qsa_paged_mqa_logits(*args, profile=self.profile)
                self.assertEqual(self.runs, [], row.cell)
                # the sweep path checks the row itself: unbucketed (buckets are ON by
                # default since R2E and allow_unverified admits non-pass buckets)
                (logits, vis), d = API.try_qsa_paged_mqa_logits(
                    *args, profile=self.profile, allow_unverified=True, buckets=False
                )
                self.assertTrue(d.ok, row.cell)
                self.assertEqual(d.cell, row.cell)
            self.assertEqual(len(self.kinds), 1, row.cell)
            self.assertEqual(len(self.runs), 1, row.cell)
            kind, a, kw = self.kinds[0]
            self.assertEqual(
                self._vid(kind, a, kw), (row.vid, dict(row.params)), row.cell
            )
            # pos64 specialization iff int64 positions; divisor default sqrt(128)
            self.assertEqual(kw.get("pos64", False), True, row.cell)
            divisor = a[1] if kind in ("draft", "decg") else a[0]
            self.assertEqual(float(divisor), math.sqrt(128), row.cell)
            self.assertEqual(tuple(logits.shape), (r["rows"]["min"], r["columns"]))
            self.assertEqual(tuple(vis.shape), (r["rows"]["min"],))
            launch_tensors = [t for t in self.runs[0][1] if isinstance(t, torch.Tensor)]
            self.assertTrue(
                any(t.data_ptr() == rs.data_ptr() for t in launch_tensors), row.cell
            )
            n += 1
        self.assertEqual(n, len(prof.rows))
        self.assertGreaterEqual(n, 20)
        # GPU-restamped rows carried by ISA identity; R2E: the 4 bf16 prefill rows (ktile
        # re-route / persist ISA change) and the 2 fp8 draft rows (wg_waves 2) are pending
        self.assertEqual(n_ver, sum(r.verified for r in prof.rows))
        self.assertGreaterEqual(n_ver, 14)

    def test_rule_samples_dispatch_their_variant(self):
        """Rule samples (allow_unverified sweep path) launch exactly the rule's variant."""
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        n = 0
        for rule in prof.rules:
            small = [
                s
                for s in rule.raw["samples"]
                if s["rows"] * s["columns"] <= 1 << 22
                and s["requests"] * s["table_width"] <= 1 << 22
            ]
            self.assertTrue(small, rule.id)
            if PR.page_of(rule.raw) not in API.kernel_page_sizes(K):
                # R2: page 8 before the page-8 kernel: refused statically, nothing launched
                s = small[0]
                args = call_args(
                    rows=s["rows"],
                    requests=s["requests"],
                    width=s["table_width"],
                    columns=s["columns"],
                    fmt=rule.raw["cache_dtype"],
                    page=PR.page_of(rule.raw),
                )
                out, d = API.try_qsa_paged_mqa_logits(
                    *args, profile=self.profile, allow_unverified=True, buckets=False
                )
                self.assertIsNone(out)
                self.assertEqual(d.reason, PR.Reason.CACHE_SHAPE, (rule.id, d))
                continue
            for s in small[:6]:
                args = call_args(
                    rows=s["rows"],
                    requests=s["requests"],
                    width=s["table_width"],
                    columns=s["columns"],
                    fmt=rule.raw["cache_dtype"],
                    page=PR.page_of(rule.raw),
                )
                self.kinds.clear()
                self.runs.clear()
                (logits, _vis), d = API.try_qsa_paged_mqa_logits(
                    *args, profile=self.profile, allow_unverified=True, buckets=False
                )
                self.assertTrue(d.ok, (rule.id, s, d))
                # the rule itself, or an exact row (precedence) -- never a bucket / other rule
                self.assertTrue(
                    d.cell == f"rule:{rule.id}"
                    or not d.cell.startswith(("rule:", "bucket:")),
                    (rule.id, s, d),
                )
                kind, a, kw = self.kinds[0]
                self.assertEqual(
                    self._vid(kind, a, kw), (rule.vid, dict(rule.params)), (rule.id, s)
                )
                self.assertEqual(tuple(logits.shape), (s["rows"], s["columns"]))
                n += 1
        self.assertGreaterEqual(n, 20)

    def test_try_and_raise(self):
        args = call_args(columns=1000)
        with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
            API.qsa_paged_mqa_logits(*args, profile=self.profile, strict=True)
        self.assertEqual(cm.exception.reason, PR.Reason.COLUMNS)
        d = API.decide(*args, profile=self.profile, native=False)
        self.assertEqual(
            (d.ok, d.path, d.vid), (False, API.PATH_NATIVE, API.NATIVE_VID)
        )
        out, d = API.try_qsa_paged_mqa_logits(*args, profile=self.profile)
        self.assertIsNone(out)
        self.assertEqual(d.reason, PR.Reason.COLUMNS)
        self.assertEqual(self.runs, [])  # nothing launched
        out, d = API.try_qsa_paged_mqa_logits(*call_args(), profile=self.profile)
        self.assertTrue(d.ok)
        self.assertEqual(len(self.runs), 1)
        self.assertEqual(self.kinds[-1][0], "decg")

    def test_caller_outputs_are_used(self):
        args = call_args(rows=16, requests=16, fmt="bf16")
        lg = torch.empty(16, 65536)
        vg = torch.empty(16, dtype=torch.int32)
        out = API.qsa_paged_mqa_logits(
            *args, logits=lg, visible_groups=vg, profile=self.profile
        )
        self.assertEqual(out[0].data_ptr(), lg.data_ptr())
        self.assertEqual(out[1].data_ptr(), vg.data_ptr())
        args = call_args(rows=16, requests=16, fmt="fp8")  # pending row: sweep path
        lg8 = torch.empty(16, 65536)
        (o8, _), d = API.try_qsa_paged_mqa_logits(
            *args, logits=lg8, profile=self.profile, allow_unverified=True
        )
        self.assertTrue(d.ok)
        self.assertEqual(o8.data_ptr(), lg8.data_ptr())


def _pass_call_classes(prof, per_rule=3):
    """(label, call_args kwargs) for every pass row and a few small samples per pass rule."""
    out = []
    for row in prof.rows:
        if row.verified:
            r = row.raw
            out.append(
                (
                    row.cell,
                    {
                        "rows": r["rows"]["min"],
                        "requests": r["requests"]["min"],
                        "width": r["table_width"],
                        "columns": r["columns"],
                        "fmt": r["cache_dtype"],
                        "page": PR.page_of(r),
                    },
                )
            )
    for rule in prof.rules:
        if not rule.verified:
            continue
        small = [
            s
            for s in rule.raw["samples"]
            if s["rows"] * s["columns"] <= 1 << 22
            and s["requests"] * s["table_width"] <= 1 << 22
        ]
        for s in small[:per_rule]:
            # R2E: page-8/16 rules pass via the d438 restamp chain -> their own page size
            out.append(
                (
                    rule.cell,
                    {
                        "rows": s["rows"],
                        "requests": s["requests"],
                        "width": s["table_width"],
                        "columns": s["columns"],
                        "fmt": rule.raw["cache_dtype"],
                        "page": PR.page_of(rule.raw),
                    },
                )
            )
    return out


class Launch(_Stubs):
    """API 1.1 ``launch``: same launch as the 1.0 path / kernel wrapper, no ``decide``."""

    def _calls(self, fn):
        self.kinds.clear()
        self.runs.clear()
        out = fn()
        self.assertEqual(len(self.runs), 1)
        return out, list(self.kinds), self.runs[0]

    def _same_launch(self, ref, got, caller_ptrs, label):
        (ref_out, _, (ref_l, ref_a)), (out, _, (l, a)) = ref, got
        # same builder + same (args, kwargs) => same lru_cache entry / compiled function
        self.assertEqual((ref_l.kind, ref_l.build_args), (l.kind, l.build_args), label)
        self.assertEqual(len(ref_a), len(a), label)
        g = next(
            i for i, x in enumerate(ref_a) if isinstance(x, int)
        )  # grid_x position
        outputs = range(g - 3, g)  # logits, visible, starts
        for i, (x, y) in enumerate(zip(ref_a, a)):
            self.assertIs(type(x), type(y), (label, i))
            if isinstance(x, torch.Tensor):
                self.assertEqual(
                    (tuple(x.shape), tuple(x.stride()), x.dtype, x.device),
                    (tuple(y.shape), tuple(y.stride()), y.dtype, y.device),
                    (label, i),
                )
                if x.data_ptr() in caller_ptrs:
                    self.assertEqual(x.data_ptr(), y.data_ptr(), (label, i))
                elif (
                    i not in outputs
                ):  # narrowed int64 metadata: fresh tensors, same values
                    self.assertTrue(torch.equal(x, y), (label, i))
            else:
                self.assertEqual(x, y, (label, i))
        self.assertEqual(tuple(ref_out[0].shape), tuple(out[0].shape), label)
        self.assertEqual(tuple(ref_out[1].shape), tuple(out[1].shape), label)

    def test_launch_matches_wrapper_and_op_for_every_pass_class(self):
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        classes = _pass_call_classes(prof)
        self.assertGreaterEqual(len(classes), 20)
        n = 0
        for label, spec in classes:
            for qp64 in (True, False):
                for meta64 in (False, True):
                    args = call_args(qp64=qp64, **spec)
                    if meta64:
                        args[2], args[3], args[5] = (
                            args[2].long(),
                            args[3].long(),
                            args[5].long(),
                        )
                    rows, cols = spec["rows"], spec["columns"]
                    lg = torch.empty(rows, cols)
                    vg = torch.empty(rows, dtype=torch.int32)
                    args[9] = torch.full((rows,), 7, dtype=torch.int32)
                    caller = {t.data_ptr() for t in args if isinstance(t, torch.Tensor)}
                    caller |= {lg.data_ptr(), vg.data_ptr()}
                    d = self.decide(args, logits=lg, visible_groups=vg)
                    self.assertTrue(d.ok, (label, d))
                    a = args
                    ref = self._calls(
                        lambda: API._launch(
                            None,
                            a[0],
                            a[1],
                            a[2],
                            a[3],
                            a[4],
                            a[5],
                            a[7],
                            a[8],
                            a[9],
                            a[11],
                            lg,
                            vg,
                        )
                    )
                    op = self._calls(
                        lambda: API.qsa_paged_mqa_logits(
                            *a, logits=lg, visible_groups=vg, profile=self.profile
                        )
                    )
                    with mock.patch.object(
                        API, "decide", side_effect=AssertionError("decide")
                    ), mock.patch.object(
                        PR, "match", side_effect=AssertionError("match")
                    ), mock.patch.object(
                        API, "_static_decision", side_effect=AssertionError("static")
                    ), mock.patch.object(
                        K,
                        "qsa_paged_mqa_logits_page4",
                        side_effect=AssertionError("wrapper"),
                    ):
                        got = self._calls(
                            lambda: API.launch(d, *a, logits=lg, visible_groups=vg)
                        )
                    self._same_launch(ref, got, caller, (label, qp64, meta64))
                    self._same_launch(op, got, caller, (label, qp64, meta64, "op"))
                    self.assertEqual(got[0][0].data_ptr(), lg.data_ptr())
                    self.assertEqual(got[0][1].data_ptr(), vg.data_ptr())
                    # pos64 specialization iff int64 positions
                    self.assertEqual(got[1][-1][2].get("pos64", False), qp64, label)
                    n += 1
        self.assertGreaterEqual(n, 80)

    def test_launch_allocates_outputs_like_the_wrapper(self):
        # R2E: the bf16 r512q2 prefill row is pending (persist ISA changed); the fp8 one
        # (r32 fp8 at 65536 columns) is a pass prefill row
        for spec in (
            {"rows": 16, "requests": 16},
            {"rows": 64, "requests": 16},
            {
                "rows": 512,
                "requests": 2,
                "width": 16384,
                "columns": 65536,
                "fmt": "fp8",
            },
        ):
            args = call_args(**spec)
            d = self.decide(args)
            self.assertTrue(d.ok, d)
            ref = self._calls(
                lambda: API._launch(
                    None,
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    None,
                    args[8],
                    None,
                    args[11],
                    None,
                    None,
                )
            )
            with mock.patch.object(API, "decide", side_effect=AssertionError("decide")):
                got = self._calls(lambda: API.launch(d, *args))
            caller = {t.data_ptr() for t in args if isinstance(t, torch.Tensor)}
            self._same_launch(ref, got, caller, spec)
            logits, vis = got[0]
            self.assertEqual(
                (logits.dtype, tuple(logits.shape), vis.dtype, tuple(vis.shape)),
                (torch.float32, (spec["rows"], args[8]), torch.int32, (spec["rows"],)),
            )

    def test_plan_is_reused_and_builder_called_once_per_pos_dtype(self):
        args = call_args(rows=64, requests=16)
        d = self.decide(args)
        for _ in range(3):
            API.launch(d, *args)
        self.assertEqual(len(self.runs), 3)
        self.assertEqual(len(self.kinds), 1)  # built once (pos64)
        args32 = call_args(rows=64, requests=16, qp64=False)
        API.launch(d, *args32)
        API.launch(d, *args32)
        self.assertEqual(len(self.kinds), 2)  # + the int32-position build
        self.assertEqual(
            [kw.get("pos64", False) for _, _, kw in self.kinds], [True, False]
        )

    def test_guards(self):
        args = call_args(rows=16, requests=16)
        d = self.decide(args)
        self.assertTrue(d.ok)
        # a declined decision without a prepared native plan refuses with its own reason
        bad = API.decide(*call_args(columns=1000), profile=self.profile, native=False)
        self.assertEqual(bad.path, API.PATH_NATIVE)
        with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
            API.launch(bad, *args)
        self.assertIs(cm.exception.decision, bad)
        self.assertEqual(cm.exception.reason, PR.Reason.COLUMNS)
        q32 = list(args)
        q32[0] = args[0].float()
        with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
            API.launch(d, *q32)
        self.assertEqual(cm.exception.reason, PR.Reason.Q_DTYPE)
        for other in (
            call_args(rows=32, requests=16),
            call_args(rows=16, requests=16, width=8192),
            call_args(rows=16, requests=16, fmt="fp8"),
        ):
            with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
                API.launch(d, *other)
            self.assertEqual(cm.exception.reason, PR.Reason.ROUTE_MISMATCH)
        with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
            API.launch(API.Decision(True, PR.Reason.OK, profile=self.profile), *args)
        self.assertEqual(cm.exception.reason, PR.Reason.PROFILE_KERNEL_MISMATCH)
        with self.assertRaises(API.QSAIndexScorerUnsupported):
            API.launch(None, *args)
        with mock.patch.object(
            API, "_KERNEL", types.SimpleNamespace()
        ):  # kernel re-imported
            with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
                API.launch(d, *args)
            self.assertEqual(cm.exception.reason, PR.Reason.PROFILE_KERNEL_MISMATCH)
        PR.clear_cache()  # profile reloaded
        with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
            API.launch(d, *args)
        self.assertEqual(cm.exception.reason, PR.Reason.PROFILE_KERNEL_MISMATCH)
        self.assertEqual(self.runs, [])
        d2 = self.decide(args)  # fresh decide works
        API.launch(d2, *args)
        self.assertEqual(len(self.runs), 1)

    def test_launch_reads_no_environment(self):
        args = call_args(rows=64, requests=16)
        d = self.decide(args)

        class _NoEnv(dict):
            def get(self, *a, **k):
                raise AssertionError("env read")

            __getitem__ = get

        # R2: the first launch of a launcher through a decision may scope the persistent
        # compile-cache directory (reads/sets FLYDSL_RUNTIME_CACHE_DIR once); every later
        # launch reads no environment.
        API.launch(d, *args)
        with mock.patch.object(os, "environ", _NoEnv()):
            API.launch(d, *args)
            API.launch(d, *args)
        self.assertEqual(len(self.runs), 3)

    def test_try_uses_shared_path_without_inspect(self):
        args = call_args(rows=16, requests=16)
        with mock.patch.object(
            inspect, "signature", side_effect=AssertionError("inspect")
        ):
            (_lg, _vg), d = API.try_qsa_paged_mqa_logits(*args, profile=self.profile)
        self.assertTrue(d.ok)
        self.assertEqual(len(self.runs), 1)


class Prewarm(_Stubs):
    """API 1.1 ``prewarm``: every pass variant, both position dtypes, idempotent."""

    def setUp(self):
        super().setUp()
        self._w = [
            mock.patch.dict(API._PREWARMED, clear=True),
            mock.patch.object(
                torch.cuda,
                "synchronize",
                side_effect=AssertionError("cuda sync on cpu"),
            ),
        ]
        for p in self._w:
            p.start()

    def tearDown(self):
        for p in reversed(self._w):
            p.stop()
        super().tearDown()

    def _expected(self, dtypes, pos=("i32", "i64")):
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        tags = set()
        for r in [*prof.rows, *prof.rules]:
            if r.verified and r.raw["cache_dtype"] in dtypes:
                for p in pos:
                    tags.add(f"{r.vid}/{r.raw['cache_dtype']}/{p}")
        return sorted(tags)

    def _expected_builds(self, dtypes, pos=("i32", "i64")):
        """Distinct (tag, params) builds: every pass row / rule / bucket variant (a pass
        kernel-native group bucket adds its group_rows 3 decode-group build)."""
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        out = set()
        for r in [*prof.rows, *prof.rules, *prof.buckets]:
            if r.verified and r.raw["cache_dtype"] in dtypes:
                for p in pos:
                    out.add(
                        (
                            f"{r.vid}/{r.raw['cache_dtype']}/{p}",
                            tuple(sorted(dict(r.params).items())),
                        )
                    )
        return out

    def test_variants_are_the_union_of_pass_rows_and_rules(self):
        got = API.prewarm(self.profile, cache_dtypes=("bf16", "fp8"), device="cpu")
        want = self._expected(("bf16", "fp8"))
        self.assertEqual(got, want)
        # GPU restamp on kernel 58051973: every row and rule passes -> page 4/8/16 x (bf16
        # draft/decg/r16/r32_kt/r32_persist + fp8 draft/decg/r16_fp8/r32_kt_fp8/r32_fp8)
        # x 2 pos dtypes
        # RULES-REQUESTS: while the widened prefill rules are pending, their r32_kt /
        # r32_persist / r32_kt_fp8 variants (3 pages x 2 pos dtypes each) are not prewarmed
        self.assertEqual(len(want), 42 if RR_PENDING else 60)
        self.assertTrue(
            all(
                any(v in t for t in want)
                for v in (
                    (".draft.v1/fp8",)
                    if RR_PENDING
                    else (".r32_kt.", ".r32_persist.", ".draft.v1/fp8", ".r32_kt_fp8.")
                )
            )
        )
        builds = self._expected_builds(("bf16", "fp8"))
        # + fp8 ktile C4 (rows > 2048) and the pass g3 decode-group buckets (bf16 + fp8),
        # 3 pages x 2 pos dtypes each
        self.assertEqual(len(builds), 54 if RR_PENDING else 78)
        self.assertEqual(
            len(self.runs), len(builds)
        )  # one private launch each (FlyDSL JIT)
        built = {(k, a, tuple(sorted(kw.items()))) for k, a, kw in self.kinds}
        self.assertEqual(len(built), len(builds))
        vids = {Dispatch._vid(k, a, kw)[0] for k, a, kw in self.kinds}
        self.assertEqual(vids, {t.split("/")[0] for t in want})
        self.assertEqual(sum(kw.get("group_rows") == 3 for _, _, kw in self.kinds), 12)
        # pending-only variants and fp8 are not built by default
        self.kinds.clear()
        self.runs.clear()
        API._PREWARMED.clear()
        self.assertEqual(
            API.prewarm(self.profile, device="cpu"), self._expected(("bf16",))
        )
        self.assertTrue(all("/bf16/" in t for t in self._expected(("bf16",))))
        self.assertEqual(len(self.runs), len(self._expected_builds(("bf16",))))
        self.assertEqual(len(self.runs), 28 if RR_PENDING else 36)

    def test_builds_exactly_what_real_calls_build(self):
        """The builder (args, kwargs) of every pass call class == a prewarmed lru_cache key."""
        API.prewarm(self.profile, cache_dtypes=("bf16", "fp8"), device="cpu")
        warmed = {(k, a, tuple(sorted(kw.items()))) for k, a, kw in self.kinds}
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        for label, spec in _pass_call_classes(prof):
            for qp64 in (True, False):
                self.kinds.clear()
                API.qsa_paged_mqa_logits(
                    *call_args(qp64=qp64, **spec), profile=self.profile
                )
                k, a, kw = self.kinds[-1]
                self.assertIn((k, a, tuple(sorted(kw.items()))), warmed, (label, qp64))

    def test_idempotent(self):
        first = API.prewarm(self.profile, device="cpu")
        n_runs, n_builds = len(self.runs), len(self.kinds)
        self.assertEqual(API.prewarm(self.profile, device="cpu"), first)
        self.assertEqual((len(self.runs), len(self.kinds)), (n_runs, n_builds))
        # one entry per distinct build (tag + params: a g3 bucket shares the decg tag)
        self.assertEqual(len(API._PREWARMED), n_builds)
        self.assertEqual(
            {(k[3], k[4]) for k in API._PREWARMED}, self._expected_builds(("bf16",))
        )
        self.assertEqual(sorted({k[3] for k in API._PREWARMED}), first)
        self.assertTrue(all(v >= 0 for v in API._PREWARMED.values()))

    def test_pos_dtypes(self):
        got = API.prewarm(self.profile, pos_dtypes=(torch.int32,), device="cpu")
        self.assertEqual(got, self._expected(("bf16",), ("i32",)))
        self.assertTrue(all(kw.get("pos64", False) is False for _, _, kw in self.kinds))
        n32 = len(self.kinds)
        self.assertEqual(n32, len(self._expected_builds(("bf16",), ("i32",))))
        got = API.prewarm(self.profile, pos_dtypes=(torch.int64,), device="cpu")
        self.assertEqual(got, self._expected(("bf16",), ("i64",)))
        # the int64 builds differ from the int32 ones by exactly pos64=True
        k64 = self.kinds[n32:]
        self.assertEqual(len(k64), n32)
        self.assertTrue(all(kw.get("pos64") is True for _, _, kw in k64))
        self.assertEqual(
            sorted(repr((k, a, sorted(kw.items()))) for k, a, kw in self.kinds[:n32]),
            sorted(
                repr((k, a, sorted((x, y) for x, y in kw.items() if x != "pos64")))
                for k, a, kw in k64
            ),
        )
        with self.assertRaises(ValueError):
            API.prewarm(self.profile, pos_dtypes=(torch.int16,), device="cpu")

    def test_dummy_launch_is_private_and_routed(self):
        API.prewarm(self.profile, cache_dtypes=("bf16", "fp8"), device="cpu")
        self.assertEqual(len(self.runs), len(self.kinds))
        for (launch, a), (kind, ba, kw) in zip(self.runs, self.kinds):
            rows, columns = a[-6], a[-5]
            vid, params = Dispatch._vid(kind, ba, kw)
            if ".r32_persist." in vid or ".r32_fp8." in vid:
                # R2E: persist / r32 fp8 are routed only above 2048 columns
                self.assertEqual((rows, columns), (32, 4096), vid)
            elif params.get("col_tiles") == 4:
                self.assertEqual(rows, 2080, vid)  # fp8 ktile C4 = the rows > 2048 band
            else:
                self.assertLessEqual(rows, 32, vid)
                self.assertLessEqual(columns, 512, vid)

    def test_off_or_uncovered_device_builds_nothing(self):
        with mock.patch.object(
            API, "_load_kernel", side_effect=AssertionError("kernel import")
        ):
            self.assertEqual(API.prewarm(None, device="cpu"), [])
            self.assertEqual(API.prewarm("no_such_profile", device="cpu"), [])
        with mock.patch.object(API, "_device_info", lambda dev: ("gfx950", 256)):
            self.assertEqual(API.prewarm(self.profile, device="cpu"), [])
        with mock.patch.object(API, "_ALLOWED_DEVICE_TYPES", ("cuda",)):
            self.assertEqual(API.prewarm(self.profile, device="cpu"), [])
        self.assertEqual((self.runs, self.kinds), ([], []))


class Scripts(unittest.TestCase):
    def test_written_scripts_compile(self):
        for rel in (
            "op_tests/test_qsa_index_scorer.py",
            "op_tests/op_benchmarks/qsa_index_scorer/perf_qsa_index_scorer.py",
            "op_tests/op_benchmarks/qsa_index_scorer/isa_audit_qsa_index_scorer.py",
            "op_tests/op_benchmarks/qsa_index_scorer/zero_overhead_qsa_index_scorer.py",
            "aiter/ops/qsa_index_scorer/gen_profile.py",
        ):
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                compile(fh.read(), rel, "exec")


if __name__ == "__main__":
    print(f"kernel import: {K_HOW}", flush=True)
    unittest.main()
