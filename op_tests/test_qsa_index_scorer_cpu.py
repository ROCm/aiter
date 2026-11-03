# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
G2_DIR = os.path.join(ROOT, "aiter", "ops", "flydsl", "kernels", "mqa_logits", "_g2")
PROFILE_JSON = os.path.join(ROOT, "aiter", "configs", "qsa_index_scorer",
                            "qsa_page4_gfx1250_v1.json")
ATOM_REL = os.path.join("atom_impl", "atom", "model_ops", "qwen4_exp", "ops", "qsa.py")


def _golden_dir() -> str:
    cands = [os.environ.get("QSA_GOLDEN_DIR", ""),
             os.path.join(ROOT, "..", "..", "..", "..", "golden_baseline"),
             "/work/task/MI455_QSA_INDEX_SCORER/golden_baseline"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, ATOM_REL)):
            return os.path.abspath(c)
    raise RuntimeError("frozen golden not found: set QSA_GOLDEN_DIR (drift guards must run)")


@contextlib.contextmanager
def _tmpdir():
    """Plain makedirs temp dir (mkdtemp's 0o700 breaks some sandboxed Windows ACLs)."""
    path = os.path.join(tempfile.gettempdir(), f"qsa_isc_{os.getpid()}_{uuid.uuid4().hex[:8]}")
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
        name, os.path.join(PKG_DIR, "__init__.py"), submodule_search_locations=[PKG_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, sys.modules[name + ".api"], sys.modules[name + ".profiles"]


def load_kernel():
    sys.path.insert(0, G2_DIR)
    try:
        import _load  # noqa: E402
    finally:
        sys.path.remove(G2_DIR)
    K, how = _load.load_kernel_module(allow_stub=True)
    return K, how


PKG, API, PR = load_pkg()
_CLEAN = {k: v for k, v in os.environ.items()
          if not k.startswith("QSA_") or k in ("QSA_GOLDEN_DIR", "QSA_RESULTS")}
_CLEAN.pop(PR.PROFILE_ENV, None)
with mock.patch.dict(os.environ, _CLEAN, clear=True):
    K, K_HOW = load_kernel()
FREEZE_REASONS = (
    "ok", "disabled", "profile_not_found", "profile_invalid", "profile_kernel_mismatch",
    "kernel_env_mismatch", "kernel_unavailable", "device_not_gpu", "device_mismatch",
    "arch_unsupported", "cu_unsupported", "compress_ratio_unsupported", "q_dtype_unsupported",
    "q_shape_unsupported", "q_layout_unsupported", "cache_format_unsupported",
    "cache_shape_unsupported", "cache_layout_unsupported", "cache_alignment_unsupported",
    "scale_layout_unsupported", "metadata_unsupported", "output_layout_unsupported",
    "empty_call", "divisor_uncovered", "max_columns_invalid", "plane_too_large",
    "rows_rule_uncovered", "columns_uncovered", "table_width_uncovered", "row_unverified",
    "route_mismatch",
)


def aligned(shape, dtype, align=128, offset=0):
    n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    back = torch.zeros(n + 2 * align, dtype=torch.uint8)
    start = (-back.data_ptr()) % align + offset
    return back[start:start + n].view(dtype).view(*shape)


def call_args(rows=64, requests=16, width=16384, columns=65536, fmt="bf16", pages=8,
              qp64=True, cache_offset=0):
    q = aligned((rows, 4, 128), torch.bfloat16)
    if fmt == "bf16":
        cache, scale = aligned((pages, 4, 1, 128), torch.bfloat16, offset=cache_offset), None
    else:
        cache = aligned((pages, 4, 1, 128), torch.float8_e4m3fn, offset=cache_offset)
        scale = torch.ones((pages, 4, 1), dtype=torch.float32)
    pt = torch.zeros((requests, width), dtype=torch.int32)
    if rows <= requests:
        t2r = torch.arange(rows, dtype=torch.int32)
    else:
        t2r = (torch.arange(rows, dtype=torch.int32) * requests) // rows
    qp = torch.arange(rows, dtype=torch.int64 if qp64 else torch.int32)
    cl = torch.full((requests,), 4 * width, dtype=torch.int32)
    return [q, cache, pt, t2r, qp, cl, 4, None, columns, None, False, scale]


class _Launch:
    def __init__(self, kind, cols=64):
        self.kind, self.cols_per_block = kind, cols


class Env(unittest.TestCase):
    """Clean QSA_* env, CPU tensors admitted, device info faked, kernel injected."""

    profile = PR.DEFAULT_PROFILE

    def setUp(self):
        self._p = [mock.patch.dict(os.environ, _CLEAN, clear=True),
                   mock.patch.object(API, "_ALLOWED_DEVICE_TYPES", ("cuda", "cpu")),
                   mock.patch.object(API, "_device_info", lambda dev: ("gfx1250", 256)),
                   mock.patch.object(API, "_KERNEL", K),
                   mock.patch.object(API, "_KERNEL_ENV", PR.effective_kernel_env(_CLEAN))]
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
        self.assertEqual(self.data["kernel"]["sha256"], sha,
                         "kernel changed: re-run gen_profile emit + GPU check")
        self.assertEqual(sha, "30c6213d492ad50a9e063456628737ca993060d73dfa4ead248558ba5abb121c",
                         "kernel source differs from champion a78c285c (ISA identity)")

    def test_rows(self):
        for r in self.data["rows"]:
            self.assertEqual(r["sig"], PR.row_signature(r))
            self.assertEqual((r["vid"], r["params"]),
                             PR.expected_variant(r["cache_dtype"], r["regime"], r["columns"]))
            self.assertEqual(PR.regime_of(r["rows"]["min"], r["requests"]["min"]), r["regime"])
            self.assertIn(r["golden"]["status"], ("pass", "pending"))
            if r["golden"]["status"] == "pass":
                self.assertTrue(r["golden"].get("cases") or r["golden"].get("checked"))
            for key in ("arch", "cu", "cache_dtype", "columns", "table_width", "vid", "params",
                        "cell", "generated_by", "notes"):
                self.assertIn(key, r)

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
        self.assertEqual(g["revision"],
                         "4b8888907e2bd3530bfde10eb081bf8cf60d49749aea4cac62ec486211b12ff8")

    def test_generator_no_drift(self):
        spec = importlib.util.spec_from_file_location(
            "qsa_gen_profile_test", os.path.join(PKG_DIR, "gen_profile.py"))
        gen = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [PKG_DIR] + sys.path):
            spec.loader.exec_module(gen)
        self.assertEqual(gen.verify(PROFILE_JSON), [])
        with _tmpdir() as td:
            out = os.path.join(td, "qsa_page4_gfx1250_v1.json")
            self.assertEqual(gen.main(["emit", "--out", out]), 0)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), self.data)
            cap = os.path.join(td, "cap.json")
            with open(cap, "w") as fh:
                json.dump([{"cache_dtype": "bf16", "rows": 128, "requests": 32, "columns": 16384,
                            "table_width": 16384}], fh)
            out2 = os.path.join(td, "qsa_extra_v1.json")
            self.assertEqual(gen.main(["emit", "--out", out2, "--plan-file", cap]), 0)
            prof = PR.load_profile(out2, use_cache=False)
            new = [r for r in prof.rows if r.cell == "decode_bf16_r128q32_c16384_w16384"]
            self.assertEqual(len(new), 1)
            self.assertFalse(new[0].verified)   # captured rows start pending
        with self.assertRaises(ValueError):
            gen.make_row(dict(cache_dtype="bf16", rows=6, requests=1, columns=64,
                              table_width=16))

    def _write(self, td, data, name="qsa_bad_v1"):
        path = os.path.join(td, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        return path

    def test_invalid_profiles(self):
        with _tmpdir() as td:
            cases = []
            d = json.loads(json.dumps(self.data)); d["profile"] = "qsa_bad_v1"
            del d["kernel_env"]; cases.append(d)
            d = json.loads(json.dumps(self.data)); d["profile"] = "qsa_bad_v1"
            d["schema_version"] = 99; cases.append(d)
            d = json.loads(json.dumps(self.data)); d["profile"] = "qsa_bad_v1"
            d["rows"][0]["columns"] = 1234; cases.append(d)          # sig no longer matches
            d = json.loads(json.dumps(self.data)); d["profile"] = "qsa_bad_v1"
            d["rows"].append(d["rows"][0]); cases.append(d)          # duplicate sig
            d = json.loads(json.dumps(self.data)); d["profile"] = "qsa_bad_v1"
            d["rows"][0]["vid"] = "qsa.page4.decg.v2"; cases.append(d)
            d = json.loads(json.dumps(self.data)); d["profile"] = "qsa_bad_v1"
            d["rows"][0]["params"] = dict(d["rows"][0]["params"], wave_cols=64); cases.append(d)
            d = json.loads(json.dumps(self.data))                     # name != file stem
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
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and ast.unparse(node.func.value) == "os.environ"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).startswith("QSA_")):
                key = node.args[0].value
                dflt = ast.unparse(node.args[1]) if len(node.args) > 1 else None
                found.setdefault(key, set()).add(dflt)
        # every import-time switch that changes code selection is pinned by the profile
        ignore = {"QSA_TRACE_ROWS_PER_BLOCK"}
        self.assertEqual(set(found) - ignore, set(PR.KERNEL_ENV_DEFAULTS))
        for key, val in PR.KERNEL_ENV_DEFAULTS.items():
            want = "_BF16_K_LOAD" if key == "QSA_FP8_K_LOAD" else repr(val)
            self.assertEqual(found[key], {want}, key)
        # the env snapshot of the injected (clean-env) kernel matches its resolved switches
        self.assertEqual((K._DECODE_GROUP, K._DECODE_GROUP_COLS, K._DRAFT_PATH, K._ACCESS,
                          K._BF16_K_LOAD, K._FP8_K_LOAD, K.TDM_PERSIST, K.TDM_PERSIST_CHUNKS,
                          K.TDM_R16_NOBAR, K._BF16_R64, K._FP8_CVT, K.R16_TDM_COLS),
                         ("all", 32, True, "raw", "auto", "auto", True, 8, True, False,
                          "fast", 32))

    def test_effective_env_inheritance(self):
        e = PR.effective_kernel_env({"QSA_BF16_K_LOAD": "RAW"})
        self.assertEqual((e["QSA_BF16_K_LOAD"], e["QSA_FP8_K_LOAD"]), ("raw", "raw"))
        self.assertEqual(PR.effective_kernel_env({}), PR.KERNEL_ENV_DEFAULTS)


# ---------------------------------------------------------------------------
class OffByDefault(unittest.TestCase):
    def test_disabled_without_profile(self):
        with mock.patch.dict(os.environ, _CLEAN, clear=True), \
                mock.patch.object(API, "_KERNEL", None), \
                mock.patch.object(API, "_device_info", side_effect=AssertionError("device query")), \
                mock.patch.object(API, "_load_kernel", side_effect=AssertionError("kernel import")):
            args = call_args()
            d = API.decide(*args)
            self.assertEqual((d.ok, d.reason.value), (False, "disabled"))
            self.assertEqual(API.eligibility(*args), (False, "disabled"))
            with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
                API.qsa_paged_mqa_logits(*args)
            self.assertEqual(cm.exception.reason, PR.Reason.DISABLED)
            out, d = API.try_qsa_paged_mqa_logits(*args)
            self.assertIsNone(out)
            self.assertEqual(API.layout_eligibility(args[1]).reason, PR.Reason.DISABLED)
            self.assertIsNone(API.selected_profile())

    def test_env_enables(self):
        env = dict(_CLEAN, **{PR.PROFILE_ENV: PR.DEFAULT_PROFILE})
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(API.selected_profile(), PR.DEFAULT_PROFILE)
        with mock.patch.dict(os.environ, dict(_CLEAN, **{PR.PROFILE_ENV: "nope_v0"}), clear=True):
            self.assertEqual(API.decide(*call_args()).reason, PR.Reason.PROFILE_NOT_FOUND)

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
            "print('NO_SIDE_EFFECTS')\n")
        env = dict(_CLEAN)
        r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                           env=env, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("NO_SIDE_EFFECTS", r.stdout)


# ---------------------------------------------------------------------------
class AbiDrift(unittest.TestCase):
    def _atom_def(self):
        path = os.path.join(_golden_dir(), ATOM_REL)
        with open(path, "rb") as fh:
            raw = fh.read()
        tree = ast.parse(raw.decode("utf-8"))
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "qsa_paged_mqa_logits")
        return raw, fn

    def test_atom_source_frozen(self):
        raw, _ = self._atom_def()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), API.ATOM_SCORER_SOURCE_SHA256,
                         "ATOM scorer source changed: re-review ATOM_SCORER_SIGNATURE")

    def test_atom_signature_matches_frozen(self):
        _, fn = self._atom_def()
        a = fn.args
        self.assertEqual((a.posonlyargs, a.vararg, a.kwarg, a.kwonlyargs), ([], None, None, []))
        names = [x.arg for x in a.args]
        defaults = [inspect.Parameter.empty] * (len(names) - len(a.defaults)) + [
            ast.literal_eval(d) for d in a.defaults]
        self.assertEqual(tuple(zip(names, defaults)), API.ATOM_SCORER_SIGNATURE)
        ret = ast.unparse(fn.returns)
        self.assertEqual(ret, "tuple[torch.Tensor, torch.Tensor]")

    def _check_ours(self, fn, extra_kwonly):
        params = list(inspect.signature(fn).parameters.values())
        lead = params[:len(API.ATOM_SCORER_SIGNATURE)]
        self.assertEqual(tuple((p.name, p.default) for p in lead), API.ATOM_SCORER_SIGNATURE)
        self.assertTrue(all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in lead))
        rest = params[len(lead):]
        self.assertTrue(all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in rest))
        self.assertEqual([p.name for p in rest], extra_kwonly)
        self.assertTrue(all(p.default is None or p.default is False for p in rest))

    def test_our_signatures(self):
        self._check_ours(API.qsa_paged_mqa_logits, ["logits", "visible_groups", "profile"])
        self._check_ours(API.eligibility, ["logits", "visible_groups", "profile"])
        self._check_ours(API.decide, ["logits", "visible_groups", "profile", "allow_unverified"])
        self.assertIs(PKG.qsa_paged_mqa_logits, API.qsa_paged_mqa_logits)

    def test_reason_enum_frozen(self):
        self.assertEqual(tuple(PR.iter_reasons()), FREEZE_REASONS)
        self.assertEqual((PR.API_VERSION, PR.PROFILE_ENV, PR.DEFAULT_PROFILE),
                         ("1.0", "AITER_QSA_INDEX_SCORER_PROFILE", "qsa_page4_gfx1250_v1"))

    def test_kernel_module_api_kept(self):
        with open(PR.KERNEL_PATH, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "qsa_paged_mqa_logits_page4")
        self.assertEqual([x.arg for x in fn.args.args],
                         ["q", "compressed_k_cache", "page_table", "token_to_request",
                          "query_positions", "context_lens", "compress_ratio", "divisor",
                          "max_columns", "compressed_k_scale", "logits", "visible_groups",
                          "row_starts"])
        self.assertTrue(callable(K.qsa_paged_mqa_logits_page4))
        # the op passes the kernel exactly the knobs it understands
        src = inspect.getsource(API._launch)
        for kw in ("compress_ratio=", "divisor=score_divisor", "max_columns=max_columns",
                   "compressed_k_scale=", "logits=logits", "visible_groups=", "row_starts="):
            self.assertIn(kw, src)

    def test_default_divisor_equals_atom(self):
        self.assertEqual(API._DEFAULT_DIVISOR, 128 ** 0.5)


# ---------------------------------------------------------------------------
class Reasons(Env):
    def test_ok_every_verified_row(self):
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        for row in prof.rows:
            r = row.raw
            args = call_args(rows=r["rows"]["min"], requests=r["requests"]["min"],
                             width=r["table_width"], columns=r["columns"], fmt=r["cache_dtype"])
            d = self.decide(args)
            if row.verified:
                self.assertTrue(d.ok, (row.cell, d))
                self.assertEqual((d.cell, d.vid, d.columns), (row.cell, row.vid, r["columns"]))
                ok, reason = d
                self.assertEqual((ok, reason), (True, "ok"))
                self.assertEqual(API.eligibility(*args, profile=self.profile), (True, "ok"))
                # int32 positions and full-width (max_columns=None) variants of the call
                a32 = call_args(rows=r["rows"]["min"], requests=r["requests"]["min"],
                                width=r["table_width"], columns=r["columns"],
                                fmt=r["cache_dtype"], qp64=False)
                self.assertTrue(self.decide(a32).ok)
            else:
                self.assertEqual(d.reason, PR.Reason.ROW_UNVERIFIED, row.cell)
                self.assertTrue(self.decide(args, allow_unverified=True).ok)

    def test_tensor_reasons(self):
        base = call_args()

        def mut(i, v):
            a = list(base); a[i] = v; return a

        self.assertReason(mut(6, 16), "compress_ratio_unsupported")
        self.assertReason(mut(0, base[0].float()), "q_dtype_unsupported")
        self.assertReason(mut(0, torch.zeros(64, 8, 128, dtype=torch.bfloat16)), "q_shape_unsupported")
        self.assertReason(mut(0, torch.zeros(64, 4, 64, dtype=torch.bfloat16)), "q_shape_unsupported")
        self.assertReason(mut(0, aligned((4, 64, 128), torch.bfloat16).transpose(0, 1)),
                          "q_layout_unsupported")
        self.assertReason(mut(0, torch.zeros(0, 4, 128, dtype=torch.bfloat16)), "empty_call")
        # cache formats: mxfp8 / fp4 / bf16+scale / fp8 without scale / fp16
        fp8 = aligned((8, 4, 1, 128), torch.float8_e4m3fn)
        a = mut(1, fp8); a[11] = torch.zeros(8, 4, 1, 4, dtype=torch.uint8)
        self.assertIn("mxfp8", self.assertReason(a, "cache_format_unsupported").detail)
        a = mut(1, aligned((8, 4, 1, 64), torch.uint8)); a[11] = torch.zeros(8, 4, 1, 4, dtype=torch.uint8)
        self.assertIn("fp4", self.assertReason(a, "cache_format_unsupported").detail)
        a = list(base); a[11] = torch.ones(8, 4, 1)
        self.assertReason(a, "cache_format_unsupported")
        self.assertReason(mut(1, fp8), "cache_format_unsupported")
        self.assertReason(mut(1, aligned((8, 4, 1, 128), torch.float16)), "cache_format_unsupported")
        self.assertReason(mut(1, aligned((8, 16, 1, 128), torch.bfloat16)), "cache_shape_unsupported")
        self.assertReason(mut(1, aligned((8, 4, 2, 128), torch.bfloat16)[:, :, :1]),
                          "cache_layout_unsupported")
        for off in (16, 64):
            self.assertReason(mut(1, aligned((8, 4, 1, 128), torch.bfloat16, offset=off)),
                              "cache_alignment_unsupported")
        a = call_args(fmt="fp8"); a[11] = torch.ones(8, 4, 2)[:, :, :1]
        self.assertReason(a, "scale_layout_unsupported")
        a = call_args(fmt="fp8"); a[11] = torch.ones(7, 4, 1)
        self.assertReason(a, "scale_layout_unsupported")
        # metadata
        self.assertReason(mut(2, base[2].to(torch.int16)), "metadata_unsupported")
        self.assertReason(mut(2, base[2].t().contiguous().t()), "metadata_unsupported")
        self.assertReason(mut(2, torch.zeros(16, 0, dtype=torch.int32)), "metadata_unsupported")
        self.assertReason(mut(3, base[3][:-1]), "metadata_unsupported")
        self.assertReason(mut(3, base[3].to(torch.int16)), "metadata_unsupported")
        self.assertReason(mut(4, base[4].view(-1, 1)), "metadata_unsupported")
        self.assertReason(mut(4, torch.zeros(128, dtype=torch.int64)[::2]), "metadata_unsupported")
        self.assertReason(mut(5, base[5][:-1]), "metadata_unsupported")
        self.assertReason(mut(3, torch.zeros(64, dtype=torch.int32, device="meta")),
                          "device_mismatch")
        # divisor / max_columns
        self.assertReason(mut(7, 2.0), "divisor_uncovered")
        self.assertTrue(self.decide(mut(7, 128 ** 0.5)).ok)
        self.assertReason(mut(8, 0), "max_columns_invalid")
        self.assertReason(mut(8, -5), "max_columns_invalid")
        self.assertTrue(self.decide(mut(10, True)).ok)        # allow_flydsl is ignored
        # outputs
        self.assertReason(base, "output_layout_unsupported",
                          logits=torch.empty(64, 1024, dtype=torch.float32))
        self.assertReason(base, "output_layout_unsupported",
                          visible_groups=torch.empty(64, dtype=torch.int64))
        self.assertReason(mut(9, torch.empty(63, dtype=torch.int32)), "output_layout_unsupported")
        self.assertTrue(self.decide(mut(9, torch.empty(64, dtype=torch.int32)),
                                    logits=torch.empty(64, 65536),
                                    visible_groups=torch.empty(64, dtype=torch.int32)).ok)
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
        with mock.patch.object(API, "_KERNEL", None), mock.patch.object(API, "_KERNEL_ENV", None), \
                mock.patch.dict(os.environ, {"QSA_DECODE_GROUP_COLS": "64"}):
            self.assertReason(call_args(), "kernel_env_mismatch")
        with mock.patch.dict(API._KERNEL_SHA, {}, clear=True), \
                mock.patch.object(API, "_kernel_sha256", lambda: "0" * 64):
            self.assertReason(call_args(), "profile_kernel_mismatch")
        with mock.patch.object(API, "_KERNEL", None), \
                mock.patch.object(API, "_load_kernel", side_effect=ImportError("no flydsl")):
            self.assertReason(call_args(), "kernel_unavailable")
        with mock.patch.object(K, "decode_group_route", lambda *a: 64):
            self.assertReason(call_args(), "route_mismatch")

    def test_coverage_reasons(self):
        self.assertReason(call_args(rows=6, requests=1), "rows_rule_uncovered")       # r16 tail
        self.assertReason(call_args(rows=96, requests=24), "rows_rule_uncovered")     # no row
        self.assertReason(call_args(rows=64, requests=16, width=8192, columns=32768),
                          "table_width_uncovered")
        self.assertReason(call_args(columns=4096), "columns_uncovered")
        self.assertTrue(self.decide(call_args(columns=None)).ok)   # full width = 65536 bucket
        d = self.assertReason(call_args(rows=16, requests=16, columns=32768), "row_unverified")
        self.assertEqual(d.cell, "draft_bf16_r16q16_c32768_w16384")

    def test_static_reason_set(self):
        self.assertIn(PR.Reason.CACHE_ALIGNMENT, PR.STATIC_REASONS)
        self.assertNotIn(PR.Reason.COLUMNS, PR.STATIC_REASONS)
        d = API.layout_eligibility(call_args()[1], profile=self.profile)
        self.assertTrue(d.ok, d)
        d = API.layout_eligibility(call_args(cache_offset=16)[1], profile=self.profile)
        self.assertEqual(d.reason, PR.Reason.CACHE_ALIGNMENT)


# ---------------------------------------------------------------------------
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

    def test_predict_route_matches_profile_variants(self):
        for dt in ("bf16", "fp8"):
            for rows, reqs in ((16, 16), (64, 16), (512, 2), (320, 2), (4, 1), (64, 2)):
                for cols in (64, 256, 1000, 32768, 65536):
                    reg = PR.regime_of(rows, reqs)
                    self.assertEqual(API.predict_route(K, dt, rows, reqs, cols, 0x1000),
                                     PR.expected_variant(dt, reg, cols), (dt, rows, reqs, cols))


class Dispatch(Env):
    """Real kernel wrapper; only _run / builders / stream lookup are intercepted."""

    def setUp(self):
        super().setUp()
        self.kinds, self.runs = [], []

        def mk(kind, cols_fn):
            def build(*a, **kw):
                lk = _Launch(kind, cols_fn(*a, **kw))
                lk.build_args = (a, kw)
                self.kinds.append((kind, a, kw))
                return lk
            return build

        def wmma_cols(divisor, rpb=16, k_load=None, **kw):
            if k_load == "persist":
                return K.WMMA_COLS * K.TDM_PERSIST_CHUNKS
            return K.R16_TDM_COLS if (rpb == 16 and k_load == "tdm") else K.WMMA_COLS

        self._d = [mock.patch.object(K, "_run", lambda launch, *a: self.runs.append((launch, a))),
                   mock.patch.object(K, "_build_wmma_kernel", mk("wmma", wmma_cols)),
                   mock.patch.object(K, "_build_wmma_fp8_kernel", mk("wmma_fp8", lambda *a, **k: 64)),
                   mock.patch.object(K, "_build_draft_kernel", mk("draft", lambda *a, **k: 32)),
                   mock.patch.object(K, "_build_decode_group_kernel", mk("decg", lambda *a, **k: 32)),
                   mock.patch.object(K, "_build_kernel", mk("scalar", lambda *a, **k: 256)),
                   mock.patch.object(torch.cuda, "current_stream", lambda dev=None: "STREAM")]
        for p in self._d:
            p.start()

    def tearDown(self):
        for p in reversed(self._d):
            p.stop()
        super().tearDown()

    @staticmethod
    def _vid(kind, a, kw):
        if kind == "draft":
            return "qsa.page4.draft.v1", {"k_load": a[2], "wave_cols": a[3]}
        if kind == "decg":
            return "qsa.page4.decg.v1", {"k_load": "tdm", "wave_cols": a[2], "barrier": False}
        if kind == "wmma":
            rpb, k_load = a[1], a[2]
            if k_load == "persist":
                return "qsa.page4.r32_persist.v1", {"k_load": "tdm", "rows_per_block": 32,
                                                    "persist_cols": K.TDM_PERSIST_CHUNKS,
                                                    "buffers": K.TDM_PERSIST_BUFFERS}
            return f"qsa.page4.r{rpb}.v1", {"k_load": k_load, "rows_per_block": rpb}
        if kind == "wmma_fp8":
            rpb, k_load = a[1], a[2]
            if rpb == 32:
                return "qsa.page4.r32_fp8.v1", {"k_load": k_load, "rows_per_block": 32,
                                                "cvt": K._FP8_CVT}
            return "qsa.page4.r16_fp8.v1", {"k_load": k_load, "rows_per_block": 16}
        return "qsa.page4.scalar.v0", {}

    def test_every_verified_row_dispatches_its_variant(self):
        prof = PR.load_profile(PR.DEFAULT_PROFILE)
        n = 0
        for row in prof.rows:
            if not row.verified:
                continue
            r = row.raw
            args = call_args(rows=r["rows"]["min"], requests=r["requests"]["min"],
                             width=r["table_width"], columns=r["columns"], fmt=r["cache_dtype"])
            rs = torch.full((r["rows"]["min"],), 7, dtype=torch.int32)
            args[9] = rs
            self.kinds.clear(); self.runs.clear()
            logits, vis = API.qsa_paged_mqa_logits(*args, profile=self.profile)
            self.assertEqual(len(self.kinds), 1, row.cell)
            self.assertEqual(len(self.runs), 1, row.cell)
            kind, a, kw = self.kinds[0]
            self.assertEqual(self._vid(kind, a, kw), (row.vid, dict(row.params)), row.cell)
            # pos64 specialization iff int64 positions; divisor default sqrt(128)
            self.assertEqual(kw.get("pos64", False), True, row.cell)
            divisor = a[1] if kind in ("draft", "decg") else a[0]
            self.assertEqual(float(divisor), math.sqrt(128), row.cell)
            self.assertEqual(tuple(logits.shape), (r["rows"]["min"], r["columns"]))
            self.assertEqual(tuple(vis.shape), (r["rows"]["min"],))
            launch_tensors = [t for t in self.runs[0][1] if isinstance(t, torch.Tensor)]
            self.assertTrue(any(t.data_ptr() == rs.data_ptr() for t in launch_tensors), row.cell)
            n += 1
        self.assertGreaterEqual(n, 15)

    def test_try_and_raise(self):
        args = call_args(columns=4096)
        with self.assertRaises(API.QSAIndexScorerUnsupported) as cm:
            API.qsa_paged_mqa_logits(*args, profile=self.profile)
        self.assertEqual(cm.exception.reason, PR.Reason.COLUMNS)
        out, d = API.try_qsa_paged_mqa_logits(*args, profile=self.profile)
        self.assertIsNone(out)
        self.assertEqual(d.reason, PR.Reason.COLUMNS)
        self.assertEqual(self.runs, [])                     # nothing launched
        out, d = API.try_qsa_paged_mqa_logits(*call_args(), profile=self.profile)
        self.assertTrue(d.ok)
        self.assertEqual(len(self.runs), 1)
        self.assertEqual(self.kinds[-1][0], "decg")

    def test_caller_outputs_are_used(self):
        args = call_args(rows=16, requests=16, fmt="fp8")
        lg = torch.empty(16, 65536)
        vg = torch.empty(16, dtype=torch.int32)
        out = API.qsa_paged_mqa_logits(*args, logits=lg, visible_groups=vg, profile=self.profile)
        self.assertEqual(out[0].data_ptr(), lg.data_ptr())
        self.assertEqual(out[1].data_ptr(), vg.data_ptr())


class Scripts(unittest.TestCase):
    def test_written_scripts_compile(self):
        for rel in ("op_tests/test_qsa_index_scorer.py",
                    "op_tests/op_benchmarks/qsa_index_scorer/perf_qsa_index_scorer.py",
                    "op_tests/op_benchmarks/qsa_index_scorer/isa_audit_qsa_index_scorer.py",
                    "aiter/ops/qsa_index_scorer/gen_profile.py"):
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                compile(fh.read(), rel, "exec")


if __name__ == "__main__":
    print(f"kernel import: {K_HOW}", flush=True)
    unittest.main()
