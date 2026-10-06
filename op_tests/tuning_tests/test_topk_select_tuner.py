# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Oracle, lookup, routing, grouping and crash-isolation tests for the topk_select tuner."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import pandas as pd
import torch

from aiter.ops import topk_select_tuning as tun
from aiter.ops.topk_select import _choose
from aiter.ops.topk_select_tuning import (
    expand_dist_field,
    half_octave_bounds,
    lookup_tuned,
    oracle_check,
    override_tuned_rows,
    reload_tuned_table,
    warn_unavailable_tuned,
)

CUDA = torch.cuda.is_available()
# Imported by name, not by file path: the tuner's spawned workers unpickle its
# functions by module name.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "csrc" / "topk_select"))
import topk_select_tune as tt

HDR = ",".join(tun.TUNED_COLUMNS) + "\n"


def _rec(**kw):
    lo, hi = half_octave_bounds(4096)
    wlo, whi = half_octave_bounds(8192)
    base = {
        "gfx": "",
        "cu_num": 0,
        "rows_lo": lo,
        "rows_hi": hi,
        "width_lo": wlo,
        "width_hi": whi,
        "k": 2048,
        "dtype": "float32",
        "ragged": False,
        "tie": None,
        "deterministic": False,
        "mode": "graph",
        "backend": "sampled",
        "aiter_rev": "",
    }
    base.update(kw)
    return base


def _csv_row(rows, k, backend, rev="abc1234"):
    """A tuned row for the band holding (rows, width 8192)."""
    rlo, rhi = half_octave_bounds(rows)
    wlo, whi = half_octave_bounds(8192)
    return (
        f"gfx950,256,{rlo},{rhi},{wlo},{whi},{k},float32,False,,False,graph,"
        f"{backend},1,plain,2,0.5,equal@{rows}x8192,1,{rev}\n"
    )


def _shape(rows, width, k, tie=None, deterministic=False, fp32=True):
    """`_choose` / `_router_choice` arguments as `topk_select` builds them on GPU 0."""
    import aiter.ops.topk_select as ts
    from aiter.ops.flydsl.kernels.tensor_shim import wave_size_of

    return (rows, width, k, wave_size_of(0), False, tie, deterministic, fp32, 0) + (
        ts._sampled_ok(0),
    )


class Payload:
    """Not a tensor or plain value: a weights_only load must refuse it."""


class TestOracleNegatives(unittest.TestCase):
    def test_five_failure_tags(self):
        m, n, k = 4, 16, 4
        x = torch.arange(m * n, dtype=torch.float32).view(m, n)
        lens = torch.full((m,), n, dtype=torch.int32)
        good = torch.stack([torch.arange(n - 1, n - 1 - k, -1) for _ in range(m)]).to(
            torch.int32
        )
        self.assertEqual(oracle_check(good, x, lens, k), "ok")

        first_k = (
            torch.arange(k, dtype=torch.int32).unsqueeze(0).expand(m, k).contiguous()
        )
        self.assertEqual(oracle_check(first_k, x, lens, k), "value_mismatch")

        dup = good.clone()
        dup[:, 1] = dup[:, 0]
        self.assertEqual(oracle_check(dup, x, lens, k), "duplicate_index")

        oob = good.clone()
        oob[0, 0] = n
        self.assertEqual(oracle_check(oob, x, lens, k), "index_out_of_range")

        past = good.clone()
        short = lens.clone()
        short[0] = 2
        past[0, 0] = 3
        self.assertEqual(oracle_check(past, x, short, k), "index_past_end")

        wrong = good.clone()
        wrong[0, 0] = 0
        self.assertEqual(oracle_check(wrong, x, lens, k), "value_mismatch")

        short_count = good.clone()
        short_count[1, 3] = -1
        self.assertTrue(
            oracle_check(short_count, x, lens, k).startswith("live_count=3")
        )

    def test_ragged_rows_with_fill_are_ok(self):
        m, n, k = 3, 16, 4
        x = torch.randn(m, n)
        lens = torch.tensor([16, 2, 0], dtype=torch.int32)
        idx = torch.full((m, k), -1, dtype=torch.int32)
        idx[0] = torch.topk(x[0], k).indices.int()
        idx[1, :2] = torch.topk(x[1, :2], 2).indices.int()
        self.assertEqual(oracle_check(idx, x, lens, k), "ok")
        bad = idx.clone()
        bad[1, 2] = 5
        self.assertEqual(oracle_check(bad, x, lens, k), "index_past_end")

    def test_nan_input_is_named(self):
        x = torch.randn(2, 16)
        x[1, 3] = float("nan")
        lens = torch.full((2,), 16, dtype=torch.int32)
        idx = torch.topk(torch.nan_to_num(x), 4, dim=1).indices.int()
        self.assertIn("NaN", oracle_check(idx, x, lens, 4))


class TestBands(unittest.TestCase):
    def test_every_value_lies_in_its_own_band_and_bands_tile(self):
        prev_hi = 0
        lo, hi = half_octave_bounds(1)
        for n in range(1, 1 << 21):
            if n > hi:
                self.assertEqual(half_octave_bounds(n)[0], prev_hi + 1, n)
                lo, hi = half_octave_bounds(n)
            self.assertTrue(lo <= n <= hi, (n, lo, hi))
            prev_hi = hi
        for n in (11, 362, 2896, 11585, 741455):
            lo, hi = half_octave_bounds(n)
            self.assertTrue(lo <= n <= hi, (n, lo, hi))


class TestLookup(unittest.TestCase):
    def setUp(self):
        reload_tuned_table()
        _choose.cache_clear()

    def tearDown(self):
        reload_tuned_table()
        _choose.cache_clear()

    def test_hit(self):
        with override_tuned_rows([_rec()]):
            self.assertEqual(
                lookup_tuned(4096, 8192, 2048, False, None, False, "gfx950", 304),
                "sampled",
            )

    def test_miss_falls_through(self):
        with override_tuned_rows([_rec()]):
            self.assertIsNone(
                lookup_tuned(64, 8192, 2048, False, None, False, "gfx950", 304)
            )

    def test_band_edges(self):
        lo, hi = half_octave_bounds(4096)
        self.assertEqual((lo, hi), (4096, 5792))
        with override_tuned_rows([_rec(rows_lo=lo, rows_hi=hi)]):
            for rows, want in ((lo, "sampled"), (hi, "sampled"), (hi + 1, None)):
                self.assertEqual(
                    lookup_tuned(rows, 8192, 2048, False, None, False, "gfx950", 304),
                    want,
                )
            self.assertIsNone(
                lookup_tuned(lo - 1, 8192, 2048, False, None, False, "gfx950", 304)
            )

    def test_tightest_band_wins(self):
        wide = _rec(rows_lo=1, rows_hi=100000, backend="plain")
        with override_tuned_rows([wide, _rec(backend="sampled")]):
            self.assertEqual(
                lookup_tuned(4096, 8192, 2048, False, None, False, "gfx950", 304),
                "sampled",
            )

    def test_unavailable_backend_warns(self):
        tun._BAD_BACKEND_WARNED.clear()
        with self.assertLogs("aiter", level="WARNING") as cm:
            warn_unavailable_tuned("sampled", 1, 8, 2)
        self.assertTrue(any("sampled" in m for m in cm.output))

    def test_expand_all(self):
        self.assertIn("equal", expand_dist_field("all"))
        self.assertEqual(expand_dist_field("randn;equal"), ["randn", "equal"])

    def test_rev_check_runs_git_once(self):
        tun.aiter_rev.cache_clear()
        with (
            mock.patch("subprocess.check_output", return_value="abc1234\n") as co,
            override_tuned_rows([_rec(aiter_rev="abc1234")]),
        ):
            for rows in range(4096, 4106):
                lookup_tuned(rows, 8192, 2048, False, None, False, "gfx950", 304)
        self.assertEqual(co.call_count, 1)
        tun.aiter_rev.cache_clear()

    def test_unknown_rev_never_runs_git(self):
        tun.aiter_rev.cache_clear()
        with (
            mock.patch("subprocess.check_output") as co,
            override_tuned_rows([_rec(aiter_rev="unknown")]),
        ):
            lookup_tuned(4096, 8192, 2048, False, None, False, "gfx950", 304)
        co.assert_not_called()


class TestTableLoading(unittest.TestCase):
    """Which files `topk_select` reads, through `AITER_CONFIGS` like every op."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.a, self.b = d / "a.csv", d / "b.csv"
        self.a.write_text(HDR + _csv_row(4096, 2048, "sampled"))
        self.b.write_text(HDR + _csv_row(512, 2048, "decode"))
        reload_tuned_table()

    def tearDown(self):
        os.environ.pop("AITER_CONFIG_TOPK_SELECT", None)
        reload_tuned_table()
        self.tmp.cleanup()

    def _lookup(self, rows):
        return lookup_tuned(rows, 8192, 2048, False, None, False, "gfx950", 256)

    def test_env_colon_list_loads_every_file(self):
        os.environ["AITER_CONFIG_TOPK_SELECT"] = f"{self.a}{os.pathsep}{self.b}"
        reload_tuned_table()
        self.assertEqual(self._lookup(4096), "sampled")
        self.assertEqual(self._lookup(512), "decode")

    def test_model_configs_merged_without_rewriting(self):
        from aiter.jit import core

        root = Path(self.tmp.name) / "root"
        mc = root / "aiter" / "configs" / "model_configs"
        mc.mkdir(parents=True)
        default = root / "aiter" / "configs" / "topk_select_tuned.csv"
        default.write_text(HDR)
        (root / "aiter" / "configs" / "topk_select_untuned.csv").write_text(
            ",".join(tt.LOOKUP_KEYS) + "\n"
        )
        two = HDR + _csv_row(4096, 2048, "sampled") + _csv_row(512, 2048, "decode")
        model = mc / "m_topk_select_tuned.csv"
        model.write_text(two)
        os.environ.pop("AITER_CONFIG_TOPK_SELECT", None)
        with (
            mock.patch.object(core, "AITER_ROOT_DIR", str(root)),
            mock.patch.object(core, "AITER_CONFIG_TOPK_SELECT", str(default)),
        ):
            reload_tuned_table()
            self.assertEqual(self._lookup(4096), "sampled")
            self.assertEqual(self._lookup(512), "decode")
        self.assertEqual(model.read_text(), two)

    def test_load_and_hit_are_logged(self):
        os.environ["AITER_CONFIG_TOPK_SELECT"] = str(self.a)
        reload_tuned_table()
        with self.assertLogs("aiter", level="INFO") as cm:
            self._lookup(4096)
        self.assertTrue(any("1 tuned row(s) loaded" in m for m in cm.output))
        with (
            mock.patch.object(tun, "AITER_LOG_TUNED_CONFIG", 1),
            self.assertLogs("aiter", level="INFO") as cm,
        ):
            self._lookup(4100)
        self.assertTrue(
            any(
                "band rows 4096-5792" in m and "backend is sampled!" in m
                for m in cm.output
            )
        )

    def test_two_tables_claiming_one_band_fall_back_without_raising(self):
        # get_config_file writes the lower-us row back to the source files and
        # raises; that call keeps the router, and the next lookup reads the
        # repaired files.
        clash = Path(self.tmp.name) / "c.csv"
        clash.write_text(HDR + _csv_row(4096, 2048, "decode"))
        os.environ["AITER_CONFIG_TOPK_SELECT"] = f"{self.a}{os.pathsep}{clash}"
        reload_tuned_table()
        with self.assertLogs("aiter", level="WARNING") as cm:
            self.assertIsNone(self._lookup(4096))
        warned = [m for m in cm.output if "tuned table not loaded" in m]
        self.assertEqual(len(warned), 1, cm.output)
        self.assertIn("duplicate shape", warned[0])
        self.assertNotIn("decode", clash.read_text())
        self.assertEqual(self._lookup(4100), "sampled")

    def test_shipped_tables_keep_the_merge_contract(self):
        # As for chunk_gdn_h_opt: the tuned table ships header-only, and the
        # untuned one carries the lookup keys get_config_file dedups on.
        cfg = Path(tt.AITER_ROOT_DIR) / "aiter" / "configs"
        tuned = (cfg / "topk_select_tuned.csv").read_text().splitlines()
        untuned = (cfg / "topk_select_untuned.csv").read_text().splitlines()
        self.assertEqual(tuned, [",".join(tun.TUNED_COLUMNS)])
        self.assertEqual(untuned, [",".join(tt.LOOKUP_KEYS)])

    def test_hand_edited_spaces_load_and_unknown_backend_is_named(self):
        spaced = HDR.replace(",", ", ") + _csv_row(4096, 2048, "sampled").replace(
            ",", ", "
        )
        self.a.write_text(spaced + _csv_row(512, 2048, "sampld"))
        os.environ["AITER_CONFIG_TOPK_SELECT"] = str(self.a)
        reload_tuned_table()
        with self.assertLogs("aiter", level="WARNING") as cm:
            self.assertEqual(self._lookup(4096), "sampled")
            self.assertIsNone(self._lookup(512))
        self.assertTrue(any("unknown backend(s) sampld" in m for m in cm.output))

    def test_missing_file_warns(self):
        os.environ["AITER_CONFIG_TOPK_SELECT"] = str(Path(self.tmp.name) / "nope.csv")
        reload_tuned_table()
        with self.assertLogs("aiter", level="WARNING") as cm:
            self.assertIsNone(self._lookup(4096))
        self.assertTrue(any("does not exist" in m for m in cm.output))


@unittest.skipUnless(CUDA, "needs a GPU")
class TestChooseAndDispatch(unittest.TestCase):
    def tearDown(self):
        reload_tuned_table()
        _choose.cache_clear()

    def test_choose_hit_and_miss(self):
        with override_tuned_rows([_rec()]):
            _choose.cache_clear()
            self.assertEqual(_choose(*_shape(4096, 8192, 2048)), "sampled")
        with override_tuned_rows([]):
            _choose.cache_clear()
            miss = _choose(*_shape(4096, 8192, 2048))
            self.assertIn(miss, ("plain", "decode", "stream", "small_k", "argmax"))

    def test_tie_and_deterministic_cannot_override(self):
        with override_tuned_rows([_rec()]):
            _choose.cache_clear()
            self.assertNotEqual(
                _choose(*_shape(4096, 8192, 2048, tie="low")), "sampled"
            )
            _choose.cache_clear()
            self.assertNotEqual(
                _choose(*_shape(4096, 8192, 2048, deterministic=True)), "sampled"
            )

    def test_same_path_dispatch(self):
        import aiter.ops.topk_select as ts

        seen = []
        orig = ts._dispatch

        def wrap(backend, *a, **kw):
            seen.append(backend)
            return orig(backend, *a, **kw)

        rlo, rhi = half_octave_bounds(32)
        rec = _rec(rows_lo=rlo, rows_hi=rhi, k=64, backend="stream")
        with override_tuned_rows([rec]), mock.patch.object(ts, "_dispatch", wrap):
            ts._choose.cache_clear()
            ts.topk_select(torch.randn(32, 8192, device="cuda"), 64)
        self.assertEqual(seen, ["stream"])

    def _raising(self, ts, bad, seen):
        orig = ts._dispatch

        def flaky(backend, *a, **kw):
            seen.append(backend)
            if backend == bad:
                raise RuntimeError("simulated launch failure")
            return orig(backend, *a, **kw)

        return mock.patch.object(ts, "_dispatch", flaky)

    def _not_router(self, ts, rows):
        router = ts._router_choice(*_shape(rows, 8192, 64))
        return router, [b for b in ("decode", "stream") if b != router]

    def test_tuned_backend_that_raises_falls_back_to_the_router(self):
        import aiter.ops.topk_select as ts

        router, (tuned, *_) = self._not_router(ts, 32)
        gfx, cu = tun._runtime_chip()
        row = _csv_row(32, 64, tuned).replace("gfx950,256,", f"{gfx},{cu},", 1)
        x = torch.randn(32, 8192, device="cuda")
        lens = torch.full((32,), 8192, dtype=torch.int32, device="cuda")
        seen = []
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.csv"
            path.write_text(HDR + row)
            tun._RETIRED.clear()
            with (
                mock.patch.dict(os.environ, {"AITER_CONFIG_TOPK_SELECT": str(path)}),
                self._raising(ts, tuned, seen),
                self.assertLogs("aiter", level="WARNING") as cm,
            ):
                reload_tuned_table()
                ts._choose.cache_clear()
                idx = ts.topk_select(x, 64)[1]
                idx2 = ts.topk_select(x, 64)[1]
        reload_tuned_table()
        ts._choose.cache_clear()
        tun._RETIRED.clear()
        self.assertEqual(seen, [tuned, router, router])
        self.assertEqual(oracle_check(idx, x, lens, 64), "ok")
        self.assertEqual(oracle_check(idx2, x, lens, 64), "ok")
        self.assertTrue(any("raised RuntimeError" in m for m in cm.output))

    def test_a_candidate_that_raises_under_the_tuner_is_not_rescued(self):
        # Rescuing it would grade the router's answer and time as the candidate's.
        import aiter.ops.topk_select as ts

        _router, (tuned, *_) = self._not_router(ts, 32)
        rlo, rhi = half_octave_bounds(32)
        rec = _rec(rows_lo=rlo, rows_hi=rhi, k=64, backend=tuned)
        seen = []
        tun._RETIRED.clear()
        with override_tuned_rows([rec]), self._raising(ts, tuned, seen):
            ts._choose.cache_clear()
            with self.assertRaisesRegex(RuntimeError, "simulated launch failure"):
                ts.topk_select(torch.randn(32, 8192, device="cuda"), 64)
        ts._choose.cache_clear()
        self.assertEqual(seen, [tuned])
        self.assertEqual(tun._RETIRED, set())

    def test_retiring_one_backend_leaves_rows_naming_another(self):
        import aiter.ops.topk_select as ts

        router, others = self._not_router(ts, 32)
        rlo, rhi = half_octave_bounds(32)
        tun._RETIRED.clear()
        shape = (32, 8192, 64, False, None, False, "float32")
        with self.assertLogs("aiter", level="WARNING"):
            retired = tun.retire_tuned("plain", router, RuntimeError("x"), *shape)
        self.assertTrue(retired)
        try:
            for b in ("plain", *others):
                rec = _rec(rows_lo=rlo, rows_hi=rhi, k=64, backend=b)
                with override_tuned_rows([rec]):
                    got = lookup_tuned(
                        32, 8192, 64, False, None, False, rec["gfx"], rec["cu_num"]
                    )
                self.assertEqual(got, None if b == "plain" else b)
        finally:
            tun._RETIRED.clear()

    def test_rows_tuned_on_one_dtype_never_apply_to_another(self):
        import aiter.ops.topk_select as ts

        f32_row = _rec(rows_lo=1, rows_hi=1, k=1, backend="small_k")
        with (
            override_tuned_rows([f32_row]),
            mock.patch.object(tun.logger, "warning") as warn,
        ):
            ts._choose.cache_clear()
            got = ts._choose(*_shape(1, 8192, 1, fp32=False), dtype="bfloat16")
        self.assertEqual(got, "argmax")
        warn.assert_not_called()
        bf16_row = _rec(rows_lo=1, rows_hi=1, k=1, backend="argmax", dtype="bfloat16")
        with override_tuned_rows([bf16_row]):
            self.assertEqual(
                lookup_tuned(1, 8192, 1, False, None, False, "", 0, dtype="bfloat16"),
                "argmax",
            )
            self.assertIsNone(lookup_tuned(1, 8192, 1, False, None, False, "", 0))
        ts._choose.cache_clear()

    def test_a_new_backend_is_a_candidate_with_no_tuner_change(self):
        import aiter.ops.topk_select as ts
        from aiter.ops.flydsl.kernels.tensor_shim import wave_size_of

        real = ts._available

        def with_new(width, k, wave_size, ragged, fp32, sampled_ok):
            return real(width, k, wave_size, ragged, fp32, sampled_ok) | {"newbe"}

        by_tie = {**ts._BACKENDS_BY_TIE, None: (*ts._BACKENDS_BY_TIE[None], "newbe")}
        rows_limit = {**ts._ROW_PREDICATES, "newbe": lambda r, w, k, device: r <= 64}
        g = {
            "key": tt.band_key(
                "gfx950", 256, 64, 8192, 64, False, None, False, "graph"
            ),
            "specs": ["equal@64x8192", "equal@90x8192"],
        }
        with (
            mock.patch.object(ts, "_available", with_new),
            mock.patch.object(ts, "_BACKENDS_BY_TIE", by_tie),
            mock.patch.object(ts, "_ROW_PREDICATES", rows_limit),
        ):
            plan = tt.plan_backends(g, wave_size_of(0))
            self.assertIn("newbe", plan["available"]["equal@64x8192"])
            self.assertNotIn("newbe", plan["available"]["equal@90x8192"])
            self.assertNotIn("newbe", plan["common"])
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "t.csv"
                p.write_text(HDR + _csv_row(64, 64, "newbe"))
                self.assertEqual(
                    [r["backend"] for r in tun._load_tuned_rows_from_path(str(p))],
                    ["newbe"],
                )


class TestDtypes(unittest.TestCase):
    def test_aliases_and_rejections(self):
        for alias, want in (
            ("", "float32"),
            (None, "float32"),
            ("fp32", "float32"),
            ("torch.bfloat16", "bfloat16"),
            ("BF16", "bfloat16"),
            ("half", "float16"),
        ):
            self.assertEqual(tun.normalize_dtype(alias), want, alias)
        with self.assertRaises(ValueError):
            tun.normalize_dtype("int8")
        self.assertIn(
            "only at k=1", tt.check_entry(8, 8192, 64, False, None, "randn", "bfloat16")
        )
        self.assertIsNone(tt.check_entry(8, 8192, 1, False, None, "randn", "bfloat16"))


class TestRouterFailureFallback(unittest.TestCase):
    G: ClassVar[dict] = {
        "key": tt.band_key(
            "gfx950", 256, 4096, 8192, 2048, False, None, False, "graph"
        ),
        "specs": ["equal@4096x8192"],
    }
    OPTS = SimpleNamespace(objective="no_regress", min_improvement_pct=3.0)

    def _summary(self, us, dropped=None, crashed=None):
        return {
            "router": {"equal@4096x8192": "plain"},
            "common": ["decode", "plain", "sampled", "stream"],
            "us": {"equal@4096x8192": us},
            "dropped": dropped or {},
            "crashed": crashed or {},
        }

    def test_broken_router_choice_is_replaced_by_the_fastest_working(self):
        s = self._summary(
            {"decode": 446.0, "sampled": 50.0, "stream": 475.0},
            dropped={"plain": "equal@4096x8192: value_mismatch"},
        )
        plan = {"router": s["router"], "common": set(s["common"])}
        rec, summary = tt.decide(self.G, plan, s["us"], self.OPTS, s)
        self.assertEqual(rec["backend"], "sampled")
        self.assertEqual(rec["us_default"], "")
        self.assertIn("equal@4096x8192", summary["why"]["router_failed"])

    def test_nothing_working_keeps_no_row_and_says_why(self):
        s = self._summary({}, crashed={"plain": "process died (exit code -6)"})
        plan = {"router": s["router"], "common": set(s["common"])}
        rec, summary = tt.decide(self.G, plan, s["us"], self.OPTS, s)
        self.assertIsNone(rec)
        self.assertIn("no other backend passed", summary["why"]["_"])

    def test_a_router_failure_in_one_process_still_routes_around_it(self):
        first = self._summary(
            {"decode": 446.0, "sampled": 50.0, "stream": 475.0},
            dropped={"plain": "equal@4096x8192: graph capture failed"},
        )
        second = self._summary(
            {"decode": 440.0, "plain": 430.0, "sampled": 52.0, "stream": 470.0}
        )
        rec, summary = tt.confirm(self.G, first, second, self.OPTS)
        self.assertEqual(rec["backend"], "sampled")
        self.assertIn("router_failed", summary["why"])


class TestPick(unittest.TestCase):
    def test_no_regress_rejects_single_sample_slowdown(self):
        us = {
            "randn": {"plain": 10.0, "sampled": 8.0},
            "equal": {"plain": 10.0, "sampled": 11.0},
        }
        pick, why = tt._pick("no_regress", us, "plain", 3.0)
        self.assertIsNone(pick)
        self.assertIn("equal", why["sampled"])

    def test_no_regress_accepts_geomean_win(self):
        us = {
            "randn": {"plain": 10.0, "sampled": 9.0},
            "equal": {"plain": 40.0, "sampled": 10.0},
        }
        pick, why = tt._pick("no_regress", us, "plain", 3.0)
        self.assertEqual(pick, "sampled")
        self.assertGreater(why["speedup"], 1.03)

    def test_minimax_allows_a_regression(self):
        us = {
            "randn": {"plain": 10.0, "sampled": 12.0},
            "equal": {"plain": 80.0, "sampled": 10.0},
        }
        pick, _ = tt._pick("minimax", us, "plain", 3.0)
        self.assertEqual(pick, "sampled")

    def test_router_baseline_is_per_sample(self):
        # The router picks plain at one size and stream at another in the same band;
        # `stream` must be judged against each sample's own router choice.
        us = {
            "a@4096x8192": {"plain": 10.0, "stream": 10.1},
            "b@5000x8192": {"plain": 30.0, "stream": 12.0},
        }
        router = {"a@4096x8192": "plain", "b@5000x8192": "stream"}
        pick, why = tt._pick("no_regress", us, router, 3.0)
        self.assertIsNone(pick)
        self.assertIn("plain", why)


class TestGrouping(unittest.TestCase):
    def _df(self, rows):
        cols = ["rows", "width", "k", "ragged", "tie", "deterministic", "dist"]
        return pd.DataFrame(rows, columns=cols)

    def test_same_band_rows_pool_into_one_group(self):
        df = self._df(
            [
                (4100, 8192, 2048, False, "", False, "equal"),
                (4600, 8192, 2048, False, "", False, "equal"),
                (5000, 8192, 2048, False, "", False, "randn;equal"),
                (600, 8192, 2048, False, "", False, "equal"),
            ]
        )
        groups, rejected = tt.groups_from_untuned(df, "gfx950", 256, "graph")
        self.assertEqual(rejected, [])
        self.assertEqual(len(groups), 2)
        big = next(g for g in groups if g["key"]["rows_lo"] == 4096)
        self.assertEqual(
            big["specs"],
            [
                "equal@4100x8192",
                "equal@4600x8192",
                "randn@5000x8192",
                "equal@5000x8192",
            ],
        )

    def test_bad_entries_are_rejected_with_reasons_and_rest_kept(self):
        df = self._df(
            [
                (8, 1024, 2048, False, "", False, "randn"),
                (8, 8192, 64, False, "", False, "randm;equal"),
                (8, 8192, 64, False, "", False, "/no/such/file.pt"),
                (8, 8192, 64, True, "", False, "randn"),
                (8, 8192, 64, False, "sideways", False, "equal"),
            ]
        )
        groups, rejected = tt.groups_from_untuned(df, "gfx950", 256, "graph")
        reasons = " | ".join(r[1] for r in rejected)
        self.assertEqual(len(rejected), 5, reasons)
        for needle in ("must be in [1", "unknown dist", "not found", "ragged", "tie"):
            self.assertIn(needle, reasons)
        self.assertEqual(rejected[0][0], "rows=8 width=1024 k=2048 dist=randn")
        self.assertEqual([g["specs"] for g in groups], [["equal@8x8192"]])

    def test_tuned_row_without_samples_is_rejected_not_dropped(self):
        df = pd.read_csv(
            io.StringIO(
                HDR + _csv_row(4096, 2048, "sampled").replace("equal@4096x8192", "")
            )
        )
        groups, rejected = tt.groups_from_tuned(df, "gfx950", 256)
        self.assertEqual(groups, [])
        self.assertIn("no samples", rejected[0][1])

    def test_spec_round_trip_with_at_sign_in_path(self):
        spec = tt.format_spec("/data/run@2/k64_0.pt", 4100, 8192)
        self.assertEqual(tt.parse_spec(spec), ("/data/run@2/k64_0.pt", 4100, 8192))
        self.assertIsNone(tt.parse_spec("randn"))

    def test_groups_rebuilt_from_tuned_rows(self):
        df = pd.read_csv(io.StringIO(HDR + _csv_row(4096, 2048, "sampled")))
        groups, rejected = tt.groups_from_tuned(df, "gfx950", 256)
        self.assertEqual(rejected, [])
        self.assertEqual(groups[0]["specs"], ["equal@4096x8192"])
        self.assertEqual(groups[0]["key"]["tie"], None)


class TestTunerBatches(unittest.TestCase):
    def test_each_batch_runs_only_its_own_bands(self):
        tuner = tt.TopkSelectTuner()
        g1 = {
            "key": tt.band_key(
                "gfx950", 256, 4096, 8192, 2048, False, None, False, "graph"
            ),
            "specs": ["equal@4096x8192"],
        }
        g2 = {
            "key": tt.band_key(
                "gfx950", 256, 512, 8192, 2048, False, None, False, "graph"
            ),
            "specs": ["equal@512x8192"],
        }
        tuner._groups = [g1, g2]
        tuner._opts = SimpleNamespace(inject_fault=False)
        batch = pd.DataFrame([tt._untuned_row(g2)])
        seen = []

        def fake(fn, payloads, devices, timeout, on_done=None, quiet=True):
            seen.extend(p["g"]["specs"][0] for p in payloads)
            return [("ok", (None, {"why": {"_": "x"}}), 0) for _ in payloads]

        with mock.patch.object(tt, "run_isolated", fake):
            tuner.tune(batch, None, SimpleNamespace(mp=1, timeout=10))
        self.assertEqual(seen, ["equal@512x8192"])

    SPEC = "randn@64x8192"

    def _two_passes(self, first_us, second_us):
        """Run tune() with each pass's measurements faked; return the rows."""
        tuner = tt.TopkSelectTuner()
        g = {
            "key": tt.band_key(
                "gfx950", 256, 64, 8192, 64, False, None, False, "graph"
            ),
            "specs": [self.SPEC],
        }
        tuner._groups = [g]
        opts = SimpleNamespace(
            inject_fault=False, objective="no_regress", min_improvement_pct=3.0
        )
        tuner._opts = opts
        calls = []

        def outcome(us):
            summary = {
                "router": {self.SPEC: "stream"},
                "common": sorted(us),
                "us": {self.SPEC: dict(us)},
            }
            plan = {"router": summary["router"], "common": set(us)}
            return tt.decide(g, plan, summary["us"], opts, summary)

        def fake(fn, payloads, devices, timeout, on_done=None, quiet=True):
            calls.append(payloads[0]["opts"].draw if payloads else None)
            us = first_us if len(calls) == 1 else second_us
            return [("ok", outcome(us), 0) for _ in payloads]

        tuner._opts.draw = 0
        with mock.patch.object(tt, "run_isolated", fake):
            recs = tuner.tune(
                pd.DataFrame([tt._untuned_row(g)]),
                None,
                SimpleNamespace(mp=1, timeout=10),
            )
        self.assertEqual(calls, [0, 1])
        return recs

    def test_win_that_does_not_reproduce_is_not_written(self):
        first = {"decode": 10.0, "stream": 12.0}
        self.assertEqual(self._two_passes(first, {"decode": 12.5, "stream": 12.0}), [])
        self.assertEqual(self._two_passes(first, {"decode": 11.9, "stream": 12.0}), [])

    def test_win_that_reproduces_is_written(self):
        recs = self._two_passes(
            {"decode": 10.0, "stream": 12.0}, {"decode": 10.1, "stream": 12.0}
        )
        self.assertEqual([r["backend"] for r in recs], ["decode"])

    def test_band_retuned_to_no_row_loses_its_old_row(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "tuned.csv")
            Path(out).write_text(
                HDR + _csv_row(64, 64, "decode") + _csv_row(512, 64, "decode")
            )
            tuner = tt.TopkSelectTuner()
            tuner._out_file = out
            tuner.tunedf = tuner.get_tuned_gemm_list(out)
            g = {
                "key": tt.band_key(
                    "gfx950", 256, 64, 8192, 64, False, None, False, "graph"
                ),
                "specs": [self.SPEC, "equal@64x8192"],
            }
            tuner._groups = [g]
            opts = SimpleNamespace(
                inject_fault=False,
                objective="no_regress",
                min_improvement_pct=3.0,
                draw=0,
            )
            tuner._opts = opts
            us = {
                self.SPEC: {"decode": 12.5, "stream": 12.0},
                "equal@64x8192": {"decode": 10.0, "stream": 12.0},
            }
            summary = {
                "router": dict.fromkeys(us, "stream"),
                "common": ["decode", "stream"],
                "us": us,
            }
            plan = {"router": summary["router"], "common": {"decode", "stream"}}

            def fake(fn, payloads, devices, timeout, on_done=None, quiet=True):
                return [
                    ("ok", tt.decide(g, plan, us, opts, dict(summary)), 0)
                    for _ in payloads
                ]

            with mock.patch.object(tt, "run_isolated", fake):
                recs = tuner.tune(
                    pd.DataFrame([tt._untuned_row(g)]),
                    None,
                    SimpleNamespace(mp=1, timeout=10, compare=False),
                )
            self.assertEqual(recs, [])
            left = pd.read_csv(out)
            self.assertEqual(left["rows_lo"].tolist(), [512])

    def test_near_tie_that_swaps_between_passes_is_still_written(self):
        recs = self._two_passes(
            {"decode": 10.0, "sampled": 10.2, "stream": 12.5},
            {"decode": 10.3, "sampled": 10.0, "stream": 12.5},
        )
        self.assertEqual([r["backend"] for r in recs], ["sampled"])


@unittest.skipUnless(CUDA, "needs a GPU")
class TestIsolation(unittest.TestCase):
    def test_aborting_child_is_reported_not_hung(self):
        res = tt.run_isolated(
            tt.time_one_backend,
            [{"g": {}, "backend": tt.FAULT_NAME, "opts": None}],
            [0],
            timeout=120,
        )
        self.assertEqual(res[0][0], "died")
        self.assertNotEqual(res[0][1], 0)

    def test_python_error_is_reported(self):
        res = tt.run_isolated(
            tt.time_one_backend,
            [{"g": {}, "backend": "plain", "opts": None}],
            [0],
            timeout=120,
        )
        self.assertEqual(res[0][0], "error")


@unittest.skipUnless(CUDA, "needs a GPU")
class TestRecorder(unittest.TestCase):
    def test_capped_per_band_and_dtype_recorded(self):
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.dict(
                os.environ,
                {"AITER_TOPK_SELECT_RECORD": d, "AITER_TOPK_SELECT_RECORD_CALLS": "2"},
            ),
        ):
            tun._RECORD_COUNTS.clear()
            tun._RECORD_OFF = False
            for rows in (4100, 4600, 5000):
                x = torch.randn(rows, 8192, device="cuda")
                lens = torch.full((rows,), 8192, dtype=torch.int32, device="cuda")
                tun.maybe_record(x, lens, 2048, False, None, False)
            h = torch.randn(8, 8192, device="cuda").half()
            tun.maybe_record(h, lens[:8], 1, False, None, False)
            self.assertEqual(len(os.listdir(Path(d) / "samples")), 3)
            df = pd.read_csv(Path(d) / "topk_untuned.csv")
            self.assertEqual(sorted(df["dtype"]), ["float16", "float32", "float32"])
            groups, rejected = tt.groups_from_untuned(df, "gfx950", 256, "graph", d)
            self.assertEqual(rejected, [])
            self.assertEqual(
                sorted(g["key"]["dtype"] for g in groups), ["float16", "float32"]
            )

    def test_bad_settings_and_failures_never_reach_the_caller(self):
        tun._RECORD_COUNTS.clear()
        tun._ENV_WARNED.clear()
        tun._RECORD_OFF = False
        x = torch.randn(8, 1024, device="cuda")
        lens = torch.full((8,), 1024, dtype=torch.int32, device="cuda")
        bad = {
            "AITER_TOPK_SELECT_RECORD_CALLS": "abc",
            "AITER_TOPK_SELECT_RECORD_ROWS": "0",
        }
        with tempfile.TemporaryDirectory() as d:
            with (
                mock.patch.dict(os.environ, {"AITER_TOPK_SELECT_RECORD": d, **bad}),
                self.assertLogs("aiter", level="WARNING") as cm,
            ):
                tun.maybe_record(x, lens, 16, False, None, False)
            self.assertEqual(len(os.listdir(Path(d) / "samples")), 1)
        self.assertTrue(any("not a positive integer" in m for m in cm.output))
        with (
            mock.patch.dict(os.environ, {"AITER_TOPK_SELECT_RECORD": "/proc/nope"}),
            self.assertLogs("aiter", level="WARNING") as cm,
        ):
            tun.maybe_record(x, lens, 16, False, None, False)
            tun.maybe_record(x, lens, 16, False, None, False)
        self.assertTrue(tun._RECORD_OFF)
        self.assertEqual(sum("recording is off" in m for m in cm.output), 1)
        tun._RECORD_OFF = False

    def test_skips_when_capturing(self):
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.dict(os.environ, {"AITER_TOPK_SELECT_RECORD": d}),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            tun._RECORD_COUNTS.clear()
            x = torch.zeros(2, 8)
            tun.maybe_record(
                x, torch.full((2,), 8, dtype=torch.int32), 2, False, None, False
            )
            self.assertEqual(os.listdir(d), [])


@unittest.skipUnless(CUDA, "needs a GPU")
class TestPreProcess(unittest.TestCase):
    def test_creates_output_and_rejects_bad_counts(self):
        tuner = tt.TopkSelectTuner()
        with tempfile.TemporaryDirectory() as d:
            i = os.path.join(d, "in.csv")
            Path(i).write_text("rows,width,k,dist\n")
            o = os.path.join(d, "new", "out.csv")
            tuner.pre_process(tuner.parser.parse_args(["-i", i, "-o", o, "--mp", "1"]))
            self.assertEqual(list(pd.read_csv(o).columns), tuner.columns)
            self.assertTrue(tuner.untunedf.empty)
            for flag, bad in (("--draws", "0"), ("--iters", "0"), ("--warmup", "-1")):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    tuner.pre_process(
                        tuner.parser.parse_args(["-i", i, "-o", o, flag, bad])
                    )

    def test_base_timing_flags_are_honoured_not_ignored(self):
        tuner = tt.TopkSelectTuner()
        args = tuner.parser.parse_args(["--warmup", "3", "--iters", "7"])
        box = tt._ArgBox(args)
        self.assertEqual((box.warmup, box.iters), (3, 7))
        self.assertNotIn("warmup", tuner._GEMM_ONLY_FLAGS)
        self.assertNotIn("iters", tuner._GEMM_ONLY_FLAGS)


class TestSummary(unittest.TestCase):
    def test_band_with_no_usable_backend_fails_the_run(self):
        ok = {"us": {"a@1x8192": {"plain": 9.0}}, "router": {"a@1x8192": "plain"}}
        self.assertEqual(tt._failure_reason(ok), "")
        self.assertEqual(tt._failure_reason({"us": {}}), "no backend could be timed")
        broken = {**ok, "dropped": {"plain": "a@1x8192: value_mismatch"}}
        self.assertIn("router's choice failed", tt._failure_reason(broken))
        tuner = tt.TopkSelectTuner()
        tuner.tune_summary("Finished")
        tuner.failed = pd.DataFrame([{"k": 64, "reason": "no backend could be timed"}])
        with self.assertRaises(SystemExit):
            tuner.tune_summary("Finished")


class TestRunConfigStatus(unittest.TestCase):
    def test_row_without_samples_is_an_error_not_a_pass(self):
        tuner = tt.TopkSelectTuner()
        row = pd.read_csv(io.StringIO(HDR + _csv_row(4096, 2048, "sampled")))
        row["dists"] = ""
        tuner.untunedf = row
        args = SimpleNamespace(
            mp=1,
            timeout=10,
            verbose=False,
            mode="graph",
            warmup=1,
            iters=1,
            objective="no_regress",
            min_improvement_pct=3.0,
        )
        spawned = []

        def fake(fn, payloads, devices, timeout, on_done=None, quiet=True):
            spawned.extend(payloads)
            return []

        with (
            mock.patch.object(tt, "run_isolated", fake),
            mock.patch.object(tuner, "get_gfx", return_value="gfx950"),
            mock.patch.object(tuner, "get_cu_num", return_value=256),
        ):
            (res,) = tuner.run_config(args)
        self.assertEqual(spawned, [])
        self.assertTrue(res["status"].startswith("error: the row lists no samples"))

    def _m(self, table, router, t, u, oracle="ok", fail=""):
        return {
            "table": table,
            "router": router,
            "table_us": t,
            "router_us": u,
            "oracle": oracle,
            "fail": fail,
        }

    def test_statuses(self):
        cases = {
            "ok": {"a@1x8192": self._m("sampled", "plain", 10.0, 40.0)},
            "mismatch": {
                "a@1x8192": self._m("sampled", "plain", 10.0, 40.0, "value_mismatch")
            },
            "error: the table's": {"a@1x8192": self._m("decode", "stream", 10.6, 10.0)},
            "error: a@1x8192 was not timed": {
                "a@1x8192": self._m("sampled", "plain", None, 40.0, fail="x")
            },
        }
        for want, per in cases.items():
            with self.subTest(want=want):
                _text, status, _us = tt.summarize_run_config("band", per)
                self.assertTrue(status.startswith(want), status)

    def test_router_noise_is_not_a_regression(self):
        per = {"a@1x8192": self._m("decode", "stream", 10.3, 10.0)}
        self.assertEqual(tt.summarize_run_config("band", per)[1], "ok")

    def _fb(self, t, u, router_failed):
        return {
            **self._m("sampled", "plain", t, u),
            "fallback": True,
            "router_failed": router_failed,
        }

    def test_fallback_row_is_ok_while_the_router_still_fails(self):
        # Slower than the router is expected: the router's answer is wrong.
        per = {"a@1x8192": self._fb(40.0, None, "a@1x8192: value_mismatch")}
        text, status, _us = tt.summarize_run_config("band", per)
        self.assertEqual(status, "ok")
        self.assertEqual(_us, 40.0)
        self.assertIn("plain still fails on a@1x8192: value_mismatch", text)
        self.assertIn("table 40.0 us (the router's choice failed)", text)

    def test_fallback_row_is_stale_once_the_router_passes_again(self):
        per = {
            "a@1x8192": self._fb(40.0, 10.0, None),
            "b@1x8192": self._fb(40.0, 10.0, None),
        }
        status = tt.summarize_run_config("band", per)[1]
        self.assertTrue(status.startswith("error: the router's choice (plain)"), status)

    def test_fallback_row_must_still_be_correct_and_timed(self):
        bad = {**self._fb(40.0, None, "x: crashed"), "oracle": "value_mismatch"}
        self.assertTrue(
            tt.summarize_run_config("band", {"a@1x8192": bad})[1].startswith("mismatch")
        )
        untimed = self._fb(None, None, "x: crashed")
        self.assertTrue(
            tt.summarize_run_config("band", {"a@1x8192": untimed})[1].startswith(
                "error: a@1x8192 was not timed"
            )
        )

    def test_run_config_flags_rows_with_no_router_time_as_fallback(self):
        tuner = tt.TopkSelectTuner()
        rows = pd.read_csv(
            io.StringIO(
                HDR + _csv_row(4096, 2048, "sampled") + _csv_row(512, 2048, "plain")
            )
        )
        rows["dists"] = "randn@4096x8192"
        rows.loc[0, ["us_default", "worst_ratio_vs_default"]] = float("nan")
        tuner.untunedf = rows
        args = SimpleNamespace(
            mp=1,
            timeout=10,
            verbose=False,
            mode="graph",
            warmup=1,
            iters=1,
            objective="no_regress",
            min_improvement_pct=3.0,
        )
        seen = []

        def fake(fn, payloads, devices, timeout, on_done=None, quiet=True):
            seen.extend(p["g"]["fallback"] for p in payloads)
            return [("died", None, 0)] * len(payloads)

        with (
            mock.patch.object(tt, "run_isolated", fake),
            mock.patch.object(tuner, "get_gfx", return_value="gfx950"),
            mock.patch.object(tuner, "get_cu_num", return_value=256),
        ):
            tuner.run_config(args)
        self.assertEqual(seen, [True, False])
        self.assertTrue(tuner.run_config_failed)


class TestPathsAndProfile(unittest.TestCase):
    def test_relative_sample_path_resolves_against_csv_dir(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "samples").mkdir()
            torch.save(
                {
                    "input": torch.randn(4, 64),
                    "row_lens": torch.full((4,), 64, dtype=torch.int32),
                    "rows": 4,
                    "width": 64,
                    "k": 8,
                    "ragged": False,
                },
                Path(d) / "samples" / "s.pt",
            )
            df = pd.DataFrame(
                [{"rows": 4, "width": 64, "k": 8, "dist": "samples/s.pt"}]
            )
            groups, rejected = tt.groups_from_untuned(df, "gfx950", 256, "graph", d)
            self.assertEqual(rejected, [])
            want = os.path.normpath(os.path.join(d, "samples", "s.pt"))
            self.assertEqual(groups[0]["specs"], [f"{want}@4x64"])
            _g, rejected = tt.groups_from_untuned(df, "gfx950", 256, "graph", "/")
            self.assertIn("not found", rejected[0][1])

    def test_samples_load_without_unpickling_code(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.pt")
            torch.save(
                {
                    "input": torch.randn(4, 64),
                    "row_lens": torch.full((4,), 64, dtype=torch.int32),
                    "width": 64,
                    "k": 8,
                    "extra": Payload(),
                },
                p,
            )
            why = tt.check_entry(4, 64, 8, False, None, p)
        self.assertIn("cannot load", why)

    def test_presets_are_timed_over_independent_draws(self):
        cpu = torch.device("cpu")
        a = list(tt.sample_chunks("randn@4x64", cpu))
        b = list(tt.sample_chunks("randn@4x64", cpu, draw=1))
        self.assertEqual([len(c["xs"]) for c in a], [tt.MAX_COPIES] * 4)
        self.assertEqual(sum(len(c["xs"]) for c in a), tt.MAX_DRAWS)
        self.assertFalse(torch.equal(a[0]["xs"][0][0], a[0]["xs"][1][0]))
        self.assertFalse(torch.equal(a[0]["xs"][0][0], b[0]["xs"][0][0]))
        again = list(tt.sample_chunks("randn@4x64", cpu))
        self.assertTrue(torch.equal(a[2]["xs"][3][0], again[2]["xs"][3][0]))
        xs, ncalls = tt._rotation(a[0])
        self.assertEqual((len(xs), ncalls), (tt.MAX_COPIES, tt.MAX_COPIES))

    def test_large_samples_get_the_minimum_draws_in_bounded_chunks(self):
        cpu = torch.device("cpu")
        nbytes = 4 * 64 * 4
        with mock.patch.object(tt, "DRAW_BYTES", nbytes):
            with mock.patch.object(tt, "CHUNK_BYTES", 5 * nbytes):
                chunks = list(tt.sample_chunks("randn@4x64", cpu))
            (one,) = tt.sample_chunks("randn@4x64", cpu)
        self.assertEqual([len(c["xs"]) for c in chunks], [5, 5, 5, 1])
        whole = torch.cat([x for c in chunks for x, _ in c["xs"]])
        self.assertTrue(torch.equal(whole, torch.cat([x for x, _ in one["xs"]])))

    def test_profile_lists_times_and_dropped(self):
        g = {
            "key": tt.band_key("gfx950", 256, 1, 8192, 64, False, None, False, "graph"),
            "specs": ["equal@1x8192"],
        }
        summary = {
            "router": {"equal@1x8192": "stream"},
            "us": {"equal@1x8192": {"sampled": 5.6, "stream": 46.8}},
            "dropped": {"plain": "equal@1x8192: graph capture failed"},
        }
        tuner = tt.TopkSelectTuner()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.csv")
            tuner._write_profile(path, [(g, 1, summary)])
            tuner._write_profile(path, [(g, 2, summary)])
            df = pd.read_csv(path)
        self.assertEqual(len(df), 6)
        self.assertEqual(sorted(df["pass"].unique().tolist()), [1, 2])
        self.assertIn("dropped: equal@1x8192", " ".join(df["status"].astype(str)))


def _record_four(d):
    os.environ["AITER_TOPK_SELECT_RECORD"] = d
    for rows in (1, 8, 64, 512):
        x = torch.randn(rows, 1024, device="cuda")
        lens = torch.full((rows,), 1024, dtype=torch.int32, device="cuda")
        tun.maybe_record(x, lens, 16, False, None, False)
    return os.getpid()


@unittest.skipUnless(CUDA, "needs a GPU")
class TestConcurrentProcesses(unittest.TestCase):
    def test_two_ranks_record_into_one_directory(self):
        with tempfile.TemporaryDirectory() as d:
            res = tt.run_isolated(_record_four, [{"d": d}, {"d": d}], [0, 0], 300)
            self.assertEqual([r[0] for r in res], ["ok", "ok"], res)
            df = pd.read_csv(Path(d) / "topk_untuned.csv")
            self.assertEqual(len(df), 8)
            self.assertEqual(df["dist"].nunique(), 8)
            for rel in df["dist"]:
                self.assertTrue((Path(d) / rel).is_file(), rel)
            groups, rejected = tt.groups_from_untuned(df, "gfx950", 256, "graph", d)
            self.assertEqual(rejected, [])
            self.assertEqual(sum(len(g["specs"]) for g in groups), 8)

    def test_hung_child_is_killed_at_timeout(self):
        import subprocess

        res = tt.run_isolated(subprocess.run, [{"args": ["sleep", "60"]}], [0], 5)
        self.assertEqual(res[0][0], "timeout")


if __name__ == "__main__":
    unittest.main()
