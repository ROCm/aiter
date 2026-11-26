# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
The FlyDSL MXFP6 MoE tuner (gemm_moe_tune.py --mxfp6-flydsl), host code only: the
search space of a stage, the rows it writes and reads back, and the options it
refuses. No kernel is built or launched.

Run:
    python3 -m unittest op_tests.tuning_tests.test_mxfp6_moe_tuner -v
"""

import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

from aiter.ops.flydsl import mxfp6_moe_policy as policy
from aiter.ops.flydsl.mxfp6_moe_policy import Mxfp6MoeConfig, Mxfp6StageConfig

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TUNE_DIR = os.path.join(REPO, "csrc", "ck_gemm_moe_2stages_codegen")
RESULTS = ["block_m", "h_layout", "us1", "kernelName1", "us2", "kernelName2"]
RESULTS += ["us", "tflops", "bw"]


def _stage(**kw):
    base = {
        "tile_m": 256,
        "tile_n": 256,
        "ring": 3,
        "group_m": 4,
        "nt_weights": True,
        "pad_rows": False,
        "warm": 0,
        "half_tiles": False,
    }
    return Mxfp6StageConfig(**{**base, **kw})


def _config(route_m=256, h_layout="tile_step", **stage2):
    return Mxfp6MoeConfig(
        route_m=route_m,
        stage1=_stage(tile_m=route_m),
        stage2=_stage(tile_m=route_m, **stage2),
        h_layout=h_layout,
    )


class _TunerCase(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, TUNE_DIR)
        try:
            self.tune = importlib.import_module("gemm_moe_tune")
            self.search = importlib.import_module(
                "csrc.ck_gemm_moe_2stages_codegen.mxfp6_moe_search"
            )
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"gemm_moe_tune not importable: {e}")
        finally:
            sys.path.pop(0)

    def tuner(self):
        return self.tune.Mxfp6FlydslTuner(
            "mxfp6FlydslTuner", list(policy.TUNED_KEYS), list(RESULTS), "test"
        )


class TestSearchSpace(_TunerCase):
    """stage_variants: what one dimension of a stage's search may be set to."""

    def variants(self, base, dim, **kw):
        space = {
            "stage1": True,
            "six": True,
            "route_m": 256,
            "n": 4096,
            "slab": 256,
            "a_bytes": 1 << 20,
        }
        return self.search.stage_variants(base, dim, **{**space, **kw})

    def test_dimensions(self):
        dims = self.search.stage_dims
        shared = ["tile", "group_m", "nt_weights"]
        self.assertEqual(dims(True, True, True), [*shared, "pad_rows"])
        self.assertEqual(dims(True, False, False), shared)
        self.assertEqual(dims(False, True, True), [*shared, "warm"])
        # a row-major h has nothing to warm
        self.assertEqual(dims(False, True, False), shared)
        self.assertEqual(dims(False, False, False), shared)

    def test_tiles_cover_whole_weight_slabs(self):
        # weights prepared in 256-column slabs: no 128-column tile
        tiles = self.variants(_stage(), "tile")
        self.assertEqual({(c.tile_m, c.tile_n) for c in tiles}, {(256, 256)})
        self.assertEqual({c.ring for c in tiles}, {2, 3})
        # 128-column slabs: both widths
        tiles = self.variants(_stage(), "tile", slab=128)
        self.assertEqual({c.tile_n for c in tiles}, {128, 256})
        # ... unless 256 does not divide the stage's columns
        tiles = self.variants(_stage(), "tile", slab=128, n=6272)
        self.assertEqual({c.tile_n for c in tiles}, {128})

    def test_tiles_keep_the_routing_tile(self):
        for route_m in (128, 256):
            tiles = self.variants(_stage(tile_m=route_m), "tile", route_m=route_m)
            self.assertEqual({c.route_m for c in tiles}, {route_m})
            self.assertFalse(any(c.half_tiles for c in tiles))

    def test_eight_bit_tiles(self):
        tiles = self.variants(_stage(tile_n=128), "tile", six=False, slab=128)
        # 256 x 256 fits two LDS stages only
        wide = {c.ring for c in tiles if (c.tile_m, c.tile_n) == (256, 256)}
        self.assertEqual(wide, {2})
        # stage 1 on 256-row routing tiles: also halves
        halves = [c for c in tiles if c.half_tiles]
        self.assertTrue(halves)
        self.assertEqual({(c.tile_m, c.route_m) for c in halves}, {(128, 256)})
        for kw in ({"stage1": False}, {"route_m": 128}):
            tiles = self.variants(
                _stage(tile_m=kw.get("route_m", 256), tile_n=128),
                "tile",
                six=False,
                slab=128,
                **kw,
            )
            self.assertFalse(any(c.half_tiles for c in tiles))

    def test_other_dimensions_change_one_field(self):
        base = _stage()
        for dim, field, values in (
            ("group_m", "group_m", {1, 2, 4, 8, 16}),
            ("nt_weights", "nt_weights", {False, True}),
            ("pad_rows", "pad_rows", {False, True}),
            ("warm", "warm", {0, 1, 2, 4, 8}),
        ):
            got = self.variants(base, dim)
            self.assertEqual({getattr(c, field) for c in got}, values)
            for c in got:
                self.assertEqual(
                    c, type(base)(**{**vars(base), field: getattr(c, field)})
                )
            self.assertIn(base, got)

    def test_no_padded_copy_past_32_bit_offsets(self):
        # the copy is 4/3 of the rows and must stay below 4 GiB - 4 KiB
        fits = 3 * ((1 << 30) - 1024)
        got = self.variants(_stage(), "pad_rows", a_bytes=fits)
        self.assertEqual([c.pad_rows for c in got], [False, True])
        got = self.variants(_stage(), "pad_rows", a_bytes=fits + 3)
        self.assertEqual([c.pad_rows for c in got], [False])

    def test_unknown_dimension(self):
        with self.assertRaises(ValueError):
            self.variants(_stage(), "tile_k")

    def test_every_variant_has_a_name(self):
        for stage1, dim in ((True, "tile"), (True, "pad_rows"), (False, "warm")):
            for c in self.variants(_stage(), dim, stage1=stage1, slab=128):
                name = policy.stage_kernel_name(stage1, c)
                self.assertEqual(policy.stage_config_from_name(name), (stage1, c))


class TestFastest(_TunerCase):
    """fastest: the pair a shape's row names."""

    def test_smallest_total_time(self):
        # b wins three of four draws, a the total
        a, b = _config(128), _config(256)
        self.assertEqual(
            self.search.fastest({a: [100.0] * 4, b: [99.0, 99.0, 99.0, 110.0]}), a
        )
        # +11%, -4%, +10%, -5% against a: faster in total
        us = {a: [111.0, 96.0, 110.0, 95.0], b: [100.0] * 4}
        self.assertEqual(self.search.fastest(us), b)

    def test_the_heuristic_is_a_candidate(self):
        # listed first: named where nothing found is faster
        heuristic, other = _config(128), _config(256)
        us = {heuristic: [100.0, 100.0], other: [100.0, 100.0]}
        self.assertEqual(self.search.fastest(us), heuristic)
        us[other] = [101.0, 100.0]
        self.assertEqual(self.search.fastest(us), heuristic)
        us[other] = [99.0, 100.0]
        self.assertEqual(self.search.fastest(us), other)


class TestWork(_TunerCase):
    def test_moe_work(self):
        flops, nbytes = self.search.moe_work(4096, 6144, 2048, 64, 4, "e2m3")
        self.assertEqual(flops, 6 * 4096 * 4 * 6144 * 2048)
        weights = 64 * 3 * 6144 * 2048 * 6 // 8
        x_and_y = 4096 * 6144 * 2 * 2
        routed = 4096 * 4 * (2048 * 6 // 8 + 6144 * 2) * 2
        self.assertEqual(nbytes, weights + x_and_y + routed)
        # 8-bit activations: only the h rows grow
        more = self.search.moe_work(4096, 6144, 2048, 64, 4, "e4m3")[1] - nbytes
        self.assertEqual(more, 4096 * 4 * 2048 * 2 // 8 * 2)


class TestRows(_TunerCase):
    """The rows the tuner writes: read back by the op's loader as the same kernels."""

    def key(self, token, gfx="gfx950", a="mxfp6_e2m3"):
        return {
            "gfx": gfx,
            "cu_num": 256,
            "token": token,
            "model_dim": 6144,
            "inter_dim": 2048,
            "expert": 64,
            "topk": 4,
            **policy.TUNED_CALL,
            "q_dtype_a": a,
            "q_dtype_w": "mxfp6_e2m3",
        }

    def result(self, config, us=500.0):
        return self.search.ShapeResult(
            heuristic=_config(),
            best=config,
            us=us,
            us_heuristic=us * 1.1,
            us1=us * 0.6,
            us2=us * 0.3,
            candidates=10,
            pairs=5,
        )

    def row(self, tuner, token, config, us=500.0, **key):
        tuner._search = self.search
        row = self.key(token, **key)
        return tuner._shape_row(row, policy.row_call(row), self.result(config, us))

    def write(self, tuner, rows, path):
        tuner.result_to_csv(pd.DataFrame(rows, columns=tuner.columns), path)

    def test_columns(self):
        tuner = self.tuner()
        self.assertEqual(tuner.columns, [*policy.TUNED_KEYS, *RESULTS])
        self.assertTrue(set(policy.TUNED_RESULTS) <= set(RESULTS))

    def test_shape_row(self):
        tuner = self.tuner()
        config = _config(128, "rows", group_m=1)
        row = self.row(tuner, 4096, config, us=500.0)
        self.assertEqual(sorted(row), sorted(tuner.columns))
        self.assertEqual(policy.row_config(row), config)
        self.assertEqual((row["us1"], row["us2"], row["us"]), (300.0, 150.0, 500.0))
        flops, nbytes = self.search.moe_work(4096, 6144, 2048, 64, 4, "e2m3")
        self.assertEqual(row["tflops"], round(flops / 500e-6 / 1e12, 2))
        self.assertEqual(row["bw"], round(nbytes / 500e-6 / 1e9, 2))

    def test_failed_row_is_not_written(self):
        tuner = self.tuner()
        failed = tuner._failed_row(self.key(8192), RuntimeError("no memory"))
        self.assertEqual(sorted(failed), sorted(tuner.columns))
        self.assertEqual(failed["us"], tuner.INVALID_TIME)
        self.assertIn("RuntimeError: no memory", failed["kernelName1"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tuned.csv")
            self.write(tuner, [self.row(tuner, 4096, _config()), failed], path)
            self.assertEqual(len(policy.load_tuned_configs(path)), 1)
        self.assertEqual(list(tuner.success["token"]), [4096])
        self.assertEqual(list(tuner.failed["token"]), [8192])

    def test_written_rows_load_as_the_kernels(self):
        tuner = self.tuner()
        configs = {
            (4096, "mxfp6_e2m3"): _config(128, "rows", group_m=1),
            (8192, "mxfp6_e2m3"): _config(256, warm=4),
            (8192, "torch.float8_e4m3fn"): _config(256, tile_n=128, ring=2),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tuned.csv")
            rows = [self.row(tuner, t, c, a=a) for (t, a), c in configs.items()]
            self.write(tuner, rows, path)
            loaded = policy.load_tuned_configs(path)
            text = pd.read_csv(path, dtype=str)
        self.assertEqual(len(loaded), 3)
        for (token, a), config in configs.items():
            a_fmt = "e4m3" if "float8" in a else "e2m3"
            key = ("gfx950", 256, token, 6144, 2048, 64, 4, a_fmt, "e2m3")
            self.assertEqual(loaded[key], config)
        # integers stay integers
        for column in ("cu_num", "token", "expert", "use_g1u1", "block_m"):
            self.assertTrue(text[column].str.fullmatch(r"\d+").all(), column)

    def test_retuned_row_replaces_and_other_rows_stay(self):
        tuner = self.tuner()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tuned.csv")
            first = [
                self.row(tuner, 4096, _config()),
                self.row(tuner, 8192, _config(warm=1)),
                self.row(tuner, 4096, _config(warm=2), gfx="gfx1250"),
            ]
            self.write(tuner, first, path)
            again = self.tuner()
            self.write(again, [self.row(again, 4096, _config(128, "rows"), 400)], path)
            loaded = policy.load_tuned_configs(path)
            frame = pd.read_csv(path)
        shape = (6144, 2048, 64, 4, "e2m3", "e2m3")
        self.assertEqual(
            loaded,
            {
                ("gfx950", 256, 4096, *shape): _config(128, "rows"),
                ("gfx950", 256, 8192, *shape): _config(warm=1),
                ("gfx1250", 256, 4096, *shape): _config(warm=2),
            },
        )
        self.assertEqual(list(frame.columns), again.columns)
        self.assertEqual(sorted(frame["us"]), [400.0, 500.0, 500.0])


class TestOptions(_TunerCase):
    def parse(self, *argv):
        argv = ["gemm_moe_tune.py", *argv]
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"):
            return self.tuner().parse_args()

    def test_defaults(self):
        args = self.parse("--mxfp6-flydsl")
        self.assertTrue(args.untune_file.endswith("untuned_mxfp6_moe.csv"))
        self.assertTrue(args.tune_file.endswith("tuned_mxfp6_moe.csv"))
        self.assertEqual(args.mxfp6_search_mode, "descent")
        # one call of a fixed seed per shape
        options = self.search.SearchOptions()
        self.assertEqual((options.draws, options.seed), (1, 2026))

    def test_refused(self):
        for argv in (
            ["--compare"],
            ["--run_config"],
            ["--e2e_tune"],
            ["--grouped-gemm"],
            ["--mxfp4-flydsl"],
            ["--ep-size", "8"],
            ["--swiglu_limit", "7"],
            ["--select", "power"],
            ["--mxfp6-search-mode", "staged"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self.parse("--mxfp6-flydsl", *argv)

    def test_shipped_rows_load(self):
        # every shipped row names a config, of a shape of the untuned list
        tuned = policy.load_tuned_configs(
            os.path.join(REPO, "aiter/configs/tuned_mxfp6_moe.csv")
        )
        untuned = pd.read_csv(os.path.join(REPO, "aiter/configs/untuned_mxfp6_moe.csv"))
        calls = {policy.row_call(row) for _, row in untuned.iterrows()}
        self.assertTrue({key[2:] for key in tuned} <= calls)

    def test_untuned_rows_are_mxfp6_calls(self):
        untuned = pd.read_csv(os.path.join(REPO, "aiter/configs/untuned_mxfp6_moe.csv"))
        self.assertEqual(list(untuned.columns), list(policy.TUNED_KEYS[2:]))
        calls = [policy.row_call(row) for _, row in untuned.iterrows()]
        self.assertEqual(len(set(calls)), len(calls))


if __name__ == "__main__":
    unittest.main()
