# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Level 1 (no GPU): knob rules and search of the gfx1250 grouped MoE tuner
(csrc/ck_gemm_moe_2stages_codegen/grouped_fmoe_search.py)."""

import csv
import glob
import os
import shutil
import tempfile
import unittest
from unittest import mock

from csrc.ck_gemm_moe_2stages_codegen import grouped_fmoe_search as gfs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# Kimi-K3 routed MoE, one GPU: 896 experts, top-16, MXFP4 a4w4.
K3 = gfs.Shape(
    token=16384, model_dim=3584, inter_dim=3072, expert=896, topk=16, a_fp4=True
)
# The shipped 16384-token row of that shape.
K3_ROW = {
    "tile_m": 128,
    "tile_n": 256,
    "tile_k": 256,
    "tile_m2": 128,
    "tile_n2": 256,
    "tile_k2": 256,
    "m_warp": 2,
    "n_warp": 4,
    "m_warp2": 1,
    "n_warp2": 4,
    "num_buffers": 3,
    "num_buffer_stage2": 3,
    "cluster_m2": 4,
    "cluster_n2": 1,
    "waves_per_tensor_tdm2": 1,
}


def k3_cfg(**changes):
    return {**gfs.resolve_config(K3_ROW), **changes}


class TestKnobRules(unittest.TestCase):
    def test_blank_cells_resolve_like_the_runtime(self):
        cfg = gfs.resolve_config(
            {
                "tile_m": "128",
                "tile_n": "",
                "n_warp": "2",
                "tile_m2": "",
                "cluster_n": "2",
                "cluster_n2": "",
                "waves_per_tensor_tdm": "1",
                "waves_per_tensor_tdm2": "nan",
                "split_k1": "0",
            }
        )
        self.assertEqual(cfg["tile_n"], 128)  # n_warp * 64
        self.assertEqual(cfg["tile_k"], 128)
        self.assertEqual(cfg["tile_m2"], 128)
        self.assertEqual(cfg["cluster_n2"], 2)
        self.assertEqual(cfg["cluster_m"], -1)
        self.assertEqual(cfg["waves_per_tensor_tdm2"], 1)
        self.assertEqual(cfg["split_k1"], 1)
        self.assertEqual(set(cfg), set(gfs.KNOB_COLUMNS))

    def test_shipped_rows_pass_the_rules(self):
        # The rules may not be stricter than the kernels: every gfx1250 row the
        # runtime ships must pass them.
        tables = glob.glob(
            os.path.join(ROOT, "aiter", "configs", "**", "*tuned_grouped_fmoe*.csv"),
            recursive=True,
        )
        checked = 0
        for path in tables:
            for row in gfs.read_rows(path):
                if not str(row.get("gfx", "")).startswith("gfx1250"):
                    continue
                shape = gfs.shape_of(row)
                why = gfs.check_config(gfs.resolve_config(row), shape)
                self.assertIsNone(
                    why, f"{os.path.basename(path)} token {row['token']}: {why}"
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_rules_reject_configs_the_kernels_reject(self):
        cases = {
            # TDM ownership: 8 waves cannot share 1 wave per tensor.
            "waves per tensor": k3_cfg(waves_per_tensor_tdm=1),
            "four waves": k3_cfg(tdm_as_in_prologue=1),
            "LDS": k3_cfg(tile_n=512, num_buffers=4),
            "does not divide": k3_cfg(tile_n2=384, n_warp2=4),
            "plain grid": k3_cfg(split_k2=2),
            "fp4 activations": {**k3_cfg(a_preshuffle=1)},
            "cluster_m 1": k3_cfg(persistent_workers2=256),
            "VGPR": k3_cfg(tile_m=256, tile_n=512, m_warp=1, n_warp=2),
        }
        shapes = {"fp4 activations": gfs.Shape(16384, 3584, 3072, 896, 16, a_fp4=False)}
        for expect, cfg in cases.items():
            why = gfs.check_config(cfg, shapes.get(expect, K3))
            self.assertIsNotNone(why, expect)
            self.assertIn(expect, why)
        self.assertIsNone(gfs.check_config(k3_cfg(), K3))
        self.assertIsNone(
            gfs.check_config(k3_cfg(persistent_workers2=256, cluster_m2=-1), K3)
        )

    def test_equivalent_rows_share_a_key(self):
        base = gfs.effective_key(k3_cfg(), K3)
        self.assertEqual(gfs.effective_key(k3_cfg(cluster_m=1), K3), base)
        self.assertEqual(gfs.effective_key(k3_cfg(waves_per_tensor_tdm=2), K3), base)
        # cluster_n=3 does not divide 3584/256 = 14 N tiles, so it is off.
        self.assertEqual(gfs.effective_key(k3_cfg(cluster_n2=3), K3), base)
        self.assertNotEqual(gfs.effective_key(k3_cfg(cluster_m2=-1), K3), base)
        # gemm1 with cluster_n 4 and a blank cluster_m clusters 4x4.
        self.assertEqual(
            gfs.effective_key(k3_cfg(cluster_n=4), K3),
            gfs.effective_key(k3_cfg(cluster_n=4, cluster_m=4), K3),
        )

    def test_moves_keep_every_knob(self):
        start = k3_cfg()
        for group, gen in gfs.GROUPS:
            for label, cfg in gen(start, K3):
                self.assertEqual(set(cfg), set(gfs.KNOB_COLUMNS), label)


class TestSeeds(unittest.TestCase):
    def _table(self, tokens, cu_num=256):
        # (token, tile_m) rows of the K3 shape, plus one row of another
        # activation, which must never become a seed.
        rows = [self._row(token) | {"tile_m": tile_m} for token, tile_m in tokens]
        rows.append(self._row(16384) | {"act_type": "ActivationType.Silu"})
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, newline=""
        ) as f:
            w = csv.DictWriter(
                f, fieldnames=["gfx", "cu_num", *rows[0], "tile_n", "us"]
            )
            w.writeheader()
            for row in rows:
                w.writerow(
                    {"gfx": "gfx1250", "cu_num": cu_num, **row, "tile_n": 512, "us": 1}
                )
        self.addCleanup(os.unlink, f.name)
        return f.name

    def _row(self, token):
        return {
            "token": token,
            "model_dim": 3584,
            "inter_dim": 3072,
            "expert": 896,
            "topk": 16,
            "act_type": "ActivationType.Situv2",
            "dtype": "torch.bfloat16",
            "q_dtype_a": gfs.FP4,
            "q_dtype_w": gfs.FP4,
            "q_type": "QuantType.per_1x32",
            "ep_fused": float("nan"),
        }

    def test_row_in_use_comes_first(self):
        table = self._table([(4096, 64), (16384, 128)])
        seeds = gfs.make_seeds(
            self._row(16384), gfs.shape_of(self._row(16384), 256), [table], "gfx1250"
        )
        self.assertIn("token 16384", seeds[0][0])
        self.assertEqual(seeds[0][1]["tile_m"], 128)
        self.assertTrue(any(s[1]["tile_m"] == 64 for s in seeds[1:]), seeds)
        self.assertEqual(len({gfs.effective_key(c, K3) for _, c in seeds}), len(seeds))

    def test_missing_bucket_starts_from_the_largest_below_like_the_lookup(self):
        table = self._table([(4096, 64), (2048, 32)])
        row = self._row(8192)
        seeds = gfs.make_seeds(row, gfs.shape_of(row, 256), [table], "gfx1250")
        self.assertIn("token 4096", seeds[0][0])

    def test_rows_of_another_cu_count_serve_when_none_match(self):
        table = self._table([(1024, 32), (4096, 64)], cu_num=192)
        row = self._row(2048)
        seeds = gfs.make_seeds(row, gfs.shape_of(row, 256), [table], "gfx1250")
        self.assertIn("token 1024", seeds[0][0])
        self.assertTrue(any("token 4096" in label for label, _ in seeds), seeds)

    def test_no_row_starts_from_the_runtime_default(self):
        row = self._row(8192)
        seeds = gfs.make_seeds(row, gfs.shape_of(row, 256), [], "gfx1250")
        self.assertEqual(seeds[0][0], "runtime default")
        self.assertEqual((seeds[0][1]["tile_m"], seeds[0][1]["tile_k"]), (64, 128))


def _cost(cfg):
    """A synthetic pipeline: gemm1 likes tile_n 512 and cluster_m 2, gemm2
    likes no M cluster and rolled_drain. A tile_n2 of 512 crashes the worker."""
    g1 = (
        1000.0
        * (0.9 if cfg["tile_n"] == 512 else 1.0)
        * (0.93 if cfg["cluster_m"] == 2 else 1.0)
    )
    g2 = (
        800.0
        * (1.4 if cfg["cluster_m2"] not in (-1, 1) else 1.0)
        * (0.9 if cfg["rolled_drain"] else 1.0)
    )
    return g1, g2


class FakeBench:
    def __init__(self, cost=_cost, clock=None, step=0.0):
        self.cost, self.calls, self.clock, self.step = cost, [], clock, step

    def measure(self, configs, rounds):
        self.calls.append((rounds, len(configs)))
        if self.clock is not None:
            self.clock[0] += self.step
        out = []
        for cfg in configs:
            if cfg["tile_n2"] == 512:
                out.append(
                    {"ok": False, "error": "worker died (exit -6) on this config"}
                )
                continue
            g1, g2 = self.cost(cfg)
            out.append(
                {
                    "ok": True,
                    "stages": {"gemm1": [g1] * rounds, "gemm2": [g2] * rounds},
                    "total": [g1 + g2] * rounds,
                }
            )
        return out


class TestSearch(unittest.TestCase):
    def _seeds(self):
        default = gfs.resolve_config(
            {"tile_m": 64, "tile_n": 256, "tile_k": 128, "n_warp": 4, "num_buffers": 3}
        )
        return [("table row", k3_cfg()), ("runtime default", default)]

    def test_finds_the_cross_stage_optimum(self):
        bench = FakeBench()
        res = gfs.search(
            K3, self._seeds(), bench.measure, gfs.SearchOptions(), log=lambda m: None
        )
        self.assertTrue(res["ok"], res.get("error"))
        cfg = res["config"]
        self.assertEqual((cfg["tile_n"], cfg["cluster_m"]), (512, 2))
        self.assertIn(cfg["cluster_m2"], (-1, 1))
        self.assertEqual(cfg["rolled_drain"], 1)
        self.assertLess(res["us"], res["start_us"])
        self.assertIsNone(gfs.check_config(cfg, K3))
        # The last call confirms start vs result with twice the rounds.
        self.assertEqual(bench.calls[-1], (10, 2))
        # Crashed candidates are recorded, not fatal.
        self.assertTrue(
            any(not r["ok"] and "worker died" in r["error"] for r in res["records"])
        )

    def test_keeps_the_start_when_nothing_beats_it(self):
        bench = FakeBench(cost=lambda cfg: (1000.0, 800.0))
        res = gfs.search(
            K3, self._seeds(), bench.measure, gfs.SearchOptions(), log=lambda m: None
        )
        self.assertEqual(res["label"], "table row")
        self.assertEqual(res["config"], k3_cfg())

    def test_gains_below_min_gain_are_not_taken(self):
        tiny = lambda cfg: (1000.0 * (0.995 if cfg["rolled_drain"] else 1.0), 800.0)
        res = gfs.search(
            K3,
            self._seeds(),
            FakeBench(cost=tiny).measure,
            gfs.SearchOptions(),
            log=lambda m: None,
        )
        self.assertEqual(res["config"]["rolled_drain"], 0)

    def test_budget_bounds_the_search_but_not_the_final_check(self):
        clock = [0.0]
        bench = FakeBench(clock=clock, step=100.0)
        opts = gfs.SearchOptions(budget_s=250.0)
        res = gfs.search(
            K3,
            self._seeds(),
            bench.measure,
            opts,
            log=lambda m: None,
            clock=lambda: clock[0],
        )
        self.assertLessEqual(len(bench.calls), 5)
        self.assertEqual(bench.calls[-1][0], 10)
        self.assertTrue(res["ok"])

    def test_descends_from_the_start_point_too(self):
        # The runtime default (tile 64x256x128) is fastest as is but a dead end
        # two moves away from the start row; the optimum is two other moves
        # away from the start row.
        def cost(cfg):
            if cfg["tile_m"] == 64 and cfg["tile_k"] == 128:
                return 900.0, 0.0
            g1 = 1000.0 * (0.8 if cfg["cluster_m"] == 2 else 1.0)
            return g1 * (0.8 if cfg["rolled_drain"] else 1.0), 0.0

        res = gfs.search(
            K3,
            self._seeds(),
            FakeBench(cost=cost).measure,
            gfs.SearchOptions(),
            log=lambda m: None,
        )
        cfg = res["config"]
        self.assertEqual(
            (cfg["tile_k"], cfg["cluster_m"], cfg["rolled_drain"]), (256, 2, 1)
        )
        self.assertEqual(res["history"][0], "table row")

    def test_batches_respect_the_batch_size(self):
        bench = FakeBench()
        gfs.search(
            K3,
            self._seeds(),
            bench.measure,
            gfs.SearchOptions(batch=4),
            log=lambda m: None,
        )
        self.assertTrue(all(n <= 4 for _, n in bench.calls))


class TestLookupKey(unittest.TestCase):
    def test_blank_cells_match_anything(self):
        key = gfs.lookup_key(
            {
                "gfx": "gfx1250",
                "cu_num": 256,
                "token": 16384,
                "topk": 16,
                "ep_fused": float("nan"),
            }
        )
        self.assertTrue(
            gfs.row_matches({"gfx": "gfx1250", "token": "16384", "cu_num": ""}, key)
        )
        self.assertTrue(
            gfs.row_matches(
                {"gfx": "gfx1250", "token": "16384.0", "ep_fused": "0"}, key
            )
        )
        self.assertFalse(
            gfs.row_matches({"gfx": "gfx1250", "token": "16384", "ep_fused": "1"}, key)
        )
        self.assertFalse(gfs.row_matches({"gfx": "gfx1250", "token": "8192"}, key))

    def test_routing_names(self):
        self.assertEqual(gfs.check_routing("zipf:0.6"), "zipf:0.6")
        self.assertEqual(gfs.check_routing("balanced,zipf:0.6"), "balanced,zipf:0.6")
        with self.assertRaises(ValueError):
            gfs.check_routing("balanced,skewed")

    def test_bucket_tokens_stay_in_the_bucket(self):
        self.assertEqual(gfs.bucket_tokens(16384, [0.887, 1.0]), [14533, 16384])
        self.assertEqual(gfs.bucket_tokens(1, [0.75, 1.0]), [1])
        with self.assertRaises(ValueError):
            gfs.bucket_tokens(8192, [0.5])


class TestWriteGuard(unittest.TestCase):
    """GroupedFmoeTuner.result_to_csv does not write a shape that another
    table of the runtime merge already holds."""

    # The grouped tuner's key: gemm_moe_tune.py's key + gate_mode, ep_fused.
    KEYS = (
        "gfx",
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
        "gate_mode",
        "ep_fused",
    )

    def _row(self, token, us):
        return {
            "gfx": "gfx1250",
            "cu_num": 256,
            "token": token,
            "model_dim": 3584,
            "inter_dim": 3072,
            "expert": 896,
            "topk": 16,
            "act_type": "ActivationType.Situv2",
            "dtype": "torch.bfloat16",
            "q_dtype_a": gfs.FP4,
            "q_dtype_w": gfs.FP4,
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "gate_mode": "GateMode.INTERLEAVE",
            "ep_fused": "",
            **gfs.resolve_config(K3_ROW),
            "kernelName1": "grouped_gemm1_fp4_gugu",
            "kernelName2": "grouped_gemm2_fp4",
            "us": us,
            "tflops": 0,
            "bw": 0,
        }

    def test_shape_held_by_a_merged_table_is_not_written(self):
        try:
            import pandas as pd

            from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"gemm_moe_tune not importable: {exc}")
        results = ["max_m", *gfs.KNOB_COLUMNS, "kernelName1", "kernelName2"]
        tuner = gemm_moe_tune.GroupedFmoeTuner(
            "groupedFmoeTuner", list(self.KEYS), results + ["us", "tflops", "bw"]
        )
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        out, peer = os.path.join(tmp, "out.csv"), os.path.join(tmp, "peer.csv")
        pd.DataFrame([self._row(16384, 3618.9)]).to_csv(peer, index=False)
        with open(peer) as f:
            peer_text = f.read()
        frame = pd.DataFrame(
            [self._row(16384, 3000.0), self._row(8192, 2000.0)], columns=tuner.columns
        )
        with mock.patch.object(tuner, "_runtime_tables", return_value=[[out, peer]]):
            tuner.result_to_csv(frame, out)
        written = pd.read_csv(out)
        self.assertEqual(list(written["token"]), [8192])
        with open(peer) as f:
            self.assertEqual(f.read(), peer_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
