# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for FhmoeTuner (gemm_moe_tune.py --fhmoe).

Level 1: no GPU. The GPU identity the tuner reads (gfx, cu_num) is mocked, and
the kernel search is never dispatched.
"""

import contextlib
import csv
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

# CI executes this file directly; imports and fixtures must use this checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CONFIGS_DIR = REPO_ROOT / "aiter" / "configs"
UNTUNED_CSV = CONFIGS_DIR / "untuned_fhmoe.csv"
TUNED_CSV = CONFIGS_DIR / "tuned_fhmoe.csv"

# Mirrors fhmoe_key / fhmoe_result_list in the __main__ block of gemm_moe_tune.py;
# test_columns_match_shipped_tuned_header fails if the column set drifts from the CSV.
KEYS = [
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
    "shared_expert_id",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
]
RESULT_LIST = ["block_m", "ksplit", "kernelName1", "kernelName2"]
LDS_BYTES = 160 * 1024


def _tune_module():
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    return gemm_moe_tune


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _tuner():
    return _tune_module().FhmoeTuner("testFhmoe", KEYS, RESULT_LIST, "test parser")


def _catalogue():
    return pd.read_csv(UNTUNED_CSV)


def _shape(token=1):
    row = _catalogue().iloc[0].copy()
    row["token"] = token
    return _tune_module().FhmoeTuner._parse_fhmoe_row(row)


def _tune_args(**overrides):
    args = {"warmup": 1, "iters": 1, "profile_file": "", "errRatio": 0.05}
    args.update(overrides)
    return SimpleNamespace(**args)


class TestFhmoeCatalogueSchema(unittest.TestCase):
    def test_columns_match_shipped_tuned_header(self):
        with open(TUNED_CSV, newline="") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(sorted(KEYS + RESULT_LIST), sorted(header))

    def test_parse_row_covers_every_key(self):
        self.assertTrue(set(KEYS) <= set(_shape()))


class TestFhmoeLookupToken(unittest.TestCase):
    def test_power_of_two_tokens_are_identity(self):
        lookup = _tune_module().FhmoeTuner._lookup_token
        for token in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
            with self.subTest(token=token):
                self.assertEqual(lookup(token), token)

    def test_other_tokens_round_up_to_their_bucket(self):
        lookup = _tune_module().FhmoeTuner._lookup_token
        self.assertEqual(lookup(3), 4)
        self.assertEqual(lookup(3072), 4096)


class TestFhmoeTuneTasks(unittest.TestCase):
    def test_non_bucket_token_is_rejected_before_any_task(self):
        tuner = _tuner()
        df = _catalogue().iloc[:1].copy()
        df["token"] = 3
        with (
            mock.patch.object(
                _tune_module().FhmoeTuner,
                "_kernel_pairs",
                side_effect=AssertionError("pairs built for a non-bucket token"),
            ),
            self.assertRaisesRegex(ValueError, "not a serving bucket"),
        ):
            tuner._fhmoe_tune_tasks(df, _tune_args())

    def test_catalogue_tokens_become_one_task_per_pair(self):
        tuner = _tuner()
        pairs = [(16, "k1a", "k2a"), (32, "k1b", "k2b")]
        df = _catalogue()
        with mock.patch.object(
            _tune_module().FhmoeTuner, "_kernel_pairs", return_value=pairs
        ), _quiet():
            tasks, in_datas = tuner._fhmoe_tune_tasks(df, _tune_args())
        self.assertEqual(len(tasks), len(df) * len(pairs))
        self.assertEqual(in_datas, [(len(pairs), ())] * len(df))
        self.assertEqual({task[0][0][2] for task in tasks}, set(df["token"]))


class TestFhmoeOwnedRows(unittest.TestCase):
    def test_only_mxfp4_weight_rows_are_kept(self):
        df = pd.DataFrame(
            {
                "token": [1, 2, 4],
                "q_dtype_w": [
                    "torch.float4_e2m1fn_x2",
                    "torch.float8_e4m3fn",
                    "torch.float4_e2m1fn_x2",
                ],
            }
        )
        with _quiet():
            owned = _tune_module().FhmoeTuner._owned_rows(df)
        self.assertEqual(owned["token"].tolist(), [1, 4])
        self.assertEqual(owned.index.tolist(), [0, 1])


class TestFhmoeKernelPairs(unittest.TestCase):
    def _pairs(self, shape):
        with mock.patch.dict(os.environ):
            os.environ.pop("TUNE_MOE_KERNEL_REGEX", None)
            return _tune_module().FhmoeTuner._kernel_pairs(shape)

    def test_pairs_respect_the_stage_cuts(self):
        mod = _tune_module()
        shape = _shape()
        pairs = self._pairs(shape)
        self.assertTrue(pairs)
        self.assertEqual(len(pairs), len(set(pairs)), "duplicate pairs")
        s1_all = mod.get_flydsl_stage1_kernels("fp8", "fp4", "bf16")
        s2_all = mod.get_flydsl_stage2_kernels("fp8", "fp4", "bf16")
        for block_m, kn1, kn2 in pairs:
            self.assertIn(block_m, mod.FhmoeTuner.BLOCK_MS)
            base1 = kn1 if kn1 in s1_all else kn1[: -len("_fp8")]
            self.assertIn(base1, s1_all, kn1)
            self.assertEqual(s1_all[base1]["tile_m"], block_m)
            self.assertEqual(shape["model_dim"] % s1_all[base1].get("k_batch", 1), 0)
            self.assertIn(kn2, s2_all)
            p2 = s2_all[kn2]
            self.assertEqual(p2["tile_m"], block_m)
            self.assertEqual(shape["inter_dim"] % p2["tile_k"], 0, kn2)
            lds = block_m * (p2["tile_k"] * 2 + p2["tile_n"] * 4)
            self.assertLessEqual(lds, LDS_BYTES, kn2)

    def test_kernel_regex_env_filters_the_pairs(self):
        shape = _shape()
        pairs = self._pairs(shape)
        _, kn1, kn2 = pairs[0]
        pattern = f"{re.escape(kn1)} {re.escape(kn2)}$"
        with mock.patch.dict(os.environ, {"TUNE_MOE_KERNEL_REGEX": pattern}):
            filtered = _tune_module().FhmoeTuner._kernel_pairs(shape)
        self.assertIn(pairs[0], filtered)
        self.assertLess(len(filtered), len(pairs))
        for _, f1, f2 in filtered:
            self.assertRegex(f"{f1} {f2}", pattern)


class TestFhmoePadDummyExpert(unittest.TestCase):
    def test_2d_tensor_gets_one_zero_expert_block(self):
        pad = _tune_module().FhmoeTuner._pad_dummy_expert
        out = pad(torch.ones((6, 4), dtype=torch.uint8, device="cpu"), n_experts=3)
        self.assertEqual(tuple(out.shape), (8, 4))
        self.assertTrue(bool(out[:6].all()))
        self.assertFalse(bool(out[6:].any()))

    def test_3d_tensor_gets_one_zero_expert(self):
        pad = _tune_module().FhmoeTuner._pad_dummy_expert
        out = pad(torch.ones((3, 2, 2), device="cpu"))
        self.assertEqual(tuple(out.shape), (4, 2, 2))
        self.assertTrue(bool(out[:3].all()))
        self.assertFalse(bool(out[3].any()))


class TestFhmoePostProcess(unittest.TestCase):
    def setUp(self):
        self.tuner = _tuner()
        self.args = _tune_args()

    def _info(self, token):
        shape = _shape(token)
        return tuple(shape[k] for k in KEYS)

    def _result(self, token, names, block_m, us, err=0.01):
        return ((self._info(token), names[0], names[1], block_m), us, err)

    def test_fastest_valid_candidate_wins_per_shape(self):
        t = self.tuner
        results = [
            self._result(1, ("k1a", "k2a"), 16, 12.0),
            self._result(1, ("k1b", "k2b"), 32, 9.0),
            self._result(1, ("k1c", "k2c"), 64, 5.0, err=0.5),
            self._result(1, ("k1d", "k2d"), 128, t.INVALID_TIME, err=1.0),
            self._result(1, ("k1e", "k2e"), 16, t.INF_TIME, err=1.0),
            self._result(2, ("k1f", "k2f"), 16, 7.0),
        ]
        with _quiet():
            df = t.post_process(results, self.args)
        self.assertEqual(list(df.columns), t.columns + ["us"])
        by_token = {int(row["token"]): row for _, row in df.iterrows()}
        self.assertEqual(set(by_token), {1, 2})
        self.assertEqual(by_token[1]["kernelName1"], "k1b")
        self.assertEqual(by_token[1]["kernelName2"], "k2b")
        self.assertEqual(by_token[1]["block_m"], 32)
        self.assertEqual(by_token[1]["us"], 9.0)
        self.assertEqual(by_token[2]["kernelName1"], "k1f")
        self.assertEqual(by_token[1]["act_type"], "ActivationType.Silu")

    def test_shape_with_only_invalid_candidates_yields_no_row(self):
        t = self.tuner
        results = [
            self._result(1, ("k1a", "k2a"), 16, t.INVALID_TIME, err=1.0),
            self._result(1, ("k1b", "k2b"), 32, t.INF_TIME, err=1.0),
            self._result(1, ("k1c", "k2c"), 64, 5.0, err=0.5),
        ]
        with _quiet():
            df = t.post_process(results, self.args)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), t.columns + ["us"])


class TestFhmoeResultToCsv(unittest.TestCase):
    def _best(self, tuner, kn1, kn2, us):
        info = tuple(_shape(1)[k] for k in KEYS)
        with _quiet():
            return tuner.post_process([((info, kn1, kn2, 32), us, 0.01)], _tune_args())

    def test_rewriting_the_same_shape_keeps_one_row(self):
        tuner = _tuner()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tuned_fhmoe.csv")
            with _quiet():
                tuner.result_to_csv(self._best(tuner, "k1a", "k2a", 9.0), path)
                tuner.result_to_csv(self._best(tuner, "k1a", "k2a", 9.0), path)
                tuner.result_to_csv(self._best(tuner, "k1b", "k2b", 8.0), path)
            rows = pd.read_csv(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.loc[0, "kernelName1"], "k1b")
        self.assertEqual(rows.loc[0, "kernelName2"], "k2b")
        self.assertEqual(list(rows.columns), KEYS + RESULT_LIST)


class TestFhmoeRunFhmoe(unittest.TestCase):
    def test_temp_candidate_csv_is_removed_when_the_write_fails(self):
        mod = _tune_module()
        real_mkstemp = tempfile.mkstemp
        created = []

        def tracking_mkstemp(*args, **kwargs):
            fd, path = real_mkstemp(*args, **kwargs)
            created.append(path)
            return fd, path

        data = {"w1": SimpleNamespace(), "w2": SimpleNamespace()}
        with (
            mock.patch.object(tempfile, "mkstemp", side_effect=tracking_mkstemp),
            mock.patch.object(
                mod.FhmoeTuner,
                "_write_candidate_csv",
                side_effect=RuntimeError("disk full"),
            ),
            self.assertRaisesRegex(RuntimeError, "disk full"),
        ):
            mod.FhmoeTuner._run_fhmoe(_shape(), data, "k1", "k2", 32)
        self.assertEqual(len(created), 1)
        self.assertFalse(os.path.exists(created[0]), "temp candidate CSV leaked")


class TestFhmoeUntunedTargetGuard(unittest.TestCase):
    """The catalogue may pin gfx/cu_num; the live GPU decides what is tuned."""

    def setUp(self):
        mod = _tune_module()
        self.mod = mod
        patches = [
            mock.patch.object(mod.FhmoeTuner, "get_gfx", return_value="gfx950"),
            mock.patch.object(mod.FhmoeTuner, "get_cu_num", return_value=256),
            mock.patch.object(mod, "get_gfx_runtime", return_value="gfx950"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.tuner = _tuner()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, text, name="untuned.csv"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as handle:
            handle.write(text)
        return path

    def _load(self, text, name="untuned.csv"):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            df = self.tuner.get_untuned_gemm_list(self._write(text, name))
        return df, buffer.getvalue()

    def test_shipped_catalogue_is_kept_on_its_target(self):
        with _quiet():
            df = self.tuner.get_untuned_gemm_list(str(UNTUNED_CSV))
        self.assertEqual(len(df), len(_catalogue()))

    def test_shipped_catalogue_is_skipped_on_another_cu_num(self):
        with mock.patch.object(self.mod.FhmoeTuner, "get_cu_num", return_value=128):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                df = self.tuner.get_untuned_gemm_list(str(UNTUNED_CSV))
        self.assertTrue(df.empty)
        self.assertIn("[fhmoe] skipping 12 rows", buffer.getvalue())
        self.assertIn("gfx950 with 128 CUs", buffer.getvalue())

    def test_rows_pinned_to_another_target_are_skipped_with_a_notice(self):
        df, out = self._load(
            "gfx,cu_num,token\ngfx942,304,1\ngfx950,256,2\ngfx950,128,4\n"
        )
        self.assertEqual(df["token"].tolist(), [2])
        self.assertIn("[fhmoe] skipping 2 rows", out)

    def test_blank_targets_are_stamped_with_the_live_gpu(self):
        df, out = self._load("gfx,cu_num,token\n,,1\ngfx950,,2\n,256,4\n")
        self.assertEqual(df["token"].tolist(), [1, 2, 4])
        self.assertEqual(set(df["gfx"]), {"gfx950"})
        self.assertEqual(set(df["cu_num"]), {256})
        self.assertEqual(out, "")

    def test_unparsable_cu_num_is_skipped_not_stamped(self):
        df, _ = self._load("gfx,cu_num,token\ngfx950,abc,1\ngfx950,256,2\n")
        self.assertEqual(df["token"].tolist(), [2])

    def test_catalogue_without_target_columns_is_untouched(self):
        df, out = self._load("token\n1\n2\n")
        self.assertEqual(list(df.columns), ["token"])
        self.assertEqual(len(df), 2)
        self.assertEqual(out, "")

    def test_tuned_file_read_back_keeps_every_target(self):
        df, out = self._load(
            "gfx,cu_num,token,kernelName1\ngfx942,304,1,k\ngfx950,256,2,k\n"
        )
        self.assertEqual(len(df), 2)
        self.assertEqual(out, "")

    def test_pre_process_tunes_only_this_gpus_rows(self):
        path = self._write(
            "gfx,cu_num,token\ngfx942,304,1\ngfx950,256,2\n,,4\n",
            name="untuned_fhmoe.csv",
        )
        args = SimpleNamespace(
            all=False,
            last=False,
            untune_file=path,
            tune_file=os.path.join(self.tmp.name, "tuned_fhmoe.csv"),
        )
        with _quiet():
            self.tuner.pre_process(args)
        self.assertEqual(self.tuner.untunedf["token"].tolist(), [2, 4])
        self.assertEqual(set(self.tuner.untunedf["gfx"]), {"gfx950"})
        self.assertEqual(set(self.tuner.untunedf["cu_num"]), {256})


if __name__ == "__main__":
    unittest.main(verbosity=2)
