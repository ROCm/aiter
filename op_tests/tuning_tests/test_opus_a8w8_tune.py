# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from aiter.ops.opus import opus_gemm
from csrc.opus_gemm import opus_gemm_a8w8_tune as tune
from csrc.opus_gemm.opus_gemm_common import (
    get_kernel_instance,
    kernels_list,
)


class TestOpusA8W8Tuner(unittest.TestCase):
    def setUp(self):
        with patch("torch.cuda.device_count", return_value=0):
            self.tuner = tune.OpusA8W8Tuner()
        for method, value in (("get_gfx", "gfx950"), ("get_cu_num", 256)):
            patcher = patch.object(self.tuner, method, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.input_file = str(Path(self.tmp.name) / "shapes.csv")
        self.output_file = str(Path(self.tmp.name) / "tuned.csv")

    def args(self, *extra):
        return self.tuner.parser.parse_args(
            [
                "--input_file",
                self.input_file,
                "--tuned_file",
                self.output_file,
                "--libtype",
                "opus",
                "--mp",
                "1",
                *extra,
            ]
        )

    def rows(self, **overrides):
        return self.tuner._normalize_rows(
            pd.DataFrame([{"M": 64, "N": 256, "K": 256, **overrides}])
        )

    def test_existing_cli_aliases(self):
        long_args = self.args()
        short_args = self.tuner.parser.parse_args(
            ["-i", self.input_file, "-o", self.output_file, "--mp", "1"]
        )
        self.assertEqual(vars(long_args), vars(short_args))
        self.assertEqual(long_args.errRatio, 0)
        self.assertNotIn("family", vars(long_args))

    def test_import_does_not_parse_or_tune(self):
        with patch("argparse.ArgumentParser.parse_args") as parse, patch(
            "aiter.utility.mp_tuner.mp_tuner"
        ) as sweep:
            importlib.reload(tune)
        parse.assert_not_called()
        sweep.assert_not_called()

    def test_scale_modes_survive_csv_write_skip_and_retune(self):
        rows = pd.concat([self.rows(scaleAB="False"), self.rows(scaleAB="True")])
        pd.concat([rows, rows]).to_csv(self.input_file, index=False)
        args = self.args()
        self.tuner.pre_process(args)
        self.assertEqual(len(self.tuner.untunedf), 2)
        for scale_ab, kid in ((False, 2), (True, 1)):
            row = rows[rows.scaleAB == scale_ab].iloc[0]
            keys = tuple(row[key] for key in self.tuner.keys)
            result = self.tuner.result_to_df([((keys, kid, 0, ""), 10.0, 0.0)])
            self.tuner.result_to_csv(result, self.output_file, concat=True)
            self.tuner.pre_process(args)
            self.assertEqual(len(self.tuner.untunedf), 0 if scale_ab else 1)
            if not scale_ab:
                self.assertTrue(self.tuner.untunedf.iloc[0].scaleAB)
        saved = self.tuner.get_tuned_gemm_list(self.output_file)
        self.assertEqual(len(saved), 2)
        self.assertEqual(set(saved.libtype), {"opus"})
        self.assertFalse(saved.duplicated(self.tuner.keys).any())
        args.all = True
        self.tuner.pre_process(args)
        self.assertEqual(len(self.tuner.untunedf), 2)

    def test_csv_defaults_and_unsupported_options(self):
        row = self.rows().iloc[0]
        self.assertEqual(row["dtype"], str(torch.float8_e4m3fn))
        self.assertEqual(row.outdtype, str(torch.float32))
        self.assertFalse(row.scaleAB)
        for changes in (
            {"bias": True},
            {"bpreshuffle": True},
            {"dtype": "bf16"},
            {"outdtype": "bf16"},
            {"libtype": "ck"},
            {"M": 1.5},
            {"K": 0},
            {"kernelId": 2.5},
            {"scaleAB": "invalid"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.rows(**changes)
        for flag in ("--splitK", "--compare"):
            with (
                self.subTest(flag=flag),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.tuner.pre_process(self.args(flag))

    def test_candidate_registry_completeness(self):
        for scale_ab, family in ((False, "a8w8"), (True, "a8w8_blockscale")):
            registered = {
                kid
                for kid in kernels_list
                if get_kernel_instance("gfx950", family, kid, "fp32") is not None
            }
            self.assertEqual(
                set(tune.candidate_kids_for_shape("gfx950", 256, 4096, 4096, scale_ab)),
                registered,
            )
        with patch.dict(kernels_list, {99999: kernels_list[2]}):
            self.assertIn(
                99999, tune.candidate_kids_for_shape("gfx950", 64, 256, 256, False)
            )

    def test_candidate_shape_and_arch_guards(self):
        for scale_ab, kid in ((False, 2), (True, 1)):
            for m, n, k in ((53, 384, 256), (64, 4096, 4096)):
                self.assertIn(
                    kid, tune.candidate_kids_for_shape("gfx950", m, n, k, scale_ab)
                )
            for shape in (
                (0, 256, 256),
                (64, 256, 128),
                (64, 256, 384),
                (64, 256, 255),
            ):
                self.assertEqual(
                    tune.candidate_kids_for_shape("gfx950", *shape, scale_ab), []
                )
            for arch in ("gfx942", "gfx1250"):
                self.assertEqual(
                    tune.candidate_kids_for_shape(arch, 64, 256, 256, scale_ab), []
                )
            self.assertEqual(
                tune.candidate_kids_for_shape("gfx950", 64, 256, 256, scale_ab, "bf16"),
                [],
            )
        self.assertEqual(
            tune.candidate_kids_for_shape("gfx950", 53, 257, 130, False), [2]
        )
        self.assertEqual(
            tune.candidate_kids_for_shape("gfx950", 53, 257, 130, True), []
        )

    def test_reference_with_nonuniform_scales(self):
        x = torch.tensor([[1, -2, 3, -4], [-1, 2, -3, 4]], dtype=torch.float8_e4m3fn)
        w = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, -1, 1, -1]], dtype=x.dtype)
        x_scale = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32)
        w_scale = torch.tensor([[7, 11]], dtype=torch.float32)
        expected = torch.tensor(
            [[113, -160, 273], [-193, 276, -469]], dtype=torch.float32
        )
        torch.testing.assert_close(tune.run_torch(x, w, x_scale, w_scale), expected)
        torch.testing.assert_close(
            tune.run_torch(x, w, None, None), x.float() @ w.float().T
        )

    def test_rare_mismatch_cannot_round_to_zero(self):
        ref = torch.zeros(100000)
        out = ref.clone()
        out[-1] = 1
        error = tune.compare_outputs(ref, out, printLog=False)
        self.assertGreater(round(error, 4), 0)
        out[-1] = float("nan")
        self.assertGreater(round(tune.compare_outputs(ref, out, printLog=False), 4), 0)

    def test_run_config_calls_the_saved_id(self):
        data = tune.generate_data(64, 256, 256, 2, device="cpu")
        reference = tune.run_torch(data["x"], data["w"], None, None)
        saved_kid = 99999
        self.tuner.untunedf = self.rows(kernelId=saved_kid, splitK=0, libtype="opus")

        def launch(x, w, out, **kwargs):
            self.assertEqual(kwargs["kid"], saved_kid)
            self.assertIs(out, data["out"])
            self.assertTrue(torch.isnan(out).all())
            out.copy_(reference)

        def measured(func, *args, **kwargs):
            return func(*args), 3.0

        with patch.dict(kernels_list, {saved_kid: kernels_list[2]}), patch.object(
            tune, "generate_data", return_value=data
        ), patch.object(tune, "opus_gemm", side_effect=launch) as kernel, patch(
            "aiter.test_common.run_perftest", side_effect=measured
        ):
            results = self.tuner.run_config(self.args())
        kernel.assert_called_once()
        self.assertEqual(results[0]["status"], "ok")
        with contextlib.redirect_stdout(io.StringIO()) as report:
            self.tuner._print_benchmark_results("Saved kid", results)
        self.assertIn("3.00", report.getvalue())
        self.assertNotIn("N/A", report.getvalue())

    def test_run_config_rejects_incompatible_saved_ids(self):
        for changes in (
            {"scaleAB": False, "kernelId": 1},
            {"scaleAB": True, "kernelId": 2},
            {"kernelId": 2, "splitK": 1},
            {"kernelId": 99999},
        ):
            self.tuner.untunedf = self.rows(
                **{"splitK": 0, "libtype": "opus", **changes}
            )
            with self.subTest(changes=changes), patch.object(
                tune, "generate_data"
            ) as data:
                with self.assertRaises(ValueError):
                    self.tuner.run_config(self.args())
                data.assert_not_called()

    def test_run_config_without_input_file_uses_tuned_file(self):
        self.rows(kernelId=2, splitK=0, libtype="opus").to_csv(
            self.output_file, index=False
        )
        args = self.tuner.parser.parse_args(
            ["--tuned_file", self.output_file, "--run_config"]
        )
        with patch.object(self.tuner, "run_config", return_value=[]) as replay:
            self.tuner.run(args)
        replay.assert_called_once()
        self.assertEqual(args.run_config, self.output_file)
        self.assertEqual(self.tuner.untunedf.iloc[0].kernelId, 2)


class TestOpusA8W8TuneGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("OPUS A8W8 GPU validation requires gfx950")
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise unittest.SkipTest(f"OPUS plain A8W8 is unavailable on {arch}")

    def test_exact_kids_and_m_padding(self):
        for scale_ab in (False, True):
            kids = tune.candidate_kids_for_shape("gfx950", 64, 256, 256, scale_ab)
            self.assertTrue(kids)
            for kid in kids:
                with self.subTest(scaleAB=scale_ab, kid=kid):
                    data = tune.generate_data(53, 256, 256, kid, device="cuda")
                    ref = tune.run_torch(
                        data["x"], data["w"], data["x_scale"], data["w_scale"]
                    )
                    data["out"].fill_(float("nan"))
                    tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), kid)
                    torch.testing.assert_close(data["out"], ref, rtol=1e-2, atol=1e-2)
                    x_pad = torch.zeros((64, 256), device="cuda", dtype=data["x"].dtype)
                    x_pad[:53].copy_(data["x"])
                    scale_pad = None
                    if scale_ab:
                        scale_pad = torch.ones(
                            (64, data["x_scale"].shape[1]), device="cuda"
                        )
                        scale_pad[:53].copy_(data["x_scale"])
                    out_pad = torch.full((64, 256), float("nan"), device="cuda")
                    opus_gemm(
                        x_pad,
                        data["w"],
                        out_pad,
                        kid=kid,
                        x_scale=scale_pad,
                        w_scale=data["w_scale"],
                    )
                    torch.testing.assert_close(out_pad[:53], ref, rtol=1e-2, atol=1e-2)
                    torch.testing.assert_close(
                        out_pad[53:], torch.zeros_like(out_pad[53:]), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        out_pad[:53], data["out"], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
