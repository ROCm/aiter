# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Focused saved-kid replay and M-padding checks for MXFP8 BMM tuning."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from aiter import dtypes
from aiter.test_common import checkAllclose
from csrc.opus_gemm import opus_bmm_mxscale_tune as tune

_MXFP8_ERR_RATIO = 0.003


class TestOpusBmmMxscaleTune(unittest.TestCase):
    def setUp(self):
        self.tuner = tune.OpusBmmMxscaleTuner()

    @staticmethod
    def args():
        return SimpleNamespace(warmup=0, iters=1, errRatio=0.02, verbose=False)

    @staticmethod
    def row(**changes):
        row = {
            "gfx": "gfx950",
            "b": 2,
            "m": 64,
            "n": 1024,
            "k": 4096,
            "libtype": "opus",
            "kernelId": 8311,
            "splitK": 1,
            "errRatio": 0.0,
        }
        row.update(changes)
        return pd.DataFrame([row])

    def test_tune_result_records_opus(self):
        frame = self.tuner.result_to_df(
            [((("gfx950", 2, 64, 1024, 4096), 8311, 1, ""), 8.5, 0.0)]
        )
        self.assertEqual(frame.loc[0, "libtype"], "opus")
        self.assertEqual(frame.loc[0, "kernelId"], 8311)
        self.assertEqual(frame.loc[0, "splitK"], 1)

    def test_run_config_calls_saved_exact_kid(self):
        activation = torch.empty((2, 1, 1))
        weight = torch.empty((2, 1, 1))
        output = torch.full((1, 2, 1), float("nan"))
        x_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
        w_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
        workspace = None
        reference = torch.arange(2, dtype=torch.float32).reshape(1, 2, 1)
        data = (
            activation,
            weight,
            output,
            x_scale,
            w_scale,
            workspace,
            reference,
        )
        self.tuner.untunedf = self.row()

        def launch(XQ, WQ, Y, **kwargs):
            self.assertIs(XQ, activation)
            self.assertIs(WQ, weight)
            self.assertEqual(Y.data_ptr(), output.data_ptr())
            self.assertEqual(kwargs["kid"], 8311)
            self.assertEqual(kwargs["layout"], "mxscale_bmm")
            self.assertIs(kwargs["x_scale"], x_scale)
            self.assertIs(kwargs["w_scale"], w_scale)
            self.assertEqual(kwargs["split_k"], 1)
            self.assertIs(kwargs["workspace"], workspace)
            Y.copy_(reference.transpose(0, 1))
            return Y

        def measured(func, *args, **kwargs):
            self.assertEqual(kwargs, {"num_warmup": 0, "num_iters": 1})
            return func(*args), 3.0

        with patch.object(
            tune, "gen_bmm_mxscale_data", return_value=data
        ) as generate, patch.object(
            tune, "opus_bmm", side_effect=launch
        ) as exact, patch(
            "aiter.test_common.run_perftest", side_effect=measured
        ):
            results = self.tuner.run_config(self.args())

        generate.assert_called_once_with(2, 64, 1024, 4096, 1, dtypes.bf16, 8311, 1)
        exact.assert_called_once()
        self.assertEqual(results[0]["status"], "ok")
        self.assertIn("kid=8311,splitK=1", results[0]["shape"])

    def test_run_config_normalizes_checked_in_local_kid(self):
        self.tuner.untunedf = self.row(kernelId=311)
        data = (
            torch.empty((2, 1, 1)),
            torch.empty((2, 1, 1)),
            torch.empty((1, 2, 1)),
            torch.empty((2, 1, 1), dtype=torch.uint8),
            torch.empty((2, 1, 1), dtype=torch.uint8),
            None,
            torch.empty((1, 2, 1)),
        )

        with patch.object(
            tune, "gen_bmm_mxscale_data", return_value=data
        ) as generate, patch.object(
            tune, "run_bmm_mxscale_bench", return_value=data[2]
        ), patch(
            "aiter.test_common.run_perftest", return_value=(data[2], 3.0)
        ), patch(
            "aiter.test_common.checkAllclose", return_value=0.0
        ):
            results = self.tuner.run_config(self.args())

        generate.assert_called_once_with(2, 64, 1024, 4096, 1, dtypes.bf16, 8311, 1)
        self.assertIn("kid=8311,splitK=1", results[0]["shape"])

    def test_shape_only_run_config_keeps_production_policy_path(self):
        from aiter.ops import batched_gemm_op_a8w8 as batched

        self.tuner.untunedf = self.row()[self.tuner.keys]
        output = torch.zeros((64, 2, 1))
        data = (
            torch.empty((2, 64, 1)),
            torch.empty((2, 1, 1)),
            torch.empty_like(output),
            torch.empty((2, 64, 1), dtype=torch.uint8),
            torch.empty((2, 1, 1), dtype=torch.uint8),
            None,
            output,
        )

        def measured(func, *args, **kwargs):
            self.assertEqual(kwargs["dtype"], dtypes.bf16)
            kwargs.pop("num_warmup")
            kwargs.pop("num_iters")
            return func(*args, **kwargs), 3.0

        with patch.object(
            tune, "gen_bmm_mxscale_data", return_value=data
        ), patch.object(
            batched, "batched_gemm_a8w8_mxscale", return_value=output
        ) as production, patch.object(
            tune, "opus_bmm"
        ) as exact, patch(
            "aiter.test_common.run_perftest", side_effect=measured
        ):
            results = self.tuner.run_config(self.args())

        production.assert_called_once()
        exact.assert_not_called()
        self.assertEqual(results[0]["status"], "ok")

    def test_run_config_rejects_non_opus_and_incompatible_saved_rows(self):
        for changes in (
            {"libtype": "ck"},
            {"kernelId": 99999},
            {"splitK": 3},
        ):
            self.tuner.untunedf = self.row(**changes)
            with self.subTest(changes=changes), patch.object(
                tune, "gen_bmm_mxscale_data"
            ) as generate:
                with self.assertRaises(ValueError):
                    self.tuner.run_config(self.args())
                generate.assert_not_called()


class TestOpusBmmMxscaleTuneGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("MXFP8 BMM GPU validation requires gfx950")
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise unittest.SkipTest(f"MXFP8 BMM is unavailable on {arch}")

    def test_saved_m64_kid_m53_padding_and_production_hit(self):
        from aiter.ops import batched_gemm_op_a8w8 as batched
        from aiter.ops.opus import policy

        b, logical_m, padded_m, n, k = 2, 53, 64, 1024, 4096
        shipped = pd.read_csv(tune.SHIPPED_CSV)
        row = shipped[
            (shipped["gfx"] == "gfx950")
            & (shipped["b"] == b)
            & (shipped["m"] == padded_m)
            & (shipped["n"] == n)
            & (shipped["k"] == k)
        ].iloc[0]
        self.assertEqual(row["libtype"], "opus")
        saved_kid = int(row["kernelId"])
        kernel_id = (
            saved_kid
            if saved_kid in tune._CODEGEN_BMM
            else tune.bmm_mxscale_global_kid(saved_kid)
        )
        split_k = int(row["splitK"])
        self.assertIn(kernel_id, tune._CODEGEN_BMM)

        O_mx, W_mx, _Y, xs_mx, ws_mx, _workspace, ref = tune.gen_bmm_mxscale_data(
            b,
            logical_m,
            n,
            k,
            7,
            dtypes.bf16,
            kernel_id,
            split_k,
        )
        O_pad = torch.zeros((b, padded_m, k), dtype=O_mx.dtype, device=O_mx.device)
        O_pad[:, :logical_m].copy_(O_mx)
        xs_pad = torch.full(
            (b, padded_m, k // tune.GROUP),
            127,
            dtype=xs_mx.dtype,
            device=xs_mx.device,
        )
        xs_pad[:, :logical_m].copy_(xs_mx)
        out_pad = torch.full(
            (padded_m, b, n),
            float("nan"),
            dtype=dtypes.bf16,
            device=O_mx.device,
        )
        workspace_numel = tune._workspace_numel(kernel_id, split_k, b, padded_m, n)
        workspace = (
            torch.empty(workspace_numel, dtype=torch.float32, device=O_mx.device)
            if workspace_numel
            else None
        )
        tune.run_bmm_mxscale_bench(
            O_pad,
            W_mx,
            out_pad,
            xs_pad,
            ws_mx,
            workspace,
            kernel_id,
            split_k,
        )
        torch.cuda.synchronize()
        padded_err = checkAllclose(
            out_pad[:logical_m].float(),
            ref.float(),
            rtol=1e-2,
            atol=1e-2,
            printLog=False,
        )
        self.assertLessEqual(padded_err, _MXFP8_ERR_RATIO)
        torch.testing.assert_close(
            out_pad[logical_m:],
            torch.zeros_like(out_pad[logical_m:]),
            rtol=0,
            atol=0,
        )

        config_row = row.to_dict()
        config_row["kernelId"] = kernel_id
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "mxscale_bmm.csv")
            pd.DataFrame([config_row]).to_csv(config_path, index=False)
            config = SimpleNamespace(
                AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE=config_path
            )
            raw_launch, exact_launch = batched._get_mxscale_bmm_launchers()
            calls = []

            def record_raw(*args, **kwargs):
                calls.append(("raw", int(args[-2]), int(args[-1])))
                return raw_launch(*args, **kwargs)

            def record_exact(*args, **kwargs):
                calls.append(("exact", int(kwargs["kid"]), int(kwargs["split_k"])))
                return exact_launch(*args, **kwargs)

            policy._load_mxscale_bmm_tuned.cache_clear()
            policy.lookup_mxscale_bmm_config.cache_clear()
            batched._get_mxscale_bmm_launch_plan.cache_clear()
            try:
                with patch.object(policy, "AITER_CONFIGS", config), patch.object(
                    batched,
                    "_get_mxscale_bmm_launchers",
                    return_value=(record_raw, record_exact),
                ):
                    self.assertEqual(
                        batched._get_mxscale_bmm_launch_plan(b, logical_m, n, k),
                        (kernel_id, split_k),
                    )
                    production_out = batched.batched_gemm_a8w8_mxscale(
                        O_mx.transpose(0, 1),
                        W_mx,
                        xs_mx.transpose(0, 1),
                        ws_mx,
                        dtype=dtypes.bf16,
                    )
                    torch.cuda.synchronize()
            finally:
                batched._get_mxscale_bmm_launch_plan.cache_clear()
                policy.lookup_mxscale_bmm_config.cache_clear()
                policy._load_mxscale_bmm_tuned.cache_clear()

        expected_route = "raw" if split_k <= 1 else "exact"
        self.assertEqual(calls, [(expected_route, kernel_id, split_k)])
        production_err = checkAllclose(
            production_out.float(),
            ref.float(),
            rtol=1e-2,
            atol=1e-2,
            printLog=False,
        )
        self.assertLessEqual(production_err, _MXFP8_ERR_RATIO)


if __name__ == "__main__":
    unittest.main()
