# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for separate FP32/E8M0 blockscale tuning contracts."""

import importlib.util
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
TUNER_PATH = ROOT / "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py"
FP32_CATALOG = "aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common"
FP32_RUNNER = "aiter.ops.flydsl.gemm_a8w8_blockscale"
KEYS = ["gfx", "cu_num", "M", "N", "K"]
RESULTS = [
    "libtype",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
]
SHAPE = ("gfx950", 256, 256, 256, 512)


class TestBlockscaleScaleDtype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "blockscale_scale_dtype_tuner", TUNER_PATH
        )
        cls.module = importlib.util.module_from_spec(spec)
        original_path = sys.path[:]
        try:
            sys.path.insert(0, str(TUNER_PATH.parent))
            spec.loader.exec_module(cls.module)
        finally:
            sys.path[:] = original_path

    def make_tuner(self):
        with mock.patch.object(torch.cuda, "device_count", return_value=1):
            return self.module.GemmA8W8BlockScaleTuner(
                "scale_contract", KEYS[:], RESULTS[:]
            )

    def prepare(self, cli):
        tuner = self.make_tuner()
        args = tuner.parser.parse_args(cli)
        with mock.patch.object(self.module.GemmCommonTuner, "run", return_value=None):
            tuner.run(args)
        return tuner, args

    def install_module(self, name, module):
        # Restore just this entry; do not unload unrelated lazily imported modules.
        previous = sys.modules.get(name)
        sys.modules[name] = module

        def restore():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

        self.addCleanup(restore)

    def fp32_catalog(self):
        catalog = ModuleType(FP32_CATALOG)
        catalog.kernels_list = {
            i: SimpleNamespace(
                name=f"flydsl_blockscale_8w_256x256x128_F8_F8_B16_ps{i}_sm1_tdma0",
                preshuffle_b=bool(i),
            )
            for i in range(2)
        }
        catalog.kernel_fits_shape = mock.Mock(return_value=True)
        self.install_module(FP32_CATALOG, catalog)
        return catalog

    def test_explicit_fp32_plain(self):
        tuner, args = self.prepare(["--libtype", "flydsl", "--scale-dtype", "fp32"])
        self.assertFalse(tuner._mxscale)
        self.assertEqual(args.tune_file, self.module.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE)
        self.assertEqual(tuner.keys, KEYS)
        self.assertEqual(
            tuner.ARG_DEFAULTS["config_env_name"], "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
        )

    def test_fp32_preshuffle_uses_original_family(self):
        tuner, args = self.prepare(
            ["--libtype", "flydsl", "--scale-dtype", "fp32", "--preshuffle"]
        )
        self.assertFalse(tuner._mxscale)
        self.assertEqual(
            args.tune_file, self.module.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE
        )
        self.assertEqual(tuner.keys, KEYS)
        self.assertEqual(
            tuner.ARG_DEFAULTS["config_env_name"],
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
        )

    def test_legacy_flydsl_remains_e8m0(self):
        for scale in ([], ["--scale-dtype", "e8m0"]):
            with self.subTest(scale=scale):
                tuner, args = self.prepare(
                    ["--libtype", "flydsl", "--preshuffle", *scale]
                )
                self.assertTrue(tuner._mxscale)
                self.assertEqual(args.scale_dtype, "e8m0")
                self.assertEqual(
                    args.tune_file,
                    self.module.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE,
                )
                self.assertEqual(tuner.keys, KEYS + ["w_scale_block"])
                self.assertEqual(
                    tuner.ARG_DEFAULTS["config_env_name"],
                    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
                )

    def test_legacy_other_backends_remain_fp32(self):
        for lib in ("all", "both", "ck", "cktile", "asm", "opus"):
            with self.subTest(lib=lib):
                tuner, args = self.prepare(["--libtype", lib])
                self.assertFalse(tuner._mxscale)
                self.assertEqual(args.scale_dtype, "fp32")
                self.assertEqual(
                    args.tune_file, self.module.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE
                )

    def test_invalid_e8m0_modes_fail_before_base_run(self):
        for cli in (
            ["--libtype", "flydsl"],
            ["--libtype", "flydsl", "--scale-dtype", "e8m0"],
            ["--libtype", "ck", "--scale-dtype", "e8m0", "--preshuffle"],
            ["--libtype", "both", "--scale-dtype", "e8m0", "--preshuffle"],
        ):
            with self.subTest(cli=cli):
                tuner = self.make_tuner()
                args = tuner.parser.parse_args(cli)
                with mock.patch.object(self.module.GemmCommonTuner, "run") as base:
                    with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                        tuner.run(args)
                    base.assert_not_called()

    def test_custom_output_is_preserved(self):
        for scale in ("fp32", "e8m0"):
            with self.subTest(scale=scale):
                _, args = self.prepare(
                    [
                        "--libtype",
                        "flydsl",
                        "--scale-dtype",
                        scale,
                        "--preshuffle",
                        "-o",
                        "custom-output.csv",
                    ]
                )
                self.assertEqual(args.tune_file, "custom-output.csv")

    def test_fp32_tasks_use_matching_layout_and_reference(self):
        catalog = self.fp32_catalog()
        for preshuffle in (False, True):
            with self.subTest(preshuffle=preshuffle):
                tuner = self.make_tuner()
                tasks = tuner.get_gemm_a8w8_blockscale_fp32_flydsl_tune_task(
                    SHAPE, 7, preshuffle, {}
                )
                self.assertEqual(len(tasks), 1)
                task = tasks[0]
                self.assertEqual(
                    task[0],
                    (
                        SHAPE,
                        int(preshuffle),
                        0,
                        catalog.kernels_list[int(preshuffle)].name,
                        "flydsl",
                        preshuffle,
                    ),
                )
                self.assertIs(task[1], self.module.generate_data)
                self.assertIs(task[3], self.module.run_gemm_a8w8_blockscale_fp32_flydsl)
                self.assertIs(task[6], self.module.run_torch)
                self.assertEqual(
                    task[4][0],
                    (
                        ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"]
                        if preshuffle
                        else ["x", "weight", "x_scale", "w_scale", "out"]
                    ),
                )
                self.assertEqual(task[4][-1], preshuffle)
                self.assertEqual(task[-1], ("out",))
        catalog.kernel_fits_shape.return_value = False
        self.assertEqual(
            tuner.get_gemm_a8w8_blockscale_fp32_flydsl_tune_task(SHAPE, 0, False, {}),
            [],
        )
        self.assertEqual(
            tuner.get_gemm_a8w8_blockscale_fp32_flydsl_tune_task(
                ("gfx942", *SHAPE[1:]), 0, False, {}
            ),
            [],
        )

    def test_missing_fp32_catalog_does_not_use_mxscale(self):
        self.install_module(FP32_CATALOG, None)
        tuner = self.make_tuner()
        with mock.patch.object(self.module.aiter.logger, "warning") as warning:
            self.assertEqual(
                tuner.get_gemm_a8w8_blockscale_fp32_flydsl_tune_task(
                    SHAPE, 0, False, {}
                ),
                [],
            )
            warning.assert_called_once()

    def test_fp32_runner_forwards_output_and_layout(self):
        runner = ModuleType(FP32_RUNNER)
        runner.run_gemm_a8w8_blockscale = mock.Mock(return_value=mock.sentinel.out)
        self.install_module(FP32_RUNNER, runner)
        args = [
            mock.sentinel.x,
            mock.sentinel.weight,
            mock.sentinel.xs,
            mock.sentinel.ws,
            mock.sentinel.out,
            "fp32_kernel",
            True,
        ]
        result = self.module.run_gemm_a8w8_blockscale_fp32_flydsl(*args)
        self.assertIs(result, mock.sentinel.out)
        runner.run_gemm_a8w8_blockscale.assert_called_once_with(*args)

    def test_e8m0_task_contract_is_unchanged(self):
        tuner, _ = self.prepare(["--libtype", "flydsl", "--preshuffle"])
        ki = SimpleNamespace(
            a_dtype="fp8", b_dtype="fp8", split_k=4, name="flydsl_mxpsh_test_sk4"
        )
        with mock.patch.object(
            self.module, "kernels_list_flydsl", {0: ki}
        ), mock.patch.object(self.module, "fits_shape_flydsl", return_value=True):
            tasks = tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(
                (*SHAPE, "128x128"), 0, True, {}
            )
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task[0][2:5], (4, ki.name, "flydsl"))
        self.assertIs(task[1], self.module.generate_data_e8m0)
        self.assertIs(task[3], self.module.run_gemm_a8w8_blockscale_flydsl)
        self.assertEqual(
            task[4],
            (["x", "weight_shuffle", "x_scale_shuf", "w_scale_shuf", "out"], ki.name),
        )
        self.assertIs(task[6], self.module.run_torch_e8m0)

    def test_backend_groups_do_not_mix_scale_contracts(self):
        builders = {
            "get_gemm_a8w8_blockscale_tune_task": "ck",
            "get_gemm_a8w8_blockscale_cktile_tune_task": "cktile",
            "get_gemm_a8w8_blockscale_asm_tune_task": "asm",
            "get_gemm_a8w8_blockscale_opus_tune_task": "opus",
            "get_gemm_a8w8_blockscale_flydsl_tune_task": "mxscale",
            "get_gemm_a8w8_blockscale_fp32_flydsl_tune_task": "fp32",
        }
        for lib, scale, expected in (
            ("all", "fp32", ["ck", "cktile", "asm", "opus", "fp32"]),
            ("all", "e8m0", ["mxscale"]),
            ("flydsl", "fp32", ["fp32"]),
            ("flydsl", "e8m0", ["mxscale"]),
            ("both", "fp32", ["ck", "cktile"]),
        ):
            with self.subTest(lib=lib, scale=scale), ExitStack() as stack:
                tuner, args = self.prepare(
                    ["--libtype", lib, "--scale-dtype", scale, "--preshuffle"]
                )
                for name, value in builders.items():
                    stack.enter_context(
                        mock.patch.object(tuner, name, return_value=[value])
                    )
                stack.enter_context(
                    mock.patch.object(tuner, "get_cu_num", return_value=256)
                )
                stack.enter_context(
                    mock.patch.object(tuner, "get_gfx", return_value="gfx950")
                )
                run = stack.enter_context(
                    mock.patch.object(self.module, "mp_tuner", return_value=[])
                )
                tuner.tune(
                    pd.DataFrame([{"M": 256, "N": 256, "K": 512}]), pd.DataFrame(), args
                )
                self.assertEqual(run.call_args.args[0], expected)

    def test_kernel_name_uses_selected_catalog(self):
        catalog = self.fp32_catalog()
        for scale in ("fp32", "e8m0"):
            with self.subTest(scale=scale):
                tuner, _ = self.prepare(
                    ["--libtype", "flydsl", "--scale-dtype", scale, "--preshuffle"]
                )
                mx = SimpleNamespace(name="flydsl_mxpsh_test")
                with mock.patch.object(self.module, "kernels_list_flydsl", {0: mx}):
                    self.assertEqual(
                        tuner.getKernelName(0, "flydsl"),
                        mx.name if scale == "e8m0" else catalog.kernels_list[0].name,
                    )
                    self.assertIsNone(tuner.getKernelName(-1, "flydsl"))

    def test_csv_writer_keeps_families_and_bmm_rows_separate(self):
        for scale in ("fp32", "e8m0"):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as tmp:
                tuner, args = self.prepare(
                    ["--libtype", "flydsl", "--scale-dtype", scale, "--preshuffle"]
                )
                key = SHAPE + (("128x128",) if scale == "e8m0" else ())
                old = dict(zip(tuner.keys, key))
                old.update(
                    libtype="flydsl" if scale == "e8m0" else "ck",
                    kernelId="bmm" if scale == "e8m0" else 0,
                    splitK=1,
                    us=5.0,
                    kernelName="existing",
                    tflops=1.0,
                    bw=1.0,
                    errRatio=0.0,
                )
                file = Path(tmp) / Path(args.tune_file).name
                pd.DataFrame([old]).to_csv(file, index=False)
                # Synthetic timing records exercise CSV selection, not kernel performance.
                info = (
                    key,
                    0,
                    1 if scale == "e8m0" else 0,
                    "selected_kernel",
                    "flydsl",
                    True,
                )
                result = tuner.result_to_df([(info, 1.0, 0.0)])
                tuner.result_to_csv(result, str(file))
                tuner.sortResults(str(file), True, tuner.sort_keys)
                stored = pd.read_csv(file)
                self.assertNotIn("scale_dtype", stored.columns)
                self.assertEqual("w_scale_block" in stored.columns, scale == "e8m0")
                self.assertEqual(len(stored), 2 if scale == "e8m0" else 1)
                if scale == "e8m0":
                    self.assertEqual(
                        stored.loc[stored.kernelId == "bmm", "kernelName"].tolist(),
                        ["existing"],
                    )
                self.assertIn("selected_kernel", stored.kernelName.tolist())

    def test_run_config_uses_selected_scale_data_and_reference(self):
        from aiter import test_common
        from aiter.ops import gemm_op_a8w8

        data = {
            name: object()
            for name in (
                "x",
                "weight",
                "weight_shuffle",
                "x_scale",
                "x_scale_t",
                "w_scale",
                "out",
                "x_scale_shuf",
                "w_scale_shuf",
                "x_deq",
                "w_deq",
            )
        }
        for scale, preshuffle in (("fp32", False), ("fp32", True), ("e8m0", True)):
            with self.subTest(scale=scale, preshuffle=preshuffle), ExitStack() as stack:
                cli = ["--libtype", "flydsl", "--scale-dtype", scale]
                tuner, args = self.prepare(
                    cli + (["--preshuffle"] if preshuffle else [])
                )
                tuner.untunedf = pd.DataFrame([{"M": 256, "N": 256, "K": 512}])
                plain_data = stack.enter_context(
                    mock.patch.object(self.module, "generate_data", return_value=data)
                )
                mx_data = stack.enter_context(
                    mock.patch.object(
                        self.module, "generate_data_e8m0", return_value=data
                    )
                )
                plain_ref = stack.enter_context(
                    mock.patch.object(self.module, "run_torch")
                )
                mx_ref = stack.enter_context(
                    mock.patch.object(self.module, "run_torch_e8m0")
                )
                perf = stack.enter_context(
                    mock.patch.object(
                        test_common, "run_perftest", return_value=(data["out"], 1.0)
                    )
                )
                stack.enter_context(
                    mock.patch.object(test_common, "checkAllclose", return_value=0.0)
                )
                stack.enter_context(mock.patch.object(torch.cuda, "empty_cache"))
                stack.enter_context(
                    mock.patch.object(
                        tuner,
                        "_get_run_config_err_ratio_limit",
                        return_value=(0.05, "0.05"),
                    )
                )
                result = tuner.run_config(args)
                self.assertEqual(result[0]["status"], "ok")
                is_mx = scale == "e8m0"
                (mx_data if is_mx else plain_data).assert_called_once_with(
                    256, 256, 512, 0
                )
                (plain_data if is_mx else mx_data).assert_not_called()
                (mx_ref if is_mx else plain_ref).assert_called_once()
                (plain_ref if is_mx else mx_ref).assert_not_called()
                op = (
                    gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle
                    if preshuffle
                    else gemm_op_a8w8.gemm_a8w8_blockscale
                )
                xs = (
                    "x_scale_shuf"
                    if is_mx
                    else "x_scale_t" if preshuffle else "x_scale"
                )
                perf.assert_called_once_with(
                    op,
                    data["x"],
                    data["weight_shuffle" if preshuffle else "weight"],
                    data[xs],
                    data["w_scale_shuf" if is_mx else "w_scale"],
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )


if __name__ == "__main__":
    unittest.main()
