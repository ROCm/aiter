# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only route tests for the A8W8 blockscale tuner.

The tuner module is loaded with small dependency stubs so this test never
queries ROCm, enumerates a GPU, builds a JIT module, or launches a kernel.
"""

import argparse
import ast
import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TUNER_PATH = (
    _REPO_ROOT
    / "csrc"
    / "ck_gemm_a8w8_blockscale"
    / "gemm_a8w8_blockscale_tune.py"
)
_PRODUCTION_OP_PATH = _REPO_ROOT / "aiter" / "ops" / "gemm_op_a8w8.py"
_PLAIN_TUNE = "/mock/a8w8_blockscale_tuned_gemm.csv"
_PRESHUFFLE_TUNE = "/mock/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
_PLAIN_UNTUNE = "aiter/configs/a8w8_blockscale_untuned_gemm.csv"
_PRESHUFFLE_UNTUNE = (
    "aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv"
)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _package(name, **attrs):
    mod = _module(name, **attrs)
    mod.__path__ = []
    return mod


class _StubGemmCommonTuner:
    """Only the argparse surface exercised by the route tests."""

    ARG_DEFAULTS = {
        "verbose": False,
        "tune_file": "",
        "untune_file": "",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
        "timeout": 1800,
        "warmup": 5,
        "iters": 101,
        "min_improvement_pct": 3.0,
    }
    INVALID_TIME = -1
    INF_TIME = float("inf")

    def __init__(self, name, keys, result_list, description=""):
        self.name = name
        self.keys = keys
        self.columns = keys + result_list
        self.parser = argparse.ArgumentParser(description=description)
        defaults = self.get_arg_defaults()
        self.parser.add_argument(
            "-i", "--untune_file", default=defaults["untune_file"]
        )
        self.parser.add_argument("-o", "--tune_file", default=defaults["tune_file"])
        self.parser.add_argument("--mp", type=int, default=0)
        self.parser.add_argument("-k", "--splitK", action="store_true")
        self.parser.add_argument("--shape_grouped", action="store_true")
        self.parser.add_argument(
            "--errRatio", type=float, default=defaults["errRatio"]
        )
        self.parser.add_argument("--batch", type=int, default=defaults["batch"])
        self.parser.add_argument("--warmup", type=int, default=defaults["warmup"])
        self.parser.add_argument("--iters", type=int, default=defaults["iters"])
        self.parser.add_argument("--timeout", type=int, default=defaults["timeout"])
        self.parser.add_argument("--compare", action="store_true")
        self.parser.add_argument("--update_improved", action="store_true")
        self._setup_specific_arguments()

    def get_arg_defaults(self):
        return self.ARG_DEFAULTS.copy()

    def parse_args(self):
        args = self.parser.parse_args()
        if args.update_improved and not args.compare:
            self.parser.error("--update_improved requires --compare")
        return args

    def run(self, args, fast_mode=False):
        return args

    def calculate(self, results, bpes=(1, 1, 2)):
        return 0, 0


def _load_tuner_module():
    dtypes = _module(
        "aiter.dtypes",
        bf16=object(),
        fp16=object(),
        fp32=object(),
        fp8=object(),
    )
    aiter = _package("aiter", dtypes=dtypes, logger=logging.getLogger("aiter-test"))
    opus_kernel = types.SimpleNamespace(
        name="gfx942_a8w8_bpreshuffle_11000",
        B_M=128,
        B_N=128,
        B_K=128,
        has_oob=True,
    )

    def get_kernel_instance(arch, family, kid, output_dtype=None):
        if (
            arch == "gfx942"
            and family == "a8w8_blockscale_bpreshuffle"
            and kid == 11000
            and output_dtype == "bf16_t"
        ):
            return opus_kernel
        return None

    stubs = {
        "aiter": aiter,
        "aiter.dtypes": dtypes,
        "aiter.jit": _package("aiter.jit"),
        "aiter.jit.core": _module(
            "aiter.jit.core",
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=_PLAIN_TUNE,
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=_PRESHUFFLE_TUNE,
            get_asm_dir=lambda: "/mock/asm",
        ),
        "aiter.jit.utils": _package("aiter.jit.utils"),
        "aiter.jit.utils.chip_info": _module(
            "aiter.jit.utils.chip_info",
            get_gfx_runtime=lambda: (_ for _ in ()).throw(
                AssertionError("GPU architecture detection must not run")
            ),
        ),
        "aiter.ops": _package("aiter.ops"),
        "aiter.ops.opus": _module("aiter.ops.opus", opus_gemm=lambda *a, **k: None),
        "aiter.ops.shuffle": _module(
            "aiter.ops.shuffle", shuffle_weight=lambda *a, **k: None
        ),
        "aiter.utility": _package("aiter.utility"),
        "aiter.utility.base_tuner": _module(
            "aiter.utility.base_tuner", GemmCommonTuner=_StubGemmCommonTuner
        ),
        "aiter.utility.mp_tuner": _module(
            "aiter.utility.mp_tuner", mp_tuner=lambda *a, **k: []
        ),
        "ck_gemm_a8w8_blockscale_bpreshuffle": _package(
            "ck_gemm_a8w8_blockscale_bpreshuffle"
        ),
        (
            "ck_gemm_a8w8_blockscale_bpreshuffle."
            "gemm_a8w8_blockscale_bpreshuffle_common"
        ): _module(
            "ck_gemm_a8w8_blockscale_bpreshuffle."
            "gemm_a8w8_blockscale_bpreshuffle_common",
            kernels_list=[],
        ),
        "gemm_a8w8_blockscale_cktile_instance": _module(
            "gemm_a8w8_blockscale_cktile_instance",
            BLOCK_PER_CU_MAX=1,
            candidate_kernels_cktile_dict={},
        ),
        "gemm_a8w8_blockscale_instance": _module(
            "gemm_a8w8_blockscale_instance", candidate_kernels_dict=[]
        ),
        "opus_gemm": _package("opus_gemm"),
        "opus_gemm.opus_gemm_common": _module(
            "opus_gemm.opus_gemm_common",
            get_kernel_instance=get_kernel_instance,
            kernels_list={11000: opus_kernel},
        ),
    }
    spec = importlib.util.spec_from_file_location(
        "_a8w8_blockscale_tune_route_test_module", _TUNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_sys_path = list(sys.path)
    try:
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


def _load_dtype_config_filter():
    """Compile only the pure policy helper, without importing aiter."""
    tree = ast.parse(_PRODUCTION_OP_PATH.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_filter_blockscale_bpreshuffle_config_for_dtype"
    )
    production_entry = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "gemm_a8w8_blockscale_bpreshuffle"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper.name
        for node in ast.walk(production_entry)
    )
    namespace = {"dtypes": types.SimpleNamespace(bf16="bf16")}
    helper_module = ast.Module(body=[helper], type_ignores=[])
    exec(compile(helper_module, str(_PRODUCTION_OP_PATH), "exec"), namespace)
    return namespace[helper.name]


_TUNER_MODULE = _load_tuner_module()
GemmA8W8BlockScaleTuner = _TUNER_MODULE.GemmA8W8BlockScaleTuner
_filter_config_for_dtype = _load_dtype_config_filter()


def _new_tuner():
    return GemmA8W8BlockScaleTuner("test", ["gfx", "cu_num", "M", "N", "K"], [])


def _parse(tuner, argv):
    with patch.object(sys, "argv", ["gemm_a8w8_blockscale_tune.py", *argv]):
        return tuner.parse_args()


class TestA8W8BlockscaleTuneRoutes(unittest.TestCase):
    def test_fp16_does_not_consume_bf16_only_opus_tuned_row(self):
        opus_config = {"libtype": "opus", "kernelId": 11000}
        ck_config = {"libtype": "ck", "kernelId": 7}

        self.assertIs(
            _filter_config_for_dtype(opus_config, "bf16"),
            opus_config,
        )
        self.assertIsNone(_filter_config_for_dtype(opus_config, "fp16"))
        self.assertIs(_filter_config_for_dtype(ck_config, "fp16"), ck_config)

    def test_plain_defaults_stay_plain(self):
        tuner = _new_tuner()
        args = _parse(tuner, [])

        self.assertEqual(args.tune_file, _PLAIN_TUNE)
        self.assertEqual(args.untune_file, _PLAIN_UNTUNE)
        self.assertEqual(
            tuner.get_arg_defaults()["config_env_name"],
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        )

    def test_preshuffle_defaults_switch_as_one_route(self):
        tuner = _new_tuner()
        args = _parse(tuner, ["--preshuffle"])

        self.assertEqual(args.tune_file, _PRESHUFFLE_TUNE)
        self.assertEqual(args.untune_file, _PRESHUFFLE_UNTUNE)
        self.assertEqual(
            tuner.get_arg_defaults()["config_env_name"],
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
        )

    def test_explicit_input_and_output_are_preserved(self):
        tuner = _new_tuner()
        args = _parse(
            tuner,
            ["--preshuffle", "-i", "/explicit/in.csv", "-o", "/explicit/out.csv"],
        )

        self.assertEqual(args.untune_file, "/explicit/in.csv")
        self.assertEqual(args.tune_file, "/explicit/out.csv")

    def test_preshuffle_mode_does_not_pollute_later_instances(self):
        first = _new_tuner()
        _parse(first, ["--preshuffle"])
        second = _new_tuner()
        args = _parse(second, [])

        self.assertEqual(args.tune_file, _PLAIN_TUNE)
        self.assertEqual(args.untune_file, _PLAIN_UNTUNE)
        self.assertEqual(
            GemmA8W8BlockScaleTuner.ARG_DEFAULTS["config_env_name"],
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        )

    def test_programmatic_namespace_gets_the_same_mode_defaults(self):
        tuner = _new_tuner()
        args = argparse.Namespace(
            preshuffle=True,
            tune_file=None,
            untune_file=None,
        )

        returned = tuner.run(args)

        self.assertIs(returned, args)
        self.assertEqual(args.tune_file, _PRESHUFFLE_TUNE)
        self.assertEqual(args.untune_file, _PRESHUFFLE_UNTUNE)

    def test_opus_only_rejects_plain_and_unregistered_arch_routes(self):
        with self.assertRaisesRegex(ValueError, "requires --preshuffle"):
            GemmA8W8BlockScaleTuner._validate_opus_route(
                "opus", False, "gfx942"
            )
        with self.assertRaisesRegex(ValueError, "no registered BF16 OPUS kernel"):
            GemmA8W8BlockScaleTuner._validate_opus_route(
                "opus", True, "gfx950"
            )

        # `all` remains valid because its CK/CKTile/ASM candidates are still useful.
        self.assertIsNone(
            GemmA8W8BlockScaleTuner._validate_opus_route(
                "all", True, "gfx950"
            )
        )

    def test_gfx942_opus_task_uses_public_kid_11000(self):
        tuner = _new_tuner()
        tasks = tuner.get_gemm_a8w8_blockscale_opus_tune_task(
            ("gfx942", 80, 1, 128, 128),
            seed=0,
            preshuffleB=True,
            run_kwargs={},
        )

        self.assertEqual(len(tasks), 1)
        info = tasks[0][0]
        self.assertEqual(info[1], 11000)
        self.assertEqual(info[4], "opus")
        self.assertTrue(info[5])
        self.assertEqual(
            tasks[0][4][0],
            ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"],
        )
        self.assertEqual(tasks[0][4][1], 11000)

    def test_opus_only_reports_shape_with_no_compatible_candidate(self):
        tuner = _new_tuner()
        tuner.get_cu_num = lambda: 80
        tuner.get_gfx = lambda: "gfx942"
        args = argparse.Namespace(
            splitK=False,
            mp=0,
            preshuffle=True,
            shape_grouped=False,
            errRatio=0.05,
            blockPerCu=[1],
            warmup=0,
            iters=1,
            timeout=1,
            verbose=False,
            libtype="opus",
        )
        shapes = pd.DataFrame([{"M": 1, "N": 64, "K": 128}])

        with self.assertRaisesRegex(ValueError, "shape M=1, N=64, K=128"):
            tuner.tune(shapes, pd.DataFrame(), args)


if __name__ == "__main__":
    unittest.main()
