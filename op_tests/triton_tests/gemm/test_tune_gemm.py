# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests: python -m unittest discover -s op_tests/triton_tests/gemm -p test_tune_gemm.py"""

import ast
import importlib.util
import io
import json
import logging
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "aiter/ops/triton/utils"
TUNING = UTILS / "_triton/tuning"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TuningTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="aiter-tuning-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.arch = "gfx950"
        packages = (
            "aiter",
            "aiter.ops",
            "aiter.ops.triton",
            "aiter.ops.triton.utils",
            "aiter.ops.triton.utils._triton",
        )
        modules = {name: types.ModuleType(name) for name in packages}
        arch_info = types.SimpleNamespace(get_arch=lambda: self.arch)
        modules["aiter.ops.triton.utils._triton"].arch_info = arch_info
        modules["aiter.ops.triton.utils._triton.arch_info"] = arch_info
        modules["aiter.ops.triton.utils.logger"] = types.SimpleNamespace(
            AiterTritonLogger=lambda: logging.getLogger(__name__)
        )
        modules["triton"] = types.ModuleType("triton")
        modules["triton.testing"] = types.SimpleNamespace(do_bench=self.benchmark)
        self.stack.enter_context(patch.dict(sys.modules, modules))
        self.config = load_module(
            "aiter.ops.triton.utils.config_utils", UTILS / "config_utils.py"
        )
        self.gemm = load_module(
            "aiter.ops.triton.utils.gemm_config_utils", UTILS / "gemm_config_utils.py"
        )
        modules["aiter.ops.triton.utils"].gemm_config_utils = self.gemm
        self.cases = load_module("gemm_cases", TUNING / "gemm_cases.py")
        self.tuner = load_module("_test_tune_gemm", TUNING / "tune_gemm.py")
        self.config.AITER_TRITON_CONFIGS_PATH = str(self.directory)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        # Exercise real orchestration and loaders without GPU tensor operations.
        self.stack.enter_context(
            patch.object(self.tuner, "snapshot", side_effect=lambda x: (x,))
        )
        self.check = self.stack.enter_context(patch.object(self.tuner, "check_outputs"))
        self.errors = io.StringIO()
        self.lookup = {
            "config_name": "GEMM-TEST",
            "M": 12,
            "N": 64,
            "K": 256,
            "bounds": None,
            "specialized_filename": None,
            "backend": "triton",
            "B": None,
        }

    def clear_caches(self):
        self.config.load_config_json.cache_clear()
        self.gemm._get_gemm_config_cached.cache_clear()

    def table(self, contents, filename="DEFAULT.json"):
        directory = Path(
            self.config.resolve_config_dir(
                "gemm", "GEMM-TEST", backend=self.lookup["backend"]
            )
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(json.dumps(contents))
        self.clear_caches()
        return path

    def benchmark(self, run, return_mode):
        self.assertEqual(return_mode, "median")
        run()
        return self.timings[self.last_choice]

    def tune(self, timings, candidates):
        self.timings = timings
        hook = self.tuner.ConfigLookup(self.gemm._get_gemm_config_cached)

        def run():
            config, _ = self.gemm.get_gemm_config(**self.lookup)
            self.last_choice = config["choice"]
            if isinstance(timings[self.last_choice], Exception):
                raise timings[self.last_choice]
            return "output"

        with patch.object(self.gemm, "_get_gemm_config_cached", hook):
            self.tuner.tune(run, hook, {"choice": candidates}, self.errors)
        self.clear_caches()
        return self.tuner.find_family(self.lookup)[3]

    def test_existing_config_is_replaced_only_when_candidate_is_faster(self):
        self.table({"any": {"choice": "base"}})
        for candidate_time in (3.0, 5.0, 7.0):
            with self.subTest(candidate_time=candidate_time):
                path = self.table(
                    {"M_LEQ_16": {"choice": "base"}, "any": {"choice": "base"}},
                    "GEMM-TEST-N=64-K=256.json",
                )
                original = path.read_bytes()
                self.tune({"base": 5.0, "candidate": candidate_time}, ["candidate"])
                winner = "candidate" if candidate_time < 5.0 else "base"
                self.assertEqual(
                    self.gemm.get_gemm_config(**self.lookup)[0], {"choice": winner}
                )
                if winner == "base":
                    self.assertEqual(path.read_bytes(), original)

    def test_missing_tuned_config_records_best_including_default(self):
        for candidate_time, winner, existing_any in (
            (8.0, "base", False),
            (2.0, "candidate", False),
            (8.0, "base", True),
        ):
            with self.subTest(winner=winner):
                self.table({"any": {"choice": "base"}})
                target = self.tuner.find_family(self.lookup)[3]
                target.unlink(missing_ok=True)
                if existing_any:
                    target.write_text(json.dumps({"any": {"choice": "base"}}))
                self.clear_caches()
                self.assertFalse(self.gemm.get_gemm_config(**self.lookup)[1])
                self.tune({"base": 5.0, "candidate": candidate_time}, ["candidate"])
                self.assertEqual(
                    self.gemm.get_gemm_config(**self.lookup), ({"choice": winner}, True)
                )

    def test_failed_baseline_continues_and_failed_candidates_are_logged(self):
        self.table({"any": {"choice": "base"}})
        target = self.tune(
            {
                "base": ValueError("baseline failure"),
                "bad": RuntimeError("compile failure"),
                "good": 3.0,
            },
            ["bad", "good"],
        )
        self.assertTrue(target.exists())
        self.assertEqual(
            self.gemm.get_gemm_config(**self.lookup)[0], {"choice": "good"}
        )
        for text in (
            '"base"',
            '"bad"',
            "Traceback",
            "baseline failure",
            "compile failure",
        ):
            self.assertIn(text, self.errors.getvalue())
        self.assertIsNone(self.check.call_args.args[1])

    def test_all_failed_configs_write_nothing(self):
        self.table({"any": {"choice": "base"}})
        with self.assertRaisesRegex(RuntimeError, "No config worked"):
            self.tune(
                {
                    "base": ValueError("baseline"),
                    "bad": ValueError("candidate"),
                    "invalid": float("nan"),
                },
                ["bad", "invalid"],
            )
        self.assertFalse(self.tuner.find_family(self.lookup)[3].exists())
        self.assertIn("Traceback", self.errors.getvalue())

    def test_arch_backend_custom_and_batched_paths_use_real_loader(self):
        for arch in ("gfx942", "gfx950", "gfx1250", "gfxfuture"):
            for backend in ("triton", "gluon"):
                self.arch, self.lookup["backend"] = arch, backend
                self.table({"any": {"choice": "base"}})
                for suffix in (None, "N4=32-N16=64-K=256"):
                    self.lookup.update(B=5, specialized_filename=suffix)
                    directory, _, source, target = self.tuner.find_family(self.lookup)
                    self.assertIn(f"/{arch}/{backend}/", directory.as_posix())
                    expected = suffix or "B=5-N=64-K=256"
                    self.assertEqual(target.name, f"GEMM-TEST-{expected}.json")
                    self.tuner.write_config(
                        source, target, self.lookup, {"choice": "winner"}
                    )
                    self.clear_caches()
                    self.assertEqual(
                        self.gemm.get_gemm_config(**self.lookup),
                        ({"choice": "winner"}, True),
                    )

    def test_custom_bounds_and_batch_fallback_preserve_other_buckets(self):
        self.table({"any": {"choice": "base"}})
        path = self.table(
            {
                "M_BOUNDS": [3, 10],
                "M_LEQ_3": {"choice": "small"},
                "M_GEQ_10": {"choice": "large"},
                "any": {"choice": "base"},
            },
            "GEMM-TEST-N=64-K=256.json",
        )
        self.lookup.update(M=100, B=5)
        _, _, source, target = self.tuner.find_family(self.lookup)
        self.assertEqual(source, path)
        self.tuner.write_config(source, target, self.lookup, {"choice": "winner"})
        self.clear_caches()
        self.assertEqual(
            self.gemm.get_gemm_config(**self.lookup)[0], {"choice": "winner"}
        )
        self.assertEqual(json.loads(target.read_text())["M_LEQ_3"], {"choice": "small"})
        self.lookup.update(M=12, bounds=(4, 16))
        self.tuner.write_config(target, target, self.lookup, {"choice": "explicit"})
        self.clear_caches()
        self.assertEqual(
            self.gemm.get_gemm_config(**self.lookup)[0], {"choice": "explicit"}
        )

    def test_arbitrary_config_keys_and_overrides(self):
        self.table({"any": {"new_key": "base", "buffer_count": 1}})
        self.table(
            {"any": {"new_key": "published", "buffer_count": 2}},
            "GEMM-TEST-N=32-K=128.json",
        )
        directory, defaults, _, _ = self.tuner.find_family(self.lookup)
        configs = list(
            self.tuner.candidate_configs(
                directory, defaults, self.lookup, {"buffer_count": [7]}
            )
        )
        self.assertEqual(
            configs,
            [
                {"new_key": "base", "buffer_count": 7},
                {"new_key": "published", "buffer_count": 7},
            ],
        )
        ignored = list(
            self.tuner.candidate_configs(
                directory, defaults, self.lookup, {"buffer_count": [7], "typo": [1]}
            )
        )
        self.assertEqual(ignored, configs)

    def test_main_restores_lookup_after_error(self):
        def case(M):
            return lambda: None

        case.space = {}
        original = self.gemm._get_gemm_config_cached
        args = types.SimpleNamespace(
            list=False, op="test", dims=["M=12"], backend=None, space=[]
        )
        with patch.object(self.tuner, "CASES", {"test": case}), patch.object(
            self.tuner, "parse_args", return_value=args
        ), patch.object(
            self.tuner, "Path", return_value=self.directory / "errors.txt"
        ), patch.object(
            self.tuner, "tune", side_effect=RuntimeError("failed")
        ), self.assertRaisesRegex(
            RuntimeError, "failed"
        ):
            self.tuner.main()
        self.assertIs(self.gemm._get_gemm_config_cached, original)

    def test_registry_coverage_and_cpu_list(self):
        missing = set()
        for path in (ROOT / "aiter/ops/triton/gemm").rglob("*.py"):
            for node in ast.parse(path.read_text()).body:
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name.startswith(
                        ("gemm_", "batched_gemm_", "fused_gemm_", "ff_")
                    )
                    and not node.name.endswith(("_", "_fake_tensor"))
                    and node.name not in self.cases.CASES
                ):
                    missing.add(node.name)
        self.assertEqual(
            missing,
            {
                "ff_a16w16_gated",
                "ff_a16w16_nogate",  # composites
                "gemm_afp4wfp4_preshuffled_weight_scales",
            },
        )  # deprecated alias
        child = subprocess.run(
            [sys.executable, str(TUNING / "tune_gemm.py"), "--list"],
            capture_output=True,
            text=True,
            check=True,
        )
        for name in self.cases.CASES:
            self.assertIn(name, child.stdout)
