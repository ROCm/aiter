# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU regressions for tuning orchestration; no torch, Triton, or GPU required.

python -m unittest discover -s op_tests/triton_tests/gemm -p test_tune_gemm.py
"""

import ast
import csv
import importlib.util
import inspect
import io
import json
import logging
import math
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "aiter/ops/triton/utils"
TUNING = UTILS / "_triton/tuning"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Tensor:
    """Small tensor double whose float() intentionally aliases float32 storage."""

    def __init__(self, values, shape=None, dtype="float32"):
        self.values = list(values)
        self.shape = shape if shape is not None else (len(values),)
        self.dtype = dtype

    def detach(self):
        return self

    def float(self):
        return self if self.dtype == "float32" else Tensor(self.values, self.shape)

    def is_floating_point(self):
        return self.dtype.startswith("float")

    def clone(self):
        return Tensor(self.values, self.shape, self.dtype)

    def norm(self):
        return Tensor([math.sqrt(sum(x * x for x in self.values))], ())

    def clamp_min(self, minimum):
        return Tensor([max(x, minimum) for x in self.values], self.shape)

    def item(self):
        return self.values[0]

    def __sub__(self, other):
        return Tensor([a - b for a, b in zip(self.values, other.values)], self.shape)

    def __truediv__(self, other):
        return Tensor([self.item() / other.item()], ())


class TuningTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aiter-tuning-test-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.arch = "gfx950"
        packages = [
            "aiter",
            "aiter.ops",
            "aiter.ops.triton",
            "aiter.ops.triton.utils",
            "aiter.ops.triton.utils._triton",
        ]
        modules = {name: types.ModuleType(name) for name in packages}
        modules["torch"] = types.SimpleNamespace(
            Tensor=Tensor,
            equal=lambda actual, expected: actual.values == expected.values,
            cuda=types.SimpleNamespace(synchronize=lambda: None),
            isfinite=lambda tensor: types.SimpleNamespace(
                all=lambda: all(math.isfinite(x) for x in tensor.values)
            ),
        )
        modules["triton"] = types.SimpleNamespace(jit=lambda fn: fn)
        modules["triton.language"] = types.ModuleType("triton.language")
        modules["triton"].language = modules["triton.language"]
        modules["triton.testing"] = types.SimpleNamespace(runtime=None)
        self.arch_info = types.SimpleNamespace(get_arch=lambda: self.arch)
        modules["aiter.ops.triton.utils._triton"].arch_info = self.arch_info
        modules["aiter.ops.triton.utils.logger"] = types.SimpleNamespace(
            AiterTritonLogger=lambda: logging.getLogger(__name__)
        )
        # flock is a runtime dependency on the Linux GPU host, not of these tests.
        modules["fcntl"] = types.SimpleNamespace(flock=Mock(), LOCK_EX=2, LOCK_UN=8)
        isolated_modules = patch.dict(sys.modules, modules)
        isolated_modules.start()
        self.addCleanup(isolated_modules.stop)
        self.config = load_module(
            "aiter.ops.triton.utils.config_utils", UTILS / "config_utils.py"
        )
        self.gemm = load_module(
            "aiter.ops.triton.utils.gemm_config_utils", UTILS / "gemm_config_utils.py"
        )
        modules["aiter.ops.triton.utils"].config_utils = self.config
        modules["aiter.ops.triton.utils"].gemm_config_utils = self.gemm
        self.cases = load_module("gemm_cases", TUNING / "gemm_cases.py")
        self.tuner = load_module("_test_tune_gemm", TUNING / "tune_gemm.py")
        self.worker = load_module(
            "_test_profile_configs", TUNING / "profile_configs.py"
        )
        self.config.AITER_TRITON_CONFIGS_PATH = str(self.directory)
        self.tuner.config_utils = self.config
        self.tuner.gemm_config_utils = self.gemm
        self.tuner.arch_info = self.arch_info
        self.tuner.CONFIGS_ROOT = str(self.directory)
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def write_table(self, table, backend="triton", filename="DEFAULT.json"):
        directory = Path(
            self.config.resolve_config_dir("gemm", "GEMM-TEST", backend=backend)
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(json.dumps(table))
        self.config.load_config_json.cache_clear()
        self.gemm._get_gemm_config_cached.cache_clear()
        return path

    def lookup(self, **changes):
        return {
            "config_name": "GEMM-TEST",
            "M": 12,
            "N": 64,
            "K": 256,
            "B": None,
            "bounds": None,
            "specialized_filename": None,
            "backend": "triton",
        } | changes

    def family(self, **changes):
        return self.tuner.find_family(
            {"lookups": [self.lookup(**changes)], "config": {"variant": "base"}},
            "test_gemm",
        )

    def test_all_arches_and_backends_write_through_real_loader(self):
        for arch in ("gfx942", "gfx950", "gfx1250", "gfxfuture"):
            for backend in ("triton", "gluon"):
                with self.subTest(arch=arch, backend=backend):
                    self.arch = arch
                    default_path = self.write_table(
                        {"any": {"kernel_variant": "base", "new_key": 1}}, backend
                    )
                    original = default_path.read_bytes()
                    family = self.family(backend=backend)
                    candidate = {"kernel_variant": "fast", "new_key": 7}
                    written = self.tuner.write_config(family, 12, candidate)
                    self.assertEqual(self.tuner.resolve(family["lookup"], 12), written)
                    self.assertEqual(written, candidate)
                    self.assertEqual(default_path.read_bytes(), original)
                    self.assertIn(
                        f"/{arch}/{backend}/", family["target_path"].replace("\\", "/")
                    )

    def test_bucket_writes_beat_existing_geq_and_preserve_neighbors(self):
        self.write_table(
            {
                "M_BOUNDS": [3, 10, 30],
                "M_LEQ_3": {"variant": "tiny"},
                "M_GEQ_30": {"variant": "large"},
                "any": {"variant": "base"},
            }
        )
        family = self.family(M=100)
        self.tuner.write_config(family, 100, {"variant": "winner"})
        self.assertEqual(
            self.tuner.resolve(family["lookup"], 100), {"variant": "winner"}
        )
        self.assertEqual(self.tuner.resolve(family["lookup"], 2), {"variant": "tiny"})
        self.assertEqual(self.tuner.resolve(family["lookup"], 9), {"variant": "base"})

    def test_explicit_bounds_override_file_bounds(self):
        self.write_table({"M_BOUNDS": [3, 10], "any": {"variant": "base"}})
        family = self.family(bounds=[4, 16])
        self.tuner.write_config(family, 12, {"variant": "winner"})
        self.assertEqual(
            self.tuner.resolve(family["lookup"], 12), {"variant": "winner"}
        )
        table = json.loads(Path(family["target_path"]).read_text())
        self.assertIn("M_LEQ_16", table)

    def test_batched_target_preserves_existing_nk_source(self):
        self.write_table({"any": {"variant": "base"}})
        source = self.write_table(
            {"M_LEQ_4": {"variant": "small"}, "any": {"variant": "shared"}},
            filename="GEMM-TEST-N=64-K=256.json",
        )
        original = source.read_bytes()
        family = self.family(B=5)
        self.assertEqual(Path(family["source_path"]), source)
        self.tuner.write_config(family, 12, {"variant": "batch"})
        self.assertEqual(self.tuner.resolve(family["lookup"], 12), {"variant": "batch"})
        self.assertEqual(self.tuner.resolve(family["lookup"], 2), {"variant": "small"})
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(
            Path(family["target_path"]).name, "GEMM-TEST-B=5-N=64-K=256.json"
        )

    def test_custom_suffix_uses_lookup_dimensions(self):
        self.write_table({"any": {"variant": "base"}})
        family = self.family(K=512, specialized_filename="N4=32-N16=64-K=512")
        self.assertEqual(
            Path(family["target_path"]).name, "GEMM-TEST-N4=32-N16=64-K=512.json"
        )
        self.tuner.write_config(family, 12, {"variant": "winner"})
        self.assertEqual(
            self.tuner.resolve(family["lookup"], 12), {"variant": "winner"}
        )

    def test_unknown_keys_keep_json_values_and_accept_overrides(self):
        self.write_table({"any": {"variant": "base", "new_key": 7}}, "gluon")
        self.arch = "gfx1250"
        self.write_table({"any": {"variant": "other", "new_key": 11}}, "gluon")
        self.write_table({"any": {"variant": "wrong_backend", "new_key": 99}})
        family = self.family(backend="gluon")
        space = self.tuner.search_space(family, 12, 64, 256, {"new_key": [13, 17]})
        self.assertEqual(space["new_key"], [13, 17])
        self.assertEqual(set(space["variant"]), {"base", "other"})

    def test_search_keeps_observed_small_tiles_and_launch_options(self):
        observed = {
            "BLOCK_SIZE_K": [32],
            "num_warps": [2],
            "matrix_instr_nonkdim": [32],
            "BLOCK_M": [64],
        }
        for key, values in observed.items():
            with self.subTest(key=key):
                choices = self.tuner.candidate_values(key, 12, 64, 256, observed)
                self.assertIn(values[0], choices)

    def test_lookup_hook_records_m_and_observes_imported_loader_alias(self):
        self.write_table({"any": {"variant": "base"}})
        imported_lookup = self.gemm.get_gemm_config
        path = self.directory / "record.json"
        hook = self.worker.LookupHook(path)
        self.assertEqual(
            imported_lookup("GEMM-TEST", 6, 64, 256)[0], {"variant": "base"}
        )
        record = json.loads(path.read_text())
        self.assertEqual(record["lookups"][0]["M"], 6)
        hook.override = {"variant": "candidate"}
        selected, tuned = imported_lookup("GEMM-TEST", 6, 64, 256)
        self.assertEqual(selected, hook.override)
        self.assertTrue(tuned)
        selected["variant"] = "mutated by wrapper"
        self.assertEqual(hook.override, {"variant": "candidate"})
        for changes in ({"M": 12}, {"backend": "gluon"}, {"config_name": "GEMM-OTHER"}):
            lookup = self.lookup(M=6) | changes
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, "changed.*lookup"
            ):
                imported_lookup(**lookup)

    def test_snapshot_detaches_reused_float32_outputs(self):
        output = Tensor([1, 2])
        reference = self.worker.output_of((output, {"second": Tensor([3])}))
        output.values[:] = [8, 9]
        self.assertEqual(reference[0].values, [1, 2])
        self.assertEqual(reference[1].values, [3])
        with self.assertRaisesRegex(ValueError, "output 0 differs"):
            self.worker.check_output(
                self.worker.output_of((output, Tensor([3]))), reference
            )

    def test_checks_second_output_and_tensor_count(self):
        reference = self.worker.output_of((Tensor([1]), Tensor([2])))
        with self.assertRaisesRegex(ValueError, "output 1 differs"):
            self.worker.check_output([Tensor([1]), Tensor([20])], reference)
        with self.assertRaisesRegex(ValueError, "tensor count"):
            self.worker.check_output([Tensor([1])], reference)

    def test_integer_scales_require_exact_values_and_matching_dtype(self):
        expected = self.worker.output_of(Tensor([127, 128], dtype="uint8"))
        with self.assertRaisesRegex(ValueError, "integer values differ"):
            self.worker.check_output([Tensor([128, 128], dtype="uint8")], expected)
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.worker.check_output([Tensor([127, 128])], expected)
        self.worker.check_output([Tensor([127, 128], dtype="uint8")], expected)

    def test_nonfinite_relative_error_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            self.worker.check_output([Tensor([-1e200])], [Tensor([1e200])])

    def test_fp8_outputs_are_converted_before_finite_checks(self):
        isfinite = self.worker.torch.isfinite

        def float32_isfinite(tensor):
            if tensor.dtype != "float32":
                raise NotImplementedError("isfinite does not support this FP8 dtype")
            return isfinite(tensor)

        with patch.object(self.worker.torch, "isfinite", float32_isfinite):
            output = Tensor([1, 2], dtype="float8_e4m3fn")
            self.worker.check_output([output], self.worker.output_of(output))
            with self.assertRaisesRegex(ValueError, "output 0 has NaN"):
                self.worker.check_output(
                    [Tensor([float("nan"), 2], dtype="float8_e4m3fn")], [output]
                )

    def test_rejects_missing_nonfinite_and_wrong_shape_outputs(self):
        for reference in (None, []):
            with self.assertRaisesRegex(ValueError, "successful current config"):
                self.worker.check_output([Tensor([1])], reference)
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "reference output"):
                self.worker.check_output([Tensor([1])], [Tensor([value])])
            with self.assertRaisesRegex(ValueError, "output 0 has NaN or Inf"):
                self.worker.check_output([Tensor([value])], [Tensor([1])])
        with self.assertRaisesRegex(ValueError, "shape"):
            self.worker.check_output([Tensor([1, 2])], [Tensor([1])])

    def run_worker(self, run, candidates, profile=None, check=True):
        self.worker.CASES = {"test_gemm": lambda **kwargs: run}
        spec = {
            "op": "test_gemm",
            "dims": {"M": 12, "N": 64, "K": 256},
            "backend": None,
            "record": str(self.directory / "record.json"),
            "status": str(self.directory / "status.txt"),
            "run_current": True,
            "check": check,
            "candidates": candidates,
        }
        path = self.directory / "spec.json"
        path.write_text(json.dumps(spec))
        with patch.object(
            self.worker, "profile", profile or (lambda run: None)
        ), patch.object(self.worker, "mark"):
            self.worker.main(path)
        return Path(spec["status"]).read_text().splitlines()

    def test_worker_checks_all_reused_outputs_and_does_not_prune_tiles(self):
        self.write_table({"any": {"variant": "base"}})
        first, second = Tensor([1]), Tensor([2])

        def run():
            config, _ = self.gemm.get_gemm_config("GEMM-TEST", 12, 64, 256)
            if config["variant"] == "resource_error":
                raise RuntimeError("OutOfResources")
            second.values[:] = [20 if config["variant"] == "wrong" else 2]
            return first, second

        statuses = self.run_worker(
            run,
            [{"variant": value} for value in ("resource_error", "wrong", "good")],
        )
        self.assertEqual(statuses[:2], ["ready", "ok"])
        self.assertIn("OutOfResources", statuses[2])
        self.assertIn("output 1 differs", statuses[3])
        self.assertEqual(statuses[4], "ok")

    def test_worker_does_not_use_reference_when_baseline_profiling_fails(self):
        attempts = []

        def profile(run):
            attempts.append(True)
            raise RuntimeError("baseline failed after first invocation")

        statuses = self.run_worker(
            lambda: Tensor([1]), [{"variant": "candidate"}], profile
        )
        self.assertIn("baseline failed", statuses[1])
        self.assertIn("successful current config", statuses[2])
        self.assertEqual(len(attempts), 1)

    def test_profile_marks_every_measured_invocation(self):
        events = []

        class Marker:
            def __getitem__(self, grid):
                return lambda buffer: events.append("marker")

        driver = types.SimpleNamespace(
            get_device_interface=lambda: types.SimpleNamespace(
                synchronize=lambda: None
            ),
            get_empty_cache_for_benchmark=lambda: types.SimpleNamespace(
                zero_=lambda: None
            ),
        )
        torch = types.SimpleNamespace(
            zeros=lambda *args, **kwargs: object(), float32="float32"
        )
        with patch.object(self.worker, "N_RUNS", 3), patch.object(
            self.worker,
            "runtime",
            types.SimpleNamespace(driver=types.SimpleNamespace(active=driver)),
        ), patch.object(self.worker, "torch", torch), patch.object(
            self.worker, "run_dummy", Marker()
        ):
            self.worker.profile(lambda: events.append("run"))
        self.assertEqual(
            events, ["marker", "run", "marker", "run", "marker", "run", "marker"]
        )

    def test_repeated_kernel_launches_sum_per_run_and_ignore_validation(self):
        path = self.directory / "trace.csv"
        rows, clock = [], 0

        def launch(name, nanoseconds=1):
            nonlocal clock
            rows.append((name, clock, clock + nanoseconds))
            clock += nanoseconds + 1

        launch("gemm_validation", 999999)
        launch("run_dummy")
        for _ in range(3):
            launch("gemm_same_name", 1000)
            launch("gemm_same_name", 2000)
            launch("_ff_a16w16_fused_", 3000)
            launch("flush_cache", 500000)
            launch("run_dummy")
        launch("split_dummy")
        launch("split_dummy")
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Kernel_Name", "Start_Timestamp", "End_Timestamp"])
            writer.writerows(reversed(rows))
        self.assertEqual(self.tuner.kernel_times(path), [6.0, None])

    def test_child_success_requires_status_exit_and_positive_finite_times(self):
        good = self.tuner.Child(["ready", "ok", "ok"], [10.0, 5.0], None, None, "")
        self.assertTrue(self.tuner.child_succeeded(good, 2))
        failures = [
            good._replace(statuses=["ready", "ok", "error: wrong output"]),
            good._replace(statuses=["ready", "ok"]),
            good._replace(ended="exited with code 3"),
            good._replace(times=[10.0, None]),
            good._replace(times=[10.0, float("nan")]),
            good._replace(times=[10.0, float("inf")]),
            good._replace(times=[10.0, 0.0]),
            good._replace(times=[10.0, -1.0]),
        ]
        for child in failures:
            with self.subTest(child=child):
                self.assertFalse(self.tuner.child_succeeded(child, 2))

    def test_resource_error_does_not_skip_other_configs_and_gpu_fault_retries_rest(
        self,
    ):
        configs = [{"variant": index} for index in range(3)]
        first = self.tuner.Child(
            ["ready", "ok", "error: OutOfResources", "error: HIP error"],
            [10.0, None],
            None,
            "exited with code 3",
            "",
        )
        second = self.tuner.Child(["ready", "ok", "ok"], [10.0, 5.0], None, None, "")
        errors = Mock()
        with patch.object(
            self.tuner, "run_child", side_effect=[first, second]
        ) as run_child:
            best, elapsed = self.tuner.try_candidates(
                {"check": True}, configs, True, 10, errors
            )
        self.assertEqual((best, elapsed), (configs[2], 5.0))
        self.assertEqual(run_child.call_args_list[0].args[1], configs)
        self.assertEqual(run_child.call_args_list[1].args[1], [configs[2]])
        self.assertEqual(errors.add.call_count, 2)

    def test_timeout_blames_only_active_config_and_retries_lost_trace_once(self):
        configs = [{"variant": index} for index in range(3)]
        timeout = self.tuner.Child(["ready", "ok", "ok"], [], None, "timed out", "")
        retried = self.tuner.Child(
            ["ready", "ok", "ok", "ok"], [10.0, None, 7.0], None, None, ""
        )
        errors = Mock()
        with patch.object(
            self.tuner, "run_child", side_effect=[timeout, retried]
        ) as run_child:
            best, elapsed = self.tuner.try_candidates(
                {"check": True}, configs, True, 10, errors
            )
        self.assertEqual((best, elapsed), (configs[2], 7.0))
        self.assertEqual(run_child.call_args_list[1].args[1], [configs[0], configs[2]])
        failed = [call.args[0] for call in errors.add.call_args_list]
        self.assertEqual(failed, [configs[1], configs[0]])

    def run_tuner_main(self, final=None, no_check=False, baseline_ok=True):
        default_path = self.write_table({"any": {"variant": "base"}})
        record = {"lookups": [self.lookup(M=6)], "config": {"variant": "base"}}
        baseline = self.tuner.Child(
            ["ready", "ok" if baseline_ok else "error: no reference"],
            [10.0] if baseline_ok else [],
            record,
            None,
            "",
        )
        args = types.SimpleNamespace(
            list=False,
            op="test_gemm",
            dims=["M=12", "N=64", "K=256"],
            backend=None,
            timeout=10,
            no_check=no_check,
            space=[],
        )
        candidate = {"variant": "candidate"}
        children = [baseline] + ([final] if final is not None else [])
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(self.tuner, "parse_args", return_value=args)
            )
            stack.enter_context(
                patch.object(
                    self.tuner,
                    "get_case",
                    return_value=types.SimpleNamespace(
                        space={}, kernel_names=("gemm",)
                    ),
                )
            )
            stack.enter_context(patch.object(self.tuner, "ErrorLog"))
            stack.enter_context(
                patch.object(self.tuner, "run_child", side_effect=children)
            )
            stack.enter_context(
                patch.object(
                    self.tuner, "try_candidates", return_value=(candidate, 1.0)
                )
            )
            self.tuner.main()
        return default_path.parent / "GEMM-TEST-N=64-K=256.json"

    def test_final_failed_or_incomplete_retime_never_writes_even_with_fast_time(self):
        success = self.tuner.Child(["ready", "ok", "ok"], [10.0, 1.0], None, None, "")
        for final in (
            success._replace(statuses=["ready", "ok", "error: output mismatch"]),
            success._replace(ended="exited with code 3"),
            success._replace(times=[10.0, None]),
        ):
            with self.subTest(final=final), self.assertRaisesRegex(
                SystemExit, "nothing written"
            ):
                self.run_tuner_main(final)
            self.assertFalse(list(self.directory.rglob("GEMM-TEST-*.json")))

    def test_no_check_explores_without_writing_when_reference_fails(self):
        target = self.run_tuner_main(no_check=True, baseline_ok=False)
        self.assertFalse(target.exists())

    def test_failed_reference_aborts_checked_run(self):
        with self.assertRaisesRegex(SystemExit, "valid reference"):
            self.run_tuner_main(baseline_ok=False)
        self.assertFalse(list(self.directory.rglob("GEMM-TEST-*.json")))

    def test_successful_retime_writes_bucket_for_actual_lookup_m(self):
        final = self.tuner.Child(["ready", "ok", "ok"], [10.0, 4.0], None, None, "")
        target = self.run_tuner_main(final)
        self.assertEqual(
            json.loads(target.read_text()),
            {"M_LEQ_8": {"variant": "candidate"}, "any": {"variant": "base"}},
        )

    def test_parse_dims_rejects_zero_negative_and_duplicate_dimensions(self):
        for tokens in (["M=0"], ["M=-1"], ["M=2", "M=3"], ["N=3"]):
            with self.subTest(tokens=tokens), self.assertRaises(SystemExit):
                self.tuner.parse_dims(tokens)

    def test_case_registry_has_explicit_dimensions_and_list_needs_no_gpu(self):
        for name, case in self.cases.CASES.items():
            parameters = inspect.signature(case).parameters
            self.assertEqual(case.__name__, name)
            self.assertIn("M", parameters)
            self.assertTrue(
                all(p.kind == p.POSITIONAL_OR_KEYWORD for p in parameters.values())
            )
        child = subprocess.run(
            [sys.executable, str(TUNING / "tune_gemm.py"), "--list"],
            capture_output=True,
            text=True,
            check=True,
        )
        for name in self.cases.CASES:
            self.assertIn(name, child.stdout)

    def test_every_public_gemm_has_a_case_or_documented_exception(self):
        exceptions = {
            "ff_a16w16_gated",  # composite: tune its constituent GEMMs
            "ff_a16w16_nogate",  # composite: tune its constituent GEMMs
            "gemm_afp4wfp4_preshuffled_weight_scales",  # deprecated alias
        }
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
        self.assertEqual(missing, exceptions)
        for node in ast.walk(ast.parse((TUNING / "gemm_cases.py").read_text())):
            if isinstance(node, ast.Call):
                self.assertFalse(
                    any(keyword.arg == "config" for keyword in node.keywords)
                )

    def test_preshuffle_backend_reaches_matching_config_lookup(self):
        path = ROOT / "aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py"
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "gemm_afp4wfp4_preshuffle"
        )
        arches = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_GLUON_PRESHUFFLE_ARCHS"
                for target in node.targets
            )
        )
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
        module = ast.Module(body=[function], type_ignores=[])

        class LookupReached(Exception):
            pass

        lookup = Mock(side_effect=LookupReached)
        namespace = {
            "torch": types.SimpleNamespace(bfloat16="bf16"),
            "arch_info": types.SimpleNamespace(
                get_arch=lambda: self.arch, is_fp4_avail=lambda: True
            ),
            "_GLUON_PRESHUFFLE_ARCHS": arches,
            "_get_config": lookup,
        }
        exec(  # noqa: S102 -- local source under test
            compile(module, str(path), "exec"), namespace
        )
        op = namespace[function.name]
        inputs = (
            types.SimpleNamespace(shape=(32, 128)),
            types.SimpleNamespace(shape=(2, 4096)),
            None,
            None,
        )
        for arch, requested, selected in (
            ("gfx1250", None, "gluon"),
            ("gfx1250", "triton", "triton"),
            ("gfx1250", "gluon", "gluon"),
            ("gfx950", None, "triton"),
        ):
            self.arch = arch
            with self.subTest(arch=arch, backend=requested), self.assertRaises(
                LookupReached
            ):
                op(*inputs, backend=requested)
            lookup.assert_called_with(32, 32, 128, True, backend=selected)
        with self.assertRaises(AssertionError):
            op(*inputs, backend="gluon")
