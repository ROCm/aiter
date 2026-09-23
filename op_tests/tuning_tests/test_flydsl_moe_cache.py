# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side regressions for the first two-stage FlyDSL MoE refactor.

These tests need the installed aiter/FlyDSL packages, but never launch GPU
kernels. Most replace compilation and device boundaries; fresh lightweight
processes also compile into private caches through the real AOT fork workers.
Parsing, validation, launcher construction, and caches are real.
Numerical and cold-process RUN_ONLY coverage lives in test_flydsl_moe_run_only.
"""

import contextlib
import csv
import importlib
import inspect
import io
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[2]
_PREFIX = "impl__flydsl_gfx942__"
_DECODE = "16_16_16_False"
_PREFILL = "64_128_128_True"
_LIGHTWEIGHT_MARKER = "FLYDSL_MOE_LIGHTWEIGHT_RESULT "
_RUNTIME_MOE_MODULES = (
    "aiter.ops.flydsl.fused_moe_gfx942",
    "aiter.fused_moe",
    "aiter.ops.enum",
    "aiter.ops.moe_op",
    "aiter.ops.moe_sorting",
    "aiter.ops.quant",
)


@contextlib.contextmanager
def _forbid_cuda_access():
    """Guard the scoped AOT work after unchanged package/staging import setup."""
    import torch

    with contextlib.ExitStack() as stack:
        guards = {
            name: stack.enter_context(
                patch.object(
                    torch.cuda,
                    name,
                    side_effect=AssertionError(f"Unexpected GPU access: {name}"),
                )
            )
            for name in (
                "_lazy_init",
                "init",
                "is_available",
                "current_device",
                "get_device_properties",
                "device",
                "device_count",
                "set_device",
                "current_stream",
                "default_stream",
                "synchronize",
            )
        }
        yield guards
        for guard in guards.values():
            guard.assert_not_called()


def _assert_lightweight_imports():
    if os.environ.get("AITER_AOT_IMPORT") != "1":
        raise AssertionError("The compile worker must use AITER_AOT_IMPORT=1")
    # FlyDSL's package imports the pure-Python implementation registry; unlike
    # the wrapper and enum op module, it does not load runtime MoE/C++ ops.
    loaded = [name for name in _RUNTIME_MOE_MODULES if name in sys.modules]
    if loaded:
        raise AssertionError(f"Lightweight AOT imported runtime MoE modules: {loaded}")


def _whole_graph_row(**updates):
    """Build a real CSV row, not a replacement for the production parser."""
    row = {
        "gfx": "gfx942",
        "cu_num": "80",
        "token": "64",
        "model_dim": "512",
        "inter_dim": "128",
        "expert": "4",
        "topk": "2",
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float8_e4m3fnuz",
        "q_dtype_w": "torch.float8_e4m3fnuz",
        "q_type": "QuantType.per_Token",
        "use_g1u1": "1",
        "doweight_stage1": "0",
        "block_m": "64",
        "ksplit": "0",
        "kernelName1": _PREFIX + _PREFILL,
        "kernelName2": "",
        "us": "1.0",
        "run_1stage": "0",
        "xbf16": "0",
        "flat": "0",
    }
    row.update({key: str(value) for key, value in updates.items()})
    return row


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _lightweight_aot_probe(phase):
    """Fresh-process regression entry; no imports here run at test collection."""
    import multiprocessing

    import torch

    # Existing optional Triton/staged-MoE imports may initialize the driver.
    # Do not hide that inherited behavior or import the runtime MoE wrapper.
    from aiter.aot.flydsl import moe
    from aiter.aot.flydsl.common import collect_aot_jobs, run_jobs_parallel

    _assert_lightweight_imports()
    evidence = {
        "phase": phase,
        "cuda_initialized_after_import_setup": torch.cuda._initialized,
    }
    with _forbid_cuda_access() as guards:
        if phase == "parse":
            path = os.environ["AITER_CONFIG_FMOE"]
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            expected = {
                (
                    row["kernelName1"],
                    *(
                        int(row[key])
                        for key in (
                            "token",
                            "model_dim",
                            "inter_dim",
                            "expert",
                            "topk",
                            "cu_num",
                        )
                    ),
                )
                for row in rows
                if row["kernelName1"].startswith(_PREFIX)
            }
            jobs = moe.parse_csv(path)
            actual = {
                (
                    job["kernel_name"],
                    *(
                        job[key]
                        for key in (
                            "token_num",
                            "model_dim",
                            "inter_dim",
                            "experts",
                            "topk",
                            "cu_num",
                        )
                    ),
                )
                for job in jobs
                if job["stage"] == "whole_graph"
            }
            if not expected or actual != expected:
                raise AssertionError(
                    f"Published Qwen jobs changed: {actual ^ expected}"
                )
            if collect_aot_jobs([path, path], moe.parse_csv) != jobs:
                raise AssertionError("Collection lost or duplicated published jobs")
            evidence.update(parsed_jobs=len(jobs), whole_graph_jobs=len(actual))
        elif phase == "fork":
            root = Path(os.environ["FLYDSL_RUNTIME_CACHE_DIR"]).parent
            path = root / "fork_moe.csv"
            _write_csv(path, [_whole_graph_row(q_type="QuantType.per_Tensor")])
            jobs = moe.parse_csv(str(path))
            if len(jobs) != 1 or jobs[0]["stage"] != "whole_graph":
                raise AssertionError(f"Expected one real whole-graph job: {jobs}")
            parent_pid = os.getpid()

            def worker(**kwargs):
                before = dict(os.environ)
                result = moe.compile_one_config(**kwargs)
                _assert_lightweight_imports()
                return {
                    **result,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "start_method": multiprocessing.get_start_method(),
                    "cuda_in_bad_fork": torch.cuda._is_in_bad_fork(),
                    "cuda_calls": sum(guard.call_count for guard in guards.values()),
                    "environment_restored": dict(os.environ) == before,
                }

            results = run_jobs_parallel(worker, jobs)
            if len(results) != 1:
                raise AssertionError(f"Unexpected fork results: {results}")
            result = results[0]
            if (
                result["compile_time"] is None
                or result["compile_arch"] != "gfx942"
                or result["pid"] == parent_pid
                or result["ppid"] != parent_pid
                or result["start_method"] != "fork"
                or result["cuda_calls"]
                or not result["environment_restored"]
            ):
                raise AssertionError(f"Unsafe or unsuccessful fork compile: {result}")
            evidence.update(worker=result)
        elif phase == "integration":
            from op_tests.tuning_tests.test_flydsl_moe_run_only import (
                _compile_in_fresh_process,
            )

            root = Path(os.environ["FLYDSL_RUNTIME_CACHE_DIR"]).parent
            path = root / "integration_moe.csv"
            _write_csv(
                path,
                [_whole_graph_row(kernelName1=_PREFIX + _DECODE, block_m=16, token=1)],
            )
            _compile_in_fresh_process(str(path), None)
        else:
            raise ValueError(f"Unknown lightweight test phase: {phase}")

        _assert_lightweight_imports()
        evidence.update(
            cuda_calls=sum(guard.call_count for guard in guards.values()),
            runtime_modules_loaded=[
                name for name in _RUNTIME_MOE_MODULES if name in sys.modules
            ],
        )
        if phase != "parse":
            artifacts = list(
                Path(os.environ["FLYDSL_RUNTIME_CACHE_DIR"]).rglob("*.pkl")
            )
            if not artifacts:
                raise AssertionError("AOT succeeded without persistent artifacts")
            evidence["artifacts"] = len(artifacts)
    print(_LIGHTWEIGHT_MARKER + json.dumps(evidence), flush=True)


class TestFlydslMoeLightweightAot(unittest.TestCase):
    def _run_probe(self, phase):
        with tempfile.TemporaryDirectory(prefix="aiter_moe_lightweight_") as directory:
            env = dict(os.environ)
            for key in ("MOE_PREFILL_TILE_K", "HSA_OVERRIDE_GFX_VERSION"):
                env.pop(key, None)
            env.update(
                PYTHONPATH=str(_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
                PYTHONDONTWRITEBYTECODE="1",
                AITER_AOT_IMPORT="1",
                AITER_TRITON_ONLY="0",
                COMPILE_ONLY="1",
                ARCH="gfx942",
                FLYDSL_GPU_ARCH="gfx942",
                GPU_ARCHS="gfx942",
                CU_NUM="80",
                FLYDSL_RUNTIME_CACHE_DIR=str(Path(directory) / "flydsl_cache"),
                FLYDSL_RUNTIME_ENABLE_CACHE="1",
                FLYDSL_RUNTIME_RUN_ONLY="0",
                FLYDSL_DUMP_IR="0",
                AITER_JIT_DIR=str(Path(directory) / "aiter_jit"),
                TRITON_CACHE_DIR=str(Path(directory) / "triton_cache"),
                AITER_CONFIG_FMOE=str(
                    _ROOT
                    / "aiter/configs/model_configs/qwen3_5_35b_fp8_ptpc_tuned_fmoe.csv"
                ),
                AITER_CONFIG_FHMOE=str(_ROOT / "aiter/configs/tuned_fhmoe.csv"),
                AITER_FLYDSL_AOT_WORKERS="1",
                AITER_FLYDSL_AOT_MAX_RETRIES="0",
                AITER_FLYDSL_AOT_TIMEOUT="0",
                AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED="0",
                AITER_FLYDSL_MOE_BF16_RTE_SIMPLIFIED="0",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; from op_tests.tuning_tests.test_flydsl_moe_cache "
                        "import _lightweight_aot_probe; _lightweight_aot_probe(sys.argv[1])"
                    ),
                    phase,
                ],
                cwd=_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(
            result.returncode, 0, result.stdout[-8000:] + result.stderr[-8000:]
        )
        records = [
            line[len(_LIGHTWEIGHT_MARKER) :]
            for line in result.stdout.splitlines()
            if line.startswith(_LIGHTWEIGHT_MARKER)
        ]
        self.assertEqual(len(records), 1, result.stdout)
        evidence = json.loads(records[0])
        self.assertEqual(evidence["cuda_calls"], 0)
        self.assertEqual(evidence["runtime_modules_loaded"], [])
        print(_LIGHTWEIGHT_MARKER + records[0], flush=True)
        return evidence

    def test_published_qwen_jobs_collect_in_lightweight_process(self):
        evidence = self._run_probe("parse")
        self.assertGreater(evidence["whole_graph_jobs"], 0)

    def test_production_fork_preloads_stages_and_auxiliaries_without_cuda(self):
        evidence = self._run_probe("fork")
        self.assertGreater(evidence["artifacts"], 0)
        self.assertEqual(evidence["worker"]["cuda_calls"], 0)

    def test_integration_compile_worker_uses_lightweight_imports(self):
        evidence = self._run_probe("integration")
        self.assertGreater(evidence["artifacts"], 0)


class _CpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep collection free of aiter imports and their extension loading.
        import torch

        cls.torch = torch
        cls.backend = importlib.import_module("aiter.ops.flydsl.fused_moe_gfx942")
        cls.host = importlib.import_module("aiter.ops.flydsl.moe_gemm_2stage")
        cls.aot = importlib.import_module("aiter.aot.flydsl.moe")
        cls.aot_common = importlib.import_module("aiter.aot.flydsl.common")
        cls.kernels = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage"
        )
        cls.common = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage.common"
        )
        cls.gemm1 = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage.gemm1"
        )
        cls.gemm2 = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage.gemm2"
        )
        cls.reduce = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage.moe_reduce"
        )
        cls.quant = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage.quant"
        )
        cls.shim = importlib.import_module("aiter.ops.flydsl.kernels.tensor_shim")

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ))
        for key in (
            "ARCH",
            "FLYDSL_GPU_ARCH",
            "CU_NUM",
            "MOE_PREFILL_TILE_K",
            "COMPILE_ONLY",
            "FLYDSL_RUNTIME_RUN_ONLY",
        ):
            os.environ.pop(key, None)
        self.stack.enter_context(
            patch.object(self.torch.cuda, "is_available", return_value=False)
        )
        for name in ("_lazy_init", "current_device", "get_device_properties", "device"):
            self.stack.enter_context(
                patch.object(
                    self.torch.cuda,
                    name,
                    side_effect=AssertionError(f"Unexpected GPU access: {name}"),
                )
            )
        self.stack.enter_context(
            patch.object(
                self.shim.flyc,
                "compile",
                side_effect=AssertionError("CPU tests must not compile GPU kernels"),
            )
        )
        self._clear_caches()
        self.addCleanup(self._clear_caches)

    def _clear_caches(self):
        self.common._get_device_cache_key.cache_clear()
        self.host._get_compiled_kernel.cache_clear()
        self.kernels.compile_gemm.cache_clear()
        for factory in (
            self.reduce.invert_sorted_ids,
            self.reduce.sorted_sum,
            self.quant.flydsl_absmax,
            self.quant.flydsl_quant_per_tensor,
        ):
            factory.cache_clear()

    def _parse(self, *rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moe.csv"
            _write_csv(path, rows)
            with contextlib.redirect_stdout(io.StringIO()):
                return self.aot.parse_csv(str(path))

    def _job(self, **updates):
        jobs = self._parse(_whole_graph_row(**updates))
        self.assertEqual(len(jobs), 1, jobs)
        return jobs[0]

    @contextlib.contextmanager
    def _devices(self, *, ambient=0, arch="gfx942"):
        """Model CUDA's context restoration without initializing a GPU."""
        state = {"current": ambient, "entered": [], "restored": []}

        def index(device):
            if device is None:
                return state["current"]
            if isinstance(device, int):
                return device
            parsed = self.torch.device(device)
            return state["current"] if parsed.index is None else parsed.index

        @contextlib.contextmanager
        def device_context(device):
            previous = state["current"]
            state["current"] = index(device)
            state["entered"].append(state["current"])
            try:
                yield
            finally:
                state["current"] = previous
                state["restored"].append(previous)

        def properties(device=None):
            self.assertIn(index(device), (0, 1))
            return SimpleNamespace(
                name="test AMD GPU",
                gcnArchName=arch + ":sramecc+:xnack-",
                multi_processor_count=80,
            )

        with (
            patch.object(self.torch.cuda, "is_available", return_value=True),
            patch.object(
                self.torch.cuda, "current_device", side_effect=lambda: state["current"]
            ),
            patch.object(self.torch.cuda, "device", side_effect=device_context),
            patch.object(
                self.torch.cuda, "get_device_properties", side_effect=properties
            ),
        ):
            yield state


class TestFlydslMoeAotParser(_CpuTest):
    def test_parser_accepts_baseline_fp8_quantization_and_activations(self):
        configs = (
            (_DECODE, 1),
            (_DECODE, 2),
            (_DECODE, 16),
            (_PREFILL, 64),
            ("64_128_256_True", 64),
            ("64_256_128_True", 64),
        )
        for (config, batch), quant, act in itertools.product(
            configs, ("per_Token", "per_Tensor"), ("Silu", "Swiglu")
        ):
            with self.subTest(config=config, batch=batch, quant=quant, act=act):
                job = self._job(
                    kernelName1=_PREFIX + config,
                    token=batch,
                    block_m=config.split("_")[0],
                    q_type="QuantType." + quant,
                    act_type="ActivationType." + act,
                )
                self.assertEqual(job["stage"], "whole_graph")
                self.assertEqual(job["config_string"], config)
                self.assertEqual(job["token_num"], batch)
                self.assertEqual(job["weight_dtype"], "fp8")
                self.assertEqual(
                    job["quant_type"], "ptpc" if quant == "per_Token" else "per_tensor"
                )
                self.assertEqual(job["act"], act.lower())

    def test_parser_rejects_new_architectures_weight_formats_and_paths(self):
        unsupported = (
            {"gfx": "gfx950", "cu_num": 256},
            {"gfx": "gfx1250"},
            {"cu_num": 256},
            {"kernelName1": "impl__flydsl_gfx950__" + _DECODE},
            {"q_dtype_w": "torch.bfloat16", "q_type": "QuantType.No"},
            {"q_dtype_w": "torch.float8_e4m3fn"},
            {"q_dtype_w": "torch.float4_e2m1fn_x2", "q_type": "QuantType.per_1x32"},
            {"q_type": "QuantType.per_1x32"},
            {"q_type": "QuantType.No"},
            {"act_type": "ActivationType.Situv2"},
            {"act_type": "ActivationType.Gelu"},
            {"dtype": "torch.float16"},
            # Later optimization encodings must not become new opt1 entry points.
            {"kernelName1": _PREFIX + _DECODE + "_True"},
            {"kernelName1": _PREFIX + _PREFILL + ":1x4_64x256:64:128"},
            {"kernelName1": _PREFIX + _PREFILL + ":8x1_compact:64:128"},
        )
        for update in unsupported:
            with self.subTest(update=update):
                self.assertEqual(self._parse(_whole_graph_row(**update)), [])

    def test_parser_requires_paired_weights_and_runtime_activation_dtype(self):
        for q_type, config in itertools.product(
            ("QuantType.per_Token", "QuantType.per_Tensor"), (_DECODE, _PREFILL)
        ):
            base = _whole_graph_row(
                q_type=q_type,
                kernelName1=_PREFIX + config,
                block_m=config.split("_")[0],
                token=16,
            )
            for dtype in (
                "torch.bfloat16",
                "torch.float16",
                "torch.float8_e4m3fn",
                "torch.int8",
            ):
                with self.subTest(config=config, q_type=q_type, q_dtype_a=dtype):
                    self.assertEqual(self._parse({**base, "q_dtype_a": dtype}), [])
            self.assertEqual(self._parse({**base, "use_g1u1": "0"}), [])
            for optional in ("use_g1u1", "q_dtype_a"):
                legacy = dict(base)
                legacy.pop(optional)
                self.assertEqual(len(self._parse(legacy)), 1)
            self.assertEqual(len(self._parse({**base, "q_dtype_a": ""})), 1)

    def test_parser_validates_target_and_missing_or_nonpositive_cu(self):
        for gfx, cu_num in itertools.product(("", "gfx942"), (None, "", 0, 80, 304)):
            row = _whole_graph_row(gfx=gfx)
            if cu_num is None:
                row.pop("cu_num")
            else:
                row["cu_num"] = str(cu_num)
            with self.subTest(gfx=gfx, cu_num=cu_num):
                jobs = self._parse(row)
                self.assertEqual(len(jobs), 1)
                self.assertEqual(jobs[0]["cu_num"], int(cu_num or 0))
        for cu_num in (-1, -80, "nan", "80.5", 256):
            with self.subTest(cu_num=cu_num):
                self.assertEqual(self._parse(_whole_graph_row(cu_num=cu_num)), [])
        legacy = _whole_graph_row()
        legacy.pop("gfx")
        self.assertEqual(len(self._parse(legacy)), 1)

    def test_parser_rejects_incompatible_request_parameters(self):
        for update in (
            {"doweight_stage1": 1},
            {"shared_expert_id": 0},
            {"block_m": 16},
            {"ksplit": 1},
            {"ksplit": -1},
        ):
            with self.subTest(update=update):
                self.assertEqual(self._parse(_whole_graph_row(**update)), [])
        for block_m in (None, "", 0, 64):
            row = _whole_graph_row()
            if block_m is None:
                row.pop("block_m")
            else:
                row["block_m"] = str(block_m)
            self.assertEqual(len(self._parse(row)), 1)

    def test_parser_rejects_missing_or_nonpositive_dimensions(self):
        for field, value in itertools.product(
            ("token", "model_dim", "inter_dim", "expert", "topk"), (None, "", 0, -1)
        ):
            row = _whole_graph_row()
            if value is None:
                row.pop(field)
            else:
                row[field] = str(value)
            with self.subTest(field=field, value=value):
                self.assertEqual(self._parse(row), [])

    def test_parser_rejects_incomplete_tiles_and_lds_overflow(self):
        cases = (
            (_DECODE, 1, 384, 128),  # Incomplete four-wave Gate/Up K tile.
            (_DECODE, 16, 384, 128),
            (_DECODE, 16, 512, 96),  # Incomplete Down K tile.
            (_DECODE, 257, 512, 128),
            ("32_16_16_False", 16, 512, 128),
            ("16_128_128_True", 64, 512, 128),
            ("64_64_128_True", 64, 512, 128),
            ("64_128_64_True", 64, 512, 128),
            ("64_256_128_True", 64, 512, 192),  # Incomplete Gate/Up N tile.
            (_PREFILL, 64, 640, 128),  # Odd Gate/Up pipeline tile count.
            ("64_128_256_True", 64, 768, 128),
            ("64_128_256_True", 64, 256, 128),
            ("256_128_256_True", 64, 512, 128),  # Gate/Up LDS > 64 KiB.
            (_PREFILL, 64, 512, 1088),  # Down LDS > 64 KiB.
            ("32_128_128_True", 64, 512, 64),  # Incomplete cooperative copy.
            ("0_128_128_True", 64, 512, 128),
            ("64_-128_128_True", 64, 512, 128),
            ("64_128_0_True", 64, 512, 128),
            ("64_128_128_true", 64, 512, 128),
        )
        for config, batch, model_dim, inter_dim in cases:
            with self.subTest(config=config, batch=batch, shape=(model_dim, inter_dim)):
                self.assertEqual(
                    self._parse(
                        _whole_graph_row(
                            kernelName1=_PREFIX + config,
                            token=batch,
                            model_dim=model_dim,
                            inter_dim=inter_dim,
                            block_m=0,
                        )
                    ),
                    [],
                )

    def test_sorted_sum_preserves_the_baseline_256_column_alignment(self):
        # The 32-thread baseline reduction handles odd multiples of 256.
        self.assertEqual(len(self._parse(_whole_graph_row(model_dim=768))), 1)
        problem = self.backend._Problem(64, 4, 256, 512, 768, 128, 2, "ptpc")
        reason = self.backend.Config.from_string(_PREFILL).unsupported_reason(problem)
        self.assertIsNone(reason)
        for model_dim in (128, 384, 640):
            with self.subTest(model_dim=model_dim):
                self.assertIsNotNone(
                    self.backend.Config.from_string(_PREFILL).unsupported_reason(
                        replace(problem, model_dim=model_dim)
                    )
                )

    def test_parser_accepts_exact_tile_and_lds_boundaries(self):
        for config, batch, model_dim, inter_dim in (
            (_DECODE, 1, 256, 64),
            (_DECODE, 256, 512, 128),
            ("32_128_128_True", 64, 512, 128),
            ("128_128_256_True", 64, 512, 512),
            (_PREFILL, 64, 512, 1024),
            (_PREFILL, 64, 768, 128),
        ):
            with self.subTest(config=config, shape=(model_dim, inter_dim)):
                self._job(
                    kernelName1=_PREFIX + config,
                    token=batch,
                    model_dim=model_dim,
                    inter_dim=inter_dim,
                    block_m=config.split("_")[0],
                )

    def test_stage_specific_parser_semantics_are_unchanged(self):
        registry = importlib.import_module("aiter.ops.flydsl.moe_kernels")
        names = (
            "flydsl_moe1_afp8_wfp8_bf16_t32x128x256_gui",
            "flydsl_moe2_afp8_wfp8_bf16_t32x128x128_reduce",
        )
        row = _whole_graph_row(
            gfx="gfx950",
            cu_num=256,
            dtype="torch.float16",
            q_dtype_a="torch.float8_e4m3fn",
            q_dtype_w="torch.float8_e4m3fn",
            act_type="ActivationType.Situv2",
            use_g1u1=0,
            doweight_stage1=1,
            block_m=32,
            ksplit=2,
            kernelName1=names[0],
            kernelName2=names[1],
        )
        jobs = self._parse(row)
        self.assertEqual([job["stage"] for job in jobs], [1, 2])
        for name, job in zip(names, jobs):
            params = registry.get_flydsl_kernel_params(name)
            self.assertIsNotNone(params)
            for key, value in params.items():
                self.assertEqual(job[key], value)
            self.assertEqual(job["act"], "situv2")
            self.assertEqual(job["cu_num"], 256)
            self.assertEqual(job["block_m"], 32)
            self.assertTrue(job["doweight_stage1"])
        rejected_whole = _whole_graph_row(use_g1u1=0)
        self.assertEqual(self._parse(rejected_whole, row, row), jobs)

    def test_whole_graph_jobs_deduplicate_without_losing_specializations(self):
        base = _whole_graph_row()
        variants = [
            base,
            {**base, "us": "2.0"},
            {**base, "q_dtype_a": ""},
            {**base, "token": "128"},
            {**base, "act_type": "ActivationType.Swiglu"},
            {**base, "q_type": "QuantType.per_Tensor"},
            {**base, "cu_num": "304"},
        ]
        jobs = self._parse(*variants)
        self.assertEqual(len(jobs), 5)
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                str(Path(directory) / name) for name in ("first.csv", "second.csv")
            ]
            for path in paths:
                _write_csv(path, variants)
            collected = self.aot_common.collect_aot_jobs(paths, self.aot.parse_csv)
        self.assertEqual(collected, jobs)


class TestFlydslMoePrecompile(_CpuTest):
    def test_worker_restores_environment_and_omits_unspecified_cu_override(self):
        for cu_num, initial in itertools.product(
            (None, 0, 80, 304),
            (
                {},
                {
                    "ARCH": "gfx950",
                    "FLYDSL_GPU_ARCH": "gfx950",
                    "CU_NUM": "17",
                    "COMPILE_ONLY": "0",
                },
            ),
        ):
            job = self._job()
            if cu_num is None:
                job.pop("cu_num")
            else:
                job["cu_num"] = cu_num

            def precompile(_cu_num=cu_num, **kwargs):
                self.assertEqual(os.environ["COMPILE_ONLY"], "1")
                self.assertEqual(os.environ["ARCH"], "gfx942")
                self.assertEqual(os.environ["FLYDSL_GPU_ARCH"], "gfx942")
                self.assertEqual(
                    os.environ.get("CU_NUM"), str(_cu_num) if _cu_num else None
                )
                self.assertEqual(kwargs["config_string"], _PREFILL)
                self.assertEqual(kwargs["batch"], 64)

            with self.subTest(cu_num=cu_num, initial=initial), patch.dict(os.environ):
                for key in ("ARCH", "FLYDSL_GPU_ARCH", "CU_NUM", "COMPILE_ONLY"):
                    os.environ.pop(key, None)
                os.environ.update(initial)
                before = dict(os.environ)
                with patch.object(
                    self.host, "precompile_flydsl_moe", side_effect=precompile
                ) as preload:
                    result = self.aot.compile_one_config(**job)
                preload.assert_called_once()
                self.assertIsNotNone(result["compile_time"], result)
                self.assertEqual(result["compile_arch"], "gfx942")
                self.assertEqual(dict(os.environ), before)

    def test_worker_reports_failure_and_restores_environment(self):
        before = dict(os.environ)
        with (
            patch.object(
                self.host,
                "precompile_flydsl_moe",
                side_effect=RuntimeError("preload failed"),
            ),
            contextlib.redirect_stdout(io.StringIO()) as messages,
        ):
            result = self.aot.compile_one_config(**self._job())
        self.assertIsNone(result["compile_time"])
        self.assertIn("preload failed", messages.getvalue())
        self.assertEqual(dict(os.environ), before)

    def test_worker_rejects_conflicting_target_or_config_identity(self):
        for update in (
            {"cu_num": -1},
            {"cu_num": 256},
            {"gfx": "gfx950"},
            {"kernel_name": "impl__flydsl_gfx950__" + _PREFILL},
            {"config_string": "64_128_256_True"},
        ):
            with (
                self.subTest(update=update),
                patch.object(self.host, "precompile_flydsl_moe") as preload,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = self.aot.compile_one_config(**{**self._job(), **update})
                self.assertIsNone(result["compile_time"], result)
                preload.assert_not_called()

    def test_swiglu_default_and_explicit_limits_are_finite_specializations(self):
        job = self._job(act_type="ActivationType.Swiglu")
        self.assertNotIn("swiglu_limit", job)
        for limit, expected in ((None, 7.0), (0.0, 7.0), (7.0, 7.0), (1.5, 1.5)):
            with self.subTest(limit=limit):
                _, resolved = self.common.validate_gemm_options(
                    "fp8", "ptpc", "ptpc", 64, "prefill_1x4", "swiglu", limit
                )
                self.assertTrue(math.isfinite(resolved))
                self.assertEqual(resolved, expected)
                with patch.object(self.host, "precompile_flydsl_moe") as preload:
                    result = self.aot.compile_one_config(**job, swiglu_limit=limit)
                self.assertIsNotNone(result["compile_time"])
                self.assertEqual(preload.call_args.kwargs["swiglu_limit"], limit)
        # This tests finite legacy defaults, not support for every possible
        # runtime limit. A cold miss for an uncompiled limit is tested on GPU.

    def test_precompile_rejects_unsupported_inputs_before_building_launchers(self):
        base = {
            "config_string": _PREFILL,
            "batch": 64,
            "model_dim": 512,
            "inter_dim": 128,
            "experts": 4,
            "topk": 2,
            "weight_dtype": "fp8",
            "quant_type": "ptpc",
            "activation": "silu",
        }
        for update in (
            {"weight_dtype": "bf16"},
            {"weight_dtype": "fp4", "quant_type": "mxfp4"},
            {"quant_type": "no"},
            {"activation": "situv2"},
            {"model_dim": 640},
            {"inter_dim": 1088},
        ):
            with (
                self.subTest(update=update),
                patch.dict(os.environ, {"ARCH": "gfx942"}),
                patch.object(self.host, "_get_compiled_kernel") as build,
                self.assertRaises(ValueError),
            ):
                self.host.precompile_flydsl_moe(**{**base, **update})
            build.assert_not_called()
        for arch in ("gfx950", "gfx1250", ""):
            with (
                self.subTest(arch=arch),
                patch.dict(os.environ, {"ARCH": arch, "FLYDSL_GPU_ARCH": arch}),
                self.assertRaisesRegex(ValueError, "gfx942"),
            ):
                self.host.precompile_flydsl_moe(**base)

    def test_compile_only_requires_an_explicit_target_without_querying_cuda(self):
        for target in ({}, {"ARCH": "gfx950"}, {"FLYDSL_GPU_ARCH": "gfx1250"}):
            with (
                self.subTest(target=target),
                patch.dict(os.environ, {"COMPILE_ONLY": "yes", **target}),
                _forbid_cuda_access(),
                patch.object(self.host, "_get_compiled_kernel") as build,
                self.assertRaisesRegex(ValueError, "gfx942 compile target"),
            ):
                self.host.precompile_flydsl_moe(
                    config_string=_DECODE,
                    batch=1,
                    model_dim=256,
                    inter_dim=64,
                    experts=2,
                    topk=1,
                    weight_dtype="fp8",
                    quant_type="ptpc",
                    activation="silu",
                    device="cuda:1",
                )
            build.assert_not_called()

    def test_aot_preloads_stages_and_all_required_auxiliaries_without_launch(self):
        for config, batch, quant_type in (
            (_DECODE, 1, "ptpc"),
            (_DECODE, 16, "per_tensor"),
            (_PREFILL, 64, "ptpc"),
            (_PREFILL, 64, "per_tensor"),
        ):
            self._clear_caches()
            recorded = []

            def preload(launcher, *args, _recorded=recorded):
                inspect.signature(launcher.func).bind(*args)
                _recorded.append(launcher.func.__qualname__)

            with (
                self.subTest(config=config, quant_type=quant_type),
                contextlib.ExitStack() as stack,
            ):
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        {
                            "ARCH": "gfx942",
                            "FLYDSL_GPU_ARCH": "gfx942",
                            "COMPILE_ONLY": "1",
                        },
                    )
                )
                stack.enter_context(_forbid_cuda_access())
                for module in (self.host, self.reduce, self.quant):
                    stack.enter_context(
                        patch.object(module, "_preload_compiled", side_effect=preload)
                    )
                for module in (self.backend, self.reduce, self.quant):
                    stack.enter_context(
                        patch.object(
                            module,
                            "_run_compiled",
                            side_effect=AssertionError("AOT launched a kernel"),
                        )
                    )
                stack.enter_context(
                    patch.object(
                        self.backend,
                        "moe_sorting",
                        side_effect=AssertionError("AOT called HIP sorting"),
                    )
                )
                stack.enter_context(
                    patch.object(
                        self.backend.aiter,
                        "get_hip_quant",
                        side_effect=AssertionError("AOT called HIP quantization"),
                    )
                )
                self.host.precompile_flydsl_moe(
                    config_string=config,
                    batch=batch,
                    model_dim=512,
                    inter_dim=128,
                    experts=4,
                    topk=2,
                    weight_dtype="fp8",
                    quant_type=quant_type,
                    activation="swiglu",
                    swiglu_limit=1.5,
                )
            expected = ["_build_moe_gemm1", "_build_moe_gemm2"]
            if config == _PREFILL:
                expected += ["_invert_sorted_ids_cached", "_sorted_sum_cached"]
                if quant_type == "per_tensor":
                    expected += [
                        "_flydsl_absmax_cached",
                        "_flydsl_quant_per_tensor_cached",
                    ]
            self.assertEqual([name.split(".")[0] for name in recorded], expected)

    def test_preload_prefers_the_no_dispatch_api_even_with_a_hot_callable(self):
        value, args = object(), (object(), object())
        launcher = SimpleNamespace(
            preload=Mock(return_value=value),
            _cf=Mock(side_effect=AssertionError("preload dispatched the hot callable")),
        )
        before = dict(os.environ)
        self.assertIs(self.shim._preload_compiled(launcher, *args), value)
        launcher.preload.assert_called_once_with(*args)
        launcher._cf.assert_not_called()
        self.shim.flyc.compile.assert_not_called()
        self.assertEqual(dict(os.environ), before)

    def test_preload_fallback_sets_compile_only_and_restores_it_on_failure(self):
        for old, fail in itertools.product((None, "0", "1"), (False, True)):
            with self.subTest(old=old, fail=fail), patch.dict(os.environ):
                if old is None:
                    os.environ.pop("COMPILE_ONLY", None)
                else:
                    os.environ["COMPILE_ONLY"] = old
                launcher = SimpleNamespace(_cf=Mock())
                value, argument = object(), object()

                def compile_only(
                    actual,
                    *args,
                    _launcher=launcher,
                    _argument=argument,
                    _fail=fail,
                    _value=value,
                ):
                    self.assertIs(actual, _launcher)
                    self.assertEqual(args, (_argument,))
                    self.assertEqual(os.environ["COMPILE_ONLY"], "1")
                    if _fail:
                        raise RuntimeError("compile failed")
                    return _value

                with patch.object(self.shim.flyc, "compile", side_effect=compile_only):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "compile failed"):
                            self.shim._preload_compiled(launcher, argument)
                    else:
                        self.assertIs(
                            self.shim._preload_compiled(launcher, argument), value
                        )
                launcher._cf.assert_not_called()
                self.assertEqual(os.environ.get("COMPILE_ONLY"), old)


class TestFlydslMoeHostCaches(_CpuTest):
    def _factories(self):
        options = {
            "N": 256,
            "K": 512,
            "weight_dtype": "fp8",
            "weight_quant_type": "ptpc",
            "TOPK": 2,
            "BLOCK_TILE_SIZE_M": 64,
            "BLOCK_TILE_SIZE_N": 128,
            "alg": "prefill_1x4",
            "E": 4,
            "act_quant_type": "ptpc",
        }

        def host_factory(module, device):
            return module._get_compiled_kernel(
                N=256,
                K=512,
                weight_dtype_str="fp8",
                quant_type_str="ptpc",
                TOPK=2,
                BLOCK_TILE_SIZE_M=64,
                BLOCK_TILE_SIZE_N=128,
                stage="gateup",
                alg="prefill_1x4",
                E=4,
                act_quant_type_str="ptpc",
                device=device,
            )

        return {
            "wrapper": lambda device: host_factory(self.backend, device),
            "host": lambda device: host_factory(self.host, device),
            "dispatch": lambda device: self.kernels.compile_gemm(
                **options, device=device
            ),
            "gateup": lambda device: self.gemm1.compile_moe_gemm1(
                **options, device=device
            ),
            "down": lambda device: self.gemm2.compile_moe_gemm2(
                **options, device=device
            ),
            "invert": lambda device: self.reduce.invert_sorted_ids(2, device=device),
            "sum": lambda device: self.reduce.sorted_sum(2, 512, device=device),
            "absmax": lambda device: self.quant.flydsl_absmax(device=device),
            "quant": lambda device: self.quant.flydsl_quant_per_tensor(
                self.torch.float8_e4m3fnuz, device=device
            ),
        }

    def test_compile_only_cache_key_needs_no_live_gpu(self):
        with patch.dict(os.environ, {"COMPILE_ONLY": "1"}), _forbid_cuda_access():
            unspecified = self.common.get_device_cache_key()
            with patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx942", "CU_NUM": "80"}):
                first = self.common.get_device_cache_key()
                self.assertNotEqual(first, unspecified)
                self.assertEqual(first, self.common.get_device_cache_key())
                for name, changed in (
                    ("ARCH", "gfx950"),
                    ("FLYDSL_GPU_ARCH", "gfx950"),
                    ("CU_NUM", "304"),
                ):
                    with (
                        self.subTest(name=name),
                        patch.dict(os.environ, {name: changed}),
                    ):
                        self.assertNotEqual(first, self.common.get_device_cache_key())
            self.assertEqual(self.common.get_device_cache_key(), unspecified)

    def test_compile_only_truth_values_skip_every_cuda_query_and_context(self):
        from flydsl.utils import env as flydsl_env

        for value in ("1", "true", "TRUE", "TrUe", "yes", "YES", "on", "ON"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {"COMPILE_ONLY": value, "ARCH": "gfx942", "CU_NUM": "80"},
                ),
                _forbid_cuda_access(),
            ):
                self.assertTrue(flydsl_env.compile.compile_only)
                first = self.common.get_device_cache_key()
                for device in (
                    None,
                    0,
                    1,
                    "cuda",
                    "cuda:1",
                    self.torch.device("cuda:1"),
                ):
                    with self.subTest(device=device):
                        self.assertEqual(
                            first, self.common.get_device_cache_key(device)
                        )
                        with self.common.device_context(device):
                            pass

    def test_false_compile_only_and_run_only_keep_physical_device_identity(self):
        from flydsl.utils import env as flydsl_env

        with self._devices(ambient=0) as state:
            first = self.common.get_device_cache_key(1)
            for value in ("0", "false", "FALSE", "no", "off", "", " true "):
                with (
                    self.subTest(value=value),
                    patch.dict(
                        os.environ,
                        {"COMPILE_ONLY": value, "FLYDSL_RUNTIME_RUN_ONLY": "1"},
                    ),
                ):
                    self.assertFalse(flydsl_env.compile.compile_only)
                    self.assertEqual(first, self.common.get_device_cache_key("cuda:1"))
                    self.assertNotEqual(first, self.common.get_device_cache_key())
                    with self.common.device_context("cuda:1"):
                        self.assertEqual(state["current"], 1)
                    self.assertEqual(state["current"], 0)

    def test_compile_only_and_runtime_host_caches_cannot_alias(self):
        for name, factory in self._factories().items():
            self._clear_caches()
            with (
                self.subTest(factory=name),
                patch.dict(os.environ, {"ARCH": "gfx942", "CU_NUM": "80"}),
            ):
                runtime_key = self.common.get_device_cache_key()
                runtime = factory(None)
                with (
                    patch.dict(os.environ, {"COMPILE_ONLY": "on"}),
                    _forbid_cuda_access(),
                ):
                    self.assertNotEqual(runtime_key, self.common.get_device_cache_key())
                    compiled = factory("cuda:1")
                    self.assertIsNot(compiled, runtime)
                    self.assertIs(compiled, factory(None))
                    self.assertIs(compiled, factory("cuda:0"))
                self.assertIs(factory(None), runtime)

    def test_wrapper_reexports_the_shared_apis_and_cache_counters(self):
        for name in (
            "Config",
            "_Problem",
            "_get_compiled_kernel_cached",
            "_get_compiled_kernel",
            "precompile_flydsl_moe",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(self.backend, name), getattr(self.host, name))
        self.assertIs(
            self.backend._get_compiled_kernel.cache_clear.__self__,
            self.host._get_compiled_kernel_cached,
        )
        self.assertIs(
            self.backend._get_compiled_kernel.cache_info.__self__,
            self.host._get_compiled_kernel_cached,
        )

    def test_device_spellings_normalize_and_explicit_device_beats_ambient(self):
        with self._devices(ambient=0):
            first = self.common.get_device_cache_key(1)
            for spelling in ("cuda:1", self.torch.device("cuda:1")):
                self.assertEqual(first, self.common.get_device_cache_key(spelling))
            self.assertNotEqual(first, self.common.get_device_cache_key())
            self.assertEqual(
                self.common.get_device_cache_key("cuda"),
                self.common.get_device_cache_key(0),
            )

    def test_all_host_caches_separate_devices_targets_and_cu_counts(self):
        for name, factory in self._factories().items():
            self._clear_caches()
            with (
                self.subTest(factory=name),
                self._devices(ambient=0) as state,
                patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx942", "CU_NUM": "80"}),
            ):
                first = factory(0)
                second = factory(self.torch.device("cuda:1"))
                self.assertIsNot(first, second)
                self.assertIs(factory(0), first)
                self.assertEqual(state["current"], 0)
                os.environ["FLYDSL_GPU_ARCH"] = "gfx950"
                self.assertIsNot(factory(0), first)
                os.environ["FLYDSL_GPU_ARCH"] = "gfx942"
                os.environ["CU_NUM"] = "304"
                self.assertIsNot(factory(0), first)
                os.environ["CU_NUM"] = "80"
                self.assertIs(factory(0), first)
                self.assertEqual(state["current"], 0)

    def test_arch_only_override_cannot_reuse_another_compile_target(self):
        # ARCH is a supported AOT/FlyDSL target override independently of
        # FLYDSL_GPU_ARCH. An outer host-cache hit must not hide its change.
        for name, factory in self._factories().items():
            self._clear_caches()
            with self.subTest(factory=name), patch.dict(os.environ, {"ARCH": "gfx942"}):
                first = factory(None)
                os.environ["ARCH"] = "gfx950"
                self.assertIsNot(factory(None), first)
                os.environ["ARCH"] = "gfx942"
                self.assertIs(factory(None), first)

    def test_tile_k_environment_is_resolved_before_every_gemm_cache(self):
        for name, factory in self._factories().items():
            if name not in ("wrapper", "host", "dispatch", "gateup", "down"):
                continue
            self._clear_caches()
            with (
                self.subTest(factory=name),
                patch.dict(os.environ, {"MOE_PREFILL_TILE_K": "128"}),
            ):
                first = factory(None)
                os.environ["MOE_PREFILL_TILE_K"] = "256"
                self.assertIsNot(factory(None), first)
                os.environ["MOE_PREFILL_TILE_K"] = "128"
                self.assertIs(factory(None), first)
                self.assertEqual(self.common.resolve_tile_k(256), 256)

    def test_swiglu_limit_changes_gateup_specialization(self):
        options = {
            "N": 256,
            "K": 512,
            "weight_dtype_str": "fp8",
            "quant_type_str": "ptpc",
            "TOPK": 2,
            "BLOCK_TILE_SIZE_M": 16,
            "BLOCK_TILE_SIZE_N": 32,
            "stage": "gateup",
            "alg": "batch1",
            "E": None,
            "activation_str": "swiglu",
        }
        default = self.backend._get_compiled_kernel(**options)
        explicit = self.backend._get_compiled_kernel(**options, swiglu_limit=1.5)
        self.assertIsNot(default, explicit)
        self.assertIs(self.backend._get_compiled_kernel(**options), default)
        self.assertIs(
            self.backend._get_compiled_kernel(**options, swiglu_limit=1.5), explicit
        )

    def test_same_device_context_and_warm_factories_skip_device_guards(self):
        for ambient in (0, 1):
            with self.subTest(ambient=ambient), self._devices(ambient=ambient) as state:
                devices = (
                    None,
                    ambient,
                    f"cuda:{ambient}",
                    self.torch.device("cuda", ambient),
                )
                for device in devices:
                    with self.common.device_context(device):
                        self.assertEqual(state["current"], ambient)
                for name, factory in self._factories().items():
                    with self.subTest(factory=name):
                        first = factory(ambient)
                        for device in devices:
                            self.assertIs(factory(device), first)
                self.assertEqual(state["entered"], [])
                self.assertEqual(state["restored"], [])
                self.torch.cuda.device.assert_not_called()

    def test_host_cache_only_guards_misses_on_the_requested_physical_device(self):
        factories = self._factories()
        for name in ("host", "wrapper"):
            self._clear_caches()
            with self.subTest(factory=name), self._devices(ambient=0) as state:
                compile_gemm = self.kernels.compile_gemm

                def build(_compile_gemm=compile_gemm, **kwargs):
                    self.assertEqual(state["current"], 1)
                    return _compile_gemm(**kwargs)

                with patch.object(
                    self.kernels, "compile_gemm", side_effect=build
                ) as compile_stage:
                    first = factories[name](self.torch.device("cuda:1"))
                    self.assertEqual(state["current"], 0)
                    self.assertEqual(state["entered"], [1])
                    self.assertEqual(state["restored"], [0])
                    with patch.object(
                        self.host,
                        "device_context",
                        side_effect=AssertionError(
                            "warm host cache entered a device context"
                        ),
                    ):
                        for alias in ("host", "wrapper"):
                            self.assertIs(factories[alias]("cuda:1"), first)
                        self.assertEqual(state["entered"], [1])
                        with self.torch.cuda.device(1):
                            self.assertIs(factories[name](None), first)
                    compile_stage.assert_called_once()
                self.assertEqual(state["current"], 0)
                self.assertEqual(state["entered"], [1, 1])
                self.assertEqual(state["restored"], [0, 0])
                info = self.host._get_compiled_kernel.cache_info()
                self.assertEqual((info.hits, info.misses, info.currsize), (3, 1, 1))

    def test_non_cuda_device_types_cannot_hit_a_warm_host_cache(self):
        with self._devices():
            for name in ("host", "wrapper"):
                factory = self._factories()[name]
                first = factory(0)
                for device in ("cpu:0", "meta:0", self.torch.device("cpu:0")):
                    with (
                        self.subTest(factory=name, device=device),
                        self.assertRaises(ValueError),
                    ):
                        factory(device)
                self.assertIs(factory(0), first)

    def test_compile_failure_restores_the_callers_device(self):
        for name in ("host", "wrapper"):
            self._clear_caches()
            with self.subTest(factory=name), self._devices(ambient=0) as state:

                def fail_build(*_args, **_kwargs):
                    self.assertEqual(state["current"], 1)
                    raise RuntimeError("build failed")

                with patch.object(
                    self.gemm1, "_build_moe_gemm1", side_effect=fail_build
                ):
                    for _ in range(2):
                        with self.assertRaisesRegex(RuntimeError, "build failed"):
                            self._factories()[name](self.torch.device("cuda:1"))
                        self.assertEqual(state["current"], 0)
                self.assertEqual(state["entered"], [1, 1])
                self.assertEqual(state["restored"], [0, 0])
                info = self.host._get_compiled_kernel.cache_info()
                self.assertEqual((info.hits, info.misses, info.currsize), (0, 2, 0))

    def test_legacy_imports_are_the_same_objects(self):
        legacy = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942"
        )
        for name in self.kernels.__all__:
            with self.subTest(symbol=name):
                self.assertIs(getattr(legacy, name), getattr(self.kernels, name))
        self.assertIs(legacy._ptr, self.common.torch_tensor_to_pointer)
        for name in (
            "_f32_to_bf16",
            "_f32_to_bf16_rta",
            "_f32_to_bf16_rte",
            "_TORCH_TO_FX",
        ):
            self.assertIs(getattr(legacy, name), getattr(self.common, name))
        utils = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage_utils"
        )
        old_utils = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942_utils"
        )
        for name in old_utils.__all__:
            with self.subTest(helper=name):
                self.assertIs(getattr(old_utils, name), getattr(utils, name))

    def test_tile_helpers_do_not_retain_region_bound_ir(self):
        utils = importlib.import_module(
            "aiter.ops.flydsl.kernels.moe_gemm_2stage_utils"
        )
        ops = utils.MoETileOps()
        first, second = object(), object()
        with (
            patch.object(utils.fx, "UniversalCopy", return_value=object()),
            patch.object(
                utils.fx, "make_copy_atom", side_effect=[first, second]
            ) as make,
        ):
            self.assertIs(ops.get_universal_copy_atom(utils.fx.BFloat16, 128), first)
            self.assertIs(ops.get_universal_copy_atom(utils.fx.BFloat16, 128), second)
        self.assertEqual(make.call_count, 2)
        self.assertEqual(vars(ops), {})


class TestFlydslMoeLaunchAbi(_CpuTest):
    def test_builders_reject_incomplete_tiles_before_compilation(self):
        base = {
            "N": 512,
            "K": 512,
            "weight_dtype": "fp8",
            "weight_quant_type": "ptpc",
            "TOPK": 2,
            "BLOCK_TILE_SIZE_M": 64,
            "BLOCK_TILE_SIZE_N": 128,
            "alg": "prefill_1x4",
        }
        cases = (
            (self.gemm1._build_moe_gemm1, {"K": 384, "tile_k": 128}),
            (self.gemm1._build_moe_gemm1, {"K": 640, "tile_k": 256}),
            (self.gemm1._build_moe_gemm1, {"tile_k": 64}),
            (
                self.gemm1._build_moe_gemm1,
                {"BLOCK_TILE_SIZE_M": 256, "tile_k": 256},
            ),
            (self.gemm1._build_moe_gemm1, {"alg": "splitk", "K": 384}),
            (self.gemm2._build_moe_gemm2, {"K": 96}),
            (self.gemm2._build_moe_gemm2, {"N": 384}),
            (self.gemm2._build_moe_gemm2, {"alg": "splitk", "K": 96}),
        )
        for builder, updates in cases:
            with (
                self.subTest(builder=builder.__name__, updates=updates),
                self.assertRaises(AssertionError),
            ):
                builder(**{**base, **updates})
        self.shim.flyc.compile.assert_not_called()

    def test_six_legacy_launch_signatures_match_runtime_and_aot_dispatch(self):
        for config_string, batch in ((_DECODE, 1), (_DECODE, 16), (_PREFILL, 64)):
            for quant_type in ("ptpc", "per_tensor"):
                with self.subTest(config=config_string, quant_type=quant_type):
                    self._check_case(config_string, batch, quant_type)

    def _check_case(self, config_string, batch, quant_type):
        torch = self.torch
        self._clear_caches()
        config = self.backend.Config.from_string(config_string)
        problem = self.backend._Problem(batch, 4, 256, 512, 512, 128, 2, quant_type)
        hidden = torch.empty((batch, 512), dtype=torch.bfloat16, device="cpu")
        w1 = torch.empty((4, 256, 512), dtype=torch.float8_e4m3fnuz, device="cpu")
        w2 = torch.empty((4, 512, 128), dtype=torch.float8_e4m3fnuz, device="cpu")
        ids = torch.zeros((batch, 2), dtype=torch.int32, device="cpu")
        routing = torch.ones((batch, 2), dtype=torch.float32, device="cpu")
        s1 = torch.ones((4, 256 if quant_type == "ptpc" else 1, 1), device="cpu")
        s2 = torch.ones((4, 512 if quant_type == "ptpc" else 1, 1), device="cpu")
        sorted_ids = torch.empty(
            batch * 2 + 4 * config.BLOCK_M, dtype=torch.int32, device="cpu"
        )
        sorted_weights = torch.empty_like(sorted_ids, dtype=torch.float32)
        experts = torch.empty(8, dtype=torch.int32, device="cpu")
        valid = torch.empty(2, dtype=torch.int32, device="cpu")
        output = torch.empty_like(hidden)
        stream = object()
        runtime_calls, aot_calls = [], []

        def launch(kernel, *args):
            bound = inspect.signature(kernel.func).bind(*args, stream).arguments
            self.assertIs(bound["p_weight"], w1 if not runtime_calls else w2)
            self.assertIs(bound["p_w_scale"], s1 if not runtime_calls else s2)
            self.assertIs(bound["stream"], stream)
            if batch == 1:
                self.assertIs(bound["p_topk_ids"], ids)
                self.assertIs(bound["p_topk_weights"], routing)
                self.assertEqual(bound["task_num"], 2)
            else:
                self.assertIs(bound["p_sorted_ids"], sorted_ids)
                self.assertIs(bound["p_sorted_weights"], sorted_weights)
                self.assertIs(bound["p_sorted_expert_ids"], experts)
                self.assertIs(bound["p_num_valid_ids"], valid)
                self.assertEqual(bound["M"], batch)
                self.assertEqual(bound["task_num"], experts.shape[0])
            runtime_calls.append(kernel)

        def preload(kernel, *args):
            inspect.signature(kernel.func).bind(*args)
            aot_calls.append(kernel)

        def quantize(value, **_kwargs):
            return (
                torch.empty_like(value, dtype=torch.float8_e4m3fnuz),
                torch.ones((value.shape[0], 1), device="cpu"),
            )

        with (
            patch.dict(os.environ, {"ARCH": "gfx942", "FLYDSL_GPU_ARCH": "gfx942"}),
            patch.object(self.backend, "_launch", side_effect=launch),
            patch.object(self.host, "_preload_compiled", side_effect=preload),
            patch.object(
                self.backend,
                "moe_sorting",
                return_value=(sorted_ids, sorted_weights, experts, valid, output),
            ),
            patch.object(self.backend.aiter, "get_hip_quant", return_value=quantize),
            patch.object(self.backend, "_quant_per_tensor", side_effect=quantize),
            patch.object(self.backend, "invert_sorted_ids"),
            patch.object(self.backend, "sorted_sum"),
            patch.object(self.host, "precompile_moe_reduction_kernels"),
            patch.object(self.host, "precompile_moe_quant_kernels"),
        ):
            request = {
                "hidden_states": hidden,
                "w1": w1,
                "w2": w2,
                "topk_weight": routing,
                "topk_ids": ids,
                "w1_scale": s1,
                "w2_scale": s2,
                "problem": problem,
                "activation_str": "swiglu",
                "swiglu_limit": 1.5,
            }
            if batch == 1:
                self.backend._run_batch1(**request)
            else:
                request.update(
                    expert_mask=None,
                    num_local_tokens=None,
                    moe_sorting_dispatch_policy=0,
                    config=config,
                )
                if config.use_prefill:
                    qt = (
                        self.backend.QuantType.per_Token
                        if quant_type == "ptpc"
                        else self.backend.QuantType.per_Tensor
                    )
                    self.backend._run_prefill(**request, quant_type=qt)
                else:
                    self.backend._run_decode(**request)
            self.backend.precompile_flydsl_moe(
                config_string=config_string,
                batch=batch,
                model_dim=512,
                inter_dim=128,
                experts=4,
                topk=2,
                weight_dtype="fp8",
                quant_type=quant_type,
                activation="swiglu",
                swiglu_limit=1.5,
            )
        self.assertEqual(len(runtime_calls), 2)
        self.assertEqual(len(aot_calls), 2)
        expected = ["p_input", "p_weight", "p_output"]
        if batch == 1:
            expected += [
                "p_topk_ids",
                "p_topk_weights",
                "p_w_scale",
                "task_num",
                "stream",
            ]
        else:
            expected += [
                "p_sorted_ids",
                "p_sorted_weights",
                "p_sorted_expert_ids",
                "p_num_valid_ids",
                "p_w_scale",
            ]
            if config.use_prefill:
                expected += ["p_a_scale"]
            expected += ["M", "task_num", "stream"]
        algorithm = "batch1" if batch == 1 else "splitk"
        if config.use_prefill:
            algorithm = "prefill_1x4"
        for stage, runtime, aot in zip(("gemm1", "gemm2"), runtime_calls, aot_calls):
            self.assertIs(runtime, aot)
            self.assertEqual(runtime.func.__name__, "launch_" + algorithm)
            self.assertIn(stage, runtime.func.__module__)
            signature = inspect.signature(runtime.func)
            self.assertEqual(list(signature.parameters), expected)
            for name, parameter in signature.parameters.items():
                if name.startswith("p_"):
                    expected_type = self.backend.fx.Pointer
                elif name == "stream":
                    expected_type = self.backend.fx.Stream
                else:
                    expected_type = self.backend.fx.Int32
                self.assertIs(parameter.annotation, expected_type)


class TestFlydslMoeInputContract(_CpuTest):
    def _request(self, *, batch=16, fake_device=None, quant_type=None):
        torch = self.torch
        registry = importlib.import_module("aiter.fused_moe_registry")
        if fake_device is not None:
            fake = importlib.import_module("torch._subclasses.fake_tensor")
            mode = fake.FakeTensorMode()
            # FakeTensor can initialize a real device context unless this
            # hardware boundary is disabled, even for a meta-backed tensor.
            self.stack.enter_context(patch.object(fake, "init_gpu_context"))

        def tensor(shape, dtype):
            value = torch.empty(
                shape, dtype=dtype, device="meta" if fake_device is not None else "cpu"
            )
            if fake_device is not None:
                return fake.FakeTensor(mode, value, torch.device(fake_device))
            return value

        w1 = tensor((4, 256, 512), torch.float8_e4m3fnuz)
        w2 = tensor((4, 512, 128), torch.float8_e4m3fnuz)
        w1.is_shuffled = w2.is_shuffled = True
        return registry.FusedMoeRequest(
            hidden_states=tensor((batch, 512), torch.bfloat16),
            w1=w1,
            w2=w2,
            topk_weight=tensor((batch, 2), torch.float32),
            topk_ids=tensor((batch, 2), torch.int32),
            w1_scale=tensor((4, 256, 1), torch.float32),
            w2_scale=tensor((4, 512, 1), torch.float32),
            activation=self.backend.ActivationType.Silu,
            quant_type=quant_type or self.backend.QuantType.per_Token,
        )

    @contextlib.contextmanager
    def _tuned_config(self, request, *, block_m=16):
        """Use the real cached selector with an isolated in-memory tuned row."""
        fused = importlib.import_module("aiter.fused_moe")
        experts, model_dim, inter_dim = fused.get_inter_dim(
            request.w1.shape, request.w2.shape
        )
        options = {
            "token": fused.get_padded_M(request.hidden_states.shape[0]),
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "expert": experts,
            "topk": request.topk_ids.shape[1],
            "dtype": request.hidden_states.dtype,
            "q_dtype_a": request.w1.dtype,
            "q_dtype_w": request.w1.dtype,
            "q_type": request.quant_type,
            "use_g1u1": True,
            "activation": request.activation,
            "doweight_stage1": False,
            "hidden_pad": 0,
            "intermediate_pad": 0,
            "input_dtype": request.hidden_states.dtype,
        }
        key = (
            "gfx942",
            80,
            options["token"],
            model_dim,
            inter_dim,
            experts,
            options["topk"],
            str(request.activation),
            str(options["dtype"]),
            str(options["q_dtype_a"]),
            str(options["q_dtype_w"]),
            str(request.quant_type),
            True,
            False,
        )
        row = {
            "kernelName1": _PREFIX + _DECODE,
            "kernelName2": "",
            "block_m": block_m,
            "ksplit": 0,
            "run_1stage": False,
        }
        fused.get_2stage_cfgs.cache_clear()
        try:
            with (
                patch.dict(
                    os.environ,
                    {"AITER_BYPASS_TUNE_CONFIG": "0", "AITER_ONLINE_TUNE": "0"},
                ),
                # Config path discovery can merge production CSVs even when
                # cfg_2stages is populated; this test must never touch them.
                patch.object(
                    fused,
                    "AITER_CONFIGS",
                    SimpleNamespace(AITER_CONFIG_FMOE_FILE="unused-host-moe.csv"),
                ),
                patch.object(fused, "cfg_2stages", ({key: row}, {})),
                patch.object(fused, "get_cu_num", return_value=80),
                patch.object(fused, "get_gfx_runtime", return_value="gfx942"),
                patch.object(fused, "get_gfx", return_value="gfx942"),
            ):
                yield fused, options
        finally:
            fused.get_2stage_cfgs.cache_clear()
            fused.get_block_size_M.cache_clear()
            fused.use_nt.cache_clear()

    def test_staged_entry_rejects_whole_graph_metadata_after_transform(self):
        class ReachedStage1(Exception):
            pass

        fused = importlib.import_module("aiter.fused_moe")
        request = self._request()
        full_impl = Mock()
        whole = fused.MOEMetadata(None, None, 16, 0, full_impl=full_impl)
        stage1 = Mock(side_effect=ReachedStage1)
        staged = fused.MOEMetadata(stage1, Mock(), 16, 0, prequant=False)
        quantize = Mock(side_effect=AssertionError("unexpected quantization"))
        sorted_experts = self.torch.empty(4, dtype=self.torch.int32, device="cpu")
        valid = self.torch.empty(2, dtype=self.torch.int32, device="cpu")
        output = self.torch.empty_like(request.hidden_states)

        def call(transform=None):
            return fused.fused_moe_2stages(
                request.hidden_states,
                request.w1,
                request.w2,
                request.topk_ids.shape[1],
                request.topk_ids.view(-1),
                request.topk_weight.view(-1),
                sorted_experts,
                valid,
                output,
                True,
                16,
                activation=request.activation,
                quant_type=request.quant_type,
                q_dtype_a=request.w1.dtype,
                q_dtype_w=request.w1.dtype,
                w1_scale=request.w1_scale,
                w2_scale=request.w2_scale,
                _metadata_transform=transform,
            )

        for initial, transform in ((whole, None), (staged, Mock(return_value=whole))):
            with (
                self.subTest(transformed=transform is not None),
                patch.object(fused, "get_2stage_cfgs", return_value=initial),
                patch.object(fused, "get_quant", return_value=quantize),
                self.assertRaisesRegex(
                    NotImplementedError, "whole-graph.*use fused_moe"
                ),
            ):
                call(transform)
            if transform is not None:
                transform.assert_called_once_with(initial)
        stage1.assert_not_called()
        full_impl.assert_not_called()
        quantize.assert_not_called()

        # Public dispatch can replace a fresh lookup with its validated staged row.
        transform = Mock(return_value=staged)
        with (
            patch.object(fused, "get_2stage_cfgs", return_value=whole),
            patch.object(fused, "get_quant", return_value=quantize),
            self.assertRaises(ReachedStage1),
        ):
            call(transform)
        transform.assert_called_once_with(whole)
        stage1.assert_called_once()
        full_impl.assert_not_called()
        quantize.assert_not_called()

    def test_selection_keys_block_size_without_poisoning_default_metadata(self):
        with self._tuned_config(self._request()) as (fused, options):
            default = fused.get_2stage_cfgs(**options)
            matching = fused.get_2stage_cfgs(**options, requested_block_size_m=16)
            mismatch = fused.get_2stage_cfgs(**options, requested_block_size_m=32)
            self.assertIsNotNone(default.full_impl)
            self.assertIsNotNone(matching.full_impl)
            self.assertEqual(matching.block_m, 16)
            self.assertIsNone(mismatch.full_impl)
            self.assertIsNotNone(mismatch.stage1)
            self.assertIsNot(default, matching)
            self.assertIs(fused.get_2stage_cfgs(**options), default)
            for block_m, expected in ((16, matching), (32, mismatch)):
                self.assertIs(
                    fused.get_2stage_cfgs(**options, requested_block_size_m=block_m),
                    expected,
                )
        # An unspecified tuned block is not evidence that an override is supported.
        with self._tuned_config(self._request(), block_m=None) as (fused, options):
            metadata = fused.get_2stage_cfgs(**options, requested_block_size_m=16)
            self.assertIsNone(metadata.full_impl)
            self.assertIsNotNone(metadata.stage1)

    def test_selection_rejects_only_active_silu_clamp_limits(self):
        for activation in (
            self.backend.ActivationType.Silu,
            self.backend.ActivationType.Swiglu,
        ):
            request = replace(self._request(), activation=activation)
            with self._tuned_config(request) as (fused, options):
                default = fused.get_2stage_cfgs(**options, swiglu_limit=None)
                self.assertIsNotNone(default.full_impl)
                for limit in (1.5, math.inf, 0.0, None):
                    with self.subTest(activation=activation, limit=limit):
                        metadata = fused.get_2stage_cfgs(**options, swiglu_limit=limit)
                        rejected = (
                            activation == self.backend.ActivationType.Silu
                            and limit == 1.5
                        )
                        self.assertEqual(metadata.full_impl is None, rejected)
                        if rejected:
                            self.assertIsNotNone(metadata.stage1)
                        self.assertIs(
                            fused.get_2stage_cfgs(**options, swiglu_limit=limit),
                            metadata,
                        )
                self.assertIs(
                    fused.get_2stage_cfgs(**options, swiglu_limit=None), default
                )

    def test_public_entry_passes_block_size_and_falls_back_before_full_dispatch(self):
        class ReachedFallback(Exception):
            pass

        request = self._request()
        output = self.torch.empty_like(request.hidden_states)

        def call(block_m):
            return fused.fused_moe(
                request.hidden_states,
                request.w1,
                request.w2,
                request.topk_weight,
                request.topk_ids,
                activation=request.activation,
                quant_type=request.quant_type,
                w1_scale=request.w1_scale,
                w2_scale=request.w2_scale,
                block_size_M=block_m,
            )

        with (
            self._tuned_config(request) as (fused, _),
            patch.object(
                fused, "get_2stage_cfgs", wraps=fused.get_2stage_cfgs
            ) as lookup,
            patch.object(
                self.backend, "run_flydsl_moe_gfx942_impl", return_value=output
            ) as dispatch,
            patch.object(fused, "moe_sorting", side_effect=ReachedFallback) as sorting,
        ):
            for block_m in (None, 16, 32, 16, None):
                lookup.reset_mock()
                dispatch.reset_mock()
                sorting.reset_mock()
                with self.subTest(block_m=block_m):
                    if block_m == 32:
                        with self.assertRaises(ReachedFallback):
                            call(block_m)
                        dispatch.assert_not_called()
                        sorting.assert_called_once()
                        self.assertEqual(sorting.call_args.args[5], block_m)
                    else:
                        self.assertIs(call(block_m), output)
                        dispatch.assert_called_once()
                        self.assertEqual(dispatch.call_args.args[0].block_size_m, 16)
                        sorting.assert_not_called()
                    lookup.assert_called_once()
                    self.assertEqual(
                        lookup.call_args.kwargs["requested_block_size_m"], block_m
                    )

    def test_wrapper_rejects_silu_clamp_but_preserves_supported_activations(self):
        silu = self.backend.ActivationType.Silu
        swiglu = self.backend.ActivationType.Swiglu
        for config, batch, runner in (
            (_DECODE, 1, "_run_batch1"),
            (_DECODE, 16, "_run_decode"),
            (_PREFILL, 64, "_run_prefill"),
        ):
            request = self._request(batch=batch, fake_device="cuda:0")
            for activation, limit in (
                (silu, 1.5),
                (silu, None),
                (silu, math.inf),
                (silu, 0.0),
                (swiglu, 1.5),
            ):
                candidate = replace(request, activation=activation, swiglu_limit=limit)
                with (
                    self.subTest(
                        config=config, batch=batch, activation=activation, limit=limit
                    ),
                    self._devices(),
                    patch.object(
                        self.backend, runner, return_value=request.hidden_states
                    ) as run,
                ):
                    if activation == silu and limit == 1.5:
                        with self.assertRaisesRegex(
                            NotImplementedError, "swiglu_limit.*Silu"
                        ):
                            self.backend.run_flydsl_moe_gfx942_impl(candidate, config)
                        run.assert_not_called()
                    else:
                        self.assertIs(
                            self.backend.run_flydsl_moe_gfx942_impl(candidate, config),
                            request.hidden_states,
                        )
                        run.assert_called_once()
                        self.assertEqual(run.call_args.args[-1], limit)

    def test_wrapper_requires_fp32_routing_weights_on_all_paths(self):
        torch = self.torch
        fake = importlib.import_module("torch._subclasses.fake_tensor")
        for config, batch, runner in (
            (_DECODE, 1, "_run_batch1"),
            (_DECODE, 16, "_run_decode"),
            (_PREFILL, 64, "_run_prefill"),
        ):
            request = self._request(batch=batch, fake_device="cuda:0")
            for dtype in (torch.float16, torch.bfloat16, torch.float64, torch.float32):
                routing = fake.FakeTensor(
                    request.topk_weight.fake_mode,
                    torch.empty(request.topk_weight.shape, dtype=dtype, device="meta"),
                    request.topk_weight.device,
                )
                candidate = replace(request, topk_weight=routing)
                with (
                    self.subTest(config=config, batch=batch, dtype=dtype),
                    self._devices(),
                    patch.object(
                        self.backend, runner, return_value=request.hidden_states
                    ) as run,
                ):
                    if dtype != torch.float32:
                        with self.assertRaisesRegex(ValueError, "topk_weight.*float32"):
                            self.backend.run_flydsl_moe_gfx942_impl(candidate, config)
                        run.assert_not_called()
                    else:
                        self.assertIs(
                            self.backend.run_flydsl_moe_gfx942_impl(candidate, config),
                            request.hidden_states,
                        )
                        run.assert_called_once()
                        self.assertIs(run.call_args.args[3], routing)

    def test_request_rejects_unsupported_parameters_before_dispatch(self):
        torch = self.torch
        request = self._request()
        updates = (
            {"bias1": request.w1_scale},
            {"bias2": request.w2_scale},
            {"doweight_stage1": True},
            {"a1_scale": request.w1_scale},
            {"a2_scale": request.w2_scale},
            {"hidden_pad": 1},
            {"intermediate_pad": 1},
            {"gate_mode": "interleave"},
            {"gate_mode": "gate_only"},
            {"dtype": torch.float16},
            {"block_size_m": 64},
            {"ksplit": 1},
            {"q_dtype_a": torch.bfloat16},
            {"q_dtype_a": torch.float8_e4m3fn},
            {"q_dtype_a": torch.int8},
            {"q_dtype_w": torch.bfloat16},
            {"q_dtype_w": torch.float8_e4m3fn},
        )
        for update in updates:
            with (
                self.subTest(update=tuple(update)),
                patch.object(self.backend, "run_flydsl_moe_gfx942") as dispatch,
                self.assertRaises(NotImplementedError),
            ):
                self.backend.run_flydsl_moe_gfx942_impl(
                    replace(request, **update), _DECODE
                )
            dispatch.assert_not_called()
        for weight in (request.w1, request.w2):
            weight.is_shuffled = False
            with self.assertRaisesRegex(NotImplementedError, "preshuffled"):
                self.backend.run_flydsl_moe_gfx942_impl(request, _DECODE)
            weight.is_shuffled = True

    def test_wrapper_rejects_invalid_layout_devices_and_scales_before_launch(self):
        torch = self.torch
        self.stack.enter_context(self._devices(ambient=0))
        request = self._request(fake_device="cuda:0")
        other = self._request(fake_device="cuda:1")
        fake = importlib.import_module("torch._subclasses.fake_tensor")

        def tensor_like(value, *, dtype=None, shape=None):
            metadata = torch.empty(
                value.shape if shape is None else shape,
                dtype=value.dtype if dtype is None else dtype,
                device="meta",
            )
            return fake.FakeTensor(value.fake_mode, metadata, value.device)

        updates = [
            {"num_local_tokens": request.topk_ids},
            {"expert_mask": request.topk_ids},
            {"activation": self.backend.ActivationType.Gelu},
            {"quant_type": self.backend.QuantType.No},
            {"quant_type": self.backend.QuantType.per_1x32},
            {"hidden_states": tensor_like(request.hidden_states, dtype=torch.float16)},
            {"w1": tensor_like(request.w1, dtype=torch.bfloat16)},
            {"w2": tensor_like(request.w2, dtype=torch.float8_e4m3fn)},
            {"w1_scale": None},
            {"w2_scale": None},
            {"w1_scale": tensor_like(request.w1_scale, dtype=torch.bfloat16)},
            {"w2_scale": tensor_like(request.w2_scale, shape=(1, 512, 1))},
            {"topk_ids": tensor_like(request.topk_ids, dtype=torch.int64)},
            {"topk_weight": tensor_like(request.topk_weight, shape=(16, 1))},
        ]
        for name in (
            "hidden_states",
            "w1",
            "w2",
            "topk_weight",
            "topk_ids",
            "w1_scale",
            "w2_scale",
        ):
            updates.append({name: getattr(other, name)})
            value = getattr(request, name)
            # A shape-preserving strided view is rejected, not silently copied.
            strided = torch.empty((*value.shape, 2), dtype=value.dtype, device="meta")[
                ..., 0
            ]
            updates.append(
                {name: fake.FakeTensor(value.fake_mode, strided, value.device)}
            )
        for update in updates:
            with self.subTest(update=tuple(update)), contextlib.ExitStack() as stack:
                for runner in ("_run_batch1", "_run_decode", "_run_prefill"):
                    stack.enter_context(
                        patch.object(
                            self.backend,
                            runner,
                            side_effect=AssertionError(
                                "invalid input reached GPU dispatch"
                            ),
                        )
                    )
                candidate = replace(request, **update)
                # Weight dtype conversions drop tensor attributes; retain the
                # shuffled-layout claim so the dtype validator is exercised.
                candidate.w1.is_shuffled = candidate.w2.is_shuffled = True
                with self.assertRaises((ValueError, RuntimeError, NotImplementedError)):
                    self.backend.run_flydsl_moe_gfx942_impl(candidate, _DECODE)

    def test_wrapper_guards_the_first_gpu_operation_and_restores_ambient_device(self):
        class ReachedGpuBoundary(Exception):
            pass

        for config, batch, boundary in (
            (_DECODE, 1, "_gateup_output"),
            (_DECODE, 16, "moe_sorting"),
            (_PREFILL, 64, "moe_sorting"),
        ):
            request = self._request(batch=batch, fake_device="cuda:1")
            with (
                self.subTest(config=config, batch=batch),
                self._devices(ambient=0) as state,
            ):

                def check_context(*_args, **_kwargs):
                    self.assertEqual(
                        state["current"],
                        1,
                        "the entire MoE operation must enter the input device before sorting/allocation",
                    )
                    raise ReachedGpuBoundary

                with (
                    patch.object(self.backend, boundary, side_effect=check_context),
                    self.assertRaises(ReachedGpuBoundary),
                ):
                    self.backend.run_flydsl_moe_gfx942_impl(request, config)
                self.assertEqual(state["current"], 0)

    def test_warm_paths_skip_redundant_stage_device_guards(self):
        torch = self.torch
        for config, batch, quant_type, guards in (
            (_DECODE, 1, self.backend.QuantType.per_Tensor, 1),
            (_DECODE, 16, self.backend.QuantType.per_Token, 1),
            (_PREFILL, 64, self.backend.QuantType.per_Token, 3),
            (_PREFILL, 64, self.backend.QuantType.per_Tensor, 7),
        ):
            for ambient in (0, 1):
                self._clear_caches()
                request = self._request(
                    batch=batch, fake_device="cuda:1", quant_type=quant_type
                )
                with (
                    self.subTest(config=config, quant_type=quant_type, ambient=ambient),
                    self._devices(ambient=ambient) as state,
                    request.hidden_states.fake_mode,
                    contextlib.ExitStack() as stack,
                ):

                    def dispatch(*_args):
                        self.assertEqual(state["current"], 1)

                    def sorting(*_args, _batch=batch, _request=request):
                        self.assertEqual(state["current"], 1)
                        return (
                            torch.empty(
                                _batch * 2 + 256, dtype=torch.int32, device="cuda:1"
                            ),
                            torch.empty(
                                _batch * 2 + 256, dtype=torch.float32, device="cuda:1"
                            ),
                            torch.empty(8, dtype=torch.int32, device="cuda:1"),
                            torch.empty(2, dtype=torch.int32, device="cuda:1"),
                            torch.empty_like(_request.hidden_states),
                        )

                    def quantize(value, **kwargs):
                        self.assertEqual(state["current"], 1)
                        return (
                            torch.empty_like(value, dtype=kwargs["quant_dtype"]),
                            torch.empty(
                                (value.shape[0], 1),
                                dtype=torch.float32,
                                device=value.device,
                            ),
                        )

                    stack.enter_context(
                        patch.object(
                            torch.cuda, "current_stream", return_value=object()
                        )
                    )
                    stack.enter_context(
                        patch.object(self.backend, "moe_sorting", side_effect=sorting)
                    )
                    stack.enter_context(
                        patch.object(
                            self.backend.aiter, "get_hip_quant", return_value=quantize
                        )
                    )
                    for module in (self.backend, self.reduce, self.quant):
                        stack.enter_context(
                            patch.object(
                                module, "_ptr", side_effect=lambda tensor: tensor
                            )
                        )
                        stack.enter_context(
                            patch.object(module, "_run_compiled", side_effect=dispatch)
                        )
                    self.backend.run_flydsl_moe_gfx942_impl(request, config)
                    before = self.host._get_compiled_kernel.cache_info()
                    state["entered"].clear()
                    state["restored"].clear()
                    with patch.object(
                        self.host,
                        "device_context",
                        side_effect=AssertionError(
                            "warm stage factory entered a device context"
                        ),
                    ):
                        for _ in range(3):
                            output = self.backend.run_flydsl_moe_gfx942_impl(
                                request, config
                            )
                            self.assertEqual(
                                output.device, request.hidden_states.device
                            )
                    after = self.host._get_compiled_kernel.cache_info()
                    self.assertEqual(after.misses, before.misses)
                    self.assertEqual(after.hits - before.hits, 6)
                    # Auxiliary reduction/quantization guards are outside this patch.
                    self.assertEqual(state["entered"], [1] * (3 * guards))
                    self.assertEqual(
                        state["restored"], ([1] * (guards - 1) + [ambient]) * 3
                    )
                    self.assertEqual(state["current"], ambient)

    def test_stage_launch_uses_the_tensor_device_and_its_current_stream(self):
        request = self._request(fake_device="cuda:1")
        stream, launcher = object(), object()

        for ambient, fail in itertools.product((0, 1), (False, True)):
            with (
                self.subTest(ambient=ambient, fail=fail),
                self._devices(ambient=ambient) as state,
            ):

                def current_stream(device):
                    self.assertEqual(device, request.hidden_states.device)
                    self.assertEqual(state["current"], 1)
                    return stream

                def dispatch(kernel, *args, _fail=fail):
                    self.assertIs(kernel, launcher)
                    self.assertIs(args[0], request.hidden_states)
                    self.assertEqual(args[1], 16)
                    self.assertIs(args[2], stream)
                    self.assertEqual(state["current"], 1)
                    if _fail:
                        raise RuntimeError("launch failed")

                with (
                    patch.object(
                        self.backend, "_ptr", side_effect=lambda tensor: tensor
                    ),
                    patch.object(
                        self.torch.cuda, "current_stream", side_effect=current_stream
                    ) as get_stream,
                    patch.object(
                        self.backend, "_run_compiled", side_effect=dispatch
                    ) as run,
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "launch failed"):
                            self.backend._launch(launcher, request.hidden_states, 16)
                    else:
                        self.backend._launch(launcher, request.hidden_states, 16)
                get_stream.assert_called_once_with(request.hidden_states.device)
                run.assert_called_once()
                self.assertEqual(state["current"], ambient)
                self.assertEqual(state["entered"], [1] if ambient == 0 else [])
                self.assertEqual(state["restored"], [0] if ambient == 0 else [])

    def test_wrapper_rejects_a_non_gfx942_device_even_with_spoofed_target_env(self):
        request = self._request(fake_device="cuda:1")
        with (
            self._devices(arch="gfx950"),
            patch.dict(
                os.environ,
                {"ARCH": "gfx942", "FLYDSL_GPU_ARCH": "gfx942", "GPU_ARCHS": "gfx942"},
            ),
            patch.object(
                self.backend,
                "_run_decode",
                side_effect=AssertionError("gfx950 reached the gfx942-only backend"),
            ),
            self.assertRaisesRegex(
                (ValueError, RuntimeError, NotImplementedError), "gfx942"
            ),
        ):
            self.backend.run_flydsl_moe_gfx942_impl(request, _DECODE)

    def test_public_fused_moe_preserves_output_identity_and_validates_buffers(self):
        fused = importlib.import_module("aiter.fused_moe")
        request = self._request()
        torch = self.torch
        output = torch.empty_like(request.hidden_states)
        expected = torch.full_like(output, 0.25)
        implementation = Mock(return_value=expected)
        metadata = fused.MOEMetadata(None, None, 16, 0, full_impl=implementation)

        def call(destination):
            return fused.fused_moe(
                request.hidden_states,
                request.w1,
                request.w2,
                request.topk_weight,
                request.topk_ids,
                activation=request.activation,
                quant_type=request.quant_type,
                w1_scale=request.w1_scale,
                w2_scale=request.w2_scale,
                output=destination,
            )

        with (
            patch.object(fused, "get_gfx", return_value="gfx942"),
            patch.object(fused, "get_2stage_cfgs", return_value=metadata) as lookup,
        ):
            self.assertIs(call(output), output)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            implementation.assert_called_once()
            bad_buffers = (
                output[:, :-1].contiguous(),
                output.float(),
                torch.empty((512, 16), dtype=torch.bfloat16, device="cpu").t(),
                torch.empty(output.shape, dtype=output.dtype, device="meta"),
                request.hidden_states,
                request.hidden_states.view_as(request.hidden_states),
            )
            for index, bad in enumerate(bad_buffers):
                lookup.reset_mock()
                with (
                    self.subTest(buffer=index),
                    self.assertRaisesRegex(RuntimeError, "output"),
                ):
                    call(bad)
                lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
