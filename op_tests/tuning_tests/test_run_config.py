# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 2: Run all existing tuned configs through --run_config to verify
the production operator works with every shape in the config CSVs.

For each tuner family, discovers all tuned CSVs (default + model_configs),
merges them via pathsep, and runs --run_config to benchmark every shape.
Any shape that errors (us=-1 or exception) is reported as a test failure.

Run:
    python3 -m unittest op_tests.tuning_tests.test_run_config -v
"""

import os
import sys
import subprocess
import unittest

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "configs")
MODEL_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "model_configs")

# Override: specify a tuner family and config CSV to test directly.
#   TUNE_TEST_FAMILY=a8w8_blockscale TUNE_TEST_CONFIG=/path/to/tuned.csv \
#     python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v
#
# TUNE_TEST_CONFIG supports pathsep (:) for merging multiple CSVs, e.g.:
#   TUNE_TEST_CONFIG="configs/a8w8_blockscale_tuned_gemm.csv:model_configs/xxx.csv"
TUNE_TEST_FAMILY = os.environ.get("TUNE_TEST_FAMILY")
TUNE_TEST_CONFIG = os.environ.get("TUNE_TEST_CONFIG")


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


def _find_tuned_csvs(pattern):
    """Find all tuned CSVs matching pattern in configs/ and model_configs/."""
    found = []
    for d in (CONFIGS_DIR, MODEL_CONFIGS_DIR):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if (
                pattern in f
                and "tuned" in f
                and "untuned" not in f
                and f.endswith(".csv")
            ):
                found.append(os.path.join(d, f))
    return found


def _merge_config_paths(csv_list):
    """Merge multiple CSV paths with os.pathsep (like AITER_CONFIG_* env)."""
    return os.pathsep.join(csv_list)


def _run_config(script, config_csv, timeout=600, extra_args=None):
    """Run tuner with --run_config <tuned_csv> and return result."""
    cmd = [
        sys.executable,
        os.path.join(AITER_ROOT, script),
        "--run_config",
        config_csv,
        "--warmup",
        "2",
        "--iters",
        "5",
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    script_dir = os.path.dirname(os.path.join(AITER_ROOT, script))
    env["PYTHONPATH"] = script_dir + ":" + env.get("PYTHONPATH", "")
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=AITER_ROOT,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"run_config timed out after {timeout}s\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout (last 500): {(e.stdout or b'')[-500:]}\n"
            f"  stderr (last 500): {(e.stderr or b'')[-500:]}"
        ) from None


TUNER_FAMILIES = {
    "a8w8": {
        "script": "csrc/ck_gemm_a8w8/gemm_a8w8_tune.py",
        "csv_pattern": "a8w8_tuned_gemm",
        "exclude_patterns": ["bpreshuffle", "blockscale", "batched"],
    },
    "a8w8_bpreshuffle": {
        "script": "csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",
        "csv_pattern": "a8w8_bpreshuffle_tuned_gemm",
        "exclude_patterns": ["blockscale"],
    },
    "a8w8_blockscale": {
        "script": "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        "csv_pattern": "a8w8_blockscale_tuned_gemm",
        "exclude_patterns": ["bpreshuffle", "fmoe"],
    },
    "a8w8_blockscale_bpreshuffle": {
        "script": "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        "csv_pattern": "a8w8_blockscale_bpreshuffle_tuned_gemm",
        "exclude_patterns": ["fmoe"],
        "extra_args": ["--preshuffle"],
    },
    "a4w4_blockscale": {
        "script": "csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py",
        "csv_pattern": "a4w4_blockscale_tuned_gemm",
        "exclude_patterns": [],
    },
    "batched_a8w8": {
        "script": "csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py",
        "csv_pattern": "a8w8_tuned_batched_gemm",
        "exclude_patterns": [],
    },
    "batched_bf16": {
        "script": "csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py",
        "csv_pattern": "bf16_tuned_batched_gemm",
        "exclude_patterns": [],
    },
    "fmoe": {
        "script": "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",
        "csv_pattern": "tuned_fmoe",
        "exclude_patterns": ["untuned", "profile"],
        "timeout": 1200,
    },
}


@unittest.skipUnless(_gpu_available(), "No GPU available")
class TestRunConfig(unittest.TestCase):
    """Run --run_config on all existing tuned CSVs to verify production ops."""

    def _test_family(self, name):
        cfg = TUNER_FAMILIES[name]
        pattern = cfg["csv_pattern"]
        excludes = cfg.get("exclude_patterns", [])
        timeout = cfg.get("timeout", 600)

        csvs = _find_tuned_csvs(pattern)
        csvs = [
            c for c in csvs if not any(ex in os.path.basename(c) for ex in excludes)
        ]

        if not csvs:
            self.skipTest(f"No tuned CSVs found for {name} (pattern={pattern})")

        merged = _merge_config_paths(csvs)
        csv_names = [os.path.basename(c) for c in csvs]
        extra_args = cfg.get("extra_args", None)

        result = _run_config(
            cfg["script"], merged, timeout=timeout, extra_args=extra_args
        )

        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"\n=== {name} run_config FAILED ===")
            print(f"CSVs: {csv_names}")
            print(f"STDOUT (last 2000):\n{result.stdout[-2000:]}")
            print(f"STDERR (last 2000):\n{result.stderr[-2000:]}")

        self.assertEqual(
            result.returncode, 0, f"{name} run_config failed (csvs={csv_names})"
        )

        # Parse benchmark result lines from the table output.
        # Status column shows: OK, ERROR, MISMATCH
        lines = output.split("\n")
        error_shapes = []
        mismatch_shapes = []
        for line in lines:
            stripped = line.strip()
            if "| " not in stripped:
                continue
            if stripped.endswith("ERROR"):
                error_shapes.append(stripped)
            elif stripped.endswith("MISMATCH"):
                mismatch_shapes.append(stripped)

        failures = []
        if error_shapes:
            failures.append(
                f"Errors ({len(error_shapes)} shapes):\n" + "\n".join(error_shapes[:20])
            )
        if mismatch_shapes:
            failures.append(
                f"Accuracy mismatches ({len(mismatch_shapes)} shapes):\n"
                + "\n".join(mismatch_shapes[:20])
            )

        self.assertEqual(
            len(failures),
            0,
            f"{name} run_config issues (csvs={csv_names}):\n" + "\n".join(failures),
        )

    def test_a8w8(self):
        self._test_family("a8w8")

    def test_a8w8_bpreshuffle(self):
        self._test_family("a8w8_bpreshuffle")

    def test_a8w8_blockscale(self):
        self._test_family("a8w8_blockscale")

    def test_a8w8_blockscale_bpreshuffle(self):
        self._test_family("a8w8_blockscale_bpreshuffle")

    def test_a4w4_blockscale(self):
        self._test_family("a4w4_blockscale")

    def test_batched_a8w8(self):
        self._test_family("batched_a8w8")

    def test_batched_bf16(self):
        self._test_family("batched_bf16")

    def test_fmoe(self):
        self._test_family("fmoe")


@unittest.skipUnless(_gpu_available(), "No GPU available")
@unittest.skipUnless(
    TUNE_TEST_FAMILY and TUNE_TEST_CONFIG,
    "Set TUNE_TEST_FAMILY and TUNE_TEST_CONFIG to run",
)
class TestRunConfigCustom(unittest.TestCase):
    """Run --run_config with user-specified family and config CSV.

    Usage:
        TUNE_TEST_FAMILY=a8w8_blockscale \
        TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_tuned_gemm.csv" \
        python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v

        # Multiple configs (merged):
        TUNE_TEST_FAMILY=a8w8_blockscale \
        TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_tuned_gemm.csv:aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv" \
        python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v
    """

    def test_custom(self):
        family = TUNE_TEST_FAMILY
        config = TUNE_TEST_CONFIG
        self.assertIn(
            family,
            TUNER_FAMILIES,
            f"Unknown family '{family}'. Available: {list(TUNER_FAMILIES.keys())}",
        )
        cfg = TUNER_FAMILIES[family]
        timeout = cfg.get("timeout", 600)

        # Resolve relative paths against AITER_ROOT
        resolved = []
        for p in config.split(os.pathsep):
            p = p.strip()
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.join(AITER_ROOT, p)
            self.assertTrue(os.path.exists(p), f"Config not found: {p}")
            resolved.append(p)
        merged = os.pathsep.join(resolved)

        print(
            f"\nRunning {family} --run_config with: {[os.path.basename(p) for p in resolved]}"
        )
        result = _run_config(cfg["script"], merged, timeout=timeout)

        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout[-3000:]}")
            print(f"STDERR:\n{result.stderr[-3000:]}")
        self.assertEqual(result.returncode, 0, f"{family} run_config failed")

        lines = output.split("\n")
        error_shapes = []
        mismatch_shapes = []
        for line in lines:
            stripped = line.strip()
            if "| " not in stripped:
                continue
            if stripped.endswith("ERROR"):
                error_shapes.append(stripped)
            elif stripped.endswith("MISMATCH"):
                mismatch_shapes.append(stripped)

        failures = []
        if error_shapes:
            failures.append(
                f"Errors ({len(error_shapes)} shapes):\n" + "\n".join(error_shapes[:20])
            )
        if mismatch_shapes:
            failures.append(
                f"Accuracy mismatches ({len(mismatch_shapes)} shapes):\n"
                + "\n".join(mismatch_shapes[:20])
            )

        self.assertEqual(
            len(failures), 0, f"{family} run_config issues:\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
