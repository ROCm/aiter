# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 2: End-to-end tuning pipeline smoke tests (GPU required).

Runs each tuner on small shapes, verifies CSV output, and tests
--shape_grouped with profile row count comparison.
"""

import os
import sys
import csv
import tempfile
import subprocess
import unittest
import pandas as pd

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


def _get_platform_dtypes():
    """Return (fp8_str, quant_type_str) based on GPU arch."""
    try:
        from aiter.jit.utils.chip_info import get_gfx

        gfx = get_gfx()
    except Exception:
        gfx = "gfx942"
    if gfx in ("gfx950", "gfx1250"):
        return "torch.float8_e4m3fn", "QuantType.per_1x128"
    else:
        return "torch.float8_e4m3fnuz", "QuantType.per_Token"


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _run_tuner(script, untuned, tuned, extra_args=None, timeout=300):
    cmd = [
        sys.executable,
        os.path.join(AITER_ROOT, script),
        "-i",
        untuned,
        "-o",
        tuned,
        "--mp",
        "1",
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
            f"Tuner timed out after {timeout}s (likely GPU hang or infinite loop)\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout (last 500): {(e.stdout or b'')[-500:]}\n"
            f"  stderr (last 500): {(e.stderr or b'')[-500:]}"
        ) from None


@unittest.skipUnless(_gpu_available(), "No GPU available")
class TestTunePipeline(unittest.TestCase):
    """Smoke test: run each tuner on 1 small shape, verify CSV output."""

    @classmethod
    def setUpClass(cls):
        fp8, qtype = _get_platform_dtypes()
        cls.TUNERS = {
            "a8w8": {
                "script": "csrc/ck_gemm_a8w8/gemm_a8w8_tune.py",
                "header": ["M", "N", "K", "q_dtype_w"],
                "shapes": [
                    (1, 1024, 512, "torch.int8"),
                    (1, 1024, 512, fp8),
                ],
                "keys": ["cu_num", "M", "N", "K", "q_dtype_w"],
            },
            "a8w8_blockscale": {
                "script": "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
                "header": ["M", "N", "K"],
                "shapes": [(1, 1024, 512)],
                "keys": ["cu_num", "M", "N", "K"],
            },
            "a8w8_bpreshuffle": {
                "script": "csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",
                "header": ["M", "N", "K", "q_dtype_w"],
                "shapes": [
                    (1, 1024, 512, "torch.int8"),
                    (1, 1024, 512, fp8),
                ],
                "keys": ["cu_num", "M", "N", "K", "q_dtype_w"],
            },
            "batched_a8w8": {
                "script": "csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py",
                "header": ["B", "M", "N", "K"],
                "shapes": [(2, 1, 512, 256)],
                "keys": ["cu_num", "B", "M", "N", "K"],
            },
            "batched_bf16": {
                "script": "csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py",
                "header": ["B", "M", "N", "K"],
                "shapes": [(2, 1, 512, 256)],
                "keys": ["cu_num", "B", "M", "N", "K"],
            },
            "fmoe": {
                "script": "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",
                "header": [
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
                ],
                "shapes": [
                    # bf16 (no quant)
                    (
                        512,
                        6144,
                        4096,
                        8,
                        2,
                        "ActivationType.Silu",
                        "torch.bfloat16",
                        "torch.bfloat16",
                        "torch.bfloat16",
                        "QuantType.No",
                        1,
                        0,
                    ),
                    # fp8 per-token (platform-aware)
                    (
                        16,
                        7168,
                        256,
                        256,
                        8,
                        "ActivationType.Silu",
                        "torch.bfloat16",
                        fp8,
                        fp8,
                        qtype,
                        1,
                        0,
                    ),
                    # int8 per-tensor
                    (
                        512,
                        6144,
                        4096,
                        8,
                        2,
                        "ActivationType.Silu",
                        "torch.bfloat16",
                        "torch.int8",
                        "torch.int8",
                        "QuantType.per_Tensor",
                        1,
                        0,
                    ),
                    # Gelu activation + doweight_stage1
                    (
                        4,
                        2304,
                        1536,
                        8,
                        2,
                        "ActivationType.Gelu",
                        "torch.bfloat16",
                        fp8,
                        fp8,
                        qtype,
                        1,
                        1,
                    ),
                ],
                "keys": [
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
                ],
                "timeout": 600,
            },
            "gradlib_bf16": {
                "script": "gradlib/gradlib/gemm_tuner.py",
                "header": [
                    "M",
                    "N",
                    "K",
                    "bias",
                    "dtype",
                    "outdtype",
                    "scaleAB",
                    "bpreshuffle",
                ],
                "shapes": [
                    # decode (M=1): hipBLASLt/ASM typically wins
                    (
                        1,
                        1024,
                        512,
                        "False",
                        "torch.bfloat16",
                        "torch.float32",
                        "False",
                        "False",
                    ),
                    # prefill (large M): FlyDSL has a chance to win
                    (
                        512,
                        5120,
                        1280,
                        "False",
                        "torch.bfloat16",
                        "torch.bfloat16",
                        "False",
                        "False",
                    ),
                ],
                "keys": ["M", "N", "K"],
                "timeout": 600,
            },
        }

    def _run_one(self, name):
        cfg = self.TUNERS[name]
        timeout = cfg.get("timeout", 300)
        with tempfile.TemporaryDirectory() as tmp:
            untuned = os.path.join(tmp, "untuned.csv")
            tuned = os.path.join(tmp, "tuned.csv")
            _write_csv(untuned, cfg["header"], cfg["shapes"])

            result = _run_tuner(cfg["script"], untuned, tuned, timeout=timeout)
            if result.returncode != 0:
                print(f"\n=== {name} STDOUT ===\n{result.stdout[-2000:]}")
                print(f"\n=== {name} STDERR ===\n{result.stderr[-2000:]}")
            self.assertEqual(
                result.returncode,
                0,
                f"{name} tuner exited with code {result.returncode}",
            )
            self.assertTrue(os.path.exists(tuned), f"{name}: tuned CSV not created")

            df = pd.read_csv(tuned)
            df.columns = df.columns.str.strip()
            self.assertGreaterEqual(
                len(df),
                len(cfg["shapes"]),
                f"{name}: expected >= {len(cfg['shapes'])} rows",
            )
            for key in cfg["keys"]:
                self.assertIn(key, df.columns, f"{name}: missing column {key}")
            for _, row in df.iterrows():
                us = float(row.get("us", -1))
                self.assertNotEqual(us, 0, f"{name}: us == 0 for {dict(row)}")

    def test_a8w8(self):
        self._run_one("a8w8")

    def test_a8w8_blockscale(self):
        self._run_one("a8w8_blockscale")

    def test_a8w8_bpreshuffle(self):
        self._run_one("a8w8_bpreshuffle")

    def test_batched_a8w8(self):
        self._run_one("batched_a8w8")

    def test_batched_bf16(self):
        self._run_one("batched_bf16")

    def test_fmoe(self):
        self._run_one("fmoe")

    def test_gradlib_bf16(self):
        """gradlib spawns an internal subprocess; use /tmp paths that persist."""
        cfg = self.TUNERS["gradlib_bf16"]
        timeout = cfg.get("timeout", 300)
        untuned = "/tmp/_test_gradlib_untuned.csv"
        tuned = "/tmp/_test_gradlib_tuned.csv"
        try:
            _write_csv(untuned, cfg["header"], cfg["shapes"])
            if os.path.exists(tuned):
                os.remove(tuned)
            result = _run_tuner(cfg["script"], untuned, tuned, timeout=timeout)
            if result.returncode != 0:
                print(f"\n=== gradlib STDOUT ===\n{result.stdout[-2000:]}")
                print(f"\n=== gradlib STDERR ===\n{result.stderr[-2000:]}")
            self.assertEqual(result.returncode, 0, "gradlib tuner failed")
            self.assertTrue(os.path.exists(tuned), "gradlib: tuned CSV not created")
        finally:
            for f in (untuned, tuned):
                if os.path.exists(f):
                    os.remove(f)


@unittest.skipUnless(_gpu_available(), "No GPU available")
class TestShapeGrouped(unittest.TestCase):
    """Test --shape_grouped: same profile count, correct tuned row count."""

    CONFIGS = {
        "a8w8_blockscale": {
            "script": "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
            "header": ["M", "N", "K"],
            "shapes": [(16, 1536, 7168), (16, 576, 7168), (16, 7168, 256)],
            "keys": ["cu_num", "M", "N", "K"],
        },
        "batched_bf16": {
            "script": "csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py",
            "header": ["B", "M", "N", "K"],
            "shapes": [(2, 1, 512, 256), (4, 16, 1024, 512)],
            "keys": ["cu_num", "B", "M", "N", "K"],
        },
    }

    def _run_grouped_vs_ref(self, name):
        cfg = self.CONFIGS[name]
        num_shapes = len(cfg["shapes"])
        with tempfile.TemporaryDirectory() as tmp:
            untuned = os.path.join(tmp, "untuned.csv")
            tuned_ref = os.path.join(tmp, "tuned_ref.csv")
            profile_ref = os.path.join(tmp, "profile_ref.csv")
            tuned = os.path.join(tmp, "tuned.csv")
            profile = os.path.join(tmp, "profile.csv")
            _write_csv(untuned, cfg["header"], cfg["shapes"])

            r_ref = _run_tuner(
                cfg["script"], untuned, tuned_ref, extra_args=["-o2", profile_ref]
            )
            self.assertEqual(
                r_ref.returncode, 0, f"{name} ref tuner failed:\n{r_ref.stderr[-1000:]}"
            )

            r = _run_tuner(
                cfg["script"],
                untuned,
                tuned,
                extra_args=["--shape_grouped", "-o2", profile],
            )
            if r.returncode != 0:
                print(f"\n=== {name} grouped STDERR ===\n{r.stderr[-2000:]}")
            self.assertEqual(r.returncode, 0, f"{name} grouped tuner failed")

            df = pd.read_csv(tuned)
            df.columns = df.columns.str.strip()
            self.assertEqual(
                len(df),
                num_shapes,
                f"{name}: expected {num_shapes} tuned rows, got {len(df)}",
            )

            if os.path.exists(profile) and os.path.exists(profile_ref):
                prof = pd.read_csv(profile)
                prof_ref = pd.read_csv(profile_ref)
                self.assertEqual(
                    len(prof),
                    len(prof_ref),
                    f"{name}: profile rows grouped={len(prof)} vs ref={len(prof_ref)}",
                )

    def test_a8w8_blockscale(self):
        self._run_grouped_vs_ref("a8w8_blockscale")

    def test_batched_bf16(self):
        self._run_grouped_vs_ref("batched_bf16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
