# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 0: Static validation of tuned/untuned CSV files (no GPU, fast).

Catches: duplicates, invalid times, high errRatio, git merge conflicts,
missing untuned files.
"""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import pandas as pd

try:
    from . import validate_hipblaslt_config as hipblaslt_check
except ImportError:  # The default CI shard runs this file directly.
    import validate_hipblaslt_config as hipblaslt_check

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "configs")


class TestCSVValidation(unittest.TestCase):
    TUNED_CSVS: ClassVar[dict[str, Any]] = {
        "a8w8": "a8w8_tuned_gemm.csv",
        "a8w8_bpreshuffle": "a8w8_bpreshuffle_tuned_gemm.csv",
        "a8w8_blockscale": "a8w8_blockscale_tuned_gemm.csv",
        "a8w8_blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        "a8w8_blockscale_mxscale_bpreshuffle": (
            "a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
        ),
        "a4w4_blockscale": "a4w4_blockscale_tuned_gemm.csv",
        "a6w6_blockscale": "a6w6_blockscale_tuned_gemm.csv",
        "a6w4_asm": "a6w4_asm_tuned_gemm.csv",
        "a4w6_asm": "a4w6_asm_tuned_gemm.csv",
        "a8w8_batched": "a8w8_tuned_batched_gemm.csv",
        "bf16": "bf16_tuned_gemm.csv",
        "bf16_batched": "bf16_tuned_batched_gemm.csv",
        "fmoe": "tuned_fmoe.csv",
    }

    def _load_csv(self, name):
        path = os.path.join(CONFIGS_DIR, self.TUNED_CSVS[name])
        if not os.path.exists(path):
            self.skipTest(f"{self.TUNED_CSVS[name]} not found")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df

    def _get_key_cols(self, df):
        candidates = [
            "gfx",
            "cu_num",
            "M",
            "N",
            "K",
            "B",
            "token",
            "model_dim",
            "inter_dim",
            "expert",
            "topk",
        ]
        return [c for c in candidates if c in df.columns]

    def _check_no_duplicates(self, name, extra_keys=None):
        df = self._load_csv(name)
        keys = self._get_key_cols(df)
        if extra_keys:
            keys.extend([k for k in extra_keys if k in df.columns])
        dupes = df[df.duplicated(subset=keys, keep=False)]
        self.assertEqual(
            len(dupes),
            0,
            f"{name}: {len(dupes)} duplicate rows (first 10):\n{dupes.head(10)}",
        )

    def _check_gemm_bandwidth_gbps(self, name, bpes):
        df = self._load_csv(name)
        lhs_bpe, rhs_bpe, out_bpe = bpes
        expected = (
            (
                df["M"] * df["K"] * lhs_bpe
                + df["N"] * df["K"] * rhs_bpe
                + df["M"] * df["N"] * out_bpe
            )
            / (df["us"] * 1e-6)
            / 1e9
        )
        error = (df["bw"] - expected.round(2)).abs()
        self.assertFalse(
            error.isna().any(),
            f"{name}: bw validation contains missing values",
        )
        self.assertLessEqual(
            float(error.max()),
            0.02,
            f"{name}: bw must be stored in GB/s; largest error:\n"
            f"{df.loc[error.idxmax()]}",
        )

    def test_a8w8_no_duplicates(self):
        self._check_no_duplicates("a8w8", extra_keys=["q_dtype_w"])

    def test_a8w8_blockscale_no_duplicates(self):
        self._check_no_duplicates("a8w8_blockscale")

    def test_a8w8_bpreshuffle_no_duplicates(self):
        self._check_no_duplicates(
            "a8w8_bpreshuffle", extra_keys=["q_dtype_w", "libtype"]
        )

    def test_a8w8_blockscale_bpreshuffle_no_duplicates(self):
        self._check_no_duplicates("a8w8_blockscale_bpreshuffle", extra_keys=["libtype"])

    def test_a8w8_blockscale_mxscale_bpreshuffle_no_duplicates(self):
        # One row per shape and block: kernelId names the backend (bmm or mxfp8
        # on gfx950), not a second contract.
        self._check_no_duplicates(
            "a8w8_blockscale_mxscale_bpreshuffle", extra_keys=["w_scale_block"]
        )

    def test_a4w4_blockscale_no_duplicates(self):
        self._check_no_duplicates("a4w4_blockscale")

    def test_a6w6_blockscale_no_duplicates(self):
        self._check_no_duplicates("a6w6_blockscale")

    def test_a6w4_asm_no_duplicates(self):
        self._check_no_duplicates("a6w4_asm")

    def test_a4w6_asm_no_duplicates(self):
        self._check_no_duplicates("a4w6_asm")

    def test_mxfp_gemm_bandwidth_is_gbps(self):
        self._check_gemm_bandwidth_gbps("a6w6_blockscale", (0.75, 0.75, 2.0))
        self._check_gemm_bandwidth_gbps("a6w4_asm", (0.75, 0.5, 2.0))
        self._check_gemm_bandwidth_gbps("a4w6_asm", (0.5, 0.75, 2.0))

    def test_a8w8_batched_no_duplicates(self):
        self._check_no_duplicates("a8w8_batched")

    def test_bf16_no_duplicates(self):
        self._check_no_duplicates(
            "bf16",
            extra_keys=[
                "bias",
                "dtype",
                "outdtype",
                "scaleAB",
                "bpreshuffle",
                "libtype",
            ],
        )

    def test_bf16_batched_no_duplicates(self):
        self._check_no_duplicates("bf16_batched")

    def test_fmoe_no_duplicates(self):
        self._check_no_duplicates(
            "fmoe",
            extra_keys=[
                "act_type",
                "dtype",
                "q_dtype_a",
                "q_dtype_w",
                "q_type",
                "use_g1u1",
                "doweight_stage1",
                "_tag",
            ],
        )

    def test_flydsl_stage2_sort_block_matches_fmoe_config(self):
        """Stage2 must consume the same sorting layout emitted by stage1."""
        tile_pattern = re.compile(r"_t([0-9]+)x[0-9]+x[0-9]+")
        sort_block_pattern = re.compile(r"_sbm([0-9]+)(?:_|$)")
        mismatches = []

        for root, _, files in os.walk(CONFIGS_DIR):
            for filename in files:
                if "tuned_fmoe" not in filename or not filename.endswith(".csv"):
                    continue
                path = os.path.join(root, filename)
                df = pd.read_csv(path)
                if "block_m" not in df.columns or "kernelName2" not in df.columns:
                    continue
                for index, row in df.iterrows():
                    kernel_name = str(row["kernelName2"])
                    if not kernel_name.startswith("flydsl_moe2_"):
                        continue
                    tile_match = tile_pattern.search(kernel_name)
                    if tile_match is None:
                        continue
                    sort_block_match = sort_block_pattern.search(kernel_name)
                    sort_block_m = int(
                        sort_block_match.group(1)
                        if sort_block_match is not None
                        else tile_match.group(1)
                    )
                    block_m = int(row["block_m"])
                    if sort_block_m != block_m:
                        relative_path = os.path.relpath(path, AITER_ROOT)
                        mismatches.append(
                            f"{relative_path}:{index + 2}: block_m={block_m}, "
                            f"sort_block_m={sort_block_m}, kernelName2={kernel_name}"
                        )

        self.assertFalse(
            mismatches,
            "FlyDSL stage2 sorting layout mismatches:\n" + "\n".join(mismatches),
        )

    def test_no_git_conflict_markers(self):
        for name, fname in self.TUNED_CSVS.items():
            with self.subTest(csv=name):
                path = os.path.join(CONFIGS_DIR, fname)
                if not os.path.exists(path):
                    continue
                with open(path, "r") as f:
                    content = f.read()
                for marker in ["<<<<<<<", "=======", ">>>>>>>"]:
                    self.assertNotIn(
                        marker, content, f"{name}: git conflict marker '{marker}' found"
                    )

    def test_no_invalid_times(self):
        for name in self.TUNED_CSVS:
            with self.subTest(csv=name):
                df = self._load_csv(name)
                if "us" not in df.columns:
                    continue
                us = pd.to_numeric(df["us"], errors="coerce")
                bad = df[us <= 0]
                self.assertEqual(
                    len(bad), 0, f"{name}: {len(bad)} rows with us <= 0:\n{bad.head(5)}"
                )

    def test_error_ratios_within_bounds(self):
        for name in self.TUNED_CSVS:
            with self.subTest(csv=name):
                df = self._load_csv(name)
                if "errRatio" not in df.columns:
                    continue
                err_col = df["errRatio"]
                if err_col.dtype == object:
                    err_col = err_col.str.rstrip("%").astype(float) / 100.0
                else:
                    err_col = pd.to_numeric(err_col, errors="coerce")
                high = df[err_col > 0.2]
                self.assertEqual(
                    len(high),
                    0,
                    f"{name}: {len(high)} rows with errRatio > 0.2:\n{high.head(5)}",
                )

    def test_untuned_csvs_exist(self):
        untuned_files = [
            "a8w8_untuned_gemm.csv",
            "a8w8_bpreshuffle_untuned_gemm.csv",
            "a8w8_blockscale_untuned_gemm.csv",
            "a6w6_blockscale_untuned_gemm.csv",
            "a6w4_asm_untuned_gemm.csv",
            "a4w6_asm_untuned_gemm.csv",
            "a8w8_untuned_batched_gemm.csv",
            "bf16_untuned_batched_gemm.csv",
            "untuned_fmoe.csv",
        ]
        for f in untuned_files:
            with self.subTest(file=f):
                path = os.path.join(CONFIGS_DIR, f)
                self.assertTrue(os.path.exists(path), f"Missing: {f}")


class TestHipblasltConfigValidation(unittest.TestCase):
    """CPU regressions for the opt-in live checker, with explicit stand-ins."""

    def test_row_names_and_gpu_selection(self):
        rows = pd.DataFrame(
            [
                {"solidx": str(index), "kernelName": name, "gfx": gfx, "cu_num": cu}
                for index, name, gfx, cu in (
                    (1, "bf16_kernel", "gfx942", "80"),
                    (2, "old_bf16_kernel", "gfx942", "80"),
                    (3, "", "gfx942", "80"),
                    (4, "0", "gfx942", "80"),
                    (5, "other_arch", "gfx950", "80"),
                    (6, "other_cu_count", "gfx942", "304"),
                )
            ]
        ).assign(libtype="hipblaslt", M="16", N="256", K="256")
        rows.loc[0, "solidx"] = "1.0"  # An integral CSV placeholder is still index 1.
        answers = {
            index: {"name": name, "error": None}
            for index, name in (
                (1, "bf16_kernel"),
                (2, "fp8_kernel"),
                (3, "x"),
                (4, "y"),
            )
        }
        with patch.object(
            hipblaslt_check, "_query_indices", return_value=(answers, {})
        ) as query:
            checked, _ = hipblaslt_check._check_rows(rows, "gfx942", 80, 10)
        query.assert_called_once_with([1, 2, 3, 4], 10)
        self.assertEqual(
            [row["status"] for row in checked],
            ["match", "mismatch", "missing_name", "missing_name"],
        )
        self.assertEqual(checked[1]["live_kernelName"], "fp8_kernel")
        self.assertEqual([row["merged_row"] for row in checked], [2, 3, 4, 5])

    def test_bad_index_is_never_sent_to_native_query(self):
        rows = pd.DataFrame(
            {"solidx": ["-1", "2147483648", "1.5", ""], "kernelName": "saved"}
        ).assign(libtype="hipblaslt", gfx="gfx942", cu_num="80")
        with patch.object(
            hipblaslt_check, "_query_indices", return_value=({}, None)
        ) as query:
            checked, _ = hipblaslt_check._check_rows(rows, "gfx942", 80, 10)
        query.assert_called_once_with([], 10)
        self.assertTrue(all(row["status"] == "query_failed" for row in checked))
        with self.assertRaisesRegex(ValueError, "no hipBLASLt rows"):
            hipblaslt_check._check_rows(rows, "gfx950", 256, 10)

    def test_native_exit_keeps_completed_queries_and_retries_rest(self):
        # Real subprocesses with a fake native API. os._exit cannot be caught
        # by Python; this verifies containment, not hipBLASLt GPU correctness.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "aiter" / "jit").mkdir(parents=True)
            (root / "aiter" / "jit" / "__init__.py").write_text("")
            (root / "aiter" / "jit" / "core.py").write_text(
                "from types import SimpleNamespace\n"
                "def get_module(name):\n"
                "    return SimpleNamespace(hipblaslt_build_version=(1, 5, 0))\n"
            )
            (root / "aiter" / "__init__.py").write_text(
                "import os\n"
                "def hipb_create_extension(): pass\n"
                "def hipb_destroy_extension(): pass\n"
                "def getHipblasltKernelName(index):\n"
                "    if index == 2: os._exit(23)\n"
                "    return 'kernel_' + str(index)\n"
            )
            (root / "torch.py").write_text(
                "from types import SimpleNamespace\n"
                "__version__ = 'CPU stand-in'\n"
                "version = SimpleNamespace(hip='CPU stand-in')\n"
            )
            with patch.dict(os.environ, {"PYTHONPATH": tmp}):
                answers, environment = hipblaslt_check._query_indices([1, 2, 3, 1], 10)
        self.assertEqual(answers[1], {"name": "kernel_1", "error": None})
        self.assertEqual(answers[3], {"name": "kernel_3", "error": None})
        self.assertIsNone(answers[2]["name"])
        self.assertIn("exit=23", answers[2]["error"])
        self.assertEqual(environment["hipblaslt_build_version"], [1, 5, 0])

    def test_child_setup_timeout_and_protocol_failures(self):
        prefix = hipblaslt_check._PREFIX
        ready = prefix + '{"event":"ready","environment":{}}\n'
        query = prefix + '{"event":"query","index":1}\n'
        result = prefix + '{"event":"result","index":1,"name":"kernel_1"}\n'
        cases = (
            (subprocess.CompletedProcess([], 1, "", "initialization failed"), "exit=1"),
            (subprocess.TimeoutExpired([], 1, (ready + query).encode()), "timed out"),
            (subprocess.CompletedProcess([], 0, ready + query, ""), "exit=0"),
            (
                subprocess.CompletedProcess([], 0, prefix + "{broken\n", ""),
                "protocol_error=True",
            ),
            (
                subprocess.CompletedProcess(
                    [], 1, ready + query + result, "cleanup failed"
                ),
                "cleanup failed",
            ),
        )
        for response, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                kwargs = (
                    {"side_effect": response}
                    if isinstance(response, Exception)
                    else {"return_value": response}
                )
                with patch.object(hipblaslt_check.subprocess, "run", **kwargs) as run:
                    answers, _ = hipblaslt_check._query_indices([1], 1)
                self.assertEqual(run.call_count, 1)
                self.assertIsNone(answers[1]["name"])
                self.assertIn(diagnostic, answers[1]["error"])

    def test_cli_exit_codes_and_saved_report(self):
        rows = pd.DataFrame(
            [
                {
                    "libtype": "hipblaslt",
                    "gfx": "gfx942",
                    "cu_num": "80",
                    "solidx": "1",
                    "kernelName": "bf16",
                }
            ]
        )
        chip = SimpleNamespace(get_gfx_runtime=lambda: "gfx942", get_cu_num=lambda: 80)
        cases = (
            (rows, {"name": "bf16", "error": None}, 0),
            (rows, {"name": "fp8", "error": None}, 1),
            (rows.assign(kernelName=""), {"name": "bf16", "error": None}, 1),
            (rows, {"name": None, "error": "native exit=23"}, 1),
            (rows.assign(gfx="gfx950"), {"name": "bf16", "error": None}, 2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            for df, answer, expected in cases:
                with (
                    self.subTest(answer=answer, expected=expected),
                    patch.dict(sys.modules, {"aiter.jit.utils.chip_info": chip}),
                    patch.object(
                        hipblaslt_check, "_load_config", return_value=(df, {})
                    ),
                    patch.object(
                        hipblaslt_check,
                        "_query_indices",
                        return_value=({1: answer}, {}),
                    ),
                    redirect_stdout(io.StringIO()) as stdout,
                ):
                    code = hipblaslt_check.main(["--report", str(report_path)])
                self.assertEqual(code, expected)
                self.assertEqual(
                    json.loads(report_path.read_text()), json.loads(stdout.getvalue())
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
