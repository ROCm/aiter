# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise runtime tuned-config merging against isolated copies.

Duplicate shapes with timings are resolved in memory; source files remain
unchanged. This module no longer rewrites repository configuration files.
"""

import csv
import os
import shutil
import tempfile
import unittest

try:  # importing aiter requires torch; skip cleanly where it is unavailable.
    from aiter.jit import core

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    core = None
    _IMPORT_ERR = e

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# (env var name, tuned-file base name) for every family registered in
# AITER_CONFIGS.*_FILE (aiter/jit/core.py). These are the families merged at
# runtime; the merge set and dedup key are resolved entirely by get_config_file.
FAMILIES = [
    ("AITER_CONFIG_GEMM_A4W4", "a4w4_blockscale_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A6W6", "a6w6_blockscale_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A8W8", "a8w8_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE", "a8w8_bpreshuffle_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", "a8w8_blockscale_tuned_gemm"),
    (
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
        "a8w8_blockscale_bpreshuffle_tuned_gemm",
    ),
    ("AITER_CONFIG_A8W8_BATCHED_GEMM", "a8w8_tuned_batched_gemm"),
    ("AITER_CONFIG_BF16_BATCHED_GEMM", "bf16_tuned_batched_gemm"),
    (
        "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE",
        "batched_gemm_a8w8_blockscale_mxscale_tuned",
    ),
    (
        "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
        "batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned",
    ),
    ("AITER_CONFIG_GEMM_BF16", "bf16_tuned_gemm"),
    ("AITER_CONFIG_FMOE", "tuned_fmoe"),
    ("AITER_CONFIG_FHMOE", "tuned_fhmoe"),
    ("AITER_CONFIG_GROUPED_FMOE", "tuned_grouped_fmoe"),
    ("AITER_CONFIG_GDN_K5_OPT", "chunk_gdn_h_opt_tuned"),
]


def _cache_clear():
    # get_config_file is an lru_cache-wrapped method; clear via the class object.
    type(core.AITER_CONFIGS).get_config_file.cache_clear()


@unittest.skipUnless(core is not None, f"aiter.jit.core not importable: {_IMPORT_ERR}")
class TestConfigShapeCollision(unittest.TestCase):
    """Drive the real runtime merge against a temp copy; fail on collisions."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="aiter_cfg_collision_")
        shutil.copytree(
            os.path.join(AITER_ROOT, "aiter", "configs"),
            os.path.join(cls._tmp, "aiter", "configs"),
        )
        cls._orig_root = core.AITER_ROOT_DIR
        core.AITER_ROOT_DIR = cls._tmp
        _cache_clear()

    @classmethod
    def tearDownClass(cls):
        core.AITER_ROOT_DIR = cls._orig_root
        _cache_clear()
        shutil.rmtree(cls._tmp, ignore_errors=True)

    @staticmethod
    def _resolve(root, env_name, name):
        """Drive the production (no-env) resolution path against `root`.
        Raises RuntimeError on a duplicate-shape collision."""
        os.environ.pop(env_name, None)
        core.AITER_ROOT_DIR = root
        _cache_clear()
        default_file = os.path.join(root, "aiter", "configs", f"{name}.csv")
        return core.AITER_CONFIGS.get_config_file(env_name, default_file, name)

    def _check_family(self, env_name, name):
        try:
            self._resolve(self._tmp, env_name, name)
        except RuntimeError as e:
            if "duplicate shape" in str(e).lower():
                self.fail(
                    f"{name}: runtime merge of configs/ + model_configs/ reports "
                    f"duplicate shapes (same key across files):\n{e}"
                )
            raise

    # ---- self-check / control: prove the harness itself detects collisions ----

    @staticmethod
    def _build_synthetic_family(root, dup):
        """Write a minimal isolated config tree for one family and return
        (env_name, name). With dup=True the model file duplicates the canonical
        row's key; with dup=False it uses a distinct shape. Uses the
        a8w8_blockscale family (untuned key = M,N,K)."""
        name = "a8w8_blockscale_tuned_gemm"
        env_name = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
        cfg = os.path.join(root, "aiter", "configs")
        os.makedirs(os.path.join(cfg, "model_configs"), exist_ok=True)
        # untuned header drives the dedup key columns.
        with open(os.path.join(cfg, "a8w8_blockscale_untuned_gemm.csv"), "w") as f:
            f.write("M,N,K\n")
        header = "gfx,cu_num,M,N,K,us\n"
        with open(os.path.join(cfg, f"{name}.csv"), "w") as f:
            f.write(header)
            f.write("gfx950,256,1,64,128,10.0\n")
        model_shape = "1,64,128" if dup else "2,64,128"
        with open(
            os.path.join(cfg, "model_configs", f"selfcheck_{name}.csv"), "w"
        ) as f:
            f.write(header)
            f.write(f"gfx950,256,{model_shape},20.0\n")
        return env_name, name

    def _run_synthetic(self, dup):
        tmp = tempfile.mkdtemp(prefix="aiter_cfg_selfcheck_")
        try:
            env_name, name = self._build_synthetic_family(tmp, dup=dup)
            try:
                self._resolve(tmp, env_name, name)
                return None
            except RuntimeError as e:
                return str(e)
        finally:
            core.AITER_ROOT_DIR = self._tmp  # restore for other tests
            _cache_clear()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_selfcheck_resolves_planted_duplicate(self):
        self.assertIsNone(self._run_synthetic(dup=True))

    def test_selfcheck_passes_on_clean(self):
        """Negative control: distinct shapes must NOT be flagged (no false
        positive)."""
        err = self._run_synthetic(dup=False)
        self.assertIsNone(err, f"harness false-positived on clean configs:\n{err}")

    def test_a4w4_blockscale(self):
        self._check_family("AITER_CONFIG_GEMM_A4W4", "a4w4_blockscale_tuned_gemm")

    def test_a6w6_blockscale(self):
        self._check_family("AITER_CONFIG_GEMM_A6W6", "a6w6_blockscale_tuned_gemm")

    def test_a8w8(self):
        self._check_family("AITER_CONFIG_GEMM_A8W8", "a8w8_tuned_gemm")

    def test_a8w8_bpreshuffle(self):
        self._check_family(
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE", "a8w8_bpreshuffle_tuned_gemm"
        )

    def test_a8w8_blockscale(self):
        self._check_family(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", "a8w8_blockscale_tuned_gemm"
        )

    def test_a8w8_blockscale_bpreshuffle(self):
        self._check_family(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
            "a8w8_blockscale_bpreshuffle_tuned_gemm",
        )

    def test_a8w8_batched(self):
        self._check_family("AITER_CONFIG_A8W8_BATCHED_GEMM", "a8w8_tuned_batched_gemm")

    def test_bf16_batched(self):
        self._check_family("AITER_CONFIG_BF16_BATCHED_GEMM", "bf16_tuned_batched_gemm")

    def test_batched_gemm_a8w8_blockscale_mxscale(self):
        self._check_family(
            "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE",
            "batched_gemm_a8w8_blockscale_mxscale_tuned",
        )

    def test_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle(self):
        self._check_family(
            "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
            "batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned",
        )

    def test_bf16(self):
        self._check_family("AITER_CONFIG_GEMM_BF16", "bf16_tuned_gemm")

    def test_fmoe(self):
        self._check_family("AITER_CONFIG_FMOE", "tuned_fmoe")

    def test_fhmoe(self):
        merged = self._resolve(
            self._tmp,
            "AITER_CONFIG_FHMOE",
            "tuned_fhmoe",
        )
        with open(merged, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            {int(row["token"]) for row in rows},
            {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048},
        )

    def test_grouped_fmoe(self):
        self._check_family("AITER_CONFIG_GROUPED_FMOE", "tuned_grouped_fmoe")

    def test_gdn_k5_opt(self):
        self._check_family("AITER_CONFIG_GDN_K5_OPT", "chunk_gdn_h_opt_tuned")


if __name__ == "__main__":
    unittest.main(verbosity=2)
