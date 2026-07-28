# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Cross-file shape-collision guard for tuned config CSVs.

At runtime ``aiter.jit.core.AITER_CONFIGS.get_config_file`` merges, per family,
the canonical ``aiter/configs/<name>.csv`` with every
``aiter/configs/model_configs/*<name>*.csv``, then ``update_config_files``
de-duplicates on a key derived from the matching *untuned* CSV's columns and
**raises** if two rows collide.

A single PR's CI only ever merges *its own* changed file with current ``main``,
so two PRs that each add the same shape to different model files both pass, then
break ``main`` once both land (cross-PR / merge-skew hazard). This test drives
the **real runtime merge** so the collision is caught statically -- there is no
re-implementation of the merge/dedup/key logic here, so it cannot drift.

How it stays side-effect free: ``update_config_files`` writes de-duplicated CSVs
back to their source paths when it finds collisions. We copy the entire
``aiter/configs/`` tree to a temp dir and point ``core.AITER_ROOT_DIR`` at it, so
all globbing, untuned-key lookups, and any write-backs hit the copy, never the
real repo.

Requires torch (importing ``aiter`` pulls it in); it does not need a GPU. It is
**not yet wired into any CI workflow** -- run it manually in a torch-enabled
environment, or add it to a suitable job (e.g. the CPU/level01 tuning tests) to
make it an actual PR/main regression guard.

Run:
    python3 -m unittest op_tests.tuning_tests.test_config_shape_collision -v
"""

import os
import shutil
import sys
import tempfile
import unittest

try:  # importing aiter requires torch; skip cleanly where it is unavailable.
    import aiter.jit.core as core

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
    ("AITER_CONFIG_GEMM_A8W8", "a8w8_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE", "a8w8_bpreshuffle_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", "a8w8_blockscale_tuned_gemm"),
    (
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
        "a8w8_blockscale_bpreshuffle_tuned_gemm",
    ),
    ("AITER_CONFIG_A8W8_BATCHED_GEMM", "a8w8_tuned_batched_gemm"),
    ("AITER_CONFIG_BF16_BATCHED_GEMM", "bf16_tuned_batched_gemm"),
    ("AITER_CONFIG_GEMM_BF16", "bf16_tuned_gemm"),
    ("AITER_CONFIG_FMOE", "tuned_fmoe"),
    ("AITER_CONFIG_GROUPED_FMOE", "tuned_grouped_fmoe"),
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

    def test_selfcheck_detects_planted_duplicate(self):
        """Positive control: a planted duplicate MUST be caught. If not, the
        detection harness (temp copy / AITER_ROOT_DIR redirect / merge call) is
        broken -- not the real config data."""
        err = self._run_synthetic(dup=True)
        self.assertIsNotNone(
            err,
            "harness FAILED to detect a planted duplicate shape -- the collision "
            "check is broken; do not trust its PASS on real configs.",
        )
        self.assertIn("duplicate shape", err.lower())

    def test_selfcheck_passes_on_clean(self):
        """Negative control: distinct shapes must NOT be flagged (no false
        positive)."""
        err = self._run_synthetic(dup=False)
        self.assertIsNone(err, f"harness false-positived on clean configs:\n{err}")

    def test_a4w4_blockscale(self):
        self._check_family("AITER_CONFIG_GEMM_A4W4", "a4w4_blockscale_tuned_gemm")

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

    def test_bf16(self):
        self._check_family("AITER_CONFIG_GEMM_BF16", "bf16_tuned_gemm")

    def test_fmoe(self):
        self._check_family("AITER_CONFIG_FMOE", "tuned_fmoe")

    def test_grouped_fmoe(self):
        self._check_family("AITER_CONFIG_GROUPED_FMOE", "tuned_grouped_fmoe")


def _fix_real_tree():
    """Resolve every family against the REAL checkout (not a temp copy) so
    `update_config_files`' existing auto-dedup (keep lowest-`us` per shape) writes
    the pruned CSVs back to the actual source files. Prints what changed; commit
    the result and re-run without --fix to confirm clean.

    This adds NO dedup logic -- it just triggers the write-back that
    aiter/jit/core.py::update_config_files already performs, on real files."""
    if core is None:
        raise SystemExit(f"aiter.jit.core not importable: {_IMPORT_ERR}")
    core.AITER_ROOT_DIR = AITER_ROOT  # operate on this checkout's real configs
    fixed = []
    for env_name, name in FAMILIES:
        os.environ.pop(env_name, None)
        _cache_clear()
        default_file = os.path.join(AITER_ROOT, "aiter", "configs", f"{name}.csv")
        try:
            core.AITER_CONFIGS.get_config_file(env_name, default_file, name)
        except RuntimeError as e:
            if "duplicate shape" in str(e).lower():
                fixed.append((name, str(e)))
            else:
                raise
    if not fixed:
        print("No duplicate shapes found; nothing to fix.")
        return
    print(f"Resolved duplicate shapes in {len(fixed)} family(ies):\n")
    for name, msg in fixed:
        print(f"### {name}\n{msg}\n")
    print(
        "Source CSVs were rewritten (lowest-`us` row kept per shape). "
        "Review `git diff`, commit, then re-run without --fix to confirm clean."
    )


if __name__ == "__main__":
    if "--fix" in sys.argv:
        # Modify the REAL config files in place. Read-only detection (the default)
        # runs on a temp copy; --fix intentionally writes back to the checkout.
        sys.argv.remove("--fix")
        _fix_real_tree()
    else:
        unittest.main(verbosity=2)
