# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Cross-file shape-collision guard for tuned config CSVs.

At runtime ``aiter.jit.core.AITER_CONFIGS.get_config_file`` merges, per family,
the canonical ``aiter/configs/<name>.csv`` with every
``aiter/configs/model_configs/*<name>*.csv``, then ``update_config_files``
de-duplicates on the family's lookup key (usually derived from the matching
*untuned* CSV's columns) and **raises** if two rows collide.

A single PR's CI only ever merges *its own* changed file with current ``main``,
so two PRs that each add the same shape to different model files both pass, then
break ``main`` once both land (cross-PR / merge-skew hazard). This test drives
the **real runtime merge** so the collision is caught statically -- there is no
re-implementation of the merge/dedup/key logic here, so it cannot drift.

How it stays side-effect free: ``update_config_files`` writes de-duplicated CSVs
back to their source paths when it finds collisions. We copy the entire
``aiter/configs/`` tree to a temp dir and point ``core.AITER_ROOT_DIR`` at it, so
all globbing, untuned-key lookups, and any write-backs hit the copy, never the
real repo. The merged-output directory is also redirected into the temp tree,
and config environment overrides are restored after each resolution.

Requires torch (importing ``aiter`` pulls it in); it does not need a GPU. The
level 0+1 tuning workflow runs it as a PR/main regression guard.

Run:
    python3 -m unittest op_tests.tuning_tests.test_config_shape_collision -v
"""

import csv
import os
import shutil
import sys
import tempfile
import unittest
from itertools import product
from unittest.mock import patch

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
    ("AITER_CONFIG_GEMM_A6W4_ASM", "a6w4_asm_tuned_gemm"),
    ("AITER_CONFIG_GEMM_A4W6_ASM", "a4w6_asm_tuned_gemm"),
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
    ("AITER_CONFIG_GEMM_MXFP8FP4", "asm_mxfp8fp4gemm"),
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
        cls._orig_output_dir = core._CONFIG_MERGE_OUTPUT_DIR
        core.AITER_ROOT_DIR = cls._tmp
        core._CONFIG_MERGE_OUTPUT_DIR = os.path.join(cls._tmp, "merged_configs")
        _cache_clear()

    @classmethod
    def tearDownClass(cls):
        core.AITER_ROOT_DIR = cls._orig_root
        core._CONFIG_MERGE_OUTPUT_DIR = cls._orig_output_dir
        _cache_clear()
        shutil.rmtree(cls._tmp, ignore_errors=True)

    @staticmethod
    def _resolve(root, env_name, name, *, via_property=True):
        """Drive the production (no-env) resolution path against `root`.
        Raises RuntimeError on a duplicate-shape collision."""
        default_file = os.path.join(root, "aiter", "configs", f"{name}.csv")
        with (
            patch.dict(os.environ),
            patch.object(core, "AITER_ROOT_DIR", root),
            patch.object(core, env_name, default_file),
        ):
            os.environ.pop(env_name, None)
            _cache_clear()
            try:
                if via_property:
                    return getattr(core.AITER_CONFIGS, f"{env_name}_FILE")
                return core.AITER_CONFIGS.get_config_file(env_name, default_file, name)
            finally:
                _cache_clear()

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

    def test_a6w6_blockscale(self):
        self._check_family("AITER_CONFIG_GEMM_A6W6", "a6w6_blockscale_tuned_gemm")

    def test_a6w4_asm(self):
        self._check_family("AITER_CONFIG_GEMM_A6W4_ASM", "a6w4_asm_tuned_gemm")

    def test_a4w6_asm(self):
        self._check_family("AITER_CONFIG_GEMM_A4W6_ASM", "a4w6_asm_tuned_gemm")

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

        # The generic config registry owns file discovery and merging; the
        # dedicated caller policy still owns public-kid normalization.
        from aiter.ops.opus import policy

        policy._load_mxscale_bmm_tuned.cache_clear()
        rows = policy._load_mxscale_bmm_tuned("opus")
        self.assertTrue(rows)
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(
            rows[("gfx950", 2, 1, 1024, 4096)]["kernelId"],
            8311,
            "legacy local OPUS kid 311 must become public global kid 8311",
        )
        self.assertEqual(
            rows[("gfx950", 8, 128, 1024, 4096)]["kernelId"],
            8653,
            "legacy local OPUS kid 653 must become public global kid 8653",
        )
        policy._load_mxscale_bmm_tuned.cache_clear()

    def test_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle(self):
        self._check_family(
            "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
            "batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned",
        )

    def test_bf16(self):
        self._check_family("AITER_CONFIG_GEMM_BF16", "bf16_tuned_gemm")

    def test_mxfp8fp4(self):
        self._check_family("AITER_CONFIG_GEMM_MXFP8FP4", "asm_mxfp8fp4gemm")

    def test_mxfp8fp4_merge_uses_lookup_key(self):
        # This family has no "tuned" token in its filename and no untuned
        # sibling. A changed kernel/split count must still collide on the same
        # lookup key, while distinct B types, A layouts and shapes coexist.
        names = ("asm_mxfp8fp4gemm", "renamed_mxfp8fp4gemm")
        env_name = "AITER_CONFIG_GEMM_MXFP8FP4"
        row = {
            "gfx": "gfx1250",
            "M": 512,
            "N": 2048,
            "K": 7168,
            "b_intype": "mxfp8",
            "a_preshuffle": 0,
            "outdtype": "torch.bfloat16",
            "splitK": 8,
            "kernelName": "kernel_a",
            "cu_num": 256,
            "_tag": "",
        }
        variants = (
            ({}, True),
            ({"splitK": 4}, True),
            ({"kernelName": "kernel_b"}, True),
            ({"cu_num": 304}, True),
            ({"_tag": "alternate"}, True),
            ({"b_intype": "mxfp4"}, False),
            ({"a_preshuffle": 1}, False),
            ({"M": 513}, False),
            ({"N": 4096}, False),
            ({"K": 8192}, False),
            ({"gfx": "gfx950"}, False),
            ({"outdtype": "torch.float16"}, False),
        )
        for name, (changes, collision) in product(names, variants):
            with (
                self.subTest(name=name, changes=changes),
                tempfile.TemporaryDirectory() as tmp,
            ):
                cfg = os.path.join(tmp, "aiter", "configs")
                model = os.path.join(cfg, "model_configs", f"selfcheck_{name}.csv")
                os.makedirs(os.path.dirname(model))
                for path, saved in (
                    (os.path.join(cfg, f"{name}.csv"), row),
                    (model, dict(row, **changes)),
                ):
                    with open(path, "w") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(saved)
                try:
                    if collision:
                        with self.assertRaisesRegex(RuntimeError, "duplicate shape"):
                            self._resolve(
                                tmp, env_name, name, via_property=name == names[0]
                            )
                    else:
                        merged = self._resolve(
                            tmp, env_name, name, via_property=name == names[0]
                        )
                        self.assertEqual(
                            os.path.dirname(merged), core._CONFIG_MERGE_OUTPUT_DIR
                        )
                        with open(merged) as f:
                            self.assertEqual(len(list(csv.DictReader(f))), 2)
                finally:
                    core.AITER_ROOT_DIR = self._tmp
                    _cache_clear()

    def test_mxfp8fp4_rejects_missing_or_empty_strings_before_merge(self):
        row = {
            "gfx": "gfx1250",
            "M": 512,
            "N": 2048,
            "K": 7168,
            "b_intype": "mxfp8",
            "a_preshuffle": 0,
            "outdtype": "torch.bfloat16",
            "splitK": 8,
            "kernelName": "kernel_a",
            "us": 10,
        }
        for column, invalid in product(
            ("gfx", "b_intype", "outdtype", "kernelName"), (None, "", " ", 0)
        ):
            with (
                self.subTest(column=column, invalid=invalid),
                tempfile.TemporaryDirectory() as tmp,
            ):
                paths = [
                    os.path.join(tmp, name) for name in ("valid.csv", "invalid.csv")
                ]
                bad = dict(row, us=1)
                if invalid is None:
                    del bad[column]
                else:
                    bad[column] = invalid
                for path, saved in zip(paths, (row, bad)):
                    with open(path, "w") as f:
                        writer = csv.DictWriter(f, fieldnames=list(saved))
                        writer.writeheader()
                        writer.writerow(saved)
                with self.assertLogs(core.logger, level="WARNING") as logs:
                    merged = core.AITER_CONFIGS.update_config_files(
                        os.pathsep.join(paths),
                        "string_validation",
                        config_name="AITER_CONFIG_GEMM_MXFP8FP4",
                    )
                self.assertIn(paths[1], "\n".join(logs.output))
                self.assertIn(column, "\n".join(logs.output))
                with open(merged) as f:
                    saved = list(csv.DictReader(f))
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["kernelName"], row["kernelName"])
                self.assertEqual(float(saved[0]["us"]), 10)

    def test_resolve_restores_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_name, name = self._build_synthetic_family(tmp, dup=False)
            with patch.dict(os.environ, {env_name: "/user/config.csv"}):
                self._resolve(tmp, env_name, name)
                self.assertEqual(os.environ[env_name], "/user/config.csv")

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
