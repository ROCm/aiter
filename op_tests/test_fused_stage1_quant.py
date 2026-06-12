"""Unit tests for the opt-in fused stage1 activation-quant kernel selection.

Covers ``_maybe_fuse_stage1_quant`` / ``_get_proven_fp4_stage1`` in
``aiter.fused_moe``. These exercise the pure kernel-name selection logic
(CSV-backed proven-sibling lookup + env gating) and require no GPU.
"""

import os
import tempfile
import unittest
from unittest import mock

from aiter import dtypes
import aiter.fused_moe as fm
from aiter.fused_moe import _get_proven_fp4_stage1, _maybe_fuse_stage1_quant


def _write_csv(rows):
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write("kernelName1\n")
        for r in rows:
            f.write(r + "\n")
    return path


class TestFuseStage1Quant(unittest.TestCase):
    def setUp(self):
        # Cache is keyed by tune_file path; clear so each test sees fresh state.
        fm._proven_fp4_stage1_cache.clear()
        self._files = []

    def tearDown(self):
        for p in self._files:
            try:
                os.remove(p)
            except OSError:
                pass

    def _csv(self, rows):
        p = _write_csv(rows)
        self._files.append(p)
        return p

    # ---- _get_proven_fp4_stage1 --------------------------------------------

    def test_get_proven_collects_only_fp4_stage1(self):
        f = self._csv(
            [
                "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2_fp4",  # kept
                "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2_bnt0",  # not _fp4
                "flydsl_moe2_afp4_wfp4_bf16_atomic_persist_fp4",  # moe2, not moe1
                "cktile_moe1_something_fp4",  # not flydsl_moe1
            ]
        )
        self.assertEqual(
            _get_proven_fp4_stage1(f),
            {"flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2_fp4"},
        )

    def test_get_proven_is_cached(self):
        f = self._csv(["flydsl_moe1_x_fp4"])
        first = _get_proven_fp4_stage1(f)
        self.assertIn(f, fm._proven_fp4_stage1_cache)
        # Mutating the file after caching must not change the cached result.
        with open(f, "a") as fh:
            fh.write("flydsl_moe1_y_fp4\n")
        self.assertEqual(_get_proven_fp4_stage1(f), first)

    def test_get_proven_missing_file_is_empty(self):
        self.assertEqual(_get_proven_fp4_stage1("/no/such/tune_file.csv"), set())

    # ---- _maybe_fuse_stage1_quant gating -----------------------------------

    def test_flag_off_is_noop(self):
        kn = "flydsl_moe1_a_t64"
        f = self._csv([kn + "_fp4"])
        with mock.patch.dict(os.environ, {"AITER_FUSE_STAGE1_QUANT": "0"}):
            self.assertEqual(_maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f), kn)

    def test_flag_on_upgrades_when_proven(self):
        kn = "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2"
        f = self._csv([kn + "_fp4"])
        with mock.patch.dict(os.environ, {"AITER_FUSE_STAGE1_QUANT": "1"}):
            self.assertEqual(
                _maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f), kn + "_fp4"
            )

    def test_flag_on_drops_bnt0_suffix(self):
        base = "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2"
        kn = base + "_bnt0"
        f = self._csv([base + "_fp4"])  # fused variant tuned without _bnt0
        with mock.patch.dict(os.environ, {"AITER_FUSE_STAGE1_QUANT": "1"}):
            self.assertEqual(
                _maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f), base + "_fp4"
            )

    def test_no_proven_sibling_unchanged(self):
        kn = "flydsl_moe1_a_t64"
        f = self._csv(["flydsl_moe1_other_t64_fp4"])  # different kernel
        with mock.patch.dict(os.environ, {"AITER_FUSE_STAGE1_QUANT": "1"}):
            self.assertEqual(_maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f), kn)

    def test_empty_tune_file_unchanged(self):
        kn = "flydsl_moe1_a_t64"
        f = self._csv([])
        with mock.patch.dict(os.environ, {"AITER_FUSE_STAGE1_QUANT": "1"}):
            self.assertEqual(_maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f), kn)

    def test_non_fp4_activation_dtype_unchanged(self):
        kn = "flydsl_moe1_a_t64"
        f = self._csv([kn + "_fp4"])
        with mock.patch.dict(os.environ, {"AITER_FUSE_STAGE1_QUANT": "1"}):
            self.assertEqual(_maybe_fuse_stage1_quant(kn, dtypes.bf16, f), kn)

    def test_already_fused_unchanged(self):
        kn = "flydsl_moe1_a_t64_fp4"
        f = self._csv([kn])
        with mock.patch.dict(os.environ, {"AITER_FUSE_STAGE1_QUANT": "1"}):
            self.assertEqual(_maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f), kn)

    def test_non_flydsl_moe1_kernel_unchanged(self):
        kn = "cktile_moe1_a_t64"
        f = self._csv(["flydsl_moe1_a_t64_fp4"])
        with mock.patch.dict(os.environ, {"AITER_FUSE_STAGE1_QUANT": "1"}):
            self.assertEqual(_maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f), kn)

    def test_min_tokens_gate(self):
        kn = "flydsl_moe1_a_t64"
        f = self._csv([kn + "_fp4"])
        with mock.patch.dict(
            os.environ,
            {"AITER_FUSE_STAGE1_QUANT": "1", "AITER_FUSE_STAGE1_MIN_TOKENS": "256"},
        ):
            # Below threshold: keep stock kernel (small-M tail).
            self.assertEqual(
                _maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f, token=128), kn
            )
            # At/above threshold: upgrade to fused-quant sibling.
            self.assertEqual(
                _maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f, token=256), kn + "_fp4"
            )

    def test_min_tokens_zero_means_no_gate(self):
        kn = "flydsl_moe1_a_t64"
        f = self._csv([kn + "_fp4"])
        with mock.patch.dict(
            os.environ,
            {"AITER_FUSE_STAGE1_QUANT": "1", "AITER_FUSE_STAGE1_MIN_TOKENS": "0"},
        ):
            self.assertEqual(
                _maybe_fuse_stage1_quant(kn, dtypes.fp4x2, f, token=1), kn + "_fp4"
            )


if __name__ == "__main__":
    unittest.main()
