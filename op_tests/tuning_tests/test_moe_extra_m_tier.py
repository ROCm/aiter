# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Extra tuned-row tiers of the fused-MoE lookup (``extra_m_tiers`` / ``tuned_M``).

A tuned fMoE row at a prefill token count between two ``get_padded_M`` buckets
(Kimi-K3.1's A4W4 row at 3584 tokens) is a lookup tier: 2049..3584 tokens key
on 3584 for the shapes that have a row there, and on the 4096 bucket otherwise.
Checks that the tier comes from the table (with and without a 3584 row), that
``_fused_moe_impl``, ``moev2.api.dispatch_key`` and the DP shared-expert path
key on the same function, and that a shape without a tier row gets exactly
its bucket's metadata.

Requires torch (importing aiter pulls it in). No GPU: ``GPU_ARCHS`` and
``CU_NUM`` stand in for the device.

Run:
    GPU_ARCHS=gfx950 CU_NUM=256 python3 -m unittest \
        op_tests.tuning_tests.test_moe_extra_m_tier -v
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("GPU_ARCHS", "gfx950")
os.environ.setdefault("CU_NUM", "256")

try:
    import aiter.fused_moe as fm
    import aiter.fused_moe_dp_shared_expert as dp
    from aiter import ActivationType, QuantType, dtypes
    from aiter.moev2.dispatch.tuned import extra_m_tiers
    from aiter.moev2.shapes import get_padded_M, tuned_M

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    _IMPORT_ERR = e

KIMI_A4W4 = Path(__file__).resolve().parents[2] / (
    "aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv"
)


def _table(path, tokens):
    """A tuned fMoE CSV holding one Kimi-K3.1 A4W4 row per token count."""
    with open(KIMI_A4W4) as f:
        rows = list(csv.DictReader(f))
    proto = next(r for r in rows if r["token"] == "4096")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(proto))
        w.writeheader()
        for t in tokens:
            w.writerow({**proto, "token": str(t)})


@unittest.skipIf(_IMPORT_ERR is not None, f"aiter import failed: {_IMPORT_ERR}")
class ExtraMTierTest(unittest.TestCase):
    def test_tiers_come_from_the_table(self):
        with tempfile.TemporaryDirectory() as d:
            with_row, without = f"{d}/with.csv", f"{d}/without.csv"
            _table(with_row, [2048, 3584, 4096, 8192])
            _table(without, [2048, 4096, 8192])
            self.assertEqual(extra_m_tiers(with_row), (3584,))
            self.assertEqual(extra_m_tiers(without), ())

    def test_decode_size_off_bucket_rows_are_not_tiers(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/t.csv"
            _table(path, [3, 40, 56, 168, 1000, 1024, 3000])
            self.assertEqual(extra_m_tiers(path), (3000,))

    def test_tuned_m(self):
        for m, want in (
            (2048, 2048),
            (2049, 3584),
            (3000, 3584),
            (3584, 3584),
            (3585, 4096),
        ):
            self.assertEqual(tuned_M(m, (3584,)), want, m)
            self.assertEqual(tuned_M(m, ()), get_padded_M(m), m)

    def test_fused_moe_and_dispatch_key_agree(self):
        from aiter.moev2 import api

        tiers = extra_m_tiers(fm.AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
        self.assertIn(3584, tiers, "the shipped Kimi-K3.1 A4W4 table has a 3584 row")
        for m in (1, 129, 2048, 2049, 3000, 3584, 3585, 4096, 16384):
            want = tuned_M(m, tiers)
            self.assertEqual(fm._tuned_M(m), want, m)
            call = mock.MagicMock()
            call.M = m
            with mock.patch.object(
                api, "resolve_activation_dtype", return_value=dtypes.fp4x2
            ), mock.patch.object(
                api,
                "_normalize_mxfp4_activation_params",
                return_value=(4.0, 25.0, None),
            ), mock.patch.object(
                api, "get_gfx_runtime", return_value="gfx950"
            ), mock.patch.object(
                api, "get_cu_num", return_value=256
            ):
                args = api.dispatch_key(call)[0]
            self.assertEqual(args[0], want, m)

    def test_dp_shared_expert_keys_on_its_own_table(self):
        tiers = extra_m_tiers(f"{dp.AITER_ROOT_DIR}/aiter/configs/tuned_fmoe.csv")
        for m in (2049, 3000, 3584, 4096):
            self.assertEqual(dp._tuned_token(m), tuned_M(m, tiers), m)

    def test_a_shape_without_a_tier_row_gets_its_bucket(self):
        """Kimi-K3.1 A8W4 (no 3584 row): token 3584 resolves exactly as 4096;
        Kimi-K3.1 A4W4 resolves its 3584 row (block_m 64)."""

        def cfg(token, qa):
            return fm.get_2stage_cfgs(
                token,
                3584,
                384 if qa == dtypes.fp4x2 else 512,
                896,
                16,
                dtypes.bf16,
                qa,
                dtypes.fp4x2,
                QuantType.per_1x32,
                True,
                ActivationType.Situv2,
                False,
                0,
                0,
                True,
                situ_beta=4.0,
                situ_linear_beta=1.0,
            )

        def desc(m):
            return (
                m.block_m,
                m.ksplit,
                m.run_1stage,
                repr(getattr(m.stage1, "keywords", m.stage1)),
                repr(getattr(m.stage2, "keywords", m.stage2)),
            )

        with mock.patch.object(
            fm, "get_gfx_runtime", return_value="gfx950"
        ), mock.patch.object(fm, "get_cu_num", return_value=256), mock.patch.object(
            fm, "get_gfx", return_value="gfx950"
        ):
            fm.get_2stage_cfgs.cache_clear()
            try:
                a8_tier, a8_bucket = cfg(3584, dtypes.fp8), cfg(4096, dtypes.fp8)
                a4_tier, a4_bucket = cfg(3584, dtypes.fp4x2), cfg(4096, dtypes.fp4x2)
            except Exception as e:  # noqa: BLE001
                self.skipTest(f"get_2stage_cfgs needs more of the environment: {e!r}")
            finally:
                fm.get_2stage_cfgs.cache_clear()
        self.assertEqual(desc(a8_tier), desc(a8_bucket))
        self.assertEqual(a4_tier.block_m, 64)
        self.assertNotEqual(desc(a4_tier), desc(a4_bucket))


if __name__ == "__main__":
    unittest.main()
