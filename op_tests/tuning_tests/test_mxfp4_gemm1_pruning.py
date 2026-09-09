# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks for GEMM1 pruning rule coverage and unknown BM errors."""

import ast
import unittest
from pathlib import Path
from typing import Any


class TestMxfp4Gemm1Pruning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the actual pure predicate without importing the GPU tuner.
        root = Path(__file__).resolve().parents[2]
        tuner_path = root / "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
        tuner_module = ast.parse(tuner_path.read_text())
        tuner_class = next(
            node
            for node in tuner_module.body
            if isinstance(node, ast.ClassDef) and node.name == "Mxfp4FlydslTuner"
        )
        predicate = next(
            node
            for node in tuner_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_g1_matches_m_est"
        )
        predicate.decorator_list = []
        namespace = {"Any": Any}
        exec(  # noqa: S102 - execute trusted repository source without GPU imports
            compile(
                ast.Module(body=[predicate], type_ignores=[]), str(tuner_path), "exec"
            ),
            namespace,
        )
        cls.matches_m_est = staticmethod(namespace["_g1_matches_m_est"])

        kname_path = root / "aiter/ops/flydsl/mxfp4_kname.py"
        variants = next(
            node.value
            for node in ast.parse(kname_path.read_text()).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MXFP4_G1_VARIANTS"
                for target in node.targets
            )
        )
        cls.g1_variants = ast.literal_eval(variants)["fp4"]

    def test_unknown_bm_requires_pruning_rule(self):
        supported_bms = {variant[0] for variant in self.g1_variants}
        for bm in (min(supported_bms) // 2, max(supported_bms) * 2):
            for m_est in (1, 16, 64, 129):
                with self.subTest(bm=bm, m_est=m_est):
                    with self.assertRaisesRegex(ValueError, f"BM{bm}") as raised:
                        self.matches_m_est({"bm": bm}, m_est)
                    message = str(raised.exception)
                    self.assertIn("add an M_est pruning rule", message)
                    self.assertIn("Mxfp4FlydslTuner._g1_matches_m_est", message)

    def test_each_supported_family_has_a_retained_variant(self):
        for bm in sorted({variant[0] for variant in self.g1_variants}):
            with self.subTest(bm=bm):
                self.assertTrue(
                    any(
                        self.matches_m_est(
                            {"bm": bm, "num_waves": 4, "k_wave": 1, "use_nt": use_nt},
                            m_est,
                        )
                        for variant_bm, use_nt, _ in self.g1_variants
                        if variant_bm == bm
                        for m_est in (1, 4, 16, 32, 33, 64, 65, 128, 129)
                    )
                )

    def test_bm128_retention_boundary(self):
        self.assertFalse(self.matches_m_est({"bm": 128}, 63))
        self.assertTrue(self.matches_m_est({"bm": 128}, 64))


if __name__ == "__main__":
    unittest.main()
