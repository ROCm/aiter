# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks for the GFX942 CK-Tile blockscale tuning matrix."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

INSTANCE_MODULE = (
    Path(__file__).resolve().parents[2]
    / "csrc"
    / "ck_gemm_a8w8_blockscale"
    / "gemm_a8w8_blockscale_cktile_instance.py"
)


def load_instance_module(gfx):
    chip_info = types.ModuleType("chip_info")
    chip_info.get_gfx = lambda: gfx
    spec = importlib.util.spec_from_file_location(
        f"gemm_a8w8_blockscale_cktile_instance_{gfx}", INSTANCE_MODULE
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"chip_info": chip_info}):
        spec.loader.exec_module(module)
    return module


class TestGfx942CkTileInstances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = load_instance_module("gfx942")

    def test_base_tile_matrix(self):
        expected = {
            (128, 128, 128, 1, 4, 1),
            (16, 128, 256, 1, 4, 1),
            (16, 64, 512, 1, 4, 1),
            (32, 64, 512, 1, 4, 1),
            (64, 256, 128, 1, 4, 1),
            (48, 128, 256, 1, 4, 1),
            (16, 64, 256, 1, 4, 1),
            (64, 128, 256, 1, 4, 1),
            (64, 128, 128, 1, 4, 1),
            (48, 64, 256, 1, 4, 1),
            (128, 64, 128, 1, 4, 1),
            (64, 64, 128, 1, 4, 1),
            (32, 256, 128, 1, 4, 1),
            (32, 128, 128, 1, 4, 1),
            (32, 64, 128, 1, 4, 1),
            (32, 128, 256, 1, 4, 1),
            (32, 64, 256, 1, 4, 1),
            (64, 64, 256, 1, 4, 1),
            (128, 128, 128, 2, 2, 1),
            (64, 128, 128, 2, 2, 1),
        }
        actual = {
            (
                kernel.M_Tile,
                kernel.N_Tile,
                kernel.K_Tile,
                kernel.M_Warp,
                kernel.N_Warp,
                kernel.K_Warp,
            )
            for kernel in self.instances.kernels_list_942.values()
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 20)

    def test_expands_each_tile_across_block_per_cu(self):
        candidates = self.instances.candidate_kernels_cktile_dict.values()
        self.assertEqual(len(self.instances.candidate_kernels_cktile_dict), 80)

        expanded = {}
        for kernel in candidates:
            tile = tuple(
                value for field, value in vars(kernel).items() if field != "BlockPerCu"
            )
            expanded.setdefault(tile, set()).add(kernel.BlockPerCu)

        self.assertEqual(len(expanded), 20)
        self.assertTrue(all(blocks == {1, 2, 3, 4} for blocks in expanded.values()))

    def test_candidates_obey_gfx942_tuner_constraints(self):
        for kernel in self.instances.candidate_kernels_cktile_dict.values():
            with self.subTest(kernel=kernel.name):
                waves = kernel.M_Warp * kernel.N_Warp * kernel.K_Warp
                self.assertEqual(waves, 4)
                self.assertEqual(kernel.K_Tile % 128, 0)
                self.assertEqual(max(kernel.N_Tile, 128) % min(kernel.N_Tile, 128), 0)
                self.assertEqual(kernel.K_Warp_Tile, 64)
                self.assertTrue(kernel.TransposeC)
                self.assertFalse(kernel.AQRowMajor)

    def test_existing_kernel_ids_remain_stable(self):
        kernels = self.instances.kernels_list_942
        self.assertEqual((kernels[0].M_Tile, kernels[0].K_Tile), (128, 128))
        self.assertEqual((kernels[1].M_Tile, kernels[1].K_Tile), (16, 256))

    def test_base_ids_and_names_are_unique(self):
        kernels = self.instances.kernels_list_942
        self.assertEqual(list(kernels), list(range(len(kernels))))
        self.assertEqual(
            len({kernel.name for kernel in kernels.values()}), len(kernels)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
