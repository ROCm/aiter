# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Launcher addresses keep the base pointer of a pitched scale plane.

Compact FP4 scales are a strided window on the dispatch wire. The grouped
GEMM launcher has to hand the kernel that base address as a rank-1 tensor
without packing the pitch out of the buffer.

No GPU. Importing the launcher pulls torch in.

Run:
    python3 -m unittest op_tests.tuning_tests.test_grouped_gemm_launch_address -v
"""

import os
import unittest

os.environ.setdefault("GPU_ARCHS", "gfx1250")
os.environ.setdefault("CU_NUM", "256")

try:
    import torch

    from aiter.ops.flydsl.grouped_gemm_mxfp4 import _launch_address

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    torch = _launch_address = None
    _IMPORT_ERR = e


class LaunchAddressTest(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERR is not None:
            self.skipTest(f"import failed: {_IMPORT_ERR}")

    def test_contiguous_buffer_flattens(self):
        weights = torch.arange(16, dtype=torch.int32)
        flat = _launch_address(weights, torch.int32)
        self.assertEqual(flat.shape, (16,))
        self.assertTrue(torch.equal(flat, weights))
        self.assertEqual(flat.data_ptr(), weights.data_ptr())

    def test_pitched_scale_keeps_base_address(self):
        # 8 scale rows of 224 B on a 512 B wire pitch, matching a compact
        # FP4 dispatch window that view(-1) rejects.
        storage = torch.arange(8 * 512, dtype=torch.uint8)
        scales = torch.as_strided(storage, (8, 224), (512, 1))
        with self.assertRaises(RuntimeError):
            scales.view(torch.int32).view(-1)
        flat = _launch_address(scales, torch.int32)
        self.assertEqual(flat.ndim, 1)
        self.assertEqual(flat.dtype, torch.int32)
        self.assertTrue(flat.is_contiguous())
        self.assertEqual(flat.data_ptr(), scales.data_ptr())
        self.assertEqual(flat.view(torch.uint8)[:224].tolist(), list(range(224)))
