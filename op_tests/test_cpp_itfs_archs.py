# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from csrc.cpp_itfs.utils import get_gpu_arch, validate_and_update_archs


class TestCppItfsArchs(unittest.TestCase):
    def setUp(self):
        get_gpu_arch.cache_clear()

    def tearDown(self):
        get_gpu_arch.cache_clear()

    def test_gpu_archs_override_is_accepted(self):
        for gfx in ("gfx942", "gfx950", "gfx1101", "gfx1151"):
            with (
                self.subTest(gfx=gfx),
                mock.patch.dict(os.environ, {"GPU_ARCHS": gfx}, clear=True),
            ):
                get_gpu_arch.cache_clear()
                self.assertEqual(validate_and_update_archs(), [gfx])

    def test_unsupported_architectures_still_raise(self):
        with (
            mock.patch.dict(os.environ, {"GPU_ARCHS": "gfx9999"}, clear=True),
            self.assertRaisesRegex(AssertionError, "invalid or not supported"),
        ):
            validate_and_update_archs()


if __name__ == "__main__":
    unittest.main()
