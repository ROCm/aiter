# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import unittest
from unittest.mock import patch

from aiter.test_common import _use_cuda_event_timing


class TestCudaEventTimingSelection(unittest.TestCase):

    def test_explicit_request_enables_event_timing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_use_cuda_event_timing(True))

    def test_environment_enables_event_timing_in_workers(self):
        with patch.dict(
            os.environ, {"AITER_PERFTEST_USE_CUDA_EVENT": "1"}, clear=True
        ):
            self.assertTrue(_use_cuda_event_timing(False))

    def test_profiler_remains_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_use_cuda_event_timing(False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
