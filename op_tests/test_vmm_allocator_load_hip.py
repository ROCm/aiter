# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""load_hip_runtime() binds to the HIP runtime already mapped. CPU only.

Run: python op_tests/test_vmm_allocator_load_hip.py
"""

import ctypes
import importlib.util
import io
import os
import unittest
from unittest import mock

PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "aiter",
    "dist",
    "device_communicators",
    "vmm_allocator.py",
)
CORE = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7"
MAPS = (
    "7f0000000000-7f0000100000 r--p 00000000 08:01 7 /usr/lib/x86_64-linux-gnu/libc.so.6\n"
    f"7f0000200000-7f0000300000 r-xp 00000000 08:01 42 {CORE}\n"
    "7f0000400000-7f0000401000 rw-p 00000000 00:00 0 \n"
)


def cdll_calls(maps: str) -> list:
    """Import vmm_allocator with ``maps`` as /proc/self/maps; return the CDLL calls."""
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/self/maps":
            return io.StringIO(maps)
        return real_open(path, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("vmm_allocator_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch("builtins.open", fake_open), mock.patch.object(
        ctypes, "CDLL"
    ) as cdll:
        spec.loader.exec_module(module)
    return cdll.call_args_list


class LoadHipRuntimeTest(unittest.TestCase):
    def test_binds_the_mapped_copy(self):
        self.assertEqual(cdll_calls(MAPS), [mock.call(CORE)])

    def test_binds_a_deleted_mapped_copy_by_path(self):
        self.assertEqual(
            cdll_calls(MAPS.replace(CORE, CORE + " (deleted)")), [mock.call(CORE)]
        )

    def test_falls_back_to_the_soname(self):
        self.assertEqual(cdll_calls("")[0], mock.call("libamdhip64.so"))


if __name__ == "__main__":
    unittest.main()
