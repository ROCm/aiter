# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""load_hip_runtime() binds to the HIP runtime already mapped. CPU only.

Run: python -m pytest -q tests/test_vmm_allocator_load_hip.py
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


def cdll_calls(maps, load: bool = True, side_effect=None) -> list:
    """Import vmm_allocator with ``maps`` as /proc/self/maps (None: missing),
    call load_hip_runtime() if ``load``, and return the CDLL calls."""
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        if path == "/proc/self/maps":
            if maps is None:
                raise FileNotFoundError(path)
            data = os.fsencode(maps)
            return io.BytesIO(data) if "b" in mode else io.StringIO(data.decode())
        return real_open(path, mode, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("vmm_allocator_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch("builtins.open", fake_open), mock.patch.object(
        ctypes, "CDLL", side_effect=side_effect
    ) as cdll:
        spec.loader.exec_module(module)
        if load:
            module.load_hip_runtime()
    return cdll.call_args_list


class LoadHipRuntimeTest(unittest.TestCase):
    def test_binds_the_mapped_copy(self):
        self.assertEqual(cdll_calls(MAPS), [mock.call(CORE)])

    def test_binds_a_deleted_mapped_copy_by_path(self):
        self.assertEqual(
            cdll_calls(MAPS.replace(CORE, CORE + " (deleted)")), [mock.call(CORE)]
        )

    def test_binds_a_mapped_copy_with_escaped_path(self):
        path = "/opt/my venv\n/lib/libamdhip64.so.7"
        maps = MAPS.replace(CORE, "/opt/my venv\\012/lib/libamdhip64.so.7")
        self.assertEqual(cdll_calls(maps), [mock.call(path)])

    def test_skips_a_mapped_path_that_is_not_utf8(self):
        maps = "7f0000500000-7f0000600000 r--p 0 08:01 9 /data/caf\udce9.bin\n" + MAPS
        self.assertEqual(cdll_calls(maps), [mock.call(CORE)])

    def test_warns_and_falls_back_if_the_mapped_copy_fails_to_load(self):
        def cdll(name):
            if name == CORE:
                raise OSError(f"{CORE}: cannot open shared object file")
            return mock.DEFAULT

        with self.assertLogs("vmm_allocator_under_test", "WARNING") as logs:
            calls = cdll_calls(MAPS, side_effect=cdll)
        self.assertEqual(calls, [mock.call(CORE), mock.call("libamdhip64.so")])
        self.assertIn(CORE, logs.output[0])

    def test_falls_back_to_the_soname(self):
        self.assertEqual(cdll_calls("")[0], mock.call("libamdhip64.so"))

    def test_falls_back_to_the_soname_without_proc_maps(self):
        self.assertEqual(cdll_calls(None)[0], mock.call("libamdhip64.so"))

    def test_import_loads_nothing(self):
        self.assertEqual(cdll_calls(MAPS, load=False), [])


if __name__ == "__main__":
    unittest.main()
