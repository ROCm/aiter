# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import unittest
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JIT_UTILS = os.path.join(REPO_ROOT, "aiter", "jit", "utils")
sys.path.insert(0, JIT_UTILS)

from build_targets import get_build_targets_env  # noqa: E402
from blob_gen import windows_blob_gen_argv  # noqa: E402
from mha_recipes import _ck_targets_flag_for_arch  # noqa: E402


WINDOWS_RDNA_TARGETS = {
    "gfx1100": 96,
    "gfx1101": 60,
    "gfx1102": 32,
    "gfx1103": 12,
    "gfx1151": 20,
    "gfx1201": 64,
}


class TestWindowsRDNACKTargets(unittest.TestCase):
    def test_offline_build_target_defaults(self):
        for gfx, cu_num in WINDOWS_RDNA_TARGETS.items():
            with self.subTest(gfx=gfx), mock.patch.dict(
                os.environ, {"GPU_ARCHS": gfx}, clear=True
            ):
                self.assertEqual(get_build_targets_env(), [(gfx, cu_num)])

    def test_ck_codegen_receives_each_rdna_target(self):
        for gfx in WINDOWS_RDNA_TARGETS:
            with self.subTest(gfx=gfx):
                self.assertEqual(
                    _ck_targets_flag_for_arch(gfx), f" --targets {gfx}"
                )

    def test_cdna_keeps_ck_generator_defaults(self):
        self.assertEqual(_ck_targets_flag_for_arch("gfx942"), "")
        self.assertEqual(_ck_targets_flag_for_arch("gfx950"), "")

    def test_blob_generator_paths_with_spaces_are_single_arguments(self):
        argv = windows_blob_gen_argv(
            r"C:\Program Files\Python\python.exe",
            (
                r"C:\Work Tree\aiter\3rdparty\composable_kernel"
                r"\example\ck_tile\01_fmha\generate.py "
                '-d fwd --receipt 100 --filter " @ " --output_dir {}'
            ),
            r"C:\Work Tree\build\blob",
        )

        self.assertEqual(
            argv,
            [
                r"C:\Program Files\Python\python.exe",
                (
                    r"C:\Work Tree\aiter\3rdparty\composable_kernel"
                    r"\example\ck_tile\01_fmha\generate.py"
                ),
                "-d",
                "fwd",
                "--receipt",
                "100",
                "--filter",
                " @ ",
                "--output_dir",
                r"C:\Work Tree\build\blob",
            ],
        )

    def test_blob_generator_equals_path_with_spaces(self):
        argv = windows_blob_gen_argv(
            r"C:\Python\python.exe",
            (
                r"C:\Work Tree\gen.py --working_path {} "
                r"--compiled_kids_sidecar=C:\Work Tree\compiled.json"
            ),
            r"C:\Work Tree\blob",
        )

        self.assertEqual(
            argv,
            [
                r"C:\Python\python.exe",
                r"C:\Work Tree\gen.py",
                "--working_path",
                r"C:\Work Tree\blob",
                "--compiled_kids_sidecar",
                r"C:\Work Tree\compiled.json",
            ],
        )


if __name__ == "__main__":
    unittest.main()
