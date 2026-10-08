# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from aiter.jit.core import get_args_of_build
from aiter.jit.utils.build_targets import (
    get_build_targets_env,
    torch_processor_count_to_cu,
)
from aiter.jit.utils.mha_recipes import (
    _apply_ck_mha_prebuild_target_policy,
    _ck_targets_flag,
    _ck_targets_flag_for_arches,
    _require_ck_batch_prefill_targets,
)


WINDOWS_RDNA_TARGETS = {
    "gfx1100": 96,
    "gfx1101": 60,
    "gfx1102": 32,
    "gfx1103": 12,
    "gfx1151": 40,
    "gfx1201": 64,
}


class TestWindowsRDNACKTargets(unittest.TestCase):
    def test_offline_build_target_defaults(self):
        for gfx, cu_num in WINDOWS_RDNA_TARGETS.items():
            with self.subTest(gfx=gfx), mock.patch.dict(
                os.environ, {"GPU_ARCHS": gfx}, clear=True
            ):
                self.assertEqual(get_build_targets_env(), [(gfx, cu_num)])

    def test_ck_codegen_receives_rdna_target(self):
        for gfx in WINDOWS_RDNA_TARGETS:
            with self.subTest(gfx=gfx):
                self.assertEqual(_ck_targets_flag_for_arches([gfx]), f" --targets {gfx}")

    def test_ck_codegen_receives_every_build_arch(self):
        self.assertEqual(
            _ck_targets_flag_for_arches(["gfx1100", "gfx1151"]),
            " --targets gfx1100,gfx1151",
        )
        self.assertEqual(_ck_targets_flag_for_arches(["gfx942", "gfx950"]), "")
        chip_info = SimpleNamespace(get_gfx_list=lambda: ["gfx942", "gfx1100"])
        with mock.patch.dict(sys.modules, {"chip_info": chip_info}):
            self.assertEqual(_ck_targets_flag(), " --targets gfx942,gfx1100")

    def test_batch_prefill_rejects_non_gfx9_architectures(self):
        _require_ck_batch_prefill_targets(["gfx942", "gfx950"])
        with self.assertRaisesRegex(RuntimeError, "gfx1100.*gfx1151"):
            _require_ck_batch_prefill_targets(["gfx942", "gfx1100", "gfx1151"])

    def test_static_prebuild_recipes_use_targets_and_skip_rdna_batch_prefill(self):
        config_path = Path(__file__).parents[1] / "aiter" / "jit" / "optCompilerConfig.json"
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        names = (
            "module_mha_fwd",
            "module_mha_varlen_fwd",
            "module_mha_batch_prefill",
            "libmha_fwd",
        )
        self.assertTrue(all(config[name]["blob_gen_cmd"] for name in names))
        static_build_args = [
            {**get_args_of_build(name), "md_name": name} for name in names
        ]

        for gfxs in (["gfx1151"], ["gfx942", "gfx1151"]):
            with self.subTest(gfxs=gfxs):
                recipes = _apply_ck_mha_prebuild_target_policy(
                    static_build_args, list(gfxs)
                )
                by_name = {recipe["md_name"]: recipe for recipe in recipes}
                targets = ",".join(gfxs)
                for name in ("module_mha_fwd", "module_mha_varlen_fwd", "libmha_fwd"):
                    for command in by_name[name]["blob_gen_cmd"]:
                        if "-d fwd " in command or "-d fwd_splitkv " in command:
                            self.assertIn(f"--targets {targets}", command)
                self.assertNotIn("module_mha_batch_prefill", by_name)
                self.assertFalse(
                    any(
                        "-d batch_prefill " in command
                        for command in by_name["libmha_fwd"]["blob_gen_cmd"]
                    )
                )

        static_by_name = dict(zip(names, static_build_args))
        gfx9_recipes = {
            recipe["md_name"]: recipe
            for recipe in _apply_ck_mha_prebuild_target_policy(
                static_build_args, ["gfx942", "gfx950"]
            )
        }
        for name in names:
            self.assertEqual(
                gfx9_recipes[name]["blob_gen_cmd"],
                static_by_name[name]["blob_gen_cmd"],
            )

    def test_cdna_and_gfx1250_processor_counts_are_not_doubled(self):
        for gfx, processor_count in (
            ("gfx942", 304),
            ("gfx950", 256),
            ("gfx1250", 256),
        ):
            with self.subTest(gfx=gfx):
                self.assertEqual(
                    torch_processor_count_to_cu(gfx, processor_count), processor_count
                )

    def test_rdna_wgp_counts_are_normalized_to_physical_cus(self):
        for gfx, wgp_count, cu_count in (
            ("gfx1101", 30, 60),
            ("gfx1151", 20, 40),
            ("gfx1201", 32, 64),
        ):
            with self.subTest(gfx=gfx):
                self.assertEqual(torch_processor_count_to_cu(gfx, wgp_count), cu_count)


if __name__ == "__main__":
    unittest.main()
