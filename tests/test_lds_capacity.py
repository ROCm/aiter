# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only LDS lookup and isolated package/codegen import regression tests."""

import ast
import builtins
from contextlib import contextmanager
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest import mock

UTILS = Path(__file__).resolve().parents[1] / "aiter" / "jit" / "utils"


@contextmanager
def isolated_wrapper(package):
    # Isolate LDS imports and the wrapper from unrelated Torch/HIP imports.
    tree = ast.parse((UTILS / "chip_info.py").read_text())
    members = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "get_lds_capacity_bytes")
        or (
            isinstance(node, ast.If)
            and any(
                isinstance(child, ast.ImportFrom) and child.module == "lds_capacity"
                for child in ast.walk(node)
            )
        )
    ]
    code = compile(ast.Module(body=members, type_ignores=[]), "chip_info.py", "exec")
    get_gfx = mock.Mock(return_value="gfx1250-strict")
    namespace = {"__package__": package, "get_gfx": get_gfx}
    with mock.patch.dict(sys.modules):
        sys.modules.pop("lds_capacity", None)
        if package:
            parent = ModuleType(package)
            parent.__path__ = [str(UTILS)]
            sys.modules[package] = parent
            sys.modules["lds_capacity"] = None
            sys.modules.pop(f"{package}.lds_capacity", None)
        with mock.patch.object(sys, "path", [str(UTILS), *sys.path]):
            exec(code, namespace)
            yield namespace["get_lds_capacity_bytes"], get_gfx


class LdsCapacityTest(unittest.TestCase):
    def test_helper_import_needs_no_external_dependencies(self):
        code = compile(
            (UTILS / "lds_capacity.py").read_text(), "lds_capacity.py", "exec"
        )
        namespace = {}
        with mock.patch.object(
            builtins, "__import__", side_effect=ImportError("imports forbidden")
        ):
            exec(code, namespace)
        self.assertEqual(namespace["get_lds_capacity_bytes"]("gfx1250-strict"), 327680)

    def test_capacities_and_target_normalization(self):
        expected_kib = {
            "gfx90a": 64,
            "gfx942": 64,
            "gfx950": 160,
            "gfx1100": 64,
            "gfx1101": 64,
            "gfx1102": 64,
            "gfx1150": 64,
            "gfx1151": 64,
            "gfx1200": 64,
            "gfx1201": 64,
            "gfx1250": 320,
            "gfx1250-strict": 320,
        }
        lookup = runpy.run_path(str(UTILS / "lds_capacity.py"))[
            "get_lds_capacity_bytes"
        ]
        for arch, kib in expected_kib.items():
            for target in (arch, arch.upper() + ":sramecc+:xnack-"):
                with self.subTest(target=target):
                    self.assertEqual(lookup(target), kib * 1024)

    def test_unknown_architecture_is_rejected(self):
        lookup = runpy.run_path(str(UTILS / "lds_capacity.py"))[
            "get_lds_capacity_bytes"
        ]
        with self.assertRaisesRegex(ValueError, "Unknown LDS capacity.*gfx9999"):
            lookup("GFX9999:sramecc+")

    def test_table_reexport_in_package_and_flattened_codegen(self):
        for package in ("_lds_test_package", ""):
            with self.subTest(package=package), isolated_wrapper(package) as (
                lookup,
                _,
            ):
                helper_name = f"{package}.lds_capacity" if package else "lds_capacity"
                table = lookup.__globals__["_LDS_CAPACITY_BYTES"]
                self.assertIs(table, sys.modules[helper_name]._LDS_CAPACITY_BYTES)
                self.assertEqual(table["gfx1250-strict"], 327680)

    def test_wrapper_in_package_and_flattened_codegen(self):
        for package in ("_lds_test_package", ""):
            with self.subTest(package=package), isolated_wrapper(package) as (
                lookup,
                get_gfx,
            ):
                self.assertEqual(lookup("GFX950:xnack-"), 163840)
                get_gfx.assert_not_called()
                self.assertEqual(lookup(), 327680)
                get_gfx.assert_called_once_with()
                with self.assertRaisesRegex(
                    ValueError, "Unknown LDS capacity.*gfx9999"
                ):
                    lookup("gfx9999")


if __name__ == "__main__":
    unittest.main()
