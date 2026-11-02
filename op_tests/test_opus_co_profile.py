# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only ASIC parsing and generated CO device-cache contract tests."""

import ast
import functools
import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_chip_parsers(output):
    source = ROOT / "aiter/jit/utils/chip_info.py"
    wanted = {
        "_parse_asic_revision",
        "_rocminfo_gpu_agents",
        "_hipinfo_gpu_agents",
        "get_asic_revision",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"), str(source))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        "functools": functools,
        "re": re,
        "IS_WINDOWS": False,
        "_GPU_INFO_TOOL": "rocminfo",
        "_gpu_info_output": lambda: output,
        "get_gfx_runtime": lambda: "gfx1250",
    }
    exec(  # noqa: S102 - trusted checkout AST, with injected CPU dependencies
        compile(tree, str(source), "exec"), namespace
    )
    return namespace


class AsicRevisionTests(unittest.TestCase):
    def test_hex_decimal_and_parentheses(self):
        parse = load_chip_parsers("")["_parse_asic_revision"]
        for text, expected in (
            ("0x0", 0),
            ("0X1", 1),
            ("0", 0),
            ("1 (B0)", 1),
            ("01", 1),
        ):
            with self.subTest(text=text):
                self.assertEqual(parse(text), expected)
        for text in ("", "-1", "unknown", "chip_rev_id=0x11", "0x"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text)

    def test_mixed_agents_keep_node_gate_minimum_not_device_selection(self):
        output = """Agent 1
  Name: CPU
  Device Type: CPU
  ASIC Revision: 0x11
Agent 2
  Name: gfx1250:sramecc+
  Device Type: GPU
  ASIC Revision: 0x1
Agent 3
  Name: gfx1250
  Device Type: GPU
  ASIC Revision: 0x0
"""
        parsers = load_chip_parsers(output)
        self.assertEqual(
            parsers["_rocminfo_gpu_agents"](), [("gfx1250", 1), ("gfx1250", 0)]
        )
        # Existing shipped-ASM policy remains node-wide/fail-closed. The CO
        # runtime deliberately uses current-device HIP asicRevision instead.
        self.assertEqual(parsers["get_asic_revision"](), 0)

    def test_windows_hipinfo_accepts_hex(self):
        parsers = load_chip_parsers(
            "device#0\ngcnArchName: gfx1250\nasicRevision: 0x1\n"
        )
        self.assertEqual(parsers["_hipinfo_gpu_agents"](), [("gfx1250", 1)])

    def test_test_helpers_support_combined_and_direct_a0_leaves(self):
        source = ROOT / "op_tests/test_opus_co_integration.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), str(source))
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_co_test_profile_dirs"
        ]
        namespace = {"Path": Path, "GFX1250": "gfx1250"}
        exec(  # noqa: S102 - trusted checkout AST, with injected CPU dependencies
            compile(tree, str(source), "exec"), namespace
        )
        profiles = namespace["_co_test_profile_dirs"]
        base = ROOT / "csrc/opus_gemm/gen_co"
        self.assertEqual(
            [relative for _, relative in profiles(base)], ["gfx1250", "gfx1250-a0"]
        )
        self.assertEqual(profiles(base / "gfx1250-a0"), [(base / "gfx1250-a0", "")])

    def test_all_profile_metadata_uses_current_local_kids(self):
        sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
        from opus_gemm_common import gfx1250_4wave_co_kernels_declared

        expected = {
            instance.name: kid
            for kid, instance in gfx1250_4wave_co_kernels_declared.items()
        }
        base = ROOT / "csrc/opus_gemm/gen_co"
        for relative in ("gfx1250", "gfx1250-a0", "named_reg/gfx1250"):
            with self.subTest(profile=relative):
                info = json.loads((base / relative / "build_info.json").read_text())
                self.assertEqual(len(info["kernels"]), len(expected))
                self.assertEqual(
                    {entry["symbol"]: entry["kid"] for entry in info["kernels"]},
                    expected,
                )

    def test_launcher_does_not_pin_the_first_device(self):
        sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
        from codegen import gen_instances_gfx1250 as gen
        from opus_gemm_common import gfx1250_4wave_co_kernels_declared

        k = gfx1250_4wave_co_kernels_declared[496]
        cg = SimpleNamespace(
            impl_path="unused", _host_instantiations=[], _device_instantiations=[]
        )
        generated = []
        with mock.patch.object(
            gen,
            "write_if_changed",
            side_effect=lambda path, text: generated.append(text),
        ):
            gen.gen_4wave_co_instance(
                cg,
                k,
                gen.TRAITS_HEADER_MAP[k.kernel_tag],
                gen.TRAITS_NAME_MAP[k.kernel_tag],
                gen.KARGS_NAME_MAP[k.kernel_tag],
            )
        self.assertEqual(len(generated), 1)
        self.assertNotIn("static AiterAsmKernelFast& kernel", generated[0])
        self.assertIn(f'co_kernel("{k.name}", XQ.device_id)', generated[0])
        self.assertEqual(cg._device_instantiations, [])


if __name__ == "__main__":
    unittest.main()
