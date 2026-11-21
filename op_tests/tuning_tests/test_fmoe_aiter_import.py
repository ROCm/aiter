# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fmoe tuner modules that read aiter.<name> import aiter at module level.

aiter is also a Python builtin (3.10+). A module that only does
`from aiter import X` and then calls `aiter.get_hip_quant(...)` imports fine
and passes ruff (F821 sees the builtin), but raises AttributeError when the
line runs. In the tuner that error only marks the candidate as failed. CPU
only: it reads the source, nothing is imported or run.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TUNER_DIRS = [
    REPO_ROOT / "aiter" / "tuning",
    REPO_ROOT / "csrc" / "ck_gemm_moe_2stages_codegen",
]


def aiter_uses_without_module_import(source):
    """Return the lines reading aiter.<name> if the module body has no `import aiter`."""
    tree = ast.parse(source)
    if any(
        isinstance(stmt, ast.Import)
        and any(alias.name == "aiter" and alias.asname is None for alias in stmt.names)
        for stmt in tree.body
    ):
        return []
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "aiter"
    )


class TestFmoeAiterImport(unittest.TestCase):
    def test_check_finds_aiter_without_module_import(self):
        source = (
            "from aiter import QuantType\n"
            "\n"
            "def run(q_type=QuantType.per_1x128):\n"
            "    return aiter.get_hip_quant(q_type)\n"
        )
        self.assertEqual(aiter_uses_without_module_import(source), [4])
        self.assertEqual(
            aiter_uses_without_module_import("import aiter\n" + source), []
        )
        local_import = "def other():\n    import aiter\n\n\n" + source
        self.assertEqual(aiter_uses_without_module_import(local_import), [8])

    def test_tuner_modules_import_aiter(self):
        files = sorted(p for d in TUNER_DIRS for p in d.rglob("*.py"))
        self.assertTrue(files)
        found = [
            f"{path.relative_to(REPO_ROOT)}:{line}"
            for path in files
            for line in aiter_uses_without_module_import(path.read_text())
        ]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
