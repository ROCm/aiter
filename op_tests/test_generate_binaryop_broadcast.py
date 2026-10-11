# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only regression for the generated 3D binary-op broadcast guard."""

import ast
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _broadcast_case_source() -> str:
    generator = (
        Path(__file__).resolve().parents[1] / "csrc/kernels/generate_binaryop.py"
    )
    tree = ast.parse(generator.read_text(encoding="utf-8"))
    codegen = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BinaryOpCodegen"
    )
    header = next(
        ast.literal_eval(node.value)
        for node in codegen.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "API_COMMON_HEADER"
            for target in node.targets
        )
    )
    start = header.index("auto broadcast_3d_case =")
    end = header.index("\n        };", start) + len("\n        };")
    return header[start:end]


def test_generated_3d_broadcast_guard_rejects_incompatible_dimensions():
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required to execute the generated guard")

    source = f"""
struct Tensor {{
  int dims[3];
  int size(int index) const {{ return dims[index]; }}
}};

bool supports(Tensor input, Tensor other) {{
  constexpr int PATTERN_BROADCAST_0 = 2;
  constexpr int PATTERN_BROADCAST_1 = 3;
  constexpr int PATTERN_BROADCAST_2 = 5;
  bool is_support = false;
  bool order_flag = true;
  int pattern = 0;
  {_broadcast_case_source()}
  for (int dim = 0; dim < 3; ++dim) broadcast_3d_case(dim);
  return is_support;
}}

int main() {{
  if (supports({{3, 4, 8}}, {{5, 4, 8}})) return 1;
  if (supports({{4, 3, 8}}, {{4, 5, 8}})) return 2;
  if (supports({{4, 8, 3}}, {{4, 8, 5}})) return 3;
  if (!supports({{1, 4, 8}}, {{5, 4, 8}})) return 4;
  if (!supports({{5, 4, 8}}, {{1, 4, 8}})) return 5;
  if (supports({{1, 4, 8}}, {{5, 1, 8}})) return 6;
  return 0;
}}
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        cpp = path / "broadcast_guard.cpp"
        executable = path / "broadcast_guard"
        cpp.write_text(source, encoding="utf-8")
        subprocess.run(
            [compiler, "-std=c++17", str(cpp), "-o", str(executable)], check=True
        )
        subprocess.run([str(executable)], check=True)
