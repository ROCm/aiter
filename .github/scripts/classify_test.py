#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
classify_test.py

Lightweight, in-memory Python AST-based test classifier for ROCm/aiter CI.
Distinguishes pure pytest test suites from legacy CLI benchmark scripts
without executing module code, without importing dependencies, and with
zero GPU/PyTorch requirements.

Usage:
    python3 .github/scripts/classify_test.py <path_to_test_file.py>

Output:
    Prints 'PYTEST' or 'PYTHON' to stdout.
    Exit code 0 on successful classification.
"""

from __future__ import annotations

import ast
import sys
from typing import Set

# Standard pytest built-in fixtures recognized across all test suites
STANDARD_FIXTURES: Set[str] = {
    "self",
    "cls",
    "monkeypatch",
    "capsys",
    "capfd",
    "capsysbinary",
    "capfdbinary",
    "tmp_path",
    "tmp_path_factory",
    "tmpdir",
    "tmpdir_factory",
    "request",
    "pytestconfig",
    "record_property",
    "record_testsuite_property",
    "recwarn",
    "benchmark",
    "cache",
    "doctest_namespace",
}


def classify_test_file(filepath: str) -> str:
    """
    Classifies a Python test file as either 'PYTEST' or 'PYTHON'.

    Returns:
        'PYTEST': Pure pytest test suite containing test functions with valid
                  signatures (no unparametrized positional arguments).
        'PYTHON': Legacy benchmark script containing CLI argument parsing,
                  custom main sweeps, or unparametrized positional test functions.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except Exception:
        # Fall back to python3 to surface syntax or encoding errors
        return "PYTHON"

    # 1. Check for module-level CLI argument parsing (e.g. parser.parse_args())
    for node in tree.body:
        if isinstance(node, (ast.Expr, ast.Assign)):
            unparsed = ast.unparse(node)
            if "parse_args(" in unparsed:
                return "PYTHON"

    # 2. Inspect `if __name__ == '__main__':` entry point
    has_main_block = False
    main_only_calls_pytest = False

    for node in tree.body:
        if isinstance(node, ast.If):
            test_unp = ast.unparse(node.test)
            if "__name__" in test_unp and "__main__" in test_unp:
                has_main_block = True
                body_unp = ast.unparse(node.body)
                # If __main__ is solely a pytest.main() dispatcher, it can run as PYTEST
                if "pytest.main" in body_unp and "argparse" not in body_unp:
                    main_only_calls_pytest = True
                break

    if has_main_block and not main_only_calls_pytest:
        return "PYTHON"

    # 3. Discover local fixture definitions in the file
    local_fixtures: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if "fixture" in ast.unparse(dec):
                    local_fixtures.add(node.name)

    recognized_fixtures = STANDARD_FIXTURES | local_fixtures

    # 4. Discover test functions and test classes
    test_function_nodes = []
    has_test_classes = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_function_nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test") or any("TestCase" in ast.unparse(b) for b in node.bases):
                has_test_classes = True

    # If there are no test functions and no test classes, it's not a pytest suite
    if not test_function_nodes and not has_test_classes:
        return "PYTHON"

    # 5. Check test functions for unparametrized positional arguments
    # (e.g. def test_gemm(dtype, m, n, k) which causes FixtureLookupError under pytest)
    for fn in test_function_nodes:
        # Total positional arguments without default values
        num_pos_no_default = len(fn.args.args) - len(fn.args.defaults)
        if num_pos_no_default > 0:
            # Check if covered by @pytest.mark.parametrize
            is_parametrized = any("parametrize" in ast.unparse(d) for d in fn.decorator_list)
            if not is_parametrized:
                pos_arg_names = [a.arg for a in fn.args.args[:num_pos_no_default]]
                unresolved = [a for a in pos_arg_names if a not in recognized_fixtures]
                if unresolved:
                    # Positional arguments require external CLI/script caller -> Legacy script
                    return "PYTHON"

    return "PYTEST"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: classify_test.py <path_to_test_file.py>", file=sys.stderr)
        sys.exit(1)

    result = classify_test_file(sys.argv[1])
    print(result)


if __name__ == "__main__":
    main()
