#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Main test runner for OPUS unit tests.

This script runs all OPUS tests (both C++ and Python) and reports results.
"""

import subprocess
import sys
import os

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AITER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))


def run_cpp_tests():
    """Build and run C++ unit tests."""
    print("=" * 60)
    print("Running C++ OPUS Unit Tests")
    print("=" * 60)

    # Check for hipcc
    result = subprocess.run(["which", "hipcc"], capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: hipcc not found. C++ tests require ROCm/HIP.")
        print("Skipping C++ tests.")
        return None

    # Try to build and run
    os.chdir(SCRIPT_DIR)

    # Build
    print("\nBuilding C++ tests...")
    result = subprocess.run(["make", "all"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        return False

    # Run
    print("\nRunning C++ tests...")
    result = subprocess.run(["./test_opus_basic"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def run_python_integration_tests():
    """Run Python integration tests."""
    print("\n" + "=" * 60)
    print("Running Python Integration Tests")
    print("=" * 60)

    os.chdir(SCRIPT_DIR)

    result = subprocess.run(
        [sys.executable, "test_integration.py"], capture_output=True, text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def run_python_type_tests():
    """Run Python type tests."""
    print("\n" + "=" * 60)
    print("Running Python Type Tests")
    print("=" * 60)

    os.chdir(SCRIPT_DIR)

    # Try pytest first
    result = subprocess.run(["which", "pytest"], capture_output=True, text=True)
    if result.returncode == 0:
        result = subprocess.run(
            ["pytest", "test_opus_types.py", "-v"], capture_output=True, text=True
        )
    else:
        # Fall back to direct execution
        result = subprocess.run(
            [sys.executable, "test_opus_types.py"], capture_output=True, text=True
        )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def main():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("OPUS (AI Operator Micro Std) Test Suite")
    print("=" * 60)

    results = {}

    # Run C++ tests
    try:
        cpp_result = run_cpp_tests()
        results["C++ Tests"] = cpp_result
    except Exception as e:
        print(f"C++ tests failed with exception: {e}")
        results["C++ Tests"] = False

    # Run Python integration tests
    try:
        py_int_result = run_python_integration_tests()
        results["Python Integration"] = py_int_result
    except Exception as e:
        print(f"Python integration tests failed with exception: {e}")
        results["Python Integration"] = False

    # Run Python type tests
    try:
        py_type_result = run_python_type_tests()
        results["Python Type Tests"] = py_type_result
    except Exception as e:
        print(f"Python type tests failed with exception: {e}")
        results["Python Type Tests"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {test_name:<25} {status}")

    print("=" * 60)

    # Return overall status
    all_passed = all(r is True or r is None for r in results.values())
    any_failed = any(r is False for r in results.values())

    if any_failed:
        print("\nSome tests FAILED")
        return 1
    elif all_passed:
        print("\nAll tests PASSED (or skipped)")
        return 0
    else:
        print("\nAll available tests PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
