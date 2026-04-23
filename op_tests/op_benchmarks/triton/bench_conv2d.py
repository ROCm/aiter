# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Thin shim that runs the conv2d benchmark via the test harness.

Forwards all CLI flags to ``op_tests.triton_tests.conv.cli`` with
``--benchmark`` and ``--test-mode models`` injected when missing.
Run ``python -m op_tests.triton_tests.conv.cli --help`` to see all options.
"""

import sys


def main():
    if "--benchmark" not in sys.argv:
        sys.argv.insert(1, "--benchmark")
    has_mode = any(a in sys.argv for a in ("--test-mode", "--mode"))
    if not has_mode:
        sys.argv.extend(["--test-mode", "models"])
    from op_tests.triton_tests.conv.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
