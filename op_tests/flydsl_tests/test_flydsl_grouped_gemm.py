#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compatibility entry point for the gfx1250 grouped GEMM benchmark."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1] / "test_flydsl_grouped_gemm_gfx1250.py"
        ),
        run_name="__main__",
    )
