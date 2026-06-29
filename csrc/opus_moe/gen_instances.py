# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate Opus MoE stage2 dispatch headers.

This file starts as the common JIT/codegen entry point. Dtype-specific commits
add manifest emitters once their private kernel sources exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Opus MoE stage2 dispatch headers"
    )
    parser.add_argument("--working_path", required=True)
    parser.add_argument(
        "--tune_files", default="", help="Accepted for JIT compatibility."
    )
    parser.add_argument(
        "--tune_file", default=None, help="Deprecated alias for --tune_files."
    )
    parser.add_argument("--arch", default=None, help="Optional arch filter")
    parser.add_argument(
        "--cu-num", type=int, default=None, help="Optional CU-count filter"
    )
    args = parser.parse_args()

    out_dir = Path(args.working_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[opus_moe gen_instances] prepared {out_dir}")


if __name__ == "__main__":
    main()
