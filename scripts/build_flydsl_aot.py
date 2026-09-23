#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build AITER's registered FlyDSL AOT kernels in an isolated process."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This file is intentionally executed by path.  Set the lightweight-import gate
# before importing the ``aiter`` package, then make the source checkout visible
# when the caller's working directory is elsewhere.
os.environ["AITER_AOT_IMPORT"] = "1"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_PROJECT_ROOT / "aiter" / "jit" / "flydsl_cache",
        help="FlyDSL cache directory to populate",
    )
    parser.add_argument(
        "--list-specs",
        action="store_true",
        help="list registered AOT operator families without compiling",
    )
    args = parser.parse_args(argv)

    from aiter.aot.flydsl.spec import (
        DEFAULT_AOT_SPEC_REGISTRY,
        register_default_specs,
    )

    register_default_specs()
    specs = DEFAULT_AOT_SPEC_REGISTRY.get_all_specs()
    if args.list_specs:
        for spec in specs:
            print(f"{spec.name}: {spec.module}")
        return 0

    from aiter.aot.flydsl.common import run_aot

    run_aot(str(args.cache_dir), specs=specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
