#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation entry point for the FlyDSL K5 mfma16_hip fork.

The implementation lives in ``chunk_gdn_h.py`` next to the baseline K5 kernel,
since the two share their csv handling, fan-out and compile driver. This module
is the ``mfma16_hip``-only view of it, which is what
``OpKind.CHUNK_GDN_H_MFMA16_HIP`` imports and what the documented
``python -m`` command below resolves to.

Usage:
    # Compile every configuration from the default tuned table
    python -m aiter.aot.flydsl.chunk_gdn_h_mfma16_hip

    # Custom table(s)
    python -m aiter.aot.flydsl.chunk_gdn_h_mfma16_hip --csv /path/to/tuned.csv

    # Cross-compile for another arch (host need not own that GPU)
    python -m aiter.aot.flydsl.chunk_gdn_h_mfma16_hip --target-arch gfx942

    # Both K5 kernels in one run
    python -m aiter.aot.flydsl.chunk_gdn_h
"""

from __future__ import annotations

from aiter.aot.flydsl.chunk_gdn_h import (
    DEFAULT_CSVS_MFMA16_HIP as DEFAULT_CSVS,
)
from aiter.aot.flydsl.chunk_gdn_h import (
    MFMA16_HIP_AOT_ARCH_DEFAULT,
)
from aiter.aot.flydsl.chunk_gdn_h import (
    compile_one_config_mfma16_hip as compile_one_config,
)
from aiter.aot.flydsl.chunk_gdn_h import (
    main_mfma16_hip as main,
)
from aiter.aot.flydsl.chunk_gdn_h import (
    parse_csv_mfma16_hip as parse_csv,
)

__all__ = [
    "DEFAULT_CSVS",
    "MFMA16_HIP_AOT_ARCH_DEFAULT",
    "compile_one_config",
    "main",
    "parse_csv",
]


if __name__ == "__main__":
    main()
