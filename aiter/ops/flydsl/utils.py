# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""General utilities shared across all FlyDSL kernel families."""

import importlib.util


def is_flydsl_available() -> bool:
    return importlib.util.find_spec("flydsl") is not None
