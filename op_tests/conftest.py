# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "full_grid: full parameter sweep (slow); activate with: pytest -m full_grid",
    )
