# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        dest="full",
        default=False,
        help=(
            "Parametrize the shape-swept tests over the full grid instead of the "
            "CI subset (slow). Currently honoured by "
            "test_flydsl_linear_attention_prefill.py."
        ),
    )
    parser.addoption(
        "--perf",
        action="store_true",
        dest="perf",
        default=False,
        help=(
            "Run tests marked ``perf`` (benchmarks). They are skipped by default "
            "because they are slow and report timings rather than assert."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "perf: benchmark rather than a correctness check; skipped unless --perf",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``perf``-marked tests unless ``--perf`` was passed."""
    if config.getoption("perf"):
        return
    skip_perf = pytest.mark.skip(reason="perf benchmark; pass --perf to run")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)
