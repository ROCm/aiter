# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The support query of qsa_sparse_prefill -- no GPU needed.

Callers pick the FlyDSL path (and the narrowest width that holds their selection)
before building it, so the query has to answer exactly what the entry accepts:
gfx942, the widths 4T + 3, T = 256/512/768/1024, and an int32 selection under
2**31 bytes. The GPU architecture is patched, so this runs on any host.
op_tests/test_flydsl_qsa_sparse_prefill.py exercises the kernels themselves.
"""

import itertools

import pytest

pytest.importorskip("flydsl", reason="QSA sparse prefill is built with FlyDSL")

from aiter.ops import qsa
from aiter.ops.flydsl import qsa_sparse_prefill as flydsl_prefill

WIDTHS = (1027, 2051, 3075, 4099)


def _set_gfx(monkeypatch, gfx):
    monkeypatch.setattr(flydsl_prefill, "get_gfx_runtime", lambda: gfx)


@pytest.fixture
def gfx942(monkeypatch):
    _set_gfx(monkeypatch, "gfx942")


def _entry_rule(rows, width):
    """What the entry accepted inline before the query existed."""
    topk = (width - 3) // 4
    return (
        width == 4 * topk + 3
        and topk % 256 == 0
        and 256 <= topk <= 1024
        and rows * width * 4 < 2**31
    )


def test_widths_are_the_kernel_budgets_in_increasing_order(gfx942):
    assert flydsl_prefill.SELECTION_WIDTHS == WIDTHS
    assert qsa.qsa_sparse_prefill_widths() == WIDTHS


@pytest.mark.parametrize("width", WIDTHS)
def test_byte_limit_is_the_last_row_under_2_31_bytes(gfx942, width):
    rows = (2**31 - 1) // (4 * width)
    assert qsa.qsa_sparse_prefill_supports_selection(rows, width)
    assert not qsa.qsa_sparse_prefill_supports_selection(rows + 1, width)


def test_query_matches_the_entry_rule(gfx942):
    widths = sorted(
        {w + d for w in WIDTHS for d in (-4, -1, 0, 1, 4)} | {0, 3, 7, 4103, 5123}
    )
    rows = [0, 1, 8192, 131072, 131073, 2**20]
    for r, w in itertools.product(rows, widths):
        assert qsa.qsa_sparse_prefill_supports_selection(r, w) == _entry_rule(r, w), (
            r,
            w,
        )


@pytest.mark.parametrize("gfx", ["gfx950", "gfx90a", "gfx1250"])
def test_other_gpus_support_nothing(monkeypatch, gfx):
    _set_gfx(monkeypatch, gfx)
    assert not flydsl_prefill.flydsl_qsa_sparse_prefill_available()
    assert qsa.qsa_sparse_prefill_widths() == ()
    assert not qsa.qsa_sparse_prefill_supports_selection(8, 2051)


def test_without_flydsl_nothing_is_supported(gfx942, monkeypatch):
    monkeypatch.setattr(qsa, "_HAS_FLYDSL_PREFILL", False)
    assert qsa.qsa_sparse_prefill_widths() == ()
    assert not qsa.qsa_sparse_prefill_supports_selection(8, 2051)
