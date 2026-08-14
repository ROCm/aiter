# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only unit tests for the deterministic MLA-decode split-cap env logic
(issue #4364): valid, invalid, zero, negative, whitespace, and combined settings.

Loads aiter/ops/_det_split.py directly by path so it needs no GPU and does not
trigger the aiter JIT build.
"""

import importlib.util
import os
import warnings

import pytest

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "aiter",
    "ops",
    "_det_split.py",
)
_spec = importlib.util.spec_from_file_location(
    "aiter_det_split_standalone", _MODULE_PATH
)
_det_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_det_split)
resolve = _det_split.resolve_det_split_cap

_MSPB = "AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH"
_DET = "AITER_MLA_DECODE_DETERMINISTIC"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_MSPB, raising=False)
    monkeypatch.delenv(_DET, raising=False)


# ---- no override: budget unchanged ----------------------------------------
@pytest.mark.parametrize("budget", [8, 1, 0, -1])
def test_unset_returns_input(budget):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert resolve(budget) == budget


# ---- valid caps ------------------------------------------------------------
@pytest.mark.parametrize(
    "cap, budget, expected",
    [
        ("1", 8, 1),  # single-split (reproducible)
        ("4", 8, 4),  # clamp down
        ("32", 8, 8),  # cap above budget -> budget unchanged (min)
        ("4", 0, 4),  # budget<=0 sentinel -> take the cap
        ("4", -1, 4),  # budget<=0 sentinel -> take the cap
        (" 1 ", 8, 1),  # whitespace tolerated by int()
        ("01", 8, 1),  # leading zero tolerated by int()
    ],
)
def test_valid_cap(monkeypatch, cap, budget, expected):
    monkeypatch.setenv(_MSPB, cap)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve(budget) == expected


# ---- deterministic shorthand ----------------------------------------------
@pytest.mark.parametrize("val", ["1", "true", "yes", "2"])
def test_deterministic_flag_forces_single_split(monkeypatch, val):
    monkeypatch.setenv(_DET, val)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve(8) == 1


@pytest.mark.parametrize("val", ["0", ""])
def test_deterministic_flag_off(monkeypatch, val):
    monkeypatch.setenv(_DET, val)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve(8) == 8


# ---- invalid / out-of-range: warn + leave budget unchanged -----------------
@pytest.mark.parametrize("cap", ["abc", "1.5", "", "  "])
def test_invalid_value_warns_and_ignored(monkeypatch, cap):
    monkeypatch.setenv(_MSPB, cap)
    with pytest.warns(
        UserWarning, match="invalid AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH"
    ):
        assert resolve(8) == 8


@pytest.mark.parametrize("cap", ["0", "-1", "-5"])
def test_out_of_range_warns_and_ignored(monkeypatch, cap):
    monkeypatch.setenv(_MSPB, cap)
    with pytest.warns(UserWarning, match="must be >= 1"):
        assert resolve(8) == 8


# ---- both set: explicit cap wins; warn only when they disagree -------------
def test_both_set_agree_no_conflict_warning(monkeypatch):
    monkeypatch.setenv(_MSPB, "1")
    monkeypatch.setenv(_DET, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve(8) == 1


def test_both_set_disagree_warns_and_cap_wins(monkeypatch):
    monkeypatch.setenv(_MSPB, "8")
    monkeypatch.setenv(_DET, "1")
    with pytest.warns(UserWarning, match="the explicit cap takes"):
        # DETERMINISTIC would imply 1, but the explicit cap (8) wins.
        assert resolve(16) == 8


def test_both_set_disagree_invalid_cap_falls_back(monkeypatch):
    # Explicit cap present but invalid -> conflict warning AND invalid warning;
    # budget is left unchanged (DETERMINISTIC does not rescue it).
    monkeypatch.setenv(_MSPB, "abc")
    monkeypatch.setenv(_DET, "1")
    with pytest.warns(UserWarning):
        assert resolve(8) == 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
