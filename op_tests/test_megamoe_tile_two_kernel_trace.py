# SPDX-License-Identifier: MIT
"""CPU-only checks for the strict EP16 two-kernel trace auditor."""

from __future__ import annotations

import pytest

from scripts.megamoe_tile.assert_ep16_two_kernel_trace import (
    KernelLaunch,
    audit_launches,
)


STAGE1 = "megamoe_tile_ep16_stage1"
STAGE2 = "megamoe_tile_ep16_stage2"


def _launch(name: str, ordinal: int) -> KernelLaunch:
    start = ordinal * 1000
    return KernelLaunch(name, start, start + 800, "synthetic.csv", ordinal + 2)


def test_two_kernel_tail_accepts_exact_pairs():
    launches = [_launch("jit_helper", 0)]
    launches.extend(
        _launch(name, ordinal + 1)
        for ordinal, name in enumerate([STAGE1, STAGE2] * 4)
    )
    result = audit_launches(
        launches,
        stage1_regex=STAGE1,
        stage2_regex=STAGE2,
        iterations=3,
    )
    assert result["selected_launches"] == 6
    assert result["sequence"] == [STAGE1, STAGE2] * 3


def test_two_kernel_tail_rejects_standalone_input_quant():
    names = [STAGE1, STAGE2, "standalone_bf16_to_a4", STAGE1, STAGE2]
    launches = [_launch(name, ordinal) for ordinal, name in enumerate(names)]
    with pytest.raises(AssertionError, match="adjacent launches"):
        audit_launches(
            launches,
            stage1_regex=STAGE1,
            stage2_regex=STAGE2,
            iterations=2,
        )


def test_two_kernel_trace_requires_unambiguous_symbols():
    launches = [
        _launch("megamoe_tile_ep16_stage1_a", 0),
        _launch(STAGE2, 1),
        _launch("megamoe_tile_ep16_stage1_b", 2),
        _launch(STAGE2, 3),
    ]
    with pytest.raises(ValueError, match="one exact kernel symbol"):
        audit_launches(
            launches,
            stage1_regex=r"megamoe_tile_ep16_stage1_.*",
            stage2_regex=STAGE2,
            iterations=2,
        )
