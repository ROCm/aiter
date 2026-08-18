# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest


def test_rank_partial_epoch_lsa_gate_factory_contract():
    """Build the FlyDSL gate and pin its registered-window ABI metadata."""

    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import compile_rank_partial_epoch_gate_lsa

    gate = compile_rank_partial_epoch_gate_lsa(NUM_RANKS=8)
    assert "rank_partial_epoch_lsa_r8_t64" in gate.kernel_name
    assert gate.num_ranks == 8
    assert gate.threads == 64
    assert gate.ready_kind == "absolute-generation"
    assert gate.publish_before_wait is True
    assert gate.requires_registered_window_handle is True
    assert gate.memory_order == "system-release-acquire"


@pytest.mark.parametrize(
    ("num_ranks", "threads"),
    ((0, 64), (65, 128), (8, 32)),
)
def test_rank_partial_epoch_lsa_gate_rejects_invalid_geometry(
    num_ranks: int, threads: int
):
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import compile_rank_partial_epoch_gate_lsa

    with pytest.raises(ValueError):
        compile_rank_partial_epoch_gate_lsa(
            NUM_RANKS=num_ranks,
            threads=threads,
        )
