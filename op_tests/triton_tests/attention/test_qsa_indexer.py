# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA preparation: explicit Q outputs, cache writes and two-stage compression."""

import pytest
import torch

from aiter.ops.triton.attention.qsa_indexer import (
    qsa_indexer_decode_prep,
    qsa_indexer_prefill_prep,
)
from op_tests import qsa_indexer_test_utils as helpers

_SHARED = (
    "qk",
    "positions",
    "state_slots",
    "key_state",
    "rope_state",
    "write_locs",
    "compressed",
    "cos_sin_cache",
    "axis_map",
    "q_weight",
    "k_weight",
    "q_eps",
    "k_eps",
)


def make_prep_inputs(mode, rows, context, device=None):
    """Shared fixture for prep tests and rotating-buffer benchmarks."""
    device = (
        torch.device("cuda", torch.cuda.current_device()) if device is None else device
    )
    if mode == "prefill":
        source = helpers.synthetic((context,), (rows,), device)
        names = (*_SHARED, "member_rows", "rope_matrix")
    else:
        window = 4 if mode == "verify" else 1
        source = helpers.decode_case(
            (context - window + 1,) * rows,
            device,
            window=window,
            context=-(-context // 64) * 64,
        )
        names = (*_SHARED, "group_locs")
    inputs = {name: source.inputs[name] for name in names}
    if mode != "prefill":
        inputs["verify"] = mode == "verify"
    return source, inputs


@pytest.mark.parametrize("length,extend", ((7, 7), (2177, 129), (262148, 128)))
def test_prefill_preparation(length, extend):
    device = helpers._gpu()
    source, inputs = make_prep_inputs("prefill", extend, length, device)
    expected = helpers.reference_prep(source.inputs, source.state)
    q_out = torch.full_like(expected.q, float("nan"))
    stats = torch.zeros((source.rows, 2), dtype=torch.int32, device=device)
    q, actual_stats = qsa_indexer_prefill_prep(**inputs, q_out=q_out, stats_out=stats)
    assert q is q_out and actual_stats is stats
    torch.testing.assert_close(q, expected.q, rtol=0, atol=0)
    helpers.assert_state(source, expected)
    assert bool((stats == -1).all())


@pytest.mark.parametrize("mode", ("decode", "verify"))
@pytest.mark.parametrize("stats_enabled", (False, True))
def test_decode_preparation(mode, stats_enabled):
    device = helpers._gpu()
    source, inputs = make_prep_inputs(mode, 3, 12000, device)
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    q_out = torch.full_like(expected.q, float("nan"))
    stats = (
        torch.zeros((source.rows, 256, 2), dtype=torch.int32, device=device)
        if stats_enabled
        else None
    )
    q = qsa_indexer_decode_prep(**inputs, q_out=q_out, stats_out=stats)
    assert q is q_out
    torch.testing.assert_close(q, expected.q, rtol=0, atol=0)
    helpers.assert_state(source, expected)
    if stats is not None:
        assert bool((stats == -1).all())


def test_prep_rejects_cpu_metadata_before_state_writes():
    device = helpers._gpu()
    source, inputs = make_prep_inputs("decode", 2, 12, device)
    inputs["group_locs"] = torch.zeros((2, 4), dtype=torch.int32, device="cpu")
    before = helpers.clone_state(source.state)
    with pytest.raises(ValueError, match="same GPU"):
        qsa_indexer_decode_prep(**inputs)
    for name in helpers.STATE_NAMES:
        assert torch.equal(source.inputs[name], before[name])
