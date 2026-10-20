# SPDX-License-Identifier: MIT
"""Validate poisoned-page fixtures on CPU before spending a GPU allocation."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "m3_scale_fixture",
    Path(__file__).with_name("test_unified_attention_scale_cache.py"),
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@pytest.mark.parametrize("lengths", [(7, 31), (31, 7), (128, 129), (129, 257)])
def test_shared_prefix_does_not_poison_valid_values(lengths):
    expected, table, values = fixture.check_case(
        "shuffle",
        "token",
        False,
        page=128,
        heads=16,
        kv_heads=1,
        q_lens=[1, 3],
        k_lens=list(lengths),
        backend="gluon",
        device="cpu",
        reference_only=True,
    )
    assert expected.isfinite().all()
    assert bool(table[0, 0] == table[1, 0]) == (min(lengths) >= 128)
    for i, length in enumerate(lengths):
        positions = torch.arange(length)
        pages, slots = table[i, positions // 128].long(), positions % 128
        assert values.float()[pages, slots].isfinite().all()
        if length % 128:
            tail = values.float()[table[i, length // 128].long(), length % 128 :]
            assert tail.isnan().all(), "The partial-page poison must remain active"
