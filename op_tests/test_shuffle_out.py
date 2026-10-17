# SPDX-License-Identifier: MIT

import pytest
import torch

from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


@pytest.mark.parametrize("gate_up", [False, True])
def test_shuffle_weight_a16w4_can_write_caller_storage(gate_up):
    experts = 2
    rows = 64 if gate_up else 32
    src = (
        torch.arange(experts * rows * 64, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(experts, rows, 64)
    )
    expected = shuffle_weight_a16w4(src, 16, gate_up)
    out = torch.empty_like(expected)

    actual = shuffle_weight_a16w4(src, 16, gate_up, out=out)

    assert actual is out
    assert actual.is_shuffled
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("gate_up", [False, True])
def test_shuffle_scale_a16w4_can_write_padded_caller_storage(gate_up):
    experts = 2
    rows = 64 if gate_up else 32
    src = (
        torch.arange(experts * rows * 5, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(experts * rows, 5)
    )
    expected = shuffle_scale_a16w4(src, experts, gate_up)
    assert expected.shape == (experts * rows, 8)
    out = torch.empty_like(expected)

    actual = shuffle_scale_a16w4(src, experts, gate_up, out=out)

    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shuffle_out_rejects_the_wrong_shape():
    src = torch.zeros((2, 32, 64), dtype=torch.uint8)

    with pytest.raises(ValueError, match="out must have shape"):
        shuffle_weight_a16w4(src, 16, False, out=torch.empty(1, dtype=torch.uint8))
