# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention.utils import lse_merge

# (batch, splits, sequence, heads): one range, ranges that fill one tile, and partial tiles.
SHAPES = [
    (1, 1, 77, 2),
    (1, 2, 1, 1),
    (1, 4, 4352, 3),
    (2, 3, 77, 2),
    (2, 7, 300, 1),
    (3, 5, 1024, 4),
]
TOLERANCES = {
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def _partials(batch, splits, sequence, heads, head_dim, dtype):
    partial_out = torch.randn(
        (batch * splits, sequence, heads, head_dim), dtype=dtype, device="cuda"
    )
    partial_lse = (
        torch.randn(
            (batch * splits, heads, sequence), dtype=torch.float32, device="cuda"
        )
        * 4
    )
    return partial_out, partial_lse


def _reference(partial_out, partial_lse, batch):
    """FP32 merge in torch. A range with LSE -inf for a row adds nothing to it."""
    splits = partial_out.shape[0] // batch
    _, sequence, heads, head_dim = partial_out.shape
    split_lse = partial_lse.view(batch, splits, heads, sequence)
    lse = torch.logsumexp(split_lse, dim=1)
    reached = torch.isfinite(lse)
    weight = torch.where(
        reached.unsqueeze(1), torch.exp(split_lse - lse.unsqueeze(1)), 0.0
    )
    empty = partial_lse.isneginf().transpose(1, 2).unsqueeze(-1)
    values = partial_out.float().masked_fill(empty, 0.0)
    out = torch.einsum(
        "bihs,bishd->bshd",
        weight,
        values.view(batch, splits, sequence, heads, head_dim),
    )
    return out, lse


@pytest.mark.parametrize("batch,splits,sequence,heads", SHAPES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("write_lse", [True, False])
def test_lse_merge_matches_reference(
    batch, splits, sequence, heads, head_dim, dtype, write_lse
):
    torch.manual_seed(0)
    partial_out, partial_lse = _partials(
        batch, splits, sequence, heads, head_dim, dtype
    )
    out = torch.empty((batch, sequence, heads, head_dim), dtype=dtype, device="cuda")
    lse = torch.empty((batch, heads, sequence), dtype=torch.float32, device="cuda")

    lse_merge(partial_out, partial_lse, out, lse if write_lse else None)

    reference_out, reference_lse = _reference(partial_out, partial_lse, batch)
    torch.testing.assert_close(out, reference_out.to(dtype), **TOLERANCES[dtype])
    if write_lse:
        torch.testing.assert_close(lse, reference_lse, atol=1e-5, rtol=1e-5)


def test_lse_merge_reads_and_writes_strided_views():
    """Only the last dimension has to be contiguous. The merge must not write past the view."""
    torch.manual_seed(0)
    batch, splits, sequence, heads, head_dim = 2, 3, 300, 2, 128
    partial_out = torch.randn(
        (batch * splits, sequence, 2 * heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )[:, :, ::2]
    partial_lse = torch.randn(
        (batch * splits, heads, sequence + 50), dtype=torch.float32, device="cuda"
    )[:, :, :sequence]
    out_storage = torch.full(
        (batch, sequence, 2 * heads, head_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = out_storage[:, :, 1::2]
    lse_storage = torch.full(
        (batch, heads, sequence + 50), float("nan"), dtype=torch.float32, device="cuda"
    )
    lse = lse_storage[:, :, :sequence]

    lse_merge(partial_out, partial_lse, out, lse)

    reference_out, reference_lse = _reference(partial_out, partial_lse, batch)
    torch.testing.assert_close(out, reference_out.to(out.dtype), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, reference_lse, atol=1e-5, rtol=1e-5)
    assert out_storage[:, :, 0::2].isnan().all()
    assert lse_storage[:, :, sequence:].isnan().all()


@pytest.mark.parametrize("empty_fill", [1e4, float("nan"), float("inf")])
@pytest.mark.parametrize("write_lse", [True, False])
def test_lse_merge_skips_empty_ranges(empty_fill, write_lse):
    """A range with no valid keys for a row has LSE -inf and must carry no weight, whatever its
    partial output holds: its weight is 0, and 0 * NaN or 0 * Inf would still be NaN. Range 1
    is empty for every row, some rows are empty in range 0 alone, and some in every range,
    which must give a 0 output and LSE -inf rather than NaN. 77 rows leave a partial tile.
    """
    torch.manual_seed(67)
    batch, splits, heads, sequence = 2, 3, 2, 77
    partial_out, partial_lse = _partials(
        batch, splits, sequence, heads, 128, torch.bfloat16
    )
    partial_lse[1::splits] = float("-inf")
    partial_lse[0::splits, :, :10] = float("-inf")
    partial_lse[:, :, 20:25] = float("-inf")
    partial_lse[splits : 2 * splits, 1, 40:42] = float("-inf")
    partial_out[partial_lse.isneginf().transpose(1, 2)] = empty_fill
    out = torch.full(
        (batch, sequence, heads, 128), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.full(
        (batch, heads, sequence), float("nan"), dtype=torch.float32, device="cuda"
    )

    lse_merge(partial_out, partial_lse, out, lse if write_lse else None)

    reference_out, reference_lse = _reference(partial_out, partial_lse, batch)
    reached = torch.isfinite(reference_lse)
    assert not out.isnan().any()
    assert (out.permute(0, 2, 1, 3)[~reached] == 0).all()
    torch.testing.assert_close(out.float(), reference_out, atol=1e-2, rtol=1e-2)
    if write_lse:
        assert torch.equal(lse[~reached], reference_lse[~reached])
        torch.testing.assert_close(lse[reached], reference_lse[reached])


@pytest.mark.parametrize(
    "sequence,heads,splits",
    [
        pytest.param(65536, 64, 5, id="splits-past-2^31"),
        pytest.param(147456, 128, 1, id="partial-past-2^31"),
    ],
)
def test_lse_merge_addresses_past_2_31_elements(sequence, heads, splits):
    """The last partial, or one partial alone, can lie past 2^31 elements. All weight sits on
    the last range, so the merge must return it exactly. An offset computed in 32 bits reads
    before the buffer and faults instead."""
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        pytest.skip("needs ~10 GiB free for partials past 2^31 elements")
    partial_out = torch.zeros(
        (splits, sequence, heads, 128), dtype=torch.bfloat16, device="cuda"
    )
    partial_lse = torch.full(
        (splits, heads, sequence), float("-inf"), dtype=torch.float32, device="cuda"
    )
    partial_out[-1].normal_()
    partial_lse[-1] = 0.5
    out = torch.empty((1, sequence, heads, 128), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((1, heads, sequence), dtype=torch.float32, device="cuda")

    lse_merge(partial_out, partial_lse, out, lse)

    assert torch.equal(out[0], partial_out[-1])
    assert torch.equal(lse[0], partial_lse[-1])


def _bad_call(case):
    """Arguments for one malformed call; every other argument is valid."""
    batch, splits, sequence, heads, head_dim = 2, 2, 16, 2, 128
    partial_out, partial_lse = _partials(
        batch, splits, sequence, heads, head_dim, torch.bfloat16
    )
    out = torch.empty(
        (batch, sequence, heads, head_dim), dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.empty((batch, heads, sequence), dtype=torch.float32, device="cuda")
    if case == "no key range":
        partial_out, partial_lse = partial_out[:0], partial_lse[:0]
    elif case == "does not fold":
        partial_out, partial_lse = partial_out[:3], partial_lse[:3]
    elif case == "partial_lse must be":
        partial_lse = partial_lse[:, :, :-1]
    elif case == "lse must be":
        lse = lse[:1]
    elif case == "contiguous last dimension":
        partial_out = partial_out.transpose(1, 3).contiguous().transpose(1, 3)
    elif case == "power of two":
        partial_out, out = partial_out[..., :96], out[..., :96]
    return partial_out, partial_lse, out, lse


@pytest.mark.parametrize(
    "case",
    [
        "no key range",
        "does not fold",
        "partial_lse must be",
        "lse must be",
        "contiguous last dimension",
        "power of two",
    ],
)
def test_lse_merge_rejects_malformed_arguments(case):
    with pytest.raises(ValueError, match=case):
        lse_merge(*_bad_call(case))
