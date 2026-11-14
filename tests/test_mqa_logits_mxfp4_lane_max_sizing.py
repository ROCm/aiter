# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""mqa_logits_mxfp4's lane maxima sizing check against the kernels' slot addressing -- no GPU.

pa_mqa_logits_mxfp4.h: row r's slots start at element (row start / 16) + 64 r and it has
64 * ceil(row_ends[r] / 1024) of them, the row start being r * out.stride(0) for a 2-D out.
The check must size the buffer from that address span, so a padded row stride cannot let
the kernels store past an accepted allocation.
"""

import itertools

import pytest
import torch

M = pytest.importorskip("aiter.ops.opus.pa_mqa_logits_mxfp4")


def _slots_end(out_stride: int, row_ends: list[int]) -> int:
    """One past the last lane maxima slot the kernels store for these rows."""
    return max(
        r * out_stride // 16 + 64 * r + 64 * -(-end // 1024)
        for r, end in enumerate(row_ends)
    )


def _strided_out(rows: int, width: int, stride: int) -> torch.Tensor:
    return torch.empty_strided((rows, width), (stride, 1), dtype=torch.float32)


def _lane_max(n: int) -> torch.Tensor:
    return torch.empty(n, dtype=torch.float32)


def test_padded_row_stride_rejects_numel_sized_buffer():
    # Two full 4096-key rows 65536 elements apart: the second row's slots are
    # [4160, 4416), past a buffer sized from out.numel() (704 elements).
    out = _strided_out(2, 4096, 65536)
    assert _slots_end(65536, [4096, 4096]) == 4416
    numel_sized = M.mqa_logits_mxfp4_lane_max_elems(out.numel(), 2)
    assert numel_sized == 704
    with pytest.raises(ValueError, match="lane_max must be"):
        M._check_lane_max(_lane_max(numel_sized), out, 2, flat=False)
    need = M.mqa_logits_mxfp4_lane_max_elems(M._lane_max_out_span(out, 2, False), 2)
    assert need >= 4416
    M._check_lane_max(_lane_max(need), out, 2, flat=False)
    with pytest.raises(ValueError, match="lane_max must be"):
        M._check_lane_max(_lane_max(need - 1), out, 2, flat=False)


@pytest.mark.parametrize(
    "rows,width,pad",
    list(itertools.product([1, 2, 3, 7], [64, 1000, 1024, 1025, 4096], [0, 64, 61440])),
)
def test_accepted_buffer_covers_every_slot(rows, width, pad):
    stride = -(-width // 64) * 64 + pad
    out = _strided_out(rows, width, stride)
    need = M.mqa_logits_mxfp4_lane_max_elems(
        M._lane_max_out_span(out, rows, False), rows
    )
    M._check_lane_max(_lane_max(need), out, rows, flat=False)
    assert need >= _slots_end(stride, [width] * rows)


def test_contiguous_out_sizing_unchanged():
    out = torch.empty(4, 2048, dtype=torch.float32)
    assert M._lane_max_out_span(out, 4, False) == out.numel()
    flat = torch.empty(10000, dtype=torch.float32)
    assert M._lane_max_out_span(flat, 3, True) == flat.numel()


def test_misaligned_row_stride_rejected():
    out = _strided_out(2, 100, 100)
    with pytest.raises(ValueError, match="aligned"):
        M._check_lane_max(_lane_max(1 << 12), out, 2, flat=False)
