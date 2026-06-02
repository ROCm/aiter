# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks for the gfx1250 MXScale host preshuffle/padding helpers.

These run anywhere (no GPU, no flydsl) and guard the FlyDSL WMMA layout contract
against silent drift — including the non-contiguous-input case that a GPU-only
end-to-end test would miss.
"""

import torch

from aiter.ops.flydsl.mxscale_layout import (
    SCALE_BLOCK,
    SCALES_PER_WMMA,
    WMMA_DIM,
    preshuffle_b_16x16,
    preshuffle_e8m0_scale_wmma,
)


def _ref_preshuffle_b(b: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Independent nested-loop reference: 16x16 byte tiles laid out
    (rows//16, cols//16, 16, 16) row-major."""
    src = b.contiguous().view(rows, cols)
    out = torch.empty(rows // 16, cols // 16, 16, 16, dtype=b.dtype)
    for ti in range(rows // 16):
        for tj in range(cols // 16):
            out[ti, tj] = src[ti * 16 : ti * 16 + 16, tj * 16 : tj * 16 + 16]
    return out.reshape(rows, cols)


def test_preshuffle_b_matches_reference():
    rows, cols = 64, 48
    b = torch.arange(rows * cols, dtype=torch.int32).remainder(251).to(torch.uint8)
    b = b.view(rows, cols)
    got = preshuffle_b_16x16(b, rows, cols)
    exp = _ref_preshuffle_b(b, rows, cols)
    assert torch.equal(got, exp)
    # It must be a pure permutation of the input bytes.
    assert torch.equal(got.view(-1).sort().values, b.view(-1).sort().values)


def test_preshuffle_b_handles_noncontiguous():
    """Non-contiguous input must give the same result as its contiguous copy."""
    rows, cols = 32, 32
    base = torch.arange(rows * cols, dtype=torch.int32).remainder(251).to(torch.uint8)
    nc = base.view(cols, rows).t()  # shape (rows, cols) but non-contiguous
    assert not nc.is_contiguous()
    got = preshuffle_b_16x16(nc, rows, cols)
    exp = preshuffle_b_16x16(nc.contiguous(), rows, cols)
    assert torch.equal(got, exp)


def test_preshuffle_e8m0_scale_shape_and_permutation():
    # warp_tile=64 -> wmma_rep=4; scale_k_per_tile=4 -> k_wmma_steps=1.
    rows, k_scale = 128, 16
    warp_tile = 64
    s = torch.arange(rows * k_scale, dtype=torch.int32).remainder(251).to(torch.uint8)
    s = s.view(rows, k_scale)
    out = preshuffle_e8m0_scale_wmma(s, warp_tile, scale_k_per_tile=4)
    # Output is a reshape/permute -> same element count, pure permutation.
    assert out.numel() == s.numel()
    assert torch.equal(out.view(-1).sort().values, s.view(-1).sort().values)
    assert SCALE_BLOCK == 32 and WMMA_DIM == 16 and SCALES_PER_WMMA == 4


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
