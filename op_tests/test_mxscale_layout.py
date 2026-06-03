# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks for the gfx1250 MXScale host layout helpers.

These run anywhere (no GPU, no flydsl) and guard the FlyDSL WMMA layout contract
against silent drift. K is never padded here: the B weight is prepared with the
ordinary ``shuffle_weight`` 16x16 byte preshuffle (shape-preserving), and only the
per-call M padding + E8M0 scale swizzle live in ``mxscale_layout``.
"""

import pytest
import torch

from aiter.utility import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.flydsl.mxscale_layout import (
    SCALE_BLOCK,
    SCALES_PER_WMMA,
    WMMA_DIM,
    align_up,
    preshuffle_e8m0_scale_wmma,
    preshuffle_mxscale_activation,
    preshuffle_mxscale_scale_for_kernel,
    validate_mxscale_kernel_shape,
)


def _ref_preshuffle_b(b: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Independent nested-loop reference: 16x16 byte tiles laid out
    (rows//16, cols//16, 16, 16) row-major. This is the B layout the gfx1250
    MXScale kernel consumes."""
    src = b.contiguous().view(rows, cols)
    out = torch.empty(rows // 16, cols // 16, 16, 16, dtype=b.dtype)
    for ti in range(rows // 16):
        for tj in range(cols // 16):
            out[ti, tj] = src[ti * 16 : ti * 16 + 16, tj * 16 : tj * 16 + 16]
    return out.reshape(rows, cols)


def test_generic_shuffle_matches_kernel_16x16_layout():
    """The ordinary shuffle_weight((16,16)) must produce the exact 16x16 byte
    layout the MXScale kernel expects (so MXScale needs no bespoke weight prep)."""
    rows, cols = 64, 64  # cols % 32 == 0 (generic shuffle requires BK=32)
    b = torch.arange(rows * cols, dtype=torch.int32).remainder(251).to(torch.uint8)
    b = b.view(rows, cols)
    got = shuffle_weight(b, layout=(16, 16)).view(torch.uint8)
    exp = _ref_preshuffle_b(b, rows, cols)
    assert torch.equal(got, exp)
    # Pure permutation of the input bytes.
    assert torch.equal(got.view(-1).sort().values, b.view(-1).sort().values)


def test_generic_shuffle_handles_noncontiguous():
    """Non-contiguous input must give the same result as its contiguous copy."""
    rows, cols = 32, 64
    base = torch.arange(rows * cols, dtype=torch.int32).remainder(251).to(torch.uint8)
    nc = base.view(cols, rows).t()  # shape (rows, cols) but non-contiguous
    assert not nc.is_contiguous()
    got = shuffle_weight(nc, layout=(16, 16)).view(torch.uint8)
    exp = shuffle_weight(nc.contiguous(), layout=(16, 16)).view(torch.uint8)
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


def test_preshuffle_b_scale_for_kernel_is_pure_permutation():
    # N=64, K=256 -> k_scale = 256//32 = 8. tile_n=32, n_warp=2 -> warp_tile_n=16.
    n, k_scale = 64, 8
    s = torch.arange(n * k_scale, dtype=torch.int32).remainder(127).to(torch.uint8)
    s = s.view(n, k_scale).view(dtypes.fp8_e8m0)
    out = preshuffle_mxscale_scale_for_kernel(s, tile_n=32, tile_k=128, n_warp=2)
    out_u8 = out.view(torch.uint8)
    assert out_u8.numel() == s.numel()
    assert torch.equal(
        out_u8.view(-1).sort().values, s.view(torch.uint8).view(-1).sort().values
    )


def test_preshuffle_b_scale_for_kernel_rejects_unaligned_n():
    n, k_scale = 48, 8  # 48 % tile_n(32) != 0
    s = torch.zeros((n, k_scale), dtype=torch.uint8).view(dtypes.fp8_e8m0)
    with pytest.raises(ValueError, match="tile_n=32"):
        preshuffle_mxscale_scale_for_kernel(s, tile_n=32, tile_k=128, n_warp=2)


def test_activation_pads_m_only_keeps_k():
    # M=17 -> padded to tile_m=128; K stays 256 (never padded). pack_a=1.
    m, k = 17, 256
    tile_m, tile_k, m_warp = 128, 128, 2
    a = torch.arange(m * k, dtype=torch.int32).remainder(251).to(torch.uint8)
    a = a.view(m, k).view(dtypes.fp8)
    a_scale = torch.arange(m * (k // SCALE_BLOCK), dtype=torch.int32)
    a_scale = a_scale.remainder(127).to(torch.uint8)
    a_scale = a_scale.view(m, k // SCALE_BLOCK).view(dtypes.fp8_e8m0)

    a_p, a_s_p = preshuffle_mxscale_activation(
        a, a_scale, data_format="fp8", tile_m=tile_m, tile_k=tile_k, m_warp=m_warp
    )
    # A is padded in M only; K is unchanged. Padded rows are zero.
    assert tuple(a_p.shape) == (align_up(m, tile_m), k)
    a_p_u8 = a_p.view(torch.uint8)
    assert torch.equal(a_p_u8[:m], a.view(torch.uint8))
    assert torch.all(a_p_u8[m:] == 0)
    # A_scale is padded in M (with E8M0 1.0) then swizzled -> pure permutation
    # of the padded scale (which contains the original bytes + E8M0_ONE pad).
    assert a_s_p.numel() == align_up(m, tile_m) * (k // SCALE_BLOCK)


def test_validate_mxscale_kernel_shape_rejects_bad_shapes():
    base = dict(N=64, tile_n=32, tile_k=128, num_buffers=2, split_k=1)
    # OK: K divisible by tile_k*split_k and enough K tiles.
    validate_mxscale_kernel_shape(K=256, **base)
    # K not divisible by WMMA_K=128 -> reject up front (no padding).
    with pytest.raises(ValueError, match="WMMA_K=128"):
        validate_mxscale_kernel_shape(K=160, **base)
    # N not divisible by tile_n -> reject.
    with pytest.raises(ValueError, match="tile_n=32"):
        validate_mxscale_kernel_shape(
            N=48, K=256, tile_n=32, tile_k=128, num_buffers=2, split_k=1
        )
    # num_buffers needs >= that many K tiles per split.
    with pytest.raises(ValueError, match="num_buffers=4"):
        validate_mxscale_kernel_shape(
            N=64, K=256, tile_n=32, tile_k=128, num_buffers=4, split_k=1
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
