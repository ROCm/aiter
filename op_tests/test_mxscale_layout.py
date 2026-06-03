# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks for the gfx1250 MXScale host preshuffle/padding helpers.

These run anywhere (no GPU, no flydsl) and guard the FlyDSL WMMA layout contract
against silent drift — including the non-contiguous-input case that a GPU-only
end-to-end test would miss.
"""

import pytest
import torch

from aiter.utility import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.flydsl.mxscale_gemm import (
    _prepare_b_scale_for_mxscale_kernel,
    shuffle_weight_mxscale,
)
from aiter.ops.flydsl.mxscale_layout import (
    E8M0_ONE,
    MXSCALE_B_LAYOUT,
    MXSCALE_B_PAD_K_MIN,
    MXSCALE_B_SCALE_LAYOUT,
    SCALE_BLOCK,
    SCALES_PER_WMMA,
    WMMA_DIM,
    align_up,
    get_padded_problem_shape,
    pad_scale_mxscale,
    pad_weight_data_mxscale,
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


def _mxscale_weight(n=64, k=160):
    w = torch.arange(n * k, dtype=torch.int32).remainder(251).to(torch.uint8)
    w = w.view(n, k).view(dtypes.fp8)
    s_cols = k // SCALE_BLOCK
    scale = torch.arange(n * s_cols, dtype=torch.int32).remainder(127).to(torch.uint8)
    scale = scale.view(n, s_cols).view(dtypes.fp8_e8m0)
    return w, scale


def test_mxscale_weight_and_scale_keep_n_and_pad_k():
    w, scale = _mxscale_weight(n=64, k=160)
    w_pad = pad_weight_data_mxscale(w, data_format="fp8")
    prepared_n = 64
    k_pad = max(align_up(160, 128), MXSCALE_B_PAD_K_MIN)
    scale_pad = pad_scale_mxscale(
        scale,
        data_format="fp8",
        logical_n=64,
        logical_k=160,
        prepared_n=prepared_n,
        padded_k=k_pad,
    )

    assert tuple(w_pad.shape) == (prepared_n, k_pad)
    assert tuple(scale_pad.shape) == (prepared_n, k_pad // SCALE_BLOCK)
    assert getattr(w_pad, "_mxscale_layout") == "rowmajor_kpad_v1"
    assert getattr(scale_pad, "_mxscale_layout") == MXSCALE_B_SCALE_LAYOUT
    assert getattr(w_pad, "_mxscale_logical_shape") == (64, 160)
    assert getattr(w_pad, "_mxscale_padded_shape") == (prepared_n, k_pad)

    w_u8 = w_pad.view(torch.uint8)
    scale_u8 = scale_pad.view(torch.uint8)
    assert torch.equal(w_u8[:, :160], w.view(torch.uint8))
    assert torch.all(w_u8[:, 160:] == 0)
    assert torch.equal(scale_u8[:, : 160 // SCALE_BLOCK], scale.view(torch.uint8))
    assert torch.all(scale_u8[:, 160 // SCALE_BLOCK :] == E8M0_ONE)


def test_get_padded_problem_shape_respects_num_buffers():
    shape = get_padded_problem_shape(
        "fp8", M=17, N=64, K=160, tile_m=128, tile_n=32, tile_k=128, num_buffers=2
    )
    assert shape["M"] == 128
    assert shape["N"] == 64
    assert shape["K"] == MXSCALE_B_PAD_K_MIN

    with pytest.raises(ValueError, match="num_buffers=4"):
        get_padded_problem_shape(
            "fp8",
            M=17,
            N=64,
            K=160,
            tile_m=128,
            tile_n=32,
            tile_k=128,
            num_buffers=4,
        )

    with pytest.raises(ValueError, match="N=64.*tile_n=128"):
        get_padded_problem_shape(
            "fp8",
            M=17,
            N=64,
            K=160,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            num_buffers=2,
        )


def test_shuffle_weight_mxscale_rejects_unaligned_n():
    w, _ = _mxscale_weight(n=17, k=160)
    with pytest.raises(ValueError, match="N=17.*16x16"):
        shuffle_weight(w, mxscale_data_format="fp8")


def test_shuffle_weight_mxscale_uses_stable_b_layout_only():
    w, scale = _mxscale_weight(n=64, k=160)
    w_pad = pad_weight_data_mxscale(w, data_format="fp8")
    prepared_n, k_pad = w_pad._mxscale_padded_shape
    scale_pad = pad_scale_mxscale(
        scale,
        data_format="fp8",
        logical_n=64,
        logical_k=160,
        prepared_n=prepared_n,
        padded_k=k_pad,
    )
    w_shuf = shuffle_weight_mxscale(w, data_format="fp8")
    prepared_n, k_pad = w_shuf._mxscale_padded_shape

    expected = preshuffle_b_16x16(w_pad, prepared_n, k_pad)
    assert torch.equal(w_shuf.view(torch.uint8), expected.view(torch.uint8))
    assert getattr(w_shuf, "is_shuffled", False) is True
    assert getattr(w_shuf, "_mxscale_layout") == MXSCALE_B_LAYOUT
    assert getattr(scale_pad, "_mxscale_layout") == MXSCALE_B_SCALE_LAYOUT

    w_api = shuffle_weight(w, mxscale_data_format="fp8")
    assert torch.equal(w_api.view(torch.uint8), w_shuf.view(torch.uint8))


def test_shuffle_weight_mxscale_rejects_old_tile_arguments():
    w, _ = _mxscale_weight(n=64, k=160)
    try:
        shuffle_weight_mxscale(w, data_format="fp8", tile_n=128)
    except TypeError:
        return
    raise AssertionError("shuffle_weight_mxscale must not accept runtime tile args")


def test_bpreshuffle_rejects_raw_mxscale_weight():
    import aiter

    w, scale = _mxscale_weight(n=64, k=160)
    a = torch.empty((3, 160), dtype=dtypes.fp8)
    a_scale = torch.empty((3, 160 // SCALE_BLOCK), dtype=dtypes.fp8_e8m0)

    with pytest.raises(ValueError, match="shuffle_weight"):
        aiter.gemm_a8w8_bpreshuffle(a, w, a_scale, scale)


def test_b_scale_kernel_prepare_is_cached_per_kernel():
    w, scale = _mxscale_weight(n=64, k=160)
    w_prepared = shuffle_weight(w, mxscale_data_format="fp8")
    prepared_n, k_pad = w_prepared._mxscale_padded_shape
    scale_prepared = pad_scale_mxscale(
        scale,
        data_format="fp8",
        logical_n=64,
        logical_k=160,
        prepared_n=prepared_n,
        padded_k=k_pad,
    )

    first = _prepare_b_scale_for_mxscale_kernel(
        w_prepared,
        scale,
        data_format="fp8",
        tile_n=32,
        tile_k=128,
        n_warp=2,
    )
    second = _prepare_b_scale_for_mxscale_kernel(
        w_prepared,
        scale,
        data_format="fp8",
        tile_n=32,
        tile_k=128,
        n_warp=2,
    )
    other = _prepare_b_scale_for_mxscale_kernel(
        w_prepared,
        scale,
        data_format="fp8",
        tile_n=64,
        tile_k=128,
        n_warp=2,
    )
    prepared = _prepare_b_scale_for_mxscale_kernel(
        w_prepared,
        scale_prepared,
        data_format="fp8",
        tile_n=32,
        tile_k=128,
        n_warp=2,
    )

    assert first.data_ptr() == second.data_ptr()
    assert first.data_ptr() != other.data_ptr()
    assert torch.equal(first.view(torch.uint8), prepared.view(torch.uint8))


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
