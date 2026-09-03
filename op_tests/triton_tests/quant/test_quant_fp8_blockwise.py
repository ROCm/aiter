# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for FP8 block-wise quantization Triton kernels.

Oracle: dequantize the kernel output with the kernel-returned (inverse) scales
and compare against the original tensor within an FP8-precision tolerance. This
is an independent check — the reference does not re-run the kernel's rounding —
so a scale-layout or index-overflow bug produces a large error rather than a
silently agreeing mirror.
"""

import math

import pytest
import torch

from aiter.ops.triton.quant.quant_fp8_blockwise import (
    quant_fp8_blockwise,
    quant_fp8_blockwise_for_act_grad,
    quant_fp8_blockwise_for_weight,
    quant_fp8_blockwise_segment_m,
    requant_fp8_row_to_col,
)

# e4m3 keeps 3 mantissa bits, so per-element relative quant error tops out
# around 2^-3; allow headroom for the block-shared scale.
_REL = 0.16


def _assert_fp8_close(dequant, ref):
    """Functional FP8 check: bounded by the per-tensor magnitude, not exact."""
    scale = ref.abs().max().clamp_min(1e-4)
    err = (dequant.float() - ref.float()).abs().max()
    assert err <= _REL * scale, f"max abs err {err:.4f} > {_REL} * {scale:.4f}"


@pytest.mark.parametrize("M, N", [(128, 128), (256, 384), (200, 300), (130, 70)])
@pytest.mark.parametrize("axis", [0, 1])
def test_quant_fp8_blockwise_roundtrip(M, N, axis):
    """quant_fp8_blockwise output dequantizes back to the input."""
    torch.manual_seed(0)
    bs = 128
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 3.0

    x_fp8, scales = quant_fp8_blockwise(x, block_size=bs, axis=axis)
    assert x_fp8.shape == (M, N)
    assert x_fp8.dtype == torch.float8_e4m3fnuz

    deq = x_fp8.float().clone()
    if axis == 1:  # one scale per (row, col-block)
        assert scales.shape == (M, math.ceil(N / bs))
        for j in range(math.ceil(N / bs)):
            deq[:, j * bs : (j + 1) * bs] *= scales[:, j : j + 1]
    else:  # one scale per (row-block, col)
        assert scales.shape == (math.ceil(M / bs), N)
        for i in range(math.ceil(M / bs)):
            deq[i * bs : (i + 1) * bs, :] *= scales[i : i + 1, :]

    _assert_fp8_close(deq, x)


@pytest.mark.parametrize("B, M, N", [(1, 128, 128), (4, 256, 384), (3, 130, 70)])
def test_quant_fp8_blockwise_for_weight(B, M, N):
    """Batched weight quant: exercises the per-batch offset (int64) path."""
    torch.manual_seed(1)
    bs = 128
    w = torch.randn(B, M, N, device="cuda", dtype=torch.bfloat16) * 2.0

    w_fp8, scales = quant_fp8_blockwise_for_weight(w, block_size=bs)
    assert w_fp8.shape == (B, M, N)
    assert scales.shape == (B, math.ceil(M / bs), math.ceil(N / bs))

    deq = w_fp8.float().clone()
    for b in range(B):
        for i in range(math.ceil(M / bs)):
            for j in range(math.ceil(N / bs)):
                deq[b, i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] *= scales[b, i, j]
    _assert_fp8_close(deq, w)


def test_quant_fp8_blockwise_for_weight_2d_input():
    """A 2-D weight is treated as batch size 1."""
    torch.manual_seed(2)
    w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    w_fp8, scales = quant_fp8_blockwise_for_weight(w, block_size=128)
    assert w_fp8.shape == (1, 256, 256)
    assert scales.shape == (1, 2, 2)


@pytest.mark.parametrize("M, N", [(128, 256), (200, 300)])
def test_quant_fp8_blockwise_for_act_grad(M, N):
    """Dual row+col quant: both copies dequantize back to the input."""
    torch.manual_seed(3)
    bs = 128
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 2.0

    x_row, s_row, x_col, s_col = quant_fp8_blockwise_for_act_grad(x, block_size=bs)
    assert x_row.shape == (M, N) and x_col.shape == (M, N)
    assert s_row.shape == (M, math.ceil(N / bs))
    assert s_col.shape == (math.ceil(M / bs), N)

    deq_row = x_row.float().clone()
    for j in range(math.ceil(N / bs)):
        deq_row[:, j * bs : (j + 1) * bs] *= s_row[:, j : j + 1]
    _assert_fp8_close(deq_row, x)

    deq_col = x_col.float().clone()
    for i in range(math.ceil(M / bs)):
        deq_col[i * bs : (i + 1) * bs, :] *= s_col[i : i + 1, :]
    _assert_fp8_close(deq_col, x)


@pytest.mark.parametrize("seg_lens", [[128, 256, 64], [100, 200, 50], [300]])
def test_quant_fp8_blockwise_segment_m(seg_lens):
    """Segmented col-wise quant: each segment dequantizes back independently."""
    torch.manual_seed(5)
    bs = 128
    N = 256
    batch_size = len(seg_lens)
    M = sum(seg_lens)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 2.0

    seg_indptr = torch.tensor(
        [0] + list(torch.tensor(seg_lens).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )
    blocks_per_seg = [math.ceil(length / bs) for length in seg_lens]
    scales_seg_indptr = torch.tensor(
        [0] + list(torch.tensor(blocks_per_seg).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )

    x_fp8, scales = quant_fp8_blockwise_segment_m(
        x, batch_size, seg_indptr, scales_seg_indptr, block_size=bs
    )
    assert x_fp8.shape == (M, N)
    assert x_fp8.dtype == torch.float8_e4m3fnuz

    # Dequant each segment's row-blocks with its col scales and compare.
    deq = x_fp8.float().clone()
    for b in range(batch_size):
        s = seg_indptr[b].item()
        e = seg_indptr[b + 1].item()
        base = scales_seg_indptr[b].item()
        for r in range(math.ceil((e - s) / bs)):
            rs = s + r * bs
            re = min(e, rs + bs)
            deq[rs:re, :] *= scales[base + r, :]
    _assert_fp8_close(deq, x)


@pytest.mark.parametrize("M, K", [(128, 256), (200, 384)])
def test_requant_fp8_row_to_col(M, K):
    """Row->col requant preserves the dequantized values within FP8 tolerance."""
    torch.manual_seed(4)
    bs = 128
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0

    # Start from a row-wise blockwise FP8 tensor.
    x_row, s_row = quant_fp8_blockwise(x, block_size=bs, axis=1)
    y_col, s_col = requant_fp8_row_to_col(x_row, s_row, block_size=bs)
    assert y_col.shape == (M, K)
    assert s_col.shape == (math.ceil(M / bs), K)

    # Dequant the row source and the col result; they must agree (double-quant
    # only adds a second FP8 rounding).
    deq_row = x_row.float().clone()
    for j in range(math.ceil(K / bs)):
        deq_row[:, j * bs : (j + 1) * bs] *= s_row[:, j : j + 1]
    deq_col = y_col.float().clone()
    for i in range(math.ceil(M / bs)):
        deq_col[i * bs : (i + 1) * bs, :] *= s_col[i : i + 1, :]

    _assert_fp8_close(deq_col, deq_row)


def test_public_package_import():
    """The wrappers must be importable from the quant package, not just the module."""
    from aiter.ops.triton.quant import (
        quant_fp8_blockwise as pkg_blockwise,
    )
    from aiter.ops.triton.quant import (
        quant_fp8_blockwise_for_act_grad,
        quant_fp8_blockwise_for_weight,
        quant_fp8_blockwise_segment_m,
    )
    from aiter.ops.triton.quant import (
        requant_fp8_row_to_col as pkg_requant,
    )

    for fn in (
        pkg_blockwise,
        quant_fp8_blockwise_for_act_grad,
        quant_fp8_blockwise_for_weight,
        quant_fp8_blockwise_segment_m,
        pkg_requant,
    ):
        assert callable(fn)


def test_quant_fp8_blockwise_rejects_bad_inputs():
    """Non-contiguous input, out-of-range fp8_max, and non-pow2 block must raise."""
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        quant_fp8_blockwise(x.t())  # non-contiguous
    with pytest.raises(AssertionError):
        quant_fp8_blockwise(x, fp8_max=1e9)  # above e4m3fnuz max
    with pytest.raises(AssertionError):
        quant_fp8_blockwise(x, block_size=100)  # not a power of two
    with pytest.raises(AssertionError):
        quant_fp8_blockwise(x, axis=2)  # invalid axis


def test_quant_fp8_blockwise_for_weight_rejects_noncontiguous():
    w = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    with pytest.raises(AssertionError):
        quant_fp8_blockwise_for_weight(w)
