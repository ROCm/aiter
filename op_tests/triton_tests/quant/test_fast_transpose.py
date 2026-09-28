# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for fast transpose Triton kernels."""

import pytest
import torch

from aiter.ops.triton.quant.fast_transpose import (
    fast_transpose_2d,
    transpose_packed_fp4,
)


@pytest.mark.parametrize(
    "M, N",
    [
        (1, 1),
        (32, 32),
        (64, 128),
        (128, 64),
        (1024, 512),
        (511, 257),  # non-power-of-2
        (1, 4096),
        (4096, 1),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fnuz],
)
def test_fast_transpose_2d_correctness(M, N, dtype):
    torch.manual_seed(0)
    if dtype == torch.float8_e4m3fnuz:
        x = torch.randn(M, N, device="cuda").to(dtype)
        ref = x.view(torch.int8).t().contiguous().view(dtype)
    else:
        x = torch.randn(M, N, dtype=dtype, device="cuda")
        ref = x.t().contiguous()

    out = fast_transpose_2d(x)

    assert out.shape == (N, M), f"shape mismatch: {out.shape} vs {(N, M)}"
    assert out.is_contiguous(), "output is not contiguous"
    assert out.dtype == dtype

    if dtype == torch.float8_e4m3fnuz:
        assert torch.equal(
            out.view(torch.int8), ref.view(torch.int8)
        ), "FP8 bit pattern mismatch"
    else:
        torch.testing.assert_close(out, ref)


def _reference_packed_fp4_transpose(data_fp4: torch.Tensor) -> torch.Tensor:
    data_fp4 = data_fp4.view(torch.uint8)
    low_nibbles = data_fp4 & 0x0F
    high_nibbles = (data_fp4 >> 4) & 0x0F
    unpacked = torch.stack((low_nibbles, high_nibbles), dim=-1).flatten(-2)
    transposed = unpacked.t().contiguous()
    return transposed[:, 0::2] | (transposed[:, 1::2] << 4)


def test_transpose_packed_fp4_preserves_nibble_order():
    logical = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 0, 1],
            [2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.uint8,
        device="cuda",
    )
    data_fp4 = logical[:, 0::2] | (logical[:, 1::2] << 4)
    expected_logical = logical.t().contiguous()
    expected = expected_logical[:, 0::2] | (expected_logical[:, 1::2] << 4)

    output = transpose_packed_fp4(data_fp4)

    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1),
        (4, 3),
        (6, 32),
        (32, 17),
        (64, 32),
        (70, 65),
        (256, 128),
    ],
)
def test_transpose_packed_fp4_matches_reference(shape):
    torch.manual_seed(0)
    data_fp4 = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    output = transpose_packed_fp4(data_fp4)
    expected = _reference_packed_fp4_transpose(data_fp4)

    assert output.shape == (shape[1] * 2, shape[0] // 2)
    assert output.is_contiguous()
    assert output.dtype == torch.uint8
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_transpose_packed_fp4_supports_strided_input():
    torch.manual_seed(1)
    storage = torch.randint(0, 256, (70, 130), dtype=torch.uint8, device="cuda")
    data_fp4 = storage[:, ::2]
    assert not data_fp4.is_contiguous()

    output = transpose_packed_fp4(data_fp4)

    torch.testing.assert_close(
        output, _reference_packed_fp4_transpose(data_fp4), atol=0, rtol=0
    )


def test_transpose_packed_fp4_with_large_unused_stride():
    # A singleton dimension preserves large stride metadata without a 4 GiB allocation.
    huge_stride = (1 << 32) + 1
    input_storage = torch.tensor([0x21, 0x43], dtype=torch.uint8, device="cuda")
    data_fp4 = torch.as_strided(input_storage, (2, 1), (1, huge_stride))
    assert data_fp4.stride(1) == huge_stride

    output = transpose_packed_fp4(data_fp4)

    expected = torch.tensor([[0x31], [0x42]], dtype=torch.uint8, device="cuda")
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="native packed FP4 dtype is unavailable",
)
def test_transpose_packed_fp4_preserves_native_dtype():
    torch.manual_seed(2)
    data_bytes = torch.randint(0, 256, (34, 17), dtype=torch.uint8, device="cuda")
    data_fp4 = data_bytes.view(torch.float4_e2m1fn_x2)

    output = transpose_packed_fp4(data_fp4)

    assert output.dtype == torch.float4_e2m1fn_x2
    torch.testing.assert_close(
        output.view(torch.uint8),
        _reference_packed_fp4_transpose(data_bytes),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("shape", [(2, 1), (6, 17), (64, 32), (70, 65)])
def test_transpose_packed_fp4_round_trip(shape):
    torch.manual_seed(2)
    data_fp4 = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    restored = transpose_packed_fp4(transpose_packed_fp4(data_fp4))

    torch.testing.assert_close(restored, data_fp4, atol=0, rtol=0)


def test_transpose_packed_fp4_validates_input():
    with pytest.raises(ValueError, match="must be 2-D"):
        transpose_packed_fp4(torch.zeros((2, 2, 2), dtype=torch.uint8, device="cpu"))
    with pytest.raises(TypeError, match="torch.uint8"):
        transpose_packed_fp4(torch.zeros((2, 2), dtype=torch.float32, device="cpu"))
    with pytest.raises(ValueError, match="non-zero"):
        transpose_packed_fp4(torch.empty((2, 0), dtype=torch.uint8, device="cpu"))
    with pytest.raises(ValueError, match="rows must be even"):
        transpose_packed_fp4(torch.zeros((3, 2), dtype=torch.uint8, device="cpu"))
