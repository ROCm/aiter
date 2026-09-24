# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.quant import dynamic_mxfp4_quant_blockscale
from aiter.utility.fp4_utils import (
    e8m0_to_f32,
    f32_to_mx_e8m0_scale,
    f32_to_mxfp4,
)
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

_BLOCK_SIZE = 32


def _torch_dynamic_mxfp4_quant_blockscale(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference matching the Triton 32x32 EVEN-scale quantizer."""
    # Keep implicit constructors on CPU when the default device changes.
    with torch.device("cpu"):
        x = x.float().cpu()
        M, N = x.shape
        tiles = (
            x.reshape(M // _BLOCK_SIZE, _BLOCK_SIZE, N // _BLOCK_SIZE, _BLOCK_SIZE)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        amax = tiles.abs().amax(dim=(-2, -1))

        scales = f32_to_mx_e8m0_scale(
            amax,
            mode=MxScaleRoundModeInt.Even,
            dtype=MxDtypeInt.FP4_E2M1,
        ).view(torch.uint8)
        scales = scales.clamp_max(254)
        scale_f32 = e8m0_to_f32(scales).float()

        scaled_tiles = tiles / scale_f32[:, :, None, None]
        scaled = scaled_tiles.permute(0, 2, 1, 3).reshape(M, N)
        packed = f32_to_mxfp4(scaled).view(torch.uint8)
    return packed, scales


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).flatten(-2)


@pytest.mark.parametrize(
    "M,N",
    [
        (32, 32),
        (32, 96),
        (64, 32),
        (64, 64),
        (96, 160),
        (256, 256),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dynamic_mxfp4_quant_blockscale_matches_reference(M: int, N: int, dtype):
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype, device="cuda") * 4.0

    actual_packed, actual_scales = dynamic_mxfp4_quant_blockscale(x)
    expected_packed, expected_scales = _torch_dynamic_mxfp4_quant_blockscale(x)

    assert actual_packed.shape == (M, N // 2)
    assert actual_scales.shape == (M // _BLOCK_SIZE, N // _BLOCK_SIZE)
    assert actual_packed.dtype == torch.uint8
    assert actual_scales.dtype == torch.uint8
    assert actual_packed.is_contiguous()
    assert actual_scales.is_contiguous()
    torch.testing.assert_close(actual_packed.cpu(), expected_packed, atol=0, rtol=0)
    torch.testing.assert_close(actual_scales.cpu(), expected_scales, atol=0, rtol=0)


def test_dynamic_mxfp4_quant_blockscale_even_scale_tie():
    below_tie = torch.nextafter(
        torch.tensor(1.75, device="cpu"), torch.tensor(0.0, device="cpu")
    ).item()
    x = torch.zeros((64, 32), dtype=torch.float32, device="cuda")
    x[0, 0] = below_tie
    x[32, 0] = 1.75

    _, scales = dynamic_mxfp4_quant_blockscale(x, scaling_mode="even")

    # EVEN uses the 1.75x threshold when rounding amax to a power of two.
    torch.testing.assert_close(
        scales.cpu(),
        torch.tensor([[125], [126]], dtype=torch.uint8, device="cpu"),
        atol=0,
        rtol=0,
    )


def test_dynamic_mxfp4_quant_blockscale_payload_rne_ties():
    positive_ties = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device="cuda",
    )
    x = torch.zeros((32, 32), dtype=torch.float32, device="cuda")
    x[0, : positive_ties.numel()] = positive_ties
    x[0, 8 : 8 + positive_ties.numel()] = -positive_ties

    packed, scales = dynamic_mxfp4_quant_blockscale(x)
    codes = _unpack_nibbles(packed)[0]

    assert scales.item() == 127
    expected_positive = torch.tensor(
        [0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8, device="cuda"
    )
    torch.testing.assert_close(
        codes[: expected_positive.numel()], expected_positive, atol=0, rtol=0
    )
    torch.testing.assert_close(
        codes[8 : 8 + expected_positive.numel()],
        expected_positive | 0x08,
        atol=0,
        rtol=0,
    )


def test_dynamic_mxfp4_quant_blockscale_is_transpose_invariant():
    torch.manual_seed(37)
    x = torch.randn((64, 96), dtype=torch.float32, device="cuda")

    packed, scales = dynamic_mxfp4_quant_blockscale(x)
    packed_t, scales_t = dynamic_mxfp4_quant_blockscale(x.t().contiguous())

    codes = _unpack_nibbles(packed)
    codes_t = _unpack_nibbles(packed_t)
    torch.testing.assert_close(codes_t, codes.t().contiguous(), atol=0, rtol=0)
    torch.testing.assert_close(scales_t, scales.t().contiguous(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dynamic_mxfp4_quant_blockscale_e8m0_endpoints(dtype):
    zeros = torch.zeros((32, 32), dtype=dtype, device="cuda")
    zero_packed, zero_scale = dynamic_mxfp4_quant_blockscale(zeros)
    assert zero_scale.item() == 0
    assert torch.count_nonzero(zero_packed).item() == 0

    smallest = torch.full((32, 32), 2.0**-125, dtype=dtype, device="cuda")
    smallest_packed, smallest_scale = dynamic_mxfp4_quant_blockscale(smallest)
    assert smallest_scale.item() == 0
    assert torch.all(smallest_packed == 0x66)

    largest = torch.full((32, 32), torch.finfo(dtype).max, dtype=dtype, device="cuda")
    largest_packed, largest_scale = dynamic_mxfp4_quant_blockscale(largest)
    assert largest_scale.item() == 254
    assert torch.all(largest_packed == 0x44)
    assert torch.count_nonzero(largest_scale == 0xFF).item() == 0


@pytest.mark.parametrize("shape", [(32,), (1, 32, 32)])
def test_dynamic_mxfp4_quant_blockscale_rejects_non_2d(shape):
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="must be 2-D"):
        dynamic_mxfp4_quant_blockscale(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_dynamic_mxfp4_quant_blockscale_rejects_dtype(dtype):
    x = torch.empty((32, 32), dtype=dtype, device="cuda")
    with pytest.raises(TypeError, match="torch.bfloat16 or torch.float32"):
        dynamic_mxfp4_quant_blockscale(x)


@pytest.mark.parametrize("shape", [(0, 32), (32, 0)])
def test_dynamic_mxfp4_quant_blockscale_rejects_empty_dimensions(shape):
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="must be non-zero"):
        dynamic_mxfp4_quant_blockscale(x)


@pytest.mark.parametrize("shape", [(31, 32), (32, 33), (64, 48)])
def test_dynamic_mxfp4_quant_blockscale_rejects_ragged_tiles(shape):
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="divisible by 32"):
        dynamic_mxfp4_quant_blockscale(x)


def test_dynamic_mxfp4_quant_blockscale_rejects_noncontiguous_input():
    x = torch.empty((32, 64), dtype=torch.bfloat16, device="cuda")[:, ::2]
    assert not x.is_contiguous()
    with pytest.raises(ValueError, match="must be contiguous"):
        dynamic_mxfp4_quant_blockscale(x)


def test_dynamic_mxfp4_quant_blockscale_rejects_unknown_scaling_mode():
    x = torch.empty((32, 32), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="scaling_mode must be 'even'"):
        dynamic_mxfp4_quant_blockscale(x, scaling_mode="ceil")
