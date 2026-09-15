# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton import arch_info
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

requires_gfx950 = pytest.mark.skipif(
    arch_info.get_arch() != "gfx950",
    reason="MXFP4 stochastic conversion requires gfx950",
)


def _dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = mxfp4_to_f32(packed)
    scale_f32 = e8m0_to_f32(scales).repeat_interleave(32, dim=-1)
    return values * scale_f32


def test_dynamic_mxfp4_quant_sr_requires_cuda():
    with pytest.raises(ValueError, match="2-D"):
        dynamic_mxfp4_quant(
            torch.zeros(32, dtype=torch.bfloat16),
            use_sr=True,
            philox_seed=1,
        )

    x = torch.zeros((2, 32), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        dynamic_mxfp4_quant(x, use_sr=True, philox_seed=1)


@requires_gfx950
def test_dynamic_mxfp4_quant_sr_validates_contract():
    x = torch.zeros((3, 32), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(TypeError, match="bfloat16 or float32"):
        dynamic_mxfp4_quant(x.to(torch.float16), use_sr=True, philox_seed=1)
    with pytest.raises(ValueError, match="scaling_mode='even'"):
        dynamic_mxfp4_quant(
            x,
            scaling_mode="ceil",
            use_sr=True,
            philox_seed=1,
        )
    with pytest.raises(ValueError, match="divisible by 32"):
        dynamic_mxfp4_quant(x[:, :6], use_sr=True, philox_seed=1)
    with pytest.raises(ValueError, match="philox_seed is required"):
        dynamic_mxfp4_quant(x, use_sr=True)
    with pytest.raises(ValueError, match="philox_seed must be"):
        dynamic_mxfp4_quant(x, use_sr=True, philox_seed=-1)
    counters_used = x.numel() // 8
    max_valid_offset = (1 << 63) - counters_used
    dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1,
        philox_offset=max_valid_offset,
    )
    with pytest.raises(ValueError, match="leave room"):
        dynamic_mxfp4_quant(
            x,
            use_sr=True,
            philox_seed=1,
            philox_offset=max_valid_offset + 1,
        )
    with pytest.raises(ValueError, match="only valid when use_sr=True"):
        dynamic_mxfp4_quant(x, philox_seed=1)


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 32), (6, 96), (64, 256)])
def test_dynamic_mxfp4_quant_sr_is_reproducible_and_reuses_rtn_scales(
    shape,
    dtype,
):
    torch.manual_seed(17)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    kwargs = {"use_sr": True, "philox_seed": 1234, "philox_offset": 5678}

    packed, scales = dynamic_mxfp4_quant(x, **kwargs)
    packed_repeat, scales_repeat = dynamic_mxfp4_quant(x, **kwargs)
    packed_next, scales_next = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1234,
        philox_offset=5679,
    )
    _, scales_rtn = dynamic_mxfp4_quant(x)

    assert packed.shape == (shape[0], shape[1] // 2)
    assert scales.shape == (shape[0], shape[1] // 32)
    assert packed.dtype == torch.uint8 and scales.dtype == torch.uint8
    assert scales.stride() == scales_rtn.stride()
    torch.testing.assert_close(packed, packed_repeat, atol=0, rtol=0)
    torch.testing.assert_close(scales, scales_repeat, atol=0, rtol=0)
    torch.testing.assert_close(scales, scales_next, atol=0, rtol=0)
    torch.testing.assert_close(scales, scales_rtn, atol=0, rtol=0)
    assert not torch.equal(packed, packed_next)


@requires_gfx950
def test_dynamic_mxfp4_quant_sr_bf16_exact_values_match_known_encoding():
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            0.0,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    ).repeat(2)[None, :]
    expected_packed = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0xA9, 0xCB, 0xED, 0x0F] * 2,
        dtype=torch.uint8,
        device="cuda",
    )[None, :]

    packed, scales = dynamic_mxfp4_quant(
        values,
        use_sr=True,
        philox_seed=1234,
        philox_offset=5678,
    )

    torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)
    assert torch.all(scales == 127)
    torch.testing.assert_close(
        _dequantize(packed, scales), values.float(), atol=0, rtol=0
    )


@requires_gfx950
def test_dynamic_mxfp4_quant_sr_accepts_noncontiguous_input():
    torch.manual_seed(19)
    x = torch.randn((96, 6), dtype=torch.float32, device="cuda").T
    assert not x.is_contiguous()

    actual = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=7,
        philox_offset=11,
    )
    expected = dynamic_mxfp4_quant(
        x.contiguous(),
        use_sr=True,
        philox_seed=7,
        philox_offset=11,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


@requires_gfx950
def test_dynamic_mxfp4_quant_sr_preserves_raw_zero_scale_endpoint():
    # Raw E8M0 zero represents 2^-127, not the 2^-126 minimum normal value.
    x = torch.full((32, 32), 2.0**-125, dtype=torch.float32, device="cuda")
    packed, scales = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1,
    )

    assert torch.count_nonzero(scales).item() == 0
    assert torch.all(packed == 0x66)
    torch.testing.assert_close(_dequantize(packed, scales), x, atol=0, rtol=0)


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dynamic_mxfp4_quant_sr_never_emits_e8m0_nan(dtype):
    x = torch.full(
        (2, 32),
        torch.finfo(dtype).max,
        dtype=dtype,
        device="cuda",
    )
    _, scales = dynamic_mxfp4_quant(x, use_sr=True, philox_seed=1)

    assert torch.all(scales == 254)


@requires_gfx950
def test_dynamic_mxfp4_quant_sr_uses_distinct_global_counters():
    torch.manual_seed(23)
    tile = torch.randn((32, 128), dtype=torch.bfloat16, device="cuda")
    x = tile.repeat(2, 16)

    packed, scales = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1234,
    )
    packed_high_offset, scales_high_offset = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1234,
        philox_offset=(1 << 32),
    )

    torch.testing.assert_close(scales[:, :4], scales[:, 4:8], atol=0, rtol=0)
    assert not torch.equal(packed[:, :64], packed[:, 64:128])
    torch.testing.assert_close(scales[:32], scales[32:], atol=0, rtol=0)
    assert not torch.equal(packed[:32], packed[32:])
    torch.testing.assert_close(scales, scales_high_offset, atol=0, rtol=0)
    assert not torch.equal(packed, packed_high_offset)


@requires_gfx950
def test_dynamic_mxfp4_quant_sr_rounds_midpoints_without_bias():
    torch.manual_seed(29)
    x = torch.full((256, 256), 1.25, dtype=torch.float32, device="cuda")
    x[:, 0::32] = 4.0
    x[:, 1::32] = 6.0

    packed, scales = dynamic_mxfp4_quant(
        x,
        use_sr=True,
        philox_seed=1234,
    )
    dequantized = _dequantize(packed, scales)
    midpoint_mask = torch.ones_like(x, dtype=torch.bool)
    midpoint_mask[:, 0::32] = False
    midpoint_mask[:, 1::32] = False
    midpoint_values = dequantized[midpoint_mask]
    round_up_fraction = (midpoint_values == 1.5).float().mean()

    assert packed[0, 0].item() == 0x76
    assert torch.all(scales == 127)
    assert torch.all((midpoint_values == 1.0) | (midpoint_values == 1.5))
    assert abs(round_up_fraction.item() - 0.5) < 0.02
