# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MXFP8 activation quantizer with the decode tier's (16, 16) code layout and AITER-shuffled scales.

The expected bytes are built in torch: per 32-K group the scale byte is clamp(ceil(log2(amax / 448)) + 127) from the bf16
pattern of the group maximum (the Triton MXFP8 quantizer's rule, zero / denormal / tiny maxima -> byte 0 and conversion
scale 2^-126), the code is x / 2^(byte - 127) rounded to nearest even e4m3, the codes are permuted into the weight's
(16, 16) preshuffled layout and the scales go through ``shuffle_scale_a16w4``. Whole outputs are compared byte for byte.
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_scale_a16w4

torch.set_default_device("cuda")


def ref_quant(x: torch.Tensor, rows: int):
    m, k = x.shape
    g = x.view(m, k // 32, 32)
    amax = (
        g.abs().amax(-1).view(torch.int16).to(torch.int32)
    )  # bf16 pattern of the max (sign cleared)
    t = (amax + 0x1F) >> 7
    sb = (torch.clamp(t, min=8) - 8).to(torch.uint8)
    scale = torch.ldexp(
        torch.ones_like(t, dtype=torch.float32), (torch.clamp(t, min=9) - 8 - 127)
    )
    codes = (
        (g.float() / scale[..., None])
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .view(m, k)
    )
    pad = torch.zeros(rows, k, dtype=torch.uint8)
    pad[:m] = codes
    r = pad.view(
        rows // 16, 16, k // 64, 4, 16
    )  # [row tile, mlane, 64-K chunk, 16-K chunk, byte]
    tiled = (
        r.permute(0, 2, 3, 1, 4).contiguous().view(rows, k)
    )  # (16, 16) preshuffled layout
    srows = (rows + 31) // 32 * 32
    spad = torch.zeros(srows, k // 32, dtype=torch.uint8)
    spad[:m] = sb
    return tiled, shuffle_scale_a16w4(spad.view(torch.float8_e8m0fnu), 1, False).view(
        torch.uint8
    )


@pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 only")
@pytest.mark.parametrize("k", [6144, 2048])
@pytest.mark.parametrize("m", [1, 5, 8, 16, 17, 24, 33, 48, 64, 96, 100, 128])
def test_quant_shuffled(m, k):
    from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import mxfp8_quantize_shuffled

    for sc in (0.01, 1.0, 30.0, 3000.0):
        x = (torch.randn(m, k) * sc).to(torch.bfloat16)
        if m % 3 == 0:
            x[: m // 2, ::7] = 0
        if m % 5 == 0:
            x[0] = 0
        rows = 16 * ((m + 15) // 16)
        codes, s = mxfp8_quantize_shuffled(x, rows)
        ref_c, ref_s = ref_quant(x, rows)
        assert torch.equal(
            codes.view(torch.uint8), ref_c
        ), f"codes differ (m={m} k={k} scale={sc})"
        assert torch.equal(
            s.view(-1)[: ref_s.numel()], ref_s.view(-1)
        ), f"scales differ (m={m} k={k} scale={sc})"


def test_rejects_non_bf16_and_unpadded_rpw():
    from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import (
        mxfp8_quantize_shuffled,
        quantize,
    )

    with pytest.raises(ValueError):
        mxfp8_quantize_shuffled(torch.ones(1, 256, dtype=torch.float32), 16)
    x = torch.ones(1, 256, dtype=torch.bfloat16)
    quant = torch.empty(16, 256, dtype=torch.float8_e4m3fn)
    scale = torch.empty(32, 8, dtype=torch.uint8)
    with pytest.raises(ValueError):
        quantize(x, quant, scale, shuffled=True, rpw=4)


if __name__ == "__main__":
    # CI runs op tests as `python3 <file>` (.github/scripts/aiter_test.sh): run the cases under pytest
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
