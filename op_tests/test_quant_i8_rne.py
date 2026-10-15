# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.quant import per_token_quant_i8_rne_hip


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="HIP INT8 quantization requires a ROCm GPU",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_per_token_quant_i8_rne_rounds_ties_and_handles_zero_rows(dtype):
    x = torch.zeros((2, 256), device="cuda", dtype=dtype)
    x[1, :9] = torch.tensor(
        [20.5, 21.5, 62.5, 63.5, -20.5, -21.5, -62.5, -63.5, 127.0],
        device="cuda",
        dtype=dtype,
    )

    quantized, scale = per_token_quant_i8_rne_hip(x)

    assert quantized.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert scale.shape == (2, 1)
    assert scale[0].item() == 0.0
    assert torch.count_nonzero(quantized[0]).item() == 0
    assert scale[1].item() == 1.0
    assert quantized[1, :9].cpu().tolist() == [
        20,
        22,
        62,
        64,
        -20,
        -22,
        -62,
        -64,
        127,
    ]


@pytest.mark.parametrize("columns", [2048, 4096, 5120, 8192, 13824, 16640])
def test_per_token_quant_i8_rne_respects_error_bound(columns):
    generator = torch.Generator(device="cuda").manual_seed(columns)
    x = torch.randn(
        (7, columns),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    quantized, scale = per_token_quant_i8_rne_hip(x)
    reconstructed = quantized.float() * scale
    error = (reconstructed - x.float()).abs()

    torch.testing.assert_close(
        scale,
        x.float().abs().amax(dim=-1, keepdim=True) / 127.0,
        rtol=0,
        atol=0,
    )
    assert torch.all(error <= scale * 0.5001)
    assert quantized.min().item() >= -127
    assert quantized.max().item() <= 127


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
