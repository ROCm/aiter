# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
from triton.experimental.gluon import language as gl

from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant
from aiter.ops.triton.utils._triton import arch_info


@pytest.mark.parametrize("quantize", [dynamic_mxfp4_quant, dynamic_mxfp8_quant])
def test_missing_scaled_downcast_uses_triton(quantize, monkeypatch):
    if arch_info.get_arch() != "gfx950":
        pytest.skip("Gluon MXFP4/MXFP8 quantization is selected only on gfx950")

    cdna4 = getattr(getattr(gl, "amd", None), "cdna4", None)
    if cdna4 is not None:
        monkeypatch.delattr(cdna4, "scaled_downcast", raising=False)

    x = torch.randn((32, 128), dtype=torch.bfloat16, device="cuda")
    actual, actual_scales = quantize(x)
    expected, expected_scales = quantize(x, backend="triton")
    torch.cuda.synchronize()

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_scales, expected_scales)
    with pytest.raises(RuntimeError, match="gl.amd.cdna4.scaled_downcast"):
        quantize(x, backend="gluon")
