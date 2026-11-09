# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for downcast_to_mxfp's pow2_scale path.

pow2_scale=True is meant to reproduce the quark-style scale derivation that
dynamic_mxfp4_quant / dynamic_mxfp8_quant already implement, so those are used
as the reference rather than a second torch model of the same arithmetic.
"""

import pytest
import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant

# Full blocks, a trailing partial block, a single block, and fewer elements than
# one block. 32 is the MX block size, so N=68 and N=20 are the interesting ones.
SHAPES = [(128, 256), (64, 68), (32, 20), (1, 32), (256, 96)]


@pytest.mark.parametrize("M, N", SHAPES)
def test_downcast_to_mxfp_pow2_scale_fp4(M: int, N: int):
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    out, scale = downcast_to_mxfp(x, torch.uint8, axis=-1, pow2_scale=True)
    ref_out, ref_scale = dynamic_mxfp4_quant(x)

    assert scale.shape == (M, triton.cdiv(N, 32))
    torch.testing.assert_close(scale, ref_scale, atol=0, rtol=0)
    torch.testing.assert_close(
        out.view(torch.uint8), ref_out.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.parametrize("M, N", SHAPES)
def test_downcast_to_mxfp_pow2_scale_fp8(M: int, N: int):
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    out, scale = downcast_to_mxfp(x, torch.float8_e4m3fn, axis=-1, pow2_scale=True)

    # dynamic_mxfp8_quant requires K % 32 == 0. Zero padding cannot raise a
    # block's amax, so padding out the last block leaves every scale and every
    # real element unchanged, making it an exact reference for partial blocks.
    pad = (-N) % 32
    ref_out, ref_scale = dynamic_mxfp8_quant(F.pad(x, (0, pad)))

    assert scale.shape == (M, triton.cdiv(N, 32))
    torch.testing.assert_close(scale, ref_scale, atol=0, rtol=0)
    torch.testing.assert_close(
        out.view(torch.uint8), ref_out[:, :N].view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.parametrize("out_dtype", [torch.uint8, torch.float8_e4m3fn])
def test_downcast_to_mxfp_pow2_scale_is_not_a_no_op(out_dtype):
    """The two scale schemes must stay distinguishable.

    A tensor quantized under one has to be dequantized under the same one, so if
    this ever stops differing the flag has silently become a no-op and callers
    lose the only signal that the schemes are incompatible.
    """
    torch.manual_seed(20)
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")

    _, scale_default = downcast_to_mxfp(x, out_dtype, axis=-1, pow2_scale=False)
    _, scale_pow2 = downcast_to_mxfp(x, out_dtype, axis=-1, pow2_scale=True)

    assert not torch.equal(scale_default, scale_pow2)
