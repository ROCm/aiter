import torch
import pytest
from aiter.ops.triton.fused_quant import (
    fused_flatten_mxfp4_quant,
    fused_rms_mxfp4_quant,
)
from op_tests.triton_tests.test_quant_mxfp4 import torch_dynamic_mxfp4_quant


@pytest.mark.parametrize("B", [1, 4, 16, 32, 1000, 10000])
@pytest.mark.parametrize("M", [32, 64])
@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_flatten_quant(B: int, M: int, N: int, dtype):
    torch.manual_seed(20)
    x = torch.randn((B, M, N), dtype=dtype, device="cuda").transpose(0, 1)

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x.flatten(1, 2))
    triton_out, triton_scale = fused_flatten_mxfp4_quant(x)

    torch.testing.assert_close(triton_scale, torch_scale)
    torch.testing.assert_close(triton_out, torch_out)
