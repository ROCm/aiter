import torch
import pytest
from aiter.ops.triton.fused_quant import fused_flatten_quant, rms_rms_mxfpx4_quant
from test_quant_mxfp4 import torch_dynamic_mxfp4_quant

@pytest.mark.parametrize(
    "B, M, N",
    [
        (16, 1, 4),
        (16, 1, 28),
        (16, 1, 32),
        (16, 1, 64),
        (16, 1, 68),
        (16, 2, 4),
        (16, 2, 28),
        (16, 2, 32),
        (16, 2, 64),
        (16, 2, 68),
        (16, 128, 4),
        (16, 128, 28),
        (16, 128, 32),
        (16, 128, 64),
        (16, 128, 68),
        (16, 256, 32),
        (16, 160, 40),
        (16, 280, 20),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_flatten_quant(B: int, M: int, N: int, dtype):
    torch.manual_seed(20)
    x = torch.randn((B, M, N), dtype=dtype, device="cuda").transpose(0, 1)

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x.flatten(1, 2))
    triton_out, triton_scale = fused_flatten_quant(x)

    torch.testing.assert_close(triton_scale, torch_scale)
    torch.testing.assert_close(triton_out, torch_out)
