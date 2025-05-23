import torch
import pytest

from aiter.ops.triton.quant import dynamic_mxfp4_quant
from .utils.quant_ref import torch_dynamic_mxfp4_quant

DEBUG_MODE = False


@pytest.mark.parametrize(
    "M, N",
    [
        (1, 4),
        (1, 28),
        (1, 32),
        (1, 64),
        (1, 68),
        (2, 4),
        (2, 28),
        (2, 32),
        (2, 64),
        (2, 68),
        (128, 4),
        (128, 28),
        (128, 32),
        (128, 64),
        (128, 68),
        (256, 32),
        (160, 40),
        (280, 20),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_dynamic_mxfp4_quant(M: int, N: int, dtype):
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype, device="cuda")

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")

    triton_out, triton_scale = dynamic_mxfp4_quant(x)
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape} triton_out={triton_out}")
        print(f"triton_scale.shape={triton_scale.shape} triton_scale={triton_scale}")

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x)
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape} torch_out={torch_out}")
        print(f"torch_scale.shape={torch_scale.shape} torch_scale={torch_scale}")

    torch.testing.assert_close(triton_scale, torch_scale)
    torch.testing.assert_close(triton_out, torch_out)
