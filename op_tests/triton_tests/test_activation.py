import torch
import torch.nn.functional as F
import pytest
from .test_quant_mxfp4 import torch_dynamic_mxfp4_quant
from .test_gemm_afp4wfp4 import shuffle_scales
from aiter.ops.triton.activation import act_mul_and_mxfp4_quant
from op_tests.triton_tests.utils.types import str_to_torch_dtype

DEBUG_MODE = False


def pad_tensor_2d(tensor, mult_m=256, mult_n=8):
    M, N = tensor.shape

    pad_rows = (mult_m - (M % mult_m)) % mult_m
    pad_cols = (mult_n - (N % mult_n)) % mult_n
    padded_tensor = torch.nn.functional.pad(
        tensor, (0, pad_cols, 0, pad_rows), mode="constant", value=0
    )

    return padded_tensor


def torch_act_mul_and_mxfp4_quant(input: torch.Tensor, activation: str) -> torch.Tensor:
    """
    The fused kernel casts the original input to float32 and does all the arithmetic
    and bit operations in float32.
    """
    input = input.to(torch.float32)
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    if activation == "silu":
        out = F.silu(x) * y
    elif activation == "gelu":
        out = F.gelu(x) * y
    else:
        out = F.gelu(x, approximate="tanh") * y
    return torch_dynamic_mxfp4_quant(out)

def get_test_params():
    full_shape_set = [
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
        (256, 512),
        (256, 1024),
        (160, 40),
        (280, 20),
        (32, 128),
    ] 
    full_set = [
        pytest.param(*shape, dtype, activation, shuffle, marks=pytest.mark.slow)
        for shape in full_shape_set
        for dtype in ["bfloat16", "float16"]
        for activation in ["silu", "gelu", "gelu_tanh"]
        for shuffle in [True, False]
    ]

    fast_shape_set = [
        (1, 4),
        (1, 64),
        (2, 28),
        (2, 32),
        (2, 68),
        (128, 28),
        (128, 68),
        (256, 1024),
        (160, 40),
    ] 
    fast_set = [
        (*shape, dtype, activation, shuffle)
        for shape in fast_shape_set
        for dtype in ["bfloat16"]
        for activation in ["silu", "gelu", "gelu_tanh"]
        for shuffle in [True, False]
    ]
    return full_set + fast_set

@pytest.mark.parametrize("M, N, dtype, activation, shuffle", get_test_params())
def test_act_mul_and_mxfp4_quant(M: int, N: int, dtype, activation: str, shuffle: bool):
    # TODO: extend tests to different shapes with proper padding
    if shuffle and (M % 256 != 0 or N % 512 != 0):
        pytest.skip()
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=str_to_torch_dtype[dtype], device="cuda")

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")

    triton_out, triton_scale = act_mul_and_mxfp4_quant(
        x, activation=activation, shuffle=shuffle
    )
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape} triton_out={triton_out}")
        print(f"triton_scale.shape={triton_scale.shape} triton_scale={triton_scale}")

    torch_out, torch_scale = torch_act_mul_and_mxfp4_quant(x, activation=activation)
    if shuffle:
        torch_scale = shuffle_scales(torch_scale)
        triton_scale = triton_scale.reshape(triton_scale.shape[0] // 32, -1)
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape} torch_out={torch_out}")
        print(f"torch_scale.shape={torch_scale.shape} torch_scale={torch_scale}")

    torch.testing.assert_close(triton_out, torch_out)
    torch.testing.assert_close(triton_scale, torch_scale)
