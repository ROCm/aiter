import torch
import torch.nn.functional as F
import pytest
from op_tests.triton_tests.quant.test_quant_mxfp4 import torch_dynamic_mxfp4_quant
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    shuffle_scales,
    un_shuffle_scales,
)
from aiter.ops.triton.activation import act_mul_and_mxfp4_quant
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEBUG_MODE = False


def pad_tensor_2d(tensor, mult_m=256, mult_n=8):
    M, N = tensor.shape

    pad_rows = (mult_m - (M % mult_m)) % mult_m
    pad_cols = (mult_n - (N % mult_n)) % mult_n
    padded_tensor = torch.nn.functional.pad(
        tensor, (0, pad_cols, 0, pad_rows), mode="constant", value=0
    )

    return padded_tensor


def torch_act_mul_and_mxfp4_quant(
    input: torch.Tensor, activation: str, shuffle: bool
) -> torch.Tensor:
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
    out, out_scale = torch_dynamic_mxfp4_quant(out)
    if shuffle:
        # out_scale_pad = out_scale
        M = out_scale.shape[0]
        N = out.shape[1] * 2
        scaleM = (M + 255) // 256 * 256
        scaleN_valid = (N + 31) // 32
        scaleN = (scaleN_valid + 7) // 8 * 8
        out_scale_pad = torch.empty(
            (scaleM, scaleN), dtype=out_scale.dtype, device=out_scale.device
        )
        out_scale_pad[:M, :scaleN] = out_scale[:M, :scaleN]
        out_scale = shuffle_scales(out_scale_pad)
        out_scale = out_scale.view(out_scale.shape[0] * 32, -1)
    return out, out_scale


@pytest.mark.parametrize(
    "M, N",
    [
        (512, 57344),
        (504, 57344),
        (1, 57344),
        (4, 57344),
        (32, 8192),
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
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("scale_shuffle_padding", [False, True])
def test_act_mul_and_mxfp4_quant(
    M: int, N: int, dtype, activation: str, shuffle: bool, scale_shuffle_padding: bool
):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    if shuffle and N % 512 != 0:
        pytest.skip()

    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=dtype, device="cuda")

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")

    triton_out, triton_scale = act_mul_and_mxfp4_quant(
        x,
        activation=activation,
        shuffle=shuffle,
        scale_shuffle_padding=scale_shuffle_padding,
    )
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape} triton_out={triton_out}")
        print(f"triton_scale.shape={triton_scale.shape} triton_scale={triton_scale}")

    torch_out, torch_scale = torch_act_mul_and_mxfp4_quant(
        x, activation=activation, shuffle=shuffle
    )

    if shuffle:
        triton_scale = un_shuffle_scales(
            triton_scale.view(triton_scale.shape[0] // 32, -1)
        )
        torch_scale = un_shuffle_scales(
            torch_scale.view(torch_scale.shape[0] // 32, -1)
        )

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape} torch_out={torch_out}")
        print(f"torch_scale.shape={torch_scale.shape} torch_scale={torch_scale}")

    scaleN_valid = (N // 2 + 31) // 32
    triton_scale = triton_scale[:M, :scaleN_valid]
    torch_scale = torch_scale[:M, :scaleN_valid]

    torch.testing.assert_close(triton_out, torch_out)
    torch.testing.assert_close(triton_scale, torch_scale)
