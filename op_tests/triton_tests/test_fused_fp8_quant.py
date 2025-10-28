import torch
import pytest
from aiter.ops.triton.fused_fp8_quant import (
    fused_rms_fp8_group_quant,
    fused_flatten_fp8_group_quant,
    fused_reduce_act_mul_fp8_group_quant,
)
from op_tests.triton_tests.test_quant_mxfp4 import torch_dynamic_mxfp4_quant
import aiter
import torch.nn.functional as F

torch.manual_seed(0)


def rmsnorm(input, weight, eps=1e-6):
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps)
    rms_norm = input * norm_factor[:, None] * weight[None, :]
    return rms_norm


def per_token_fp8_group_quant(x, dtype_quant, group_size=128):
    DTYPE_MAX = torch.finfo(dtype_quant).max
    M, N = x.shape
    x_reshape = x.reshape(M, N // group_size, group_size).to(torch.float32)
    x_max = torch.max(torch.abs(x_reshape), dim=-1, keepdim=True)[0]
    x_max = torch.where(x_max < 1e-10, 1e-10, x_max).to(torch.float32)
    x_scale = x_max / DTYPE_MAX
    scale_recip = 1.0 / x_scale
    x_quant = torch.clamp(x_reshape * scale_recip, -DTYPE_MAX, DTYPE_MAX).to(
        dtype_quant
    )
    x_quant = x_quant.reshape(M, N)
    x_scale = x_scale.squeeze(-1)

    return x_quant, x_scale


def upcast(x, s, dtype, group_size=128):
    x_N = x.shape[1]
    x = x.reshape(-1, x_N // group_size, group_size).to(torch.float32) * s.reshape(
        -1, s.shape[1], 1
    )
    x = x.reshape(-1, x_N)
    return x.to(dtype=dtype)


def run_torch_rms_fp8_group_quant(
    x1, w1, eps1, x2, w2, eps2, res1, dtype_quant, group_size
):
    s = x1 + res1
    y1 = rmsnorm(s, w1, eps1)
    y2 = rmsnorm(x2, w2, eps2)
    y1_q, y1_s = per_token_fp8_group_quant(y1, dtype_quant, group_size)
    return (y1_q, y1_s), y1.to(x1.dtype), y2.to(x1.dtype), s.to(x1.dtype)


def generate_fused_rms_quant_data(M, N1, N2, dtype=torch.bfloat16):
    x1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    x2 = torch.randn((M, N2), dtype=dtype, device="cuda") / 10
    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    return x1, w1, x2, w2, res1


@pytest.mark.parametrize("M", [1, 32, 256])
@pytest.mark.parametrize("N1, N2", [(128, 128), (128, 7168), (7168, 7168)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rms_fp8_group_quant(M: int, N1: int, N2: int, dtype):
    group_size = 128
    dtype_quant = aiter.dtypes.fp8
    x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)

    (y1_q_torch, y1_s_torch), y1_torch, y2_torch, y1_res_torch = (
        run_torch_rms_fp8_group_quant(
            x1, w1, 1e-6, x2, w2, 1e-6, res1, dtype_quant, group_size
        )
    )

    (y1_q_triton, y1_s_triton), y1_triton, y2_triton, y1_res_triton = (
        fused_rms_fp8_group_quant(
            x1,
            w1,
            1e-6,
            inp2=x2,
            inp2_weight=w2,
            inp2_epsilon=1e-6,
            group_size=group_size,
            dtype_quant=dtype_quant,
            res1=res1,
            output_unquantized_inp1=True,
        )
    )

    torch.testing.assert_close(y1_torch, y1_triton, atol=0.1, rtol=0.1)
    torch.testing.assert_close(y2_torch, y2_triton, atol=0.1, rtol=0.1)
    torch.testing.assert_close(y1_res_torch, y1_res_triton, atol=0.1, rtol=0.1)

    y1_upcast_torch = upcast(
        y1_q_torch, y1_s_torch, dtype=torch.float32, group_size=group_size
    )
    y1_upcast_triton = upcast(
        y1_q_triton, y1_s_triton, dtype=torch.float32, group_size=group_size
    )
    torch.testing.assert_close(y1_upcast_torch, y1_upcast_triton, atol=0.1, rtol=0.1)


def run_torch_flatten_fp8_group_quant(x, dtype_quant, group_size):
    y_q, y_s = per_token_fp8_group_quant(
        x.reshape(x.shape[0], -1), dtype_quant, group_size
    )
    return y_q, y_s


@pytest.mark.parametrize("M", [1, 32, 256])
@pytest.mark.parametrize("N1, N2", [(16, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_flatten_fp8_group_quant(M: int, N1: int, N2: int, dtype):
    group_size = 128
    dtype_quant = aiter.dtypes.fp8
    x = torch.randn((N1, M, N2), dtype=dtype, device="cuda") / 10
    x = x.transpose(0, 1)

    y_q_torch, y_s_torch = run_torch_flatten_fp8_group_quant(x, dtype_quant, group_size)

    y_q_triton, y_s_triton = fused_flatten_fp8_group_quant(
        x,
        group_size=group_size,
        dtype_quant=dtype_quant,
    )

    y_upcast_torch = upcast(
        y_q_torch, y_s_torch, dtype=torch.float32, group_size=group_size
    )
    y_upcast_triton = upcast(
        y_q_triton, y_s_triton, dtype=torch.float32, group_size=group_size
    )
    torch.testing.assert_close(y_upcast_torch, y_upcast_triton, atol=0.1, rtol=0.1)


def run_torch_reduce_act_mul_fp8_group_quant(
    x, x2, activation, dtype, dtype_quant, group_size=128
):
    x = x.clone()
    y2 = None
    if x.dim() == 3:
        x = x.sum(axis=0)
        y2 = x2.sum(axis=0).to(dtype=dtype)
    else:
        assert x2 is None, "x2 must be None in x.dim() == 2 cases"
    n = x.shape[1] // 2
    x, x_mul = x.split([n, n], dim=-1)
    if activation == "silu":
        x = F.silu(x) * x_mul
    elif activation == "gelu":
        x = F.gelu(x) * x_mul

    y_q, y_s = per_token_fp8_group_quant(x, dtype_quant, group_size)

    return (y_q, y_s), y2


def generate_fused_reduce_act_mul_fp8_group_quant(
    M: int,
    N1: int,
    dtype=torch.bfloat16,
    SPK: int = 1,
    N2: int = 1,
):
    if SPK == 1:
        x = torch.randn((M, N1 * 2), dtype=dtype).cuda() / 10
    else:
        x = torch.randn((SPK, M, N1 * 2), dtype=torch.float32).cuda() / 10
    x2 = None
    if SPK > 1:
        x2 = torch.randn((SPK, M, N2), dtype=torch.float32).cuda() / 10

    return x, x2


@pytest.mark.parametrize("M", [1, 32, 256, 131072])
@pytest.mark.parametrize("N1, N2", [(256, 256)])
@pytest.mark.parametrize("SPK", [1, 4, 14])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_fused_reduce_act_mul_fp8_group_quant(
    M: int, N1: int, N2: int, SPK: int, dtype, activation
):
    group_size = 128
    dtype_quant = aiter.dtypes.fp8

    x, x2 = generate_fused_reduce_act_mul_fp8_group_quant(
        M, N1, dtype=dtype, SPK=SPK, N2=N2
    )

    (y_q_torch, y_s_torch), y2_torch = run_torch_reduce_act_mul_fp8_group_quant(
        x, x2, activation, dtype, dtype_quant, group_size
    )

    (y_q_triton, y_s_triton), y2_triton = fused_reduce_act_mul_fp8_group_quant(
        x,
        activation=activation,
        x2=x2,
        group_size=group_size,
        dtype_quant=dtype_quant,
        dtype=dtype,
    )

    torch.testing.assert_close(y2_torch, y2_triton, atol=0.1, rtol=0.1)

    y_upcast_torch = upcast(
        y_q_torch, y_s_torch, dtype=torch.float32, group_size=group_size
    )
    y_upcast_triton = upcast(
        y_q_triton, y_s_triton, dtype=torch.float32, group_size=group_size
    )
    torch.testing.assert_close(y_upcast_torch, y_upcast_triton, atol=0.1, rtol=0.1)
