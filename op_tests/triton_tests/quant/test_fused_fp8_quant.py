import torch
import pytest
from aiter.ops.triton.fused_fp8_quant import (
    fused_rms_fp8_per_tensor_static_quant,
    fused_rms_fp8_group_quant,
    fused_flatten_fp8_group_quant,
    fused_reduce_act_mul_fp8_group_quant,
    fused_reduce_rms_fp8_group_quant,
)
from op_tests.triton_tests.quant.test_quant_mxfp4 import torch_dynamic_mxfp4_quant
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
    if N % group_size > 0:
        num_pad = group_size - (N % group_size)
        x_reshape = F.pad(x, (0, num_pad, 0, 0), "constant", 0)
        x_reshape = x_reshape.reshape(
            M, (N + group_size - 1) // group_size, group_size
        ).to(torch.float32)
    else:
        x_reshape = x.reshape(M, N // group_size, group_size).to(torch.float32)
    x_max = torch.max(torch.abs(x_reshape), dim=-1, keepdim=True)[0]
    x_max = torch.where(x_max < 1e-10, 1e-10, x_max).to(torch.float32)
    x_scale = x_max / DTYPE_MAX
    scale_recip = 1.0 / x_scale
    x_quant = torch.clamp(x_reshape * scale_recip, -DTYPE_MAX, DTYPE_MAX).to(
        dtype_quant
    )
    x_quant = x_quant.reshape(M, (N + group_size - 1) // group_size * group_size)[:, :N]
    x_scale = x_scale.squeeze(-1)

    return x_quant, x_scale


def per_tensor_fp8_static_quant(x, dtype_quant, x_scale):
    DTYPE_MAX = torch.finfo(dtype_quant).max
    scale_recip = 1.0 / x_scale
    x_quant = torch.clamp(x * scale_recip, -DTYPE_MAX, DTYPE_MAX).to(dtype_quant)
    return x_quant


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


def run_torch_rms_fp8_per_tensor_static_quant(
    x1, w1, eps1, x2, w2, eps2, res1, dtype_quant, x1_scale
):
    s = x1 + res1
    y1 = rmsnorm(s, w1, eps1)
    y2 = rmsnorm(x2, w2, eps2)
    y1_q = per_tensor_fp8_static_quant(y1, dtype_quant, x1_scale)
    return y1_q, y1.to(x1.dtype), y2.to(x1.dtype), s.to(x1.dtype)


@pytest.mark.parametrize("M", [1, 32, 256])
@pytest.mark.parametrize("N1, N2", [(128, 128), (128, 7168), (7168, 7168)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rms_fp8_per_tensor_static_quant(M: int, N1: int, N2: int, dtype):
    dtype_quant = aiter.dtypes.fp8
    scale = torch.randn(1, dtype=torch.float32, device="cuda")
    x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)

    y1_q_torch, y1_torch, y2_torch, y1_res_torch = (
        run_torch_rms_fp8_per_tensor_static_quant(
            x1, w1, 1e-6, x2, w2, 1e-6, res1, dtype_quant, scale
        )
    )

    y1_q_triton, y1_triton, y2_triton, y1_res_triton = (
        fused_rms_fp8_per_tensor_static_quant(
            x1,
            w1,
            1e-6,
            scale,
            inp2=x2,
            inp2_weight=w2,
            inp2_epsilon=1e-6,
            dtype_quant=dtype_quant,
            res1=res1,
            output_unquantized_inp1=True,
        )
    )

    torch.testing.assert_close(y1_torch, y1_triton, atol=0.1, rtol=0.1)
    torch.testing.assert_close(y2_torch, y2_triton, atol=0.1, rtol=0.1)
    torch.testing.assert_close(y1_res_torch, y1_res_triton, atol=0.1, rtol=0.1)

    y1_upcast_torch = y1_q_torch.to(torch.float32) * scale
    y1_upcast_triton = y1_q_triton.to(torch.float32) * scale
    torch.testing.assert_close(y1_upcast_torch, y1_upcast_triton, atol=0.1, rtol=0.1)


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


@pytest.mark.parametrize("M", [1, 32, 256])
@pytest.mark.parametrize("N1, N2", [(128, 128), (128, 7168), (7168, 7168)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rms_fp8_group_quant_transpose_scale(M: int, N1: int, N2: int, dtype):
    """Test that transpose_scale parameter returns scale with transposed memory layout."""
    group_size = 128
    dtype_quant = aiter.dtypes.fp8
    x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)

    # Call with transpose_scale=False (original behavior)
    (y1_q_orig, y1_s_orig), y1_orig, y2_orig, y1_res_orig = fused_rms_fp8_group_quant(
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
        transpose_scale=False,
    )

    # Call with transpose_scale=True
    (
        (y1_q_transposed, y1_s_transposed),
        y1_transposed,
        y2_transposed,
        y1_res_transposed,
    ) = fused_rms_fp8_group_quant(
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
        transpose_scale=True,
    )

    num_bs_cols = (N1 + group_size - 1) // group_size

    # Verify that both outputs have the same shape
    assert y1_s_orig.shape == (
        M,
        num_bs_cols,
    ), f"Expected shape (M, num_bs_cols), got {y1_s_orig.shape}"
    assert y1_s_transposed.shape == (
        M,
        num_bs_cols,
    ), f"Expected shape (M, num_bs_cols), got {y1_s_transposed.shape}"

    # Verify that transpose_scale=True version is equivalent to .transpose().contiguous().view()
    y1_s_expected = y1_s_orig.transpose(0, 1).contiguous().view(*y1_s_orig.shape)

    # Verify that both have the same shape and strides (row-major)
    assert (
        y1_s_orig.stride() == y1_s_transposed.stride()
    ), "Both should have row-major strides"
    assert (
        y1_s_orig.is_contiguous() and y1_s_transposed.is_contiguous()
    ), "Both should be contiguous"

    # Verify numerical correctness - values should match the transpose().contiguous().view() pattern
    torch.testing.assert_close(y1_s_transposed, y1_s_expected, atol=1e-6, rtol=1e-6)

    # Verify that other outputs are identical
    # For fp8 tensors, use exact bitwise comparison
    torch.testing.assert_close(y1_q_transposed, y1_q_orig, atol=0, rtol=0)
    torch.testing.assert_close(y1_transposed, y1_orig, atol=0.1, rtol=0.1)
    torch.testing.assert_close(y2_transposed, y2_orig, atol=0.1, rtol=0.1)
    torch.testing.assert_close(y1_res_transposed, y1_res_orig, atol=0.1, rtol=0.1)


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


def run_torch_reduce_rms_fp8_group_quant(
    x1, w1, eps1, x2, w2, eps2, res1, x3, dtype_quant, dtype, group_size
):
    out_dtype = dtype if dtype is not None else x1.dtype
    if x1.dim() == 3:
        x1 = torch.sum(x1, dim=0)
        x2 = torch.sum(x2, dim=0)
        assert x3 is not None
        x3 = torch.sum(x3, dim=0).to(out_dtype)
    else:
        assert x3 is None
    if res1 is not None:
        s = x1 + res1
        y_res1 = s.to(out_dtype)
    else:
        s = x1
        y_res1 = None
    y1 = rmsnorm(s, w1, eps1)
    y2 = rmsnorm(x2, w2, eps2)
    y1_q, y1_s = per_token_fp8_group_quant(y1, dtype_quant, group_size)
    return (y1_q, y1_s), y1.to(out_dtype), y2.to(out_dtype), y_res1, x3


def generate_fused_reduce_rms_quant_data(M, N1, N2, N3, SPK, dtype=torch.bfloat16):
    if SPK > 1:
        x1 = torch.randn((SPK, M, N1), dtype=torch.float32, device="cuda") / 10
        x2 = torch.randn((SPK, M, N2), dtype=torch.float32, device="cuda") / 10
        x3 = torch.randn((SPK, M, N3), dtype=torch.float32, device="cuda") / 10
    else:
        x1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
        x2 = torch.randn((M, N2), dtype=dtype, device="cuda") / 10
        x3 = None

    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    return x1, w1, x2, w2, res1, x3


@pytest.mark.parametrize("M", [1, 32, 256, 8192])
@pytest.mark.parametrize(
    "N1, N2, N3", [(128, 128, 128), (1536, 512, 64), (7168, 7168, 7168)]
)
@pytest.mark.parametrize("SPK", [1, 4, 14])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_reduce_rms_fp8_group_quant(
    M: int, N1: int, N2: int, N3: int, SPK: int, dtype
):
    group_size = 128
    dtype_quant = aiter.dtypes.fp8
    x1, w1, x2, w2, res1, x3 = generate_fused_reduce_rms_quant_data(
        M, N1, N2, N3, SPK, dtype
    )
    (y1_q_torch, y1_s_torch), y1_torch, y2_torch, y1_res_torch, y3_torch = (
        run_torch_reduce_rms_fp8_group_quant(
            x1, w1, 1e-6, x2, w2, 1e-6, res1, x3, dtype_quant, dtype, group_size
        )
    )

    (y1_q_triton, y1_s_triton), y1_triton, y2_triton, y1_res_triton, y3_triton = (
        fused_reduce_rms_fp8_group_quant(
            x1,
            w1,
            1e-6,
            inp2=x2,
            inp2_weight=w2,
            inp2_epsilon=1e-6,
            inp3=x3,
            group_size=group_size,
            dtype_quant=dtype_quant,
            dtype=dtype,
            res1=res1,
            output_unquantized_inp1=True,
        )
    )

    torch.testing.assert_close(y1_torch, y1_triton, atol=0.1, rtol=0.1)
    torch.testing.assert_close(y2_torch, y2_triton, atol=0.1, rtol=0.1)

    if y1_res_torch is not None:
        torch.testing.assert_close(y1_res_torch, y1_res_triton, atol=0.1, rtol=0.1)

    y1_upcast_torch = upcast(
        y1_q_torch, y1_s_torch, dtype=torch.float32, group_size=group_size
    )
    y1_upcast_triton = upcast(
        y1_q_triton, y1_s_triton, dtype=torch.float32, group_size=group_size
    )
    torch.testing.assert_close(y1_upcast_torch, y1_upcast_triton, atol=0.1, rtol=0.1)

    if y3_torch is not None:
        torch.testing.assert_close(y3_torch, y3_triton, atol=0.1, rtol=0.1)
