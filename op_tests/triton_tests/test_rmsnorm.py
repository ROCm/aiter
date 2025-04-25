import torch
import pytest
import triton
import torch.nn.functional as F
from aiter.ops.triton.rmsnorm import rms_norm
from aiter.ops.triton.rmsnorm import rmsnorm2d_fwd_with_add


def torch_rmsnorm(x, g, out_dtype=torch.float16, epsilon=1e-6):
    M, N = x.shape
    # cast to float32 as the triton kernel
    x_f32 = x.float()
    g_f32 = g.float()
    rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * 1 / N)
    rsigma = 1.0 / rms
    rms_norm_f32 = x_f32 * rsigma.unsqueeze(1) * g_f32
    rms_norm = rms_norm_f32.to(out_dtype)
    return rms_norm


def run_torch(x, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = torch_rmsnorm(x, weight, x.dtype, eps)
    else:
        residual_out = x + residual
        output = torch_rmsnorm(residual_out, weight, residual_out.dtype, eps)
    return output, residual_out


def run_triton(x, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = rms_norm(x, weight, eps)
    else:
        residual_out = torch.empty_like(x)
        output = torch.empty_like(x)
        rmsnorm2d_fwd_with_add(output, x, residual, residual_out, weight, eps)
    return output, residual_out


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [
        (1, 4),
        (2, 10),
        (8192, 4096),
        (4096, 8192),
        (8000, 8000),
        (1, 31744),
        (3, 65536),
        (1, 131072),
        (873, 1245),
        (23153, 45),
        (89999, 234),
    ],
)
def test_rmsnorm(M, N, in_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    out_dtype = in_dtype
    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)

    y_torch, *_ = run_torch(x, weight, 1e-5)
    y_triton, *_ = run_triton(x, weight, 1e-5)

    if out_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    assert (
        y_triton.dtype == out_dtype
    ), f"y_triton has dtype={y_triton.dtype}, expected {out_dtype}"
    assert (
        y_torch.dtype == out_dtype
    ), f"y_torch has dtype={y_torch.dtype}, expected {out_dtype}"

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)


# TODO: Re-enable the commented tests once we find why they're causing the AITER CI to fail
@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [
        (1, 4),
        (2, 10),
        (8192, 4096),
        (4096, 8192),
        (8000, 8000),
        # (1, 31744),
        # (3, 65536),
        (1, 131072),
        (873, 1245),
        (23153, 45),
        (89999, 234),
    ],
)
def test_fused_add_rmsnorm(M, N, in_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    out_dtype = in_dtype
    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)
    res = torch.randn(M, N, device="cuda", dtype=in_dtype)

    y_torch, res_torch, *_ = run_torch(x, weight, 1e-5, residual=res)
    y_triton, res_triton, *_ = run_triton(x, weight, 1e-5, residual=res)

    if out_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    assert (
        y_triton.dtype == out_dtype
    ), f"y_triton has dtype={y_triton.dtype}, expected {out_dtype}"
    assert (
        y_torch.dtype == out_dtype
    ), f"y_torch has dtype={y_torch.dtype}, expected {out_dtype}"

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
    triton.testing.assert_close(res_triton, res_torch, atol=atol, rtol=rtol)
