import triton
import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.norm import layer_norm
from aiter.ops.triton.norm import layernorm2d_fwd_with_add


def run_torch(input, weight, bias, eps, residual=None, x_bias=None):
    if residual is None:
        residual_out = None
        output = F.layer_norm(
            input=input,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    else:
        residual_out = input + residual
        output = F.layer_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    return output, residual_out


def run_triton(input, weight, bias, eps, residual=None, x_bias=None):
    if residual is None:
        residual_out = None
        output = layer_norm(input, weight, bias, eps, x_bias)
    else:
        residual_out = torch.empty_like(input)
        output = torch.empty_like(input)
        layernorm2d_fwd_with_add(
            output, input, residual, residual_out, weight, bias, eps, x_bias
        )
    return output, residual_out


# pytest
@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [
        (1823, 781),
        (2, 128),
        (1, 4),
        (128, 2),
        (1, 128),
        (8192, 8192),
        (4096, 8192),
        (359, 1),
        (1, 359),
        (1, 131072),
        (1, 89999),
    ],
)
def test_layernorm(M, N, dtype_str, eps=1e-5):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w_shape = (N,)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)

    # forward pass
    y_torch, *_ = run_torch(x, w, b, eps)
    y_triton, *_ = run_triton(x, w, b, eps)

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)


# pytest
@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [
        (1823, 781),
        (2, 128),
        (1, 4),
        (128, 2),
        (1, 128),
        (8192, 8192),
        (4096, 8192),
        (359, 1),
        (1, 359),
        (1, 131072),
        (1, 89999),
    ],
)
def test_fused_add_layernorm(M, N, dtype_str, eps=1e-5):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w_shape = (N,)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)

    # forward pass
    y_torch, res_torch, *_ = run_torch(x, w, b, eps, residual=res)
    y_triton, res_triton, *_ = run_triton(x, w, b, eps, residual=res)

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
    triton.testing.assert_close(res_triton, res_torch, atol=atol, rtol=rtol)
