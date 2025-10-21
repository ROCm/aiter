import torch
import pytest
from aiter.ops.triton.swiglu_with_gemm import swiglu_with_gemm
from aiter.ops.triton.utils.types import str_to_torch_dtype


# pytest
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
    ],
)
def test_swiglu_with_gemm(M, N, K, dtype):
    dtype = str_to_torch_dtype[dtype]
    torch.manual_seed(0)
    silu = torch.nn.SiLU()
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    w = torch.randn(N, 2 * K, dtype=dtype, device="cuda")
    xw = x @ w
    y_triton = swiglu_with_gemm(x, w)
    y_torch = silu(xw[:, :K]) * xw[:, K:]

    atol, rtol = 1e-1, 1e-1

    torch.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
