import torch
import pytest
from aiter.ops.triton.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad
import torch.nn.functional as F


def generate_inputs(M, N, has_res, dtype):
    x = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn((N,), dtype=dtype, device="cuda")
    res = torch.randn((M, N), dtype=dtype, device="cuda") if has_res else None
    return x, weight, res


def run_torch(x, weight, eps=1e-6, res=None, pad_to_multiple=0):
    dtype = x.dtype
    x = x.to(torch.float32)
    if res is not None:
        x = x + res.to(torch.float32)
        res = x.to(dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * weight * torch.rsqrt(variance + eps)
    N = x.shape[-1]
    if pad_to_multiple > 0:
        pad = (N + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple - N
        if pad > 0:
            x = F.pad(x, (0, pad, 0, 0), "constant", 0.0)
    x = x.to(dtype)
    if res is not None:
        return x, res
    return x


@pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 256, 8192])
@pytest.mark.parametrize("N", [4, 16, 320, 640, 2880])
@pytest.mark.parametrize("has_res", [False, True])
@pytest.mark.parametrize("pad_to_multiple", [0, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mul_add(M: int, N: int, has_res: bool, pad_to_multiple: int, dtype):

    x, weight, res = generate_inputs(M, N, has_res, dtype)

    if has_res:
        x_torch, res_torch = run_torch(x, weight, 1e-6, res, pad_to_multiple)
        x_triton, res_triton = fused_add_rmsnorm_pad(
            x, weight, 1e-6, res, pad_to_multiple
        )
    else:
        x_torch = run_torch(x, weight, 1e-6, res, pad_to_multiple)
        x_triton = fused_add_rmsnorm_pad(x, weight, 1e-6, res, pad_to_multiple)

    torch.testing.assert_close(x_torch.to(dtype), x_triton)
    if has_res:
        torch.testing.assert_close(res_torch.to(dtype), res_triton)
