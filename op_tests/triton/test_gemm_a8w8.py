import torch
import triton
import triton.language as tl
import pytest
from aiter.ops.triton.gemm_a8w8 import gemm_a8w8
import torch.nn.functional as F


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    return gemm_a8w8(x, weight, x_scale, w_scale, bias)


def is_cdna4():
    return triton.runtime.driver.active.get_current_target().arch == "gfx950"


e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz
e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz

name_to_torch_types = {
    "int8": torch.int8,
    "int32": torch.int32,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp8e5": e5m2_type,
    "fp8e4": e4m3_type,
}


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
    x_vals += [
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
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    return x_vals


def generate_gemm_a8w8_inputs(M, N, K, dtype):
    x = torch.randint(-20, 20, (M, K), dtype=torch.int8).cuda()
    weight = torch.randint(-20, 20, (N, K), dtype=torch.int8).cuda()
    x_scale = torch.rand([M, 1], dtype=torch.float32).cuda() + 1e-6
    w_scale = torch.rand([1, N], dtype=torch.float32).cuda() + 1e-6
    bias = torch.rand([1, N], dtype=dtype).cuda() * 10

    return x, weight, x_scale, w_scale, bias


@pytest.mark.parametrize(
    "dtype, m, n, k", [(dtype, *shape) for dtype in ["bf16"] for shape in get_x_vals()]
)
def test_gemm(dtype, M, N, K):
    dtype = name_to_torch_types[dtype]
    x, weight, x_scale, w_scale, bias = generate_gemm_a8w8_inputs(M, N, K)

    a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b = run_triton(x, weight, x_scale, w_scale, bias, dtype)

    triton.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
