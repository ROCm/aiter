import torch
import triton
import pytest
from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale
import torch.nn.functional as F


block_shape = (128, 128)


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x_scale = x_scale.repeat_interleave(block_shape_k, dim=1)
    x = x.to(x_scale.dtype) * x_scale[:m, :k]
    x = x.view(m, k)
    w_scale = w_scale.repeat_interleave(block_shape_k, dim=0)
    w_scale = w_scale.repeat_interleave(block_shape_n, dim=1)
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))

    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    return gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


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
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536)]
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
        (2048, 2048, 2049),
        (159, 17389, 597),
        (16, 576, 7168),
    ]
    return x_vals


def generate_gemm_a8w8_blockscale_inputs(M, N, K, block_shape_n, block_shape_k):
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    return x, weight, x_scale, w_scale


@pytest.mark.parametrize(
    "dtype, M, N, K", [(dtype, *shape) for dtype in ["bf16"] for shape in get_x_vals()]
)
def test_gemm(dtype, M, N, K):
    block_shape_n, block_shape_k = block_shape

    dtype = name_to_torch_types[dtype]
    x, weight, x_scale, w_scale = generate_gemm_a8w8_blockscale_inputs(
        M, N, K, block_shape_n, block_shape_k
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)
    b = run_triton(x, weight, x_scale, w_scale, dtype)

    triton.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
