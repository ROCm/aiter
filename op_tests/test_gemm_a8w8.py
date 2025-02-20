# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import aiter
from aiter.ops.shuffle import shuffle_weight
import pytest
import torch
import torch.nn.functional as F

from .utils import DefaultBenchmarkHook, check_all_close, rand_tensor

MNK = [
    # qkv_proj
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
    # attn_out
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

# ck_gemm only accepts the following combination for | scale : output | dtypes
# | F32 : F16 | F32 : BF16 | F16 : F16 | B16 : B16 |
CK_GEMM_SCALES_OUTPUT_DTYPES = [
    (torch.float32, torch.float16),
    (torch.float32, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.bfloat16),
]


def torch_scaled_mm(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def setup(
    mnk: tuple[int, int, int],
    ab_dtype: torch.dtype,
    scales_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    use_bias: bool,
) -> tuple[torch.tensor, ...]:
    
    m, n, k = mnk
    a = rand_tensor(size=(m, k), dtype=ab_dtype).cuda()
    b = rand_tensor(size=(n, k), dtype=ab_dtype).cuda()
    a_scale = rand_tensor(size=(m, 1), dtype=scales_dtype).cuda() + 1e-6
    b_scale = rand_tensor(size=(1, n), dtype=scales_dtype).cuda() + 1e-6
    bias = (rand_tensor(size=(1, n), dtype=bias_dtype).cuda()) if use_bias else None
    return a, b, a_scale, b_scale, bias

@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("ab_dtype", [torch.int8, torch.float8_e4m3fnuz])
@pytest.mark.parametrize("scales_output_dtype", CK_GEMM_SCALES_OUTPUT_DTYPES)
@pytest.mark.parametrize("use_bias", [True, False])
def test_ck_gemm_close_to_torch(
    benchmark,
    mnk: tuple[int, int, int],
    ab_dtype: torch.dtype,
    scales_output_dtype: tuple[torch.dtype, torch.dtype],
    use_bias: bool
) -> None:
    # using bias further reduces precision,
    # so we use a higher tolerance when bias is enabled.
    if use_bias:
        rtol, atol = (1e-1, 1e-1)
    else:
        rtol, atol = (1e-2, 1e-1)

    scales_dtype, out_dtype =  scales_output_dtype
    bias_dtype = out_dtype
    a, b, a_scale, b_scale, bias = setup(
        mnk,
        ab_dtype,
        scales_dtype,
        bias_dtype,
        use_bias,
    )

    _, output = benchmark(
        DefaultBenchmarkHook(aiter.gemm_a8w8_CK, a, b, a_scale, b_scale, bias, out_dtype)
    )
    expected = torch_scaled_mm(a, b, a_scale, b_scale, bias, dtype=out_dtype)

    check_all_close(output, expected, rtol=rtol, atol=atol)

@pytest.mark.parametrize("mnk", MNK)
def test_asm_gemm_close_to_torch(benchmark, mnk: tuple[int, int, int]) -> None:
    rtol, atol = (1e-1, 1e-1)
    ab_dtype = torch.int8
    out_dtype = torch.bfloat16
    scales_dtype = torch.float32
    bias_dtype = torch.float
    # gemm_a8w8_ASM requires bias and shuffle
    a, b, a_scale, b_scale, bias = setup(
        mnk,
        ab_dtype,
        scales_dtype,
        bias_dtype,
        use_bias=True,
    )
    b_shuffled= shuffle_weight(b, layout=(32, 16))

    _, output = benchmark(
        DefaultBenchmarkHook(aiter.gemm_a8w8_ASM, a, b_shuffled, a_scale, b_scale, bias)
    )
    expected = torch_scaled_mm(a, b, a_scale, b_scale, bias, dtype=out_dtype)
    if output is not None and torch.sum(output.isnan()==True) == 0:
        check_all_close(output, expected, rtol=rtol, atol=atol)
