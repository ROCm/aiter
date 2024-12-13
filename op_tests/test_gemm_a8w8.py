from ater.test_common import checkAllclose, perftest, tensor_dump
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import ater
from ater.ops.shuffle import shuffle_weight


@perftest()
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest()
def run_gemm_b(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    return ater.gemm_a8w8_bias(x, weight, x_scale, w_scale, bias, dtype=dtype)


def test_gemm(dtype, m, n, k):
    dim = (m, n, k)
    x = torch.randint(-20, 20, (m, k), dtype=torch.int8).cuda()
    weight = torch.randint(-20, 20, (n, k), dtype=torch.int8).cuda()
    x_scale = torch.rand([m, 1], dtype=torch.float32).cuda()
    w_scale = torch.rand([1, n], dtype=torch.float32).cuda()
    bias = torch.rand([1, n], dtype=dtype).cuda()
    tensor_dump(x, 'x')
    tensor_dump(weight, 'weight')
    tensor_dump(shuffle_weight(weight), 'weight_shuffled')
    tensor_dump(x_scale, 'x_scale')
    tensor_dump(w_scale, 'w_scale')
    tensor_dump(bias, 'bias')

    (a, *_), avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    (b, *_), avg_b = run_gemm_b(x, weight, x_scale, w_scale, bias, dtype)
    tensor_dump(a, 'output')

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, B avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg, rtol=1e-3, atol=1000)


for dtype in [torch.bfloat16]:
    # qkv_proj
    for (m, n, k) in [
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
    ]:
        test_gemm(dtype, m, n, k)
    # attn_out
    for (m, n, k) in [
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
    ]:
        test_gemm(dtype, m, n, k)
