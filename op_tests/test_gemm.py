from ater.test_common import checkAllclose, perftest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
if 1:
    _path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, f'{_path}/../../')
    from ater.tuned_gemm import tgemm


@perftest()
def run_torch(x, weight, bias=None):

    return F.linear(x, weight, bias)


@perftest()
def run_gemm_b(x, weight, bias=None):
    return tgemm.mm(x, weight, bias)


def test_gemm(dtype, m, n, k, bias=False):
    dim = (m, n, k)
    x = torch.randn(m, k, dtype=dtype).cuda()
    weight = torch.randn(n, k, dtype=dtype).cuda()
    if bias:
        bias = torch.randn(n, dtype=dtype).cuda()
    else:
        bias=None
    (a, *_), avg_a = run_torch(x, weight, bias)
    (b, *_), avg_b = run_gemm_b(x, weight, bias)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, B avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg)


test_gemm(torch.bfloat16, 128, 32, 8192)
# for dtype in [torch.float16, torch.bfloat16]:
#     # # qkv_proj
#     # for (m, n, k) in [(4096, 1280, 8192),
#     #                   (128, 1280, 8192),
#     #                   (128, 1024, 8192),
#     #                   (128, 128, 8192),
#     #                   ]:
#     #     test_gemm(dtype, m, n, k)
#     # # attn_out
#     # for (m, n, k) in [(4096, 8192, 1024),
#     #                   (128, 8192, 1024)]:
#     #     test_gemm(dtype, m, n, k)
#     # test_gemm(dtype, 128, 1024, 8192)
#     test_gemm(dtype, 128, 32, 8192)
#     # # gating
#     # for (m, n, k) in [(4096, 32, 8192),
#     #                   (128, 32, 8192)]:
#     #     test_gemm(dtype, m, n, k)
#     # # gating
#     # for (m, n, k) in [(1, 19392, 8192),
#     #                   (128, 19392, 8192)]:
#     #     test_gemm(dtype, m, n, k)
