# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.utility import fp4_utils
from aiter.test_common import checkAllclose, perftest, benchmark
from einops import rearrange
from einops import repeat as eirp
import pandas as pd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
# SCALE_GROUP_SIZE = 32
block_shape = (128, 128)


# @perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a4w4_blockscale_CK(x, weight, x_scale, w_scale, dtype)


# @benchmark()
def test_gemm(dtype, m, n, k):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ["gfx950"]:
        return

    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((m, k), dtype=dtype)
    w = torch.randn((n, k), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    out_ck = torch.empty((m + 255) // 256 * 256, n, dtype=dtype)
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)

    print(x.shape)
    print(x_scales.shape)
    print(x_scales.stride())

    # b, avg_b = run_gemm_ck(x, w, x_scales_shuffle, w_scales_shuffle, out_ck)



    # dim = (m, n, k)
    # block_shape_n, block_shape_k = block_shape
    # scale_n = (n + block_shape_n - 1) // block_shape_n
    # scale_k = (k + block_shape_k - 1) // block_shape_k
    # x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    # weight = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    # x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    # w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    # b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, dtype)

    # msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    # checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)

    # return {"us": avg_b}


df = []
for dtype in [dtypes.bf16]:
    # deepseek-r1
    for m in [64]:
        for n, k in [
            (512, 1024),
        ]:
            ret = test_gemm(dtype, m, n, k)
#             df.append(ret)
# df = pd.DataFrame(df)
# aiter.logger.info(f"summary:\n{df}")
