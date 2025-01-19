from ater.test_common import checkAllclose, perftest, tensor_dump
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import ater
from ater.ops.shuffle import shuffle_weight

@perftest()
def test_ater_perTensorQuantFp8(input, sacle=None):
    out = torch.empty(input.shape, dtype=torch.float8_e4m3fnuz, device=input.device)
    if sacle is None:
        sacle = torch.zeros(1, dtype=torch.float32, device=input.device)
        ater.dynamic_scaled_fp8_quant(out, input, sacle)
    else:
        ater.static_scaled_fp8_quant(out, input, sacle)
    return out, sacle

@perftest()
def test_torch_perTensorQuantFp8(input, sacle=None):
    if sacle is None:
        max_val = torch.max(torch.abs(input.view(-1)))
        # dtypeMax = torch.finfo(torch.float8_e4m3fnuz).max
        dtypeMax = 240.0
        sacle = max_val.to(torch.float32) / dtypeMax
        out = (input / sacle).to(torch.float8_e4m3fnuz)
    else:
        out = (input / sacle).to(torch.float8_e4m3fnuz)
    return out, sacle.view(1)
    

@perftest()
def test_ater_perTokenQuantFp8(input):
    # orign_shape = input.shape
    token_num = input.view(-1, input.shape[-1]).shape[0]
    out = torch.empty(input.shape, dtype=torch.float8_e4m3fnuz, device=input.device)
    sacles = torch.empty(token_num, dtype=torch.float32, device=input.device)
    ater.dynamic_per_token_scaled_fp8_quant(out, input, sacles)
    return out, sacles


@perftest()
def test_torch_perTokenQuantFp8(input):
    return ater.pertoken_quant(input, torch.float, quant_dtype=torch.float8_e4m3fnuz)


def test_quant(m, n, dtype=torch.bfloat16):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device='cuda')
    max_val = torch.max(torch.abs(input.view(-1)))
    print(f'{max_val=}')

    # (a, a_sacle), avg_a = test_torch_perTensorQuantFp8(input)
    # (b, b_sacle), avg_b = test_ater_perTensorQuantFp8(input)
    # msg = f"[perf] dynamic_perTensorQuantFp8 dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ater avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    # checkAllclose(a.to(torch.float), b.to(torch.float), msg=msg)
    # checkAllclose(a_sacle, b_sacle, msg='sacle')
    # print(f'{a_sacle=}, {b_sacle=}')

    # scale = a_sacle
    (a, a_sacle), avg_a = test_torch_perTensorQuantFp8(input)
    (b, b_sacle), avg_b = test_ater_perTensorQuantFp8(input)
    msg = f"[perf] static_perTensorQuantFp8 dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ater avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a.to(torch.float), b.to(torch.float), rtol=0.125, atol=1e-3, msg=msg)

    (a, a_sacle), avg_a = test_torch_perTokenQuantFp8(input)
    (b, b_sacle), avg_b = test_ater_perTokenQuantFp8(input)
    msg = f"[perf] dynamic_perTokenQuantFp8 dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ater avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a.to(torch.float), b.to(torch.float), rtol=0.125, atol=1e-3, msg=msg)
    checkAllclose(a_sacle.view(-1), b_sacle, msg='sacle')


for dtype in [torch.float16, torch.bfloat16][1:]:
    for m in [1, 16, 32, 64, 128, 192, 256]:
        for n in [4096, 8192]:
            test_quant(m, n, dtype=dtype)
    


