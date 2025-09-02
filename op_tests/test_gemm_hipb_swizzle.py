from aiter import hipb_mm, hipb_create_extension
from aiter.utility import dtypes
import torch
import pdb
from aiter.test_common import checkAllclose, perftest
from aiter.ops.shuffle import shuffle_weight
import aiter







@perftest()
def aiter_hip(inp, weights, scaleA, scaleB, dtype):
    return hipb_mm(
    inp, 
    weights.t(), 
    solution_index=-1, 
    bias=None, 
    out_dtype=dtype, 
    scaleA=scaleA, 
    scaleB=scaleB.t(), 
    scaleOut=None,
    swizzle=True
)


@perftest()
def torch_gemm(inp, weights, scale_a, scale_b, out_dtype):
    output = torch._scaled_mm(inp,
                            weights.t(),
                            out_dtype=out_dtype,
                            scale_a=scale_a,
                            scale_b=scale_b.t(),
                            bias=None)
    
    return output



hipb_create_extension()
dtype = dtypes.bf16


def swizzle_test(m, n, k):
    dtype = dtypes.bf16

    inp = torch.rand((m, k), dtype=dtype, device="cuda")
    weights = torch.rand((n, k), dtype=dtype, device="cuda")
    inp, x_scale = aiter.pertoken_quant(inp, quant_dtype=dtypes.fp8)
    weights, w_scale = aiter.pertoken_quant(weights, quant_dtype=dtypes.fp8)
    weight_swizzle = shuffle_weight(weights, layout=(16, 16), use_int4=False)
    bias = torch.rand([1, n], dtype=dtype, device="cuda") * 10



    out, time = aiter_hip(inp, weight_swizzle, x_scale, w_scale, torch.bfloat16)
    out_torch, time_torch = torch_gemm(inp, weights, x_scale, w_scale, torch.bfloat16)


    err_b = checkAllclose(out, out_torch, msg=f"swizzle time:{time}, hipblaslt time:{time_torch} ", rtol=1e-2, atol=1e-2)

    if err_b == 0:
        msg = f'mnk={m} {n} {k}: swizzle time:{time}, hipblaslt time:{time_torch} All Close!'
    else:
        msg = f'mnk={m} {n} {k}: swizzle time:{time}, hipblaslt time:{time_torch} Not All Close!'
    

    return msg




n, k = 7392, 8192
msg_all = []

for m in [16, 32, 48, 64, 4096]:

    msg = swizzle_test(m, n, k)
    msg_all.append(msg)

for msg in msg_all:
    print(msg)
