from aiter import hipb_mm, hipb_create_extension
from aiter.utility import dtypes
import torch
import pdb
from aiter.test_common import checkAllclose, perftest
from aiter.ops.shuffle import shuffle_weight
import aiter







@perftest()
def aiter_hip_swizzle(inp, weights, scaleA, scaleB, dtype):
    if scaleB is not None:
        scaleB = scaleB.t()
    
    
    return hipb_mm(
    inp, 
    weights.t(), 
    solution_index=-1, 
    bias=None, 
    out_dtype=dtype, 
    scaleA=scaleA, 
    scaleB=scaleB, 
    scaleOut=None,
    swizzle=True
)




@perftest()
def aiter_hip(inp, weights, scaleA, scaleB, dtype):
    if scaleB is not None:
        scaleB = scaleB.t()

    return hipb_mm(
    inp, 
    weights.t(), 
    solution_index=-1, 
    bias=None, 
    out_dtype=dtype, 
    scaleA=scaleA, 
    scaleB=scaleB, 
    scaleOut=None
)



@perftest()
def torch_gemm_fp8(inp, weights, scale_a, scale_b, out_dtype):
    output = torch._scaled_mm(inp,
                            weights.t(),
                            out_dtype=out_dtype,
                            scale_a=scale_a,
                            scale_b=scale_b.t(),
                            bias=None)
    
    return output



@perftest()
def torch_gemm_bf16(inp, weights):
    return torch.matmul(inp, weights.t())



@perftest()
def run_gemm_ck_bpreshuffle(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale, None, dtype)




hipb_create_extension()
dtype = dtypes.bf16


def swizzle_test_fp8(m, n, k):
    dtype = dtypes.bf16

    inp = torch.randn((m, k), dtype=dtype, device="cuda")
    weights = torch.randn((n, k), dtype=dtype, device="cuda")
    inp, x_scale = aiter.pertoken_quant(inp, quant_dtype=dtypes.fp8)
    weights, w_scale = aiter.pertoken_quant(weights, quant_dtype=dtypes.fp8)

    out_torch, time_torch = torch_gemm_fp8(inp, weights, x_scale, w_scale, torch.bfloat16)

    out_hip, time_hip = aiter_hip(inp, weights, x_scale, w_scale, torch.bfloat16)


    weight_swizzle = shuffle_weight(weights, layout=(16, 16), use_int4=False)
    
    out_sw, time_sw = aiter_hip_swizzle(inp, weight_swizzle, x_scale, w_scale, torch.bfloat16)

    out_ck, time_ck = run_gemm_ck_bpreshuffle(inp, weight_swizzle, x_scale, w_scale, torch.bfloat16)
    


    err_b = checkAllclose(out_sw, out_torch, msg=f"swizzle time:{time_sw}, hipblaslt time:{time_torch} ", rtol=1e-2, atol=1e-2)

    if err_b == 0:
        msg = f'mnk={m} {n} {k}: swizzle time:{time_sw}, hipb_mm time:{time_hip}, ck time:{time_ck}, torch._scaled_mm time:{time_torch} All Close!'
    else:
        msg = f'mnk={m} {n} {k}: swizzle time:{time_sw}, hipb_mm time:{time_hip}, ck time:{time_ck}, torch._scaled_mm time:{time_torch} Not All Close!'

    # msg = f'{time_sw} {time_hip} {time_ck} {time_torch}'
    

    return msg





def swizzle_test_bf16(m, n, k):
    dtype = dtypes.bf16

    inp = torch.randn((m, k), dtype=dtype, device="cuda")
    weights = torch.randn((n, k), dtype=dtype, device="cuda")

    out_torch, time_torch = torch_gemm_bf16(inp, weights)
    out_hipb, time_hipb = aiter_hip(inp, weights, None, None, torch.bfloat16)


    weight_swizzle = shuffle_weight(weights, layout=(16, 16), use_int4=False)
    out_swizzle, time_swizzle = aiter_hip_swizzle(inp, weight_swizzle, None, None, torch.bfloat16)
    


    err_b = checkAllclose(out_swizzle, out_torch, msg=f"swizzle time:{time_swizzle}, hipblaslt time:{time_torch} ", rtol=1e-2, atol=1e-2)

    if err_b == 0:
        msg = f'mnk={m} {n} {k}: swizzle time:{time_swizzle}, hipb_mm time:{time_hipb}, hipblaslt time:{time_torch} All Close!'
    else:
        msg = f'mnk={m} {n} {k}: swizzle time:{time_swizzle}, hipb_mm time:{time_hipb}, hipblaslt time:{time_torch} Not All Close!'
    

    # msg = f'{time_swizzle} {time_hipb} {time_torch}'

    return msg




n, k = 7424, 8192
msg_all = []
func_dtype = 'bf16'


func = swizzle_test_fp8 if func_dtype == 'fp8' else swizzle_test_bf16


for m in [16, 32, 48, 64, 4096, 5120, 8192]:

    msg = func(m, n, k)
    msg_all.append(msg)

for msg in msg_all:
    print(msg)
