import torch
import torch.nn.functional as F
import numpy as np
import ater
import argparse
from ater.test_common import checkAllclose, perftest

num_iters = 100
num_warmup = 20

def pertoken_quant(hidden_states_input, y_scale_dtype, x_scale = None):
    # assume output int8, hidden_states is [m, n] shape and in fp16/bf16
    if x_scale is None:
        hidden_states = hidden_states_input
    else:
        # smooth quant
        hidden_states = hidden_states_input.to(x_scale) * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(
        input=torch.abs(hidden_states), 
        dim=-1, 
        keepdim=True
    )
    per_token_scale = per_token_amax.to(dtype=torch.float32) / 127.0

    # quant hidden_states
    hidden_states = (hidden_states / per_token_scale).to(dtype=torch.int8)

    return hidden_states, per_token_scale.to(y_scale_dtype)
    # hidden_states now is int8 will feed to next layer as intput
    # per_token_scale will be used as dequant factor later layer

@perftest()
def run_torch(input, weight, bias, eps, residual=None, x_scale = None, y_scale_dtype = None):
    if residual is None:
        residual_out = None
        output = F.layer_norm(
            input=input,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps
        )
    else:
        residual_out = input + residual
        output = F.layer_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps
        )
    if y_scale_dtype is None:
        y_scale = None
    else:
        output, y_scale = pertoken_quant(output, y_scale_dtype, x_scale=x_scale)
    return output, residual_out, y_scale

@perftest()
def run_ck(input, weight, bias, eps, residual=None, x_scale = None, y_scale_dtype = None):
    if y_scale_dtype is None:
        y_scale = None
        if residual is None:
            residual_out = None
            output = torch.empty_like(input)
            ater.layer_norm(
                input,
                weight,
                bias,
                eps
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            output = torch.empty_like(input)
            ater.layernorm2d_fwd_with_add(
                output,
                input,
                residual,
                residual_out,
                weight,
                bias,
                eps
            )
    elif x_scale is None:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            ater.layernorm2d_fwd_with_dynamicquant(
                output,
                input,
                y_scale,
                weight,
                bias,
                eps
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            ater.layernorm2d_fwd_with_add_dynamicquant(
                output,
                input,
                residual,
                residual_out,
                y_scale,
                weight,
                bias,
                eps
            )
    else:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            ater.layernorm2d_fwd_with_smoothquant(
                output,
                input,
                x_scale,
                y_scale,
                weight,
                bias,
                eps
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            ater.layernorm2d_fwd_with_add_smoothquant(
                output,
                input,
                residual,
                residual_out,
                x_scale,
                y_scale,
                weight,
                bias,
                eps
            )
        
    return output, residual_out, y_scale



def test_layernorm2d_instance(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    (a, *_), avg_a = run_torch(input, weight, bias, 1e-5)
    (b, *_), avg_b = run_ck(input, weight, bias, 1e-5)
    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    checkAllclose(a, b)
    print(f"[passed~]")


def test_layernorm2d_fuseAdd_instance(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    (a, res_a, *_), avg_a = run_torch(input, weight, bias, 1e-5, residual=res)
    (b, res_b, *_), avg_b = run_ck(input, weight, bias, 1e-5, residual=res)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    checkAllclose(a, b, rtol=1e-2, atol=1e-1)
    checkAllclose(res_a, res_b)
    print(f" [passed~]")


def test_layernorm2d_fuseSmoothquant_instance(dtype, m, n, xscaleType, yscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    xscale = torch.randn(n, dtype=xscaleType, device="cuda")
    (a, _, yscale_a), avg_a = run_torch(input, weight, bias, 1e-5, x_scale=xscale, y_scale_dtype=yscaleType)
    (b, _, yscale_b), avg_b = run_ck(input, weight, bias, 1e-5, x_scale=xscale, y_scale_dtype=yscaleType)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)
    print(f" [passed~]")

def test_layernorm2d_fuseAdd_Smoothquant_instance(dtype, m, n, xscaleType, yscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    xscale = torch.randn(n, dtype=xscaleType, device="cuda")
    (a, res_a, yscale_a), avg_a = run_torch(input, weight, bias, 1e-5, residual=res, x_scale=xscale, y_scale_dtype=yscaleType)
    (b, res_b, yscale_b), avg_b = run_ck(input, weight, bias, 1e-5, residual=res, x_scale=xscale, y_scale_dtype=yscaleType)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(res_a, res_b)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)
    print(f" [passed~]")


def test_layernorm2d_fuseDynamicquant_instance(dtype, m, n, yscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    (a, _, yscale_a), avg_a = run_torch(input, weight, bias, 1e-5, y_scale_dtype=yscaleType)
    (b, _, yscale_b), avg_b = run_ck(input, weight, bias, 1e-5,  y_scale_dtype=yscaleType)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(yscale_a, yscale_b)
    print(f" [passed~]")

def test_layernorm2d_fuseAdd_Dynamicquant_instance(dtype, m, n, yscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    (a, res_a, yscale_a), avg_a = run_torch(input, weight, bias, 1e-5, residual=res, y_scale_dtype=yscaleType)
    (b, res_b, yscale_b), avg_b = run_ck(input, weight, bias, 1e-5, residual=res, y_scale_dtype=yscaleType)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(res_a, res_b)
    checkAllclose(yscale_a, yscale_b)
    print(f" [passed~]")

def test_layernorm2d():
    print('\nstart layernorm2d test')
    for dtype in [torch.float16, torch.bfloat16]:
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for n in [4096, 8192, 16384, 32768, 65536]:
                test_layernorm2d_instance(dtype, m, n)

def test_layernorm2d_fuseAdd():
    print('\nstart layernorm2d fuse add test')
    for dtype in [torch.float16, torch.bfloat16]:
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for n in [4096, 8192, 16384, 32768, 65536]:
                test_layernorm2d_fuseAdd_instance(dtype, m, n)

def test_layernorm2d_fuseSmoothquant():
    print('\nstart layernorm2d fuse Smoothquant test')
    for scaleType in [ torch.float32]:
        for dtype in [torch.float16, torch.bfloat16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [10, 4096, 8192]:
                    test_layernorm2d_fuseSmoothquant_instance(dtype, m, n, xscaleType=scaleType, yscaleType=scaleType)

def test_layernorm2d_fuseAdd_Smoothquant():
    print('\nstart layernorm2d fuse add Smoothquant test')
    for scaleType in [torch.float32]:
        for dtype in [torch.float16, torch.bfloat16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [4096, 8192]:
                    test_layernorm2d_fuseAdd_Smoothquant_instance(dtype, m, n, xscaleType=scaleType, yscaleType=scaleType)

def test_layernorm2d_fuseDynamicquant():
    print('\nstart layernorm2d fuse Smoothquant test')
    for scaleType in [ torch.float32]:
        for dtype in [torch.float16, torch.bfloat16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [1024, 2048]:
                    test_layernorm2d_fuseDynamicquant_instance(dtype, m, n, yscaleType=scaleType)

def test_layernorm2d_fuseAdd_Dynamicquant():
    print('\nstart layernorm2d fuse add Smoothquant test')
    for scaleType in [torch.float32]:
        for dtype in [torch.float16, torch.bfloat16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [1024, 2048]:
                    test_layernorm2d_fuseAdd_Dynamicquant_instance(dtype, m, n, yscaleType=scaleType)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_layernorm2dFusedSQuant",
        description="Test ck layernorm2d Fused add and SmoothQuant")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="1: test_layernorm2d, \n2:test_layernorm2d_fuseAdd, \n"+
            "3:test_layernorm2d_fuseSmoothquant, \n4:test_layernorm2d_fuseAdd_Smoothquant"+
            "5:test_layernorm2d_fuseDynamicquant, \n6:test_layernorm2d_fuseAdd_Dynamicquant",
        default=1,
    )
    # parser.add_argument(
    #     "--GPUID",
    #     type=str,
    #     help="This script uses single GPU. Specify the GPU to use for tuning",
    #     default="0",
    # )
    args =  parser.parse_args()
    if args.mode == 1:
        test_layernorm2d()
    elif args.mode == 2:
        test_layernorm2d_fuseAdd()
    elif args.mode == 3:
        test_layernorm2d_fuseSmoothquant()
    elif args.mode == 4:
        test_layernorm2d_fuseAdd_Smoothquant()
    elif args.mode == 5:
        test_layernorm2d_fuseDynamicquant()
    elif args.mode == 6:
        test_layernorm2d_fuseAdd_Dynamicquant()