# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, perftest, run_perftest
from aiter import dtypes
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight
import argparse
import pandas as pd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


@perftest(num_iters=5)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    aiter.logger.info(f"x shape: {x.shape}, w shape: {w.shape}, x_f32 shape: {x_f32.shape}, w_f32 shape: {w_f32.shape}, x_scale shape: {x_scales.shape}, w_scale shape: {w_scales.shape}")
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    aiter.logger.info(f"x_f32 shape: {x_f32.shape}, x_scales_f32 shape: {x_scales_f32.shape}")
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, out):
    return aiter.gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out)


@perftest()
def run_triton(x, w, x_scales, w_scales, out, dtype=dtypes.bf16):
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4

    gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, out)
    return out


@perftest()
def run_gemm_asm(
    x,
    weightshuffle,
    x_scale,
    w_scale,
    out,
    kernelName="",
    bias=None,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    log2_k_split=None,
):
    # if log2_k_split is not None and log2_k_split > 0:
    #     out_reset = torch.zeros(
    #         (out.shape[0] + 31) // 32 * 32, out.shape[1], dtype=dtype
    #     )
    #     out = out_reset

    aiter.gemm_a4w4_asm(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=log2_k_split,
    )
    return out


@benchmark()
def test_gemm_asm(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ["gfx950"]:
        return
    ret = {}
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    out1 = torch.empty(M, N, dtype=dtype)
    out2 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    out3 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    bias_f32 = None
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    a, avg_a = run_torch(x, w, x_scales, w_scales, dtype)
    # b, avg_b = run_triton(x, w.T, x_scales, w_scales, out1, dtype)
    # b, avg_b = a, 0
    # err_b = checkAllclose(a, b, msg="triton        ")

    try:
        c, us = run_perftest(
            aiter.gemm_a4w4,
            x,
            wshuffle,
            x_scales_shuffle,
            w_scales_shuffle,
            out2,
            bpreshuffle=True,
        )
        err = checkAllclose(a, c[:M], msg="unified api")
    except RuntimeError as e:
        err = str(e)
        us = float("nan")
        aiter.logger.warning(f"gemm_a4w4 failed: dtype={dtype} M={M} N={N} K={K} error={err}")
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err

    # kernelName = ""  # "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"
    # log2_k_split = 1
    # d, us = run_gemm_asm(
    #     x,
    #     wshuffle,
    #     x_scales_shuffle,
    #     w_scales_shuffle,
    #     out3,
    #     kernelName,
    #     bias_f32,
    #     bpreshuffle=True,
    #     log2_k_split=log2_k_split,
    # )
    # err = checkAllclose(a, d[:M], msg=f"asm {kernelName} log2_k_split_{log2_k_split}")
    # tag = "asm_dbg"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    # e, us = run_gemm_ck(x, wshuffle, x_scales_shuffle, w_scales_shuffle, out3)
    # err = checkAllclose(a, e[:M], msg="ck            ")
    # tag = "ck"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    return ret

@benchmark()
def test_gemm_triton(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4

    if get_gfx() not in ["gfx950"]:
        return
    ret = {}
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    out1 = torch.empty(M, N, dtype=dtype)
    out2 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    out3 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    bias_f32 = None
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    a, avg_a = run_torch(x, w, x_scales, w_scales, dtype)
    # b, avg_b = run_triton(x, w.T, x_scales, w_scales, out1, dtype)
    # b, avg_b = a, 0
    # err_b = checkAllclose(a, b, msg="triton        ")

    c, us = run_perftest(
        gemm_afp4wfp4,
        x.view(torch.uint8),
        w.view(torch.uint8),
        x_scales,
        w_scales,
        dtype,
        out1,
    )
    err = checkAllclose(a, c[:M], msg="triton gemm")
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err
    return ret

@benchmark()
def test_gemm_baseline(dtype, M, N, K):
    ret = {}
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    bias_f32 = None

    _, us = run_perftest(
        torch.mm,
        x,
        w.T
    )
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = 0
    return ret

l_dtype = ["bf16"]
l_mnk = [
    # pure_compute
    (256, 2048, 8192),
    (2048, 8192, 8192),
    (16384, 16384, 16384),
    (32768, 106496, 16384),
    (32768, 16384, 53248),
    (32768, 18432, 16384),
    (32768, 16384, 16384),
    (128, 106496, 16384),
    (128, 16384, 53248),
    (128, 18432, 16384),
    (128, 16384, 16384),
    (64, 106496, 16384),
    (64, 16384, 53248),
    (64, 18432, 16384),
    (64, 16384, 16384),
    (64, 106496, 16384),
    (32, 106496, 16384),
    (32, 16384, 53248),
    (32, 18432, 16384),
    (32, 16384, 16384),
    # qkv_proj
    (1, 1280, 8192),
    (64, 1280, 8192),
    (127, 1280, 8192),
    (129, 1280, 8192),
    (65, 1280, 8192),
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
    # tune
    (1552, 8192, 8192),
    (1664, 8192, 8192),
    (1792, 8192, 8192),
    (1920, 8192, 8192),
    (3072, 8192, 8192),
    (1552, 10240, 8192),
    (1664, 10240, 8192),
    (1792, 10240, 8192),
    (1920, 10240, 8192),
    (3072, 10240, 8192),
    (1552, 57344, 8192),
    (1664, 57344, 8192),
    (1792, 57344, 8192),
    (1920, 57344, 8192),
    (3072, 57344, 8192),
    (1552, 8192, 28672),
    (1664, 8192, 28672),
    (1792, 8192, 28672),
    (1920, 8192, 28672),
    (3072, 8192, 28672),
]

# _l_m = [1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
_l_m = [64]
# _l_m = [1]
l_mnk = [
    # dsr1
    # qkv_a_proj
    *((i, 2112, 7168) for i in _l_m),
    # o_proj
    *((i, 7168, 256) for i in _l_m),
    # q_b_proj
    *((i, 3072, 1536) for i in _l_m),
    # shared_experts.gate_up_proj
    *((i, 512, 7168) for i in _l_m),
    # shared_experts.down_proj
    *((i, 7168, 2048) for i in _l_m),
    # gate_up_proj (dense mlp)
    *((i, 4608, 7168) for i in _l_m),
    # down_proj (dense mlp)
    *((i, 7168, 2304) for i in _l_m),
    # w_kc
    # (1, 8192, 128),  # shape error during quant
    # w_vc
    *((i, 2048, 512) for i in _l_m),
    # kv_b_proj,
    *((i, 4096, 512) for i in _l_m),
]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-mnk",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.shape is not None:
    l_mnk = [args.shape]

df_asm = []
df_triton = []
df_base = []
for dtype in l_dtype:
    for m, n, k in l_mnk:
        ret = test_gemm_asm(dtype, m, n, k)
        df_asm.append(ret)
        ret = test_gemm_triton(dtype, m, n, k)
        df_triton.append(ret)
        ret = test_gemm_baseline(dtype, m, n, k)
        df_base.append(ret)
df_asm = pd.DataFrame(df_asm)
df_triton = pd.DataFrame(df_triton)
df_base = pd.DataFrame(df_base)
aiter.logger.info(f"summary:\n{df_asm}")
aiter.logger.info(f"triton summary:\n{df_triton}")
aiter.logger.info(f"baseline summary:\n{df_base}")
