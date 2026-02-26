# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import os
import torch
from aiter.ops.triton.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffle,
    _get_config,
    get_splitk,
    set_use_gemm_splitk_bf16
)
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.ops.shuffle import shuffle_weight
import argparse
import time



def shuffle_scales(scales: torch.Tensor):
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape
    scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


def un_shuffle_scales(scales_shuffled: torch.Tensor):
    scales = scales_shuffled.clone()
    sm, sn = scales.shape
    scales = scales.view(sm * 32, sn // 32)
    sm, sn = scales.shape
    scales = scales.view(sm // 32, sn // 8, 4, 16, 2, 2, 1)
    scales = scales.permute(0, 5, 3, 1, 4, 2, 6).contiguous()
    scales = scales.view(sm, sn)
    return scales


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
    M,
    N,
    K,
    layout="TN",
    output=True,
    shuffle_weight_fg=False,
    shuffle_scales_fg=False,
    dtype=torch.float32,
):
    if shuffle_weight_fg:
        assert (
            shuffle_scales_fg
        ), "weight shuffling is only supported with scale shuffling"

    torch.manual_seed(5)

    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
        x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    else:
        x_low = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T
        x_high = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T

    if layout[1] == "N":
        w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T
        w_high = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T

    x = (
        x_high << 4 | x_low
    )  # Doing this computation on GPU tensors results in NaNs, so move it to GPU afterwards
    x = x.to(device="cuda")

    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    M_pad = (M + 255) // 256 * 256
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M_pad), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.T
    w_scales = w_scales.T
    if shuffle_scales_fg:
        if M >= 32:
            x_scales_shuffled = shuffle_scales(x_scales)
        else:
            x_scales_shuffled = x_scales.contiguous()
        w_scales_shuffled = shuffle_scales(w_scales)
    else:
        x_scales_shuffled = x_scales
        w_scales_shuffled = w_scales

    if shuffle_weight_fg:
        use_int4 = False
        weight_shuffle_layout = (16, 16)
        w_shuffed = shuffle_weight(
            w, layout=weight_shuffle_layout, use_int4=use_int4
        ).reshape(
            w.shape[0] // weight_shuffle_layout[0],
            w.shape[1] * weight_shuffle_layout[0],
        )
    else:
        w_shuffed = w

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype).cuda()
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return (
        x,
        w,
        w_shuffed,
        x_scales[:M],
        w_scales,
        x_scales_shuffled[:M],
        w_scales_shuffled,
        y,
    )


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch(x, w, x_scales, w_scales, dtype):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)

args = argparse.ArgumentParser()
args.add_argument("--M", type=int, default=1024)
args.add_argument("--N", type=int, default=1024)
args.add_argument("--K", type=int, default=1024)
args.add_argument("--shuffle_weight_scales", type=int, default=1)
args.add_argument("--repeat", type=int, default=1)
args.add_argument("--warmup", type=int, default=0)
args.add_argument("--num_copies", type=int, default=1)
args.add_argument("--out_dtype", type=str, default="float32")
args.add_argument("--skip_reduce", type=int, default=0)
args.add_argument('--override_config', type=int, default=0, help='override the config with the one in the script')
args.add_argument('--acc_check', type=int, default=0, help='check the accuracy of the result')

args = args.parse_args()

layout = "TN"
dtype = str_to_torch_dtype[args.out_dtype]
print("dtype", dtype)
(
    x,
    w,
    w_triton,
    x_scales,
    w_scales,
    x_scales_triton,
    w_scales_triton,
    y,
) = generate_gemm_afp4wfp4_inputs(
    args.M,
    args.N,
    args.K,
    layout=layout,
    output=True,
    shuffle_scales_fg=args.shuffle_weight_scales,
    shuffle_weight_fg=args.shuffle_weight_scales,
    dtype=dtype,
)

use_aot = False
skip_reduce = args.skip_reduce
# config hack
if args.override_config:
    config = _get_config(args.M, 16384, 53248// 2, shuffle=args.shuffle_weight_scales)[0]
    # calculate next power of 2 for args.M
    next_power_of_2 = 2 ** ((args.M - 1).bit_length())
    # if args.M == 128 or True:
    config["waves_per_eu"] = 2
    config["num_stages"] = 2
    config["BLOCK_SIZE_M"] = max(next_power_of_2, 16)
    config["BLOCK_SIZE_N"] = 256
    config["BLOCK_SIZE_K"] = 256
    config["NUM_KSPLIT"] = 1
    config["num_warps"] = 4
    config["GROUP_SIZE_M"] = 1

    print(f"new config: {config}")
    y_pp = torch.empty((config["NUM_KSPLIT"], args.M, args.N), dtype=torch.float32, device="cuda")
else:
    config = _get_config(args.M, args.N, args.K // 2, shuffle=args.shuffle_weight_scales)[0]
    y_pp = torch.empty((config["NUM_KSPLIT"], args.M, args.N), dtype=torch.float32, device="cuda")
    config = None



if args.shuffle_weight_scales:
    def run_once(x_in, w_in, x_scales_in, w_scales_in, y_out):
        return gemm_afp4wfp4_preshuffle(
            x_in,
            w_in,
            x_scales_in,
            w_scales_in,
            dtype,
            y_out,
            use_aot=use_aot,
            config=config,
            skip_reduce=skip_reduce,
        )
else:
    def run_once(x_in, w_in, x_scales_in, w_scales_in, y_out):
        return gemm_afp4wfp4(
            x_in, w_in, x_scales_in, w_scales_in, dtype, y_out, config=config
        )
num_copies = max(1, int(args.num_copies))
input_sets = []
total_input_size = 0
total_input_size += x.numel() * x.element_size()
total_input_size += w_triton.numel() * w_triton.element_size()
total_input_size += x_scales_triton.numel() * x_scales_triton.element_size()
total_input_size += w_scales_triton.numel() * w_scales_triton.element_size()
total_input_size = total_input_size * num_copies
print(f"total_input_size: {total_input_size / (1000*1000)} MB")
if args.acc_check:
    print("Checking accuracy...")
    print("*"*40)
    run_once(x, w_triton, x_scales_triton, w_scales_triton, y)
    y_torch = run_torch(x, w, x_scales, w_scales, dtype)
    max_diff = (y_torch - y).abs().max().item()    
    print("Max difference:", max_diff)
    if max_diff > 1e-2:
        print("Accuracy check failed!")
    else:
        print("Accuracy check passed!")
    print("*"*40)
else:
    for _ in range(num_copies):
        input_sets.append(
            (
                x.clone(),
                w_triton.clone(),
                x_scales_triton.clone(),
                w_scales_triton.clone(),
                torch.empty_like(y),
            )
        )
    i = 0
    for _ in range(args.warmup):
        idx = i % num_copies
        x_i, w_i, x_scales_i, w_scales_i, y_i = input_sets[idx]
        run_once(x_i, w_i, x_scales_i, w_scales_i, y_i)
        i += 1
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(args.repeat):
        idx = i % num_copies
        x_i, w_i, x_scales_i, w_scales_i, y_i = input_sets[idx]
        run_once(x_i, w_i, x_scales_i, w_scales_i, y_i)
        i += 1
    end_time = time.time()
    elapsed_s = end_time - start_time
    print(f"time: {elapsed_s}")
    print(f"time per iteration in us: {(elapsed_s / args.repeat) * 1e6}")

