# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest
from aiter.ops.shuffle import shuffle_weight
from gemm_a8w8_bpreshuffle_common import kernelInstance, kernels_list
import argparse
from einops import rearrange
from einops import repeat as eirp

block_shape = (128, 128)

def checkClose(a, b, rtol=1e-3, atol=0.01):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        return True
    else:
        percent = (a[mask]).numel()/a.numel()
        if percent > 0.01:
            return False
        else:
            return True

def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n =  (n + block_shape_n - 1) // block_shape_n
    scale_k =  (k + block_shape_k - 1) // block_shape_k
    # x_scale = rearrange(x_scale.view(-1, 1).repeat(1, block_shape_n*block_shape_k).view(m, scale_k, 1, block_shape_k),
    #                           'num_blk_n num_blk_k blk_n blk_k ->(num_blk_n blk_n) (num_blk_k blk_k)')
    x = x.to(x_scale.dtype).view(m, k//block_shape[1], block_shape[1]) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(w_scale.view(-1, 1).repeat(1, block_shape_n*block_shape_k).view(scale_n, scale_k, block_shape_n, block_shape_k),
                              'num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)')
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))
    # scale = torch.matmul(x_scale, w_scale)
    # out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)

def run_torch2(x, weight, x_scale, w_scale, dtype=torch.float16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]

    x_scale_ = eirp(x_scale, "m k -> m (k repeat)", repeat=block_shape_k)
    x_scale_ = x_scale_[:m, :k]

    w_scale_ = eirp(w_scale, "n k -> (n repeat) k", repeat=block_shape_n)
    w_scale_ = eirp(w_scale_, "n k -> n (k repeat)", repeat=block_shape_k)
    w_scale_ = w_scale_[:n, :k]

    x_ = x.to(x_scale.dtype) * x_scale_
    weight_ = weight.to(w_scale.dtype) * w_scale_

    out = F.linear(x_.to(torch.float32), weight_.to(torch.float32))
    return out.to(dtype)
def get_untuned_gemm_list(untuned_gemm_file):
    assert os.path.exists(untuned_gemm_file), f"Not exist a8w8_bpreshuffle_untuned_gemm.csv file: {untuned_gemm_file}"
    untunedf = pd.read_csv(untuned_gemm_file)
    return untunedf

def get_tuned_gemm_list(tuned_gemm_file):
    if os.path.exists(tuned_gemm_file):
        tunedf = pd.read_csv(tuned_gemm_file)
    else:
        tunedf = pd.DataFrame(columns=["M", "N", "K", "kernelId", "splitK", "us", "kernelName"])
    return tunedf

@perftest()
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a8w8_bpreshuffle_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out


def tune_gemm(m, n, k, useSplitK = False):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = (torch.rand((m, k), dtype=torch.float32, device="cuda") / 10).to(
        torch.float8_e4m3fnuz
    )
    weight = (torch.rand((n, k), dtype=torch.float32, device="cuda") / 10).to(
        torch.float8_e4m3fnuz
    )

    x_scale = torch.ones([scale_k, scale_m], dtype=torch.float32, device="cuda")
    w_scale = torch.ones([scale_k, scale_n], dtype=torch.float32, device="cuda")

    x_scale_trans = torch.transpose(x_scale, 0, 1)
    w_scale_trans = torch.transpose(w_scale, 0, 1)

    flat_weight = weight.view(n // 16, 16, k // 64, 4, 16)
    flat_weight = flat_weight.permute(0, 2, 3, 1, 4).contiguous()
    flat_weight = flat_weight.view(n, -1)
    ref_out = run_torch2(x, weight, x_scale_trans, w_scale_trans, torch.float16)
    # avg_c = run_gemm_ck_bpreshuffle(x, flat_weight, x_scale, w_scale, dtype)
    out = torch.empty(m, n, dtype=torch.float16, device="cuda")

    print(f"*******************M:{m} X N:{n} X K:{k}**************************")
    print(f"Start tuning a8w8 bpreshuffle gemm kernel for M:{m}, N:{n}, K:{k}")
    kernels_num = len(kernels_list)
    best_kernelConfig = (-1, 0)
    best_time = -1
    for i in range(kernels_num):
        kernel = kernels_list[i]
        maxsplitK = aiter.compute_gemm_SplitK(m, n, k, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK) \
            if useSplitK else 0
        for splitK in range(maxsplitK+1):
            try:
                (out), avg_t = kernel_instance_test(x, flat_weight, x_scale, w_scale, out, i, splitK)
                isClosed = checkClose(ref_out, out, rtol=1e-2, atol=0.1)
                if isClosed:
                    print(f"{str(dim):<20} kernelid:{i:<3d}\t avg: {avg_t:<8.2f} us, {kernel.name}, {splitK=}")
                    if best_time < 0 or avg_t < best_time:
                        best_kernelConfig = (i, splitK)
                        best_time = avg_t
                else:
                    print(f"{str(dim):<20} kernelid:{i:<3d}\t No pass         , {kernel.name}, {splitK=}") 
            except RuntimeError as e:
                print(e)
                print(f"{str(dim):<20} kernelid:{i:<3d}\t No support      , {kernel.name}, {splitK=}") 

    best_kernelId, splitK = best_kernelConfig
    if best_kernelConfig[0] == -1:
        print(f"No kernel can be used for M:{m}, N:{n}, K:{k}")
        best_time = 'nan'
    else:
        best_time = round(best_time, 4)
        
        print(f"Tuning result for M:{m}, N:{n}, K:{k} is kernelId={best_kernelId} {kernels_list[best_kernelId].name} {splitK=}, {best_time}us")
    print(f"*******************M:{m} X N:{n} X K{k}**************************")
    
    return best_kernelId, splitK, best_time


def tune_gemm_list(untunedf, tunedf, issorted = False, useSplitK = False):
    for i in range(len(untunedf)):
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]
        
        if tunedf[(tunedf["M"]==M) & (tunedf["N"]==N) & (tunedf["K"]==K)].empty:
            kernelId, splitK, time = tune_gemm(M, N, K, useSplitK)
            kernelName = 'None' if kernelId == -1 else kernels_list[kernelId].name
            temp = pd.DataFrame({"M":[M], "N":[N], "K":[K], "kernelId":[kernelId], "splitK":[splitK], 
                           "us":[time], "kernelName":[kernelName]})
            tunedf = pd.concat([tunedf, temp], ignore_index=True)

        else:
            print(f"M:{M}, N:{N}, K{K} is in tuned gemm, skip!!!")
        print()
        print()
    if issorted:
        tunedf = tunedf.sort_values(by=["M", "N", "K"])
    print("Totall tuning result:")
    print(tunedf)
    return tunedf



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm a8w8 kernel",
    )

    parser.add_argument(
        "-i",
        "--untune_file",
        default="aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv",
        required=False,
        help="input"
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv",
        required=False,
        help="output: tuning result store this file"
    )

    parser.add_argument(
        "-k",
        "--splitK",
        action='store_true',
        required=False,
        help="Use splitK kernels"
    )

    parser.add_argument(
        "--sort",
        action='store_true',
        required=False,
        help="Arranged according to the M N K size"
    )

    args = parser.parse_args()
    untunedf = get_untuned_gemm_list(args.untune_file)
    tunedf = get_tuned_gemm_list(args.tune_file)
    tunedf = tune_gemm_list(untunedf, tunedf, args.sort, args.splitK)
    tunedf.to_csv(args.tune_file, index=False)
