# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from gemm_a8w8_common import kernels_params_dict
import argparse

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
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)

def get_untuned_gemm_list(untuned_gemm_file):
    assert os.path.exists(untuned_gemm_file), f"Not exist a8w8_untuned_gemm.csv file: {untuned_gemm_file}"
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
    aiter.gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out

def random_tensor(a: torch.tensor, b: torch.tensor, dtype: torch.dtype) -> torch.tensor:
    if dtype == torch.int8:
        return torch.randint(-20, 20, (a, b), dtype=dtype, device="cuda")
    elif dtype == torch.float8_e4m3fnuz:
        return torch.rand((a, b), device="cuda").to(dtype)
    raise RuntimeError("Unsupported data type.")

def tune_gemm(m, n, k, dtype: torch.dtype, useSplitK = False):
    dim = (m, n, k)
    x = random_tensor(m, k, dtype)
    weight = random_tensor(n, k, dtype)
    x_scale = torch.rand([m, 1], dtype=torch.bfloat16, device="cuda")
    w_scale = torch.rand([1, n], dtype=torch.bfloat16, device="cuda")
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

    ref_out = run_torch(x, weight, x_scale, w_scale)

    print(f"*******************M:{m} X N:{n} X K:{k}**************************")
    print(f"Start tuning a8w8 gemm kernel for M:{m}, N:{n}, K{k}:")
    kernels_num = len(kernels_params_dict)
    best_kernelConfig = (-1, 0)
    best_time = -1
    for i in range(kernels_num):
        kernel = kernels_params_dict[i]
        maxsplitK = aiter.compute_gemm_SplitK(m, n, k, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK) \
            if useSplitK else 0
        for splitK in range(maxsplitK+1):
            try:
                (out), avg_t = kernel_instance_test(x, weight, x_scale, w_scale, out, i, splitK)
                isClosed = checkClose(ref_out, out, rtol=1e-2, atol=0.01)
                if isClosed:
                    print(f"{str(dim):<20} kernelid:{i:<3d}\t avg: {avg_t:<8.2f} us, {kernel.name}, {splitK=}")
                    if best_time < 0 or avg_t < best_time:
                        best_kernelConfig = (i, splitK)
                        best_time = avg_t
                else:
                    print(f"{str(dim):<20} kernelid:{i:<3d}\t No pass         , {kernel.name}, {splitK=}") 
            except RuntimeError as e:
                print(str(e))
                print(f"{str(dim):<20} kernelid:{i:<3d}\t No support      , {kernel.name}, {splitK=}") 

    best_kernelId, splitK = best_kernelConfig
    if best_kernelConfig[0] == -1:
        print(f"No kernel can be used for M:{m}, N:{n}, K:{k}")
        best_time = 'nan'
    else:
        best_time = round(best_time, 4)
        
        print(f"Tuning result for M:{m}, N:{n}, K:{k} is kernelId={best_kernelId} {kernel.name} {splitK=}, {best_time}us")
    print(f"*******************M:{m} X N:{n} X K{k}**************************")
    
    return best_kernelId, splitK, best_time


def tune_gemm_list(untunedf, tunedf, dtype: torch.dtype, issorted = False, useSplitK = False):
    print("untuned df is \n\n", untunedf)
    for i in range(len(untunedf)):
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]
        
        if tunedf[(tunedf["M"]==M) & (tunedf["N"]==N) & (tunedf["K"]==K)].empty:
            kernelId, splitK, time = tune_gemm(M, N, K, dtype, useSplitK)
            kernelName = 'None' if kernelId == -1 else kernels_params_dict[kernelId].name
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
        default="aiter/configs/a8w8_untuned_gemm.csv",
        required=False,
        help="input"
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_gemm.csv",
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
        "-d",
        "--dtype",
        required=False,
        default="int8",
        help="int8 or fp8"
    )

    parser.add_argument(
        "--sort",
        action='store_true',
        required=False,
        help="Arranged according to the M N K size"
    )

    args = parser.parse_args()
    dtype = torch.float8_e4m3fnuz if args.dtype == "fp8" else torch.int8
    untunedf = get_untuned_gemm_list(args.untune_file)
    tunedf = get_tuned_gemm_list(args.tune_file)
    tunedf = tune_gemm_list(untunedf, tunedf, dtype, args.sort, args.splitK)
    tunedf.to_csv(args.tune_file, index=False)
