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
from batched_gemm_a8w8_common import kernelInstance, kernels_list
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
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    for b in range(B):
        b_x = F.linear(x[b, :, :].to(torch.float32), weight[b, :, :].to(torch.float32))
        b_scale = torch.matmul(x_scale[b, :, :], w_scale[b, :, :])
        b_out = torch.mul(b_x, b_scale)
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)

def get_untuned_batched_gemm_list(untuned_batched_gemm_file):
    assert os.path.exists(untuned_batched_gemm_file), f"Not exist a8w8_untuned_batched_gemm.csv file: {untuned_batched_gemm_file}"
    untunedf = pd.read_csv(untuned_batched_gemm_file)
    return untunedf

def get_tuned_batched_gemm_list(tuned_batched_gemm_file):
    if os.path.exists(tuned_batched_gemm_file):
        tunedf = pd.read_csv(tuned_batched_gemm_file)
    else:
        tunedf = pd.DataFrame(columns=["B", "M", "N", "K", "kernelId", "splitK", "us", "kernelName"])
    return tunedf

@perftest()
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.batched_gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out


def tune_batched_gemm(b, m, n, k, useSplitK = False):
    dim = (b, m, n, k)
    x = torch.randint(-20, 20, (b, m, k), dtype=torch.int8, device="cuda")
    weight = torch.randint(-20, 20, (b, n, k), dtype=torch.int8, device="cuda")
    x_scale = torch.rand([b, m, 1], dtype=torch.bfloat16, device="cuda")
    w_scale = torch.rand([b, 1, n], dtype=torch.bfloat16, device="cuda")
    out = torch.empty(b, m, n, dtype=torch.bfloat16, device="cuda")

    ref_out = run_torch(x, weight, x_scale, w_scale)

    print(f"*******************B:{b} X M:{m} X N:{n} X K:{k}**************************")
    print(f"Start tuning a8w8 batched_gemm kernel for B:{b}, M:{m}, N:{n}, K{k}:")
    kernels_num = len(kernels_list)
    best_kernelConfig = (-1, 0)
    best_time = -1
    for i in range(kernels_num):
        kernel = kernels_list[i]
        maxsplitK = aiter.compute_batched_gemm_SplitK(b, m, n, k, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK) \
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
                print(f"error = {e}")
                print(f"{str(dim):<20} kernelid:{i:<3d}\t No support      , {kernel.name}, {splitK=}")

    best_kernelId, splitK = best_kernelConfig
    if best_kernelConfig[0] == -1:
        print(f"No kernel can be used for B{b}, M:{m}, N:{n}, K:{k}")
        best_time = 'nan'
    else:
        best_time = round(best_time, 4)

        print(f"Tuning result for B:{b}, M:{m}, N:{n}, K:{k} is kernelId={best_kernelId} {kernels_list[best_kernelId].name} {splitK=}, {best_time}us")
    print(f"*******************B:{b} X M:{m} X N:{n} X K{k}**************************")

    return best_kernelId, splitK, best_time


def tune_batched_gemm_list(untunedf, tunedf, issorted = False, useSplitK = False):
    for i in range(len(untunedf)):
        B = untunedf.loc[i, "B"]
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]

        if tunedf[(tunedf["B"]==B) & (tunedf["M"]==M) & (tunedf["N"]==N) & (tunedf["K"]==K)].empty:
            kernelId, splitK, time = tune_batched_gemm(B, M, N, K, useSplitK)
            kernelName = 'None' if kernelId == -1 else kernels_list[kernelId].name
            temp = pd.DataFrame({"B":[B], "M":[M], "N":[N], "K":[K], "kernelId":[kernelId], "splitK":[splitK],
                           "us":[time], "kernelName":[kernelName]})
            tunedf = pd.concat([tunedf, temp], ignore_index=True)

        else:
            print(f"B:{B}, M:{M}, N:{N}, K{K} is in tuned batched_gemm, skip!!!")
        print()
        print()
    if issorted:
        tunedf = tunedf.sort_values(by=["B", "M", "N", "K"])
    print("Totall tuning result:")
    print(tunedf)
    return tunedf



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK batched_gemm a8w8 kernel",
    )

    parser.add_argument(
        "-i",
        "--untune_file",
        default="aiter/configs/a8w8_untuned_batched_gemm.csv",
        required=False,
        help="input"
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_batched_gemm.csv",
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
        help="Arranged according to the B M N K size"
    )

    args = parser.parse_args()
    untunedf = get_untuned_batched_gemm_list(args.untune_file)
    tunedf = get_tuned_batched_gemm_list(args.tune_file)
    tunedf = tune_batched_gemm_list(untunedf, tunedf, args.sort, args.splitK)
    tunedf.to_csv(args.tune_file, index=False)
