# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.test_common import perftest
from gemm_a8w8_blockscale_common import kernels_list
import argparse
from einops import rearrange

block_shape = (128, 128)


def checkClose(a, b, rtol=1e-3, atol=0.01):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        return True
    else:
        percent = (a[mask]).numel() / a.numel()
        if percent > 0.01:
            return False
        else:
            return True


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    # x_scale = rearrange(x_scale.view(-1, 1).repeat(1, block_shape_n*block_shape_k).view(m, scale_k, 1, block_shape_k),
    #                           'num_blk_n num_blk_k blk_n blk_k ->(num_blk_n blk_n) (num_blk_k blk_k)')
    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
    # scale = torch.matmul(x_scale, w_scale)
    # out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def get_untuned_gemm_list(untuned_gemm_file):
    assert os.path.exists(
        untuned_gemm_file
    ), f"Not exist a8w8_untuned_gemm.csv file: {untuned_gemm_file}"
    untunedf = pd.read_csv(untuned_gemm_file)
    return untunedf


def get_tuned_gemm_list(tuned_gemm_file):
    if os.path.exists(tuned_gemm_file):
        tunedf = pd.read_csv(tuned_gemm_file)
    else:
        tunedf = pd.DataFrame(
            columns=["cu_num", "M", "N", "K", "kernelId", "splitK", "us", "kernelName"]
        )
    return tunedf


@perftest()
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a8w8_blockscale_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out


def tune_gemm(m, n, k, useSplitK=False):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
    out = torch.empty(m, n, dtype=dtypes.bf16, device="cuda")

    ref_out = run_torch(x, weight, x_scale, w_scale)

    print(f"*******************M:{m} X N:{n} X K:{k}**************************")
    print(f"Start tuning a8w8 gemm kernel for M:{m}, N:{n}, K{k}:")
    kernels_num = len(kernels_list)
    best_kernelConfig = (-1, 0)
    best_time = -1
    for i in range(kernels_num):
        kernel = kernels_list[i]
        maxsplitK = (
            aiter.compute_gemm_SplitK(
                m, n, k, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK
            )
            if useSplitK
            else 0
        )
        for splitK in range(maxsplitK + 1):
            try:
                (out), avg_t = kernel_instance_test(
                    x, weight, x_scale, w_scale, out, i, splitK
                )
                isClosed = checkClose(ref_out, out, rtol=1e-2, atol=0.1)
                if isClosed:
                    print(
                        f"{str(dim):<20} kernelid:{i:<3d}\t avg: {avg_t:<8.2f} us, {kernel.name}, {splitK=}"
                    )
                    if best_time < 0 or avg_t < best_time:
                        best_kernelConfig = (i, splitK)
                        best_time = avg_t
                else:
                    print(
                        f"{str(dim):<20} kernelid:{i:<3d}\t No pass         , {kernel.name}, {splitK=}"
                    )
            except RuntimeError as e:
                print(e)
                print(
                    f"{str(dim):<20} kernelid:{i:<3d}\t No support      , {kernel.name}, {splitK=}"
                )

    best_kernelId, splitK = best_kernelConfig
    if best_kernelConfig[0] == -1:
        print(f"No kernel can be used for M:{m}, N:{n}, K:{k}")
        best_time = "nan"
    else:
        best_time = round(best_time, 4)

        print(
            f"Tuning result for M:{m}, N:{n}, K:{k} is kernelId={best_kernelId} {kernels_list[best_kernelId].name} {splitK=}, {best_time}us"
        )
    print(f"*******************M:{m} X N:{n} X K{k}**************************")

    return best_kernelId, splitK, best_time


def tune_gemm_list(untunedf, tunedf, issorted=False, useSplitK=False):
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    for i in range(len(untunedf)):
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]

        if tunedf[
            (tunedf["M"] == M)
            & (tunedf["N"] == N)
            & (tunedf["K"] == K)
            & (tunedf["cu_num"] == cu_num)
        ].empty:
            kernelId, splitK, time = tune_gemm(M, N, K, useSplitK)
            kernelName = "None" if kernelId == -1 else kernels_list[kernelId].name
            temp = pd.DataFrame(
                {
                    "M": [M],
                    "N": [N],
                    "K": [K],
                    "cu_num": [cu_num],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                }
            )
            tunedf = pd.concat([tunedf, temp], ignore_index=True)

        else:
            print(f"M:{M}, N:{N}, K{K} is in tuned gemm, skip!!!")
        print()
        print()
    if issorted:
        tunedf = tunedf.sort_values(by=["cu_num", "M", "N", "K"])
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
        default="aiter/configs/a8w8_blockscale_untuned_gemm.csv",
        required=False,
        help="input",
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/a8w8_blockscale_tuned_gemm.csv",
        required=False,
        help="output: tuning result store this file",
    )

    parser.add_argument(
        "-k", "--splitK", action="store_true", required=False, help="Use splitK kernels"
    )

    parser.add_argument(
        "--sort",
        action="store_true",
        required=False,
        help="Arranged according to the M N K size",
    )

    args = parser.parse_args()
    untunedf = get_untuned_gemm_list(args.untune_file)
    tunedf = get_tuned_gemm_list(args.tune_file)
    tunedf = tune_gemm_list(untunedf, tunedf, args.sort, args.splitK)
    tunedf.to_csv(args.tune_file, index=False)
