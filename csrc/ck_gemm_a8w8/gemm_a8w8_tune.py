# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.test_common import perftest
from gemm_a8w8_common import kernels_list
from aiter.utility.mp_tuner import mp_tuner
import argparse


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
    x = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def get_untuned_gemm_list(untuned_gemm_file):
    assert os.path.exists(
        untuned_gemm_file
    ), f"Not exist a8w8_untuned_gemm.csv file: {untuned_gemm_file}"
    untunedf = pd.read_csv(untuned_gemm_file)
    filtered_df = untunedf.drop_duplicates().reset_index(drop=True)
    return filtered_df


def get_tuned_gemm_list(tuned_gemm_file):
    if os.path.exists(tuned_gemm_file):
        tunedf = pd.read_csv(tuned_gemm_file)
    else:
        tunedf = pd.DataFrame(
            columns=["M", "N", "K", "kernelId", "splitK", "us", "kernelName"]
        )
    return tunedf


def generate_data(m, n, k):
    x = torch.randint(-20, 20, (m, k), dtype=dtypes.i8, device="cuda")
    weight = torch.randint(-20, 20, (n, k), dtype=dtypes.i8, device="cuda")
    x_scale = torch.rand([m, 1], dtype=dtypes.bf16, device="cuda")
    w_scale = torch.rand([1, n], dtype=dtypes.bf16, device="cuda")
    out = torch.empty(m, n, dtype=dtypes.bf16, device="cuda")
    # x.share_memory_()
    # weight.share_memory_()
    # x_scale.share_memory_()
    # w_scale.share_memory_()
    return x, weight, x_scale, w_scale, out


def gemm_a8w8_ref(x, weight, x_scale, w_scale):
    return run_torch(x, weight, x_scale, w_scale)


def run_gemm_a8w8(x, weight, x_scale, w_scale, out, kernelId, splitK):

    aiter.gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernelId, splitK)
    return out


def tune_gemm_list(
    untunedf, tunedf, issorted=False, useSplitK=False, mp_num=1, shape_grouped=False
):
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    print(f"cu_num:{cu_num}")
    task = []
    tasks_data = []
    for i in range(len(untunedf)):
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]
        kernels_num = len(kernels_list)

        if tunedf[
            (tunedf["M"] == M)
            & (tunedf["N"] == N)
            & (tunedf["K"] == K)
            & (tunedf["cu_num"] == cu_num)
        ].empty:
            input_datas = generate_data(M, N, K)
            total_kernel_nums = 0
            for i in range(kernels_num):
                kernel = kernels_list[i]
                maxsplitK = (
                    aiter.compute_gemm_SplitK(
                        M, N, K, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK
                    )
                    if useSplitK
                    else 0
                )
                for splitK in range(maxsplitK + 1):
                    info = ((cu_num, M, N, K), i, splitK)
                    task.append(
                        (
                            info,
                            run_gemm_a8w8,
                            (i, splitK),
                            {},
                            gemm_a8w8_ref,
                            (),
                            {},
                            None,
                            1e-2,
                            1e-2,
                        )
                    )
                total_kernel_nums = total_kernel_nums + 1

            tasks_data.append((total_kernel_nums, input_datas))
    if task:
        ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped)
        for el in ret:
            info, time, err_ratio = el
            (cu_num, M, N, K), kernelId, splitK = info
            kernelName = (
                "None"
                if kernelId == -1 or time == "nan"
                else kernels_list[kernelId].name
            )
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
        help="input",
    )

    parser.add_argument(
        "--mp",
        type=int,
        default=torch.cuda.device_count(),
        help="Tuning on multiple GPUs using multiple processes",
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_gemm.csv",
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
    tunedf = tune_gemm_list(untunedf, tunedf, args.sort, args.splitK, args.mp)
    tunedf.to_csv(args.tune_file, index=False)
