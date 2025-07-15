# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.test_common import perftest
from batched_gemm_a8w8_common import kernels_list
import argparse
from aiter.utility.mp_tuner import mp_tuner
import time

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
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=dtypes.bf16, device="cuda")
    for b in range(B):
        b_x = F.linear(x[b, :, :].to(dtypes.fp32), weight[b, :, :].to(dtypes.fp32))
        b_scale = torch.matmul(x_scale[b, :, :], w_scale[b, :, :])
        b_out = torch.mul(b_x, b_scale)
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


def get_untuned_batched_gemm_list(untuned_batched_gemm_file):
    assert os.path.exists(
        untuned_batched_gemm_file
    ), f"Not exist a8w8_untuned_batched_gemm.csv file: {untuned_batched_gemm_file}"
    untunedf = pd.read_csv(untuned_batched_gemm_file)
    return untunedf


def get_tuned_batched_gemm_list(tuned_batched_gemm_file):
    if os.path.exists(tuned_batched_gemm_file):
        tunedf = pd.read_csv(tuned_batched_gemm_file)
    else:
        tunedf = pd.DataFrame(
            columns=["B", "M", "N", "K", "kernelId", "splitK", "us", "kernelName"]
        )
    return tunedf


def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.batched_gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out


def generate_data(b, m, n, k):
    x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.i8, device="cuda")
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.i8, device="cuda")
    x_scale = torch.rand([b, m, 1], dtype=dtypes.bf16, device="cuda")
    w_scale = torch.rand([b, 1, n], dtype=dtypes.bf16, device="cuda")
    out = torch.empty(b, m, n, dtype=dtypes.bf16, device="cuda")
    return x, weight, x_scale, w_scale, out


def tune_batched_gemm_list(untunedf, tunedf, issorted=False, useSplitK=False, mp_num=1):
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    task = []
    tasks_data = []
    for i in range(len(untunedf)):
        B = untunedf.loc[i, "B"]
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]

        if tunedf[
            (tunedf["B"] == B)
            & (tunedf["M"] == M)
            & (tunedf["N"] == N)
            & (tunedf["K"] == K)
            & (tunedf["cu_num"] == cu_num)
        ].empty:
            kernels_num = len(kernels_list)
            input_datas = generate_data(B, M, N, K)
            print(
                f"*******************B:{b} X M:{m} X N:{n} X K{k}**************************"
            )
            # kernelId, splitK, time = tune_batched_gemm(B, M, N, K, useSplitK)
            total_kernel_nums = 0
            for i in range(kernels_num):
                kernel = kernels_list[i]
                maxsplitK = (
                    aiter.compute_batched_gemm_SplitK(
                        B, M, N, K, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK
                    )
                    if useSplitK
                    else 0
                )
                for splitK in range(maxsplitK + 1):
                    info = ((cu_num, B, M, N, K), i, splitK)
                    task.append(
                        (
                            info,
                            kernel_instance_test,
                            (i, splitK),
                            {},
                            run_torch,
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
        shape_grouped = False
        ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped)
        for el in ret:
            info, time, err_ratio = el
            (cu_num, B, M, N, K), kernelId, splitK = info
            kernelName = (
                "None"
                if kernelId == -1 or time == "nan"
                else kernels_list[kernelId].name
            )
            temp = pd.DataFrame(
                {
                    "cu_num": [cu_num],
                    "B": [B],
                    "M": [M],
                    "N": [N],
                    "K": [K],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                }
            )
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
        help="input",
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_batched_gemm.csv",
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
        help="Arranged according to the B M N K size",
    )

    parser.add_argument(
        "--mp",
        type=int,
        default=torch.cuda.device_count(),
        help="Tuning on multiple GPUs using multiple processes",
    )

    args = parser.parse_args()
    untunedf = get_untuned_batched_gemm_list(args.untune_file)
    tunedf = get_tuned_batched_gemm_list(args.tune_file)
    start = time.time()
    tunedf = tune_batched_gemm_list(untunedf, tunedf, args.sort, args.splitK, args.mp)
    print("Tuning time: ", time.time() - start)
    tunedf.to_csv(args.tune_file, index=False)
