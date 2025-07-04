# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.utility import fp4_utils
from aiter.test_common import perftest
from aiter.ops.shuffle import shuffle_weight
from gemm_a4w4_blockscale_common import kernels_list
import argparse
from aiter.utility.mp_tuner import mp_tuner

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
torch.random.manual_seed(0)
SCALE_GROUP_SIZE = 32
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


def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


def get_untuned_gemm_list(untuned_gemm_file):
    assert os.path.exists(
        untuned_gemm_file
    ), f"Not exist a4w4_untuned_gemm.csv file: {untuned_gemm_file}"
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
    aiter.gemm_a4w4_blockscale_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out


def run_gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out, kernel_id, splitK):
    m, k = x.shape
    n, k = weight.shape
    aiter.gemm_a4w4_blockscale_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out[:m]


def generate_data(m, n, k, useSplitK=False, dtype=dtypes.fp16):
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((m, k), dtype=dtype)
    w = torch.randn((n, k), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    w_shuffle = shuffle_weight(w)
    out_ck = torch.empty((m + 255) // 256 * 256, n, dtype=dtype)
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    return (
        x,
        w,
        x_scales,
        w_scales,
        w_shuffle,
        x_scales_shuffle,
        w_scales_shuffle,
        out_ck,
    )


def tune_gemm_list(
    untunedf, tunedf, issorted=False, useSplitK=False, mp_num=1, shape_grouped=False
):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ["gfx950"]:
        print(f"tuning is not supported in this chip {get_gfx()}")
        return tunedf
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    task = []
    tasks_in_data = []
    kernels_num = len(kernels_list)
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
            input_data = generate_data(M, N, K)
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
                            run_gemm_a4w4_blockscale,
                            (
                                input_data[0],
                                input_data[4],
                                input_data[5],
                                input_data[6],
                                input_data[7],
                                i,
                                splitK,
                            ),
                            {},
                            run_torch,
                            (
                                input_data[0],
                                input_data[1],
                                input_data[2],
                                input_data[3],
                                dtypes.fp16,
                            ),
                            {},
                            None,
                            1e-2,
                            0.1,
                        )
                    )
                    total_kernel_nums = total_kernel_nums + 1

            tasks_in_data.append((total_kernel_nums, ()))
    if task:
        ret = mp_tuner(task, tasks_in_data, mp_num, False, shape_grouped)
        for el in ret:
            info, time, err_ratio = el
            (cu_num, M, N, K), kernelId, splitK = info
            kernelName = (
                "None"
                if kernelId == -1 or time == "nan"
                else kernels_list[kernelId].name
            )
            print(
                f"Tuning result for M:{M}, N:{N}, K:{K}, cu_num:{cu_num} is kernelId={kernelId} {kernels_list[kernelId].name} {splitK=}, {time}us"
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
        tunedf = tunedf.sort_values(by=["cu_num", "M", "N", "K"])
    print("Totall tuning result:")
    print(tunedf)
    return tunedf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm a4w4 kernel",
    )

    parser.add_argument(
        "-i",
        "--untune_file",
        default="aiter/configs/a4w4_blockscale_untuned_gemm.csv",
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
        default="aiter/configs/a4w4_blockscale_tuned_gemm.csv",
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
