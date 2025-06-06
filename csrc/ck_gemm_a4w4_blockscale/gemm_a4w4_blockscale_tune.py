# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.utility import fp4_utils
from aiter.test_common import checkAllclose, perftest
from aiter.ops.shuffle import shuffle_weight
from gemm_a4w4_blockscale_common import kernels_list
import argparse
from einops import rearrange

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


def tune_gemm(m, n, k, useSplitK=False, dtype=dtypes.fp16):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ["gfx950"]:
        return

    dim = (m, n, k)
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

    ref_out = run_torch(x, w, x_scales, w_scales, dtype)

    print(f"*******************M:{m} X N:{n} X K:{k}**************************")
    print(f"Start tuning a4w4 gemm kernel for M:{m}, N:{n}, K{k}:")
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
                (out_ck), avg_t = kernel_instance_test(
                    x, w_shuffle, x_scales_shuffle, w_scales_shuffle, out_ck, i, splitK
                )
                isClosed = checkClose(ref_out, out_ck[:m], rtol=1e-2, atol=0.1)
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
    tunedf = tune_gemm_list(untunedf, tunedf, args.sort, args.splitK)
    tunedf.to_csv(args.tune_file, index=False)
