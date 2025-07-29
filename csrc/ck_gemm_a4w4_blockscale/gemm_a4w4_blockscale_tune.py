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
from aiter.jit.core import get_asm_dir

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
    filtered_df = untunedf.drop_duplicates().reset_index(drop=True)
    return filtered_df


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
    res = aiter.gemm_a4w4_blockscale_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return res[:m]


def run_gemm_a4w4_blockscale_asm(
    x,
    weight_shuffle,
    x_scale,
    w_scale,
    out,
    kernelName,
    bias,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    splitK=None,
):
    m, k = x.shape
    if splitK != 0:
        out_reset = torch.zeros(out.shape[0], out.shape[1], dtype=dtype)
        out = out_reset
    res = aiter.gemm_a4w4_asm(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=splitK,
    )
    return res[:m]


def generate_data(m, n, k, useSplitK=False, dtype=dtypes.bf16):
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
    bias_f32 = None
    return (
        x,
        w,
        x_scales,
        w_scales,
        w_shuffle,
        x_scales_shuffle,
        w_scales_shuffle,
        out_ck,
        bias_f32,
    )


def get_asm_kernels(file):
    if not os.path.exists(file):
        print(f"ASM kernel list file not exist: {file}")
        return {}
    df = pd.read_csv(file)
    shuffle_df = (
        df[df["bpreshuffle"] == 1]
        .reset_index()
        .sort_values(by=["tile_m", "tile_n", "splitK"])
    )
    kernel_dict = (
        shuffle_df.groupby(["tile_m", "tile_n"])["knl_name"].apply(list).to_dict()
    )
    return kernel_dict


def tune_gemm_list(
    untunedf,
    tunedf,
    issorted=False,
    useSplitK=False,
    mp_num=1,
    shape_grouped=False,
    errRatio=0.05,
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
    ck_kernels_num = len(kernels_list)
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
            for i in range(ck_kernels_num):
                kernel = kernels_list[i]
                maxsplitK = (
                    aiter.compute_gemm_SplitK(
                        M, N, K, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK
                    )
                    if useSplitK
                    else 0
                )
                for splitK in range(maxsplitK + 1):
                    info = ((cu_num, M, N, K), i, splitK, "")
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
                                dtypes.bf16,
                            ),
                            {},
                            None,
                            1e-2,
                            0.01,
                        )
                    )
                    total_kernel_nums = total_kernel_nums + 1
            ### asm kernels
            asm_kernels_id = ck_kernels_num + 1
            asm_kernel_list_csv = f"{get_asm_dir()}/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
            asm_kernels = get_asm_kernels(asm_kernel_list_csv)
            #asm_tiles = [(256, 256), (128, 512)]
            asm_tiles = [key for key in asm_kernels.keys()]
            for tile_m, tile_n in asm_tiles:
                maxsplitK = (
                    aiter.compute_gemm_SplitK(M, N, K, tile_m, tile_n, 256)
                    if useSplitK
                    else 0
                )
                kernelName = asm_kernels.get((tile_m, tile_n), [])
                if len(kernelName) == 0:
                    print(f"no kernel name for ({tile_m}, {tile_n})!!!!")
                    continue
                if len(kernelName) == 1:
                    maxsplitK = 0
                for splitK in range(maxsplitK + 1):
                    kernel_name = kernelName[0] if splitK == 0 else kernelName[1]
                    info = ((cu_num, M, N, K), asm_kernels_id, splitK, kernel_name)
                    task.append(
                        (
                            info,
                            run_gemm_a4w4_blockscale_asm,
                            (
                                input_data[0],
                                input_data[4],
                                input_data[5],
                                input_data[6],
                                input_data[7],
                                kernel_name,
                                input_data[8],
                                dtypes.bf16,
                                True,
                                splitK,
                            ),
                            {},
                            run_torch,
                            (
                                input_data[0],
                                input_data[1],
                                input_data[2],
                                input_data[3],
                                dtypes.bf16,
                            ),
                            {},
                            None,
                            1e-2,
                            0.01,
                        )
                    )
                    asm_kernels_id = asm_kernels_id + 1
                    total_kernel_nums = total_kernel_nums + 1
            tasks_in_data.append((total_kernel_nums, ()))
        else:
            print(f"M:{M}, N:{N}, K{K} is in tuned gemm, skip!!!")
            print()
            print()
    if task:
        ret = mp_tuner(task, tasks_in_data, mp_num, False, shape_grouped, errRatio)
        for el in ret:
            info, time, err_ratio = el
            (cu_num, M, N, K), kernelId, splitK, kernel_name = info
            if kernelId < ck_kernels_num:
                kernelName = (
                    "None"
                    if kernelId == -1 or time == "nan" or kernelId > ck_kernels_num
                    else kernels_list[kernelId].name
                )

                print(
                    f"Tuning result for M:{M}, N:{N}, K:{K}, cu_num:{cu_num} is kernelId={kernelId} {kernels_list[kernelId].name} {splitK=}, {time}us"
                )
            else:
                kernelName = kernel_name
                print(
                    f"Tuning result for M:{M}, N:{N}, K:{K}, cu_num:{cu_num} is {kernelName} {splitK=}, {time}us"
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
        default=1,  # torch.cuda.device_count(),
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
    parser.add_argument(
        "--errRatio",
        type=float,
        default=0.05,
        help="tolerable error ratio, default 0.05.",
    )

    args = parser.parse_args()
    untunedf = get_untuned_gemm_list(args.untune_file)
    tunedf = get_tuned_gemm_list(args.tune_file)
    tunedf = tune_gemm_list(
        untunedf, tunedf, args.sort, args.splitK, args.mp, errRatio=args.errRatio
    )
    tunedf.to_csv(args.tune_file, index=False)
