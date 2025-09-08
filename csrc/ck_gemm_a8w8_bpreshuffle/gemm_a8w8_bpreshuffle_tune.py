# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.test_common import perftest
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.ops.shuffle import shuffle_weight
from gemm_a8w8_bpreshuffle_common import kernels_list
import argparse
from aiter.utility.mp_tuner import mp_tuner


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


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest()
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a8w8_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def run_gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a8w8_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def generate_data(m, n, k, seed, dtype=dtypes.fp16, device="cuda"):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=dtype, device=device)
    weight = torch.randn((n, k), dtype=dtype, device=device)
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=dtypes.fp8)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtype, device=device)
    return x, weight_shuffle, x_scale, w_scale, out, weight


class Gemma8W8BPreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": "aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv",
        "untune_file": "aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv",
    }

    def _setup_specific_arguments(self):
        pass

    def calculate(self, results, bpes=(1, 1, 2)):
        ## bpes = (inbpe, w_bpe, outbpe)
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId):
        if kernelId < 0 or kernelId > len(kernels_list):
            return None
        return kernels_list[kernelId].name

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        issorted = args.sort
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = True
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        seed = 10000
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            kernels_num = len(kernels_list)
            gemm_a8w8_idx = [0, 1, 2, 3, 4]  # input index in generate_data
            ref_data_idx = [0, 5, 2, 3]
            if tunedf[
                (tunedf["M"] == M)
                & (tunedf["N"] == N)
                & (tunedf["K"] == K)
                & (tunedf["cu_num"] == cu_num)
            ].empty:
                seed = seed + 1
                total_kernel_nums = 0
                for i in range(kernels_num):
                    kernel = kernels_list[i]
                    maxsplitK = (
                        aiter.compute_gemm_SplitK(
                            M,
                            N,
                            K,
                            kernel.MPerBLOCK,
                            kernel.NPerBLOCK,
                            kernel.KPerBLOCK,
                        )
                        if useSplitK
                        else 0
                    )
                    for splitK in range(maxsplitK + 1):
                        info = ((cu_num, M, N, K), i, splitK, "")
                        task.append(
                            (
                                info,
                                generate_data,
                                (M, N, K, seed),
                                run_gemm_a8w8_bpreshuffle,
                                (
                                    gemm_a8w8_idx,
                                    i,
                                    splitK,
                                ),
                                {},
                                run_torch,
                                (
                                    ref_data_idx,
                                    None,
                                    dtypes.fp16,
                                ),
                                {},
                                None,
                                1e-2,
                                0.1,
                            )
                        )
                        total_kernel_nums = total_kernel_nums + 1

                tasks_data.append((total_kernel_nums, ()))
            else:
                print(f"M:{M}, N:{N}, K{K} is in tuned gemm, skip!!!")
                print()
                print()
        ret = []
        if task:
            ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped, errRatio)

        return ret


if __name__ == "__main__":
    ## use default key and resultList
    tuner = Gemma8W8BPreShuffleTuner(
        "Gemma8W8BPreShuffleTuner",
        description="gen API for CK gemm a8w8 bpreshuffle kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
