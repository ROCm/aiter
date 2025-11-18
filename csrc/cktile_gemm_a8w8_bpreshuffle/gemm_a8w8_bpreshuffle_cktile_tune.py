# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_CKTILE
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.ops.shuffle import shuffle_weight
from gemm_a8w8_bpreshuffle_cktile_common import kernels_list
import argparse
from aiter.utility.mp_tuner import mp_tuner


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_gemm_a8w8_bpreshuffle_cktile(
    x, weight, x_scale, w_scale, out, kernel_id, splitK=0
):
    aiter.gemm_a8w8_bpreshuffle_cktile_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def generate_data(
    m, n, k, seed, dtype=dtypes.bf16, q_dtype_w=dtypes.fp8, device="cuda"
):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=dtype, device=device)
    weight = torch.randn((n, k), dtype=dtype, device=device)
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=q_dtype_w)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=q_dtype_w)
    bias_f32 = None
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtype, device=device)
    return x, weight_shuffle, x_scale, w_scale, out, weight, bias_f32


class GemmA8W8BpreShuffleCktileTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_CKTILE}",
        "untune_file": "aiter/configs/a8w8_bpreshuffle_cktile_untuned_gemm.csv",
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

    def get_cktile_gemm_a8w8_bpreshuffle_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
    ):
        (cu_num, M, N, K, q_dtype_w) = info_keys
        if eval(q_dtype_w) != dtypes.fp8:
            print(
                f"Warning: q_dtype_w only support {dtypes.fp8}, actual q_dtype_w is {q_dtype_w}!"
            )
            return []
        kernels_num = len(kernels_list)
        gemm_a8w8_idx = [0, 1, 2, 3, 4]  # input index in generate_data
        ref_data_idx = [0, 5, 2, 3, 6]
        tasks_ck = []
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
                info = (info_keys, i, splitK, "")
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w)),
                        run_gemm_a8w8_bpreshuffle_cktile,
                        (
                            gemm_a8w8_idx,
                            i,
                            splitK,
                        ),
                        {},
                        run_torch,
                        (
                            ref_data_idx,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                    )
                )
        return tasks_ck

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        issorted = args.sort
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = False
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        seed = 10000
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            q_dtype_w = untunedf.loc[i, "q_dtype_w"]
            seed = seed + 1
            total_kernel_nums = 0
            kernels_num = len(kernels_list)
            info_keys = (cu_num, M, N, K, q_dtype_w)
            task.extend(
                self.get_cktile_gemm_a8w8_bpreshuffle_tune_task(
                    info_keys,
                    useSplitK,
                    seed,
                )
            )

            total_kernel_nums = len(task)

            tasks_data.append((total_kernel_nums, ()))
        ret = []
        if task:
            ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped, errRatio)

        return ret


if __name__ == "__main__":
    ## use default key and resultList
    key = ["cu_num", "M", "N", "K", "q_dtype_w"]
    tuner = GemmA8W8BpreShuffleCktileTuner(
        "GemmA8W8BpreShuffleCktileTuner",
        key=key,
        description="gen API for gemm a8w8 bpreshuffle cktile kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
