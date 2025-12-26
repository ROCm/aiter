# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8
from aiter.utility.base_tuner import GemmCommonTuner
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
            columns=["cu_num", "M", "N", "K", "kernelId", "splitK", "us", "kernelName"]
        )
    return tunedf


def generate_data(m, n, k, seed, device="cuda"):
    torch.manual_seed(seed)
    x = torch.randint(-20, 20, (m, k), dtype=dtypes.i8, device=device)
    weight = torch.randint(-20, 20, (n, k), dtype=dtypes.i8, device=device)
    x_scale = torch.rand([m, 1], dtype=dtypes.bf16, device=device)
    w_scale = torch.rand([1, n], dtype=dtypes.bf16, device=device)
    out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
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


class GemmA8W8Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8}",
        "untune_file": "aiter/configs/a8w8_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
    }

    def getKernelName(self, kernelId):
        if kernelId >= len(kernels_list) or kernelId < 0:
            return None
        return kernels_list[kernelId].name

    def _setup_specific_arguments(self):
        # self.parser.add_argument()
        pass

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=(1, 1, 2))

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
        tasks_data = []
        gemm_a8w8_data_idx = [0, 1, 2, 3, 4]  # input index in generate_data
        ref_data_idx = [0, 1, 2, 3]
        seed = 0
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            kernels_num = len(kernels_list)
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
                            run_gemm_a8w8,
                            (gemm_a8w8_data_idx, i, splitK),
                            {
                                "num_warmup": args.warmup,
                                "num_iters": args.iters,
                            },
                            gemm_a8w8_ref,
                            (ref_data_idx,),
                            {},
                            None,
                            1e-2,
                            1e-2,
                        )
                    )
                    total_kernel_nums = total_kernel_nums + 1

            tasks_data.append((total_kernel_nums, ()))

        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_data,
                mp_num,
                False,
                shape_grouped,
                errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )
        return ret


if __name__ == "__main__":

    ## tuner =GemmA8W8Tuner("GemmA8W8Tuner", key, resultList,"gen API for CK gemm a8w8 kernel")
    ## use default key and resultList
    tuner = GemmA8W8Tuner(
        "GemmA8W8Tuner",  # key, resultList,
        description="gen API for CK gemm a8w8 kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
