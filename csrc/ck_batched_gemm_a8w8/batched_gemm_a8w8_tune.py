# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_A8W8_BATCHED_GEMM
from aiter.utility.base_tuner import GemmCommonTuner
from batched_gemm_a8w8_common import kernels_list
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


def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.batched_gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out


def generate_data(b, m, n, k, device="cuda"):
    x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.i8, device=device)
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.i8, device=device)
    x_scale = torch.rand([b, m, 1], dtype=dtypes.bf16, device=device)
    w_scale = torch.rand([b, 1, n], dtype=dtypes.bf16, device=device)
    out = torch.empty(b, m, n, dtype=dtypes.bf16, device=device)
    # index of data [0, 1, 2, 3, 4]
    return x, weight, x_scale, w_scale, out


class BatchedGemma8W8Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_A8W8_BATCHED_GEMM}",
        "untune_file": "aiter/configs/a8w8_untuned_batched_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
    }

    def _setup_specific_arguments(self):
        pass

    def calculate(self, results, bpes=(1, 1, 2)):
        info, time, err_ratio = results
        if time == -1:
            return -1, -1
        print(info[0])
        cu_num, b, m, n, k = info[0]
        flops = m * n * k * 2 * b
        tflops = round(flops / (time * 1000000), 2)
        lhs_bpe, rhs_bpe, out_bpe = bpes
        bw = round(
            b
            * (m * k * lhs_bpe + n * k * rhs_bpe + m * n * out_bpe)
            / (time * 1e-6)
            / 1e9,
            2,
        )
        return tflops, bw

    def getKernelName(self, kernelId):
        if kernelId >= len(kernels_list) or kernelId < 0:
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
        shape_grouped = False
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        task = []
        tasks_data = []
        for i in range(len(untunedf)):
            B = untunedf.loc[i, "B"]
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]

            kernels_num = len(kernels_list)

            print(
                f"******************tune B:{B} X M:{M} X N:{N} X K{K}*******************"
            )
            # kernelId, splitK, time = tune_batched_gemm(B, M, N, K, useSplitK)
            total_kernel_nums = 0
            for i in range(kernels_num):
                kernel = kernels_list[i]
                maxsplitK = (
                    aiter.compute_batched_gemm_SplitK(
                        B,
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
                    info = ((cu_num, B, M, N, K), i, splitK, "")
                    task.append(
                        (
                            info,
                            generate_data,
                            (B, M, N, K),
                            kernel_instance_test,
                            (
                                [0, 1, 2, 3, 4],
                                i,
                                splitK,
                            ),  # [0, 1, 2, 3, 4] is index of paramters for kernel_instance_test in generate_data
                            {
                                "num_warmup": args.warmup,
                                "num_iters": args.iters,
                            },
                            run_torch,
                            ([0, 1, 2, 3],),
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
            shape_grouped = False
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

    key = [
        "cu_num",
        "B",
        "M",
        "N",
        "K",
    ]
    resultList = [
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "errRatio",
        "tflops",
        "bw",
    ]

    tuner = BatchedGemma8W8Tuner(
        "BatchGemmA8W8Tuner",
        key,
        resultList,
        "gen API for CK batch gemm a8w8 kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
