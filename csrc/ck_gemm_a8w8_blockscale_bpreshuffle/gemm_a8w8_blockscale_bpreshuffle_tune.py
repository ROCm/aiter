# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from gemm_a8w8_blockscale_bpreshuffle_common import kernels_list

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import perftest
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

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
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest()
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a8w8_blockscale_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def run_gemm_a8w8_blockscale_bpreshuffle(
    x, weight, x_scale, w_scale, out, kernel_id, splitK
):
    aiter.gemm_a8w8_blockscale_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def generate_data(m, n, k, seed, device="cuda"):
    torch.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    return x, weight_shuffle, x_scale_t, w_scale, out, weight, x_scale


class GemmA8W8BlockscaleBPreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": "aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        "untune_file": "aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    }

    def _setup_specific_arguments(self):
        pass

    def calculate(self, results, bpes=(1, 1, 2)):
        ## bpes = (inbpe, w_bpe, outbpe)
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId):
        if kernelId < 0 or kernelId >= len(kernels_list):
            return None
        return kernels_list[kernelId].name

    def tune(self, untunedf, tunedf, args):
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
            ref_data_idx = [0, 5, 6, 3]

            if tunedf[
                (tunedf["M"] == M)
                & (tunedf["N"] == N)
                & (tunedf["K"] == K)
                & (tunedf["cu_num"] == cu_num)
            ].empty:
                seed = seed + 1
                total_kernel_nums = 0
                for j in range(kernels_num):  # Use 'j' instead of 'i' to avoid conflict
                    kernel = kernels_list[j]
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
                        info = ((cu_num, M, N, K), j, splitK, "")
                        task.append(
                            (
                                info,
                                generate_data,
                                (M, N, K, seed),
                                run_gemm_a8w8_blockscale_bpreshuffle,
                                (
                                    gemm_a8w8_idx,
                                    j,
                                    splitK,
                                ),
                                {},
                                run_torch,
                                (
                                    ref_data_idx,
                                    None,
                                    dtypes.bf16,
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
                print(f"M:{M}, N:{N}, K:{K} is in tuned gemm, skip!!!")
                print()
                print()

        ret = []
        if task:
            ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped, errRatio)

        return ret


if __name__ == "__main__":
    ## use default key and resultList
    tuner = GemmA8W8BlockscaleBPreShuffleTuner(
        "GemmA8W8BlockscaleBPreShuffleTuner",
        description="gen API for CK gemm a8w8 blockscale bpreshuffle kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
