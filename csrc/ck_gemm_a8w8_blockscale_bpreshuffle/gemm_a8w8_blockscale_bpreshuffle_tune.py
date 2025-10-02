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
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

block_shape = (128, 128)


class Gemma8W8BlockScaleBPreShuffleTuner(GemmCommonTuner):
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
        if kernelId < 0 or kernelId > len(kernels_list):
            return None
        return kernels_list[kernelId].name

    @staticmethod
    def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
        block_shape_n, block_shape_k = block_shape
        m, k = x.shape
        n = weight.shape[0]
        scale_n = (n + block_shape_n - 1) // block_shape_n
        scale_k = (k + block_shape_k - 1) // block_shape_k
        # x_scale = rearrange(x_scale.view(-1, 1).repeat(1, block_shape_n*block_shape_k).view(m, scale_k, 1, block_shape_k),
        #                           'num_blk_n num_blk_k blk_n blk_k ->(num_blk_n blk_n) (num_blk_k blk_k)')
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
        # scale = torch.matmul(x_scale, w_scale)
        # out = torch.mul(x, scale)
        if bias is not None:
            out = out.to(bias) + bias
        return out.to(dtype)

    @staticmethod
    def run_gemm_a8w8_blockscale_bpreshuffle(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    ):
        aiter.gemm_a8w8_blockscale_bpreshuffle_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )
        return out

    @staticmethod
    def generate_data(m, n, k, seed, device="cuda"):
        torch.manual_seed(seed)
        block_shape_n, block_shape_k = block_shape
        scale_n = (n + block_shape_n - 1) // block_shape_n
        scale_k = (k + block_shape_k - 1) // block_shape_k
        x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
        weight = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(
            dtypes.fp8
        )
        x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device=device)
        w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
        weight_shuffle = shuffle_weight(weight, layout=(16, 16))
        out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
        x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
        return x, weight_shuffle, x_scale_t, w_scale, out, weight, x_scale

    def get_ck_tasks(self, info_keys, useSplitK, seed):
        cu_num, M, N, K = info_keys
        kernels_num = len(kernels_list)
        gemm_a8w8_data_idx = [0, 1, 2, 3, 4]
        ref_data_idx = [0, 5, 6, 3]
        total_kernel_nums = 0
        task = []
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
                        Gemma8W8BlockScaleBPreShuffleTuner.generate_data,
                        (M, N, K, seed),
                        Gemma8W8BlockScaleBPreShuffleTuner.run_gemm_a8w8_blockscale_bpreshuffle,
                        (
                            gemm_a8w8_data_idx,
                            i,
                            splitK,
                        ),
                        {},
                        Gemma8W8BlockScaleBPreShuffleTuner.run_torch,
                        (ref_data_idx, None, dtypes.bf16),
                        {},
                        None,
                        1e-2,
                        0.01,
                    )
                )
                total_kernel_nums = total_kernel_nums + 1
        return task

    def tune(self, untunedf, tunedf, args):
        useSplitK = args.splitK
        shape_grouped = False
        mp_num = args.mp
        cu_num = self.get_cu_num()
        task = []
        tasks_data = []

        seed = 0
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            seed = seed + 1
            total_kernel_nums = 0

            info_keys = [untunedf.loc[i, key] for key in self.keys]
            task.extend(self.get_ck_tasks(info_keys, useSplitK, seed))
            total_kernel_nums = len(task)

            tasks_data.append((total_kernel_nums, ()))
        ret = []
        if task:
            ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped)
        return ret


if __name__ == "__main__":
    ## use default key and resultList
    tuner = Gemma8W8BlockScaleBPreShuffleTuner(
        "Gemma8W8BlockScaleBPreShuffleTuner",
        description="gen API for CK gemm a8w8 bpreshuffle kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)  # fast_mode = False
