# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE
from einops import rearrange
from gemm_a8w8_blockscale_bpreshuffle_common import kernels_list
from aiter.jit.core import get_asm_dir

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

block_shape = (128, 128)


class Gemma8W8BlockScaleBPreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}",
        "untune_file": "aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    }

    def _setup_specific_arguments(self):
        pass

    def calculate(self, results, bpes=(1, 1, 2)):
        ## bpes = (inbpe, w_bpe, outbpe)
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="ck"):
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
    def run_gemm_a8w8_blockscale_bpreshuffle_asm(
        x,
        weight,
        A_scale,
        B_scale,
        out,
        bias=None,
        splitK=None,
        kernelName="",
        bpreshuffle=True,
    ):
        aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
            x, weight, out, A_scale, B_scale, bias, splitK, kernelName, bpreshuffle
        )
        return out

    @staticmethod
    def generate_data(m, n, k, seed, is_asm=False, device="cuda"):
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
        if is_asm:
            weight_shuffle = shuffle_weight(weight, layout=(32, 16))
        else:
            weight_shuffle = shuffle_weight(weight, layout=(16, 16))
        # bias = torch.zeros(1, scale_n, dtype=dtypes.fp16, device=device)
        bias_f32 = None  # bias.to(dtypes.fp32)
        out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
        x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
        return x, weight_shuffle, x_scale_t, w_scale, out, weight, x_scale, bias_f32

    def get_asm_kernels(self, file):
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
            shuffle_df.groupby(["tile_m", "tile_n", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def get_asm_gemm_f8_tasks(self, info_keys, useSplitK, kernel_id_start, seed=0):
        task = []
        (cu_num, M, N, K) = info_keys
        asm_kernel_list_csv = (
            f"{get_asm_dir()}/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv"
        )
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
        asm_tiles = [key for key in asm_kernels.keys()]

        gemm_asm_data_idx = [0, 1, 6, 3, 4, 7]  # input index in generate_data
        torch_data_idx = [0, 5, 6, 3]
        asm_kernels_id = kernel_id_start
        for key in asm_tiles:
            tile_m, tile_n, splitk = key
            maxsplitK = 8 if useSplitK else 1
            kernelName = asm_kernels.get((tile_m, tile_n, splitk), [])
            if len(kernelName) == 0:
                print(f"no kernel name for ({tile_m}, {tile_n})!!!!")
                continue
            if splitk == 0:
                maxsplitK = 1
            for splitK in range(1, maxsplitK + 1):
                kernel_name = kernelName[0]
                info = ((cu_num, M, N, K), asm_kernels_id, splitK, kernel_name, "asm")
                task.append(
                    (
                        info,
                        Gemma8W8BlockScaleBPreShuffleTuner.generate_data,
                        (M, N, K, seed, True),
                        Gemma8W8BlockScaleBPreShuffleTuner.run_gemm_a8w8_blockscale_bpreshuffle_asm,
                        (
                            gemm_asm_data_idx,
                            splitK,
                            kernel_name,
                            True,
                        ),
                        {
                            "num_warmup": 10,
                            "num_iters": 101,
                        },
                        Gemma8W8BlockScaleBPreShuffleTuner.run_torch,
                        (
                            torch_data_idx,
                            None,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                    )
                )
            asm_kernels_id = asm_kernels_id + 1
        return task

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
                info = ((cu_num, M, N, K), i, splitK, "", "ck")
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
            task.extend(self.get_asm_gemm_f8_tasks(info_keys, useSplitK, 0))
            total_kernel_nums = len(task)

            tasks_data.append((total_kernel_nums, ()))
        ret = []
        if task:
            ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped)
        return ret

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME
                else (
                    self.getKernelName(kernelId, libtype)
                    if kernelName == ""
                    else kernelName
                )
            )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))

            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} {kernelName} {splitK=}, {time}us, {err_ratio=}, {tflops=} TFLOPS, {bw=} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "errRatio": [err_ratio],
                    "tflops": [tflops],
                    "bw": [bw],
                }
            )
            temp = pd.DataFrame(key_dict)
            if resultdf.empty:
                resultdf = temp
            else:
                resultdf = pd.concat([resultdf, temp], ignore_index=True)
        return resultdf


if __name__ == "__main__":
    ## use default key and resultList

    resultList = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner = Gemma8W8BlockScaleBPreShuffleTuner(
        "Gemma8W8BlockScaleBPreShuffleTuner",
        resultList=resultList,
        description="gen API for CK gemm a8w8 bpreshuffle kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)  # fast_mode = False
