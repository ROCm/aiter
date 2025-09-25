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
from aiter.jit.core import get_asm_dir


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


def run_gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a8w8_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def run_gemm_a8w8_asm(
    x,
    weight,
    x_scale,
    w_scale,
    out,
    bias,
    kernelName,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    splitK=None,
):

    return aiter.gemm_a8w8_asm(
        x,
        weight,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        splitK=splitK,
    )


def generate_data(
    m, n, k, seed, dtype=dtypes.bf16, q_dtype_w=dtypes.fp8, is_asm=False, device="cuda"
):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=dtype, device=device)
    weight = torch.randn((n, k), dtype=dtype, device=device)
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=q_dtype_w)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=q_dtype_w)
    bias_f32 = None
    if is_asm:
        weight_shuffle = shuffle_weight(weight, layout=(32, 16))
        bias = torch.zeros(1, n, dtype=dtype, device=device)
        bias_f32 = bias.to(dtypes.fp32)
    else:
        weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtype, device=device)
    return x, weight_shuffle, x_scale, w_scale, out, weight, bias_f32


class GemmA8W8BpreShuffleTuner(GemmCommonTuner):
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

    def get_asm_gemm_i8_tasks(self, info_keys, useSplitK, kernel_id_start, seed=0):
        total_kernel_nums = 0
        task = []
        (cu_num, M, N, K, q_dtype_w) = info_keys
        if q_dtype_w != dtypes.i8:
            return task
        asm_kernel_list_csv = f"{get_asm_dir()}/i8gemm/i8gemm_bf16_perTokenI8.csv"
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
        asm_tiles = [key for key in asm_kernels.keys()]

        gemm_asm_data_idx = [0, 1, 2, 3, 4, 6]  # input index in generate_data
        torch_data_idx = [0, 5, 2, 3, 6]
        asm_kernels_id = kernel_id_start
        for key in asm_tiles:
            tile_m, tile_n, splitk = key
            if tile_m != 192:
                continue
            maxsplitK = 8 if useSplitK else 1
            kernelName = asm_kernels.get((tile_m, tile_n, splitk), [])
            if len(kernelName) == 0:
                print(f"no kernel name for ({tile_m}, {tile_n})!!!!")
                continue
            if splitk == 0:
                maxsplitK = 0
            for splitK in range(1, maxsplitK + 1):
                kernel_name = kernelName[0]
                info = (info_keys, asm_kernels_id, splitK, kernel_name)
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, q_dtype_w, True),
                        run_gemm_a8w8_asm,
                        (
                            gemm_asm_data_idx,
                            kernel_name,
                            dtypes.bf16,
                            True,
                            splitK,
                        ),
                        {
                            "num_warmup": 10,
                            "num_iters": 101,
                        },
                        run_torch,
                        (
                            torch_data_idx,
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

    def get_ck_gemm_a8w8_bpreshuffle_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
    ):
        (cu_num, M, N, K, q_dtype_w) = info_keys
        if q_dtype_w != dtypes.fp8:
            return []
        kernels_num = len(kernels_list)
        gemm_a8w8_idx = [0, 1, 2, 3, 4]  # input index in generate_data
        ref_data_idx = [0, 5, 2, 3, 6]
        tasks_ck = []
        for i in range(3):
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
                        (M, N, K, seed, dtypes.bf16, q_dtype_w),
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
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.1,
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
            q_dtype_w = eval(untunedf.loc[i, "q_dtype_w"])
            seed = seed + 1
            total_kernel_nums = 0
            kernels_num = len(kernels_list)
            info_keys = (cu_num, M, N, K, q_dtype_w)
            task.extend(
                self.get_ck_gemm_a8w8_bpreshuffle_tune_task(
                    info_keys,
                    useSplitK,
                    seed,
                )
            )

            task.extend(
                self.get_asm_gemm_i8_tasks(info_keys, useSplitK, kernels_num + 1, seed)
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
    tuner = GemmA8W8BpreShuffleTuner(
        "GemmA8W8BpreShuffleTuner",
        key=key,
        description="gen API for gemm a8w8 bpreshuffle kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
