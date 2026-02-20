# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BLOCKSCALE
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.shuffle import shuffle_weight
from aiter.jit.utils.chip_info import get_gfx

# ck
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ck_gemm_a8w8_blockscale_bpreshuffle.gemm_a8w8_blockscale_bpreshuffle_common import (
    kernels_list as candidate_kernels_bpreshuffle_dict,
)
from gemm_a8w8_blockscale_instance import candidate_kernels_dict

# cktile
from gemm_a8w8_blockscale_cktile_instance import candidate_kernels_cktile_dict

block_shape = (128, 128)


"""
a8w8_blockscale_gemm tuning for ck and ck_tile
"""


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    """
    Run the reference GEMM operation using PyTorch.
    """

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


def run_gemm_a8w8_blockscale_cktile(
    x, weight, x_scale, w_scale, out, kernel_id, splitK, preshuffleB
):
    """
    Run gemm a8w8 blockscale tuned kernel for ck_tile type.
    """

    if preshuffleB:
        return aiter.gemm_a8w8_blockscale_bpreshuffle_cktile_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )
    else:
        return aiter.gemm_a8w8_blockscale_cktile_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )


def run_gemm_a8w8_blockscale(
    x, weight, x_scale, w_scale, out, kernel_id, splitK, preshuffleB
):
    """
    Run gemm a8w8 blockscale tuned kernel for ck type.
    """

    if preshuffleB:
        return aiter.gemm_a8w8_blockscale_bpreshuffle_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )
    else:
        return aiter.gemm_a8w8_blockscale_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )


def generate_data(m, n, k, seed, device="cuda"):
    """
    Generate random data for testing the gemm a8w8 blockscale kernel.
    """

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
    return (x, weight, x_scale, w_scale, out, weight_shuffle, x_scale_t)


class GemmA8W8BlockScaleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}",
        "untune_file": "aiter/configs/a8w8_blockscale_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",  # for both results
    }

    def __init__(self, name, keys, resultList, description=""):
        """
        Initialize the Gemm A8W8 BlockScale Tuner.
        """

        super().__init__(name, keys, resultList, description)

    def _setup_specific_arguments(self):
        """
        Setup specific arguments for the tuner.
        """

        self.parser.add_argument(
            "--libtype",
            type=str,
            default="both",
            choices=["ck", "cktile", "both", "all"],
            required=False,
            help="CK gemm a8w8 blockscale type to tune: ck, cktile, both or all (covers (ck, cktile) x (standard, preshuffleB))",
        )

        self.parser.add_argument(
            "--preshuffle",
            action="store_true",
            help="Enable B-matrix preshuffle for CK gemm a8w8 blockscale",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        """
        Calculate performance metrics based on results.
        """

        return super().calculate(results, bpes=(1, 1, 2))

    def getKernelName(self, kernelId, libType="ck", preshuffleB=False):
        """
        Get the kernel name based on the kernel ID for different types.
        """
        if libType == "ck":
            kernel_list = (
                candidate_kernels_bpreshuffle_dict
                if preshuffleB
                else candidate_kernels_dict
            )
        elif libType == "cktile":
            # kernel_list = candidate_kernels_bpreshuffle_cktile_dict if preshuffleB else candidate_kernels_cktile_dict
            kernel_list = candidate_kernels_cktile_dict
        else:
            return None

        if kernelId >= len(kernel_list) or kernelId < 0:
            return None
        return kernel_list[kernelId].name

    def get_gemm_a8w8_blockscale_cktile_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
    ):
        cu_num, M, N, K = info_keys
        # kernel_list = candidate_kernels_bpreshuffle_cktile_dict if preshuffleB else candidate_kernels_cktile_dict
        kernel_list = candidate_kernels_cktile_dict
        kernels_num = len(kernel_list)
        # gemm_a8w8_idx = [0, 5 if preshuffleB else 1, 2, 3, 4]
        gemm_a8w8_idx = [0, 5, 6, 3, 4] if preshuffleB else [0, 1, 2, 3, 4]
        ref_data_idx = [0, 1, 2, 3]
        tasks_cktile = []
        for i in range(kernels_num):
            kernel = kernel_list[i]
            if not get_gfx().startswith("gfx95"):
                if (kernel.M_Warp * kernel.N_Warp * kernel.K_Warp == 8) or (
                    kernel.K_Warp_Tile > 64  # gfx942 not support
                ):
                    continue

            maxsplitK = (
                aiter.compute_gemm_SplitK(
                    M,
                    N,
                    K,
                    kernel.M_Tile,
                    kernel.N_Tile,
                    kernel.K_Tile,
                )
                if useSplitK
                else 0
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "cktile", preshuffleB)
                tasks_cktile.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a8w8_blockscale_cktile,
                        (
                            gemm_a8w8_idx,
                            i,
                            splitK,
                            preshuffleB,
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
                        0.01,
                    )
                )
        return tasks_cktile

    def get_gemm_a8w8_blockscale_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
    ):
        cu_num, M, N, K = info_keys
        kernel_list = (
            candidate_kernels_bpreshuffle_dict
            if preshuffleB
            else candidate_kernels_dict
        )
        kernels_num = len(kernel_list)
        gemm_a8w8_idx = [0, 5, 6, 3, 4] if preshuffleB else [0, 1, 2, 3, 4]
        ref_data_idx = [0, 1, 2, 3]
        tasks_ck = []
        for i in range(kernels_num):
            kernel = kernel_list[i]
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
                info = (info_keys, i, splitK, "", "ck", preshuffleB)
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a8w8_blockscale,
                        (
                            gemm_a8w8_idx,
                            i,
                            splitK,
                            preshuffleB,
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
        useSplitK = args.splitK
        mp_num = args.mp
        isPreshuffleB = args.preshuffle
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
            seed = seed + 1
            total_kernel_nums = 0
            # kernels_num = len(candidate_kernels_dict)
            info_keys = (cu_num, M, N, K)
            if args.libtype == "ck" or args.libtype == "both":
                task.extend(
                    self.get_gemm_a8w8_blockscale_tune_task(
                        info_keys, useSplitK, seed, isPreshuffleB
                    )
                )
            if args.libtype == "cktile" or args.libtype == "both":
                task.extend(
                    self.get_gemm_a8w8_blockscale_cktile_tune_task(
                        info_keys, useSplitK, seed, isPreshuffleB
                    )
                )
            if args.libtype == "all":
                for preshuffleB in [True, False]:
                    task.extend(
                        self.get_gemm_a8w8_blockscale_cktile_tune_task(
                            info_keys, useSplitK, seed, preshuffleB
                        )
                    )
                    task.extend(
                        self.get_gemm_a8w8_blockscale_tune_task(
                            info_keys, useSplitK, seed, preshuffleB
                        )
                    )
            total_kernel_nums = len(task)

            tasks_data.append((total_kernel_nums, ()))
        ret = []
        if task:
            ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped, errRatio)
        # print(ret)
        return ret

    def result_to_df(self, results):
        """
        post-process the tuning results into a DataFrame.
        """

        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype, preshuffleB = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME
                else (
                    self.getKernelName(kernelId, libtype, preshuffleB)
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
    key = ["cu_num", "M", "N", "K"]
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
    tuner = GemmA8W8BlockScaleTuner(
        "GemmA8W8BlockScaleTuner",
        key,
        resultList,
        description="gen API for CK gemm a8w8 blockscale kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
