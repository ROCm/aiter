# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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

from gemm_a8w8_blockscale_instance_legacy import candidate_kernels_dict_legacy
from gemm_a8w8_blockscale_instance_tile import candidate_kernels_dict_tile

block_shape = (128, 128)


"""
a8w8_blockscale_gemm tuning for legacy CK and tile CK
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


def run_ck_gemm_a8w8_blockscale_tile(
    x, weight, x_scale, w_scale, out, kernel_id, splitK, isBpreshuffled
):
    """
    Run gemm a8w8 blockscale tuned kernel for ck_tile type.
    """

    return aiter.gemm_a8w8_blockscale_tune_tile(
        x, weight, x_scale, w_scale, out, kernel_id, splitK, isBpreshuffled
    )


def run_ck_gemm_a8w8_blockscale_legacy(
    x, weight, x_scale, w_scale, out, kernel_id, splitK
):
    """
    Run gemm a8w8 blockscale tuned kernel for ck_legacy type.
    """

    return aiter.gemm_a8w8_blockscale_tune_legacy(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )


def generate_data(m, n, k, seed, isBpreshuffled=True, device="cuda"):
    """
    Generate random data for testing the gemm a8w8 blockscale kernel.
    If isBpreshuffled is True, return shuffled weight and transposed x_scale.
    """

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
    out = torch.empty(m, n, dtype=dtypes.bf16, device=device)

    if isBpreshuffled:
        weight_shuffle = shuffle_weight(weight, layout=(16, 16))
        x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
        return x, weight_shuffle, x_scale_t, w_scale, out, weight, x_scale
    else:
        return x, weight, x_scale, w_scale, out


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
            choices=["ck_legacy", "ck_tile", "both"],
            required=False,
            help="CK gemm a8w8 blockscale type to tune: ck_legacy, ck_tile or both",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        """
        Calculate performance metrics based on results.
        """

        return super().calculate(results, bpes=(1, 1, 2))

    def getKernelName(self, kernelId, type="ck_legacy"):
        """
        Get the kernel name based on the kernel ID for different types.
        """

        candidate_kernels_dict = {}
        if type == "ck_legacy":
            if kernelId >= len(candidate_kernels_dict_legacy) or kernelId < 0:
                return None
            candidate_kernels_dict = candidate_kernels_dict_legacy
        elif type == "ck_tile":
            if kernelId >= len(candidate_kernels_dict_tile) or kernelId < 0:
                return None
            candidate_kernels_dict = candidate_kernels_dict_tile
        else:
            return None
        return candidate_kernels_dict[kernelId].name

    def get_ck_tile_gemm_a8w8_blockscale_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        isBpreshuffled,
    ):
        (cu_num, M, N, K) = info_keys
        kernels_num = len(candidate_kernels_dict_tile)
        gemm_a8w8_idx = [0, 1, 2, 3, 4]  # input index in generate_data
        ref_data_idx = [0, 5, 6, 3] if isBpreshuffled else [0, 1, 2, 3]
        tasks_ck_tile = []
        for i in range(kernels_num):
            kernel = candidate_kernels_dict_tile[i]
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
                info = (info_keys, i, splitK, "", "ck_tile", isBpreshuffled)
                tasks_ck_tile.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, isBpreshuffled),
                        run_ck_gemm_a8w8_blockscale_tile,
                        (
                            gemm_a8w8_idx,
                            i,
                            splitK,
                            isBpreshuffled,
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
        return tasks_ck_tile

    def get_ck_legacy_gemm_a8w8_blockscale_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
    ):
        (cu_num, M, N, K) = info_keys
        kernels_num = len(candidate_kernels_dict_legacy)
        gemm_a8w8_idx = [0, 1, 2, 3, 4]  # input index in generate_data
        ref_data_idx = [0, 1, 2, 3]
        tasks_ck_legacy = []
        for i in range(kernels_num):
            kernel = candidate_kernels_dict_legacy[i]
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
                info = (info_keys, i, splitK, "", "ck_legacy", False)
                tasks_ck_legacy.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, False),
                        run_ck_gemm_a8w8_blockscale_legacy,
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
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                    )
                )
        return tasks_ck_legacy

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
            seed = seed + 1
            total_kernel_nums = 0
            # kernels_num = len(candidate_kernels_dict_legacy)
            info_keys = (cu_num, M, N, K)
            if "both" in args.libtype or "ck_legacy" in args.libtype:
                task.extend(
                    self.get_ck_legacy_gemm_a8w8_blockscale_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                    )
                )
            if "both" in args.libtype or "ck_tile" in args.libtype:
                for b_preshuffled in [True, False]:
                    task.extend(
                        self.get_ck_tile_gemm_a8w8_blockscale_tune_task(
                            info_keys,
                            useSplitK,
                            seed,
                            b_preshuffled
                        )
                    )

            total_kernel_nums = len(task)

            tasks_data.append((total_kernel_nums, ()))
        ret = []
        if task:
            ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped, errRatio)

        return ret

    def result_to_df(self, results):
        """
        post-process the tuning results into a DataFrame.
        """

        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype, isBpreshuffled = info
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
                    "isBpreshuffled": [isBpreshuffled],
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
        "isBpreshuffled",
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
