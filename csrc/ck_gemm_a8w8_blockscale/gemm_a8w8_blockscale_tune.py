# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BLOCKSCALE, get_asm_dir
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.shuffle import shuffle_weight
from aiter.jit.utils.chip_info import get_gfx

sys.path.insert(0, str(Path(__file__).parent.parent))
from ck_gemm_a8w8_blockscale_bpreshuffle.gemm_a8w8_blockscale_bpreshuffle_common import (
    kernels_list as candidate_kernels_bpreshuffle_dict,
)
from gemm_a8w8_blockscale_instance import candidate_kernels_dict

# cktile
from gemm_a8w8_blockscale_cktile_instance import (
    candidate_kernels_cktile_dict,
    BLOCK_PER_CU_MAX,
)

block_shape = (128, 128)


def get_valid_asm_splitK_list(K: int, max_splitK: int, tile_k: int = 128):
    """Filter splitK values to only those that produce valid TileK-aligned partitions."""
    valid = []
    for sk in range(1, max_splitK + 1):
        k_per_split = (K + sk - 1) // sk
        k_per_split_aligned = ((k_per_split + tile_k - 1) // tile_k) * tile_k
        actual_ksplit = (K + k_per_split_aligned - 1) // k_per_split_aligned
        if actual_ksplit == sk:
            valid.append(sk)
    return valid if valid else [1]


"""
a8w8_blockscale_gemm tuning for ck, ck_tile and asm
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


def run_gemm_a8w8_blockscale_asm(
    x,
    weight,
    x_scale,
    w_scale,
    out,
    zero_bias_buf,
    kernel_name,
    splitK=1,
    preshuffleB=True,
):
    """
    Run gemm a8w8 blockscale tuned kernel for asm type.
    """

    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
        x,
        weight,
        out,
        x_scale,
        w_scale,
        None,
        splitK,
        kernel_name,
        preshuffleB,
        zero_bias_buf,
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
    zero_bias = torch.zeros((1, n), dtype=torch.float32, device=device)
    return (x, weight, x_scale, w_scale, out, weight_shuffle, x_scale_t, zero_bias)


class GemmA8W8BlockScaleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}",
        "untune_file": "aiter/configs/a8w8_blockscale_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",  # for both results
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    }

    def __init__(self, name, keys, resultList, description=""):
        """
        Initialize the Gemm A8W8 BlockScale Tuner.
        """

        super().__init__(name, keys, resultList, description)

    def _clear_op_caches(self):
        from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type

        get_GEMM_config_with_quant_type.cache_clear()
        if hasattr(get_GEMM_config_with_quant_type, "file_cache"):
            get_GEMM_config_with_quant_type.file_cache.clear()

    def _setup_specific_arguments(self):
        """
        Setup specific arguments for the tuner.
        """

        self.parser.add_argument(
            "--libtype",
            type=str,
            default="all",
            choices=["ck", "cktile", "asm", "all", "both"],
            required=False,
            help="CK gemm a8w8 blockscale type to tune: ck, cktile, asm, both or all (covers all supported backends across standard/preshuffleB modes)",
        )

        self.parser.add_argument(
            "--preshuffle",
            action="store_true",
            help="Enable B-matrix preshuffle for CK gemm a8w8 blockscale",
        )

        self.parser.add_argument(
            "--blockPerCu",
            nargs="+",
            type=int,
            default=list(range(1, BLOCK_PER_CU_MAX + 1)),
            help="List of BlockPerCu values to tune (CKTile only)",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        """
        Calculate performance metrics based on results.
        """

        _info, time, _err_ratio = results
        if time == self.INVALID_TIME or time == self.INF_TIME:
            return 0, 0
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

    def get_asm_kernels(self, file, preshuffleB):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}

        df = pd.read_csv(file)
        asm_df = (
            df[df["bpreshuffle"] == int(preshuffleB)]
            .reset_index(drop=True)
            .sort_values(by=["tile_m", "tile_n", "splitK"])
        )
        kernel_dict = (
            asm_df.groupby(["tile_m", "tile_n", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def get_gemm_a8w8_blockscale_cktile_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        block_per_cu,
        run_kwargs,
    ):
        cu_num, M, N, K = info_keys
        # kernel_list = candidate_kernels_bpreshuffle_cktile_dict if preshuffleB else candidate_kernels_cktile_dict
        kernel_list = {
            k: v
            for k, v in candidate_kernels_cktile_dict.items()
            if v.BlockPerCu in block_per_cu
        }
        # gemm_a8w8_idx = [0, 5 if preshuffleB else 1, 2, 3, 4]
        gemm_a8w8_idx = [0, 5, 6, 3, 4] if preshuffleB else [0, 1, 2, 3, 4]
        ref_data_idx = [0, 1, 2, 3]
        tasks_cktile = []
        for i, kernel in kernel_list.items():
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
                        dict(run_kwargs),
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
        run_kwargs,
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
                        dict(run_kwargs),
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

    def run_config(self, args):
        from aiter.ops.gemm_op_a8w8 import (
            gemm_a8w8_blockscale,
            gemm_a8w8_blockscale_bpreshuffle,
        )
        from aiter.test_common import run_perftest, checkAllclose

        is_preshuffle = args.preshuffle
        untunedf = self.untunedf
        run_kwargs = {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
        }
        results = []
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            shape_str = f"({M}, {N}, {K})"
            try:
                x, weight, x_scale, w_scale, out, weight_shuffle, x_scale_t, _ = (
                    generate_data(M, N, K, 0)
                )
                if is_preshuffle:
                    out, us = run_perftest(
                        gemm_a8w8_blockscale_bpreshuffle,
                        x,
                        weight_shuffle,
                        x_scale_t,
                        w_scale,
                        **run_kwargs,
                    )
                else:
                    out, us = run_perftest(
                        gemm_a8w8_blockscale,
                        x,
                        weight,
                        x_scale,
                        w_scale,
                        **run_kwargs,
                    )
                ref = run_torch(x, weight, x_scale, w_scale)
                err_ratio = checkAllclose(out, ref, msg=f"run_config {shape_str}")
                status = "ok" if err_ratio <= args.errRatio else "mismatch"
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
        return results

    def get_gemm_a8w8_blockscale_asm_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        cu_num, M, N, K = info_keys
        asm_kernel_list_csv = (
            f"{get_asm_dir()}/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv"
        )
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv, preshuffleB)
        if not asm_kernels:
            return []

        gemm_asm_idx = [0, 5, 6, 3, 4, 7] if preshuffleB else [0, 1, 2, 3, 4, 7]
        ref_data_idx = [0, 1, 2, 3]
        tasks_asm = []
        asm_kernel_id = 0
        for key, kernel_names in asm_kernels.items():
            _tile_m, tile_n, splitk_supported = key
            # Respect ASM kernel tile constraints from the config CSV.
            if N % tile_n != 0:
                continue
            splitK_list = (
                get_valid_asm_splitK_list(K, 8)
                if useSplitK and int(splitk_supported) == 1
                else [1]
            )
            for kernel_name in kernel_names:
                for splitK in splitK_list:
                    info = (
                        info_keys,
                        asm_kernel_id,
                        splitK,
                        kernel_name,
                        "asm",
                        preshuffleB,
                    )
                    tasks_asm.append(
                        (
                            info,
                            generate_data,
                            (M, N, K, seed),
                            run_gemm_a8w8_blockscale_asm,
                            (
                                gemm_asm_idx,
                                kernel_name,
                                splitK,
                                preshuffleB,
                            ),
                            dict(run_kwargs),
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
                    asm_kernel_id += 1
        return tasks_asm

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        useSplitK = args.splitK
        mp_num = args.mp
        isPreshuffleB = args.preshuffle
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        block_per_cu = args.blockPerCu
        cu_num = self.get_cu_num()
        run_kwargs = {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
        }
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        seed = 10000
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            seed = seed + 1
            prev_task_count = len(task)
            info_keys = (cu_num, M, N, K)
            lib = args.libtype
            if lib in ("ck", "both", "all"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
            if lib in ("cktile", "both", "all"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_cktile_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        block_per_cu,
                        run_kwargs,
                    )
                )
            if lib in ("asm", "all"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_asm_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
            shape_kernel_nums = len(task) - prev_task_count

            tasks_data.append((shape_kernel_nums, ()))
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
                if time == self.INVALID_TIME or time == self.INF_TIME
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
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} "
                    f"{kernelName} splitK={splitK}, {time}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bw} GB/s"
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
        description="Tune a8w8 blockscale GEMM (CK, CKTile, ASM backends)",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
