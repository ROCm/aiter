# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: E402

import os
import sys
from pathlib import Path

# Force local source tree first to avoid importing stale site-packages aiter.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import pandas as pd
import argparse
from aiter.jit.core import (
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
    get_asm_dir,
)
from einops import rearrange
from gemm_a8w8_blockscale_bpreshuffle_common import kernels_list

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

block_shape = (128, 128)
DEFAULT_TUNED_CONFIG = "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
DEFAULT_UNTUNED_CONFIG = "a8w8_blockscale_bpreshuffle_untuned_gemm.csv"


def libtype_list(string):
    values = string.split(",")
    for value in values:
        if value not in ["all", "asm", "ck"]:
            raise argparse.ArgumentTypeError(f"Invalid libtype: {value}")
    return values


def get_current_cu_num():
    cu_num = os.getenv("CU_NUM")
    if cu_num:
        return int(cu_num)
    if torch.cuda.is_available():
        gpu = torch.cuda.current_device()
        return torch.cuda.get_device_properties(gpu).multi_processor_count
    return None


def resolve_machine_specific_config(config_path: str, config_kind: str) -> str:
    default_name = (
        DEFAULT_TUNED_CONFIG if config_kind == "tuned" else DEFAULT_UNTUNED_CONFIG
    )
    if Path(config_path).name != default_name:
        return config_path

    cu_num = get_current_cu_num()
    if cu_num is None:
        return config_path

    model_config_dir = PROJECT_ROOT / "aiter" / "configs" / "model_configs"
    pattern = f"a8w8_blockscale_bpreshuffle_{config_kind}_gemm_*_cu{cu_num}.csv"
    matches = sorted(model_config_dir.glob(pattern))
    if len(matches) == 1:
        resolved_path = str(matches[0])
        print(
            f"Auto-selected cu{cu_num} {config_kind} config for tuning: {resolved_path}"
        )
        return resolved_path
    if len(matches) > 1:
        print(
            f"Found multiple cu{cu_num} {config_kind} configs matching {pattern}, "
            f"keeping requested path: {config_path}"
        )
    return config_path


class Gemma8W8BlockScaleBPreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}",
        "untune_file": "aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    }

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--libtype",
            type=libtype_list,
            default=["all"],
            required=False,
            help="choose libtype to be tuned, support ['all', 'asm', 'ck']",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        # bpes = (inbpe, w_bpe, outbpe)
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="ck"):
        if libtype != "ck":
            return None
        if kernelId < 0 or kernelId >= len(kernels_list):
            return None
        return kernels_list[kernelId].name

    @staticmethod
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
        x, weight, out, x_scale, w_scale, kernel_name, splitK=1
    ):
        return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
            x,
            weight,
            out,
            x_scale,
            w_scale,
            None,
            splitK,
            kernel_name,
            True,
        )

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

    def get_asm_kernels(self, file):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}
        df = pd.read_csv(file)
        shuffle_df = (
            df[df["bpreshuffle"] == 1]
            .reset_index(drop=True)
            .sort_values(by=["tile_m", "tile_n", "splitK"])
        )
        kernel_dict = (
            shuffle_df.groupby(["tile_m", "tile_n", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def get_ck_tasks(self, info_keys, useSplitK, seed, num_warmup, num_iters):
        cu_num, M, N, K = info_keys
        kernels_num = len(kernels_list)
        gemm_a8w8_data_idx = [0, 1, 2, 3, 4]
        ref_data_idx = [0, 5, 6, 3]
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
                        {
                            "num_warmup": num_warmup,
                            "num_iters": num_iters,
                        },
                        Gemma8W8BlockScaleBPreShuffleTuner.run_torch,
                        (ref_data_idx, None, dtypes.bf16),
                        {},
                        None,
                        1e-2,
                        0.01,
                    )
                )
        return task

    def get_asm_tasks(self, info_keys, useSplitK, seed, num_warmup, num_iters):
        cu_num, M, N, K = info_keys
        asm_kernel_list_csv = (
            f"{get_asm_dir()}/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv"
        )
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
        if not asm_kernels:
            return []

        gemm_asm_data_idx = [0, 1, 4, 2, 3]
        ref_data_idx = [0, 5, 6, 3]

        asm_tasks = []
        asm_kernel_id = 0
        for key, kernel_names in asm_kernels.items():
            tile_m, tile_n, splitk_supported = key
            # Respect ASM kernel tile constraints from the config CSV.
            if N % tile_n != 0:
                continue
            splitK_list = (
                list(range(1, 9)) if useSplitK and int(splitk_supported) == 1 else [1]
            )
            for kernel_name in kernel_names:
                for splitK in splitK_list:
                    info = (
                        (cu_num, M, N, K),
                        asm_kernel_id,
                        splitK,
                        kernel_name,
                        "asm",
                    )
                    asm_tasks.append(
                        (
                            info,
                            Gemma8W8BlockScaleBPreShuffleTuner.generate_data,
                            (M, N, K, seed),
                            Gemma8W8BlockScaleBPreShuffleTuner.run_gemm_a8w8_blockscale_bpreshuffle_asm,
                            (
                                gemm_asm_data_idx,
                                kernel_name,
                                splitK,
                            ),
                            {
                                "num_warmup": num_warmup,
                                "num_iters": num_iters,
                            },
                            Gemma8W8BlockScaleBPreShuffleTuner.run_torch,
                            (ref_data_idx, None, dtypes.bf16),
                            {},
                            None,
                            1e-2,
                            0.01,
                        )
                    )
                    asm_kernel_id += 1
        return asm_tasks

    def tune(self, untunedf, tunedf, args):
        useSplitK = args.splitK
        shape_grouped = False
        mp_num = args.mp
        errRatio = args.errRatio
        task = []
        tasks_data = []

        seed = 0
        for i in range(len(untunedf)):
            seed = seed + 1
            total_kernel_nums = 0

            info_keys = [untunedf.loc[i, key] for key in self.keys]
            if "all" in args.libtype or "ck" in args.libtype:
                task.extend(
                    self.get_ck_tasks(
                        info_keys,
                        useSplitK,
                        seed,
                        args.warmup,
                        args.iters,
                    )
                )
            if "all" in args.libtype or "asm" in args.libtype:
                task.extend(
                    self.get_asm_tasks(
                        info_keys,
                        useSplitK,
                        seed,
                        args.warmup,
                        args.iters,
                    )
                )
            total_kernel_nums = len(task)

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

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME or time == self.INF_TIME
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
                    f"Tuning result for {str(key_dict).strip('{}')} is "
                    f"kernelId={kernelId} {kernelName} splitK={splitK}, "
                    f"{time}us, err_ratio={err_ratio}, tflops={tflops} TFLOPS, bw={bw} GB/s"
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
    tuner = Gemma8W8BlockScaleBPreShuffleTuner(
        "Gemma8W8BlockScaleBPreShuffleTuner",
        key=key,
        resultList=resultList,
        description="gen API for CK/ASM gemm a8w8 blockscale bpreshuffle kernel",
    )

    args = tuner.parse_args()
    args.tune_file = resolve_machine_specific_config(args.tune_file, "tuned")
    args.untune_file = resolve_machine_specific_config(args.untune_file, "untuned")
    tuner.run(args, False)  # fast_mode = False
