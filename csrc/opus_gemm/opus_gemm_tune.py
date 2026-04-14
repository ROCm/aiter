# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_OPUS_GEMM_A16W16
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.deepgemm import opus_gemm_a16w16_tune as _opus_gemm_a16w16_tune
from opus_gemm_common import a16w16_kernels_list


a16w16_kernel_ids = sorted(a16w16_kernels_list.keys())


def generate_data(batch, m, n, k, seed, device="cuda"):
    torch.manual_seed(seed)
    XQ = torch.randn((batch, m, k), dtype=dtypes.bf16, device=device)
    WQ = torch.randn((batch, n, k), dtype=dtypes.bf16, device=device)
    Y = torch.empty((batch, m, n), dtype=dtypes.bf16, device=device)
    return XQ, WQ, Y


def opus_gemm_ref(XQ, WQ):
    return torch.bmm(XQ.float(), WQ.float().transpose(-1, -2)).to(dtypes.bf16)


def run_opus_gemm(XQ, WQ, Y, kernelId, splitK):
    _opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)
    return Y


class OpusGemmA16W16Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_OPUS_GEMM_A16W16}",
        "untune_file": "aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
    }

    def getKernelName(self, kernelId):
        k = a16w16_kernels_list.get(kernelId)
        return k.name if k else None

    def _setup_specific_arguments(self):
        pass

    def calculate(self, results, bpes=(2, 2, 2)):
        return super().calculate(results, bpes=(2, 2, 2))

    def tune(self, untunedf, tunedf, args):
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = False
        errRatio = args.errRatio
        cu_num = self.get_cu_num()

        task = []
        tasks_data = []
        opus_data_idx = [0, 1, 2]
        ref_data_idx = [0, 1]
        seed = 0

        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            batch = int(untunedf.loc[i, "batch"]) if "batch" in untunedf.columns else 1
            seed = seed + 1

            total_kernel_nums = 0
            info_keys = (cu_num, M, N, K)

            for kid in a16w16_kernel_ids:
                maxsplitK = 0
                for splitK in range(maxsplitK + 1):
                    info = (info_keys, kid, splitK, "")
                    task.append(
                        (
                            info,
                            generate_data,
                            (batch, M, N, K, seed),
                            run_opus_gemm,
                            (opus_data_idx, kid, splitK),
                            {
                                "num_warmup": args.warmup,
                                "num_iters": args.iters,
                            },
                            opus_gemm_ref,
                            (ref_data_idx,),
                            {},
                            None,
                            1e-2,
                            1e-2,
                        )
                    )
                    total_kernel_nums += 1

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
    key = ["cu_num", "M", "N", "K"]
    resultList = [
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner = OpusGemmA16W16Tuner(
        "OpusGemmA16W16Tuner",
        key=key,
        resultList=resultList,
        description="Tune opus GEMM a16w16 (bf16) kernels",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
