# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pandas as pd
import time
import os
import sys
from aiter import QuantType
from aiter.jit.core import (
    AITER_CSRC_DIR,
    AITER_META_DIR,
    AITER_CONFIG_FMOE,
)
from aiter.utility.mp_tuner import mp_tuner

from aiter import dtypes
from aiter import ActivationType as ActivationType

sys.path.insert(0, f"{AITER_META_DIR}/hsa/gfx942")
from fmoe_2stages.tune import FmoeTuner


sys.path.insert(0, f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/")


torch.set_default_device("cuda")
torch.int4 = getattr(torch, "int4", torch.uint32)


def get_kernels_dict(file, key="tile_m"):
    if not os.path.exists(file):
        print(f"ASM kernel list file not exist: {file}")
        return {}
    df = pd.read_csv(file)
    kernel_dict = df.groupby(key)["knl_name"].apply(list).to_dict()
    return kernel_dict


class FmoeTuner950(FmoeTuner):
    ARG_DEFAULTS = {
        "verbose": False,
        "tune_file": f"{AITER_CONFIG_FMOE}",
        "untune_file": "aiter/configs/untuned_fmoe.csv",
        "errRatio": 0.5,
        "batch": 100,
        "profile_file": "aiter/configs/profile_fmoe.csv",  # for all results
    }

    def get_1stage_file_info(self, q_type, q_dtype_a, doweight_stage1):
        extraInfo_1stage = ""
        if q_dtype_a == dtypes.i8:
            quantDtype = "Int8"
        elif q_dtype_a == dtypes.fp8:
            quantDtype = "Fp8"
        else:
            quantDtype = ""
        if doweight_stage1:
            extraInfo_1stage = "_tkw1"
        if q_type == QuantType.No:
            quantDtype_1stage = "noquant"
        elif q_type == QuantType.per_1x128:
            quantDtype_1stage = "blockscale" + quantDtype
        elif q_type == QuantType.per_1x32:
            quantDtype_1stage = "pertoken" + "MXfp4"
        else:
            quantDtype_1stage = "pertoken" + quantDtype
        return quantDtype_1stage, extraInfo_1stage

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        mp_num = args.mp
        startTS = time.perf_counter()
        # blockMs = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
        blockMs = [16, 32, 64, 128]

        args = self.keys
        print(untunedf[args])
        tasks = []
        tasks_ck = []
        task_1stage = []
        in_data = []
        for line in untunedf[args].values:
            (
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            ) = line
            dtype = eval(dtype)
            q_dtype_a = eval(q_dtype_a)
            q_dtype_w = eval(q_dtype_w)
            q_type = eval(q_type)
            q_type = QuantType.per_1x128 if q_type == QuantType.per_128x128 else q_type
            print("\nStart tuning", line)
            if not use_g1u1:
                print("no moe solution(g1u0) can tune for ", line)
                continue
            act_type = eval(act_type)
            info = (
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            )
            tasks.extend(self.gen_2stages_asm1_task(info, blockMs))
            tasks_ck.extend(self.gen_2stages_task(info, blockMs))
            task_1stage.extend(self.gen_1stage_asm_task(info))
            if tasks is None and tasks_ck is None and task_1stage is None:
                print("no moe solution can tune for ", line)
                continue
            print(
                f"stage1 asm tasks is {len(tasks)}, tasks_ck is {len(tasks_ck)}, task_1stage is {len(task_1stage)}"
            )
        in_data.append((len(tasks) + len(tasks_ck) + len(task_1stage), ()))
        rets = []
        if len(tasks) + len(tasks_ck) + len(task_1stage) > 0:
            ### shape_grouped should be False as multiple stages
            rets = mp_tuner(
                tasks + tasks_ck + task_1stage, in_data, mp_num, True, False
            )
        if not rets:
            print("no shape to tune or no solution found")
            return []
        else:
            return rets


if __name__ == "__main__":

    key = [
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    resultList = [
        "block_m",
        "ksplit",
        "us1",
        "kernelName1",
        "err1",
        "us2",
        "kernelName2",
        "err2",
        "us",
        "run_1stage",
        "tflops",
        "bw",
    ]
    tuner = FmoeTuner950("fmoeTuner950", key, resultList, "fmoe tuner on gfx950")
    args = tuner.parse_args()

    tuner.run(args, False)
