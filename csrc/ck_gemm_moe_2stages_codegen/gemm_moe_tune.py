# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.tuning.moe.fused_moe import (
    FmoeTuner,
)
from aiter.tuning.moe.grouped import GroupedFmoeTuner
from aiter.tuning.moe.mxfp4 import Mxfp4FlydslTuner

torch.set_default_device("cuda")
torch.int4 = getattr(torch, "int4", torch.uint32)


if __name__ == "__main__":
    key = [
        "gfx",
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
    # ep_fused is part of the shape identity (gemm2 doing the EP scatter shifts
    # the stage2 tile optimum), so it must be in the dedup key or the write-back
    # collapses an ep_fused row into the generic one for the same shape.
    grouped_key = key + ["gate_mode", "ep_fused"]
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
        "xbf16",
        # Serialized AsmMoeMode value from the manifest.
        "flat",
        "tflops",
        "bw",
    ]
    grouped_result_list = [
        "max_m",
        "tile_m",
        "tile_n",
        "tile_k",
        "m_warp",
        "n_warp",
        "num_buffers",
        "grouped_persistent_m",
        "persistent_workers",
        "kernelName1",
        "kernelName2",
        "us",
        "tflops",
        "bw",
    ]
    use_grouped = "--grouped-gemm" in sys.argv
    use_mxfp4_flydsl = "--mxfp4-flydsl" in sys.argv
    if use_grouped:
        if get_gfx() != "gfx1250":
            raise SystemExit("--grouped-gemm is only supported on gfx1250")
        tuner = GroupedFmoeTuner(
            "groupedFmoeTuner",
            grouped_key,
            grouped_result_list,
            "grouped fmoe tuner",
        )
    elif use_mxfp4_flydsl:
        tuner = Mxfp4FlydslTuner(
            "mxfp4FlydslTuner", key, resultList, "mxfp4 a4w4 flydsl port fmoe tuner"
        )
    else:
        tuner = FmoeTuner("fmoeTuner", key, resultList + ["nt"], "fmoe tuner")
    args = tuner.parse_args()

    if args.e2e_tune:
        tuner.pre_process(args)
        tuner.e2e_tune(args)
    else:
        tuner.run(args, False)
