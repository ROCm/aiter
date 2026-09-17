# SPDX-License-Identifier: MIT
"""Run one DSV4.1 A8W4 stage1 tuner candidate and print each oracle result."""

import json

import torch

from aiter import ActivationType
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner
from csrc.ck_gemm_moe_2stages_codegen.mxfp4_v2_tune_utils import (
    v2_stage1_output_error,
)

TOKENS = 32
HIDDEN = 5120
INTERMEDIATE = 2304
EXPERTS = 96
TOPK = 6
BLOCK_M = 32
KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8"


def main():
    params = get_flydsl_kernel_params(KERNEL)
    assert params is not None
    data = FmoeTuner.generate_v2_stage1_data(
        TOKENS,
        HIDDEN,
        INTERMEDIATE,
        EXPERTS,
        TOPK,
        BLOCK_M,
        "fp8",
        "fp4",
        ActivationType.Silu,
    )
    result = FmoeTuner.run_flydsl_v2_stage1_out(
        data["a1_qt"],
        data["w1_shuf"],
        data["w1_scale_shuf"],
        data["a1_scale_sort"],
        data["sti"],
        data["sei"],
        data["cumsum"],
        data["isq"],
        data["n"],
        HIDDEN,
        INTERMEDIATE,
        EXPERTS,
        TOPK,
        BLOCK_M,
        "fp8",
        ActivationType.Silu,
        params,
    )
    reference = FmoeTuner.run_v2_stage1_sorted_ref(
        data["ref1"],
        data["ref1_scale"],
        data["topk_ids"],
        data["sti"],
        data["sei"],
        data["n"],
        TOKENS,
        INTERMEDIATE,
        BLOCK_M,
    )
    payload_error = v2_stage1_output_error(
        reference[0],
        result[0],
        inter_dim=INTERMEDIATE,
        adtype="fp8",
        printLog=False,
    )
    scale_error = v2_stage1_output_error(
        reference[1],
        result[1],
        inter_dim=INTERMEDIATE,
        adtype="fp8",
        printLog=False,
    )
    print(
        json.dumps(
            {
                "kernel": KERNEL,
                "params": {
                    "a_scale_one": params.get("a_scale_one", False),
                    "out_dtype": params["out_dtype"],
                },
                "payload_error": payload_error,
                "scale_error": scale_error,
                "scale_shapes": [list(reference[1].shape), list(result[1].shape)],
                "scale_dtypes": [str(reference[1].dtype), str(result[1].dtype)],
                "scale_prefix": [
                    reference[1].view(torch.uint8).flatten()[:16].tolist(),
                    result[1].view(torch.uint8).flatten()[:16].tolist(),
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
