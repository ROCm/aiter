#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune new FlyDSL MXFP4 GEMM1 with mixed_moe_gemm_2stage GEMM2.

Input accepts one CSV or a path-separated list. Both untuned shape CSVs and
existing tuned CSVs are reduced to their shape keys, merged, and deduplicated.
Output rows require ``flydsl_mxmoe_g1_a4w4_*`` GEMM1 and
``flydsl_moe2_*`` mixed GEMM2 names.

Example:
    python csrc/ck_gemm_moe_2stages_codegen/tune_mxfp4_flydsl.py \
      -i "untuned.csv:tuned.csv" -o replacement_tuned.csv \
      --all --mp 6 --timeout 45 --errRatio 0.1
"""

try:
    from .gemm_moe_tune import Mxfp4FlydslTuner
except ImportError:
    from gemm_moe_tune import Mxfp4FlydslTuner


KEY_COLUMNS = [
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

RESULT_COLUMNS = [
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
    "flat",
    "tflops",
    "bw",
]


def main():
    tuner = Mxfp4FlydslTuner(
        "mxfp4FlydslTuner",
        KEY_COLUMNS,
        RESULT_COLUMNS,
        "replacement mxfp4 a4w4 FlyDSL MoE tuner",
    )
    args = tuner.parse_args()
    tuner.run(args, False)


if __name__ == "__main__":
    main()
