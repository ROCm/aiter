# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
    EightWaveKernelInstance,
    kernels_list_8wave,
)


def test_gfx950_int8_8wave_name_contract():
    candidate = EightWaveKernelInstance(
        128,
        256,
        1,
        8,
        q_dtype_a="int8",
        q_dtype_w="int8",
    )
    assert candidate.name == "flydsl_bpreshuffle_8w_128x256x128_I8_I8_B16_1x8"
    int8_candidates = [
        ki for ki in kernels_list_8wave.values() if ki.q_dtype_w == "int8"
    ]
    fp8_candidates = [ki for ki in kernels_list_8wave.values() if ki.q_dtype_w == "fp8"]
    assert len(int8_candidates) == len(fp8_candidates) > 0
