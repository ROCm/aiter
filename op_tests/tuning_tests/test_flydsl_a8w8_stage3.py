# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
    k_split_candidates,
    kernel_fits_shape,
    kernel_instance_estimated_lds_bytes,
    kernels_list_950,
)


def test_gfx950_int8_stage3_candidate_contract():
    candidates = [
        candidate for candidate in kernels_list_950.values() if candidate.lds_stage == 3
    ]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert (candidate.tile_m, candidate.tile_n, candidate.tile_k) == (128, 128, 128)
    assert candidate.q_dtype_a == candidate.q_dtype_w == "int8"
    assert candidate.use_async_copy == 1
    assert candidate.xcd_swizzle == 8
    assert candidate.k_split == 1
    assert kernel_instance_estimated_lds_bytes(candidate) == 3 * 128 * 128
    assert kernel_fits_shape(candidate, 16384, 5120, 1280)
    assert k_split_candidates(candidate, 16384, 5120, 1280) == []
