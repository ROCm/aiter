# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU regression tests for varlen attention kernel dtype selection."""

import fnmatch

import pytest

from aiter.jit.utils.mha_recipes import compose_mha_fwd_variant_suffix_and_filter

# Kernel names emitted by CK 96afae1d for receipt 200 and target gfx1250.
BF16_KERNEL = (
    "fmha_fwd_d32_bf16_group_b64x64x32x32x32x64_r4x1x1_r4x1x1"
    "_w16x16x32_w16x16x32_qr_tdm_vr_pssk_nlogits_nbias_nmask_nlse"
    "_ndropout_nskip_nqscale_ntrload_kvlp_plk_sink_gfx125"
)
FP8BF16_KERNEL = (
    "fmha_fwd_d128_fp8bf16_group_b64x128x128x128x128x128_r4x1x1_r4x1x1"
    "_w16x16x128_w16x16x128_qr_tdm_vr_pssk_nlogits_nbias_nmask_nlse"
    "_ndropout_nskip_nqscale_ntrload_nkvlp_nplk_nsink_gfx125"
)


@pytest.mark.parametrize(
    "dtype,accepted,rejected",
    [
        ("bf16", BF16_KERNEL, FP8BF16_KERNEL),
        ("fp8bf16", FP8BF16_KERNEL, BF16_KERNEL),
    ],
    ids=["bf16-excludes-fp8", "fp8-remains-reachable"],
)
def test_dtype_selection(dtype, accepted, rejected):
    _, pattern = compose_mha_fwd_variant_suffix_and_filter(
        dtype=dtype,
        logits_positive=False,
        has_bias=False,
        has_alibi=False,
        use_mask=False,
        return_lse=False,
        dropout_zero=True,
        skip_zero=True,
        has_qscale=True,
    )

    assert fnmatch.fnmatchcase(accepted, pattern)
    assert not fnmatch.fnmatchcase(rejected, pattern)
