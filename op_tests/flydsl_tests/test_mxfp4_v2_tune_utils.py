# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes


@pytest.mark.l2_device
def test_a8w8_v2_data_uses_fp8_weights_and_e8m0_scales():
    from aiter.ops.flydsl.mxfp4_v2_tune_utils import build_v2_inputs, gen

    with torch.device("cuda"):
        data = gen(8, 256, 128, 4, 2, 32, adtype="fp8", b_dtype="fp8")
        inputs = build_v2_inputs(data, 8, 256, 128, 4, 2, 32)
    assert data["w1_qt"].dtype == dtypes.fp8
    assert data["w2_qt"].dtype == dtypes.fp8
    assert data["w1_scale"].dtype == dtypes.fp8_e8m0
    assert data["w2_scale"].dtype == dtypes.fp8_e8m0
    assert inputs["w2u8"].numel() == data["w2_qt"].numel()
