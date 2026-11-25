# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.flydsl import gemm_a16w16_gfx1250 as gemm


def test_split_cat_is_opt_in():
    meta = {"dtype": torch.bfloat16, "device": "meta"}
    x, w = torch.empty(31, 512, **meta), torch.empty(128 * 256, 512, **meta)
    k_out = torch.empty_strided((31, 128, 192), (128 * 200 + 16, 200, 1), **meta)
    v_out, rope = torch.empty(31, 128, 128, **meta), torch.empty(31, 1, 64, **meta)
    split = (x, w, torch.bfloat16, None)
    # no K/V outputs: None keeps the plain GEMM launcher
    assert gemm._validate_split_cat(*split, None, None, None, None) is None
    spec = gemm._validate_split_cat(*split, k_out, v_out, rope, (128, 128))
    assert (spec["heads"], spec["rope_size"], spec["k_head_stride"]) == (128, 64, 200)
    with pytest.raises(ValueError, match="provided together"):
        gemm._validate_split_cat(*split, k_out, None, rope, (128, 128))
