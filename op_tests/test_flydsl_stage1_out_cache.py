# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.fused_moe import (
    _FLYDSL_STAGE1_OUT_CACHE,
    _get_flydsl_stage1_out,
)


@pytest.fixture(autouse=True)
def clear_stage1_out_cache():
    _FLYDSL_STAGE1_OUT_CACHE.clear()
    yield
    _FLYDSL_STAGE1_OUT_CACHE.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_flydsl_stage1_out_is_reused_per_stream():
    device = torch.device("cuda:0")
    shape = (1024, 1536)

    output = _get_flydsl_stage1_out(shape, device)
    reused = _get_flydsl_stage1_out(shape, device)

    other_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(other_stream):
        other_stream_output = _get_flydsl_stage1_out(shape, device)

    assert reused.data_ptr() == output.data_ptr()
    assert other_stream_output.data_ptr() != output.data_ptr()
    assert len(_FLYDSL_STAGE1_OUT_CACHE) == 2
