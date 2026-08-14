# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import replace

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import MOEMetadata, _is_prepared_stage1_input


@pytest.mark.parametrize(
    (
        "prequant",
        "input_dtype",
        "q_dtype",
        "hidden_cols",
        "scale_dtype",
        "scale_cols",
        "expected",
    ),
    [
        (True, dtypes.fp8, dtypes.fp8, 32, dtypes.fp8_e8m0, 1, True),
        (False, dtypes.fp8, dtypes.fp8, 32, dtypes.fp8_e8m0, 1, False),
        (True, torch.bfloat16, dtypes.fp8, 32, dtypes.fp8_e8m0, 1, False),
        (True, torch.bfloat16, torch.bfloat16, 32, dtypes.fp8_e8m0, 1, False),
        (True, dtypes.fp8, dtypes.fp8, 32, None, 0, False),
        (True, dtypes.fp8, dtypes.fp8, 32, torch.uint8, 1, False),
        (True, dtypes.fp8, dtypes.fp8, 32, torch.float32, 1, False),
        (True, dtypes.fp8, dtypes.fp8, 64, dtypes.fp8_e8m0, 1, False),
    ],
)
def test_prepared_stage1_input_gate(
    prequant: bool,
    input_dtype: torch.dtype,
    q_dtype: torch.dtype,
    hidden_cols: int,
    scale_dtype: torch.dtype | None,
    scale_cols: int,
    expected: bool,
):
    metadata = replace(
        MOEMetadata(stage1=None, stage2=None, block_m=32, ksplit=0),
        prequant=prequant,
    )
    hidden_states = torch.empty((3, hidden_cols), dtype=input_dtype)
    scale = (
        torch.empty((3, scale_cols), dtype=scale_dtype)
        if scale_dtype is not None
        else None
    )

    assert (
        _is_prepared_stage1_input(metadata, hidden_states, q_dtype, scale) is expected
    )
