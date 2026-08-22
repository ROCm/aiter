# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import (
    _PREPARED_STAGE1_SCALE_LAYOUT,
    MOEMetadata,
    _is_prepared_stage1_input,
)


def _accepts_prepared_input(
    *,
    prequant=True,
    input_dtype=dtypes.fp8,
    q_dtype=dtypes.fp8,
    hidden_cols=32,
    scale_dtype=dtypes.fp8_e8m0,
    scale_shape=(32, 1),
    sorted_shape=(32,),
    scale_layout=_PREPARED_STAGE1_SCALE_LAYOUT,
    expected_sorted_blocks=None,
):
    metadata = MOEMetadata(
        stage1=None,
        stage2=None,
        block_m=32,
        ksplit=0,
        prequant=prequant,
        expected_sorted_blocks=expected_sorted_blocks,
    )
    scale = (
        torch.empty(scale_shape, dtype=scale_dtype) if scale_dtype is not None else None
    )
    return _is_prepared_stage1_input(
        metadata,
        torch.empty((3, hidden_cols), dtype=input_dtype),
        q_dtype,
        scale,
        torch.empty(sorted_shape, dtype=torch.int32),
        scale_layout,
    )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        pytest.param({}, True, id="valid"),
        pytest.param({"prequant": False}, False, id="not-prequant"),
        pytest.param({"input_dtype": torch.bfloat16}, False, id="bf16-input"),
        pytest.param({"q_dtype": torch.bfloat16}, False, id="bf16-q-dtype"),
        pytest.param({"scale_dtype": None}, False, id="no-scale"),
        pytest.param({"scale_dtype": torch.uint8}, False, id="wrong-scale-dtype"),
        pytest.param({"hidden_cols": 64}, False, id="wrong-scale-cols"),
        pytest.param({"scale_layout": None}, False, id="missing-layout"),
        pytest.param({"scale_layout": "token_major"}, False, id="wrong-layout"),
        pytest.param({"scale_shape": (31, 1)}, False, id="undersized-scale"),
        pytest.param({"scale_shape": (3, 1)}, False, id="token-major-scale"),
        pytest.param({"scale_shape": (64, 1)}, True, id="extra-scale-rows"),
        pytest.param({"sorted_shape": (8, 4)}, False, id="non-vector-sorted-ids"),
        pytest.param(
            {"scale_shape": (32, 1), "expected_sorted_blocks": 2},
            False,
            id="undersized-metadata-extent",
        ),
        pytest.param(
            {"scale_shape": (64, 1), "expected_sorted_blocks": 2},
            True,
            id="metadata-extent",
        ),
    ],
)
def test_prepared_stage1_input_gate(overrides, expected):
    assert _accepts_prepared_input(**overrides) is expected
