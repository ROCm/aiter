# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect

import pytest
import torch

from aiter.ops.mx_scale_layout import (
    from_mx_scale_layout,
    mx_scale_buffer_shape,
    resolve_mx_scale_layout,
    to_mx_scale_layout,
    validate_mx_scale_buffer,
)
from aiter.ops.shuffle import shuffle_scale, shuffle_scale_f4
from aiter.utility.mx_types import MXScaleLayoutInt


def test_mx_scale_layout_int_values_are_stable():
    assert (
        MXScaleLayoutInt.ROW_MAJOR,
        MXScaleLayoutInt.AITER_E8M0,
        MXScaleLayoutInt.OPUS_F4,
    ) == (0, 1, 2)


@pytest.mark.parametrize("rows", [1, 5, 16, 31, 32, 33, 128, 255, 256, 257])
@pytest.mark.parametrize("k_groups", [1, 4, 5, 8, 9])
@pytest.mark.parametrize(
    "layout",
    [
        MXScaleLayoutInt.ROW_MAJOR,
        MXScaleLayoutInt.AITER_E8M0,
        MXScaleLayoutInt.OPUS_F4,
    ],
)
def test_mx_scale_layout_cpu_roundtrip(rows, k_groups, layout):
    row_major = (
        torch.arange(rows * k_groups, dtype=torch.int64)
        .to(torch.uint8)
        .view(rows, k_groups)
    )
    packed = to_mx_scale_layout(row_major, layout)
    assert tuple(packed.shape) == mx_scale_buffer_shape(rows, k_groups, layout)
    validate_mx_scale_buffer(packed, rows, k_groups, layout)
    torch.testing.assert_close(
        from_mx_scale_layout(packed, rows, k_groups, layout), row_major
    )


def test_layout_helpers_reuse_canonical_shuffles():
    row_major = torch.arange(33 * 9, dtype=torch.int64).to(torch.uint8).view(33, 9)
    assert torch.equal(
        to_mx_scale_layout(row_major, MXScaleLayoutInt.AITER_E8M0),
        shuffle_scale(row_major),
    )
    assert torch.equal(
        to_mx_scale_layout(row_major, MXScaleLayoutInt.OPUS_F4),
        shuffle_scale_f4(row_major, intype=7),
    )


def test_legacy_shuffle_alias_and_conflict():
    assert resolve_mx_scale_layout(None, False) == MXScaleLayoutInt.ROW_MAJOR
    assert resolve_mx_scale_layout(None, True) == MXScaleLayoutInt.AITER_E8M0
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_mx_scale_layout(MXScaleLayoutInt.AITER_E8M0, False)


def test_f4gemm_fake_and_schema_include_layout_contract():
    from aiter.ops import quant as _quant
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4_fake
    from aiter.ops.rmsnorm import add_rmsnorm_quant, rmsnorm_quant

    params = inspect.signature(gemm_a4w4_fake).parameters
    assert params["a_scale_layout"].default is None
    assert params["b_scale_layout"].default is None

    a = torch.empty((5, 128), dtype=torch.uint8, device="meta")
    b = torch.empty((16, 128), dtype=torch.uint8, device="meta")
    a_scale = torch.empty((32, 8), dtype=torch.uint8, device="meta")
    b_scale = torch.empty((32, 8), dtype=torch.uint8, device="meta")
    assert gemm_a4w4_fake(a, b, a_scale, b_scale).shape == (5, 16)
    for layout in (MXScaleLayoutInt.AITER_E8M0, MXScaleLayoutInt.OPUS_F4):
        out = gemm_a4w4_fake(
            a,
            b,
            a_scale,
            b_scale,
            a_scale_layout=layout,
            b_scale_layout=layout,
        )
        assert out.shape == (5, 16)
    with pytest.raises(ValueError, match="unknown MX scale layout"):
        gemm_a4w4_fake(a, b, a_scale, b_scale, a_scale_layout=99)

    assert "scale_layout" in str(torch.ops.aiter.gemm_a4w4.default._schema)
    assert "scale_layout" in str(torch.ops.aiter.gemm_a4w4o8.default._schema)
    assert "scale_layout" in str(torch.ops.aiter.rmsnorm_quant.default._schema)
    assert "scale_layout" in str(torch.ops.aiter.add_rmsnorm_quant.default._schema)
    assert "scale_layout" in str(torch.ops.aiter.quant_mxfp4.default._schema)
    assert rmsnorm_quant is not None and add_rmsnorm_quant is not None
    assert _quant.quant_mxfp4 is not None
