# SPDX-License-Identifier: MIT
"""CPU-only API and dispatch coverage for fused AR + RMSNorm + MXFP8."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from aiter.dist import communication_op
from aiter.dist.communication_op import (
    _canonical_fused_ar_rms_quant_type,
    _normalize_fused_ar_rms_quant_type,
)
from aiter.dist.parallel_state import fused_allreduce_rmsnorm_mxfp8_quant_fake
from aiter.ops.enum import QuantType
from aiter.utility.dtypes import fp8, fp8_e8m0


@pytest.mark.parametrize(
    "alias",
    ["mxfp8", "MXFP8", "mxfp8_per_1x32", "per_1x32_mxfp8"],
)
def test_mxfp8_aliases_normalize(alias):
    assert _normalize_fused_ar_rms_quant_type(alias) == "mxfp8"
    assert _canonical_fused_ar_rms_quant_type(alias) == "mxfp8"


def test_per_1x32_enum_keeps_its_existing_mxfp4_meaning():
    assert _canonical_fused_ar_rms_quant_type(QuantType.per_1x32) == "mxfp4"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_size": 16}, "group_size=32"),
        ({"emit_bf16": True}, "does not support emit_bf16"),
        ({"transpose_scale": True}, "does not support emit_bf16"),
        ({"hadamard_rotate": True}, "MXFP4-only"),
    ],
)
def test_mxfp8_public_dispatch_rejects_incompatible_options(kwargs, message):
    tensor = torch.empty((2, 32), dtype=torch.bfloat16)
    options = {"group_size": 32, **kwargs}
    with (
        patch.object(communication_op, "has_custom_group", return_value=False),
        patch.object(communication_op, "get_tp_group") as get_tp_group,
        pytest.raises(ValueError, match=message),
    ):
        communication_op.tensor_model_parallel_fused_allreduce_rmsnorm_quant(
            tensor,
            tensor,
            tensor[0],
            1e-6,
            quant_type="mxfp8",
            **options,
        )
    get_tp_group.assert_not_called()


def test_mxfp8_helper_returns_the_group_result():
    tensor = torch.empty((2, 32), dtype=torch.bfloat16)
    result = (object(), object(), object())
    group = SimpleNamespace(fused_allreduce_rmsnorm_quant=Mock(return_value=result))
    with (
        patch.object(communication_op, "has_custom_group", return_value=False),
        patch.object(communication_op, "get_tp_group", return_value=group),
    ):
        actual = (
            communication_op.tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant(
                tensor,
                tensor,
                tensor[0],
                1e-6,
                prefill_support=True,
                gemma_norm=True,
            )
        )

    assert actual is result
    group.fused_allreduce_rmsnorm_quant.assert_called_once_with(
        tensor,
        tensor,
        tensor[0],
        1e-6,
        True,
        quant_type="mxfp8",
        group_size=32,
        emit_bf16=False,
        transpose_scale=False,
        gemma_norm=True,
        hadamard_rotate=False,
    )


def test_mxfp8_fake_declares_native_value_and_scale_layouts():
    tensor = torch.empty((3, 64), dtype=torch.bfloat16)
    values, residual, scales = fused_allreduce_rmsnorm_mxfp8_quant_fake(
        tensor,
        tensor,
        tensor[0],
        1e-6,
        "tp",
    )

    assert values.shape == tensor.shape
    assert values.dtype == fp8
    assert residual.shape == tensor.shape
    assert residual.dtype == tensor.dtype
    assert scales.shape == (3, 2)
    assert scales.dtype == fp8_e8m0
