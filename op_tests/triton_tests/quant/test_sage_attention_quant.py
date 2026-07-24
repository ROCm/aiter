# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

import aiter
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant,
    sage_quant_mxfp4,
)


def _make_inputs(layout: str):
    torch.manual_seed(20)
    shape = (1, 64, 2, 128) if layout == "bshd" else (1, 2, 64, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    if layout == "bshd":
        v[:, :, 0, 0] = 0
        v[:, :, 1, :] = 0
    else:
        v[:, 0, :, 0] = 0
        v[:, 1, :, :] = 0
    return q, k, v


def _expected_v_quant(v: torch.Tensor, layout: str):
    fp8_dtype = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8_dtype).max
    sequence_dim = 1 if layout == "bshd" else 2
    scales = v.abs().amax(dim=sequence_dim).to(torch.float32) / fp8_max
    safe_scales = torch.where(scales == 0, 1.0, scales)
    broadcast_dim = sequence_dim
    quantized = (v / safe_scales.unsqueeze(broadcast_dim)).to(fp8_dtype)
    return quantized, scales


@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("quantizer", ["sage_v1", "sage_mxfp4"])
def test_sage_v_quantization_defines_zero_scale_channels(layout, quantizer):
    if quantizer == "sage_mxfp4" and not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")

    q, k, v = _make_inputs(layout)
    fp8_dtype = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8_dtype).max

    if quantizer == "sage_v1":
        result = sage_quant(
            q,
            k,
            v,
            fp8_dtype,
            fp8_max,
            BLKQ=64,
            BLKK=64,
            layout=layout,
            smooth_k=False,
        )
    else:
        result = sage_quant_mxfp4(
            q,
            k,
            v,
            fp8_dtype,
            fp8_max,
            BLKQ=64,
            BLKK=64,
            layout=layout,
            smooth_k=False,
        )

    v_fp8, v_scale = result[4:6]
    expected_v_fp8, expected_v_scale = _expected_v_quant(v, layout)

    torch.testing.assert_close(v_scale, expected_v_scale, rtol=0, atol=0)
    torch.testing.assert_close(v_fp8.float(), expected_v_fp8.float(), rtol=0, atol=0)
    assert torch.isfinite(v_fp8.float()).all()
    assert torch.isfinite(v_scale).all()
    assert torch.isfinite(
        v_fp8.float() * v_scale.unsqueeze(1 if layout == "bshd" else 2)
    ).all()


@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
def test_sage_attention_is_finite_with_zero_v_scale_channels(layout):
    q, k, v = _make_inputs(layout)
    output = fav3_sage_wrapper_func(
        q,
        k,
        v,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        causal=False,
        return_lse=False,
        layout=layout,
        smooth_k=False,
    )

    assert torch.isfinite(output).all()
    zero_head = output[:, :, 1, :] if layout == "bshd" else output[:, 1, :, :]
    torch.testing.assert_close(zero_head, torch.zeros_like(zero_head))
