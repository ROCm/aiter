# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    mha_v4,
    mha_v4_packed,
)


def test_attention_format_ids_are_stable():
    assert int(AttentionFormat.FP32) == 0
    assert int(AttentionFormat.FP16) == 1
    assert int(AttentionFormat.BF16) == 2
    assert int(AttentionFormat.FP8_E4M3) == 3
    assert AttentionFormat.FP8 is AttentionFormat.FP8_E4M3
    assert int(AttentionFormat.FP8_E4M3_FNUZ) == 4
    assert int(AttentionFormat.FP8_E5M2) == 5
    assert int(AttentionFormat.FP8_E5M2_FNUZ) == 6
    assert int(AttentionFormat.FP6_E2M3) == 7
    assert AttentionFormat.MXFP6 is AttentionFormat.FP6_E2M3
    assert int(AttentionFormat.FP6_E3M2) == 8
    assert AttentionFormat.MXBF6 is AttentionFormat.FP6_E3M2
    assert int(AttentionFormat.FP4_E2M1) == 9
    assert AttentionFormat.MXFP4 is AttentionFormat.FP4_E2M1
    assert int(AttentionFormat.INT8) == 10
    assert int(AttentionFormat.UINT8) == 11
    assert int(AttentionFormat.INT4) == 12
    assert int(AttentionFormat.UINT4) == 13


def test_mha_v4_rejects_unsupported_contracts():
    q = torch.empty((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="do not produce LSE"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            return_lse=True,
        )
    with pytest.raises(ValueError, match="matching Q and K formats"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
        )


@pytest.mark.parametrize(
    "q_format",
    [
        AttentionFormat.FP16,
        AttentionFormat.FP8_E5M2,
        AttentionFormat.FP8_E5M2_FNUZ,
        AttentionFormat.UINT8,
        AttentionFormat.INT4,
        AttentionFormat.UINT4,
    ],
)
def test_mha_v4_rejects_reserved_raw_formats(q_format):
    q = torch.empty((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises((ValueError, NotImplementedError)):
        mha_v4(
            q,
            q,
            q,
            q_format,
            q_format,
            AttentionFormat.FP8,
        )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
def test_mha_v4_packed_rejects_wrong_scale_recipe():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.int8)
    v = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="unsupported scale recipe"):
        mha_v4_packed(
            q,
            q,
            v,
            scale,
            scale,
            scale,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.F32_PER_CHANNEL,
        )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
def test_mha_v4_packed_rejects_wrong_fp8_encoding():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must be FP8 E4M3 FNUZ"):
        mha_v4_packed(
            q,
            q,
            q,
            scale,
            scale,
            scale,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
        )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
@pytest.mark.parametrize("q_format", [AttentionFormat.INT8, AttentionFormat.FP8])
def test_mha_v4_zero_inputs_are_finite(q_format):
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    out = mha_v4(q, q, q, q_format, q_format, AttentionFormat.FP8)
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
    assert torch.isfinite(out).all()


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
def test_mha_v4_packed_i8fp8_compile_parity():
    torch.manual_seed(17)
    q = torch.randint(-32, 33, (1, 512, 5, 128), device="cuda", dtype=torch.int8)
    k = torch.randint(-32, 33, (1, 512, 5, 128), device="cuda", dtype=torch.int8)
    v = torch.randn((1, 512, 5, 128), device="cuda").to(torch.float8_e4m3fn)
    q_descale = torch.tensor([0.02], device="cuda")
    k_descale = torch.tensor([0.03], device="cuda")
    v_descale = torch.tensor([0.04], device="cuda")
    scale = 128**-0.5

    eager = mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.INT8,
        AttentionFormat.INT8,
        AttentionFormat.FP8,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=scale,
    )
    compiled = torch.compile(mha_v4_packed, fullgraph=True)(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.INT8,
        AttentionFormat.INT8,
        AttentionFormat.FP8,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(eager, compiled)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        (AttentionFormat.INT8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.FP8),
        (AttentionFormat.MXFP4, AttentionFormat.FP8),
        (AttentionFormat.MXFP4, AttentionFormat.MXFP4),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.FP8),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP4),
    ],
)
def test_mha_v4_raw_compile_parity(q_format, v_format):
    torch.manual_seed(31)
    q = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    eager_out = torch.empty_like(q)
    compiled_out = torch.empty_like(q)

    eager = mha_v4(q, k, v, q_format, q_format, v_format, out=eager_out)
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q, k, v, q_format, q_format, v_format, out=compiled_out
    )
    churn = torch.empty((16 * 1024 * 1024,), device="cuda", dtype=torch.uint8)
    consumed = compiled.contiguous()
    torch.cuda.synchronize()

    assert eager.data_ptr() == eager_out.data_ptr()
    assert compiled.data_ptr() == compiled_out.data_ptr()
    assert torch.equal(eager, compiled)
    assert torch.isfinite(consumed).all()
    assert churn.numel() == 16 * 1024 * 1024
