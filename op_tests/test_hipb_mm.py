# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.gradlib import HipblasLtMatmulScaleMode as ScaleMode
from aiter.test_common import checkAllclose

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.fixture(scope="module", autouse=True)
def hipb_extension():
    aiter.hipb_create_extension()
    yield
    aiter.hipb_destroy_extension()


@pytest.mark.parametrize("rows", [1, 32])
@pytest.mark.parametrize("mode", [ScaleMode.SCALAR_32F, ScaleMode.OUTER_VEC_32F])
@pytest.mark.parametrize("explicit", [(), ("A",), ("B",), ("A", "B")])
def test_hipb_mm_scale_mode_inference(rows, mode, explicit):
    x = torch.randint(-2, 3, (rows, 256), device="cuda").to(dtypes.fp8)
    w = torch.randint(-2, 3, (128, 256), device="cuda").to(dtypes.fp8)
    scale_a = torch.full((1, 1), 0.25, device="cuda")
    scale_b = torch.full((1, 1), 2.0, device="cuda")
    if mode == ScaleMode.OUTER_VEC_32F:
        scale_a = torch.exp2(torch.arange(rows, device="cuda").float() % 3 - 1)[:, None]
        scale_b = torch.exp2(torch.arange(128, device="cuda").float() % 3 - 1)[None, :]
    modes = {f"scale{operand}_mode": mode for operand in explicit}
    out = aiter.hipb_mm(
        x,
        w.t(),
        solution_index=-1,
        out_dtype=dtypes.bf16,
        scaleA=scale_a,
        scaleB=scale_b,
        **modes,
    )
    ref = ((x.float() * scale_a) @ (w.float().t() * scale_b)).to(dtypes.bf16)
    # A wrong inferred mode can launch successfully but apply the wrong scale factors.
    err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, catastrophic_check=True)
    assert err <= 0.05


@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize(
    "rows, scale_a_shape, scale_b_shape, provided",
    [
        (1, (1, 1), (1, 1), ScaleMode.OUTER_VEC_32F),
        (32, (1, 1), (1, 1), ScaleMode.OUTER_VEC_32F),
        (1, (1, 1), (1, 128), ScaleMode.SCALAR_32F),
        (32, (32, 1), (1, 128), ScaleMode.SCALAR_32F),
    ],
    ids=["scalar-1-row", "scalar-32-rows", "rowwise-1-row", "rowwise-32-rows"],
)
def test_hipb_mm_rejects_scale_mode_conflict(
    operand, rows, scale_a_shape, scale_b_shape, provided
):
    x = torch.zeros((rows, 256), device="cuda", dtype=dtypes.fp8)
    w = torch.zeros((128, 256), device="cuda", dtype=dtypes.fp8)
    scale_a = torch.ones(scale_a_shape, device="cuda")
    scale_b = torch.ones(scale_b_shape, device="cuda")
    with pytest.raises(
        RuntimeError, match=f"scale{operand}_mode disagrees with inferred"
    ):
        aiter.hipb_mm(
            x,
            w.t(),
            solution_index=-1,
            out_dtype=dtypes.bf16,
            scaleA=scale_a,
            scaleB=scale_b,
            **{f"scale{operand}_mode": provided},
        )


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16, dtypes.fp32])
def test_hipb_mm_without_scales(dtype):
    x = torch.zeros((32, 256), device="cuda", dtype=dtype)
    w = torch.zeros((128, 256), device="cuda", dtype=dtype)
    aiter.hipb_mm(x, w.t(), solution_index=-1, out_dtype=dtype)


@pytest.mark.parametrize("operand", ["A", "B"])
def test_hipb_mm_single_scale(operand):
    x = torch.randint(-2, 3, (32, 256), device="cuda").to(dtypes.fp8)
    w = torch.randint(-2, 3, (128, 256), device="cuda").to(dtypes.fp8)
    scale = torch.full((1, 1), 0.5, device="cuda")
    out = aiter.hipb_mm(
        x,
        w.t(),
        solution_index=-1,
        out_dtype=dtypes.bf16,
        **{f"scale{operand}": scale},
    )
    ref = ((x.float() @ w.float().t()) * scale).to(dtypes.bf16)
    err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, catastrophic_check=True)
    assert err <= 0.05


@pytest.mark.parametrize(
    "mode_a, mode_b, scale_a_shape, scale_b_shape, scale_dtype",
    [
        (
            ScaleMode.SCALAR_32F,
            ScaleMode.OUTER_VEC_32F,
            (1, 1),
            (1, 128),
            torch.float32,
        ),
        (ScaleMode.VEC32_UE8M0, ScaleMode.VEC32_UE8M0, (32, 8), (128, 8), torch.uint8),
    ],
    ids=["mixed", "block"],
)
def test_hipb_mm_rejects_unrepresentable_tuning_modes(
    monkeypatch, tmp_path, mode_a, mode_b, scale_a_shape, scale_b_shape, scale_dtype
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIP_ONLINE_TUNING", "1")
    x = torch.zeros((32, 256), device="cuda", dtype=dtypes.fp8)
    w = torch.zeros((128, 256), device="cuda", dtype=dtypes.fp8)
    scale_a = torch.ones(scale_a_shape, dtype=scale_dtype, device="cuda")
    scale_b = torch.ones(scale_b_shape, dtype=scale_dtype, device="cuda")
    with pytest.raises(
        RuntimeError,
        match="HIP_ONLINE_TUNING only supports per-tensor or per-row scaling modes",
    ):
        aiter.hipb_mm(
            x,
            w.t(),
            solution_index=-1,
            out_dtype=dtypes.bf16,
            scaleA=scale_a,
            scaleB=scale_b,
            scaleA_mode=mode_a,
            scaleB_mode=mode_b,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
