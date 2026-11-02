# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import get_module
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gradlib import HipblasLtMatmulScaleMode as ScaleMode
from aiter.test_common import assertAllclose

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
requires_fp8 = pytest.mark.skipif(
    dtypes.fp8 == dtypes.u8, reason="FP8 is not supported on this device"
)
requires_mxfp8 = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="MXFP8 hipb_mm requires gfx950"
)


@pytest.fixture(scope="module", autouse=True)
def hipb_extension():
    aiter.hipb_create_extension()
    yield
    aiter.hipb_destroy_extension()


@requires_fp8
@pytest.mark.parametrize("rows", [1, 32])
@pytest.mark.parametrize("mode", [ScaleMode.SCALAR_32F, ScaleMode.OUTER_VEC_32F])
@pytest.mark.parametrize("explicit", [(), ("A",), ("B",), ("A", "B")])
@pytest.mark.parametrize("use_findallsols", [False, True])
def test_hipb_mm_scale_mode_inference(rows, mode, explicit, use_findallsols: bool):
    x = torch.randint(-2, 3, (rows, 256), device="cuda").to(dtypes.fp8)
    w = torch.randint(-2, 3, (128, 256), device="cuda").to(dtypes.fp8)
    scale_a = torch.full((1, 1), 0.25, device="cuda")
    scale_b = torch.full((1, 1), 2.0, device="cuda")
    if mode == ScaleMode.OUTER_VEC_32F:
        scale_a = torch.exp2(torch.arange(rows, device="cuda").float() % 3 - 1)[:, None]
        scale_b = torch.exp2(torch.arange(128, device="cuda").float() % 3 - 1)[None, :]
    modes = {f"scale{operand}_mode": mode for operand in explicit}
    solution_index = -1
    if use_findallsols:
        solutions = aiter.hipb_findallsols(
            x,
            w.t(),
            out_dtype=dtypes.bf16,
            scaleA=scale_a,
            scaleB=scale_b,
            **modes,
        )
        assert solutions, "hipb_findallsols returned no solutions"
        solution_index = solutions[0]
    out = aiter.hipb_mm(
        x,
        w.t(),
        solution_index=solution_index,
        out_dtype=dtypes.bf16,
        scaleA=scale_a,
        scaleB=scale_b,
        **modes,
    )
    ref = ((x.float() * scale_a) @ (w.float().t() * scale_b)).to(dtypes.bf16)
    # A wrong inferred mode can launch successfully but apply the wrong scale factors.
    assertAllclose(ref, out, rtol=1e-2, atol=1e-2, catastrophic_check=True)


@requires_fp8
@pytest.mark.parametrize(
    "rows, scale_a_shape, scale_b_shape, provided, operand",
    [
        (1, (1, 1), (1, 1), ScaleMode.OUTER_VEC_32F, "A"),
        (32, (1, 1), (1, 1), ScaleMode.OUTER_VEC_32F, "B"),
        (1, (1, 1), (1, 128), ScaleMode.SCALAR_32F, "A"),
        (32, (32, 1), (1, 128), ScaleMode.SCALAR_32F, "B"),
    ],
    ids=["scalar-1-row", "scalar-32-rows", "rowwise-1-row", "rowwise-32-rows"],
)
@pytest.mark.parametrize(
    "op, op_kwargs",
    [
        pytest.param(aiter.hipb_mm, {"solution_index": -1}, id="mm"),
        pytest.param(aiter.hipb_findallsols, {}, id="findallsols"),
    ],
)
def test_hipb_mm_rejects_scale_mode_conflict(
    rows, scale_a_shape, scale_b_shape, provided, operand, op, op_kwargs
):
    x = torch.zeros((rows, 256), device="cuda", dtype=dtypes.fp8)
    w = torch.zeros((128, 256), device="cuda", dtype=dtypes.fp8)
    scale_a = torch.ones(scale_a_shape, device="cuda")
    scale_b = torch.ones(scale_b_shape, device="cuda")
    with pytest.raises(
        RuntimeError, match=f"scale{operand}_mode disagrees with inferred"
    ):
        op(
            x,
            w.t(),
            out_dtype=dtypes.bf16,
            scaleA=scale_a,
            scaleB=scale_b,
            **op_kwargs,
            **{f"scale{operand}_mode": provided},
        )


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16, dtypes.fp32])
def test_hipb_mm_without_scales(dtype):
    x = torch.zeros((32, 256), device="cuda", dtype=dtype)
    w = torch.zeros((128, 256), device="cuda", dtype=dtype)
    aiter.hipb_mm(x, w.t(), solution_index=-1, out_dtype=dtype)


@requires_fp8
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
    assertAllclose(ref, out, rtol=1e-2, atol=1e-2, catastrophic_check=True)


@requires_fp8
@requires_mxfp8
@pytest.mark.parametrize(
    "mode_a, mode_b, scale_a_shape, scale_b_shape, scale_dtype",
    [
        (ScaleMode.VEC32_UE8M0, ScaleMode.VEC32_UE8M0, (33, 8), (128, 8), torch.uint8),
        (
            ScaleMode.VEC32_UE8M0,
            ScaleMode.VEC32_UE8M0,
            (32, 8),
            (129, 8),
            dtypes.fp8_e8m0,
        ),
    ],
    ids=["block", "block-e8m0"],
)
def test_hipb_mm_falls_back_for_unrepresentable_tuning_modes(
    monkeypatch, tmp_path, mode_a, mode_b, scale_a_shape, scale_b_shape, scale_dtype
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIP_ONLINE_TUNING", "1")
    x = torch.zeros((32, 256), device="cuda", dtype=dtypes.fp8)
    w = torch.zeros((128, 256), device="cuda", dtype=dtypes.fp8)
    scale_a = torch.ones(scale_a_shape, dtype=scale_dtype, device="cuda")
    scale_b = torch.ones(scale_b_shape, dtype=scale_dtype, device="cuda")
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
    assert not (tmp_path / "hip_online_tuning_res.csv").exists()


@requires_fp8
@pytest.mark.parametrize(
    "mode, scale_a_shape, scale_b_shape, operand",
    [
        (ScaleMode.BLK32_UE8M0_32_8_EXT, (63, 24), (128, 24), "A"),
        (ScaleMode.BLK32_UE8M0_32_8_EXT, (64, 24), (128, 32), "B"),
        (ScaleMode.VEC32_UE8M0, (32, 24), (128, 24), "A"),
    ],
    ids=[
        "short-rows-A",
        "extra-columns-B",
        "unswizzled-short-rows-A",
    ],
)
@pytest.mark.parametrize(
    "op, op_kwargs",
    [
        pytest.param(aiter.hipb_mm, {"solution_index": -1}, id="mm"),
        pytest.param(aiter.hipb_findallsols, {}, id="findallsols"),
    ],
)
def test_hipb_mm_rejects_invalid_scale_shape(
    mode, scale_a_shape, scale_b_shape, operand, op, op_kwargs
):
    if mode == ScaleMode.BLK32_UE8M0_32_8_EXT and get_module(
        "module_hipbsolgemm"
    ).hipblaslt_build_version < (1, 3, 0):
        pytest.skip("Swizzled MXFP8 requires hipBLASLt >=1.3")
    x = torch.zeros((33, 768), device="cuda", dtype=dtypes.fp8)
    w = torch.zeros((128, 768), device="cuda", dtype=dtypes.fp8)
    scale_a = torch.ones(scale_a_shape, device="cuda", dtype=dtypes.u8)
    scale_b = torch.ones(scale_b_shape, device="cuda", dtype=dtypes.u8)
    with pytest.raises(RuntimeError, match=f"scale{operand} must have shape"):
        op(
            x,
            w.t(),
            out_dtype=dtypes.bf16,
            scaleA=scale_a,
            scaleB=scale_b,
            scaleA_mode=mode,
            scaleB_mode=mode,
            **op_kwargs,
        )


@requires_fp8
@requires_mxfp8
@pytest.mark.parametrize(
    "mode", [ScaleMode.VEC32_UE8M0, ScaleMode.BLK32_UE8M0_32_8_EXT]
)
@pytest.mark.parametrize("scale_dtype", [dtypes.u8, dtypes.fp8_e8m0])
def test_hipb_mm_accepts_mxfp8_scale_dtype(mode, scale_dtype):
    if mode == ScaleMode.BLK32_UE8M0_32_8_EXT and get_module(
        "module_hipbsolgemm"
    ).hipblaslt_build_version < (1, 3, 0):
        pytest.skip("Swizzled MXFP8 requires hipBLASLt >=1.3")
    x = torch.zeros((32, 256), device="cuda", dtype=dtypes.fp8)
    w = torch.zeros((128, 256), device="cuda", dtype=dtypes.fp8)
    # E8M0 1.0 has byte value 127; constant scales need no swizzling.
    scale_a = torch.full((32, 8), 127, device="cuda", dtype=dtypes.u8)
    scale_b = torch.full((128, 8), 127, device="cuda", dtype=dtypes.u8)
    aiter.hipb_mm(
        x,
        w.t(),
        solution_index=-1,
        out_dtype=dtypes.bf16,
        scaleA=scale_a.view(scale_dtype),
        scaleB=scale_b.view(scale_dtype),
        scaleA_mode=mode,
        scaleB_mode=mode,
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
