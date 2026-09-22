# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerical and replay contracts for native group32 FP8 projections."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32 import (
    gemm_a8w8_blockscale_group32,
)
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.test_common import checkAllclose

pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="CDNA4 packed MFMA")


def _operands(m, n, k):
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    xs = torch.randint(122, 131, (m, k // 32), device="cuda", dtype=torch.uint8)
    ws = torch.randint(
        122, 131, ((n + 31) // 32, k // 32), device="cuda", dtype=torch.uint8
    )
    return x, weight, xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu)


def _reference(x, weight, xs, ws):
    # FP64 is independent of the MFMA's internal block accumulation and of
    # either implementation's K reduction order.
    a = x.double() * xs.double().repeat_interleave(32, -1)
    b = weight.double() * ws.double().repeat_interleave(32, 0)[
        : weight.shape[0]
    ].repeat_interleave(32, -1)
    return a @ b.T


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 5120, 576),
        (3, 2053, 1280),
        (4, 8192, 1280),
        (8, 5120, 1152),
        (16, 4096, 1280),
        (31, 4096, 1280),
        (3, 5120, 2048),
        (8, 5120, 2304),
        (4, 5120, 4096),
        (3, 2304, 5120),
        (1, 5120, 8192),
        # Untuned geometries exercise the default configuration with tails.
        (63, 8193, 1280),
        (129, 4097, 576),
        (255, 16385, 1152),
        (1023, 4097, 1280),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_group32_projection_panel_scales_and_tails(m, n, k, dtype):
    x, weight, xs, ws = _operands(m, n, k)
    actual = gemm_a8w8_blockscale_group32(x, weight, xs, ws, dtype=dtype)
    expected = _reference(x, weight, xs, ws)
    peak = expected.abs().max().item()
    # Keep the established native-MFMA FP32 bound. BF16 additionally rounds
    # output; near cancellation still uses the same peak-relative floor.
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=(
            0.016
            if dtype == torch.bfloat16
            else 0.002 if dtype == torch.float16 else 3e-5
        ),
        atol=5e-5 * peak,
    )


@pytest.mark.parametrize("m,packed", [(3, True), (63, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tuned_variants_with_column_and_k_tails(m, packed, dtype):
    # Reuse a measured tile with irregular N and K to exercise masks in both
    # optimized variants; an untuned geometry alone would use DEFAULT.json.
    config, tuned = get_gemm_config("GEMM-A8W8_BLOCKSCALE_GROUP32", m, 8192, 1280)
    assert tuned
    assert ("packed" in config) == packed
    if not packed:
        assert config["N_FIRST"]
    x, w, xs, ws = _operands(m, 8193, 1312)
    actual = gemm_a8w8_blockscale_group32(x, w, xs, ws, dtype=dtype, config=config)
    expected = _reference(x, w, xs, ws)
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=(
            0.016
            if dtype == torch.bfloat16
            else 0.002 if dtype == torch.float16 else 3e-5
        ),
        atol=5e-5 * expected.abs().max().item(),
    )


@pytest.mark.parametrize(
    "a_code,b_code", [(0, 254), (254, 0), (128, 0), (255, 127), (127, 255)]
)
def test_group32_projection_extreme_scale_codes(a_code, b_code):
    x = torch.ones(3, 1280, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.ones(2048, 1280, device="cuda").to(torch.float8_e4m3fn)
    xs = torch.full((3, 40), a_code, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    ws = torch.full((64, 40), b_code, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    actual = gemm_a8w8_blockscale_group32(x, weight, xs, ws, dtype=torch.float32)
    if 255 in (a_code, b_code):
        assert actual.isnan().all()
    else:
        expected = torch.full_like(actual, 1280 * 2.0 ** (a_code + b_code - 254))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,n,k,split_k", [(3, 4096, 1280, None), (33, 8193, 576, 3)]
)
def test_group32_projection_graph_reads_live_inputs_and_scales(rows, n, k, split_k):
    x, weight, xs, ws = _operands(2 * rows, n, k)
    gemm_a8w8_blockscale_group32(
        x, weight, xs, ws, dtype=torch.float32, split_k=split_k
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gemm_a8w8_blockscale_group32(
            x, weight, xs, ws, dtype=torch.float32, split_k=split_k
        )
    for factor in (2, 0.5):
        x.copy_((x.float() * factor).to(x.dtype))
        xs.view(torch.uint8).add_(1)
        graph.replay()
        expected = _reference(x, weight, xs, ws)
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=3e-5,
            atol=5e-5 * expected.abs().max().item(),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_row_scaled_fp8_projection_with_token_and_column_tails(dtype):
    x, weight, xs, ws = _operands(97, 8193, 576)
    # Per-row scales share the scheduling kernel but not the compact weight grid.
    ws = ws.repeat_interleave(32, 0)[: weight.shape[0]].contiguous()
    actual = gemm_a8w8_blockscale_group32(
        x, weight, xs, ws, weight_group_rows=1, dtype=dtype, split_k=3
    )
    expected = (x.double() * xs.double().repeat_interleave(32, -1)) @ (
        weight.double() * ws.double().repeat_interleave(32, -1)
    ).T
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=(
            0.016
            if dtype == torch.bfloat16
            else 0.002 if dtype == torch.float16 else 3e-5
        ),
        atol=5e-5 * expected.abs().max().item(),
    )


@pytest.mark.parametrize(
    "m,n,k", [(0, 65, 64), (3, 2053, 1280), (66, 8193, 576), (512, 4096, 1280)]
)
@pytest.mark.parametrize("group_n", [1, 32])
def test_public_group32_dispatch(m, n, k, group_n):
    x, w, xs, ws = _operands(m, n, k)
    if group_n == 1:
        ws = ws.repeat_interleave(32, 0)[:n].contiguous()
    expected = gemm_a8w8_blockscale_group32(x, w, xs, ws, weight_group_rows=group_n)
    actual = gemm_a8w8_blockscale(x, w, xs, ws)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("split_k", [None, 1, 3, 100])
def test_raw_views_and_preallocated_output(split_k):
    x, w, xs, ws = _operands(63, 8193, 576)
    output = torch.empty((63, 8193), dtype=torch.bfloat16, device="cuda")
    actual = gemm_a8w8_blockscale_group32(
        x.view(torch.uint8),
        w.view(torch.uint8),
        xs.view(torch.uint8),
        ws.view(torch.uint8),
        y=output,
        split_k=split_k,
    )
    assert actual is output
    expected = _reference(x, w, xs, ws).to(output.dtype)
    checkAllclose(
        expected,
        actual,
        rtol=0.016,
        atol=5e-5 * expected.abs().max().item(),
        catastrophic_check=True,
    )


@pytest.mark.parametrize(
    "invalid", ["stride", "scale_shape", "scale_dtype", "output", "split"]
)
def test_invalid_group32_contract(invalid):
    x, w, xs, ws = _operands(3, 2053, 1280)
    kwargs = {}
    if invalid == "stride":
        x = x.T.contiguous().T
    elif invalid == "scale_shape":
        ws = ws[:-1]
    elif invalid == "scale_dtype":
        xs = xs.float()
    elif invalid == "output":
        kwargs["y"] = torch.empty((3, 2052), dtype=torch.bfloat16, device="cuda")
    else:
        kwargs["split_k"] = 0
    with pytest.raises(AssertionError):
        gemm_a8w8_blockscale_group32(x, w, xs, ws, **kwargs)


def test_public_group32_compile_dynamic_rows():
    def forward(x, w, xs, ws):
        return gemm_a8w8_blockscale(x, w, xs, ws)

    compiled = torch.compile(forward, fullgraph=True, dynamic=True)
    for m in (3, 7, 65):
        x, w, xs, ws = _operands(m, 4096, 1280)
        torch.testing.assert_close(
            compiled(x, w, xs, ws), forward(x, w, xs, ws), rtol=0, atol=0
        )
