# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerical and replay contracts for native group32 FP8 projections."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import gemm_op_a8w8
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
from aiter.ops.triton.gemm.basic import gemm_afp8wfp8 as afp8wfp8_op
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32 import (
    gemm_a8w8_blockscale_group32,
)
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.test_common import checkAllclose

pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="CDNA4 packed MFMA")


def generate_inputs(m, n, k, weight_group_rows=32):
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    xs = torch.randint(122, 131, (m, k // 32), device="cuda", dtype=torch.uint8)
    ws = torch.randint(
        122,
        131,
        (-(-n // weight_group_rows), k // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    return x, weight, xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu)


def run_torch(x, weight, xs, ws, weight_group_rows=32):
    # FP64 is independent of the MFMA's internal block accumulation and of
    # the kernel's K reduction order.
    a = x.double() * xs.double().repeat_interleave(32, -1)
    b = weight.double() * ws.double().repeat_interleave(weight_group_rows, 0)[
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
    x, weight, xs, ws = generate_inputs(m, n, k)
    actual = gemm_a8w8_blockscale_group32(x, weight, xs, ws, dtype=dtype)
    expected = run_torch(x, weight, xs, ws)
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


@pytest.mark.parametrize("m,packed", [(3, True), (129, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tuned_variants_use_production_config(m, packed, dtype):
    n, k = 8192, 1280
    config, tuned = get_gemm_config("GEMM-AFP8WFP8_A32_W32X32", m, n, k)
    assert tuned
    assert ("packed" in config) == packed
    if not packed:
        assert config["N_FIRST"]
    x, w, xs, ws = generate_inputs(m, n, k)
    actual = gemm_a8w8_blockscale_group32(x, w, xs, ws, dtype=dtype)
    expected = run_torch(x, w, xs, ws)
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
    x, weight, xs, ws = generate_inputs(2 * rows, n, k)
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
        expected = run_torch(x, weight, xs, ws)
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=3e-5,
            atol=5e-5 * expected.abs().max().item(),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_row_scaled_fp8_projection_with_token_and_column_tails(dtype):
    x, weight, xs, ws = generate_inputs(97, 8193, 576)
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
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("raw_views", [False, True])
def test_public_group32_dispatch(m, n, k, group_n, configured, raw_views, monkeypatch):
    if configured:
        monkeypatch.setattr(
            gemm_op_a8w8, "get_CKGEMM_config", lambda *args: {"libtype": "triton"}
        )
    x, w, xs, ws = generate_inputs(m, n, k)
    if group_n == 1:
        ws = ws.repeat_interleave(32, 0)[:n].contiguous()
    expected = gemm_a8w8_blockscale_group32(x, w, xs, ws, weight_group_rows=group_n)
    operands = (x, w, xs, ws)
    if raw_views:
        operands = tuple(t.view(torch.uint8) for t in operands)
    actual = gemm_a8w8_blockscale(*operands)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("split_k", [None, 1, 3, 100])
def test_raw_views_and_preallocated_output(split_k):
    x, w, xs, ws = generate_inputs(63, 8193, 576)
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
    expected = run_torch(x, w, xs, ws).to(output.dtype)
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
    x, w, xs, ws = generate_inputs(3, 2053, 1280)
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


@pytest.mark.parametrize("split_k", [None, 3])
def test_public_group32_compile_dynamic_rows(split_k):
    def forward(x, w, xs, ws):
        return gemm_a8w8_blockscale(x, w, xs, ws, split_k=split_k)

    compiled = torch.compile(forward, fullgraph=True, dynamic=True)
    for m in (3, 7, 65):
        x, w, xs, ws = generate_inputs(m, 4096, 1280)
        torch.testing.assert_close(
            compiled(x, w, xs, ws), forward(x, w, xs, ws), rtol=0, atol=0
        )


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("split_k", [None, 3])
def test_public_group32_graph_replay(configured, split_k, monkeypatch):
    if configured:
        monkeypatch.setattr(
            gemm_op_a8w8, "get_CKGEMM_config", lambda *args: {"libtype": "triton"}
        )
    x, w, xs, ws = generate_inputs(3, 4096, 1280)
    gemm_a8w8_blockscale(x, w, xs, ws, split_k=split_k)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gemm_a8w8_blockscale(x, w, xs, ws, split_k=split_k)
    x.copy_((x.float() * 0.5).to(x.dtype))
    xs.view(torch.uint8).add_(1)
    graph.replay()
    expected = gemm_a8w8_blockscale_group32(x, w, xs, ws, split_k=split_k)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 2, 16, 31, 32, 33])
@pytest.mark.parametrize("group_n", [1, 32])
@pytest.mark.parametrize("raw_views", [False, True])
def test_public_small_n_scale_layouts(n, group_n, raw_views):
    x, w, xs, ws = generate_inputs(3, n, 128, weight_group_rows=group_n)
    expected = run_torch(x, w, xs, ws, weight_group_rows=group_n)
    operands = (x, w, xs, ws)
    if raw_views:
        operands = tuple(t.view(torch.uint8) for t in operands)
    actual = gemm_a8w8_blockscale(*operands, dtype=torch.float32)
    torch.testing.assert_close(
        actual.double(), expected, rtol=3e-5, atol=5e-5 * expected.abs().max().item()
    )


_FUSED_TILE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 1,
    "cache_modifier": "",
    "num_warps": 2,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "NUM_KSPLIT": 5,
    "N_FIRST": False,
    "REDUCE_BLOCK_SIZE_M": 32,
    "REDUCE_BLOCK_SIZE_N": 32,
    "FUSED_SPLITK": True,
}
_FUSED_PACKED = dict(
    _FUSED_TILE,
    packed={
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 256,
        "K_PACK": 2,
        "cache_modifier": "",
        "NUM_KSPLIT": 5,
        "num_warps": 2,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
    },
)
_FUSED = {"tile": _FUSED_TILE, "packed": _FUSED_PACKED}


def _assert_matches_reference(actual, x, w, xs, ws, dtype, weight_group_rows=32):
    expected = run_torch(x, w, xs, ws, weight_group_rows)
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


@pytest.mark.parametrize("variant", ["tile", "packed"])
@pytest.mark.parametrize(
    "m,n,k", [(1, 512, 5120), (3, 2053, 1280), (16, 1152, 5120), (37, 5120, 2304)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_split_k_matches_reference(variant, m, n, k, dtype):
    x, w, xs, ws = generate_inputs(m, n, k)
    actual = gemm_a8w8_blockscale_group32(
        x, w, xs, ws, dtype=dtype, config=_FUSED[variant]
    )
    _assert_matches_reference(actual, x, w, xs, ws, dtype)


@pytest.mark.parametrize("variant", ["tile", "packed"])
@pytest.mark.parametrize("m,n,k", [(3, 2053, 1280), (37, 5120, 2336)])
def test_weight_cache_modifier_keeps_results(variant, m, n, k):
    x, w, xs, ws = generate_inputs(m, n, k)
    config = _FUSED[variant]
    cached = dict(config, cache_modifier=".cg")
    if variant == "packed":
        cached["packed"] = dict(config["packed"], cache_modifier=".cg")
    run = gemm_a8w8_blockscale_group32
    expected = run(x, w, xs, ws, dtype=torch.float32, config=config)
    actual = run(x, w, xs, ws, dtype=torch.float32, config=cached)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fused_split_k_row_scales():
    x, w, xs, ws = generate_inputs(5, 2053, 1280, weight_group_rows=1)
    actual = gemm_a8w8_blockscale_group32(
        x, w, xs, ws, weight_group_rows=1, dtype=torch.float32, config=_FUSED_TILE
    )
    _assert_matches_reference(actual, x, w, xs, ws, torch.float32, 1)


@pytest.mark.parametrize("variant", ["tile", "packed"])
def test_fused_split_k_is_deterministic_and_rezeroes_counters(variant):
    x, w, xs, ws = generate_inputs(4, 1152, 5120)
    run = lambda: gemm_a8w8_blockscale_group32(
        x, w, xs, ws, dtype=torch.float32, config=_FUSED[variant]
    )
    first = run()
    # A stale counter or an early partial read would change a later sum.
    assert all(torch.equal(first, run()) for _ in range(50))
    torch.cuda.synchronize()
    assert not afp8wfp8_op._split_counters(x.device).any()


@pytest.mark.parametrize("variant", ["tile", "packed"])
def test_fused_split_k_graph_replay_on_warmed_stream(variant):
    x, w, xs, ws = generate_inputs(6, 1152, 5120)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gemm_a8w8_blockscale_group32(
            x, w, xs, ws, dtype=torch.float32, config=_FUSED[variant]
        )
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = gemm_a8w8_blockscale_group32(
            x, w, xs, ws, dtype=torch.float32, config=_FUSED[variant]
        )
    for factor in (2, 0.5):
        x.copy_((x.float() * factor).to(x.dtype))
        xs.view(torch.uint8).add_(1)
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches_reference(actual, x, w, xs, ws, torch.float32)


@pytest.mark.parametrize("variant", ["tile", "packed"])
def test_fused_split_k_first_use_inside_capture(variant):
    # Before torch 2.10 counters cannot be allocated mid-capture: runs unfused.
    x, w, xs, ws = generate_inputs(6, 1152, 5120)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gemm_a8w8_blockscale_group32(
            x, w, xs, ws, dtype=torch.float32, config=_FUSED[variant]
        )
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(actual, x, w, xs, ws, torch.float32)


@pytest.mark.parametrize("variant", ["tile", "packed"])
def test_fused_split_k_concurrent_streams(variant):
    inputs = [generate_inputs(m, 1152, 5120) for m in (3, 5)]
    expected = [
        gemm_a8w8_blockscale_group32(*t, dtype=torch.float32, config=_FUSED[variant])
        for t in inputs
    ]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in inputs]
    outputs = [[], []]
    # Unsynchronized interleaving keeps both streams' reductions in flight.
    for _ in range(40):
        for i, (stream, operands) in enumerate(zip(streams, inputs)):
            with torch.cuda.stream(stream):
                outputs[i].append(
                    gemm_a8w8_blockscale_group32(
                        *operands, dtype=torch.float32, config=_FUSED[variant]
                    )
                )
    torch.cuda.synchronize()
    for want, got in zip(expected, outputs):
        assert all(torch.equal(want, y) for y in got)
