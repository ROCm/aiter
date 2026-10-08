# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import (
    _FLYDSL_STAGE1_OUT_CACHE,
    _get_flydsl_stage1_out,
)
from aiter.jit.utils.chip_info import get_gfx

_NEED_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)
_NEED_GFX950 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 FlyDSL required",
)


@pytest.fixture(autouse=True)
def clear_stage1_out_cache():
    _FLYDSL_STAGE1_OUT_CACHE.clear()
    yield
    _FLYDSL_STAGE1_OUT_CACHE.clear()


@_NEED_GPU
def test_flydsl_stage1_out_is_reused_per_stream():
    device = torch.device("cuda:0")
    shape = (1024, 1536)

    output = _get_flydsl_stage1_out(shape, device)
    reused = _get_flydsl_stage1_out(shape, device)

    other_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(other_stream):
        other_stream_output = _get_flydsl_stage1_out(shape, device)

    assert reused.data_ptr() == output.data_ptr()
    assert other_stream_output.data_ptr() != output.data_ptr()
    assert len(_FLYDSL_STAGE1_OUT_CACHE) == 2


@_NEED_GPU
def test_flydsl_stage1_out_is_keyed_by_shape():
    """A key without the shape would return an undersized buffer."""
    device = torch.device("cuda:0")

    small = _get_flydsl_stage1_out((512, 768), device)
    large = _get_flydsl_stage1_out((1024, 768), device)

    assert small.data_ptr() != large.data_ptr()
    assert small.shape == (512, 768)
    assert large.shape == (1024, 768)
    assert len(_FLYDSL_STAGE1_OUT_CACHE) == 2


@_NEED_GPU
def test_flydsl_stage1_out_is_shared_across_graph_captures():
    """Captures on one stream share one buffer, separate from eager execution."""
    device = torch.device("cuda:0")
    shape = (512, 768)
    capture_stream = torch.cuda.Stream(device=device)
    pool = torch.cuda.graph_pool_handle()

    eager = _get_flydsl_stage1_out(shape, device)

    graphs, captured = [], []
    for _ in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
            buf = _get_flydsl_stage1_out(shape, device)
            buf.view(torch.uint8).zero_()
        graphs.append(graph)
        captured.append(buf)

    assert captured[0].data_ptr() == captured[1].data_ptr()
    assert captured[0].data_ptr() != eager.data_ptr()
    assert (device, capture_stream.cuda_stream, shape) in _FLYDSL_STAGE1_OUT_CACHE
    assert len(_FLYDSL_STAGE1_OUT_CACHE) == 2


@_NEED_GPU
def test_flydsl_stage1_out_outlives_the_graph_that_allocated_it():
    """A buffer allocated during capture stays valid after that graph is freed."""
    device = torch.device("cuda:0")
    shape = (512, 768)
    capture_stream = torch.cuda.Stream(device=device)
    pool = torch.cuda.graph_pool_handle()

    first = torch.cuda.CUDAGraph()
    with torch.cuda.graph(first, pool=pool, stream=capture_stream):
        buf = _get_flydsl_stage1_out(shape, device)
        buf.view(torch.uint8).zero_()

    second = torch.cuda.CUDAGraph()
    with torch.cuda.graph(second, pool=pool, stream=capture_stream):
        reused = _get_flydsl_stage1_out(shape, device)
        reused.view(torch.uint8).fill_(7)
    assert reused.data_ptr() == buf.data_ptr()

    del first
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    filler = torch.full(shape, 3, dtype=torch.uint8, device=device)

    second.replay()
    torch.cuda.synchronize()

    assert torch.all(buf.view(torch.uint8) == 7)
    assert torch.all(filler == 3)


def _pick_kernel(**want):
    """Return a registered single-k_batch kernel matching ``want``, or skip."""
    from aiter.ops.flydsl.moe_kernels import _KERNEL_PARAMS

    for name, params in _KERNEL_PARAMS.items():
        if params.get("k_batch", 1) != 1:
            continue
        if all(params.get(key) == value for key, value in want.items()):
            return name, params
    pytest.skip(f"no registered FlyDSL kernel matching {want}")


def _observe_stage1_out(monkeypatch, kernel_name, params, device):
    """Return the ``out`` the wrapper passes to stage1, with the expected shape.

    No registered kernel declares an fp8 output, so pass out_dtype the way
    fused_moe does.
    """
    from aiter import fused_moe

    moe_kernels = fused_moe._get_flydsl_moe_kernels()
    seen = {}

    def _capture(**kwargs):
        seen["out"] = kwargs["out"]
        return None, None

    monkeypatch.setattr(moe_kernels, "flydsl_moe_stage1", _capture)

    inter_dim, model_dim, experts, topk = 256, 512, 8, 4
    sorted_token_ids = torch.zeros(128, dtype=torch.int32, device=device)
    sorted_expert_ids = torch.zeros(4, dtype=torch.int32, device=device)
    fused_moe._flydsl_stage1_wrapper(
        hidden_states=torch.empty((64, model_dim), dtype=torch.uint8, device=device),
        w1=torch.empty(
            (experts, inter_dim * 2, model_dim), dtype=torch.uint8, device=device
        ),
        w2=None,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=torch.tensor([64], dtype=torch.int32, device=device),
        out=None,
        topk=topk,
        kernelName=kernel_name,
        activation=fused_moe.ActivationType.Situv2,
        out_dtype="fp8",
        v2_output_layout=True,
    )
    expected_rows = max(
        sorted_token_ids.shape[0], sorted_expert_ids.shape[0] * params["tile_m"]
    )
    return seen["out"], (expected_rows, inter_dim)


@_NEED_GPU
def test_flydsl_stage1_wrapper_allocates_from_cache(monkeypatch):
    device = torch.device("cuda:0")
    name, params = _pick_kernel(a_dtype="fp8", b_dtype="fp4")

    out, expected_shape = _observe_stage1_out(monkeypatch, name, params, device)

    assert out is not None, "eligible call did not take a cached buffer"
    assert tuple(out.shape) == expected_shape
    assert out.data_ptr() == _get_flydsl_stage1_out(expected_shape, device).data_ptr()


@_NEED_GPU
def test_flydsl_stage1_wrapper_skips_cache_for_a16w4(monkeypatch):
    """a16w4 returns its own intermediate and would silently drop a cached buffer."""
    device = torch.device("cuda:0")
    name, params = _pick_kernel(a_dtype="bf16", b_dtype="fp4")

    out, _ = _observe_stage1_out(monkeypatch, name, params, device)

    assert out is None, "a16w4 was handed a cached buffer it will silently drop"
    assert not _FLYDSL_STAGE1_OUT_CACHE


_TILE_M = 32
_INTER_DIM = 256


def _a8w4_situv2_data(seed):
    from op_tests.flydsl_tests.test_flydsl_moe import _generate_a8w4_situv2_vec4_data

    return _generate_a8w4_situv2_vec4_data(64, 512, _INTER_DIM, 16, 4, 32, seed=seed)


def _sorted_rows(d):
    return max(d["sorted_ids"].shape[0], d["sorted_expert_ids"].shape[0] * _TILE_M)


def _stage1_situv2_a8w4(d, out):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
    from op_tests.flydsl_tests.test_flydsl_moe import (
        SITUV2_BETA,
        SITUV2_LINEAR_BETA,
    )

    return flydsl_moe_stage1(
        a=d["a_q"],
        w1=d["w1_q_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        out=out,
        topk=d["topk"],
        tile_m=_TILE_M,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="fp8",
        act="situv2",
        situ_beta=SITUV2_BETA,
        situ_linear_beta=SITUV2_LINEAR_BETA,
        w1_scale=d["w1_scale_shuf"],
        a1_scale=d["a_scale_sort"],
        v2_output_layout=True,
    )


@_NEED_GFX950
def test_flydsl_v2_stage1_result_independent_of_buffer_contents():
    """Bytes the kernel writes must not depend on what the reused buffer held."""
    d = _a8w4_situv2_data(seed=21)

    def run(poison):
        buf = torch.full(
            (_sorted_rows(d), _INTER_DIM), poison, dtype=torch.uint8, device="cuda"
        ).view(dtypes.fp8)
        out, _scale = _stage1_situv2_a8w4(d, buf)
        torch.cuda.synchronize()
        return out.view(torch.uint8)

    first = run(0xA5).clone()
    second = run(0x5A)

    untouched = first != second
    assert not bool(untouched.all()), "kernel wrote nothing; the test is vacuous"
    assert torch.equal(first[untouched], torch.full_like(first[untouched], 0xA5))
    assert torch.equal(second[untouched], torch.full_like(second[untouched], 0x5A))


@_NEED_GFX950
def test_flydsl_v2_stage1_rejects_wrong_output_dtype():
    d = _a8w4_situv2_data(seed=22)
    bad_buf = torch.empty(
        (_sorted_rows(d), _INTER_DIM), dtype=torch.bfloat16, device="cuda"
    )

    with pytest.raises(ValueError, match="stage1 out has dtype"):
        _stage1_situv2_a8w4(d, bad_buf)


@_NEED_GFX950
def test_flydsl_v2_stage1_rejects_wrong_output_shape():
    """An undersized buffer must raise; the kernel would write past its end."""
    d = _a8w4_situv2_data(seed=23)
    undersized = torch.empty(
        (_sorted_rows(d) - 1, _INTER_DIM), dtype=dtypes.fp8, device="cuda"
    )

    with pytest.raises(ValueError, match="stage1 out has shape"):
        _stage1_situv2_a8w4(d, undersized)


@_NEED_GFX950
def test_flydsl_v2_stage1_accepts_uint8_byte_buffer():
    """The MoE tuner passes uint8 storage; it must match an fp8 buffer byte for byte."""
    d = _a8w4_situv2_data(seed=25)
    shape = (_sorted_rows(d), _INTER_DIM)

    as_fp8, _ = _stage1_situv2_a8w4(
        d, torch.zeros(shape, dtype=torch.uint8, device="cuda").view(dtypes.fp8)
    )
    as_u8, _ = _stage1_situv2_a8w4(
        d, torch.zeros(shape, dtype=torch.uint8, device="cuda")
    )
    torch.cuda.synchronize()

    assert torch.equal(as_u8.view(torch.uint8), as_fp8.view(torch.uint8))


@_NEED_GFX950
def test_flydsl_v2_stage1_accepts_the_fused_moe_cache_buffer():
    """fused_moe and moe_kernels compute this shape separately; they must agree."""
    d = _a8w4_situv2_data(seed=24)
    assert d["w1_q_shuf"].shape[1] // 2 == _INTER_DIM

    cached = _get_flydsl_stage1_out(
        (_sorted_rows(d), _INTER_DIM), torch.device("cuda:0")
    )
    out, _scale = _stage1_situv2_a8w4(d, cached)
    torch.cuda.synchronize()

    assert out.data_ptr() == cached.data_ptr()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
