# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.fused_moe import (
    _FLYDSL_STAGE1_OUT_CACHE,
    _get_flydsl_stage1_out,
)


@pytest.fixture(autouse=True)
def clear_stage1_out_cache():
    _FLYDSL_STAGE1_OUT_CACHE.clear()
    yield
    _FLYDSL_STAGE1_OUT_CACHE.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_flydsl_stage1_out_is_keyed_by_shape():
    """Two shapes must not share a buffer.

    The cache returns whatever the key maps to, and the kernel takes
    out.view(-1); a key that ignored the shape would hand back an undersized
    buffer and write out of bounds.
    """
    device = torch.device("cuda:0")

    small = _get_flydsl_stage1_out((512, 768), device)
    large = _get_flydsl_stage1_out((1024, 768), device)

    assert small.data_ptr() != large.data_ptr()
    assert small.shape == (512, 768)
    assert large.shape == (1024, 768)
    assert len(_FLYDSL_STAGE1_OUT_CACHE) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_flydsl_stage1_out_is_shared_across_graph_captures():
    """Captures must share one buffer, rather than retain one each.

    One buffer per captured graph is the behaviour this cache exists to remove,
    so every capture asking for the same shape has to land on the same
    allocation. Capture runs on its own stream, so captures share with each
    other and not with eager execution: the cache holds one buffer per stream,
    which is the point, instead of one per graph.
    """
    device = torch.device("cuda:0")
    shape = (512, 768)

    eager = _get_flydsl_stage1_out(shape, device)

    captured = []
    for _ in range(2):
        graph = torch.cuda.CUDAGraph()
        # Empty graph on purpose: this probes the allocator, not a kernel, so
        # torch warns that nothing was captured.
        with torch.cuda.graph(graph):
            captured.append(_get_flydsl_stage1_out(shape, device))

    assert captured[0].data_ptr() == captured[1].data_ptr()
    assert captured[0].data_ptr() != eager.data_ptr()
    assert len(_FLYDSL_STAGE1_OUT_CACHE) == 2


def _pick_kernel(**want):
    """Return a registered single-k_batch kernel matching ``want``, or skip.

    Selecting by parsed parameters rather than by a literal name keeps these
    tests working as the registry gains and loses kernels.
    """
    from aiter.ops.flydsl.moe_kernels import _KERNEL_PARAMS

    for name, params in _KERNEL_PARAMS.items():
        if params.get("k_batch", 1) != 1:
            continue
        if all(params.get(key) == value for key, value in want.items()):
            return name, params
    pytest.skip(f"no registered FlyDSL kernel matching {want}")


def _observe_stage1_out(monkeypatch, kernel_name, params, device):
    """Drive the wrapper far enough to see which buffer it passed down.

    out_dtype is passed explicitly because no registered kernel declares an
    fp8 output; the registry only carries bf16 and f16. The gate is therefore
    reachable only through this override, which is how fused_moe drives it.
    """
    import aiter.fused_moe as fused_moe

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_flydsl_stage1_wrapper_allocates_from_cache(monkeypatch):
    """The eligible path must reach the cache, not torch.empty per call.

    Every other test here exercises the helper directly, so without this one
    the gate that decides to call it is unverified.
    """
    device = torch.device("cuda:0")
    name, params = _pick_kernel(a_dtype="fp8", b_dtype="fp4")

    out, expected_shape = _observe_stage1_out(monkeypatch, name, params, device)

    assert out is not None, "eligible call did not take a cached buffer"
    assert tuple(out.shape) == expected_shape
    assert out.data_ptr() == _get_flydsl_stage1_out(expected_shape, device).data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_flydsl_stage1_wrapper_skips_cache_for_a16w4(monkeypatch):
    """a16w4 must keep allocating its own buffer.

    It returns its own sorted intermediate before it looks at ``out``, so a
    buffer handed to it is dropped silently rather than rejected; dropping the
    carve-out would waste the allocation and produce no error.
    """
    device = torch.device("cuda:0")
    name, params = _pick_kernel(a_dtype="bf16", b_dtype="fp4")

    out, _ = _observe_stage1_out(monkeypatch, name, params, device)

    assert out is None, "a16w4 was handed a cached buffer it will silently drop"
    assert not _FLYDSL_STAGE1_OUT_CACHE
