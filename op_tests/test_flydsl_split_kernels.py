# SPDX-License-Identifier: MIT
"""Correctness and graph replay coverage for the opt-in gfx950 split kernels."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import build_gemm_bf16_split_fp32, build_softmax_split

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="requires gfx950"
)


def replay(fn, out):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    out.fill_(float("nan"))
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), "empty capture or nonfinite result"


@pytest.mark.parametrize(
    "shape", [(1, 32768), (1, 131072), (1, 89999), (2, 131071), (4, 65536), (1, 262144)]
)
@pytest.mark.parametrize(
    "dtype,name",
    [(torch.bfloat16, "bf16"), (torch.float16, "f16"), (torch.float32, "f32")],
)
@pytest.mark.parametrize("distribution", ["norm", "uniform", "constant", "wide"])
def test_softmax(shape, dtype, name, distribution):
    gen = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)
    if distribution == "uniform":
        x.uniform_(-1, 1, generator=gen)
    elif distribution == "constant":
        x.fill_(1)
    elif distribution == "wide":
        x.mul_(50)
    x = x.to(dtype)
    out = torch.empty_like(x)
    reference = x.float().softmax(-1)
    launch = build_softmax_split(*shape, name)
    replay(lambda: launch(x, out), out)
    actual = out.float()
    scaled = (
        (actual - reference).abs()
        / (reference.abs() + reference.amax(-1, keepdim=True))
    ).max()
    assert scaled <= {"bf16": 0.02, "f16": 0.005, "f32": 1e-5}[name]
    rounded_error = (reference.to(dtype).float().sum(-1) - 1).abs()
    assert torch.all(
        (actual.sum(-1) - 1).abs()
        <= rounded_error + {"bf16": 0.01, "f16": 0.002, "f32": 1e-5}[name]
    )


@pytest.mark.parametrize("k,split", [(7168, 4), (7168, 8), (16384, 8)])
@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("distribution", ["norm", "uniform", "zero", "constant"])
def test_gemm(k, split, seed, distribution):
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((32, k), device="cuda", dtype=torch.bfloat16, generator=gen)
    b = torch.randn((384, k), device="cuda", dtype=torch.bfloat16, generator=gen)
    for x in (a, b):
        if distribution == "uniform":
            x.uniform_(-1, 1, generator=gen)
        elif distribution in ("zero", "constant"):
            x.fill_(0 if distribution == "zero" else 1)
    out = torch.empty((32, 384), device="cuda", dtype=torch.bfloat16)
    cfg = {
        "block_m": 16,
        "block_n": 32,
        "block_k": 128,
        "stages": 4,
        "split_k": split,
        "m_waves": 1,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "policy": "ft",
    }
    launch = build_gemm_bf16_split_fp32(32, 384, cfg)
    reference = a.float() @ b.float().T
    replay(lambda: launch(a, b, out), out)
    delta = (out.float() - reference).abs()
    rms = reference.square().mean().sqrt().item()
    assert torch.all(delta <= 0.02 * reference.abs() + 0.01 * max(rms, 1e-6))
    assert delta.square().mean().sqrt().item() / max(rms, 1e-20) <= 0.01


def test_reject_invalid_contracts():
    with pytest.raises(ValueError, match="32768"):
        build_softmax_split(1, 1024)
    with pytest.raises(ValueError, match="dtype_str"):
        build_softmax_split(1, 32768, "int8")
    with pytest.raises(ValueError, match="positive"):
        build_gemm_bf16_split_fp32(0, 384)
    with pytest.raises(ValueError, match="split_k"):
        build_gemm_bf16_split_fp32(32, 384, {"split_k": 1})
    x = torch.randn((1, 32768), device="cuda", dtype=torch.bfloat16)
    launch = build_softmax_split(1, 32768)
    with pytest.raises(ValueError, match="aliasing"):
        launch(x, x)
    with pytest.raises(ValueError, match="matching contiguous"):
        launch(x.float(), torch.empty_like(x))
    gemm = build_gemm_bf16_split_fp32(32, 384)
    with pytest.raises(ValueError, match="expected"):
        gemm(x, x, x)


def test_separate_stream_workspaces():
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    x = torch.randn((1, 89999), device="cuda", dtype=torch.bfloat16)
    outputs = [torch.empty_like(x) for _ in streams]
    launches = [build_softmax_split(1, 89999) for _ in streams]
    torch.cuda.synchronize()
    for stream, launch, out in zip(streams, launches, outputs, strict=True):
        with torch.cuda.stream(stream):
            launch(x, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
