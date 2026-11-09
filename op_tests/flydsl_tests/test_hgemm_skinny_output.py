# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm,
    flydsl_hgemm_kernel_name,
    get_flydsl_hgemm_kernel_params,
)

pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="skinny is gfx950 only")


def _config(m):
    # Same wave-private ring for both epilogues, with room for FP32 partials.
    return {
        "block_m": 16 if m <= 16 else 32,
        "block_n": 16,
        "block_k": 64,
        "stages": 3,
        "split_k": 1,
        "m_waves": 1,
        "n_waves": 1,
        "k_waves": 4,
        "group_m": 0,
        "policy": "sk",
    }


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 128, 1024),
        (7, 896, 7168),
        (16, 128, 1024),
        (17, 128, 1024),
        (23, 896, 7168),
        (32, 128, 1024),
    ],
)
def test_output_contract(m, n, k, monkeypatch):
    from aiter import tuned_gemm

    torch.manual_seed(2026)
    # Non-contiguous inputs are copied by the public wrapper, as at runtime.
    a = torch.randn((m, k * 2), device="cuda", dtype=torch.bfloat16)[:, ::2]
    w = torch.randn((n * 2, k), device="cuda", dtype=torch.bfloat16)[::2]
    cfg = _config(m)
    ref = a.float() @ w.float().t()
    for dtype in (torch.bfloat16, torch.float32, torch.bfloat16, torch.float32):
        name = flydsl_hgemm_kernel_name(
            dtype=a.dtype, out_dtype=dtype, config=cfg, has_bias=False
        )
        parsed = get_flydsl_hgemm_kernel_params(name)
        assert parsed is not None and parsed["out_dtype"] == (
            "fp32" if dtype == torch.float32 else "bf16"
        )
        guard = torch.full((m * n + 32,), 8192, dtype=dtype, device="cuda")
        out = guard[16:-16].view(m, n)
        out.fill_(float("nan"))
        # Route through the production public out= API, not a kernel helper.
        monkeypatch.setattr(
            tuned_gemm,
            "get_GEMM_A16W16_config",
            lambda name=name, **kwargs: {
                "libtype": "flydsl",
                "solidx": 0,
                "kernelName": name,
                "splitK": 1,
            },
        )
        tuned_gemm.gemm_a16w16_out(a, w, out, otype=dtype)
        torch.testing.assert_close(
            out,
            ref.to(dtype),
            rtol=2e-4 if dtype == torch.float32 else 0.01,
            atol=5e-4 if dtype == torch.float32 else 0.01,
        )
        assert torch.all(guard[:16] == 8192) and torch.all(guard[-16:] == 8192)
        if dtype == torch.float32:
            assert not torch.equal(out, out.bfloat16().float())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tuned_gemm.gemm_a16w16_out(a, w, out, otype=torch.float32)
    a.neg_()
    for _ in range(5):
        graph.replay()
    torch.testing.assert_close(out, -ref, rtol=2e-4, atol=5e-4)


def test_streams_and_tuner():
    from aiter.ops.flydsl.gemm_a16w16_policy import get_flydsl_a16w16_configs

    torch.manual_seed(79)
    a = torch.randn((23, 1024), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((128, 1024), dtype=torch.bfloat16, device="cuda")
    ref = a.float() @ w.float().t()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs.append(
                flydsl_hgemm(
                    a, w, out_dtype=torch.float32, stream=stream, **_config(23)
                )
            )
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for out in outputs:
        torch.testing.assert_close(out, ref, rtol=2e-4, atol=5e-4)
    configs = get_flydsl_a16w16_configs(
        16, 896, 7168, torch.bfloat16, torch.float32, False
    )
    assert any(c.get("policy") == "sk" for c in configs)
    biased = get_flydsl_a16w16_configs(
        16, 896, 7168, torch.bfloat16, torch.float32, True
    )
    assert all(c.get("policy") != "sk" for c in biased)
    fp16 = get_flydsl_a16w16_configs(16, 896, 7168, torch.float16, torch.float32, False)
    assert all(c.get("policy") != "sk" for c in fp16)
