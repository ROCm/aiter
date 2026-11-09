# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16_oriented import gemm_a16w16_oriented
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(get_arch() != "gfx950", reason="gfx950 configs")


@pytest.mark.parametrize(
    "m,n,k",
    [(3, 37, 133), (65, 97, 385), (64, 896, 7168), (128, 224, 7168), (256, 4480, 7168)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_oriented(m, n, k, dtype, fp32, bias):
    torch.manual_seed(2026)
    a = torch.randn((m, k * 2), device="cuda", dtype=dtype)[:, ::2]
    w = torch.randn((n * 2, k), device="cuda", dtype=dtype)[::2]
    outdtype = torch.float32 if fp32 else dtype
    b = torch.randn((n * 2,), device="cuda", dtype=outdtype)[::2] if bias else None
    ref = a.float() @ w.float().t()
    if b is not None:
        ref += b.float()
    out = torch.full((m, n), float("nan"), device="cuda", dtype=outdtype)
    result = gemm_a16w16_oriented(a, w, bias=b, dtype=outdtype, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        result,
        ref.to(outdtype),
        rtol=2e-4 if fp32 else 0.02,
        atol=5e-4 if fp32 else 0.125,
    )
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gemm_a16w16_oriented(a, w, bias=b, dtype=outdtype, out=out)
    a.neg_()
    ref = a.float() @ w.float().t()
    if b is not None:
        ref += b.float()
    for _ in range(3):
        g.replay()
    torch.testing.assert_close(
        out, ref.to(outdtype), rtol=2e-4 if fp32 else 0.02, atol=5e-4 if fp32 else 0.125
    )


def test_multistream():
    a = torch.randn((64, 7168), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((896, 7168), device="cuda", dtype=torch.bfloat16)
    results = []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results.append(gemm_a16w16_oriented(a, w, dtype=torch.float32))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    ref = a.float() @ w.float().t()
    for result in results:
        torch.testing.assert_close(result, ref, rtol=2e-4, atol=5e-4)


def test_dispatch_out(monkeypatch):
    import aiter.tuned_gemm as tg

    # The CSV selects a backend family; tile values stay in the JSON config.
    monkeypatch.setattr(
        tg,
        "get_GEMM_A16W16_config",
        lambda **kwargs: {
            "libtype": "triton",
            "solidx": 0,
            "kernelName": "triton_a16w16_oriented",
        },
    )
    a = torch.randn((2, 32, 128), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((224, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((2, 32, 224), device="cuda", dtype=torch.float32)
    tg.gemm_a16w16_out(a, w, out, otype=torch.float32)
    torch.testing.assert_close(out, a.float() @ w.float().t(), rtol=2e-4, atol=5e-4)


def test_invalid_out():
    a = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((64, 16), device="cuda", dtype=torch.bfloat16).t()
    with pytest.raises(ValueError, match="contiguous"):
        gemm_a16w16_oriented(a, w, out=out)
