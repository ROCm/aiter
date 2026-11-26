# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from itertools import pairwise

import pytest
import torch


def _inputs(counts, k, n, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(35)

    def rand(*shape):
        return torch.randn(*shape, device=device, generator=gen).to(torch.bfloat16)

    rows = sum(counts)
    offsets = torch.tensor([0, *counts], device=device).cumsum(0, dtype=torch.int32)
    return rand(rows, k), rand(len(counts), n, k), rand(rows, n), offsets


def _reference(x, weight, dy, offsets):
    """FP32 dgrad and wgrad, one independent GEMM per expert."""
    dx = torch.empty_like(x, dtype=torch.float32)
    dw = torch.zeros_like(weight, dtype=torch.float32)
    bounds = offsets.tolist()
    for e, (start, end) in enumerate(pairwise(bounds)):
        if start != end:
            grad = dy[start:end].float()
            dx[start:end] = grad @ weight[e].float()
            dw[e] = grad.T @ x[start:end].float()
    return dx, dw


@pytest.mark.parametrize("counts", [(0, 1, 5, 0, 2), (0, 0)])
@pytest.mark.parametrize("k,n", [(7, 3), (3, 7)])
def test_reference_against_autograd(counts, k, n):
    x, weight, dy, offsets = _inputs(counts, k, n)
    dx, dw = _reference(x, weight, dy, offsets)
    ids = torch.arange(len(counts)).repeat_interleave(torch.tensor(counts))
    xf = x.float().requires_grad_()
    wf = weight.float().requires_grad_()
    y = torch.bmm(wf[ids], xf.unsqueeze(-1)).squeeze(-1)
    ref_dx, ref_dw = torch.autograd.grad(y, (xf, wf), dy.float())
    torch.testing.assert_close(dx, ref_dx)
    torch.testing.assert_close(dw, ref_dw)


def test_input_contract():
    from aiter.ops.opus.latent_moe_grouped_gemm_bwd import latent_moe_grouped_gemm_bwd

    x, w, dy, offsets = _inputs((2, 0, 1), 7, 3)
    with pytest.raises(TypeError, match="int32"):
        latent_moe_grouped_gemm_bwd(x, w, dy, offsets.long())
    with pytest.raises(ValueError, match="E[+]1"):
        latent_moe_grouped_gemm_bwd(x, w, dy, offsets[:-1])
    with pytest.raises(ValueError, match="agree"):
        latent_moe_grouped_gemm_bwd(x, w, dy[:-1], offsets)


@pytest.mark.parametrize("k,n", [(3584, 3072), (3072, 3584)])
def test_gfx950_projection(k, n):
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires gfx950")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("requires gfx950")
    from aiter.ops.opus.latent_moe_grouped_gemm_bwd import latent_moe_grouped_gemm_bwd

    args = _inputs((1, 0, 17), k, n, device="cuda:0")
    dx, dw = latent_moe_grouped_gemm_bwd(*args)
    ref_dx, ref_dw = _reference(*args)
    torch.testing.assert_close(dx, ref_dx.bfloat16(), atol=0.03125, rtol=0.015625)
    torch.testing.assert_close(dw, ref_dw, atol=1e-4, rtol=1e-4)


def test_gfx950_empty_and_unaligned_on_current_stream():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires gfx950")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("requires gfx950")
    from aiter.ops.opus.latent_moe_grouped_gemm_bwd import latent_moe_grouped_gemm_bwd

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, w, dy, off = _inputs((0, 3), 7, 5, device="cuda:0")
        storage = x.new_empty(x.numel() + 1)
        x = storage[1:].view_as(x).copy_(x)
        for ends in ((0, 0, 3), (0, 3, 3)):
            off.copy_(torch.tensor(ends, device=off.device, dtype=off.dtype))
            dx, dw = latent_moe_grouped_gemm_bwd(x, w, dy, off)
            ref_dx, ref_dw = _reference(x, w, dy, off)
            torch.testing.assert_close(dx, ref_dx.bfloat16())
            torch.testing.assert_close(dw, ref_dw)
        dx, dw = latent_moe_grouped_gemm_bwd(x[:0], w, dy[:0], off.zero_())
        assert dx.shape == (0, 7)
        assert torch.count_nonzero(dw).item() == 0
