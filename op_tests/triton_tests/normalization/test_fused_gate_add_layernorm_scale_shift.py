# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.normalization.fused_gate_add_layernorm_scale_shift import (
    fused_gate_add_layernorm_scale_shift,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype


def generate_inputs(M, N, G, dtype):
    x = torch.randn((M, N), dtype=dtype, device="cuda")
    attn = torch.randn((M, N), dtype=dtype, device="cuda")
    gate = torch.randn((G, N), dtype=dtype, device="cuda")
    scale = torch.randn((G, N), dtype=dtype, device="cuda") * 0.1
    shift = torch.randn((G, N), dtype=dtype, device="cuda") * 0.1
    return x, attn, gate, scale, shift


def run_torch(x, attn, gate, scale, shift, epsilon):
    M, _N = x.shape
    G = gate.shape[0]
    rpg = M // G
    gate_b = gate.repeat_interleave(rpg, dim=0)
    scale_b = scale.repeat_interleave(rpg, dim=0)
    shift_b = shift.repeat_interleave(rpg, dim=0)
    # fp32 throughout to match kernel (no intermediate bf16 rounding)
    h = x.float() + gate_b.float() * attn.float()
    hf = h
    mean = hf.mean(dim=-1, keepdim=True)
    var = ((hf - mean) ** 2).mean(dim=-1, keepdim=True)
    hn = (hf - mean) * torch.rsqrt(var + epsilon)
    out = hn * (1.0 + scale_b.float()) + shift_b.float()
    return out.to(x.dtype), h.to(x.dtype)


def get_vals():
    # (M, N, G) -- G groups each spanning M//G rows
    return [
        (2, 16, 1),
        (4, 128, 2),
        (2202, 1536, 2),  # SD3.5-medium 512x512: M=2*(1024+77), N=1536, G=2
        (8192, 3072, 2),
        (256, 1024, 4),
        (1101, 1536, 1),
        (4096, 2048, 8),
        (4, 2048, 1),  # N == BLOCK_SIZE_N, mask all-true
        (4, 2049, 1),  # N > 2048, BLOCK_SIZE_N=4096, mask active
        (4, 128, 4),  # G == M, rows_per_group=1
        (1, 512, 1),  # M=1, single row
    ]


@pytest.mark.parametrize("in_dtype_str", ["bf16"])
@pytest.mark.parametrize("M, N, G", [(shape) for shape in get_vals()])
def test_fused_gate_add_layernorm_scale_shift(M, N, G, in_dtype_str):
    in_dtype = str_to_torch_dtype[in_dtype_str]
    torch.manual_seed(0)

    x, attn, gate, scale, shift = generate_inputs(M, N, G, in_dtype)
    epsilon = 1e-6

    out_torch, h_torch = run_torch(x, attn, gate, scale, shift, epsilon)
    out_triton, h_triton = fused_gate_add_layernorm_scale_shift(
        x, attn, gate, scale, shift, epsilon
    )

    atol, rtol = 1e-2, 1e-2
    assert out_triton.dtype == in_dtype
    torch.testing.assert_close(h_triton, h_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_triton, out_torch, atol=atol, rtol=rtol)
    # x is updated in-place: h_triton must be the same storage as x
    assert (
        h_triton.data_ptr() == x.data_ptr()
    ), "h (residual output) must alias x (in-place update)"


def test_fused_gate_add_layernorm_scale_shift_noncontiguous():
    """Non-contiguous inputs are made contiguous; output must still be correct."""
    torch.manual_seed(7)
    DEV, DT = "cuda", torch.bfloat16
    M, N, G = 8, 128, 2
    epsilon = 1e-6
    # Create non-contiguous tensors via transpose round-trip
    x_base = torch.randn(N, M, device=DEV, dtype=DT).t()  # (M, N), non-contiguous
    a_base = torch.randn(N, M, device=DEV, dtype=DT).t()
    assert not x_base.is_contiguous()
    gate = torch.randn(G, N, device=DEV, dtype=DT) * 0.2
    scale = torch.randn(G, N, device=DEV, dtype=DT) * 0.1
    shift = torch.randn(G, N, device=DEV, dtype=DT) * 0.1
    # reference on contiguous copies
    x_c = x_base.contiguous()
    a_c = a_base.contiguous()
    out_ref, h_ref = fused_gate_add_layernorm_scale_shift(
        x_c, a_c, gate, scale, shift, epsilon
    )
    # call with non-contiguous inputs
    x_nc = x_base.clone().t().t()  # fresh non-contiguous copy
    a_nc = a_base.clone().t().t()
    out_nc, h_nc = fused_gate_add_layernorm_scale_shift(
        x_nc, a_nc, gate, scale, shift, epsilon
    )
    torch.testing.assert_close(h_nc, h_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_nc, out_ref, atol=1e-2, rtol=1e-2)


def test_fused_gate_add_layernorm_scale_shift_inplace_isolation():
    """Verify in-place does not corrupt unrelated memory adjacent to x."""
    torch.manual_seed(42)
    DEV, DT = "cuda", torch.bfloat16
    M, N, G = 8, 64, 2
    epsilon = 1e-6
    # allocate a larger buffer and take a slice so we can check neighbours
    buf = torch.zeros(M + 4, N, device=DEV, dtype=DT)
    sentinel = buf[M:]  # rows after x, should be untouched
    x = buf[:M]
    x.copy_(torch.randn(M, N, device=DEV, dtype=DT))
    attn = torch.randn(M, N, device=DEV, dtype=DT)
    gate = torch.randn(G, N, device=DEV, dtype=DT) * 0.2
    scale = torch.randn(G, N, device=DEV, dtype=DT) * 0.1
    shift = torch.randn(G, N, device=DEV, dtype=DT) * 0.1
    sentinel_before = sentinel.clone()
    fused_gate_add_layernorm_scale_shift(x, attn, gate, scale, shift, epsilon)
    assert torch.equal(
        sentinel, sentinel_before
    ), "in-place write corrupted memory beyond x boundary"
