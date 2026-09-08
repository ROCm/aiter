# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl

from . import _get_compiled


@triton.jit
def _rmsnorm_constexpr_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    eps,
    stride_x,
    stride_out,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    x_hat = x / tl.sqrt(variance + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_out + cols, (x_hat * w).to(tl.float16), mask=mask)


def triton_rmsnorm(x, weight, eps, out):
    M, N = x.shape
    _rmsnorm_constexpr_kernel[(M,)](
        x, weight, out, eps, x.stride(0), out.stride(0), N, triton.next_power_of_2(N)
    )
    return out


def torch_rmsnorm(x, weight, eps):
    x_f32 = x.to(torch.float32)
    return (x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps) * weight).to(
        x.dtype
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(32, 64), (64, 128), (128, 256)])
def test_compile_constexpr_mutation(M, N, dtype):
    """Triton kernel with tl.constexpr + shared-storage views under torch.compile."""
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    head_dim = N // 4
    q_dim = 2 * head_dim
    k_dim = head_dim
    eps = 1e-6

    qkv = torch.randn(M, N, device="cuda", dtype=dtype)
    w_q = torch.ones(q_dim, device="cuda", dtype=dtype)
    w_k = torch.ones(k_dim, device="cuda", dtype=dtype)

    def fn(qkv, w_q, w_k):
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim : q_dim + k_dim]
        v = qkv[:, q_dim + k_dim :]
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        triton_rmsnorm(q, w_q, eps, q_out)
        triton_rmsnorm(k, w_k, eps, k_out)
        return q_out, k_out, v

    q_ref = torch_rmsnorm(qkv[:, :q_dim], w_q, eps)
    k_ref = torch_rmsnorm(qkv[:, q_dim : q_dim + k_dim], w_k, eps)
    v_ref = qkv[:, q_dim + k_dim :].clone()

    compiled_fn = _get_compiled(fn)
    q_out, k_out, v_out = compiled_fn(qkv, w_q, w_k)
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_out, v_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
