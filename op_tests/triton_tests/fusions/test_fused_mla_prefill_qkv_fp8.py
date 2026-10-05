# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.ops.triton.fusions.fused_mla_prefill_qkv_fp8 import (
    fused_mla_prefill_qkv_fp8,
)
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() not in ("gfx950", "gfx1250"),
    reason="fused_mla_prefill_qkv_fp8 configs ship for gfx950 and gfx1250 only",
)

# Kimi-K3: qk_nope 128, qk_rope 64, v 128.
_NOPE, _ROPE, _V = 128, 64, 128
_Q_LORA, _KV_LORA, _GATE = 1536, 512, 1536


def torch_mla_prefill_qkv_fp8_ref(q, k_nope, v, k_pe, num_heads_out, out_dtype):
    heads = q.shape[1]
    k = torch.cat((k_nope, k_pe.expand(-1, heads, -1)), dim=-1)
    reps = -(-num_heads_out // heads)
    return tuple(
        x.repeat(1, reps, 1)[:, :num_heads_out].contiguous().to(out_dtype)
        for x in (q, k, v)
    )


def generate_inputs(n_tokens, heads, dtype=torch.bfloat16, rope=_ROPE):
    torch.manual_seed(0)
    q = torch.randn(n_tokens, heads, _NOPE + rope, device="cuda", dtype=dtype)
    kv = torch.randn(n_tokens, heads, _NOPE + _V, device="cuda", dtype=dtype)
    k_nope, v = kv.split([_NOPE, _V], dim=-1)
    fused = torch.randn(
        n_tokens, _Q_LORA + _KV_LORA + rope + _GATE, device="cuda", dtype=dtype
    )
    k_pe = fused[:, _Q_LORA + _KV_LORA : _Q_LORA + _KV_LORA + rope].unsqueeze(1)
    return q, k_nope, v, k_pe


def assert_bit_exact(out, ref):
    for name, o, r in zip("qkv", out, ref):
        assert o.is_contiguous(), f"{name}_out must be contiguous"
        assert o.shape == r.shape and o.dtype == r.dtype, f"{name}_out shape/dtype"
        assert torch.equal(
            o.view(torch.uint8), r.view(torch.uint8)
        ), f"{name}_out bytes"


@pytest.mark.parametrize("n_tokens", [1, 7, 256, 1112, 7784], ids=lambda n: f"T{n}")
@pytest.mark.parametrize(
    "heads,heads_out",
    [(12, 16), (16, 16), (8, 16), (24, 32), (40, 48)],
    ids=lambda h: str(h),
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_mla_prefill_qkv_fp8(n_tokens, heads, heads_out, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q, k_nope, v, k_pe = generate_inputs(n_tokens, heads, dtype)
    ref = torch_mla_prefill_qkv_fp8_ref(q, k_nope, v, k_pe, heads_out, dtypes.fp8)
    out = fused_mla_prefill_qkv_fp8(q, k_nope, v, k_pe, heads_out, dtypes.fp8)
    assert_bit_exact(out, ref)


def test_fused_mla_prefill_qkv_fp8_preallocated_and_2d_k_pe():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q, k_nope, v, k_pe = generate_inputs(1112, 12)
    ref = torch_mla_prefill_qkv_fp8_ref(q, k_nope, v, k_pe, 16, dtypes.fp8)
    outs = tuple(torch.empty_like(r) for r in ref)
    ret = fused_mla_prefill_qkv_fp8(
        q, k_nope, v, k_pe.squeeze(1), 16, dtypes.fp8, *outs
    )
    assert all(a is b for a, b in zip(ret, outs))
    assert_bit_exact(ret, ref)


def test_fused_mla_prefill_qkv_fp8_nope_only():
    """NoPE MLA (qk_rope_head_dim == 0): K is k_nope alone."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q, k_nope, v, k_pe = generate_inputs(256, 12, rope=0)
    ref = torch_mla_prefill_qkv_fp8_ref(q, k_nope, v, k_pe, 16, dtypes.fp8)
    assert_bit_exact(fused_mla_prefill_qkv_fp8(q, k_nope, v, k_pe, 16, dtypes.fp8), ref)
