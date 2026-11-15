# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_bf16_bmm_cat_quant_pad_mla import (
    fused_bf16_bmm_cat_quant_pad_mla,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

KV_LORA, D_NOPE, D_ROPE, D_V = 512, 128, 64, 128
SUPPORTED_ARCHS = ("gfx950",)


def vllm_fp8_max() -> float:
    """vLLM's static fp8 quant clamp bound (get_fp8_min_max)."""
    fp8 = get_fp8_e4m3_dtype()
    return 224.0 if fp8 == torch.float8_e4m3fnuz else torch.finfo(fp8).max


def generate_inputs(
    M: int,
    QH: int,
    kv_lora: int = KV_LORA,
    d_nope: int = D_NOPE,
    d_rope: int = D_ROPE,
    d_v: int = D_V,
    dtype=torch.bfloat16,
):
    """q_nope, W_UK^T and q_pe as the strided views vLLM's MLA decode passes:
    q_nope/q_pe split from the q_b_proj output, W_UK^T from split_kv_b_proj on a
    bf16 kv_b_proj weight."""
    q = torch.randn((M, QH, d_nope + d_rope), dtype=dtype, device="cuda")
    q_nope, q_pe = q.split((d_nope, d_rope), dim=-1)
    kv_b = torch.randn((QH * (d_nope + d_v), kv_lora), dtype=dtype, device="cuda")
    w_uk, _ = (
        (kv_b * 0.05).T.view(kv_lora, QH, d_nope + d_v).split((d_nope, d_v), dim=-1)
    )
    return q_nope.transpose(0, 1), w_uk.permute(1, 2, 0), q_pe


def cat_quant_fp8(ql_nope, q_pe, q_scale, fp8_max: float):
    """vLLM's _DecodeConcatQuantFP8 on its QuantFP8.forward_native path."""
    q = torch.cat((ql_nope, q_pe), dim=-1)
    q_flat = q.reshape(q.shape[0], -1)
    out = q_flat.to(torch.float32) * q_scale.to(torch.float32).reciprocal()
    out = out.clamp(-fp8_max, fp8_max).to(get_fp8_e4m3_dtype())
    return out.view(q.shape)


# vLLM compiles the op standalone with a dynamic token dim; the default
# automatic-dynamic mode reaches the same symbolic-M inductor kernel.
compiled_cat_quant_fp8 = torch.compile(cat_quant_fp8)


def ref_bmm_cat_quant_pad(q_nope, w_uk_t, q_pe, q_scale, num_padded_heads, fp8_max):
    """The unfused vLLM sequence: torch.bmm into a token-major buffer, the
    compiled cat + static fp8 quant, then get_mla_padded_q's repeat_interleave."""
    qh, m, _ = q_nope.shape
    ql_nope = q_nope.new_empty((m, qh, w_uk_t.shape[-1]))
    torch.bmm(q_nope, w_uk_t, out=ql_nope.transpose(0, 1))
    q = compiled_cat_quant_fp8(ql_nope, q_pe, q_scale, fp8_max)
    return q.repeat_interleave(num_padded_heads // qh, dim=1)


# M = 257 is above the largest M bucket, so it runs the "any" config with grouped tiles.
@pytest.mark.parametrize("M", [1, 15, 64, 257])
@pytest.mark.parametrize("QH, num_padded_heads", [(8, 16), (4, 16), (6, 12), (16, 16)])
@pytest.mark.parametrize("scale", [1.0, 0.37, 1e-3])
# K = 192 is not a multiple of BLOCK_SIZE_K, so it runs a masked K tail.
@pytest.mark.parametrize("d_nope", [D_NOPE, 192])
def test_fused_bf16_bmm_cat_quant_pad_mla(M, QH, num_padded_heads, scale, d_nope):
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"fused_bf16_bmm_cat_quant_pad_mla has no config for {get_arch()}")
    torch.manual_seed(0)
    q_nope, w_uk_t, q_pe = generate_inputs(M, QH, d_nope=d_nope)
    q_scale = torch.tensor(scale, dtype=torch.float32, device="cuda")
    fp8_max = vllm_fp8_max()

    out = fused_bf16_bmm_cat_quant_pad_mla(
        q_nope,
        w_uk_t,
        q_pe,
        q_scale,
        num_padded_heads,
        fp8_max=fp8_max,
    )
    ref = ref_bmm_cat_quant_pad(
        q_nope, w_uk_t, q_pe, q_scale, num_padded_heads, fp8_max
    )
    assert out.shape == ref.shape and out.dtype == ref.dtype

    # q_pe and the duplicated heads are pure data movement: bit-exact.
    assert torch.equal(
        out[..., KV_LORA:].view(torch.uint8), ref[..., KV_LORA:].view(torch.uint8)
    )
    dup = num_padded_heads // QH
    for d in range(1, dup):
        assert torch.equal(
            out[:, d::dup].view(torch.uint8), out[:, ::dup].view(torch.uint8)
        )

    # A different accumulation order can flip a bf16 rounding of the BMM result,
    # which moves the fp8 value by at most one step.
    out_nope, ref_nope = out[..., :KV_LORA], ref[..., :KV_LORA]
    torch.testing.assert_close(
        out_nope.float(), ref_nope.float(), atol=2**-9, rtol=2**-3
    )
    mismatch = (out_nope.view(torch.uint8) != ref_nope.view(torch.uint8)).float()
    assert mismatch.mean().item() < 0.01
