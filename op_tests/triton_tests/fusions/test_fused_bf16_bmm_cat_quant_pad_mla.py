# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_bf16_bmm_cat_quant_pad_mla import (
    fused_bf16_bmm_cat_quant_pad_mla,
    fused_bf16_bmm_rope_cat_quant_pad_cache_mla,
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


def check_against_ref(out, ref, num_padded_heads, qh):
    assert out.shape == ref.shape and out.dtype == ref.dtype

    # q_pe and the duplicated heads are pure data movement: bit-exact.
    assert torch.equal(
        out[..., KV_LORA:].view(torch.uint8), ref[..., KV_LORA:].view(torch.uint8)
    )
    dup = num_padded_heads // qh
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


def generate_rope_cache_inputs(
    M: int,
    kv_lora: int = KV_LORA,
    d_rope: int = D_ROPE,
    block_size: int = 16,
    max_pos: int = 4096,
):
    """kv_c and the unroped k_pe as strided views of one (M, kv_lora + d_rope)
    buffer, as vLLM's MLA passes them, a bf16 cos | sin cache, and slots in a
    paged cache with every 5th token from token 1 on padded (slot -1)."""
    kv_lora_out = torch.randn(
        (M, kv_lora + d_rope), dtype=torch.bfloat16, device="cuda"
    )
    kv_c, k_pe = kv_lora_out.split((kv_lora, d_rope), dim=-1)
    angles = torch.rand((max_pos, d_rope // 2), dtype=torch.float32, device="cuda")
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    positions = torch.randint(0, max_pos, (M,), device="cuda")
    num_blocks = M // block_size + 2
    slot_mapping = torch.randperm(num_blocks * block_size, device="cuda")[:M]
    slot_mapping[1::5] = -1
    kv_cache = torch.randint(
        0,
        256,
        (num_blocks, block_size, kv_lora + d_rope),
        dtype=torch.uint8,
        device="cuda",
    )
    return kv_c, k_pe, cos_sin_cache, positions, slot_mapping, kv_cache


def rope_vllm(x, positions, cos_sin_cache, is_neox):
    """vLLM's concat_and_cache_mla_rope_fused RoPE numerics. For a pair (a, b):
    a' = a*cos - bf16(b*sin), b' = a*sin + bf16(b*cos). The a products stay
    exact in fp32 because hipcc fuses them into FMAs."""
    half = x.shape[-1] // 2
    cs = cos_sin_cache[positions].float()
    cos, sin = cs[:, None, :half], cs[:, None, half:]
    if x.dim() == 2:
        cos, sin = cos[:, 0], sin[:, 0]
    a_idx, b_idx = (
        (slice(None, half), slice(half, None))
        if is_neox
        else (slice(0, None, 2), slice(1, None, 2))
    )
    a, b = x[..., a_idx].float(), x[..., b_idx].float()
    out = torch.empty_like(x)
    out[..., a_idx] = (a * cos - (b * sin).bfloat16().float()).bfloat16()
    out[..., b_idx] = (a * sin + (b * cos).bfloat16().float()).bfloat16()
    return out


def ref_rope_cache(
    q_pe,
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    k_scale,
    positions,
    cos_sin_cache,
    is_neox,
):
    """vLLM's concat_and_cache_mla_rope_fused on a uint8 fp8 cache: ropes q_pe
    and k_pe and writes fp8([kv_c | k_pe] / k_scale) to the slot. Tokens with
    slot < 0 keep q_pe unroped and do not write the cache."""
    valid = slot_mapping >= 0
    q_pe = torch.where(
        valid[:, None, None], rope_vllm(q_pe, positions, cos_sin_cache, is_neox), q_pe
    )
    k_pe = rope_vllm(k_pe, positions, cos_sin_cache, is_neox)
    fp8 = get_fp8_e4m3_dtype()
    entry = torch.cat((kv_c, k_pe), dim=-1).float() / k_scale
    entry = entry.clamp(-torch.finfo(fp8).max, torch.finfo(fp8).max).to(fp8)
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    flat[slot_mapping[valid]] = entry.view(torch.uint8)[valid]
    return q_pe


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
    check_against_ref(out, ref, num_padded_heads, QH)


@pytest.mark.parametrize("M", [1, 15, 64, 257])
@pytest.mark.parametrize("QH, num_padded_heads", [(8, 16), (16, 16)])
@pytest.mark.parametrize("is_neox", [False, True])
# k_scale = 1e-3 saturates the fp8 cache.
@pytest.mark.parametrize("k_scale", [1.0, 1e-3])
def test_fused_bf16_bmm_rope_cat_quant_pad_cache_mla(
    M, QH, num_padded_heads, is_neox, k_scale
):
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(
            f"fused_bf16_bmm_rope_cat_quant_pad_cache_mla has no config for {get_arch()}"
        )
    torch.manual_seed(0)
    q_nope, w_uk_t, q_pe = generate_inputs(M, QH)
    kv_c, k_pe, cos_sin_cache, positions, slot_mapping, kv_cache = (
        generate_rope_cache_inputs(M)
    )
    q_scale = torch.tensor(0.37, dtype=torch.float32, device="cuda")
    k_scale = torch.tensor(k_scale, dtype=torch.float32, device="cuda")
    fp8_max = vllm_fp8_max()
    fp8 = get_fp8_e4m3_dtype()
    ref_cache = kv_cache.clone()

    out = fused_bf16_bmm_rope_cat_quant_pad_cache_mla(
        q_nope,
        w_uk_t,
        q_pe,
        q_scale,
        kv_c,
        k_pe,
        kv_cache.view(fp8),
        slot_mapping,
        k_scale,
        positions,
        cos_sin_cache,
        is_neox,
        num_padded_heads,
        fp8_max=fp8_max,
    )
    roped_q_pe = ref_rope_cache(
        q_pe,
        kv_c,
        k_pe,
        ref_cache,
        slot_mapping,
        k_scale,
        positions,
        cos_sin_cache,
        is_neox,
    )
    ref = ref_bmm_cat_quant_pad(
        q_nope, w_uk_t, roped_q_pe, q_scale, num_padded_heads, fp8_max
    )
    check_against_ref(out, ref, num_padded_heads, QH)
    # The cache is pure data movement too; padded tokens leave their bytes alone.
    assert torch.equal(kv_cache, ref_cache)
