# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the separated-rope sparse MLA decode API (gfx950 gluon):
``aiter.ops.triton.attention.sparse_mla_decode.sparse_mla_decode_fwd``.

Covers the three cache formats (bf16, flat per-tensor fp8, vLLM fp8_ds_mla),
both query-head counts GLM-5 ships (8 = TP8, 16 = TP4), ragged and -1-padded
index streams, split-K on/off, the vLLM cache shapes, and the 64-bit (>2 GB
emulated) gather path. The per-tensor scale-folding is additionally pinned by a
bitwise test against the bf16 path.
"""

import os

import pytest
import torch

from aiter.ops.triton.utils._triton import arch_info

if arch_info.get_arch() == "gfx950":
    from aiter.ops.triton.attention.sparse_mla_decode import sparse_mla_decode_fwd

FP8_MAX = 448.0
TINY = 1.1754944e-38
KV_LORA, ROPE = 512, 64
D_QK = KV_LORA + ROPE


def _skip_unless_gfx950():
    if arch_info.get_arch() != "gfx950":
        pytest.skip("sparse_mla_decode_fwd is gfx950-only")


# ---------------------------------------------------------------- packers


def quantize_flat_fp8(kv):
    """GLM-5 production layout: whole row (rope included) fp8 with ONE
    per-tensor scale (what concat_and_cache_mla writes with layer._k_scale)."""
    scale = (kv.float().abs().amax() / FP8_MAX).clamp_min(TINY).reshape(1)
    q8 = (kv.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q8.view(torch.uint8), scale.to(torch.float32)


def pack_ds_mla(kv, block_size=64):
    """vLLM ``fp8_ds_mla``: 656-byte rows [512 fp8 | 4 x f32 per-128 | 64 bf16]."""
    T = kv.shape[0]
    nb = (T + block_size - 1) // block_size
    cache = torch.zeros(nb * block_size, 656, dtype=torch.uint8, device=kv.device)
    x = kv[:, :KV_LORA].float().reshape(T, 4, 128)
    scale = (x.abs().amax(-1) / FP8_MAX).clamp_min(TINY)
    q8 = (x / scale[..., None]).to(torch.float8_e4m3fn).view(torch.uint8)
    cache[:T, :KV_LORA] = q8.reshape(T, KV_LORA)
    cache[:T, 512:528].view(torch.float32).copy_(scale)
    cache[:T, 528:].view(torch.bfloat16).copy_(kv[:, KV_LORA:])
    return cache.view(nb, block_size, 656)


def dequant_flat_fp8(u8, scale):
    return u8.view(torch.float8_e4m3fn).float() * scale


def dequant_ds_mla(cache):
    flat = cache.reshape(-1, 656)
    x = flat[:, :KV_LORA].view(torch.float8_e4m3fn).float().reshape(-1, 4, 128)
    sc = flat[:, 512:528].view(torch.float32)
    nope = (x * sc[..., None]).reshape(-1, KV_LORA)
    rope = flat[:, 528:].view(torch.bfloat16).float()
    return torch.cat([nope, rope], dim=-1)


def ragged_indices(C, pool, topk, device, seed, mode):
    """mode: 'uniform' (full topk), 'ragged' (true prefix sum, no sentinels --
    the production shape), 'padded' (-1 tail padding, needs has_invalid)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lens = (
        torch.full((C,), topk)
        if mode == "uniform"
        else torch.randint(1, topk + 1, (C,), generator=g)
    )
    rows, flat_lens = [], []
    for c in range(C):
        k = int(lens[c])
        sel = torch.randperm(pool, generator=g)[:k]
        if mode == "padded":
            row = torch.full((topk,), -1, dtype=torch.int64)
            row[:k] = sel
            rows.append(row)
            flat_lens.append(topk)
        else:
            rows.append(sel)
            flat_lens.append(k)
    indices = torch.cat(rows)
    indptr = torch.zeros(C + 1, dtype=torch.int64)
    indptr[1:] = torch.tensor(flat_lens).cumsum(0)
    return (
        indices.to(torch.int32).to(device),
        indptr.to(torch.int32).to(device),
    )


def reference(q, kv_truth, indices, indptr, sm_scale):
    """f32 attention over the dequantized cache (tests the kernel, not fp8)."""
    C = q.shape[0]
    outs = []
    for c in range(C):
        sel = indices[indptr[c] : indptr[c + 1]].long()
        sel = sel[sel >= 0]
        kvs = kv_truth[sel].float()
        s = torch.einsum("hd,td->ht", q[c].float(), kvs) * sm_scale
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("ht,td->hd", p, kvs[:, :KV_LORA]))
    return torch.stack(outs)


def rel_err(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-20)).item()


def _build(fmt, C, H, topk, pool, mode, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(C, H, D_QK, dtype=torch.bfloat16, device="cuda") * 0.125
    kv = torch.randn(pool, D_QK, dtype=torch.bfloat16, device="cuda") * 0.125
    kv[:: max(1, pool // 97)] *= 7.0  # a few heavy-tail tokens
    idx, ptr = ragged_indices(C, pool, topk, "cuda", seed + 7, mode)
    ks = None
    if fmt == "bf16":
        cache, truth = kv, kv.float()
    elif fmt == "tensor":
        cache, ks = quantize_flat_fp8(kv)
        truth = dequant_flat_fp8(cache, ks)
    else:
        cache = pack_ds_mla(kv)
        truth = dequant_ds_mla(cache)[:pool]
    return q, cache, ks, idx, ptr, truth


# ---------------------------------------------------------------- tests


@pytest.mark.parametrize("fmt", ["bf16", "tensor", "dsmla"])
@pytest.mark.parametrize("H", [8, 16])
@pytest.mark.parametrize(
    "topk,mode",
    [(2048, "uniform"), (500, "ragged"), (2048, "padded")],
    ids=["topk2048", "ragged500", "padded2048"],
)
def test_sparse_mla_decode(fmt, H, topk, mode):
    _skip_unless_gfx950()
    C, pool = 8, 1 << 16
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, truth = _build(fmt, C, H, topk, pool, mode)
    ref = reference(q, truth.to(torch.bfloat16), idx, ptr, sm)
    out = sparse_mla_decode_fwd(
        q, cache, ptr, idx, sm, kv_scale=ks, has_invalid=(mode == "padded")
    )
    e = rel_err(out, ref)
    assert e < 2e-2, f"{fmt} H={H} topk={topk} {mode}: rel-err {e:.3e}"


@pytest.mark.parametrize("kv_splits", [1, None], ids=["splits1", "splitsauto"])
def test_split_k(kv_splits):
    _skip_unless_gfx950()
    C, pool, topk = 4, 1 << 15, 2048
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, truth = _build("tensor", C, 16, topk, pool, "uniform")
    ref = reference(q, truth.to(torch.bfloat16), idx, ptr, sm)
    out = sparse_mla_decode_fwd(q, cache, ptr, idx, sm, kv_scale=ks,
                                kv_splits=kv_splits)
    e = rel_err(out, ref)
    assert e < 2e-2, f"splits={kv_splits}: rel-err {e:.3e}"


def test_per_tensor_scale_folding_bitwise():
    """kv_scale=1.0 on fp8-exact values must equal the bf16 path bit-for-bit:
    the K-scale fold into qk_scale and the V-scale fold into p are exact, and
    fp8 -> bf16 staging is exact."""
    _skip_unless_gfx950()
    torch.manual_seed(3)
    C, H, topk, pool = 4, 16, 512, 1 << 15
    sm = D_QK**-0.5
    raw = torch.randn(pool, D_QK, dtype=torch.bfloat16, device="cuda") * 0.25
    exact = raw.float().to(torch.float8_e4m3fn)
    kv_bf16 = exact.to(torch.bfloat16)
    cache_u8 = exact.view(torch.uint8)
    ks = torch.ones(1, dtype=torch.float32, device="cuda")
    q = torch.randn(C, H, D_QK, dtype=torch.bfloat16, device="cuda") * 0.125
    idx, ptr = ragged_indices(C, pool, topk, "cuda", 11, "ragged")
    o_t = sparse_mla_decode_fwd(q, cache_u8, ptr, idx, sm, kv_scale=ks)
    o_b = sparse_mla_decode_fwd(q, kv_bf16, ptr, idx, sm)
    assert torch.equal(o_t.view(torch.int16), o_b.view(torch.int16))


def test_vllm_cache_shapes():
    """The [slots,1,1,R] (asm mla_decode_fwd) and [nb,block,R] (vLLM paged)
    views must produce identical results to the flat [slots,R] form."""
    _skip_unless_gfx950()
    C, pool, topk = 4, 1 << 14, 512
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, _ = _build("tensor", C, 16, topk, pool, "uniform")
    o_flat = sparse_mla_decode_fwd(q, cache, ptr, idx, sm, kv_scale=ks)
    o_4d = sparse_mla_decode_fwd(
        q, cache.view(pool, 1, 1, D_QK), ptr, idx, sm, kv_scale=ks
    )
    o_paged = sparse_mla_decode_fwd(
        q, cache.view(pool // 64, 64, D_QK), ptr, idx, sm, kv_scale=ks
    )
    assert torch.equal(o_flat, o_4d) and torch.equal(o_flat, o_paged)


@pytest.mark.parametrize("fmt", ["tensor", "dsmla"])
def test_global_load_path(fmt, monkeypatch):
    """Force the 64-bit gather path (production: pools cross buffer_load's 2 GB
    offset limit at ~3.7M tokens) and check correctness is unchanged."""
    _skip_unless_gfx950()
    monkeypatch.setenv("AITER_PA_DECODE_FORCE_GLOBAL_LOAD", "1")
    C, pool, topk = 8, 1 << 16, 2048
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, truth = _build(fmt, C, 16, topk, pool, "uniform")
    ref = reference(q, truth.to(torch.bfloat16), idx, ptr, sm)
    out = sparse_mla_decode_fwd(q, cache, ptr, idx, sm, kv_scale=ks)
    e = rel_err(out, ref)
    assert e < 2e-2, f"{fmt} 64-bit path: rel-err {e:.3e}"
