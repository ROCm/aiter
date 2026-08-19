# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``sparse_mla_decode_fwd`` (gfx950 gluon, separated-rope MLA).

Target-oriented: the production formats (bf16 and flat per-tensor fp8) get the
shape matrix; fp8_ds_mla gets a single smoke case. Index streams are uniform or
ragged with no -1 sentinels, matching what vLLM's converter emits.
"""

import pytest
import torch

from aiter.ops.triton.utils._triton import arch_info

if arch_info.get_arch() == "gfx950":
    import aiter.ops.triton.attention.sparse_mla_decode as smd
    from aiter.ops.triton.attention.sparse_mla_decode import sparse_mla_decode_fwd

FP8_MAX = 448.0
KV_LORA, ROPE = 512, 64
D_QK = KV_LORA + ROPE


def _skip_unless_gfx950():
    if arch_info.get_arch() != "gfx950":
        pytest.skip("sparse_mla_decode_fwd is gfx950-only")


def quantize_flat_fp8(kv):
    """vLLM's production layout: whole row fp8 with one per-tensor scale."""
    scale = (kv.float().abs().amax() / FP8_MAX).clamp_min(1e-30).reshape(1)
    q8 = (kv.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q8.view(torch.uint8), scale.to(torch.float32)


def pack_ds_mla(kv, block_size=64):
    """vLLM fp8_ds_mla: 656 B rows [512 fp8 | 4 x f32 per-128 | 64 bf16 rope]."""
    T = kv.shape[0]
    nb = (T + block_size - 1) // block_size
    cache = torch.zeros(nb * block_size, 656, dtype=torch.uint8, device=kv.device)
    x = kv[:, :KV_LORA].float().reshape(T, 4, 128)
    scale = (x.abs().amax(-1) / FP8_MAX).clamp_min(1e-30)
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


def make_indices(C, pool, topk, device, seed, ragged):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lens = (
        torch.randint(1, topk + 1, (C,), generator=g)
        if ragged
        else torch.full((C,), topk)
    )
    rows = [torch.randperm(pool, generator=g)[: int(lens[c])] for c in range(C)]
    indptr = torch.zeros(C + 1, dtype=torch.int64)
    indptr[1:] = lens.cumsum(0)
    return (
        torch.cat(rows).to(torch.int32).to(device),
        indptr.to(torch.int32).to(device),
    )


def reference(q, kv_truth, indices, indptr, sm_scale):
    """f32 attention over the dequantized cache (tests the kernel, not fp8)."""
    outs = []
    for c in range(q.shape[0]):
        kvs = kv_truth[indices[indptr[c] : indptr[c + 1]].long()].float()
        s = torch.einsum("hd,td->ht", q[c].float(), kvs) * sm_scale
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("ht,td->hd", p, kvs[:, :KV_LORA]))
    return torch.stack(outs)


def rel_err(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


def _build(fmt, C, H, topk, pool, ragged, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(C, H, D_QK, dtype=torch.bfloat16, device="cuda") * 0.125
    kv = torch.randn(pool, D_QK, dtype=torch.bfloat16, device="cuda") * 0.125
    idx, ptr = make_indices(C, pool, topk, "cuda", seed + 7, ragged)
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


def _run_and_check(fmt, C, H, topk, ragged, tol=2e-2, pool=1 << 16, **kwargs):
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, truth = _build(fmt, C, H, topk, pool, ragged)
    ref = reference(q, truth.to(torch.bfloat16), idx, ptr, sm)
    out = sparse_mla_decode_fwd(q, cache, ptr, idx, sm, kv_scale=ks, **kwargs)
    e = rel_err(out, ref)
    assert e < tol, f"{fmt} C={C} H={H} topk={topk} ragged={ragged}: rel-err {e:.3e}"


@pytest.mark.parametrize("fmt", ["bf16", "tensor"])
@pytest.mark.parametrize("H", [8, 16])
@pytest.mark.parametrize(
    "topk,ragged", [(2048, False), (500, True)], ids=["topk2048", "ragged500"]
)
def test_sparse_mla_decode(fmt, H, topk, ragged):
    _skip_unless_gfx950()
    _run_and_check(fmt, C=8, H=H, topk=topk, ragged=ragged)


def test_ds_mla_format():
    _skip_unless_gfx950()
    _run_and_check("dsmla", C=8, H=16, topk=2048, ragged=True)


@pytest.mark.parametrize("kv_splits", [1, None], ids=["splits1", "splitsauto"])
def test_split_k(kv_splits):
    _skip_unless_gfx950()
    _run_and_check("tensor", C=4, H=16, topk=2048, ragged=False,
                   pool=1 << 15, kv_splits=kv_splits)


def test_per_tensor_scale_folding_bitwise():
    """kv_scale=1.0 on fp8-exact values must equal the bf16 path bit-for-bit
    (K-scale folds into qk_scale, V-scale into p; fp8 -> bf16 is exact)."""
    _skip_unless_gfx950()
    torch.manual_seed(3)
    C, H, topk, pool = 4, 16, 512, 1 << 15
    sm = D_QK**-0.5
    exact = (torch.randn(pool, D_QK, dtype=torch.bfloat16, device="cuda") * 0.25
             ).float().to(torch.float8_e4m3fn)
    ks = torch.ones(1, dtype=torch.float32, device="cuda")
    q = torch.randn(C, H, D_QK, dtype=torch.bfloat16, device="cuda") * 0.125
    idx, ptr = make_indices(C, pool, topk, "cuda", 11, ragged=True)
    o_t = sparse_mla_decode_fwd(q, exact.view(torch.uint8), ptr, idx, sm, kv_scale=ks)
    o_b = sparse_mla_decode_fwd(q, exact.to(torch.bfloat16), ptr, idx, sm)
    assert torch.equal(o_t.view(torch.int16), o_b.view(torch.int16))


def test_vllm_cache_shapes():
    """[slots,1,1,R] (asm view) and [nb,block,R] (vLLM paged) must match the
    flat [slots,R] form exactly."""
    _skip_unless_gfx950()
    C, pool, topk = 4, 1 << 14, 512
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, _ = _build("tensor", C, 16, topk, pool, ragged=False)
    o_flat = sparse_mla_decode_fwd(q, cache, ptr, idx, sm, kv_scale=ks)
    o_4d = sparse_mla_decode_fwd(
        q, cache.view(pool, 1, 1, D_QK), ptr, idx, sm, kv_scale=ks
    )
    o_paged = sparse_mla_decode_fwd(
        q, cache.view(pool // 64, 64, D_QK), ptr, idx, sm, kv_scale=ks
    )
    assert torch.equal(o_flat, o_4d) and torch.equal(o_flat, o_paged)


def test_global_load_path(monkeypatch):
    """Force the 64-bit gather path (production pools cross buffer_load's 2 GB
    offset limit at ~3.7M tokens)."""
    _skip_unless_gfx950()
    monkeypatch.setattr(smd, "MAX_BYTES", 0)
    _run_and_check("tensor", C=8, H=16, topk=2048, ragged=False)
