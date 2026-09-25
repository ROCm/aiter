# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (
    cache_format,
    paged_mxfp4_mqa_logits,
    unshuffle_scales,
    unshuffle_values,
)
from aiter.ops.triton.fusions.k_norm_rope_mxfp4_cache import k_norm_rope_mxfp4_cache
from aiter.ops.triton.rope.q_rope_mxfp4_quant import q_rope_mxfp4_quant

HEAD_SIZE, ROPE_DIM, SCALE_GROUP = 128, 64, 32
_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DEV = "cuda"


def _e2m1(x):
    """fp32 -> e2m1 code: nearest magnitude, ties to the even code, saturating;
    the sign bit is x's, so -0.0 packs as 8."""
    mag = torch.tensor(_MAG, device=x.device)
    d = (x.abs().unsqueeze(-1) - mag).abs()
    ties = d == d.min(dim=-1, keepdim=True).values
    codes = torch.arange(8, device=x.device)
    # A tie is between two adjacent codes, one of them even: prefer it.
    code = torch.where(ties, codes % 2 * 8 + codes, 99).argmin(dim=-1).to(torch.uint8)
    return torch.where(torch.signbit(x), code | 8, code)


def _quantize(x):
    """fp32 [..., D] -> (packed u8 [..., D // 2], e8m0 u8 [..., D // 32])."""
    *prefix, d = x.shape
    xb = x.reshape(*prefix, d // SCALE_GROUP, SCALE_GROUP)
    amax = xb.abs().amax(dim=-1).clamp(min=6.0 * 2**-126)
    lr = torch.ceil(torch.log2(amax * (1.0 / 6.0))).clamp(-127, 127)
    codes = _e2m1(xb * torch.exp2(-lr).unsqueeze(-1)).reshape(*prefix, d)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)), (lr + 127).to(torch.uint8)


def _dequantize(packed, e8m0):
    *prefix, dh = packed.shape
    grid = torch.tensor(_MAG, device=packed.device)
    grid = torch.cat([grid, -grid])
    nib = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(*prefix, dh * 2)
    vals = grid[nib.long()].reshape(*prefix, -1, SCALE_GROUP)
    return (vals * torch.exp2(e8m0.float() - 127).unsqueeze(-1)).reshape(*prefix, -1)


def _rope_gptj(x, cos, sin):
    """GPT-J (interleaved pairs) rotation of x's last 2 * cos.shape[-1] dims."""
    nope, rot = x[..., : x.shape[-1] - 2 * cos.shape[-1]], x[..., -2 * cos.shape[-1] :]
    even, odd = rot[..., 0::2], rot[..., 1::2]
    r = torch.stack([even * cos - odd * sin, odd * cos + even * sin], dim=-1)
    return torch.cat([nope, r.flatten(-2).bfloat16().float()], dim=-1)


def _cos_sin(max_pos, gen):
    angle = torch.rand(max_pos, ROPE_DIM // 2, generator=gen) * 6.28
    return torch.cat([angle.cos(), angle.sin()], dim=-1).to(DEV)


def _k_reference(k, positions, cos_sin, w, eps, ratio):
    kf = k.float()
    kn = (
        (kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + eps) * w)
        .bfloat16()
        .float()
    )
    cs = cos_sin[(positions // ratio) * ratio]
    return _quantize(_rope_gptj(kn, cs[:, : ROPE_DIM // 2], cs[:, ROPE_DIM // 2 :]))


def _paged_pool(num_pages, page_size, pitch_pad, fill):
    """Pages a pitch apart in one allocation, as vLLM keeps a layer's pages."""
    row = HEAD_SIZE // 2 + HEAD_SIZE // SCALE_GROUP
    pitch = page_size * row + pitch_pad
    pool = torch.full((num_pages * pitch,), fill, dtype=torch.uint8, device=DEV)
    return pool, pool.as_strided((num_pages, page_size, row), (pitch, row, 1))


def _shuffle(num_heads, page_size, scale_mode=1):
    """preshuffle_cache()'s pattern for this launch, as the cache op takes it.
    Scale mode 1 (the logits kernel's) splits a token's scales over
    64 // n_per_tile lanes; mode 0 keeps them whole."""
    fmt = cache_format(num_heads, HEAD_SIZE, page_size)
    n = fmt["n_per_tile"]
    lanes = 64 // n if scale_mode == 1 else HEAD_SIZE // SCALE_GROUP
    return n, fmt["d_per_tile"], lanes


def _read_back(cache, page_size, shuffle=None, scale_mode=1):
    """Natural-order (values, scales) [pages, page, .] from a written cache."""
    flat = cache.reshape(cache.shape[0], -1)
    hb, ns = HEAD_SIZE // 2, HEAD_SIZE // SCALE_GROUP
    vals = flat[:, : page_size * hb].reshape(-1, page_size, hb)
    scales = flat[:, page_size * hb :].reshape(-1, page_size, ns)
    if shuffle is not None:
        n, d, _ = shuffle
        vals = unshuffle_values(vals, n, d)
        scales = unshuffle_scales(scales, n, scale_mode)
    return vals, scales


def _k_case(num_tokens, page_size, ratio, seed):
    gen = torch.Generator().manual_seed(seed)
    num_pages = 2 * num_tokens // page_size + 3
    slots = torch.randperm(num_pages * page_size, generator=gen)[:num_tokens]
    slots[::7] = -1
    positions = torch.randint(0, 4096, (num_tokens,), generator=gen)
    return dict(
        k=torch.randn(num_tokens, HEAD_SIZE, generator=gen).bfloat16().to(DEV),
        positions=positions.to(DEV),
        slots=slots.to(DEV),
        w=(1 + 0.1 * torch.randn(HEAD_SIZE, generator=gen)).bfloat16().to(DEV),
        cos_sin=_cos_sin(4096, gen),
        num_pages=num_pages,
        ratio=ratio,
    )


@pytest.mark.parametrize("num_heads", [16, 32, 64])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("ratio", [1, 2])
def test_k_cache_layout(num_heads, page_size, ratio):
    """A shuffled store holds exactly the natural store's bytes in the order
    unshuffle_* undoes, in either scale order, and only the tokens that
    publish a key write."""
    c = _k_case(300, page_size, ratio, seed=num_heads + page_size + ratio)
    caches = []
    for scale_mode, shuffle in (
        (1, None),
        (1, _shuffle(num_heads, page_size)),
        (0, _shuffle(num_heads, page_size, scale_mode=0)),
    ):
        _, cache = _paged_pool(c["num_pages"], page_size, 512, fill=0xAA)
        k_norm_rope_mxfp4_cache(
            c["k"],
            c["positions"],
            c["cos_sin"],
            c["w"],
            1e-6,
            cache,
            c["slots"],
            ratio,
            shuffle=shuffle,
        )
        caches.append(_read_back(cache, page_size, shuffle, scale_mode))
    for shuffled in caches[1:]:
        for a, b in zip(caches[0], shuffled):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    vals, scales = caches[0]
    writes = (c["slots"] >= 0) & ((c["positions"] + 1) % ratio == 0)
    written = torch.zeros(c["num_pages"] * page_size, dtype=torch.bool, device=DEV)
    written[c["slots"][writes]] = True
    untouched = ~written.reshape(c["num_pages"], page_size)
    assert (vals[untouched] == 0xAA).all() and (scales[untouched] == 0xAA).all()


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("page_size", [1, 64])
def test_k_cache_matches_reference(ratio, page_size):
    """Values against k_norm -> RoPE -> MXFP4 in torch. Only rounding the two
    implementations order differently may differ: a bf16 ulp in the norm or an
    fp32 log2 at a power-of-two amax. The natural order takes any page size."""
    c = _k_case(512, page_size, ratio, seed=ratio)
    _, cache = _paged_pool(c["num_pages"], page_size, 0, fill=0)
    k_norm_rope_mxfp4_cache(
        c["k"], c["positions"], c["cos_sin"], c["w"], 1e-6, cache, c["slots"], ratio
    )
    vals, scales = _read_back(cache, page_size)
    writes = (c["slots"] >= 0) & ((c["positions"] + 1) % ratio == 0)
    slot = c["slots"][writes]
    got_v = vals.reshape(-1, HEAD_SIZE // 2)[slot]
    got_s = scales.reshape(-1, HEAD_SIZE // SCALE_GROUP)[slot]
    ref_v, ref_s = _k_reference(
        c["k"][writes],
        c["positions"][writes],
        c["cos_sin"],
        c["w"].float(),
        1e-6,
        ratio,
    )
    assert (got_v == ref_v).float().mean() > 0.995
    assert (got_s == ref_s).float().mean() > 0.995
    got, ref = _dequantize(got_v, got_s), _dequantize(ref_v, ref_s)
    assert (got - ref).abs().max() <= 2 * ref.abs().max() * 0.25


@pytest.mark.parametrize("shuffle", [(48, 16, 2), (32, 24, 2), (32, 16, 3)])
def test_k_cache_rejects_a_pattern_that_does_not_tile(shuffle):
    """A pattern that does not tile the page's tokens, value bytes or scales
    is refused before anything is written."""
    c = _k_case(8, 64, 1, seed=0)
    _, cache = _paged_pool(c["num_pages"], 64, 0, fill=0xAA)
    with pytest.raises(ValueError, match="does not tile"):
        k_norm_rope_mxfp4_cache(
            c["k"],
            c["positions"],
            c["cos_sin"],
            c["w"],
            1e-6,
            cache,
            c["slots"],
            shuffle=shuffle,
        )
    assert (cache == 0xAA).all()


@pytest.mark.parametrize(
    "num_tokens, num_heads, with_weights",
    [(64, 20, False), (1000, 64, True), (4200, 64, True)],
)
def test_q_quant_matches_reference(num_tokens, num_heads, with_weights):
    gen = torch.Generator().manual_seed(num_heads)
    t, h = num_tokens, num_heads
    q = (torch.randn(t, h + 3, HEAD_SIZE, generator=gen) * 3).bfloat16().to(DEV)
    q = q[:, :h]
    positions = torch.randint(0, 4096, (t,), generator=gen).to(DEV)
    cos_sin = _cos_sin(4096, gen)
    weights = torch.randn(t, h, generator=gen).bfloat16().to(DEV)
    q_packed, q_scale, w_out = q_rope_mxfp4_quant(
        q, positions, cos_sin, weights if with_weights else None, 0.25 * 0.125
    )

    cs = cos_sin[positions][:, None]
    ref_v, ref_s = _quantize(
        _rope_gptj(q.float(), cs[..., : ROPE_DIM // 2], cs[..., ROPE_DIM // 2 :])
    )
    # The NoPE half has no rounding the two could order differently.
    nope = (HEAD_SIZE - ROPE_DIM) // 2
    assert (q_packed[..., :nope] == ref_v[..., :nope]).all()
    assert (q_packed == ref_v).float().mean() > 0.995
    assert (q_scale == ref_s).float().mean() > 0.995
    if with_weights:
        torch.testing.assert_close(
            w_out, weights.float() * 0.25 * 0.125, rtol=0, atol=0
        )
    else:
        assert w_out is None


@pytest.mark.parametrize("page_size", [64, 128])
def test_round_trip_through_logits(page_size):
    """Keys written by the cache op and a query from the quant op score as the
    dequantized values do: the writer and paged_mxfp4_mqa_logits agree."""
    gen = torch.Generator().manual_seed(page_size)
    batch, next_n, num_heads, ctx = 2, 2, 32, 3 * page_size - 5
    per_seq = -(-ctx // page_size)
    num_pages = batch * per_seq + 2
    block_table = torch.randperm(num_pages - 1, generator=gen)[: batch * per_seq] + 1
    block_table = block_table.reshape(batch, per_seq).int().to(DEV)
    pos = torch.arange(ctx)
    slots = block_table.cpu()[:, pos // page_size] * page_size + pos % page_size
    k = torch.randn(batch * ctx, HEAD_SIZE, generator=gen).bfloat16().to(DEV)
    positions = pos.repeat(batch).to(DEV)
    cos_sin = _cos_sin(4096, gen)
    w = torch.ones(HEAD_SIZE, dtype=torch.bfloat16, device=DEV)
    _, cache = _paged_pool(num_pages, page_size, 256, fill=0)
    shuffle = _shuffle(num_heads, page_size)
    k_norm_rope_mxfp4_cache(
        k,
        positions,
        cos_sin,
        w,
        1e-6,
        cache,
        slots.reshape(-1).to(DEV),
        shuffle=shuffle,
    )

    q = (
        torch.randn(batch * next_n, num_heads, HEAD_SIZE, generator=gen)
        .bfloat16()
        .to(DEV)
    )
    q_pos = (ctx - next_n + torch.arange(next_n)).repeat(batch).to(DEV)
    weights = torch.randn(batch * next_n, num_heads, generator=gen).to(DEV)
    q_packed, q_scale, w_out = q_rope_mxfp4_quant(q, q_pos, cos_sin, weights)
    ctx_lens = torch.full((batch,), ctx, dtype=torch.int32, device=DEV)
    logits = paged_mxfp4_mqa_logits(
        q_packed.view(batch, next_n, num_heads, -1),
        q_scale.view(batch, next_n, num_heads, -1),
        cache.unsqueeze(2),
        w_out,
        ctx_lens,
        block_table,
        per_seq * page_size,
    )

    vals, scales = _read_back(cache, page_size, shuffle)
    kv = _dequantize(vals, scales)[block_table.long()].reshape(batch, -1, HEAD_SIZE)
    qd = _dequantize(q_packed, q_scale).reshape(batch, next_n, num_heads, HEAD_SIZE)
    for b in range(batch):
        for n in range(next_n):
            end = ctx - next_n + n + 1
            ref = (
                torch.relu(qd[b, n] @ kv[b, :end].T) * w_out[b * next_n + n, :, None]
            ).sum(0)
            torch.testing.assert_close(
                logits[b * next_n + n, :end], ref, rtol=1e-4, atol=1e-3
            )
