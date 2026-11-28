# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the fused DSA indexer prepare op (block-FP8 indexer weights).

The reference projects in fp64 after the same 1x128 activation quant the
kernel applies, rounds to bf16 where every implementation materializes bf16
(GEMM output, after the norm, after RoPE, after the Hadamard), and compares
the dequantized fp8 values relative to each row's amax: one e4m3 code at the
top binade is 32/448 of amax, so that is the bound. The ue8m0 scale may land
one power of two away from the reference when a row's amax sits on a binade
edge under a different rounding chain; the value check covers that case.
"""

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight

device = "cuda"

H, D, HID, QK, PAGE, STRIDE = 32, 128, 6144, 2048, 64, 132
EPS = 1e-6
FP8 = torch.float8_e4m3fn
WEIGHTS_SCALE = D**-0.5 * H**-0.5
# One e4m3 code at the top binade, relative to the row amax.
VAL_TOL = 32 / 448
# Served decode rows (GLM-5.2 FP8 TP4/TP8/EP8) plus the non-power-of-two sizes
# a CUDA-graph capture can ask for, which run the masked 16-row tiles.
_TOKENS = [1, 2, 3, 4, 8, 12, 16, 32]
_SEEDS = [0, 100]  # benign / cancellation draw


def _import_op():
    try:
        import triton.experimental.gluon  # noqa: F401
    except ImportError:
        pytest.skip("Triton Gluon not available")
    from aiter.ops.triton.attention.dsa_indexer_prepare import dsa_indexer_prepare

    return dsa_indexer_prepare


def _requires_gfx950():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip(f"Gluon DSA indexer prepare is gfx950-only, got {get_arch()}")


def make_inputs(tokens, seed, hidden=HID, q_rank=QK, heads=H, pages=4096, pad=0):
    """seed < 100: benign draws. seed >= 100: cancellation draws -- x = [a, -a]
    over the two K halves with the second weight half equal to the first except
    ~1/128 of the bytes nudged by one code, so every projection is a sum of
    large partials that cancel to a small residual. ``pad`` trailing slots are -1."""
    g = torch.Generator(device=device).manual_seed(seed)
    cancel = seed >= 100

    def randn(*shape, scale=1.0):
        return (
            torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * scale
        )

    def rand(*shape):
        return torch.rand(*shape, generator=g, device=device, dtype=torch.float32)

    def activation(rows, k):
        if not cancel:
            return randn(rows, k).to(torch.bfloat16)
        a = randn(rows, k // 2).to(torch.bfloat16)
        return torch.cat([a, -a], 1).contiguous()

    def fp8_weight(n, k):
        if not cancel:
            return randn(n, k).clamp(-448, 448).to(FP8)
        wa = randn(n, k // 2).clamp(-448, 448).to(FP8).view(torch.uint8)
        mag, sign = wa & 0x7F, wa & 0x80
        nudge = (rand(n, k // 2) < 1 / 128).to(torch.int16) * torch.where(
            rand(n, k // 2) < 0.5, -1, 1
        ).to(torch.int16)
        wb = (mag.to(torch.int16) + nudge).clamp(0, 0x7E).to(torch.uint8) | sign
        return torch.cat([wa, wb], 1).contiguous().view(FP8)

    def block_scales(rows, groups):
        if not cancel:
            return (0.5 + rand(rows, groups)) * 2e-2
        s = (0.5 + rand(rows, groups // 2)) * 2e-2
        return torch.cat([s, s], 1).contiguous()

    def bf16_weight(n, k):
        if not cancel:
            return randn(n, k, scale=k**-0.5).to(torch.bfloat16)
        wa = randn(n, k // 2).to(torch.bfloat16)
        nudge = (rand(n, k // 2) < 1 / 128).to(torch.int16)
        wb = (wa.view(torch.int16) + nudge).view(torch.bfloat16)
        return torch.cat([wa, wb], 1).contiguous()

    max_pos = 8192
    angles = randn(max_pos, 32, scale=3.0)
    slots = torch.randperm(pages * PAGE, generator=g, device=device)[:tokens]
    if pad:
        slots[tokens - pad :] = -1
    wq_b = fp8_weight(heads * D, q_rank)
    wk = fp8_weight(D, hidden)
    return {
        "x": activation(tokens, hidden),
        "q_lora": activation(tokens, q_rank),
        "wq_b_plain": wq_b,
        "wq_b": shuffle_weight(wq_b, layout=(16, 16)),
        "wq_b_scale": block_scales(heads, q_rank // 128),
        "wk_plain": wk,
        "wk": shuffle_weight(wk, layout=(16, 16)),
        "wk_scale": block_scales(1, hidden // 128),
        "weights_proj": bf16_weight(heads, hidden),
        "norm_weight": (1.0 + randn(D, scale=0.1)).to(torch.bfloat16),
        "norm_bias": randn(D, scale=0.1).to(torch.bfloat16),
        "positions": torch.randint(0, max_pos, (tokens,), generator=g, device=device),
        "cos_cache": angles.cos().to(torch.bfloat16).contiguous(),
        "sin_cache": angles.sin().to(torch.bfloat16).contiguous(),
        "slot_mapping": slots.contiguous(),
        "kv_cache": torch.randint(
            0, 256, (pages, PAGE, STRIDE), generator=g, device=device, dtype=torch.uint8
        ),
    }


def _bf16(t):
    return t.to(torch.bfloat16).float()


def _hadamard(n):
    i = torch.arange(n, device=device)
    bits = i[:, None] & i[None, :]
    parity = torch.zeros_like(bits)
    while bool((bits > 0).any()):
        parity ^= bits & 1
        bits >>= 1
    return 1.0 - 2.0 * parity.float()


def _rope_interleaved(x, cos, sin):
    x = x.clone()
    even, odd = x[..., 0:64:2].clone(), x[..., 1:64:2].clone()
    x[..., 0:64:2] = even * cos - odd * sin
    x[..., 1:64:2] = odd * cos + even * sin
    return x


def _ue8m0_quant(v):
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (v / scale).to(FP8), scale.squeeze(-1)


def _fp8_projection(x, w_plain, w_scale):
    """fp64 block-FP8 projection with the kernel's 1x128 activation quant."""
    m, k = x.shape
    n = w_plain.shape[0]
    xg = x.float().view(m, k // 128, 128)
    sx = xg.abs().amax(-1).clamp_min(1e-10) / 448.0
    xq = (xg / sx[..., None]).to(FP8).double()
    w = w_plain.double().view(n, k // 128, 128)
    ws = w_scale.double()[torch.arange(n, device=device) // 128]
    partial = torch.einsum("mgk,ngk->mng", xq, w)
    return (partial * sx.double()[:, None, :] * ws[None]).sum(-1).float()


def ref_prepare(inp, hadamard=True):
    pos = inp["positions"]
    cos, sin = inp["cos_cache"].float()[pos], inp["sin_cache"].float()[pos]
    hm = _hadamard(D) * D**-0.5
    k = _bf16(_fp8_projection(inp["x"], inp["wk_plain"], inp["wk_scale"]))
    mu = k.mean(-1, keepdim=True)
    var = ((k - mu) ** 2).mean(-1, keepdim=True)
    k = _bf16(
        (k - mu) * torch.rsqrt(var + EPS) * inp["norm_weight"].float()
        + inp["norm_bias"].float()
    )
    k = _bf16(_rope_interleaved(k, cos, sin))
    if hadamard:
        k = _bf16(k @ hm.t())
    heads = inp["weights_proj"].shape[0]
    q = _bf16(
        _fp8_projection(inp["q_lora"], inp["wq_b_plain"], inp["wq_b_scale"])
    ).view(-1, heads, D)
    q = _bf16(_rope_interleaved(q, cos[:, None], sin[:, None]))
    if hadamard:
        q = _bf16(q @ hm.t())
    q_codes, q_scale = _ue8m0_quant(q)
    gate = _bf16(inp["x"].double() @ inp["weights_proj"].double().t())
    return {
        "k": k,
        "q": q,
        "q_codes": q_codes,
        "q_scale": q_scale,
        "weights": gate * q_scale * WEIGHTS_SCALE,
    }


def decode_cache(cache, slots):
    """Preshuffled index-K cache -> (codes [T, 128], scales [T], code byte index, scale byte index)."""
    flat = cache.view(torch.uint8).reshape(-1)
    s = slots.clamp_min(0)
    page, off = s // PAGE, s % PAGE
    d = torch.arange(D, device=device)
    base = page * PAGE * STRIDE
    idx = (
        base[:, None]
        + (off // 16 * 2048)[:, None]
        + (d // 16 * 256)[None, :]
        + (off % 16 * 16)[:, None]
        + (d % 16)[None, :]
    )
    sidx = base + PAGE * D + off * 4
    scales = (
        torch.stack([flat[sidx + i] for i in range(4)], -1)
        .contiguous()
        .view(torch.float32)
        .view(-1)
    )
    return flat[idx].view(FP8), scales, idx, sidx


def _assert_values(codes, scale, ref_val, what):
    deq = codes.float() * scale.float().unsqueeze(-1)
    amax = ref_val.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    err = ((deq - ref_val).abs() / amax).max().item()
    assert err <= VAL_TOL, f"{what}: max |deq - ref| / amax = {err:.4f} > {VAL_TOL:.4f}"


def _run(inp, key_only=False, **kwargs):
    op = _import_op()
    return op(
        inp["x"],
        None if key_only else inp["q_lora"],
        None if key_only else inp["wq_b"],
        None if key_only else inp["wq_b_scale"],
        inp["wk"],
        inp["wk_scale"],
        None if key_only else inp["weights_proj"],
        inp["norm_weight"],
        inp["norm_bias"],
        inp["positions"],
        inp["cos_cache"],
        inp["sin_cache"],
        inp["slot_mapping"],
        inp["kv_cache"],
        epsilon=EPS,
        weights_scale=WEIGHTS_SCALE,
        key_only=key_only,
        **kwargs,
    )


def _check_cache(inp, before, ref):
    slots = inp["slot_mapping"]
    valid = slots >= 0
    codes, scale, idx, sidx = decode_cache(inp["kv_cache"], slots)
    _assert_values(codes[valid], scale[valid], ref["k"][valid], "index-K cache")
    touched = torch.zeros(inp["kv_cache"].numel(), dtype=torch.bool, device=device)
    touched[idx[valid].reshape(-1)] = True
    for i in range(4):
        touched[sidx[valid] + i] = True
    after = inp["kv_cache"].reshape(-1)
    assert torch.equal(
        after[~touched], before.reshape(-1)[~touched]
    ), "bytes outside the written slots changed"


# --------------------------------------------------------------------------
# The reference is itself under test.
# --------------------------------------------------------------------------


def test_reference_hadamard_is_orthogonal():
    hm = _hadamard(D) * D**-0.5
    torch.testing.assert_close(
        hm @ hm.t(), torch.eye(D, device=device), atol=1e-5, rtol=0
    )


def test_reference_cache_decode_roundtrip():
    inp = make_inputs(8, 0)
    codes, scale, idx, sidx = decode_cache(inp["kv_cache"], inp["slot_mapping"])
    assert codes.shape == (8, D) and scale.shape == (8,)
    assert idx.unique().numel() == 8 * D and sidx.unique().numel() == 8


# --------------------------------------------------------------------------
# Kernel tests.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", _TOKENS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_dsa_indexer_prepare_full(tokens, seed):
    _requires_gfx950()
    inp = make_inputs(tokens, seed, pad=max(1, tokens // 8) if seed % 2 else 0)
    before = inp["kv_cache"].clone()
    ref = ref_prepare(inp)
    q_fp8, weights = _run(inp)
    torch.cuda.synchronize()
    assert q_fp8.shape == (tokens, H, D) and q_fp8.dtype is FP8
    assert weights.shape == (tokens, H) and weights.dtype is torch.float32
    _check_cache(inp, before, ref)
    # The kernel's ue8m0 q scale is folded into the weights. Where a head's amax
    # sits on a binade edge it may be one power of two off the reference; read
    # that flip back from the weights ratio and dequantize q at the kernel's scale.
    ratio = weights / ref["weights"].abs().clamp_min(1e-30) * ref["weights"].sign()
    flip = torch.round(torch.log2(ratio.abs().clamp_min(1e-30))).clamp(-1, 1)
    assert (
        flip != 0
    ).float().mean().item() <= 0.02, "too many ue8m0 scale flips vs reference"
    kernel_scale = ref["q_scale"] * torch.exp2(flip)
    _assert_values(q_fp8, kernel_scale, ref["q"], "q_fp8")
    # weights = bf16(gate) * q_scale * weights_scale. The gate is a bf16-rounded
    # fp32 accumulation: rtol 1e-2 covers one bf16 ulp and the accumulation order.
    # On the cancellation draw the gate is a small residual of large partials, so
    # the accumulation-order error is bounded against the magnitude budget instead.
    expect = ref["weights"] * torch.exp2(flip)
    budget = (
        (inp["x"].float().abs() @ inp["weights_proj"].float().abs().t())
        * kernel_scale
        * WEIGHTS_SCALE
    )
    atol = budget * 2.0**-16 if seed >= 100 else torch.zeros_like(budget)
    bad = (weights - expect).abs() > 1e-2 * expect.abs() + atol
    assert (
        not bad.any()
    ), f"weights: {bad.float().mean().item():.4f} of entries beyond tolerance"


@pytest.mark.parametrize("tokens", _TOKENS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_dsa_indexer_prepare_key_only(tokens, seed):
    _requires_gfx950()
    inp = make_inputs(tokens, seed, pad=1 if seed % 2 else 0)
    before = inp["kv_cache"].clone()
    ref = ref_prepare(inp)
    out = _run(inp, key_only=True)
    torch.cuda.synchronize()
    assert out == (None, None)
    _check_cache(inp, before, ref)


def test_dsa_indexer_prepare_matches_aiter_writer_cache_layout():
    """The cache bytes must be what ``indexer_k_quant_and_cache(preshuffle=True)``
    writes for the same (bf16, un-rotated) key, so the two writers can share a pool."""
    _requires_gfx950()
    from aiter.ops.cache import indexer_k_quant_and_cache

    inp = make_inputs(16, 0)
    ref = ref_prepare(inp, hadamard=False)
    other = inp["kv_cache"].clone()
    # The HIP writer takes the normalized, rotated bf16 key; it does not apply the Hadamard.
    indexer_k_quant_and_cache(
        ref["k"].to(torch.bfloat16),
        other.view(FP8),
        inp["slot_mapping"],
        128,
        "ue8m0",
        preshuffle=True,
    )
    codes_a, scale_a, _, _ = decode_cache(other, inp["slot_mapping"])
    _assert_values(
        codes_a, scale_a, ref["k"], "aiter writer vs reference (layout decode)"
    )
    _run(inp)
    torch.cuda.synchronize()
    codes_b, scale_b, _, _ = decode_cache(inp["kv_cache"], inp["slot_mapping"])
    # Same layout: both decode to their own key; the values differ by the Hadamard only.
    assert codes_b.shape == codes_a.shape and torch.isfinite(scale_b).all()


def test_dsa_indexer_prepare_pad_rows_skip_cache_and_keep_q():
    _requires_gfx950()
    inp = make_inputs(8, 0, pad=3)
    before = inp["kv_cache"].clone()
    q_fp8, weights = _run(inp)
    torch.cuda.synchronize()
    # Only the 5 live slots are written (checked by _check_cache); q and the
    # weights of the 3 pad rows are still produced, as the graph-padded decode needs.
    _check_cache(inp, before, ref_prepare(inp))
    assert torch.isfinite(weights).all() and q_fp8.shape == (8, H, D)


def test_dsa_indexer_prepare_int64_path_matches_buffer_path():
    """A cache >= 2 GiB takes the int64 addressing arm; same bytes as the 32-bit arm."""
    _requires_gfx950()
    free = torch.cuda.mem_get_info()[0]
    pages = (1 << 31) // (PAGE * STRIDE) + 1
    if free < pages * PAGE * STRIDE + (1 << 30):
        pytest.skip("not enough free memory for a 2 GiB index-K cache")
    inp = make_inputs(16, 0)
    big = torch.zeros((pages, PAGE, STRIDE), dtype=torch.uint8, device=device)
    shift = (pages - inp["kv_cache"].shape[0]) * PAGE
    q_small, w_small = _run(inp)
    small_codes, small_scale, _, _ = decode_cache(inp["kv_cache"], inp["slot_mapping"])
    inp_big = dict(
        inp, kv_cache=big, slot_mapping=(inp["slot_mapping"] + shift).contiguous()
    )
    q_big, w_big = _run(inp_big)
    torch.cuda.synchronize()
    big_codes, big_scale, _, _ = decode_cache(big, inp_big["slot_mapping"])
    assert torch.equal(q_small.view(torch.uint8), q_big.view(torch.uint8))
    assert torch.equal(w_small, w_big)
    assert torch.equal(small_codes.view(torch.uint8), big_codes.view(torch.uint8))
    assert torch.equal(small_scale, big_scale)
    del big


def test_dsa_indexer_prepare_preallocated_outputs():
    _requires_gfx950()
    inp = make_inputs(4, 0)
    q_out = torch.empty((4, H, D), dtype=FP8, device=device)
    w_out = torch.empty((4, H), dtype=torch.float32, device=device)
    q, w = _run(inp, q_out=q_out, weights_out=w_out)
    assert q is q_out and w is w_out


def test_dsa_indexer_prepare_determinism():
    _requires_gfx950()
    inp = make_inputs(16, 0)
    cache0 = inp["kv_cache"].clone()
    q1, w1 = _run(inp)
    c1 = inp["kv_cache"].clone()
    inp["kv_cache"].copy_(cache0)
    q2, w2 = _run(inp)
    torch.cuda.synchronize()
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8))
    assert torch.equal(w1, w2)
    assert torch.equal(c1, inp["kv_cache"])


def test_dsa_indexer_prepare_supported_rejects():
    from aiter.ops.triton.attention.dsa_indexer_prepare import (
        dsa_indexer_prepare_supported,
    )

    ok, reason = dsa_indexer_prepare_supported(*[torch.empty(0)] * 14)
    if not ok and ("gfx950" in reason or "Triton" in reason):
        pytest.skip(reason)
    inp = make_inputs(2, 0)
    args = [
        inp["x"],
        inp["q_lora"],
        inp["wq_b"],
        inp["wq_b_scale"],
        inp["wk"],
        inp["wk_scale"],
        inp["weights_proj"],
        inp["norm_weight"],
        inp["norm_bias"],
        inp["positions"],
        inp["cos_cache"],
        inp["sin_cache"],
        inp["slot_mapping"],
        inp["kv_cache"],
    ]
    assert dsa_indexer_prepare_supported(*args) == (True, "")
    bad = list(args)
    bad[4] = inp["wk_plain"].to(torch.bfloat16)
    ok, reason = dsa_indexer_prepare_supported(*bad)
    assert not ok and "wk" in reason
    bad = list(args)
    bad[13] = inp["kv_cache"][:, :32]
    ok, reason = dsa_indexer_prepare_supported(*bad)
    assert not ok and "kv_cache" in reason
    bad = list(args)
    bad[0] = torch.empty((64, HID), dtype=torch.bfloat16, device=device)
    ok, reason = dsa_indexer_prepare_supported(*bad)
    assert not ok and "decode rows" in reason
    bad = list(args)
    bad[12] = inp["slot_mapping"].to(torch.int32)
    ok, reason = dsa_indexer_prepare_supported(*bad)
    assert not ok and "slot_mapping" in reason
