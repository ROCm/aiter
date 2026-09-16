# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``flydsl_flash_attn_func`` (gfx1201 / RDNA4)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.flydsl import (
    flydsl_flash_attn_func,
    flydsl_fp8_quant,
)


def _is_gfx1201() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return False
    return arch.lower().split(":")[0].startswith("gfx1201")


pytestmark = pytest.mark.skipif(
    not _is_gfx1201(),
    reason="flydsl_flash_attn_func is gfx1201/RDNA4 only",
)


def _ref_sdpa_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """SDPA reference with BSHD inputs/outputs."""
    out_bhsd = F.scaled_dot_product_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        is_causal=causal,
    )
    return out_bhsd.transpose(1, 2).contiguous()


def _make_qkv(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, seq_len, num_heads, head_dim)
    q = torch.randn(shape, generator=g, dtype=dtype, device=device)
    k = torch.randn(shape, generator=g, dtype=dtype, device=device)
    v = torch.randn(shape, generator=g, dtype=dtype, device=device)
    return q, k, v


@pytest.mark.parametrize(
    "batch,seq_q,seq_kv,num_heads,head_dim",
    [
        (1, 1024, 512, 12, 128),  # Wan-style long Q, short text K/V.
        (1, 4096, 512, 12, 128),
        (1, 2048, 1400, 8, 128),  # K/V pad to BN=64 (1408), not BM=256 (1536).
        (1, 500, 1400, 8, 128),  # Short, unaligned Q with longer unaligned K/V.
        (1, 2048, 500, 8, 64),  # unaligned K/V (500 % BLOCK_N != 0 -> tail mask).
    ],
)
def test_flydsl_fmha_correctness_cross_attention(
    batch, seq_q, seq_kv, num_heads, head_dim
):
    """Cross-attention (seqlen_q != seqlen_k), bf16, non-causal."""
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(
        batch,
        seq_q,
        num_heads,
        head_dim,
        generator=g,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        batch,
        seq_kv,
        num_heads,
        head_dim,
        generator=g,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn(
        batch,
        seq_kv,
        num_heads,
        head_dim,
        generator=g,
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == ref.shape == (batch, seq_q, num_heads, head_dim)
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > 0.999, f"mean_cos={cos.mean().item():.6f}"


@pytest.mark.parametrize(
    "batch,seq_q,seq_kv,num_heads,head_dim",
    [
        (1, 1024, 512, 12, 128),  # Wan-style long Q, short text K/V.
        (1, 500, 1400, 8, 128),  # Short, unaligned Q with longer unaligned K/V.
        (1, 2048, 500, 8, 64),  # unaligned K/V (500 % BLOCK_N != 0 -> tail mask).
    ],
)
def test_flydsl_fmha_correctness_fp8_cross_attention(
    batch, seq_q, seq_kv, num_heads, head_dim
):
    """Per-tensor fp8 cross-attention (seqlen_q != seqlen_k). Same fp8 kernel as
    self-attn, just with independent Q vs K/V lengths; output is bf16."""
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(
        batch,
        seq_q,
        num_heads,
        head_dim,
        generator=g,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        batch,
        seq_kv,
        num_heads,
        head_dim,
        generator=g,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn(
        batch,
        seq_kv,
        num_heads,
        head_dim,
        generator=g,
        dtype=torch.bfloat16,
        device="cuda",
    )
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v)
    out = flydsl_flash_attn_func(
        qq, kk, vv, causal=False, q_descale=sq, k_descale=sk, v_descale=sv
    )
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == ref.shape == (batch, seq_q, num_heads, head_dim)
    assert out.dtype == torch.bfloat16
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.mean().item() > 0.998, f"mean_cos={cos.mean().item():.6f}"


@pytest.mark.parametrize("seq_len", [672, 640, 32])
def test_flydsl_fmha_all_zero_head(seq_len):
    """An all-zero head (Q=K=V=0) must give all-zero output, not NaN.

    Head-padding schemes (e.g. Ulysses "pad heads to an SP-divisible count")
    zero-pad extra heads, run attention, then slice the pads off. Every score in
    such a head is 0, so softmax is uniform and ``o = (Σ 1·v)/S = 0``. A NaN here
    propagates through the whole model, so this guards the degenerate
    online-softmax row.
    """
    batch, num_heads, head_dim = 1, 8, 128
    zero = slice(6, num_heads)  # the "padding" heads
    real = slice(0, 6)
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    q[:, :, zero] = 0
    k[:, :, zero] = 0
    v[:, :, zero] = 0

    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert not out.isnan().any().item(), "kernel produced NaN"
    zero_max = out[:, :, zero].abs().max().item()
    assert zero_max == 0.0, f"zero-head output not zero: max_abs={zero_max}"

    cos = F.cosine_similarity(
        out[:, :, real].float().reshape(-1, head_dim),
        ref[:, :, real].float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"


def test_flydsl_fmha_rejects_causal_cross_attention():
    """Causal + cross-attention must raise. Both kernels bound the causal KV
    loop by seqlen_q, so a shorter K/V is read past its allocation — measured
    on gfx1201 this silently returns garbage (cos ~0.5 vs SDPA) rather than
    faulting, so the wrapper has to reject it up front."""
    q = torch.randn(1, 4096, 12, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 512, 12, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 512, 12, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="causal cross-attention"):
        flydsl_flash_attn_func(q, k, v, causal=True)


def test_flydsl_fmha_rejects_malformed_fp8_descale():
    """FP8 descales must be 1-element fp32 tensors on q's device; a wrong dtype
    or element count would otherwise be read as a bogus scale by the kernel."""
    q, k, v = _make_qkv(1, 1024, 8, 128, torch.bfloat16)
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v)

    bad_dtype = sq.to(torch.float64)
    with pytest.raises(ValueError, match="q_descale"):
        flydsl_flash_attn_func(
            qq, kk, vv, q_descale=bad_dtype, k_descale=sk, v_descale=sv
        )

    bad_numel = torch.ones(2, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="k_descale"):
        flydsl_flash_attn_func(
            qq, kk, vv, q_descale=sq, k_descale=bad_numel, v_descale=sv
        )

    # A host float is the easy mistake; it must raise, not AttributeError.
    with pytest.raises(ValueError, match="v_descale"):
        flydsl_flash_attn_func(qq, kk, vv, q_descale=sq, k_descale=sk, v_descale=1.0)


def test_flydsl_fmha_out_buffer_with_padding():
    """Unaligned seqlen_q still needs the padded temporary, so ``out`` is filled
    by a copy. Same observable contract as the in-place path."""
    batch, seq_len, num_heads, head_dim = 1, 1000, 8, 128  # 1000 % 128 != 0
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)

    out = torch.empty(
        batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    ret = flydsl_flash_attn_func(q, k, v, causal=False, out=out)
    assert ret.data_ptr() == out.data_ptr()

    ref = flydsl_flash_attn_func(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_flydsl_fmha_rejects_gqa():
    """Grouped-query attention (num_heads_q != num_heads_k) is unsupported; the
    kernel assumes equal head counts."""
    q = torch.randn(1, 1024, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="num_heads"):
        flydsl_flash_attn_func(q, k, v)


@pytest.mark.parametrize(
    "batch,seq_len,num_heads,head_dim",
    [
        (2, 1000, 8, 128),  # 1000 % 32 != 0 (BLOCK_N=32 tile) -> tail mask.
        (1, 1400, 12, 128),  # 1400 % 64 != 0 (BLOCK_N=64 tile) -> tail mask.
    ],
)
def test_flydsl_fmha_correctness_unaligned_noncausal(
    batch, seq_len, num_heads, head_dim
):
    """Non-causal, non-BLOCK_N-aligned seq_len. The kernel's per-column tail
    mask must exclude padded K/V columns from the softmax so the padding does
    not leak exp(0)=1 into the denominator."""
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == ref.shape == (batch, seq_len, num_heads, head_dim)
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > 0.999, f"mean_cos={cos.mean().item():.6f}"


@pytest.mark.parametrize("head_dim", [160, 192])
@pytest.mark.parametrize("use_fp8", [False, True], ids=["bf16", "fp8"])
def test_flydsl_fmha_partial_kv_load_batch(head_dim, use_fp8):
    """Cover load schedules whose final cooperative batch is only partial."""
    batch, seq_len, num_heads = 1, 96, 2
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    if use_fp8:
        qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v, rotation=False)
        out = flydsl_flash_attn_func(
            qq,
            kk,
            vv,
            causal=False,
            q_descale=sq,
            k_descale=sk,
            v_descale=sv,
        )
        min_cos, mean_cos = 0.98, 0.995
    else:
        out = flydsl_flash_attn_func(q, k, v, causal=False)
        min_cos, mean_cos = 0.99, 0.999

    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > min_cos, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > mean_cos, f"mean_cos={cos.mean().item():.6f}"


@pytest.mark.parametrize(
    "head_dim,use_fp8,expected_bytes",
    [
        (512, False, 66048),
        (992, True, 67584),
    ],
)
def test_flydsl_fmha_rejects_lds_overflow(head_dim, use_fp8, expected_bytes):
    """Reject shapes whose narrowest supported KV tile still exceeds 64 KiB."""
    batch, seq_len, num_heads = 1, 2048, 1  # non-causal selector uses BLOCK_N=64
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    kwargs = {}
    if use_fp8:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        scale = torch.ones(1, dtype=torch.float32, device=q.device)
        kwargs = {"q_descale": scale, "k_descale": scale, "v_descale": scale}

    with pytest.raises(
        ValueError,
        match=rf"requires {expected_bytes} bytes of LDS.*65536-byte hardware limit",
    ):
        flydsl_flash_attn_func(q, k, v, causal=False, **kwargs)


@pytest.mark.parametrize(
    "head_dim,use_fp8,selected_bytes,fallback_bytes",
    [
        (256, False, 66560, 33280),
        (512, True, 67840, 34944),
    ],
)
def test_flydsl_fmha_lds_falls_back_to_block_n32(
    monkeypatch, head_dim, use_fp8, selected_bytes, fallback_bytes
):
    """An oversized BN64 selection retries BN32 without changing BLOCK_M."""
    from aiter.ops.flydsl import fmha_kernels

    assert (
        fmha_kernels._gfx1201_fmha_lds_bytes(head_dim, 64, fp8=use_fp8)
        == selected_bytes
    )
    assert (
        fmha_kernels._gfx1201_fmha_lds_bytes(head_dim, 32, fp8=use_fp8)
        == fallback_bytes
    )

    calls = []

    def _fake_get_kernel(**kwargs):
        calls.append(kwargs)

        def _launch(*args, **launch_kwargs):
            return None

        return _launch

    cache_name = "_get_fp8_gfx1201_kernel" if use_fp8 else "_get_kernel"
    monkeypatch.setattr(fmha_kernels, cache_name, _fake_get_kernel)

    batch, seq_len, num_heads = 1, 2048, 1
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    kwargs = {}
    if use_fp8:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        scale = torch.ones(1, dtype=torch.float32, device=q.device)
        kwargs = {"q_descale": scale, "k_descale": scale, "v_descale": scale}

    flydsl_flash_attn_func(q, k, v, causal=False, **kwargs)
    assert len(calls) == 1
    assert calls[0]["block_m"] == 256
    assert calls[0]["block_n"] == 32
    assert calls[0]["head_dim"] == head_dim


@pytest.mark.parametrize(
    "head_dim,use_fp8,expected_bytes",
    [
        (224, False, 58368),
        (480, True, 63616),
    ],
)
def test_flydsl_fmha_accepts_nearest_lds_boundary(
    monkeypatch, head_dim, use_fp8, expected_bytes
):
    """The nearest supported multiple-of-32 below each LDS limit remains valid."""
    from aiter.ops.flydsl import fmha_kernels

    assert (
        fmha_kernels._gfx1201_fmha_lds_bytes(head_dim, 64, fp8=use_fp8)
        == expected_bytes
    )

    calls = []

    def _fake_get_kernel(**kwargs):
        calls.append(kwargs)

        def _launch(*args, **launch_kwargs):
            return None

        return _launch

    cache_name = "_get_fp8_gfx1201_kernel" if use_fp8 else "_get_kernel"
    monkeypatch.setattr(fmha_kernels, cache_name, _fake_get_kernel)

    batch, seq_len, num_heads = 1, 2048, 1
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    kwargs = {}
    if use_fp8:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        scale = torch.ones(1, dtype=torch.float32, device=q.device)
        kwargs = {"q_descale": scale, "k_descale": scale, "v_descale": scale}

    flydsl_flash_attn_func(q, k, v, causal=False, **kwargs)
    assert len(calls) == 1
    assert calls[0]["block_n"] == 64
    assert calls[0]["head_dim"] == head_dim


@pytest.mark.parametrize(
    "batch,seq_len,num_heads,head_dim",
    [
        (1, 1536, 24, 128),  # Flux compute-bound shape (fp8's target win).
        (1, 4096, 24, 128),  # flux/wan family, fp8 wins at long S.
        (1, 8192, 12, 128),
    ],
)
def test_flydsl_fmha_correctness_fp8(batch, seq_len, num_heads, head_dim):
    """Per-tensor fp8 self-attention. Output is bf16; cosine vs fp32 SDPA is the
    correctness signal. Measured on gfx1201 the mean cosine sits in a tight
    ~0.9986 cluster (min >=0.9926) across these shapes; the bounds below leave a
    safe margin while still catching real numerical regressions."""
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v)
    out = flydsl_flash_attn_func(
        qq, kk, vv, causal=False, q_descale=sq, k_descale=sk, v_descale=sv
    )
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == ref.shape == (batch, seq_len, num_heads, head_dim)
    assert out.dtype == torch.bfloat16
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > 0.998, f"mean_cos={cos.mean().item():.6f}"


def test_flydsl_fmha_correctness_fp8_causal():
    """Exercise causal masking in the gfx1201 FP8 consumer end to end."""
    batch, seq_len, num_heads, head_dim = 1, 1024, 8, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v)
    out = flydsl_flash_attn_func(
        qq, kk, vv, causal=True, q_descale=sq, k_descale=sk, v_descale=sv
    )
    ref = _ref_sdpa_bshd(q, k, v, causal=True)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.98, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > 0.997, f"mean_cos={cos.mean().item():.6f}"


def test_flydsl_fp8_quant_producer_invariants():
    """Direct coverage of the FlyDSL fp8 producer (fp8_quant_gfx1201), which the
    end-to-end tests only exercise transitively: the per-tensor scale contract,
    rotation=False dequant accuracy, and the rotation-cancellation invariant
    ``(Q@R)(K@R)^T == Q@K^T`` that attention relies on (the rotation is never
    undone in the kernel, so a wrong/asymmetric rotation would hide here)."""
    b, s, h, d = 1, 1024, 8, 128  # head_dim==128 -> FlyDSL path
    q, k, v = _make_qkv(b, s, h, d, torch.bfloat16)

    # Scale contract: e4m3 outputs, 1-elem fp32 descales (real = fp8 * scale).
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v, rotation=False)
    for t8 in (qq, kk, vv):
        assert t8.dtype == torch.float8_e4m3fn
    for sc in (sq, sk, sv):
        assert sc.shape == (1,) and sc.dtype == torch.float32

    # rotation=False: dequant ≈ original (only e4m3 rounding).
    cos = F.cosine_similarity(
        (qq.float() * sq).reshape(-1, d), q.float().reshape(-1, d), dim=1
    )
    assert (
        cos.mean().item() > 0.99
    ), f"rot=False dequant mean_cos={cos.mean().item():.6f}"

    # rotation=True: rotation must cancel in QK^T. Dequant gives Q@R and K@R; their
    # inner product must match the unrotated Q@K^T (one head).
    qr, kr, _, sqr, skr, _ = flydsl_fp8_quant(q, k, v, rotation=True)
    Qr = (qr.float() * sqr)[0, :, 0]
    Kr = (kr.float() * skr)[0, :, 0]
    scores_rot = Qr @ Kr.T
    scores_ref = q[0, :, 0].float() @ k[0, :, 0].float().T
    cos_s = F.cosine_similarity(
        scores_rot.reshape(1, -1), scores_ref.reshape(1, -1), dim=1
    ).item()
    assert cos_s > 0.99, f"QK-preservation cos={cos_s:.6f}"


def test_flydsl_fp8_quant_backend_agreement():
    """The flydsl and torch producers use different orthonormal rotations, so they
    agree only at the attention-output level (the rotation cancels). Guards the new
    FlyDSL producer against the torch reference on the same inputs."""
    b, s, h, d = 1, 2048, 12, 128
    q, k, v = _make_qkv(b, s, h, d, torch.bfloat16)
    ref = _ref_sdpa_bshd(q, k, v)

    outs = {}
    for be in ("flydsl", "torch"):
        qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v, backend=be)
        outs[be] = flydsl_flash_attn_func(
            qq, kk, vv, causal=False, q_descale=sq, k_descale=sk, v_descale=sv
        )
        cos = F.cosine_similarity(
            outs[be].float().reshape(-1, d), ref.float().reshape(-1, d), dim=1
        )
        assert cos.mean().item() > 0.998, f"{be} mean_cos={cos.mean().item():.6f}"

    cos_fb = F.cosine_similarity(
        outs["flydsl"].float().reshape(-1, d),
        outs["torch"].float().reshape(-1, d),
        dim=1,
    )
    assert (
        cos_fb.mean().item() > 0.998
    ), f"flydsl-vs-torch mean_cos={cos_fb.mean().item():.6f}"


def test_flydsl_fp8_quant_fp16_fallback_and_direct_guard():
    """FP16 must use a safe public fallback and fail fast at the bf16-only
    low-level FlyDSL producer instead of being reinterpreted as bf16 bits."""
    from aiter.ops.flydsl.kernels.fp8_quant_gfx1201 import (
        flydsl_fp8_pertensor_quant,
    )

    q, k, v = _make_qkv(1, 128, 2, 128, torch.float16)
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v, rotation=False, backend="flydsl")
    for original, quantized, scale in zip((q, k, v), (qq, kk, vv), (sq, sk, sv)):
        cos = F.cosine_similarity(
            original.float().reshape(-1, 128),
            (quantized.float() * scale).reshape(-1, 128),
            dim=1,
        )
        assert cos.mean().item() > 0.99

    with pytest.raises(TypeError, match="requires bfloat16 input"):
        flydsl_fp8_pertensor_quant(q, rotate=False)


def test_flydsl_fp8_quant_fp16_rotation_uses_fp16_matrix(monkeypatch):
    from aiter.ops.flydsl import fmha_kernels

    q, k, v = _make_qkv(1, 128, 2, 128, torch.float16)
    seen_dtypes = []
    original = fmha_kernels._hadamard_matrix

    def _record_dtype(head_dim, device, dtype):
        seen_dtypes.append(dtype)
        return original(head_dim, device, dtype)

    monkeypatch.setattr(fmha_kernels, "_hadamard_matrix", _record_dtype)
    flydsl_fp8_quant(q, k, v, rotation=True, backend="torch")
    assert seen_dtypes == [torch.float16]


def test_flydsl_hadamard_matrix_uses_target_device():
    """Build the rotation matrix directly on the requested non-current GPU."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 visible GPUs")

    from aiter.ops.flydsl import fmha_kernels

    current_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        target = torch.device("cuda:1")
        fmha_kernels._HADAMARD_CACHE.clear()
        rotation = fmha_kernels._hadamard_matrix(128, target, torch.bfloat16)
        assert rotation.device == target
    finally:
        torch.cuda.set_device(current_device)


def test_flydsl_hadamard_cache_concurrent_first_use_and_reuse():
    """Concurrent streams share one fully-produced cached matrix."""
    import threading

    from aiter.ops.flydsl import fmha_kernels

    device = torch.device("cuda", 0)
    fmha_kernels._HADAMARD_CACHE.clear()
    barrier = threading.Barrier(2)
    results = [None, None]
    errors = []

    def _worker(index):
        try:
            torch.cuda.set_device(device)
            stream = torch.cuda.Stream(device=device)
            barrier.wait()
            with torch.cuda.stream(stream):
                matrix = fmha_kernels._hadamard_matrix(128, device, torch.bfloat16)
                checksum = matrix.float().sum()
            stream.synchronize()
            results[index] = (matrix.data_ptr(), checksum.item())
        except Exception as exc:  # noqa: BLE001 - surfaced in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert results[0][0] == results[1][0]
    assert results[0][1] == pytest.approx(math.sqrt(128), rel=2e-2)
    assert results[1][1] == pytest.approx(math.sqrt(128), rel=2e-2)

    reuse_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(reuse_stream):
        reused = fmha_kernels._hadamard_matrix(128, device, torch.bfloat16)
        reused_checksum = reused.float().sum()
    reuse_stream.synchronize()
    assert reused.data_ptr() == results[0][0]
    assert reused_checksum.item() == pytest.approx(math.sqrt(128), rel=2e-2)


def test_stream_readiness_concurrent_producers_chain_same_storage():
    """The latest registration covers every earlier in-flight producer."""
    import threading

    from aiter.ops.flydsl.stream_readiness import (
        register_ready,
        wait_ready,
    )

    device = torch.device("cuda", 0)
    tensor = torch.zeros(2, dtype=torch.int32, device=device)
    slow_stream = torch.cuda.Stream(device=device)
    fast_stream = torch.cuda.Stream(device=device)
    consumer_stream = torch.cuda.Stream(device=device)
    source_stream = torch.cuda.current_stream(device)
    lhs = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device) * 0.01
    rhs = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device) * 0.01
    slow_registered = threading.Event()
    errors = []

    def _slow_producer():
        try:
            torch.cuda.set_device(device)
            slow_stream.wait_stream(source_stream)
            with torch.cuda.stream(slow_stream):
                work = lhs
                for _ in range(3):
                    work = work @ rhs
                # The write cannot execute until all queued matrix work does.
                tensor[0].copy_((work[0, 0] * 0).to(torch.int32) + 1)
            register_ready((tensor,), stream=slow_stream)
            slow_registered.set()
        except Exception as exc:  # noqa: BLE001 - surfaced in main thread
            errors.append(exc)
            slow_registered.set()

    def _fast_producer():
        try:
            torch.cuda.set_device(device)
            slow_registered.wait()
            fast_stream.wait_stream(source_stream)
            with torch.cuda.stream(fast_stream):
                tensor[1].fill_(2)
            register_ready((tensor,), stream=fast_stream)
        except Exception as exc:  # noqa: BLE001 - surfaced in main thread
            errors.append(exc)

    threads = [
        threading.Thread(target=_slow_producer),
        threading.Thread(target=_fast_producer),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors

    wait_ready(consumer_stream, (tensor,))
    with torch.cuda.stream(consumer_stream):
        observed = tensor.clone()
    # If the latest registration did not chain the slow producer, this clone
    # would capture tensor[0] before its matrix-dependent write.
    consumer_stream.synchronize()
    slow_stream.synchronize()
    torch.testing.assert_close(observed.cpu(), torch.tensor([1, 2], dtype=torch.int32))


def test_flydsl_fp8_quant_rejects_unknown_backend():
    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    with pytest.raises(ValueError, match="unsupported fp8 quant backend"):
        flydsl_fp8_quant(q, k, v, backend="flydls")
    with pytest.raises(TypeError, match="backend must be a string"):
        flydsl_fp8_quant(q, k, v, backend=None)


def test_flydsl_arch_detection_does_not_require_rocminfo(monkeypatch):
    from aiter.ops.flydsl import fmha_kernels

    def _broken_rocminfo():
        raise RuntimeError("rocminfo unavailable")

    monkeypatch.setattr(fmha_kernels, "get_gfx_runtime", _broken_rocminfo)
    assert fmha_kernels._live_gfx(torch.device("cuda:0")) == "gfx1201"


def test_flydsl_fp8_quant_validates_input_contract():
    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    with pytest.raises(ValueError, match="share head_dim"):
        flydsl_fp8_quant(q, k[..., :64], v)
    with pytest.raises(ValueError, match="dtype must match"):
        flydsl_fp8_quant(q, k.to(torch.float16), v)
    with pytest.raises(ValueError, match="CUDA/HIP tensors"):
        flydsl_fp8_quant(q.cpu(), k.cpu(), v.cpu())


@pytest.mark.parametrize("backend", ["flydsl", "triton", "torch"])
@pytest.mark.parametrize("empty_index", [0, 1, 2])
def test_flydsl_fp8_quant_rejects_empty_inputs(backend, empty_index):
    q, k, v = _make_qkv(1, 8, 2, 128, torch.bfloat16)
    tensors = [q, k, v]
    tensors[empty_index] = tensors[empty_index][:, :0]
    with pytest.raises(ValueError, match="q/k/v must be non-empty"):
        flydsl_fp8_quant(*tensors, backend=backend)


def test_flydsl_fp8_quant_non_current_stream():
    from aiter.ops.flydsl.kernels.fp8_quant_gfx1201 import (
        flydsl_fp8_pertensor_quant,
    )

    # Transpose gives the quantizer a non-contiguous input produced on the
    # current stream, exercising both producer ordering and staging lifetime.
    backing = torch.randn(128, 1024, dtype=torch.bfloat16, device="cuda")
    x = backing.T
    expected = x.float().clone()
    stream = torch.cuda.Stream()
    xq, scale = flydsl_fp8_pertensor_quant(x, rotate=False, stream=stream)
    del x, backing

    # Encourage allocator reuse before the non-current stream finishes. The
    # quantizer must keep the original input and contiguous staging allocation
    # alive until their reads complete.
    torch.empty(1024, 128, dtype=torch.bfloat16, device="cuda").fill_(float("nan"))
    stream.synchronize()

    cos = F.cosine_similarity(expected, xq.float() * scale, dim=1)
    assert cos.mean().item() > 0.99


def test_flydsl_fp8_quant_reuses_out_across_non_current_streams():
    """A second stream must wait before overwriting a registered output."""
    from aiter.ops.flydsl.kernels.fp8_quant_gfx1201 import (
        flydsl_fp8_pertensor_quant,
    )

    shape = (1, 4096, 2, 128)
    first = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    second = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()

    flydsl_fp8_pertensor_quant(first, rotate=True, out=out, stream=first_stream)
    result, scale = flydsl_fp8_pertensor_quant(
        second, rotate=False, out=out, stream=second_stream
    )
    second_stream.synchronize()

    assert result.data_ptr() == out.data_ptr()
    cos = F.cosine_similarity(
        second.float().reshape(-1, 128),
        (result.float() * scale).reshape(-1, 128),
        dim=1,
    )
    assert cos.mean().item() > 0.99


def test_low_level_fp8_quant_waits_for_registered_attention_input():
    """Low-level quant waits for an async attention producer on another stream."""
    from aiter.ops.flydsl.kernels.fp8_quant_gfx1201 import (
        flydsl_fp8_pertensor_quant,
    )

    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    producer = torch.cuda.Stream(device=q.device)
    consumer = torch.cuda.Stream(device=q.device)
    attention = flydsl_flash_attn_func(q, k, v, stream=producer)

    quantized, scale = flydsl_fp8_pertensor_quant(
        attention, rotate=False, stream=consumer
    )
    consumer.synchronize()

    cos = F.cosine_similarity(
        attention.float().reshape(-1, 128),
        (quantized.float() * scale).reshape(-1, 128),
        dim=1,
    )
    assert cos.mean().item() > 0.99


def test_flydsl_fp8_quant_cross_stream_consumer_needs_no_manual_sync():
    """Attention waits for FP8 tensors produced on a different stream."""
    device = torch.device("cuda")
    producer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        q = torch.randn(1, 129, 4, 128, dtype=torch.bfloat16, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(q, k, v)

    # New tensor objects sharing the registered allocations must retain the
    # producer dependency.
    q8, k8, v8 = q8.view_as(q8), k8.view_as(k8), v8.view_as(v8)
    sq, sk, sv = sq.view(1), sk.view(1), sv.view(1)

    # Deliberately consume on the default/current stream without synchronizing
    # or explicitly waiting for the producer stream.
    out = flydsl_flash_attn_func(q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv)
    ref = _ref_sdpa_bshd(q, k, v)
    cosine = F.cosine_similarity(
        out.float().reshape(-1, 128), ref.float().reshape(-1, 128), dim=1
    )
    assert cosine.mean().item() > 0.998


def test_low_level_fp8_quant_cross_stream_consumer_needs_no_manual_sync():
    from aiter.ops.flydsl.kernels.fp8_quant_gfx1201 import (
        flydsl_fp8_pertensor_quant,
    )

    q, k, v = _make_qkv(1, 129, 4, 128, torch.bfloat16)
    producer = torch.cuda.Stream(device=q.device)
    q_out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_out = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_out = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    q8, sq = flydsl_fp8_pertensor_quant(q, rotate=True, out=q_out, stream=producer)
    k8, sk = flydsl_fp8_pertensor_quant(k, rotate=True, out=k_out, stream=producer)
    v8, sv = flydsl_fp8_pertensor_quant(v, rotate=False, out=v_out, stream=producer)

    out = flydsl_flash_attn_func(q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv)
    ref = _ref_sdpa_bshd(q, k, v)
    cosine = F.cosine_similarity(
        out.float().reshape(-1, 128), ref.float().reshape(-1, 128), dim=1
    )
    assert cosine.mean().item() > 0.998


def test_flydsl_fp8_quant_validates_low_level_contract():
    from aiter.ops.flydsl.kernels.fp8_quant_gfx1201 import (
        flydsl_fp8_pertensor_quant,
    )

    x = torch.randn(1024, 128, dtype=torch.bfloat16, device="cuda")
    bad_outputs = (
        torch.empty(1024, 64, dtype=torch.float8_e4m3fn, device="cuda"),
        torch.empty_like(x),
        torch.empty(128, 1024, dtype=torch.float8_e4m3fn, device="cuda").T,
        torch.empty(1024, 128, dtype=torch.float8_e4m3fn, device="cpu"),
    )
    for out in bad_outputs:
        with pytest.raises(ValueError, match="out must be a contiguous"):
            flydsl_fp8_pertensor_quant(x, rotate=False, out=out)

    with pytest.raises(ValueError, match="head_dim=128"):
        flydsl_fp8_pertensor_quant(x[:, :64], rotate=False)


def test_flydsl_fp8_quant_rejects_out_overlapping_input_storage():
    from aiter.ops.flydsl.kernels.fp8_quant_gfx1201 import (
        flydsl_fp8_pertensor_quant,
    )

    x = torch.randn(1024, 128, dtype=torch.bfloat16, device="cuda")
    # Reinterpret the first half of x's byte storage as a same-shaped FP8 view.
    # It satisfies the output shape/dtype/contiguity contract but aliases x.
    out = (
        x.view(torch.uint8)
        .reshape(-1)[: x.numel()]
        .view(torch.float8_e4m3fn)
        .reshape(x.shape)
    )
    assert (
        out.is_contiguous()
        and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    )
    with pytest.raises(ValueError, match="must not overlap input storage"):
        flydsl_fp8_pertensor_quant(x, rotate=False, out=out)


def test_flydsl_fp8_ignores_bf16_lds_vec_width_toggle(monkeypatch):
    """The BF16 vec8 diagnostic toggle must not alter FP8's fixed vec16 layout."""
    from aiter.ops.flydsl import fmha_kernels

    monkeypatch.setenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "0")
    fmha_kernels._get_fp8_gfx1201_kernel.cache_clear()

    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v, rotation=False)
    out = flydsl_flash_attn_func(
        qq, kk, vv, causal=False, q_descale=sq, k_descale=sk, v_descale=sv
    )
    ref = _ref_sdpa_bshd(q, k, v)
    cos = F.cosine_similarity(
        out.float().reshape(-1, 128), ref.float().reshape(-1, 128), dim=1
    )
    assert cos.mean().item() > 0.998


def test_flydsl_fmha_missing_fp8_descale_raises():
    """FP8 inputs without descales must raise (they are required)."""
    q, k, v = _make_qkv(1, 1024, 8, 128, torch.bfloat16)
    qq, kk, vv, *_ = flydsl_fp8_quant(q, k, v)
    with pytest.raises(ValueError, match="descale"):
        flydsl_flash_attn_func(qq, kk, vv, causal=False)


def test_flydsl_fmha_softmax_scale():
    """A scalar-tensor softmax_scale is normalized and matches SDPA."""
    batch, seq_len, num_heads, head_dim = 2, 2048, 8, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    scale_value = 0.05
    scale = torch.tensor(scale_value)
    out = flydsl_flash_attn_func(q, k, v, causal=False, softmax_scale=scale)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        is_causal=False,
        scale=scale_value,
    ).transpose(1, 2)

    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"

    with pytest.raises(ValueError, match="softmax_scale must be a scalar"):
        flydsl_flash_attn_func(q, k, v, causal=False, softmax_scale=torch.ones(2))
    with pytest.raises(ValueError, match="tensor softmax_scale must be on CPU"):
        flydsl_flash_attn_func(
            q, k, v, causal=False, softmax_scale=torch.tensor(0.05, device="cuda")
        )


@pytest.mark.parametrize(
    "softmax_scale", [0.0, -0.125, float("inf"), float("-inf"), float("nan")]
)
def test_flydsl_fmha_invalid_softmax_scale_raises(softmax_scale):
    """Non-positive and non-finite scales fail before kernel compilation."""
    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    with pytest.raises(ValueError, match="softmax_scale must be finite and positive"):
        flydsl_flash_attn_func(q, k, v, causal=False, softmax_scale=softmax_scale)


def test_flydsl_fmha_out_buffer():
    """Preallocated ``out=`` buffer is written in place and returned.

    ``seq_len`` is BLOCK_M-aligned, so this is the no-copy path where ``out`` is
    the kernel's own destination; the NaN prefill proves it is fully overwritten
    rather than partially filled.
    """
    batch, seq_len, num_heads, head_dim = 2, 2048, 8, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    out = torch.full(
        (batch, seq_len, num_heads, head_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    ret = flydsl_flash_attn_func(q, k, v, causal=False, out=out)
    assert ret.data_ptr() == out.data_ptr()
    assert not out.isnan().any().item(), "kernel did not overwrite the buffer"

    ref = flydsl_flash_attn_func(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_flydsl_fmha_fp8_out_buffer_and_softmax_scale():
    """Practical fp8 example exercising every fp8-path public extra together:
    per-tensor descales, a custom ``softmax_scale``, and a preallocated bf16
    ``out=`` buffer. Mirrors a caller that quantizes q/k/v, uses a non-default
    scale, and reuses an output buffer. The fp8 output is bf16 regardless of the
    fp8 input dtype."""
    batch, seq_len, num_heads, head_dim = 1, 4096, 24, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v)
    scale = 0.05
    out = torch.empty(
        batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    ret = flydsl_flash_attn_func(
        qq,
        kk,
        vv,
        causal=False,
        softmax_scale=scale,
        q_descale=sq,
        k_descale=sk,
        v_descale=sv,
        out=out,
    )
    assert ret.data_ptr() == out.data_ptr()
    assert out.dtype == torch.bfloat16

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        is_causal=False,
        scale=scale,
    ).transpose(1, 2)
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.mean().item() > 0.998, f"mean_cos={cos.mean().item():.6f}"


def test_flydsl_fmha_rejects_bad_out_buffer():
    """``out=`` with the wrong shape, dtype or device must raise (guards the
    preallocated-buffer path). A foreign-device ``out`` matters most: on the
    no-copy path it would be handed to the kernel as its destination."""
    q, k, v = _make_qkv(1, 1024, 8, 128, torch.bfloat16)
    bad_shape = torch.empty(1, 512, 8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="shape"):
        flydsl_flash_attn_func(q, k, v, out=bad_shape)
    bad_dtype = torch.empty(1, 1024, 8, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        flydsl_flash_attn_func(q, k, v, out=bad_dtype)
    if torch.cuda.device_count() >= 2:
        q_index = (
            q.device.index
            if q.device.index is not None
            else torch.cuda.current_device()
        )
        other_index = (q_index + 1) % torch.cuda.device_count()
        bad_device = torch.empty(
            1,
            1024,
            8,
            128,
            dtype=torch.bfloat16,
            device=f"cuda:{other_index}",
        )
        with pytest.raises(ValueError, match="must be on"):
            flydsl_flash_attn_func(q, k, v, out=bad_device)


def test_flydsl_fmha_positional_backcompat():
    """The original positional signature (q, k, v, causal, waves_per_eu, daz,
    stream) must keep working so pre-existing callers are unaffected by the new
    optional params (which are appended after it)."""
    q, k, v = _make_qkv(1, 1024, 8, 128, torch.bfloat16)
    out_pos = flydsl_flash_attn_func(q, k, v, False, 2, True)  # positional
    out_kw = flydsl_flash_attn_func(q, k, v, causal=False, waves_per_eu=2, daz=True)
    torch.testing.assert_close(out_pos, out_kw, rtol=0, atol=0)


def test_flydsl_fmha_non_current_stream_padding_and_out():
    """Padding, kernel launch, and the final out copy share the requested stream."""
    q, k, v = _make_qkv(1, 97, 2, 128, torch.bfloat16)
    out = torch.empty_like(q)
    launch_stream = torch.cuda.Stream(device=q.device)

    result = flydsl_flash_attn_func(q, k, v, stream=launch_stream, out=out)
    launch_stream.synchronize()
    ref = _ref_sdpa_bshd(q, k, v)

    assert result.data_ptr() == out.data_ptr()
    cos = F.cosine_similarity(
        out.float().reshape(-1, 128), ref.float().reshape(-1, 128), dim=1
    )
    assert cos.min().item() > 0.99


def test_flydsl_fmha_reused_out_orders_non_current_streams():
    """A second asynchronous write waits for the first write to shared out."""
    q1, k1, v1 = _make_qkv(1, 129, 2, 128, torch.bfloat16)
    q2, k2, v2 = _make_qkv(1, 129, 2, 128, torch.bfloat16)
    q2 = q2 + 0.5
    k2 = k2 - 0.25
    v2 = -v2
    out = torch.empty_like(q1)
    first_stream = torch.cuda.Stream(device=q1.device)
    second_stream = torch.cuda.Stream(device=q1.device)

    first = flydsl_flash_attn_func(q1, k1, v1, stream=first_stream, out=out)
    second = flydsl_flash_attn_func(q2, k2, v2, stream=second_stream, out=out)

    # Synchronizing only the second stream is sufficient because its write is
    # chained after the first output event.
    second_stream.synchronize()
    ref = _ref_sdpa_bshd(q2, k2, v2)
    assert first.data_ptr() == second.data_ptr() == out.data_ptr()
    cosine = F.cosine_similarity(
        out.float().reshape(-1, 128), ref.float().reshape(-1, 128), dim=1
    )
    assert cosine.min().item() > 0.99


def test_bf16_attention_result_chains_across_streams():
    """A BF16 result can immediately become Q/K/V on another stream."""
    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    first_stream = torch.cuda.Stream(device=q.device)
    second_stream = torch.cuda.Stream(device=q.device)

    first = flydsl_flash_attn_func(q, k, v, stream=first_stream)
    second = flydsl_flash_attn_func(first, first, first, stream=second_stream)

    second_stream.synchronize()
    first_ref = _ref_sdpa_bshd(q, k, v)
    second_ref = _ref_sdpa_bshd(first_ref, first_ref, first_ref)
    cosine = F.cosine_similarity(
        second.float().reshape(-1, 128),
        second_ref.float().reshape(-1, 128),
        dim=1,
    )
    assert cosine.mean().item() > 0.999


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_async_attention_result_feeds_public_fp8_quant(backend):
    """Public quant backends wait for an asynchronous attention producer."""
    if backend == "triton":
        from aiter.ops.flydsl import fmha_kernels

        if not fmha_kernels._HAS_TRITON:
            pytest.skip("Triton is unavailable")

    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    produced_ref = _ref_sdpa_bshd(q, k, v)
    producer = torch.cuda.Stream(device=q.device)
    backing = torch.empty(1, 2, 128, 128, dtype=torch.bfloat16, device=q.device)
    producer_out = backing.transpose(1, 2)
    assert producer_out.shape == q.shape and not producer_out.is_contiguous()
    produced = flydsl_flash_attn_func(q, k, v, stream=producer, out=producer_out)

    q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(
        produced, produced, produced, backend=backend
    )
    del q, k, v, produced, producer_out, backing
    # Encourage reuse of both the producer allocation and quant staging
    # allocations before any explicit synchronization occurs.
    torch.empty(1, 128, 2, 128, dtype=torch.bfloat16, device="cuda").fill_(float("nan"))
    out = flydsl_flash_attn_func(q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv)
    ref = _ref_sdpa_bshd(produced_ref, produced_ref, produced_ref)
    cosine = F.cosine_similarity(
        out.float().reshape(-1, 128), ref.float().reshape(-1, 128), dim=1
    )
    assert cosine.mean().item() > 0.998


@pytest.mark.parametrize(
    "shape_q,shape_kv",
    [
        ((0, 128, 2, 128), (0, 128, 2, 128)),
        ((1, 0, 2, 128), (1, 0, 2, 128)),
        ((1, 128, 0, 128), (1, 128, 0, 128)),
        ((1, 128, 2, 128), (1, 0, 2, 128)),
    ],
)
def test_flydsl_fmha_rejects_zero_dimensions(shape_q, shape_kv):
    q = torch.empty(shape_q, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(shape_kv, dtype=torch.bfloat16, device="cuda")
    v = torch.empty_like(k)
    with pytest.raises(ValueError, match="must all be non-zero"):
        flydsl_flash_attn_func(q, k, v)


@pytest.mark.parametrize("source", ["q", "k", "v"])
def test_flydsl_fmha_rejects_out_alias(source):
    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    out = {"q": q, "k": k, "v": v}[source]
    with pytest.raises(ValueError, match="must not overlap"):
        flydsl_flash_attn_func(q, k, v, out=out)


@pytest.mark.parametrize("layout", ["expanded", "strided"])
def test_flydsl_fmha_rejects_unsafe_out_layout(layout):
    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    if layout == "expanded":
        out = torch.empty(1, 1, 1, 1, dtype=q.dtype, device=q.device).expand_as(q)
    else:
        backing = torch.empty(1, 128, 2, 256, dtype=q.dtype, device=q.device)
        out = backing[..., ::2]
    assert out.shape == q.shape and not out.is_contiguous()
    with pytest.raises(ValueError, match="internal overlap or unsupported striding"):
        flydsl_flash_attn_func(q, k, v, out=out)


def test_flydsl_fmha_rejects_out_overlapping_fp8_descale():
    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    q8, k8, v8, _sq, sk, sv = flydsl_fp8_quant(q, k, v)
    out = torch.empty_like(q)
    alias_scale = out.view(torch.uint8).reshape(-1)[:4].view(torch.float32)
    assert alias_scale.shape == (1,)
    with pytest.raises(ValueError, match="FP8 descale storage"):
        flydsl_flash_attn_func(
            q8,
            k8,
            v8,
            q_descale=alias_scale,
            k_descale=sk,
            v_descale=sv,
            out=out,
        )


@pytest.mark.parametrize("seq_len", [96, 1408])
def test_flydsl_fmha_masks_block_m_padding_when_block_n_aligned(seq_len):
    q, k, v = _make_qkv(1, seq_len, 8, 128, torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)
    cosine = F.cosine_similarity(
        out.float().reshape(-1, 128), ref.float().reshape(-1, 128), dim=1
    )
    assert cosine.min().item() > 0.99
    assert cosine.mean().item() > 0.999


@pytest.mark.parametrize("seq_len", [96, 1408])
def test_flydsl_fmha_block_n_aligned_skips_tail_specialization(monkeypatch, seq_len):
    """BLOCK_M-only padding must not compile unnecessary KV tail predicates."""
    from aiter.ops.flydsl import fmha_kernels

    calls = []

    def _fake_get_kernel(**kwargs):
        calls.append(kwargs)

        def _launch(*args, **launch_kwargs):
            return None

        return _launch

    monkeypatch.setattr(fmha_kernels, "_get_kernel", _fake_get_kernel)
    q, k, v = _make_qkv(1, seq_len, 2, 128, torch.bfloat16)
    flydsl_flash_attn_func(q, k, v, causal=False)
    assert len(calls) == 1
    assert calls[0]["tail_mask"] is False


def test_flydsl_fmha_bf16_vec_width_participates_in_cache_key(monkeypatch):
    from aiter.ops.flydsl import fmha_kernels

    builds = []

    def _fake_build(**kwargs):
        builds.append(kwargs)
        return object()

    monkeypatch.setattr(fmha_kernels, "build_flash_attn_func_module", _fake_build)
    fmha_kernels._get_kernel.cache_clear()
    common = {
        "device_index": 0,
        "num_heads": 2,
        "head_dim": 128,
        "causal": False,
        "dtype_str": "bf16",
        "waves_per_eu": 2,
        "daz": True,
        "block_m": 128,
        "block_n": 32,
        "softmax_scale": None,
        "tail_mask": False,
        "cross_attn": False,
    }
    fmha_kernels._get_kernel(**common, lds_vec_width=16)
    fmha_kernels._get_kernel(**common, lds_vec_width=8)
    fmha_kernels._get_kernel(**common, lds_vec_width=16)
    assert len(builds) == 2
    assert [build["lds_vec_width"] for build in builds] == [16, 8]
    fmha_kernels._get_kernel.cache_clear()


def test_flydsl_fmha_rejects_invalid_stream_type():
    q, k, v = _make_qkv(1, 128, 2, 128, torch.bfloat16)
    with pytest.raises(TypeError, match="torch.cuda.Stream or None"):
        flydsl_flash_attn_func(q, k, v, stream=object())


def test_low_level_fp8_quant_rejects_invalid_stream_type():
    from aiter.ops.flydsl.kernels.fp8_quant_gfx1201 import (
        flydsl_fp8_pertensor_quant,
    )

    x = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(TypeError, match="torch.cuda.Stream or None"):
        flydsl_fp8_pertensor_quant(x, rotate=False, stream=object())


@pytest.mark.parametrize(
    "field",
    ["seq_len", "seq_len_kv_real", "seq_len_kv"],
)
def test_gfx1201_launch_limits_reject_int32_overflow(field):
    from aiter.ops.flydsl import fmha_kernels

    lengths = {"seq_len": 128, "seq_len_kv_real": 128, "seq_len_kv": 128}
    lengths[field] = 1 << 31
    with pytest.raises(ValueError, match=rf"{1 << 31}.*Int32 limit"):
        fmha_kernels._validate_gfx1201_launch_limits(
            **lengths, num_heads=1, head_dim=64, fp8=True
        )


def test_gfx1201_launch_limits_validate_bf16_buffer_descriptor():
    from aiter.ops.flydsl import fmha_kernels

    max_seq_below_limit = ((1 << 32) - 1) // (64 * 2)
    fmha_kernels._validate_gfx1201_launch_limits(
        seq_len=128,
        seq_len_kv_real=max_seq_below_limit,
        seq_len_kv=max_seq_below_limit,
        num_heads=1,
        head_dim=64,
        fp8=False,
    )
    with pytest.raises(ValueError, match=r"byte count below 2\^32"):
        fmha_kernels._validate_gfx1201_launch_limits(
            seq_len=128,
            seq_len_kv_real=max_seq_below_limit + 1,
            seq_len_kv=max_seq_below_limit + 1,
            num_heads=1,
            head_dim=64,
            fp8=False,
        )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_flydsl_fmha_cross_attention_batch_two(head_dim):
    q = torch.randn(2, 1000, 8, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(2, 777, 8, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)
    cosine = F.cosine_similarity(
        out.float().reshape(-1, head_dim), ref.float().reshape(-1, head_dim), dim=1
    )
    assert cosine.min().item() > 0.99
    assert cosine.mean().item() > 0.999


def test_gfx1201_compiled_caches_are_device_specific():
    """Identical BF16, FP8, and producer configs compile per concrete GPU."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 visible GPUs")

    from aiter.ops.flydsl import fmha_kernels
    from aiter.ops.flydsl.kernels import fp8_quant_gfx1201

    fmha_kernels._get_kernel.cache_clear()
    fmha_kernels._get_fp8_gfx1201_kernel.cache_clear()
    fp8_quant_gfx1201._compile.cache_clear()

    for index in (0, 1):
        device = torch.device("cuda", index)
        with torch.cuda.device(device):
            q = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16, device=device)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            bf16_out = flydsl_flash_attn_func(q, k, v)
            ref = _ref_sdpa_bshd(q, k, v)
            assert bf16_out.device == device
            bf16_cos = F.cosine_similarity(
                bf16_out.float().reshape(-1, 128),
                ref.float().reshape(-1, 128),
                dim=1,
            )
            assert bf16_cos.mean().item() > 0.999

            q8, sq = fp8_quant_gfx1201.flydsl_fp8_pertensor_quant(q, rotate=True)
            k8, sk = fp8_quant_gfx1201.flydsl_fp8_pertensor_quant(k, rotate=True)
            v8, sv = fp8_quant_gfx1201.flydsl_fp8_pertensor_quant(v, rotate=False)
            assert all(t.device == device for t in (q8, k8, v8, sq, sk, sv))

            fp8_out = flydsl_flash_attn_func(
                q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv
            )
            assert fp8_out.device == device
            fp8_cos = F.cosine_similarity(
                fp8_out.float().reshape(-1, 128),
                ref.float().reshape(-1, 128),
                dim=1,
            )
            assert fp8_cos.mean().item() > 0.998


def test_public_triton_quant_uses_input_device_when_current_device_differs():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 visible GPUs")
    from aiter.ops.flydsl import fmha_kernels

    if not fmha_kernels._HAS_TRITON:
        pytest.skip("Triton is unavailable")

    original_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        for index in (1, 0, 1):
            device = torch.device("cuda", index)
            q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device=device)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(q, k, v, backend="triton")

            assert torch.cuda.current_device() == 0
            assert all(t.device == device for t in (q8, k8, v8, sq, sk, sv))
            assert all(torch.isfinite(s).all().item() for s in (sq, sk, sv))

            out = flydsl_flash_attn_func(
                q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv
            )
            ref = _ref_sdpa_bshd(q, k, v)
            cosine = F.cosine_similarity(
                out.float().reshape(-1, 64), ref.float().reshape(-1, 64), dim=1
            )
            assert out.device == device
            assert cosine.mean().item() > 0.998
    finally:
        torch.cuda.set_device(original_device)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
