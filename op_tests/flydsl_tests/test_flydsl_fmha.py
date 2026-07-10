# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``flydsl_flash_attn_func`` (gfx1201 / RDNA4)."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")
from aiter.ops.flydsl import (  # noqa: E402
    is_flydsl_available,
    flydsl_flash_attn_func,
    flydsl_fp8_quant,
)

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)


def _is_gfx1201() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, seq_len, num_heads, head_dim)
    q = torch.randn(shape, generator=g, dtype=dtype, device=device)
    k = torch.randn(shape, generator=g, dtype=dtype, device=device)
    v = torch.randn(shape, generator=g, dtype=dtype, device=device)
    return q, k, v


@pytest.mark.parametrize(
    "batch,seq_len,num_heads,head_dim",
    [
        # Aligned production-like Wan2.1 1.3B shape, padded to multiple of 128.
        (1, 32768, 12, 128),
        # Smaller aligned shape (sanity).
        (2, 1024, 8, 128),
        # Unaligned shape — exercises the auto-padding path. 32760 → 32768.
        (1, 32760, 12, 128),
        # Flux self-attn, short seq (128/32 tile).
        (1, 512, 24, 128),
        # Flux self-attn, long seq (256/64 tile).
        (1, 1536, 24, 128),
        # SD3 joint-attn seq.
        (1, 1024, 24, 128),
        # head_dim=64 self-attn (128/64 tile).
        (1, 2048, 16, 64),
    ],
)
def test_flydsl_fmha_correctness_bf16(batch, seq_len, num_heads, head_dim):
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v)

    assert out.shape == ref.shape == (batch, seq_len, num_heads, head_dim)
    assert out.dtype == ref.dtype == torch.bfloat16

    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    # bf16 attention is noisy; cosine is the right correctness signal.
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > 0.999, f"mean_cos={cos.mean().item():.6f}"


@pytest.mark.parametrize(
    "batch,seq_q,seq_kv,num_heads,head_dim",
    [
        (1, 1024, 512, 12, 128),  # Wan-style long Q, short text K/V.
        (1, 4096, 512, 12, 128),
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
        batch, seq_q, num_heads, head_dim, generator=g,
        dtype=torch.bfloat16, device="cuda",
    )
    k = torch.randn(
        batch, seq_kv, num_heads, head_dim, generator=g,
        dtype=torch.bfloat16, device="cuda",
    )
    v = torch.randn(
        batch, seq_kv, num_heads, head_dim, generator=g,
        dtype=torch.bfloat16, device="cuda",
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


def test_flydsl_fmha_rejects_unsupported_head_dim():
    q = torch.randn(1, 256, 8, 48, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="head_dim"):
        flydsl_flash_attn_func(q, q.clone(), q.clone())


def test_flydsl_fmha_rejects_dtype_mismatch():
    q = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 1024, 8, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        flydsl_flash_attn_func(q, k, v)


def test_flydsl_fmha_rejects_gqa():
    """Grouped-query attention (num_heads_q != num_heads_k) is unsupported; the
    kernel assumes equal head counts."""
    q = torch.randn(1, 1024, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="num_heads"):
        flydsl_flash_attn_func(q, k, v)


def test_flydsl_fmha_correctness_f16():
    """f16 dtype coverage — Wan2.1 1.3B-style shape, non-causal."""
    batch, seq_len, num_heads, head_dim = 1, 32768, 12, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.float16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == ref.shape == (batch, seq_len, num_heads, head_dim)
    assert out.dtype == ref.dtype == torch.float16

    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > 0.999, f"mean_cos={cos.mean().item():.6f}"


def test_flydsl_fmha_correctness_causal_small():
    """Causal masking coverage — small bf16 shape."""
    batch, seq_len, num_heads, head_dim = 2, 4096, 8, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=True)
    ref = _ref_sdpa_bshd(q, k, v, causal=True)

    assert out.shape == ref.shape == (batch, seq_len, num_heads, head_dim)
    assert out.dtype == ref.dtype == torch.bfloat16

    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > 0.999, f"mean_cos={cos.mean().item():.6f}"


def test_flydsl_fmha_correctness_multi_device():
    """Kernel runs on q's device when it differs from the current device.

    Runs the kernel on device 1 while the current device is 0, in a subprocess
    so a HIP context-pollution failure cannot leak into the rest of the test
    session. Exercises the ``with torch.cuda.device(...)`` wrap in
    ``flydsl_flash_attn_func``.

    If the FlyDSL runtime pins to device 0 internally (a runtime limitation, not
    a wrapper bug), the subprocess raises ``hipErrorInvalidDevice`` and the test
    is xfail'd; the same-device guard test below still covers the device check.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 visible GPUs")

    import subprocess
    import textwrap

    script = textwrap.dedent("""
        import torch
        import torch.nn.functional as F
        from aiter.ops.flydsl import flydsl_flash_attn_func

        torch.cuda.set_device(0)
        dev1 = torch.device("cuda", 1)
        B, S, H, D = 1, 1024, 8, 128
        g = torch.Generator(device=dev1).manual_seed(0)
        shape = (B, S, H, D)
        q = torch.randn(shape, generator=g, dtype=torch.bfloat16, device=dev1)
        k = torch.randn(shape, generator=g, dtype=torch.bfloat16, device=dev1)
        v = torch.randn(shape, generator=g, dtype=torch.bfloat16, device=dev1)

        out = flydsl_flash_attn_func(q, k, v, causal=False)
        torch.cuda.synchronize(dev1)
        assert out.device == dev1, f"expected cuda:1 got {out.device}"

        with torch.cuda.device(dev1):
            ref_bhsd = F.scaled_dot_product_attention(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                is_causal=False,
            )
            ref = ref_bhsd.transpose(1, 2).contiguous()
        cos = F.cosine_similarity(
            out.float().reshape(-1, D),
            ref.float().reshape(-1, D),
            dim=1,
        )
        cm = cos.min().item()
        assert cm > 0.99, f"min_cos={cm:.6f}"
        print("MULTI_DEVICE_OK", flush=True)
        """)

    proc = subprocess.run(
        ["python", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "MULTI_DEVICE_OK" in proc.stdout:
        return
    if "hipErrorInvalidDevice" in combined or "invalid device ordinal" in combined:
        pytest.xfail(
            "FlyDSL runtime pins to device 0; wrapper-level device-context "
            "switch is in place but underlying runtime does not honor it"
        )
    raise AssertionError(
        f"multi-device subprocess failed unexpectedly:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


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


def test_flydsl_fmha_allows_tight_padding():
    """Wan2.1 production case (S_real=32760 -> S_pad=32768) must produce
    SDPA-equivalent output. The kernel bounds the non-causal KV loop at the
    real length and tail-masks the straddling last tile, so tight padding is
    exact. Regression guard for the production hot path.
    """
    batch, seq_len, num_heads, head_dim = 1, 32760, 12, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == ref.shape == (batch, seq_len, num_heads, head_dim)
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    # Tight padding is exact in practice (cos_min ~0.99999 on this shape);
    # 0.9999 is a conservative bf16 regression bound.
    assert cos.min().item() > 0.9999, f"min_cos={cos.min().item():.6f}"


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


def test_flydsl_fmha_missing_fp8_descale_raises():
    """FP8 inputs without descales must raise (they are required)."""
    q, k, v = _make_qkv(1, 1024, 8, 128, torch.bfloat16)
    qq, kk, vv, *_ = flydsl_fp8_quant(q, k, v)
    with pytest.raises(ValueError, match="descale"):
        flydsl_flash_attn_func(qq, kk, vv, causal=False)


def test_flydsl_fmha_softmax_scale():
    """Custom softmax_scale must match SDPA with the same scale."""
    batch, seq_len, num_heads, head_dim = 2, 2048, 8, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    scale = 0.05
    out = flydsl_flash_attn_func(q, k, v, causal=False, softmax_scale=scale)
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
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"


def test_flydsl_fmha_out_buffer():
    """Preallocated ``out=`` buffer is written in place and returned."""
    batch, seq_len, num_heads, head_dim = 2, 2048, 8, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    out = torch.empty(
        batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    ret = flydsl_flash_attn_func(q, k, v, causal=False, out=out)
    assert ret.data_ptr() == out.data_ptr()

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
        qq, kk, vv, causal=False, softmax_scale=scale,
        q_descale=sq, k_descale=sk, v_descale=sv, out=out,
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
    """``out=`` with the wrong shape or dtype must raise (guards the
    preallocated-buffer path)."""
    q, k, v = _make_qkv(1, 1024, 8, 128, torch.bfloat16)
    bad_shape = torch.empty(1, 512, 8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="shape"):
        flydsl_flash_attn_func(q, k, v, out=bad_shape)
    bad_dtype = torch.empty(1, 1024, 8, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        flydsl_flash_attn_func(q, k, v, out=bad_dtype)


def test_flydsl_fmha_positional_backcompat():
    """The original positional signature (q, k, v, causal, waves_per_eu, daz,
    stream) must keep working so pre-existing callers are unaffected by the new
    optional params (which are appended after it)."""
    q, k, v = _make_qkv(1, 1024, 8, 128, torch.bfloat16)
    out_pos = flydsl_flash_attn_func(q, k, v, False, 2, True)  # positional
    out_kw = flydsl_flash_attn_func(
        q, k, v, causal=False, waves_per_eu=2, daz=True
    )
    torch.testing.assert_close(out_pos, out_kw, rtol=0, atol=0)


def test_flydsl_fmha_rejects_device_mismatch():
    """Same-device check (#6) — q on device 0, k/v on device 1 must raise."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 visible GPUs")

    q = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda:0")
    k = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda:1")
    v = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda:1")
    with pytest.raises(ValueError, match="same device"):
        flydsl_flash_attn_func(q, k, v)
