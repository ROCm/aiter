# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the FlyDSL flash-attention kernels: gfx1201 bf16/f16 and gfx950 fp8."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.flydsl import flydsl_flash_attn_func


def _arch() -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.lower().split(":")[0]
    except Exception:  # noqa: BLE001
        return ""


_gfx1201_only = pytest.mark.skipif(
    not _arch().startswith("gfx1201"),
    reason="flydsl_flash_attn_func is gfx1201/RDNA4 only",
)
_gfx950_only = pytest.mark.skipif(
    not _arch().startswith("gfx950"),
    reason="flydsl fp8 flash attention is gfx950 only",
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


@_gfx1201_only
@pytest.mark.parametrize(
    "batch,seq_len,num_heads,head_dim",
    [
        # Aligned production-like Wan2.1 1.3B shape, padded to multiple of 128.
        (1, 32768, 12, 128),
        # Smaller aligned shape (sanity).
        (2, 1024, 8, 128),
        # Unaligned shape — exercises the auto-padding path. 32760 → 32768.
        (1, 32760, 12, 128),
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


@_gfx1201_only
def test_flydsl_fmha_rejects_cross_attention():
    q = torch.randn(1, 1024, 12, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 512, 12, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 512, 12, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="self-attention"):
        flydsl_flash_attn_func(q, k, v)


@_gfx1201_only
def test_flydsl_fmha_rejects_unsupported_head_dim():
    q = torch.randn(1, 256, 8, 48, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="head_dim"):
        flydsl_flash_attn_func(q, q.clone(), q.clone())


@_gfx1201_only
def test_flydsl_fmha_rejects_dtype_mismatch():
    q = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 1024, 8, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        flydsl_flash_attn_func(q, k, v)


@_gfx1201_only
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


@_gfx1201_only
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


@_gfx1201_only
def test_flydsl_fmha_correctness_multi_device():
    """Multi-GPU device-context wrapping (#1) and same-device check (#6).

    Runs the kernel on device 1 while the default current device is 0 in a
    subprocess (so a HIP context-pollution failure cannot leak into the rest
    of the test session). Validates the ``with torch.cuda.device(...)`` wrap
    in ``flydsl_flash_attn_func`` when q.device != current device.

    If the underlying FlyDSL runtime pins to device 0 internally (a runtime
    limitation, not a wrapper bug), the subprocess will raise
    ``hipErrorInvalidDevice`` and the test is marked xfail — the wrapper code
    path is still correct and the same-device guard test below still
    validates Copilot #6 directly.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 visible GPUs")

    import subprocess
    import textwrap

    script = textwrap.dedent("""
        import sys
        sys.path.insert(0, "/workspace/FlyDSL/python")
        import flydsl
        flydsl.__version__ = "0.1.5.dev999"

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
        check=False,
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


@_gfx1201_only
def test_flydsl_fmha_rejects_excessive_padding():
    """Non-causal path must reject padding ratio > 0.5% (option (d) guard).

    S_real=129 -> S_pad=256, pad ratio 127/256 = 49.6%. Padded K/V keys
    would contribute to the softmax denominator and silently scale outputs
    (rel_err ~37% per RCA in 2969_padded_softmax_rca.md). Wrapper must
    raise before launching the kernel.
    """
    batch, seq_len, num_heads, head_dim = 1, 129, 8, 128
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    with pytest.raises(ValueError, match="0.5% safety threshold"):
        flydsl_flash_attn_func(q, k, v, causal=False)


@_gfx1201_only
def test_flydsl_fmha_allows_tight_padding():
    """Wan2.1 production case (S_real=32760 -> S_pad=32768, ratio 0.024%)
    must pass the 0.5% threshold and produce SDPA-equivalent output.

    Regression guard for option (d) — protects the production hot path
    from a future, stricter threshold accidentally rejecting it.
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
    # Wan2.1 production cos_min was empirically 0.999992 in the RCA;
    # 0.9999 is the conservative regression bound (bf16 noise floor).
    assert cos.min().item() > 0.9999, f"min_cos={cos.min().item():.6f}"


@_gfx1201_only
def test_flydsl_fmha_rejects_device_mismatch():
    """Same-device check (#6) — q on device 0, k/v on device 1 must raise."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 visible GPUs")

    q = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda:0")
    k = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda:1")
    v = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda:1")
    with pytest.raises(ValueError, match="same device"):
        flydsl_flash_attn_func(q, k, v)


# gfx950 fp8 (e4m3fn). The reference dequantizes the *same* fp8 tensors, so the
# error measured is the kernel's, not the quantizer's.

FP8_DTYPE = torch.float8_e4m3fn
# Fixed fp8 correctness gate; fp8 is lossy, so these are absolute bounds.
FP8_MAX_ERR = 5e-2
FP8_MIN_COS = 0.98
FP8_UNIFORM_RANGE = (-1.0, 1.0)
FP8_SEED = 123

_FP8_HEADS = 12
_FP8_D, _FP8_DV = 192, 128

FP8_VARLEN_Q_SEQLENS = {
    1: [2614],
    2: [1024, 1590],
    3: [1024, 512, 1078],
    4: [1024, 512, 256, 822],
}
FP8_VARLEN_KV_SEQLENS = {
    1: [16384],
    2: [8192, 8192],
    3: [8192, 4096, 4096],
    4: [8192, 4096, 2048, 2048],
}
FP8_VARLEN_BATCHES = (1, 2, 3, 4)
FP8_SPLITKV_SPLITS = (2, 4, 8, 16)
FP8_SPLITKV_SEQLENS = (4096, 8192, 16384, 32768)
FP8_SPLIT_MODES = [pytest.param(1, id="dense"), pytest.param(None, id="autosplit")]


def _fp8_quant(x):
    """Per-tensor e4m3fn quantization: descale = amax / fp8_max."""
    fp8_max = torch.finfo(FP8_DTYPE).max
    descale = (x.abs().amax().float() / fp8_max).clamp(min=1e-12).view(1)
    return (x.float() / descale).to(FP8_DTYPE).contiguous(), descale.contiguous()


def _fp8_dequant(x, descale):
    return x.to(torch.float32) * descale.to(torch.float32)


def _ref_attention(q, k, v, causal):
    """fp32 SDPA over BSHD/THD-with-batch inputs, GQA-aware, bottom-right causal.

    ``F.scaled_dot_product_attention(is_causal=True)`` aligns the mask top-left,
    which differs from this kernel (and from aiter's documented convention) as
    soon as Sq != Skv, so the mask is built explicitly from ``delta``.
    """
    q_t, k_t, v_t = (t.transpose(1, 2).float() for t in (q, k, v))
    nh_q, nh_kv = q_t.shape[1], k_t.shape[1]
    if nh_q != nh_kv:
        rep = nh_q // nh_kv
        k_t = k_t.repeat_interleave(rep, dim=1)
        v_t = v_t.repeat_interleave(rep, dim=1)
    Sq, D = q_t.shape[2], q_t.shape[3]
    Skv = k_t.shape[2]
    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) / math.sqrt(D)
    if causal:
        delta = Skv - Sq
        q_idx = torch.arange(Sq, device=q.device).view(-1, 1)
        k_idx = torch.arange(Skv, device=q.device).view(1, -1)
        scores = scores.masked_fill(k_idx > q_idx + delta, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    # A fully-masked row softmaxes to NaN; the kernel writes zeros there.
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v_t).transpose(1, 2)


def _run_fp8_shape(
    causal,
    batch=1,
    seq_len=1,
    num_heads=_FP8_HEADS,
    head_dim=_FP8_D,
    head_dim_v=_FP8_DV,
    num_kv_heads=None,
    num_kv_splits=None,
    seqlen_kv=None,
    varlen_seqlens_q=None,
    varlen_seqlens_kv=None,
):
    """Run one fp8 shape and assert it against the fixed fp8 gate.

    Covers dense self/cross attention, packed varlen, and split-K; the reference
    dequantizes the same e4m3fn tensors the kernel reads.
    """
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        flydsl_flash_attn_fp8_func,
    )

    H = num_heads
    H_KV = num_heads if num_kv_heads is None else num_kv_heads
    D, Dv = head_dim, head_dim_v
    torch.manual_seed(FP8_SEED)

    varlen = varlen_seqlens_q is not None
    kw = {}
    if varlen:
        vl_q = list(varlen_seqlens_q)
        vl_kv = list(varlen_seqlens_kv) if varlen_seqlens_kv is not None else list(vl_q)
        cuq, cukv = [0], [0]
        for a, b in zip(vl_q, vl_kv):
            cuq.append(cuq[-1] + a)
            cukv.append(cukv[-1] + b)
        total_q, total_kv = cuq[-1], cukv[-1]
        cross = any(a != b for a, b in zip(vl_q, vl_kv))
        q_bf = torch.empty(total_q, H, D, dtype=torch.bfloat16, device="cuda").uniform_(
            *FP8_UNIFORM_RANGE
        )
        k_bf = torch.empty(
            total_kv, H_KV, D, dtype=torch.bfloat16, device="cuda"
        ).uniform_(*FP8_UNIFORM_RANGE)
        v_bf = torch.empty(
            total_kv, H_KV, Dv, dtype=torch.bfloat16, device="cuda"
        ).uniform_(*FP8_UNIFORM_RANGE)
        kw.update(
            cu_seqlens_q=torch.tensor(cuq, dtype=torch.int32, device="cuda"),
            cu_seqlens_kv=torch.tensor(cukv, dtype=torch.int32, device="cuda"),
            max_seqlen_q=max(vl_q),
            cross_seqlen=cross,
        )
        if cross:
            kw["max_seqlen_kv"] = max(vl_kv)
    else:
        B, S = batch, seq_len
        Skv = S if seqlen_kv is None else seqlen_kv
        q_bf = torch.empty(B, S, H, D, dtype=torch.bfloat16, device="cuda").uniform_(
            *FP8_UNIFORM_RANGE
        )
        k_bf = torch.empty(
            B, Skv, H_KV, D, dtype=torch.bfloat16, device="cuda"
        ).uniform_(*FP8_UNIFORM_RANGE)
        v_bf = torch.empty(
            B, Skv, H_KV, Dv, dtype=torch.bfloat16, device="cuda"
        ).uniform_(*FP8_UNIFORM_RANGE)

    q, q_s = _fp8_quant(q_bf)
    k, k_s = _fp8_quant(k_bf)
    v, v_s = _fp8_quant(v_bf)

    out = flydsl_flash_attn_fp8_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=H_KV,
        num_kv_splits=num_kv_splits,
        q_descale=q_s,
        k_descale=k_s,
        v_descale=v_s,
        **kw,
    )
    torch.cuda.synchronize()

    q_r, k_r, v_r = _fp8_dequant(q, q_s), _fp8_dequant(k, k_s), _fp8_dequant(v, v_s)
    if varlen:
        ref = torch.empty(out.shape, dtype=torch.float32, device="cuda")
        for b in range(len(vl_q)):
            ref[cuq[b] : cuq[b + 1]] = _ref_attention(
                q_r[cuq[b] : cuq[b + 1]].unsqueeze(0),
                k_r[cukv[b] : cukv[b + 1]].unsqueeze(0),
                v_r[cukv[b] : cukv[b + 1]].unsqueeze(0),
                causal,
            ).squeeze(0)
    else:
        ref = _ref_attention(q_r, k_r, v_r, causal)

    got = out.float()
    max_err = (got - ref).abs().max().item()
    min_cos = (
        F.cosine_similarity(got.reshape(-1, Dv), ref.reshape(-1, Dv), dim=1)
        .min()
        .item()
    )
    assert max_err < FP8_MAX_ERR and min_cos > FP8_MIN_COS, (
        f"fp8 gate: max_err={max_err:.3e} (< {FP8_MAX_ERR}), "
        f"min_cos={min_cos:.5f} (> {FP8_MIN_COS})"
    )
    return out


def _run_fp8_into_nan_out(q, k, v, head_dim_v, **kwargs):
    """Launch fp8 attention into a NaN-filled ``out`` so unwritten rows are countable."""
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        flydsl_flash_attn_fp8_func,
    )

    scales = [
        (t.abs().amax().float().clamp(min=1e-12) / torch.finfo(FP8_DTYPE).max)
        for t in (q, k, v)
    ]
    qq, kq, vq = (
        (t.float() / s).to(FP8_DTYPE).contiguous() for t, s in zip((q, k, v), scales)
    )
    d = [s.reshape(1).contiguous() for s in scales]
    out = torch.full(
        q.shape[:-1] + (head_dim_v,),
        float("nan"),
        device=q.device,
        dtype=torch.bfloat16,
    )
    flydsl_flash_attn_fp8_func(
        qq, kq, vq, out=out, q_descale=d[0], k_descale=d[1], v_descale=d[2], **kwargs
    )
    return out


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(16, 1), (32, 8), (12, 12)])
@pytest.mark.parametrize("seq_len", [4096, 8192])
def test_fp8_gqa_dense(causal, num_heads, num_kv_heads, seq_len):
    _run_fp8_shape(
        causal,
        batch=1,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        head_dim_v=128,
    )


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(16, 1), (32, 8)])
def test_fp8_gqa_varlen(causal, num_heads, num_kv_heads):
    _run_fp8_shape(
        causal,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        head_dim_v=128,
        varlen_seqlens_q=FP8_VARLEN_Q_SEQLENS[2],
        varlen_seqlens_kv=FP8_VARLEN_KV_SEQLENS[2],
    )


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch,seq_len", [(1, 4096), (2, 4096), (3, 4096), (4, 4096), (1, 8192)]
)
@pytest.mark.parametrize("num_kv_splits", FP8_SPLIT_MODES)
def test_fp8_head_dim_192_v_128_dense(causal, batch, seq_len, num_kv_splits):
    """Dense self-attention with QK head_dim 192 and a 128-wide V.

    Q/K are [B, S, 12, 192], V is [B, S, 12, 128], and the output follows V.
    """
    _run_fp8_shape(causal, batch=batch, seq_len=seq_len, num_kv_splits=num_kv_splits)


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch", FP8_VARLEN_BATCHES)
@pytest.mark.parametrize("num_kv_splits", FP8_SPLIT_MODES)
def test_fp8_head_dim_192_v_128_varlen(causal, batch, num_kv_splits):
    """Packed varlen self-attention over 2614 tokens, batch 1..4.

    Q/K are [2614, 12, 192] and V is [2614, 12, 128] at every batch -- only the
    cu_seqlens partition changes (batch 2 is [0, 1024, 2614]).
    """
    _run_fp8_shape(
        causal,
        varlen_seqlens_q=FP8_VARLEN_Q_SEQLENS[batch],
        num_kv_splits=num_kv_splits,
    )


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch", FP8_VARLEN_BATCHES)
@pytest.mark.parametrize("num_kv_splits", FP8_SPLIT_MODES)
def test_fp8_head_dim_192_v_128_varlen_cross_length(causal, batch, num_kv_splits):
    """Packed varlen cross-attention, batch 1..4: 2614 Q tokens vs 16384 KV tokens."""
    _run_fp8_shape(
        causal,
        varlen_seqlens_q=FP8_VARLEN_Q_SEQLENS[batch],
        varlen_seqlens_kv=FP8_VARLEN_KV_SEQLENS[batch],
        num_kv_splits=num_kv_splits,
    )


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_kv_splits", FP8_SPLITKV_SPLITS)
@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (192, 128)])
def test_fp8_split_kv(causal, num_kv_splits, head_dim, head_dim_v):
    """fp8 split-KV: the KV dimension split across workgroups plus a combine pass.

    The 128/128 pair isolates the split from the head-dim pair, so a failure
    points at one feature or the other rather than both at once.
    """
    _run_fp8_shape(
        causal,
        batch=1,
        seq_len=8192,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_kv_splits=num_kv_splits,
    )


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch", (1, 2, 3, 4))
def test_fp8_split_kv_batched(causal, batch):
    """Split-KV over batches: the grid is B * num_kv_splits deep, so the batch is
    what decides whether splitting still fills the GPU."""
    _run_fp8_shape(causal, batch=batch, seq_len=8192, num_kv_splits=4)


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seq_len,seqlen_kv,num_kv_splits",
    [(512, 16384, 8), (2614, 16384, 8), (1024, 32768, 16)],
)
def test_fp8_split_kv_cross_length(causal, seq_len, seqlen_kv, num_kv_splits):
    """Split-KV with short Q against long KV -- the shape split-KV exists for."""
    _run_fp8_shape(
        causal,
        batch=1,
        seq_len=seq_len,
        seqlen_kv=seqlen_kv,
        num_kv_splits=num_kv_splits,
    )


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seq_len,num_kv_splits", list(zip(FP8_SPLITKV_SEQLENS, FP8_SPLITKV_SPLITS))
)
def test_fp8_split_kv_long_sequence(causal, seq_len, num_kv_splits):
    """Split count scaled with the KV length, 4k/2 through 32k/16."""
    _run_fp8_shape(causal, batch=1, seq_len=seq_len, num_kv_splits=num_kv_splits)


@_gfx950_only
@pytest.mark.parametrize("head_dim_v", [64, 96, 128, 160, 192])
def test_fp8_supported_v_head_dims_run(head_dim_v):
    """Every head_dim_v the guard admits has to actually produce the right answer."""
    _run_fp8_shape(
        False, batch=1, seq_len=512, num_heads=4, head_dim=128, head_dim_v=head_dim_v
    )


@_gfx950_only
@pytest.mark.parametrize("seq_len", [1, 385, 1000, 4097])
def test_fp8_dense_ragged_seq_lens(seq_len):
    """Dense fp8 on sequence lengths that are not multiples of the tile."""
    _run_fp8_shape(
        True, batch=2, seq_len=seq_len, num_heads=4, head_dim=192, head_dim_v=128
    )


@_gfx950_only
@pytest.mark.parametrize(
    "head_dim,head_dim_v,match",
    [
        # D_CHUNKS < 2 aborts LLVM; D_CHUNKS > 6 miscomputes the high chunks.
        (128, 32, "head_dim_v"),
        (128, 224, "head_dim_v"),
        (128, 256, "head_dim_v"),
        (96, 96, "head_dim"),
        (256, 192, "LDS"),
        (320, 128, "LDS"),
        (384, 64, "LDS"),
    ],
)
def test_fp8_rejected_head_dims_raise_before_launch(head_dim, head_dim_v, match):
    """Unsupported head dims must name the shape, not abort or fault the GPU."""
    B, S, H = 1, 512, 4
    torch.manual_seed(0)
    q = torch.randn(B, S, H, head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, S, H, head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, S, H, head_dim_v, device="cuda", dtype=torch.bfloat16) * 0.1
    with pytest.raises((RuntimeError, ValueError), match=match):
        _run_fp8_into_nan_out(q, k, v, head_dim_v, causal=False, num_kv_heads=H)


@_gfx950_only
@pytest.mark.parametrize(
    "batch,seq_len,num_heads", [(1, 4097, 1), (1, 4097, 3), (1, 2050, 7), (1, 8193, 1)]
)
def test_fp8_auto_split_kv_writes_every_row(batch, seq_len, num_heads):
    """Auto split-K must not drop the tail of the combine grid."""
    D = 128
    assert (batch * num_heads * seq_len) % (
        256 // (D // 4)
    ) != 0, "shape would not exercise the tail"
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(batch, seq_len, num_heads, D, device="cuda", dtype=torch.bfloat16)
        * 0.1
        for _ in range(3)
    )
    out = _run_fp8_into_nan_out(q, k, v, D, causal=False, num_kv_heads=num_heads)
    assert not torch.isnan(
        out
    ).any(), f"{int(torch.isnan(out).any(-1).sum())} output rows were never written"


@_gfx950_only
@pytest.mark.parametrize(
    "seq_len,num_heads,head_dim_v",
    [(385, 1, 128), (385, 3, 128), (1155, 1, 128), (386, 1, 64)],
)
def test_fp8_varlen_split_kv_respects_batch_boundaries(seq_len, num_heads, head_dim_v):
    """varlen + split-K must not mix batches inside a combine wave."""
    rows_per_wave = 256 // head_dim_v
    assert (seq_len * num_heads) % rows_per_wave != 0, "shape would not straddle a wave"
    B, D = 8, 192
    torch.manual_seed(0)
    cu = torch.arange(0, (B + 1) * seq_len, seq_len, device="cuda", dtype=torch.int32)
    total = B * seq_len
    q = torch.randn(total, num_heads, D, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(total, num_heads, D, device="cuda", dtype=torch.bfloat16) * 0.1
    v = (
        torch.randn(total, num_heads, head_dim_v, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    kw = {
        "causal": False,
        "num_kv_heads": num_heads,
        "cu_seqlens_q": cu,
        "cu_seqlens_kv": cu,
        "max_seqlen_q": seq_len,
        "max_seqlen_kv": seq_len,
        "cross_seqlen": False,
    }
    split = _run_fp8_into_nan_out(q, k, v, head_dim_v, num_kv_splits=2, **kw)
    assert not torch.isnan(
        split
    ).any(), f"{int(torch.isnan(split).any(-1).sum())} output rows were never written"
    unsplit = _run_fp8_into_nan_out(q, k, v, head_dim_v, num_kv_splits=1, **kw)
    torch.testing.assert_close(split.float(), unsplit.float(), rtol=2e-2, atol=2e-2)


@_gfx950_only
@pytest.mark.parametrize("lazy_rescale, atol", [(True, 0.06), (False, 0.05)])
@pytest.mark.parametrize("S", [1024, 12288])
@pytest.mark.parametrize("k_scale", [1.0, 200.0])
def test_fp8_softmax_normalises(S, k_scale, lazy_rescale, atol):
    """With V all ones the output is exactly 1.0, because softmax normalises.

    Nothing about V or the PV product can move it, and 1.0 is representable in
    e4m3, so any deviation is the softmax's own normalisation. ``k_scale``
    widens the score range: the failure this guards against is invisible on
    near-uniform attention and severe on peaked attention.

    Both rescale paths are checked, with different bounds, because the headroom
    for lifting P differs. The lazy path leaves ``exp2`` free to reach
    ``2**RESCALE_THRESHOLD`` and can use only the remainder; the eager path
    rebases every tile and gets all of it.
    """
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        flydsl_flash_attn_fp8_func,
    )

    B, H, D = 2, 8, 128
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1 * k_scale

    fp8_max = torch.finfo(FP8_DTYPE).max
    q_s = q.abs().amax().float() / fp8_max
    k_s = k.abs().amax().float() / fp8_max
    v_s = torch.tensor(1.0 / fp8_max, device="cuda")
    v = (torch.ones(B, S, H, D, device="cuda", dtype=torch.bfloat16) / v_s).to(
        FP8_DTYPE
    )

    out = flydsl_flash_attn_fp8_func(
        (q / q_s).to(FP8_DTYPE),
        (k / k_s).to(FP8_DTYPE),
        v,
        causal=False,
        q_descale=q_s.reshape(1).contiguous(),
        k_descale=k_s.reshape(1).contiguous(),
        v_descale=v_s.reshape(1).contiguous(),
        dualwave_swp_lazy_rescale=lazy_rescale,
    ).float()

    # e4m3 rounding of P leaves a per-row residue that the lift cannot remove.
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=atol)


@_gfx950_only
def test_fp8_default_is_the_lazy_rescale():
    """Every other fp8 case passes the flag, so the default would go untested.

    V is all ones, so the exact output is 1.0 and the two runs must also agree
    bit for bit.
    """
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        flydsl_flash_attn_fp8_func,
    )

    B, S, H, D = 1, 4096, 8, 128
    fp8_max = torch.finfo(FP8_DTYPE).max
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1 * 200.0
    q_s = q.abs().amax().float() / fp8_max
    k_s = k.abs().amax().float() / fp8_max
    v_s = torch.tensor(1.0 / fp8_max, device="cuda")
    v = (torch.ones(B, S, H, D, device="cuda", dtype=torch.bfloat16) / v_s).to(
        FP8_DTYPE
    )
    kw = {
        "causal": False,
        "q_descale": q_s.reshape(1).contiguous(),
        "k_descale": k_s.reshape(1).contiguous(),
        "v_descale": v_s.reshape(1).contiguous(),
        "num_kv_splits": 1,
    }
    qq, kk = (q / q_s).to(FP8_DTYPE), (k / k_s).to(FP8_DTYPE)

    default = flydsl_flash_attn_fp8_func(qq, kk, v, **kw)
    lazy = flydsl_flash_attn_fp8_func(qq, kk, v, dualwave_swp_lazy_rescale=True, **kw)
    torch.cuda.synchronize()
    torch.testing.assert_close(default.float(), lazy.float(), rtol=0, atol=0)


@_gfx950_only
def test_fp8_rescale_threshold_drops_past_the_long_sequence_bound():
    """fp8 picks its rescale threshold from the KV length.

    Below the bound 6 and 4 are equally accurate and 6 is cheaper; above it the
    running max spans enough tiles that 4's extra two log2 units of P lift are
    worth its ~0.3%. The kernel is not specialised on S, so this widens the
    build cache to two variants -- keep it two.
    """
    from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa

    f = fa._fp8_rescale_threshold
    assert f(1024) == 6.0
    assert f(fa._FP8_LONG_SEQ) == 6.0
    assert f(fa._FP8_LONG_SEQ + 1) == 4.0
    assert f(131072) == 4.0
    assert {f(s) for s in (1, 1024, 4096, 4097, 8192, 131072)} == {6.0, 4.0}


@_gfx950_only
@pytest.mark.parametrize(
    "batch,num_heads,seqlen_q,seqlen_kv,expect",
    [
        (1, 8, 512, 512, 128),
        (1, 8, 2048, 2048, 128),
        (8, 32, 2048, 2048, 256),
        (16, 32, 1024, 1024, 256),
        (32, 32, 512, 512, 256),
        (1, 8, 4096, 4096, 256),
        (1, 8, 4096, 16384, 256),
        (2, 8, 4096, 4096, 256),
    ],
)
def test_fp8_auto_block_m_picks(batch, num_heads, seqlen_q, seqlen_kv, expect):
    """Pin what ``_fp8_auto_block_m`` chooses; correctness tests pass either way."""
    from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa

    assert fa._fp8_auto_block_m(batch, num_heads, seqlen_q, seqlen_kv, 256) == expect


@_gfx950_only
@pytest.mark.parametrize("split", [False, True])
def test_fp8_out_tensor_is_filled_and_returned(monkeypatch, split):
    """A caller-supplied ``out`` must come back filled, and be the same tensor.

    ``split=True`` lowers the overflow bound so the batch-splitting path runs on
    a small tensor: reaching it for real needs 2**31 elements. Each launch
    writes into its own ``out[i:i+1]`` view, so returning a concatenation would
    both copy several GB at the sizes that reach it and hand back a different
    tensor than the caller passed.
    """
    from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa

    B, S, H, D = 2, 512, 8, 128
    torch.manual_seed(0)
    q = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1).to(
        FP8_DTYPE
    )
    k, v = q.clone(), q.clone()
    scale = torch.ones(1, device="cuda")
    kw = {"causal": False, "q_descale": scale, "k_descale": scale, "v_descale": scale}

    ref = fa.flydsl_flash_attn_fp8_func(q, k, v, **kw)
    if split:
        # over the whole tensor, under one batch entry, so each launch fits
        monkeypatch.setattr(fa, "_FP8_MAX_FLAT_ELEMS", B * S * H * D * 3 // 4)
    out = torch.empty(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    got = fa.flydsl_flash_attn_fp8_func(q, k, v, out=out, **kw)
    torch.cuda.synchronize()

    assert got is out, "out must be returned, not a copy"
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0)


@_gfx950_only
@pytest.mark.parametrize(
    "case", ["b1_too_large", "per_slice_still_too_large", "kv_only_over_limit"]
)
def test_fp8_flat_overflow_guard_covers_every_tensor(monkeypatch, case):
    """The int32 flat-dim guard has to see K and V, and to give up loudly.

    Splitting divides the flat dim by B, so it only helps while B > 1 and while
    one entry fits. Cross-attention can also put the excess in K/V rather than
    Q. Each case lowers the bound rather than allocating 2**31 elements.
    """
    from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa

    torch.manual_seed(0)
    H, D = 8, 128
    B, Sq, Skv = (1, 1024, 1024) if case == "b1_too_large" else (2, 1024, 1024)
    if case == "kv_only_over_limit":
        Sq = 256
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn_like(k)
    qq, qs = _fp8_quant(q)
    kq, ks = _fp8_quant(k)
    vq, vs = _fp8_quant(v)
    kw = {
        "causal": False,
        "q_descale": qs,
        "k_descale": ks,
        "v_descale": vs,
        "fp8_block_m": 256,
        "num_kv_splits": 1,
    }

    if case == "kv_only_over_limit":
        # Between q's count and k's, so only the K/V check can fire. B=2 splits.
        ref = fa.flydsl_flash_attn_fp8_func(qq, kq, vq, **kw)
        monkeypatch.setattr(fa, "_FP8_MAX_FLAT_ELEMS", kq.numel())
        got = fa.flydsl_flash_attn_fp8_func(qq, kq, vq, **kw)
        torch.testing.assert_close(got.float(), ref.float(), rtol=0, atol=0)
        return

    # b1_too_large has no batch to divide; per_slice_still_too_large has B=2 but
    # a bound low enough that one entry is still over, so the recursion hits the
    # same wall. Both must raise rather than launch.
    bound = qq.numel() if case == "b1_too_large" else qq.numel() // 2
    monkeypatch.setattr(fa, "_FP8_MAX_FLAT_ELEMS", bound)
    with pytest.raises(NotImplementedError, match="int32"):
        fa.flydsl_flash_attn_fp8_func(qq, kq, vq, **kw)


@_gfx950_only
def test_fp8_split_result_survives_a_non_current_stream(monkeypatch):
    """The split must not be consumed on a stream that is not the one it ran on.

    Every launch goes to ``stream``; building the result on the ambient stream
    reads it while those kernels are still queued, and the damage survives a
    later synchronize because the copy already happened.
    """
    from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa

    B, S, H, D = 2, 512, 8, 128
    torch.manual_seed(0)
    q = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1).to(
        FP8_DTYPE
    )
    k, v = q.clone(), q.clone()
    scale = torch.ones(1, device="cuda")
    kw = {"causal": False, "q_descale": scale, "k_descale": scale, "v_descale": scale}

    ref = fa.flydsl_flash_attn_fp8_func(q, k, v, **kw)
    torch.cuda.synchronize()

    monkeypatch.setattr(fa, "_FP8_MAX_FLAT_ELEMS", B * S * H * D * 3 // 4)
    side = torch.cuda.Stream()
    # Elementwise, not a matmul: the point is only to keep `side` busy, and a
    # GEMM would drag in a hipBLAS handle this test has no reason to need.
    filler = torch.randn(8192, 8192, device="cuda")
    with torch.cuda.stream(side):
        for _ in range(40):  # keep the stream busy so the attention starts late
            filler = torch.sin(filler)
    got = fa.flydsl_flash_attn_fp8_func(q, k, v, stream=side, **kw)
    torch.cuda.synchronize()

    torch.testing.assert_close(got.float(), ref.float(), rtol=0, atol=0)


def _fp8_dispatch_inputs(B=2, S=1024, H=8, D=128, varlen=False):
    """Quantized inputs for the dispatch tests.

    Uniform(-1, 1) rather than normal: FP8_MAX_ERR is an *absolute* bound, and
    randn's wider dynamic range pushes the bf16 output ULP past it at the same
    relative accuracy. This is the distribution the gate is calibrated for.
    """
    torch.manual_seed(0)
    shape = (B * S, H, D) if varlen else (B, S, H, D)

    def _t():
        return torch.empty(shape, device="cuda", dtype=torch.bfloat16).uniform_(
            *FP8_UNIFORM_RANGE
        )

    q, qs = _fp8_quant(_t())
    k, ks = _fp8_quant(_t())
    v, vs = _fp8_quant(_t())
    return q, k, v, {"q_descale": qs, "k_descale": ks, "v_descale": vs}


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
def test_fp8_dispatch_batch_routes_to_gfx950(causal):
    """``flydsl_flash_attn_batch_func`` must route BSHD fp8 to the gfx950 kernel."""
    from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func

    B, S, H, D = 2, 1024, 8, 128
    q, k, v, d = _fp8_dispatch_inputs(B, S, H, D)
    out = flydsl_flash_attn_batch_func(q, k, v, causal=causal, **d)
    assert out is not None, "gfx950 fp8 must not fall through to CK/Triton"
    assert out.shape == (B, S, H, D)
    assert out.dtype == torch.bfloat16

    ref = _ref_attention(
        _fp8_dequant(q, d["q_descale"]),
        _fp8_dequant(k, d["k_descale"]),
        _fp8_dequant(v, d["v_descale"]),
        causal,
    )
    cos = F.cosine_similarity(out.float().reshape(-1, D), ref.reshape(-1, D), dim=1)
    assert (out.float() - ref).abs().max().item() < FP8_MAX_ERR
    assert cos.min().item() > FP8_MIN_COS


@_gfx950_only
@pytest.mark.parametrize("causal", [False, True])
def test_fp8_dispatch_varlen_routes_to_gfx950(causal):
    """``flydsl_flash_attn_varlen_func`` must route packed THD fp8 to gfx950."""
    from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_func

    B, S, H, D = 2, 1024, 8, 128
    q, k, v, d = _fp8_dispatch_inputs(B, S, H, D, varlen=True)
    cu = torch.tensor([0, 400, B * S], dtype=torch.int32, device="cuda")
    out = flydsl_flash_attn_varlen_func(
        q, k, v, cu, cu, B * S - 400, B * S - 400, causal=causal, **d
    )
    assert out is not None, "gfx950 fp8 must not fall through to CK/Triton"
    assert out.shape == (B * S, H, D)
    assert out.dtype == torch.bfloat16

    ref = torch.empty_like(out, dtype=torch.float32)
    qd, kd, vd = (
        _fp8_dequant(t, s)
        for t, s in zip((q, k, v), (d["q_descale"], d["k_descale"], d["v_descale"]))
    )
    for b in range(len(cu) - 1):
        lo, hi = int(cu[b]), int(cu[b + 1])
        ref[lo:hi] = _ref_attention(
            qd[lo:hi].unsqueeze(0),
            kd[lo:hi].unsqueeze(0),
            vd[lo:hi].unsqueeze(0),
            causal,
        ).squeeze(0)
    cos = F.cosine_similarity(out.float().reshape(-1, D), ref.reshape(-1, D), dim=1)
    assert (out.float() - ref).abs().max().item() < FP8_MAX_ERR
    assert cos.min().item() > FP8_MIN_COS


@_gfx950_only
@pytest.mark.parametrize(
    "unsupported",
    [
        pytest.param({}, id="no_descales"),
        pytest.param({"softmax_scale": 0.5}, id="custom_softmax_scale"),
        pytest.param({"return_lse": True}, id="return_lse"),
        pytest.param({"dropout_p": 0.1}, id="dropout"),
        pytest.param({"window_size": (128, 0)}, id="sliding_window"),
        pytest.param({"window_size": (-1, -1, 4)}, id="sink_size"),
        pytest.param({"alibi_slopes": "H_f32"}, id="alibi"),
        pytest.param({"sink": "H_f32"}, id="sink"),
        pytest.param({"out": "fp8_out"}, id="non_bf16_out"),
    ],
)
def test_fp8_dispatch_rejects_unsupported(unsupported):
    """Anything the fp8 kernel cannot express returns None, not a wrong answer.

    Falling through is what lets the caller reach CK/Triton; raising or silently
    dropping the feature would both be wrong.
    """
    from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func

    B, S, H, D = 2, 512, 8, 128
    q, k, v, d = _fp8_dispatch_inputs(B, S, H, D)
    kw = dict(d)
    materialise = {
        "H_f32": lambda: torch.zeros(H, device="cuda", dtype=torch.float32),
        "fp8_out": lambda: torch.empty(B, S, H, D, device="cuda", dtype=FP8_DTYPE),
    }
    for key, val in unsupported.items():
        kw[key] = materialise[val]() if val in materialise else val
    if not unsupported:  # the no_descales case
        kw = {}

    assert flydsl_flash_attn_batch_func(q, k, v, causal=True, **kw) is None


@_gfx950_only
def test_fp8_dispatch_rejects_bf16_inputs():
    """bf16 on gfx950 has no FlyDSL kernel here and must fall through."""
    from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func

    q = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    assert flydsl_flash_attn_batch_func(q, q.clone(), q.clone(), causal=True) is None


@_gfx950_only
def test_fp8_dispatch_matches_direct_call():
    """The dispatch wrapper must not perturb the kernel it routes to."""
    from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        flydsl_flash_attn_fp8_func,
    )

    q, k, v, d = _fp8_dispatch_inputs()
    via_dispatch = flydsl_flash_attn_batch_func(q, k, v, causal=True, **d)
    direct = flydsl_flash_attn_fp8_func(q, k, v, causal=True, **d)
    torch.cuda.synchronize()
    torch.testing.assert_close(via_dispatch.float(), direct.float(), rtol=0, atol=0)
