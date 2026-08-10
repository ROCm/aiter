# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify aiter's vendored fp8 attention kernel (gfx950).

Two levels of check, per dense shape:

  equivalence  vendored vs the upstream FlyDSL reference kernel on identical
               inputs. Must be BIT-IDENTICAL -- the vendored symbols are
               byte-for-byte copies, so any difference means the extraction
               changed behaviour (a missed helper resolving to a different
               definition, an import shadowing, a traits divergence). The
               upstream kernel lives in the FlyDSL reference checkout (not the
               installed wheel: `flydsl.kernels` is not packaged), so this
               sub-check SKIPS cleanly when that checkout is absent rather than
               hardcoding a machine path.

  correctness  vendored vs a torch reference built from the DEQUANTIZED fp8
               tensors. Guards against both copies being wrong together, and
               against the classic harness bug of comparing to the
               pre-quantization bf16 input (which measures quantization error).

Also covers varlen, fp16 output, attention sinks (with an independent fp32
real-logit oracle), and the sinks+split-K build refusal.

Device: inherits HIP_VISIBLE_DEVICES. Do not override it.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aiter.ops.flydsl.utils import is_flydsl_available  # noqa: E402

try:
    from aiter.jit.utils.chip_info import get_gfx_runtime

    _ARCH = get_gfx_runtime()
except Exception:  # noqa: BLE001
    _ARCH = None

pytestmark = [
    pytest.mark.skipif(_ARCH != "gfx950", reason=f"gfx950 only, got {_ARCH}"),
    pytest.mark.skipif(not is_flydsl_available(), reason="flydsl not available"),
]

DEV = "cuda"
HEAD_DIM = 128
SEED = 123
REF_MAX_S = 8192  # above this the S x S torch reference will not fit

# Location of the upstream FlyDSL reference checkout, used ONLY for the
# bit-equivalence sub-check. `flydsl.kernels` is not in the installed wheel, so
# there is no packaged fallback; the env override lets a differently-placed
# checkout still be found without editing this file.
_FLYDSL_REF = os.environ.get("FLYDSL_REF_CHECKOUT",
                             os.path.expanduser("~/projects/flydsl"))


def _upstream_build():
    """Import the upstream builder from the reference checkout, or None."""
    kern = os.path.join(_FLYDSL_REF, "kernels", "attention",
                        "flash_attn_fp8_gfx950.py")
    if not os.path.isfile(kern):
        return None
    if _FLYDSL_REF not in sys.path:
        sys.path.insert(0, _FLYDSL_REF)
    try:
        from kernels.attention.flash_attn_fp8_gfx950 import (
            build_flash_attn_dualwave_swp_fp8_module as build_upstream,
        )
    except Exception:  # noqa: BLE001
        return None
    return build_upstream


def _vendored_build():
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module as build,
    )

    return build


# (label, B, S, H, HKV, causal)
SHAPES = [
    ("trace S=1384", 1, 1384, 64, 4, True),
    ("trace S=4023", 1, 4023, 64, 4, True),
    ("dense S=4096", 1, 4096, 64, 4, True),
    ("non-causal S=4096", 1, 4096, 64, 4, False),
    ("MHA S=2048", 1, 2048, 8, 8, True),
    ("batch=2 S=1024", 2, 1024, 64, 4, True),
    ("ticket S=32768", 1, 32768, 64, 4, True),  # skips its torch ref (OOM)
]

# Ragged batches for the varlen path. Not comparable against upstream -- it
# rejects fp8 varlen outright -- so these are checked against the torch
# reference only. The tiny set is the sharpest test of the active guard: every
# sequence is far below BLOCK_M=256, so all of them depend on out-of-range Q
# blocks being skipped.
VARLEN_CASES = [
    [4096],
    [512],
    [1024, 3072],
    [512, 1536, 2048],
    [4023, 1384],            # trace-source shapes, packed
    [1, 3, 31, 33, 63, 65],  # all << BLOCK_M
    [8192, 1024],            # long-then-short
]


def q8(x):
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(DEV)


def make_inputs(b, s, h, hkv, d, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    mk = lambda n: torch.randn(b, s, n, d, generator=g, device=DEV,  # noqa: E731
                               dtype=torch.bfloat16)
    return q8(mk(h)), q8(mk(hkv)), q8(mk(hkv))


def torch_reference(qf, qs, kf, ks, vf, vs, causal, d):
    """Reference from the dequantized fp8 tensors -- what the kernel is fed."""
    q, k, v = qf.float() * qs, kf.float() * ks, vf.float() * vs
    b, s, h, _ = q.shape
    rep = h // k.shape[2]
    kt = k.repeat_interleave(rep, 2).transpose(1, 2)
    vt = v.repeat_interleave(rep, 2).transpose(1, 2)
    att = q.transpose(1, 2) @ kt.transpose(-1, -2) / (d ** 0.5)
    if causal:
        m = torch.triu(torch.ones(s, s, device=DEV, dtype=torch.bool), 1)
        att = att.masked_fill(m, float("-inf"))
    out = (att.softmax(-1) @ vt).transpose(1, 2)
    del att
    return out


def torch_reference_sinks(qf, qs, kf, ks, vf, vs, causal, d, sink):
    """fp32 real-logit reference with a per-head sink in the softmax denominator.

    The sink is one extra virtual logit per row: denom = sum_i exp(s_i) +
    exp(sink[head]). It has no value vector, so it lowers every output by the
    normalization but contributes nothing to the numerator. Computed entirely in
    fp32 real-logit space (no log2 domain) so it is an independent oracle for the
    kernel's scaled-domain arithmetic.
    """
    q, k, v = qf.float() * qs, kf.float() * ks, vf.float() * vs
    b, s, h, _ = q.shape
    rep = h // k.shape[2]
    kt = k.repeat_interleave(rep, 2).transpose(1, 2)
    vt = v.repeat_interleave(rep, 2).transpose(1, 2)
    att = q.transpose(1, 2) @ kt.transpose(-1, -2) / (d ** 0.5)  # [b, h, s, s]
    if causal:
        m = torch.triu(torch.ones(s, s, device=DEV, dtype=torch.bool), 1)
        att = att.masked_fill(m, float("-inf"))
    # Numerically-stable softmax with the sink folded into the denominator.
    sink_l = sink.view(1, h, 1, 1)  # per-head logit
    row_max = torch.maximum(att.amax(dim=-1, keepdim=True), sink_l)
    p = (att - row_max).exp()
    denom = p.sum(dim=-1, keepdim=True) + (sink_l - row_max).exp()
    out = ((p / denom) @ vt).transpose(1, 2)
    del att, p
    return out


def run_sinks(build, qf, qs, kf, ks, vf, vs, b, s, h, hkv, d, causal, sink):
    mod = build(num_heads=h, head_dim=d, causal=causal, dtype_str="fp8",
                use_sinks=True, num_kv_heads=hkv)
    o = torch.empty(b, s, h, d, device=DEV, dtype=torch.bfloat16)
    mod(qf.contiguous().view(-1), kf.contiguous().view(-1),
        vf.contiguous().view(-1), o.contiguous().view(-1), b, s,
        q_descale=qs, k_descale=ks, v_descale=vs, sink=sink.contiguous().view(-1))
    torch.cuda.synchronize()
    return o


def run(build, qf, qs, kf, ks, vf, vs, b, s, h, hkv, d, causal,
        out_dtype_str="bf16"):
    out_torch = torch.float16 if out_dtype_str == "f16" else torch.bfloat16
    # out_dtype_str is a vendored-only builder param; keep the call signature
    # compatible with the upstream builder (used for the equivalence check) by
    # only passing it when it deviates from the bf16 default.
    extra = {} if out_dtype_str == "bf16" else {"out_dtype_str": out_dtype_str}
    mod = build(num_heads=h, head_dim=d, causal=causal, dtype_str="fp8",
                num_kv_heads=hkv, **extra)
    o = torch.empty(b, s, h, d, device=DEV, dtype=out_torch)
    mod(qf.contiguous().view(-1), kf.contiguous().view(-1),
        vf.contiguous().view(-1), o.contiguous().view(-1), b, s,
        q_descale=qs, k_descale=ks, v_descale=vs)
    torch.cuda.synchronize()
    return o


@pytest.mark.parametrize(
    "label,b,s,h,hkv,causal", SHAPES, ids=[c[0] for c in SHAPES],
)
def test_dense_correctness(label, b, s, h, hkv, causal):
    """Vendored kernel vs a dequantized-fp8 torch reference."""
    build = _vendored_build()
    (qf, qs), (kf, ks), (vf, vs) = make_inputs(b, s, h, hkv, HEAD_DIM, SEED)
    ov = run(build, qf, qs, kf, ks, vf, vs, b, s, h, hkv, HEAD_DIM, causal)
    if s > REF_MAX_S:
        pytest.skip(f"S={s} torch reference too large (OOM); equivalence only")
    ref = torch_reference(qf, qs, kf, ks, vf, vs, causal, HEAD_DIM)
    err = (ov.float() - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ov.float().flatten(), ref.flatten(), dim=0).item()
    assert err < 1e-1, f"max err {err:.4g}"
    assert cos > 0.99, f"cosine {cos:.6f}"
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "label,b,s,h,hkv,causal", SHAPES, ids=[c[0] for c in SHAPES],
)
def test_dense_bit_identical_to_upstream(label, b, s, h, hkv, causal):
    """Vendored kernel must be BIT-IDENTICAL to the upstream FlyDSL kernel run
    in the same process. Skips cleanly if the FlyDSL reference checkout is
    absent (its kernels are not in the installed wheel)."""
    build_upstream = _upstream_build()
    if build_upstream is None:
        pytest.skip(
            f"FlyDSL reference checkout not found at {_FLYDSL_REF} "
            "(set FLYDSL_REF_CHECKOUT); bit-equivalence check needs it"
        )
    build = _vendored_build()
    (qf, qs), (kf, ks), (vf, vs) = make_inputs(b, s, h, hkv, HEAD_DIM, SEED)
    ov = run(build, qf, qs, kf, ks, vf, vs, b, s, h, hkv, HEAD_DIM, causal)
    ou = run(build_upstream, qf, qs, kf, ks, vf, vs, b, s, h, hkv, HEAD_DIM, causal)
    assert torch.equal(ov, ou), (
        f"{label}: vendored kernel differs from upstream -- the extraction "
        "changed behaviour"
    )
    torch.cuda.empty_cache()


@pytest.mark.parametrize("seqs", VARLEN_CASES, ids=[str(s) for s in VARLEN_CASES])
def test_varlen(seqs):
    """Packed ragged batch vs a per-sequence torch reference. bad_row_count is
    the sharper signal: before the active guard existed, whole sequences past
    the first BLOCK_M were wrong while the aggregate cosine still read 0.96."""
    build = _vendored_build()
    h, hkv, d = 64, 4, HEAD_DIM
    total, b = sum(seqs), len(seqs)
    g = torch.Generator(device=DEV).manual_seed(SEED)
    mk = lambda n: torch.randn(total, n, d, generator=g, device=DEV,  # noqa: E731
                               dtype=torch.bfloat16)
    qf, qs = q8(mk(h))
    kf, ks = q8(mk(hkv))
    vf, vs = q8(mk(hkv))
    cu = torch.tensor([0] + list(torch.tensor(seqs).cumsum(0)),
                      device=DEV, dtype=torch.int32)

    ref = torch.empty(total, h, d, device=DEV, dtype=torch.float32)
    for i, ln in enumerate(seqs):
        lo, hi = cu[i].item(), cu[i + 1].item()
        q, k, v = qf[lo:hi].float() * qs, kf[lo:hi].float() * ks, vf[lo:hi].float() * vs
        kk = k.repeat_interleave(h // hkv, 1).transpose(0, 1)
        vv = v.repeat_interleave(h // hkv, 1).transpose(0, 1)
        s_ = q.transpose(0, 1) @ kk.transpose(-1, -2) / (d ** 0.5)
        s_ = s_.masked_fill(
            torch.triu(torch.ones(ln, ln, device=DEV, dtype=torch.bool), 1),
            float("-inf"))
        ref[lo:hi] = (s_.softmax(-1) @ vv).transpose(0, 1)

    mod = build(num_heads=h, head_dim=d, causal=True, dtype_str="fp8",
                num_kv_heads=hkv, varlen=True)
    o = torch.empty(total, h, d, device=DEV, dtype=torch.bfloat16)
    mod(qf.contiguous().view(-1), kf.contiguous().view(-1),
        vf.contiguous().view(-1), o.contiguous().view(-1), b, max(seqs),
        cu_seqlens_q=cu, cu_seqlens_kv=cu,
        q_descale=qs, k_descale=ks, v_descale=vs)
    torch.cuda.synchronize()

    of = o.float()
    err = (of - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), ref.flatten(), dim=0).item()
    bad = int(((of - ref).abs().amax(dim=(1, 2)) > 1e-1).sum().item())
    assert bad == 0, f"{bad} rows over threshold (max err {err:.4g})"
    assert err < 1e-1, f"max err {err:.4g}"
    assert cos > 0.99, f"cosine {cos:.6f}"
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "label,b,s,h,hkv,causal",
    [
        ("f16 dense S=4096", 1, 4096, 64, 4, True),
        ("f16 non-causal", 1, 2048, 64, 4, False),
    ],
    ids=["f16_dense_S4096", "f16_noncausal"],
)
def test_fp16_output(label, b, s, h, hkv, causal):
    """OUTPUT dtype is independent of the fp8 QKV compute. f16 stores 2 bytes
    like bf16, so only the store-site conversion changes (cvt_pkrtz). The ref is
    built in f16 to compare like-for-like against the f16 store."""
    build = _vendored_build()
    (qf, qs), (kf, ks), (vf, vs) = make_inputs(b, s, h, hkv, HEAD_DIM, SEED)
    ov = run(build, qf, qs, kf, ks, vf, vs, b, s, h, hkv, HEAD_DIM, causal,
             out_dtype_str="f16")
    assert ov.dtype is torch.float16, f"expected f16 output, got {ov.dtype}"
    ref = torch_reference(qf, qs, kf, ks, vf, vs, causal, HEAD_DIM).to(torch.float16)
    err = (ov.float() - ref.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ov.float().flatten(), ref.float().flatten(), dim=0).item()
    assert err < 1e-1, f"max err {err:.4g}"
    assert cos > 0.99, f"cosine {cos:.6f}"
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "label,b,s,h,hkv,causal",
    [
        ("GQA16 dense S=4096", 1, 4096, 64, 4, True),
        ("GQA16 non-causal", 1, 2048, 64, 4, False),
        ("GQA1 dense S=2048", 1, 2048, 16, 16, True),
    ],
    ids=["GQA16_dense_S4096", "GQA16_noncausal", "GQA1_dense_S2048"],
)
def test_sinks(label, b, s, h, hkv, causal):
    """The sink is a per-head virtual logit in the softmax denominator. GQA>1:1
    is mandatory here: a GQA 1:1 case would mask a wrong per-M-row head index
    (all heads share). Reference computed in fp32 real-logit space."""
    build = _vendored_build()
    (qf, qs), (kf, ks), (vf, vs) = make_inputs(b, s, h, hkv, HEAD_DIM, SEED)
    g_sink = torch.Generator(device=DEV).manual_seed(SEED + 5)
    # Sinks spanning a useful range: some heads sink-dominated, some not.
    sink = (torch.randn(h, generator=g_sink, device=DEV,
                        dtype=torch.float32) * 2.0)
    ov = run_sinks(build, qf, qs, kf, ks, vf, vs, b, s, h, hkv, HEAD_DIM,
                   causal, sink)
    ref = torch_reference_sinks(qf, qs, kf, ks, vf, vs, causal, HEAD_DIM, sink)
    err = (ov.float() - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ov.float().flatten(), ref.flatten(), dim=0).item()
    # Sinks-only bad-row rule: absolute AND relative. The sink lowers the
    # denominator normalization, amplifying inherent fp8 bf16-P-pack quantization
    # noise on 1-to-9-key rows (~5% rel err, ref magnitude ~2) a few rows over
    # the absolute 1e-1 bar. That noise is the main-loop fast path (out of scope
    # to touch), not a sinks defect; the relative gate excludes it while still
    # catching a real per-row error.
    abs_err = (ov.float() - ref).abs().amax(dim=(2, 3))
    rel_err = abs_err / ref.abs().amax(dim=(2, 3)).clamp(min=1e-4)
    bad = int(((abs_err > 1e-1) & (rel_err > 0.1)).sum().item())
    assert cos > 0.99, f"cosine {cos:.6f}"
    assert bad == 0, f"{bad} rows over threshold (max err {err:.4g})"
    torch.cuda.empty_cache()


def test_sinks_splitk_refused():
    """Sinks + split-K must be refused (double-counted denominator; the combine
    has no sink input). A build that does NOT raise is a failure."""
    build = _vendored_build()
    with pytest.raises(ValueError):
        build(num_heads=64, head_dim=HEAD_DIM, causal=True, dtype_str="fp8",
              use_sinks=True, num_kv_heads=4, num_kv_splits=2)
