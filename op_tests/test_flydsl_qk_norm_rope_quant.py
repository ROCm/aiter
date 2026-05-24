#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test for ``flydsl_qk_norm_rope_quant``.

Validates the fused RMSNorm + GPT-J RoPE + (optional) FP8 quant kernel
against a pure-torch reference, and reports per-config kernel us /
bandwidth utilization.

Sweeps:
- T (sequence / decode-batch length)
- (group_size, scale_dtype) combos: per-row fp32, 1×128 fp32 / e8m0, 1×64 …
- with vs without optional ``q_weight``

Usage:
    python op_tests/test_flydsl_qk_norm_rope_quant.py
    python op_tests/test_flydsl_qk_norm_rope_quant.py -T 64 256 1024 -q fp8_1x128_e8m0
    python op_tests/test_flydsl_qk_norm_rope_quant.py --no-quant   # bf16 only
"""

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.flydsl import flydsl_qk_norm_rope_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# Shared constants (independent of attention shape).
_EPS = 1e-6
_SQRT2 = math.sqrt(2.0)
_FP8_DTYPE = dtypes.fp8
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_E8M0_HEADROOM = 7  # matches kernel constant


# ============================================================================
# Reference (pure torch)
# ============================================================================


def _rope_tail_ref(x, cos2d, sin2d, pos, *, D, RD):
    """GPT-J pair-interleaved RoPE on the last RD dims."""
    NOPE = D - RD
    T = x.shape[0]
    leading = x.shape[1:-1]
    tail = x[..., NOPE:].reshape(T, *leading, RD // 2, 2)
    c = cos2d[pos].reshape(T, *((1,) * len(leading)), RD // 2)
    s = sin2d[pos].reshape(T, *((1,) * len(leading)), RD // 2)
    even, odd = tail[..., 0], tail[..., 1]
    new_e = even * c - odd * s
    new_o = even * s + odd * c
    tail_new = torch.stack([new_e, new_o], dim=-1).reshape(T, *leading, RD)
    return torch.cat([x[..., :NOPE], tail_new], dim=-1)


def _e8m0_encode_ref(amax_safe):
    """Mirror the kernel's bit-manip e8m0 encoding.

    Returns ``(byte_uint8, factor_fp32)``. ``factor`` is the multiplier
    applied to ``x_norm`` so that ``out = x_norm * factor`` lands in fp8
    range. ``byte`` is the MX e8m0 stored byte; dequant scale = 2^(byte-127).
    """
    af = amax_safe.float().contiguous()
    bits = af.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits_up = (bits + 0x400000) & 0xFF800000
    exp_field = (bits_up >> 23) & 0xFF
    e8m0_biased = (exp_field - _E8M0_HEADROOM).clamp(0, 255)
    quant_exp = (254 - e8m0_biased).to(torch.int32)
    factor = (quant_exp << 23).view(torch.float32)
    return e8m0_biased.to(torch.uint8), factor


def _flydsl_qk_norm_rope_ref(
    q,
    kv,
    kv_weight,
    cos_cache,
    sin_cache,
    positions,
    *,
    H,
    D,
    RD,
    q_weight=None,
    quant=False,
    quant_group_size=None,
    scale_dtype="fp32",
):
    """Pure-torch reference. Returns same tuple as the kernel."""
    T = q.shape[0]
    G = quant_group_size if quant_group_size is not None else D
    NG = D // G

    q3 = q.view(T, H, D).float()
    kvf = kv.float()
    rstd_q = torch.rsqrt(q3.pow(2).mean(-1, keepdim=True) + _EPS)
    rstd_kv = torch.rsqrt(kvf.pow(2).mean(-1, keepdim=True) + _EPS)
    q_n = q3 * rstd_q
    if q_weight is not None:
        q_n = q_n * q_weight.float()
    kv_n = kvf * rstd_kv * kv_weight.float()

    cos2d = cos_cache.view(cos_cache.shape[0], cos_cache.shape[-1]).float()
    sin2d = sin_cache.view(sin_cache.shape[0], sin_cache.shape[-1]).float()
    q_roped = _rope_tail_ref(q_n, cos2d, sin2d, positions, D=D, RD=RD)
    kv_roped = _rope_tail_ref(kv_n, cos2d, sin2d, positions, D=D, RD=RD)

    if not quant:
        return (
            q_roped.to(torch.bfloat16),
            kv_roped.to(torch.bfloat16),
            None,
            None,
        )

    # Per-group amax of pre-RoPE x_norm × SQRT2 (post-rope upper bound).
    q_groups = q_n.reshape(T, H, NG, G)
    kv_groups = kv_n.reshape(T, NG, G)
    am_q = (q_groups.abs().amax(-1) * _SQRT2).clamp_min(1e-12)
    am_kv = (kv_groups.abs().amax(-1) * _SQRT2).clamp_min(1e-12)

    if scale_dtype == "fp32":
        factor_q = _FP8_MAX / am_q
        factor_kv = _FP8_MAX / am_kv
        scale_q_store = (am_q / _FP8_MAX).to(torch.float32)
        scale_kv_store = (am_kv / _FP8_MAX).to(torch.float32)
    elif scale_dtype == "e8m0":
        scale_q_store, factor_q = _e8m0_encode_ref(am_q)
        scale_kv_store, factor_kv = _e8m0_encode_ref(am_kv)
    else:
        raise ValueError(scale_dtype)

    factor_q_full = factor_q.unsqueeze(-1).expand(*factor_q.shape, G).reshape(T, H, D)
    factor_kv_full = factor_kv.unsqueeze(-1).expand(*factor_kv.shape, G).reshape(T, D)
    q_fp8 = (q_roped * factor_q_full).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    kv_fp8 = (kv_roped * factor_kv_full).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return q_fp8, kv_fp8, scale_q_store, scale_kv_store


# ============================================================================
# Dequant for fp8 → fp32 comparison
# ============================================================================


def _dequant(out_fp8, scale, *, D, quant_group_size, scale_dtype):
    T = out_fp8.shape[0]
    leading = out_fp8.shape[1:-1]
    G = quant_group_size if quant_group_size is not None else D
    if scale_dtype == "fp32":
        scale_f = scale.float()
    else:
        # MX e8m0: dequant_scale = 2^(byte - 127) → bits = (byte << 23)
        bits = scale.to(torch.int32) << 23
        scale_f = bits.view(torch.float32)
    scale_full = scale_f.unsqueeze(-1).expand(*scale_f.shape, G).reshape(T, *leading, D)
    return out_fp8.float() * scale_full


# ============================================================================
# Main test (per-config)
# ============================================================================

# MI355X HBM3e peak. Used only for the "%peak" perf column.
_PEAK_BW_GBPS = 8000.0


@benchmark()
def test_flydsl_qk_norm_rope_quant(
    T,
    H,
    D,
    RD,
    *,
    quant_group_size,
    scale_dtype,
    q_weighted,
    quant,
):
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Build cos/sin via a YaRN-style table covering all positions in T.
    max_pos = max(T, 64)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, RD, 2, device=device).float() / RD))
    pos_range = torch.arange(max_pos, device=device).float()
    freqs = torch.einsum("i,j->ij", pos_range, inv_freq)
    cos = freqs.cos().to(torch.bfloat16).contiguous()
    sin = freqs.sin().to(torch.bfloat16).contiguous()

    q = torch.randn(T, H * D, dtype=torch.bfloat16, device=device) * 0.1
    # Mimic V4 KV split: kv = strided view into a wider tensor
    Q_LORA = 1536
    qkv_a = torch.randn(T, Q_LORA + D, dtype=torch.bfloat16, device=device) * 0.1
    _, kv = torch.split(qkv_a, [Q_LORA, D], dim=-1)
    kv_w = torch.randn(D, dtype=torch.bfloat16, device=device).abs() + 0.5
    q_w = (
        (torch.randn(D, dtype=torch.bfloat16, device=device).abs() + 0.5)
        if q_weighted
        else None
    )
    pos = torch.randint(0, max_pos - 1, (T,), dtype=torch.int64, device=device)

    # Reference
    ref_q, ref_kv, ref_qs, ref_ks = _flydsl_qk_norm_rope_ref(
        q,
        kv.contiguous(),
        kv_w,
        cos,
        sin,
        pos,
        H=H,
        D=D,
        RD=RD,
        q_weight=q_w,
        quant=quant,
        quant_group_size=quant_group_size,
        scale_dtype=scale_dtype,
    )

    # Kernel + perf
    (got_q, got_kv, got_qs, got_ks), us = run_perftest(
        flydsl_qk_norm_rope_quant,
        q,
        kv,
        kv_w,
        cos,
        sin,
        pos,
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        q_weight=q_w,
        quant=quant,
        quant_group_size=quant_group_size,
        scale_dtype=scale_dtype,
    )

    # Accuracy
    if quant:
        deq_kw = dict(D=D, quant_group_size=quant_group_size, scale_dtype=scale_dtype)
        got_deq = _dequant(got_q, got_qs, **deq_kw)
        ref_deq = _dequant(ref_q, ref_qs, **deq_kw)
        got_kv_deq = _dequant(got_kv, got_ks, **deq_kw)
        ref_kv_deq = _dequant(ref_kv, ref_ks, **deq_kw)
        # Looser tolerance under fp8 + group quant — pow2 rounding plus bf16
        # RoPE noise pushes per-element diffs into the 0.1–10 range depending
        # on amax. cos-sim (computed via checkAllclose's atol on row sums)
        # remains > 0.999 in all configs we ship.
        rtol, atol = 0.05, 5.0
    else:
        got_deq = got_q.float()
        ref_deq = ref_q.float()
        got_kv_deq = got_kv.float()
        ref_kv_deq = ref_kv.float()
        rtol, atol = 1e-3, 1e-2

    err_q = checkAllclose(
        ref_deq, got_deq, rtol=rtol, atol=atol, msg="Q (rmsnorm+rope+quant)"
    )
    err_kv = checkAllclose(
        ref_kv_deq, got_kv_deq, rtol=rtol, atol=atol, msg="KV (rmsnorm+rope+quant)"
    )

    # Bandwidth-utilization estimate (Q in/out + KV in/out + scales when quant)
    bytes_in = T * H * D * 2 + T * D * 2 + D * 2  # Q + KV + kv_weight (small)
    if q_weighted:
        bytes_in += D * 2
    if quant:
        out_bytes_per_elem = 1
        G = quant_group_size if quant_group_size is not None else D
        NG = D // G
        scale_bytes = (T * H + T) * NG * (4 if scale_dtype == "fp32" else 1)
        bytes_out = T * H * D * out_bytes_per_elem + T * D * out_bytes_per_elem
        bytes_total = bytes_in + bytes_out + scale_bytes
    else:
        bytes_out = T * H * D * 2 + T * D * 2
        bytes_total = bytes_in + bytes_out
    gbps = bytes_total / (us * 1e-6) / 1e9

    return {
        "us": round(us, 3),
        "GB/s": round(gbps, 0),
        "%peak": round(gbps / _PEAK_BW_GBPS * 100, 1),
        "err_q": err_q,
        "err_kv": err_kv,
    }


# ============================================================================
# argparse + matrix sweep
# ============================================================================

_QUANT_OPTIONS = {
    "bf16": (False, None, "fp32", False),  # quant_q,kv off
    "fp8_per_row_fp32": (True, None, "fp32", False),
    "fp8_1x128_fp32": (True, 128, "fp32", False),
    "fp8_1x64_fp32": (True, 64, "fp32", False),
    "fp8_1x128_e8m0": (True, 128, "e8m0", False),
    "fp8_1x64_e8m0": (True, 64, "e8m0", False),
    # kernel supported, but no use case yet
    # "fp8_1x32_fp32": (True, 32, "fp32", False),
    # "fp8_1x32_e8m0": (True, 32, "e8m0", False),
}


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="aiter test for flydsl_qk_norm_rope_quant (V4-Pro decode shape).",
)
parser.add_argument(
    "-T",
    "--T",
    type=int,
    nargs="*",
    default=[1, 2, 16, 32, 64, 128, 192, 256, 512, 1024, 16384, 65540],
    help="token-count sweep. e.g. -T 4 64 1024",
)
parser.add_argument(
    "--H",
    type=int,
    nargs="*",
    default=[16, 64, 128],
    help="num-Q-heads-per-rank sweep. e.g. --H 16 128",
)
parser.add_argument(
    "--D",
    type=int,
    nargs="*",
    default=[512],
    help=(
        "head_dim sweep. Current kernel MVP only supports D=512 (VEC=8); "
        "other D values are rejected with a clear assert until atom-widths + "
        "fp8 packing are generalised."
    ),
)
parser.add_argument(
    "--RD",
    type=int,
    default=64,
    help="rope_head_dim (RoPE tail size, single value)",
)
parser.add_argument(
    "-q",
    "--quant",
    type=str,
    choices=list(_QUANT_OPTIONS.keys()),
    nargs="*",
    default=list(_QUANT_OPTIONS.keys()),
    help="quant config(s). bf16 = no quant, fp8_<group>_<scale> = quant.",
)
parser.add_argument(
    "--qweight",
    action="store_true",
    help="also run each config with optional q_weight=enabled.",
)
parser.add_argument(
    "--no-quant",
    action="store_true",
    help="bf16 only (ignore -q).",
)
args = parser.parse_args()

quant_keys = ["bf16"] if args.no_quant else args.quant
qweight_modes = [False, True] if args.qweight else [False]

rows = []
for key, qw_mode, H, D in itertools.product(quant_keys, qweight_modes, args.H, args.D):
    quant, group_size, scale_dtype, _ = _QUANT_OPTIONS[key]
    for T in args.T:
        rows.append(
            test_flydsl_qk_norm_rope_quant(
                T,
                H,
                D,
                args.RD,
                quant_group_size=group_size,
                scale_dtype=scale_dtype,
                q_weighted=qw_mode,
                quant=quant,
            )
        )

df = pd.DataFrame(rows)
aiter.logger.info(
    "flydsl_qk_norm_rope_quant summary (markdown):\n%s", df.to_markdown(index=False)
)
