# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Microbench + correctness for fused_qk_norm_rope_2way_fp8_perhead_quant.

Compares two paths on Qwen-Image (batch=1, num_tokens0=512 txt,
num_tokens1=4096 img, Hq=Hk=24, head_size=128, bf16):
  1. fused_qk_norm_rope_2way + per_tensor_quant x2 (baseline)
  2. fused_qk_norm_rope_2way_fp8_perhead_quant   (fused, per-(batch,head))
"""

import os
import sys

import torch
from torch import Tensor

# Make sure we hit the in-tree aiter (where the new op lives), not /opt/aiter.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import aiter  # noqa: E402
from aiter.test_common import perftest  # noqa: E402


@perftest()
def run_baseline_split(
    q0,
    k0,
    q1,
    k1,
    w_q0,
    w_k0,
    w_q1,
    w_k1,
    cos_sin0,
    cos_sin1,
    B,
    T0,
    T1,
    Hq,
    Hk,
    D,
    is_interleaved,
    eps,
):
    q01 = torch.empty((B, T0 + T1, Hq, D), dtype=q0.dtype, device=q0.device)
    k01 = torch.empty((B, T0 + T1, Hk, D), dtype=k0.dtype, device=k0.device)
    aiter.fused_qk_norm_rope_2way(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        B,
        T0,
        T1,
        Hq,
        Hk,
        D,
        is_interleaved,
        eps,
        q01,
        k01,
    )
    q_fp8, q_scale = aiter.per_tensor_quant(q01, quant_dtype=torch.float8_e4m3fnuz)
    k_fp8, k_scale = aiter.per_tensor_quant(k01, quant_dtype=torch.float8_e4m3fnuz)
    return q_fp8, k_fp8, q_scale, k_scale, q01, k01


@perftest()
def run_fused_perhead(
    q0,
    k0,
    q1,
    k1,
    w_q0,
    w_k0,
    w_q1,
    w_k1,
    cos_sin0,
    cos_sin1,
    B,
    T0,
    T1,
    Hq,
    Hk,
    D,
    is_interleaved,
    eps,
):
    # Pass explicit out_q01/out_k01 so correctness can compare the bf16
    # intermediate against the existing fused rope+rms path.
    out_q01 = torch.empty((B, T0 + T1, Hq, D), dtype=q0.dtype, device=q0.device)
    out_k01 = torch.empty((B, T0 + T1, Hk, D), dtype=k0.dtype, device=k0.device)
    return aiter.fused_qk_norm_rope_2way_fp8_perhead_quant(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        B,
        T0,
        T1,
        Hq,
        Hk,
        D,
        is_interleaved,
        eps,
        out_q01,
        out_k01,
    )


def max_rel_err(ref: Tensor, got: Tensor) -> float:
    ref = ref.to(torch.float32)
    got = got.to(torch.float32)
    denom = ref.abs().clamp(min=1e-3)
    return ((ref - got).abs() / denom).max().item()


def main():
    torch.manual_seed(0)
    B, T0, T1, Hq, Hk, D = 1, 512, 4096, 24, 24, 128
    is_interleaved = (
        False  # Qwen Image uses is_neox_style = True => is_interleaved=False
    )
    eps = 1e-6
    dtype = torch.bfloat16
    dev = "cuda"

    def rn(*shape):
        return torch.randn(*shape, dtype=dtype, device=dev)

    q0 = rn(B, T0, Hq, D)
    k0 = rn(B, T0, Hk, D)
    q1 = rn(B, T1, Hq, D)
    k1 = rn(B, T1, Hk, D)
    w_q0 = rn(D)
    w_k0 = rn(D)
    w_q1 = rn(D)
    w_k1 = rn(D)
    cos_sin0 = rn(T0, D)
    cos_sin1 = rn(T1, D)

    (_, _, qb_scale, kb_scale, qb_bf16, kb_bf16), t_base = run_baseline_split(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        B,
        T0,
        T1,
        Hq,
        Hk,
        D,
        is_interleaved,
        eps,
    )
    (qh_fp8, kh_fp8, qh_scale, kh_scale, qh_bf16, kh_bf16), t_ph = run_fused_perhead(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        B,
        T0,
        T1,
        Hq,
        Hk,
        D,
        is_interleaved,
        eps,
    )

    # Microbench V quant: baseline vs 2-way per-(batch, head).
    @perftest()
    def run_v_per_tensor(v):
        return aiter.per_tensor_quant(v, quant_dtype=torch.float8_e4m3fnuz)

    @perftest()
    def run_v_2way_perhead(v0, v1):
        return aiter.v_2way_per_head_fp8_quant(v0, v1)

    v0 = torch.randn(B, T0, Hq, D, dtype=dtype, device=dev).contiguous()
    v1 = torch.randn(B, T1, Hq, D, dtype=dtype, device=dev).contiguous()
    v = torch.cat([v0, v1], dim=1).contiguous()
    (_v_pt, _v_pt_s), t_vpt = run_v_per_tensor(v)
    (_v_ph, _v_ph_s), t_vph = run_v_2way_perhead(v0, v1)

    # Dequantize and compare against the bf16 reference (qb_bf16/kb_bf16 from the
    # already-correct baseline split pipeline).
    q_ref = qb_bf16.to(torch.float32)
    k_ref = kb_bf16.to(torch.float32)

    # Per-head dequant: scale shape (B, Hq) -> broadcast across (T, D)
    q_ph = qh_fp8.to(torch.float32) * qh_scale.view(B, 1, Hq, 1).to(torch.float32)
    k_ph = kh_fp8.to(torch.float32) * kh_scale.view(B, 1, Hk, 1).to(torch.float32)

    # Also compare unquantized bf16 outputs between baseline and fused paths -
    # these should be byte-identical modulo bf16 rounding.
    bf16_diff_ph_q = (
        (qb_bf16.to(torch.float32) - qh_bf16.to(torch.float32)).abs().max().item()
    )
    bf16_diff_ph_k = (
        (kb_bf16.to(torch.float32) - kh_bf16.to(torch.float32)).abs().max().item()
    )

    print(
        f"\n[shapes] q_descale baseline={tuple(qb_scale.shape)} perhead={tuple(qh_scale.shape)}"
        f"  k_descale baseline={tuple(kb_scale.shape)} perhead={tuple(kh_scale.shape)}"
    )

    print("\n[bf16 sanity] max |baseline - fused| in unquantized bf16 outputs:")
    print(f"  perhead  : q={bf16_diff_ph_q:.3e}  k={bf16_diff_ph_k:.3e}")

    print("\n[fp8 max rel err vs bf16 reference]")
    print(
        f"  perhead: q={max_rel_err(q_ref, q_ph):.3%}  k={max_rel_err(k_ref, k_ph):.3%}"
    )

    print("\n[perf — fused rope+rms+fp8 quant for Q/K only]")
    print(f"  baseline (rope+per_tensor x2)       : {t_base:8.2f} us")
    print(f"  fused perhead                       : {t_ph:8.2f} us")
    print(f"  perhead vs baseline                 : {(t_ph/t_base - 1)*100:+.2f}%")

    print("\n[perf — V quant only]")
    print(f"  per_tensor_quant (V)                : {t_vpt:8.2f} us")
    print(f"  v_2way_per_head_fp8_quant           : {t_vph:8.2f} us")
    print(
        f"  2way perhead vs pertensor           : {(t_vph/max(t_vpt,1e-9) - 1)*100:+.2f}%"
    )


if __name__ == "__main__":
    main()
