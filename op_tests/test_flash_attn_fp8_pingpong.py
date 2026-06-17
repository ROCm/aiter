#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the fp8 E4M3 flash-attention forward kernel.

Generates random Q,K,V (bf16), quantizes each per-tensor to fp8 E4M3,
runs the FlyDSL fp8 kernel, and compares against an fp32 SDPA reference
computed on the *dequantized* fp8 inputs (so the only error sources are
fp8 input rounding + fp8 P rounding + reduction order).

Run:
    PYTHONPATH=/data/work/flydsl:/data/work/aiter HIP_VISIBLE_DEVICES=0 \\
        python op_tests/test_flash_attn_fp8_pingpong.py
"""

import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_AITER_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

import flydsl.compiler as flyc  # noqa: E402,F401

from aiter.ops.flydsl.kernels.flash_attn_fp8_pingpong import (  # noqa: E402
    build_flash_attn_fp8_module,
)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def per_tensor_quant_fp8(x):
    """Per-tensor symmetric fp8 E4M3 quant.  Returns (q_fp8, descale)."""
    amax = x.abs().max().clamp(min=1e-8)
    descale = (amax / FP8_MAX).float()
    q = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, descale


def _as_i8(t):
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def run_shape(batch, seq_len, num_heads, head_dim=128, seed=0, verbose=True):
    device = "cuda"
    torch.manual_seed(seed)
    B, S, H, D = batch, seq_len, num_heads, head_dim

    q = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    k = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    v = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device).uniform_(-1, 1)

    q_fp8, q_descale = per_tensor_quant_fp8(q)
    k_fp8, k_descale = per_tensor_quant_fp8(k)
    v_fp8, v_descale = per_tensor_quant_fp8(v)

    # ---- Reference: fp32 SDPA on the dequantized fp8 inputs ----
    q_deq = q_fp8.float() * q_descale
    k_deq = k_fp8.float() * k_descale
    v_deq = v_fp8.float() * v_descale
    qf = q_deq.transpose(1, 2)  # B,H,S,D
    kf = k_deq.transpose(1, 2)
    vf = v_deq.transpose(1, 2)
    ref = F.scaled_dot_product_attention(qf, kf, vf, is_causal=False)
    ref = ref.transpose(1, 2).contiguous()  # B,S,H,D

    # ---- Kernel ----
    exe = build_flash_attn_fp8_module(num_heads=H, head_dim=D)

    q_flat = _as_i8(q_fp8).contiguous().view(-1)
    k_flat = _as_i8(k_fp8).contiguous().view(-1)
    v_flat = _as_i8(v_fp8).contiguous().view(-1)
    o_flat = torch.zeros(B * S * H * D, dtype=torch.bfloat16, device=device)

    # Call the JIT launcher directly (auto-compiles + caches); the explicit
    # flyc.compile() fast-dispatch path does not support scalar f32 args.
    exe(
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        float(q_descale),
        float(k_descale),
        float(v_descale),
        B,
        S,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    out = o_flat.view(B, S, H, D).float()
    ref_f = ref.float()

    cos = F.cosine_similarity(out.reshape(-1), ref_f.reshape(-1), dim=0).item()
    per_row = F.cosine_similarity(out.reshape(-1, D), ref_f.reshape(-1, D), dim=1)
    min_cos = per_row.min().item()
    max_err = (out - ref_f).abs().max().item()
    mean_err = (out - ref_f).abs().mean().item()

    if verbose:
        print(
            f"  [B={B} S={S} H={H} D={D}] cosine(flat)={cos:.6f} "
            f"min_cos(row)={min_cos:.6f} max_err={max_err:.4f} mean_err={mean_err:.5f}"
        )
        print(f"    out[0,0,0,:4]={out[0,0,0,:4].tolist()}")
        print(f"    ref[0,0,0,:4]={ref_f[0,0,0,:4].tolist()}")
    return cos, min_cos, max_err


def main():
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available")
        return 1

    shapes = [
        (1, 1, 256),
        (1, 1, 512),
    ]
    if os.getenv("FP8_ATTN_BIG", "1") == "1":
        shapes.append((1, 5, 4096))

    all_ok = True
    for B, H, S in shapes:
        cos, min_cos, max_err = run_shape(B, S, H)
        ok = cos >= 0.997
        all_ok = all_ok and ok
        print(f"  -> {'PASS' if ok else 'FAIL'} (cosine={cos:.6f} >= 0.997)")
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 2


def test_flash_attn_fp8_small():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA/ROCm not available")
    cos, _, _ = run_shape(1, 256, 1)
    assert cos >= 0.997, f"cosine {cos} < 0.997"


if __name__ == "__main__":
    sys.exit(main())
