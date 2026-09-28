# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Validate the optional fp8 Q output of fused_qk_norm_rope_cache_pts_quant_shuffle.

When ``q_out_fp8`` + ``per_tensor_q_scale`` are supplied, the kernel fp8-quantizes
the rotated Q in the same epilogue that writes fp8 K/V (mirroring per_tensor_k/v
scale), so callers can drop a separate scaled_quant launch. bf16 ``q_out`` is still
written for callers that consume it.

Run:
  CUDA_VISIBLE_DEVICES=0 python3 op_tests/test_fused_qk_norm_rope_cache_q_fp8.py
"""
import torch

import aiter
from aiter import get_dtype_fp8

DEV = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-6


def _run(head_size, q_scale_val, with_qfp8):
    cache_dtype = get_dtype_fp8()  # e4m3fnuz on gfx942
    nt, hq, hk, hv = 1, 4, 1, 1
    block_size = 1
    torch.manual_seed(0)
    qkv = torch.randn(nt, (hq + hk + hv) * head_size, dtype=DTYPE, device=DEV)
    qw = torch.randn(head_size, dtype=DTYPE, device=DEV)
    kw = torch.randn(head_size, dtype=DTYPE, device=DEV)
    cos_sin = torch.randn(8, head_size, dtype=DTYPE, device=DEV)
    positions = torch.zeros(nt, dtype=torch.int64, device=DEV)
    q_out = torch.empty(nt, hq * head_size, dtype=DTYPE, device=DEV)
    k_cache = torch.zeros(1, block_size, hk, head_size, dtype=cache_dtype, device=DEV)
    v_cache = torch.zeros(1, block_size, hv, head_size, dtype=cache_dtype, device=DEV)
    slot_mapping = torch.zeros(nt, dtype=torch.int64, device=DEV)
    k_scale = torch.ones(1, dtype=torch.float32, device=DEV)
    v_scale = torch.ones(1, dtype=torch.float32, device=DEV)
    x = 16 // k_cache.element_size()

    q_out_fp8 = q_scale = None
    if with_qfp8:
        q_out_fp8 = torch.zeros(nt, hq * head_size, dtype=cache_dtype, device=DEV)
        q_scale = torch.full((1,), q_scale_val, dtype=torch.float32, device=DEV)

    aiter.fused_qk_norm_rope_cache_pts_quant_shuffle(
        qkv.clone(), qw, kw, cos_sin, positions, nt, hq, hk, hv, head_size,
        True, EPS, q_out, k_cache, v_cache, slot_mapping, k_scale, v_scale,
        None, None, False, False, block_size, x, 0, False, q_out_fp8, q_scale,
    )
    return q_out, q_out_fp8, cache_dtype


def main():
    ok = True
    for head_size in (128, 256):
        for q_scale_val in (1.0, 0.5):
            q_ref, _, _ = _run(head_size, q_scale_val, with_qfp8=False)
            q_bf16, q_fp8, cache_dtype = _run(head_size, q_scale_val, with_qfp8=True)
            # bf16 q_out is byte-identical whether or not fp8 Q is requested.
            same = torch.equal(q_ref, q_bf16)
            # Kernel fp8 Q must equal a torch fp8 quant of the same bf16 Q by the
            # same per-tensor scale: out = fp8(q / scale).
            expected = (q_bf16.float() / q_scale_val).to(cache_dtype)
            exact = (q_fp8.float() == expected.float()).float().mean().item()
            passed = same and exact > 0.98
            ok = ok and passed
            print(
                f"D={head_size} qscale={q_scale_val} bf16_unchanged={same} "
                f"fp8_exact_frac={exact:.4f} -> {'PASS' if passed else 'FAIL'}"
            )
    assert ok, "fp8 Q output validation failed"
    print("ALL PASS")


if __name__ == "__main__":
    main()
