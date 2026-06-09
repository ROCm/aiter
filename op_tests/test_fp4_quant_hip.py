# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + optional perf test for the HIP FP4 per-channel/kblock quantizer.

Run from the aiter repo root, for example:

  python3 op_tests/test_fp4_quant_hip.py
  python3 op_tests/test_fp4_quant_hip.py --perf --seq-len 131072 --iters 50
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable, Tuple

import torch

from aiter import quantize_fp4_e8m0_per_channel_kblock_hip


_FP4_MAX = 6.0


def _torch_ref_quantize_fp4_e8m0_per_channel_kblock(
    v: torch.Tensor, kblock_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bit-equivalent torch reference for the HIP kernel."""
    bh, t, d = v.shape
    grouped = v.float().reshape(bh, t // kblock_size, kblock_size, d)
    amax = grouped.abs().amax(dim=2).clamp_min(1e-30)
    exp = torch.ceil(torch.log2(amax / _FP4_MAX)).clamp(-127.0, 127.0)
    scale = torch.exp2(exp)
    scale_row = scale.repeat_interleave(kblock_size, dim=1)
    v_norm = (v.float() / scale_row).clamp(-_FP4_MAX, _FP4_MAX)

    ax = v_norm.abs()
    idx = (ax > 0.25).to(torch.uint8)
    idx += (ax > 0.75).to(torch.uint8)
    idx += (ax > 1.25).to(torch.uint8)
    idx += (ax > 1.75).to(torch.uint8)
    idx += (ax > 2.5).to(torch.uint8)
    idx += (ax > 3.5).to(torch.uint8)
    idx += (ax > 5.0).to(torch.uint8)
    sign = torch.where(v_norm < 0.0, 8, 0).to(torch.uint8)
    nibble = idx | sign

    lo = nibble[..., 0::2].to(torch.int32)
    hi = nibble[..., 1::2].to(torch.int32)
    packed = ((hi << 4) | lo).to(torch.uint8)
    scale_byte = (exp.to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
    return packed, scale_byte


def _maybe_triton_impl():
    """Import the local development Triton implementation when available."""
    dynq = Path("/home/dynamicquant")
    if dynq.is_dir() and str(dynq) not in sys.path:
        sys.path.insert(0, str(dynq))
    try:
        from block_quant import quantize_fp4_e8m0_per_channel_kblock_triton

        return quantize_fp4_e8m0_per_channel_kblock_triton
    except Exception:
        return None


def _bench(fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
           warmup: int,
           iters: int):
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return out, statistics.median(times), min(times), max(times)


def _check_equal(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    equal = torch.equal(got, ref)
    max_diff = int((got.to(torch.int16) - ref.to(torch.int16)).abs().max().item())
    print(f"{name}: equal={equal} max_diff={max_diff}")
    if not equal:
        raise AssertionError(f"{name} mismatch: max_diff={max_diff}")


def run_case(bh: int,
             seq_len: int,
             head_dim: int,
             kblock_size: int,
             perf: bool,
             warmup: int,
             iters: int) -> None:
    assert seq_len % kblock_size == 0
    assert head_dim % 2 == 0

    torch.manual_seed(0)
    v = torch.randn((bh, seq_len, head_dim), device="cuda", dtype=torch.bfloat16).contiguous()
    print(f"\nshape=(BH={bh}, T={seq_len}, D={head_dim}), kblock_size={kblock_size}")

    ref_packed, ref_scale = _torch_ref_quantize_fp4_e8m0_per_channel_kblock(v, kblock_size)
    hip_packed, hip_scale = quantize_fp4_e8m0_per_channel_kblock_hip(v, kblock_size)
    torch.cuda.synchronize()

    print("HIP vs torch reference")
    _check_equal("packed", hip_packed, ref_packed)
    _check_equal("scale ", hip_scale, ref_scale)

    triton_impl = _maybe_triton_impl()
    if triton_impl is not None:
        triton_packed, triton_scale = triton_impl(v, kblock_size)
        torch.cuda.synchronize()
        print("HIP vs Triton reference")
        _check_equal("packed", hip_packed, triton_packed)
        _check_equal("scale ", hip_scale, triton_scale)
    else:
        print("Triton reference skipped: /home/dynamicquant/block_quant.py not available")

    if not perf:
        return

    (_, _), hip_med, hip_min, hip_max = _bench(
        lambda: quantize_fp4_e8m0_per_channel_kblock_hip(v, kblock_size),
        warmup,
        iters,
    )
    print(
        "HIP quant: "
        f"median={hip_med:.4f} ms min={hip_min:.4f} ms max={hip_max:.4f} ms"
    )

    if triton_impl is not None:
        (_, _), tri_med, tri_min, tri_max = _bench(
            lambda: triton_impl(v, kblock_size),
            warmup,
            iters,
        )
        print(
            "Triton quant: "
            f"median={tri_med:.4f} ms min={tri_min:.4f} ms max={tri_max:.4f} ms"
        )
        print(f"speedup HIP/Triton: {tri_med / hip_med:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test AITER HIP FP4 per-channel/kblock quantization."
    )
    parser.add_argument("--bh", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kblock-size", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    run_case(
        bh=args.bh,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        kblock_size=args.kblock_size,
        perf=args.perf,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
