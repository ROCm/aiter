# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Standalone perf harness for the FlyDSL fp8 attention kernel.

Reports latency + TFLOPS using bench_sage's FLOPS convention
``2 * b * h * sq * sk * (d + dv)`` so the number is comparable to the ASM
``aiter_fp8`` kernel. Run::

    PYTHONPATH=/data/work/flydsl:/data/work/aiter HIP_VISIBLE_DEVICES=0 \
        python op_tests/bench_flash_attn_fp8_pingpong.py --b 1 --hq 5 --sq 75520 --d 128
"""

import argparse
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from aiter.ops.flydsl.kernels.flash_attn_fp8_pingpong import (  # noqa: E402
    build_flash_attn_fp8_module,
)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def per_tensor_quant_fp8(x):
    amax = x.abs().max().clamp(min=1e-8)
    descale = (amax / FP8_MAX).float()
    q = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, descale


def _as_i8(t):
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def bench(batch, seq_len, num_heads, head_dim=128, iters=50, warmup=10, seed=0):
    device = "cuda"
    torch.manual_seed(seed)
    B, S, H, D = batch, seq_len, num_heads, head_dim

    q = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    k = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    v = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    q_fp8, q_descale = per_tensor_quant_fp8(q)
    k_fp8, k_descale = per_tensor_quant_fp8(k)
    v_fp8, v_descale = per_tensor_quant_fp8(v)

    exe = build_flash_attn_fp8_module(num_heads=H, head_dim=D)
    q_flat = _as_i8(q_fp8).contiguous().view(-1)
    k_flat = _as_i8(k_fp8).contiguous().view(-1)
    v_flat = _as_i8(v_fp8).contiguous().view(-1)
    o_flat = torch.zeros(B * S * H * D, dtype=torch.bfloat16, device=device)

    stream = torch.cuda.current_stream()
    args = (
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        float(q_descale),
        float(k_descale),
        float(v_descale),
        B,
        S,
        stream,
    )

    for _ in range(warmup):
        exe(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        exe(*args)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    flops = 2.0 * B * H * S * S * (D + D)  # bench_sage convention, non-causal
    tflops = flops / (ms * 1e-3) / 1e12
    print(
        f"[B={B} S={S} H={H} D={D}] {ms*1e3:.1f} us/iter  {tflops:.1f} TFLOPS  "
        f"(non-causal, dense)"
    )
    return tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--hq", type=int, default=5)
    ap.add_argument("--sq", type=int, default=75520)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    torch.set_default_device("cuda")
    bench(args.b, args.sq, args.hq, args.d, iters=args.iters, warmup=args.warmup)


if __name__ == "__main__":
    main()
