# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL FMHA backward benchmark — direct (no autograd, no Triton fallback).

Unlike op_tests/op_benchmarks/triton/bench_mha.py (which routes through
flash_attn_func().backward() and silently falls back to Triton when the shape is
unsupported — e.g. Sq % 64 != 0), this calls the FlyDSL host wrapper
`flydsl_flash_attn_backward` directly. It either runs FlyDSL or prints that the
shape is unsupported — it never measures Triton by accident.

Same CLI flags as the Triton bench, so the same command style works:
  cd /workspace && HIP_VISIBLE_DEVICES=$GPU \
    python op_tests/op_benchmarks/flydsl/bench_mha.py \
    -fn bwd -b $B -hq $HQ -hk $HK -sq $SQ -sk $SK -d $D -causal $CAUSAL \
    -metric throughput \
    2>&1 | tee /my_notes/flydsl/logs/flydsl_backward.log

Constraints (else "unsupported → would fall back to Triton"): bwd only, non-causal,
Hq==Hk (MHA), d=128, bf16/fp16, Sq/Sk multiples of 64.
"""

import argparse
import math

import torch

from aiter.ops.triton.attention.mha import flash_attn_func
from aiter.ops.flydsl.fmha_bwd_kernels import flydsl_flash_attn_backward

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def main():
    p = argparse.ArgumentParser(description="FlyDSL FMHA backward benchmark (direct)")
    p.add_argument("-fn", default="bwd", help="only 'bwd' is supported")
    p.add_argument("-b", type=int, default=1)
    p.add_argument("-hq", type=int, default=5)
    p.add_argument("-hk", type=int, default=5)
    p.add_argument("-sq", type=int, default=75648)
    p.add_argument("-sk", type=int, default=None, help="defaults to -sq")
    p.add_argument("-d", type=int, default=128)
    p.add_argument("-causal", default="False")
    p.add_argument("-dtype", default="bf16", choices=list(_DTYPES))
    p.add_argument(
        "-metric",
        default="throughput",
        choices=["time", "throughput"],
    )
    p.add_argument("-warmup", type=int, default=5)
    p.add_argument("-iters", type=int, default=20)
    args = p.parse_args()

    if args.fn != "bwd":
        raise SystemExit(f"only -fn bwd is supported (got {args.fn})")

    B, HQ, HK, D = args.b, args.hq, args.hk, args.d
    SQ = args.sq
    SK = args.sk if args.sk is not None else SQ
    causal = _str2bool(args.causal)
    dtype = _DTYPES[args.dtype]
    sm_scale = 1.0 / math.sqrt(D)
    device = "cuda"

    torch.manual_seed(42)
    q = torch.randn(B, SQ, HQ, D, dtype=dtype, device=device)
    k = torch.randn(B, SK, HK, D, dtype=dtype, device=device)
    v = torch.randn(B, SK, HK, D, dtype=dtype, device=device)

    # Forward (Triton) only to obtain out + softmax_lse for the backward inputs.
    with torch.no_grad():
        out, lse = flash_attn_func(
            q, k, v, softmax_scale=sm_scale, causal=causal, return_lse=True
        )
    do = torch.randn_like(out)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    def run():
        return flydsl_flash_attn_backward(
            q, k, v, out, do, lse, dq, dk, dv, sm_scale, causal
        )

    handled = run()
    if not handled:
        raise SystemExit(
            "UNSUPPORTED by FlyDSL for this shape (would fall back to Triton). "
            "Require: non-causal, Hq==Hk, d=128, bf16/fp16, Sq/Sk multiples of 64."
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters

    # Backward FLOPs = 5 matmuls of [S,S]@[S,D] = 10 * B * H * Sq * Sk * D
    # (same 2.5x-forward convention as the plan's CK/Triton table).
    flops = 10 * B * HQ * SQ * SK * D
    tflops = flops / (ms * 1e-3) / 1e12

    print(
        f"FlyDSL backward | B={B} HQ={HQ} HK={HK} Sq={SQ} Sk={SK} D={D} "
        f"{args.dtype} causal={causal}"
    )
    if args.metric == "time":
        print(f"  {ms:.3f} ms")
    else:
        print(f"  {ms:.3f} ms   {tflops:.1f} TFLOPS   (CK target ~836)")


if __name__ == "__main__":
    main()
