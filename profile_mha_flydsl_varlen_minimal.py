#!/usr/bin/env python3
"""Minimal FlyDSL varlen MHA driver for rocprofv3 ATT experiments."""

from __future__ import annotations

import argparse
import math
import os

# Keep aiter package import light; this script imports the FlyDSL kernel module
# directly instead of routing through aiter.ops.mha or op_tests.
os.environ.setdefault("AITER_AOT_IMPORT", "1")
os.environ.setdefault("GPU_ARCHS", "gfx1250")

import torch

from aiter.ops.flydsl.kernels.fmha_gfx1250.fmha_kernel import (
    flash_attn_varlen_d192_gfx1250,
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in ("1", "true", "yes", "y")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal", type=parse_bool, default=True)
    parser.add_argument("--return_lse", type=parse_bool, default=True)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-nh", "--nheads", type=int, default=32)
    parser.add_argument("-sq", "--seqlen_q", type=int, default=8192)
    parser.add_argument("-sk", "--seqlen_k", type=int, default=8192)
    parser.add_argument("--random-value", type=parse_bool, default=False)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    d_qk = 192
    d_v = 128
    total_q = args.batch_size * args.seqlen_q
    total_k = args.batch_size * args.seqlen_k

    if args.random_value:
        torch.manual_seed(42)
        q = torch.randn(total_q, args.nheads, d_qk, dtype=torch.bfloat16, device=device)
        k = torch.randn(total_k, args.nheads, d_qk, dtype=torch.bfloat16, device=device)
        v = torch.randn(total_k, args.nheads, d_v, dtype=torch.bfloat16, device=device)
    else:
        q = torch.full((total_q, args.nheads, d_qk), 0.25, dtype=torch.bfloat16, device=device)
        k = torch.full((total_k, args.nheads, d_qk), 0.25, dtype=torch.bfloat16, device=device)
        v = torch.full((total_k, args.nheads, d_v), 0.25, dtype=torch.bfloat16, device=device)

    cu_q = torch.arange(
        0,
        (args.batch_size + 1) * args.seqlen_q,
        args.seqlen_q,
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.arange(
        0,
        (args.batch_size + 1) * args.seqlen_k,
        args.seqlen_k,
        dtype=torch.int32,
        device=device,
    )
    out = torch.empty(total_q, args.nheads, d_v, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(d_qk)

    def run_once():
        return flash_attn_varlen_d192_gfx1250(
            q,
            k,
            v,
            cu_q,
            cu_k,
            args.seqlen_q,
            args.seqlen_k,
            softmax_scale=scale,
            causal=args.causal,
            out=out,
            return_lse=args.return_lse,
        )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(args.repeat):
        start_event.record()
        result = run_once()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    torch.cuda.synchronize()
    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    if args.return_lse:
        out_result, lse = result
        checksum = float(out_result[0, 0, 0].float().item() + lse[0, 0].float().item())
    else:
        checksum = float(result[0, 0, 0].float().item())
    print(
        "minimal_flydsl_mha "
        f"B={args.batch_size} H={args.nheads} SQ={args.seqlen_q} SK={args.seqlen_k} "
        f"causal={args.causal} return_lse={args.return_lse} "
        f"avg_ms={avg_ms:.6f} checksum={checksum:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
