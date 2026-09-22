# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused Qwen3-Next GDN decode op.

Reports achieved HBM bandwidth against the arch peak. The kernel is bound by the
FP32 recurrent state, which it reads and writes once per token per value head
and which dominates every other tensor by an order of magnitude, so bytes moved
is the right roofline denominator -- it performs no MFMA and no gl.dot.

Timed under CUDA graph replay. Timing the Python call instead measures host
overhead (shape checks, allocation), which at small batches is ~2.5x the kernel
itself and makes every configuration look identical.

The state pool is sized independently of the batch (--slots). Deployment sizes
the mamba pool for max concurrency, so state loads are scattered regardless of
batch; sizing it to the batch puts small batches entirely in cache and inflates
them.

Usage:
    python3 bench_fused_gdn_decode_qkvz.py
    python3 bench_fused_gdn_decode_qkvz.py --batch 1,32,256 --no-fp8
"""

import argparse
import sys

import torch

from aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz import (
    fused_gdn_decode_qkvz,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

# Peak theoretical HBM bandwidth, bytes/s.
_PEAK_BW = {"gfx950": 8.0e12, "gfx942": 5.3e12}
_HEAD_DIM = 128
_CONV_WIDTH = 4


def make_inputs(batch, num_k_heads, head_dim, slots, device="cuda"):
    num_v_heads = 2 * num_k_heads
    ratio = num_v_heads // num_k_heads
    group_width = 2 * head_dim + 2 * ratio * head_dim
    channels = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    slots = max(slots, batch + 1)
    bf16 = {"dtype": torch.bfloat16, "device": device}
    return {
        "projected_qkvz": torch.randn(batch, num_k_heads * group_width, **bf16),
        "projected_ba": torch.randn(batch, 2 * num_v_heads, **bf16),
        "conv_state": torch.randn(slots, channels, _CONV_WIDTH - 1, **bf16),
        "ssm_state": torch.randn(
            slots, num_v_heads, head_dim, head_dim, dtype=torch.float32, device=device
        ),
        "ssm_state_indices": torch.arange(
            1, batch + 1, dtype=torch.int32, device=device
        ),
        "conv_weight": torch.randn(channels, _CONV_WIDTH, **bf16),
        "conv_bias": torch.randn(channels, **bf16),
        "A_log": torch.randn(num_v_heads, dtype=torch.float32, device=device),
        "dt_bias": torch.randn(num_v_heads, **bf16),
        "norm_weight": torch.randn(head_dim, **bf16),
    }


def bytes_moved(batch, num_k_heads, head_dim, fp8):
    """Compulsory traffic for one launch."""
    num_v_heads = 2 * num_k_heads
    channels = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    state = batch * num_v_heads * head_dim * head_dim * 4  # fp32
    total = 2 * state  # read + write, the dominant term
    total += 2 * batch * channels * (_CONV_WIDTH - 1) * 2  # conv window r+w, bf16
    total += batch * num_k_heads * (2 * head_dim + 4 * head_dim) * 2  # packed qkvz
    total += batch * num_v_heads * head_dim * 2  # bf16 out
    if fp8:
        total += batch * num_v_heads * head_dim  # fp8 out
        total += batch * num_v_heads * 4  # scales
    return total


def bench_one(batch, num_k_heads, head_dim, slots, quant_dtype, iters, warmup=25):
    inp = make_inputs(batch, num_k_heads, head_dim, slots)
    args = (
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["ssm_state"],
        inp["ssm_state_indices"],
        inp["conv_weight"],
        inp["conv_bias"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_weight"],
    )
    kwargs = {"scale": head_dim**-0.5, "norm_eps": 1e-6, "quant_dtype": quant_dtype}

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fused_gdn_decode_qkvz(*args, **kwargs)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_gdn_decode_qkvz(*args, **kwargs)
    torch.cuda.synchronize()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--num-k-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=_HEAD_DIM)
    ap.add_argument("--slots", type=int, default=512)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no-fp8", action="store_true", help="bf16 output only")
    args = ap.parse_args()

    arch = get_arch()
    if arch != "gfx950":
        print(f"fused_gdn_decode_qkvz is gfx950-only; this is {arch}")
        return 0

    quant_dtype = None if args.no_fp8 else torch.float8_e4m3fn
    peak = _PEAK_BW.get(arch)
    print(
        f"fused_gdn_decode_qkvz on {torch.cuda.get_device_name(0)} ({arch}), "
        f"num_k_heads={args.num_k_heads} head_dim={args.head_dim} "
        f"slots={args.slots} fp8={not args.no_fp8}"
    )
    # us per call; this op is always a single launch, so also per launch.
    print(f"{'batch':>7}{'us/call':>10}{'GB moved':>11}{'TB/s':>9}{'% peak':>9}")
    for b in [int(x) for x in args.batch.split(",")]:
        us = bench_one(
            b, args.num_k_heads, args.head_dim, args.slots, quant_dtype, args.iters
        )
        nbytes = bytes_moved(
            b, args.num_k_heads, args.head_dim, quant_dtype is not None
        )
        tbs = nbytes / (us * 1e-6) / 1e12
        pct = 100.0 * tbs * 1e12 / peak if peak else float("nan")
        print(f"{b:>7}{us:>10.2f}{nbytes / 1e9:>11.4f}{tbs:>9.2f}{pct:>8.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
