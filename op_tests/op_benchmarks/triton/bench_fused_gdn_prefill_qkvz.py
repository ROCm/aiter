# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused Qwen3-Next GDN prefill op.

Reports latency, prefill throughput (tokens/s) and achieved HBM bandwidth per
(batch, seqlen). Unlike the decode op -- which is bound by the FP32 recurrent
state it streams once per token -- prefill runs the chunked delta-rule scan with
MFMA/``gl.dot`` and is closer to compute-bound, so tokens/s is the headline and
%peak-BW is only a loose roofline proxy.

Timed under CUDA graph replay: the per-tile grids are computed on the host from
(tokens, batch), so shapes are static and capture is valid. Timing the Python
call instead would fold in host overhead (shape checks, allocation).

Each (batch, seqlen) pair selects a tile via the wrapper's profile map; the
default sweep hits all four tiles. Usage:
    python3 bench_fused_gdn_prefill_qkvz.py
    python3 bench_fused_gdn_prefill_qkvz.py --shapes 1x1024,4x2048,8x1600
"""

import argparse
import sys

import torch

from aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz import (
    _select_tile_key,
    fused_gdn_prefill_qkvz,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

# Peak theoretical HBM bandwidth, bytes/s.
_PEAK_BW = {"gfx950": 8.0e12, "gfx942": 5.3e12}
_HEAD_DIM = 128
_CONV_WIDTH = 4

# Default sweep: (batch, seqlen) pairs chosen to cover every tile.
_DEFAULT_SHAPES = "1x1024,1x2048,1x3072,1x8192,3x2048,1x16384,8x1600"


def make_inputs(batch, seqlen, num_k_heads, head_dim, device="cuda"):
    num_v_heads = 2 * num_k_heads
    ratio = num_v_heads // num_k_heads
    group_width = 2 * head_dim + 2 * ratio * head_dim
    channels = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    m = batch * seqlen
    slots = batch + 1
    bf16 = dict(dtype=torch.bfloat16, device=device)
    cu = torch.arange(0, m + 1, seqlen, dtype=torch.int32, device=device)
    return dict(
        projected_qkvz=torch.randn(m, num_k_heads * group_width, **bf16),
        projected_ba=torch.randn(m, 2 * num_v_heads, **bf16),
        conv_state=torch.randn(slots, channels, _CONV_WIDTH - 1, **bf16),
        delta_state=torch.randn(
            slots, num_v_heads, head_dim, head_dim, dtype=torch.float32, device=device
        ),
        cache_indices=torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        cu_seqlens=cu,
        has_initial_state=torch.ones(batch, dtype=torch.bool, device=device),
        conv_weight=torch.randn(channels, _CONV_WIDTH, **bf16),
        conv_bias=torch.randn(channels, **bf16),
        A_log=torch.randn(num_v_heads, dtype=torch.float32, device=device),
        dt_bias=torch.randn(num_v_heads, **bf16),
        norm_weight=torch.randn(head_dim, **bf16),
    )


def bytes_moved(m, batch, num_k_heads, head_dim):
    """Compulsory traffic for one launch (fp8 epilogue always on)."""
    num_v_heads = 2 * num_k_heads
    channels = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    total = m * num_k_heads * (2 * head_dim + 4 * head_dim) * 2  # packed qkvz, bf16
    total += m * 2 * num_v_heads * 2  # packed ba, bf16
    total += 2 * batch * num_v_heads * head_dim * head_dim * 4  # delta state r+w, fp32
    total += 2 * batch * channels * (_CONV_WIDTH - 1) * 2  # conv window r+w, bf16
    total += m * num_v_heads * head_dim * 2  # bf16 normalized out
    total += m * num_v_heads * head_dim  # fp8 out
    total += m * num_v_heads * 4  # scales
    return total


def bench_one(batch, seqlen, num_k_heads, head_dim, iters, warmup=25):
    inp = make_inputs(batch, seqlen, num_k_heads, head_dim)
    args = (
        inp["projected_qkvz"], inp["projected_ba"], inp["conv_state"],
        inp["delta_state"], inp["cache_indices"], inp["cu_seqlens"],
        inp["has_initial_state"], inp["conv_weight"], inp["conv_bias"],
        inp["A_log"], inp["dt_bias"], inp["norm_weight"],
    )
    kwargs = dict(scale=head_dim**-0.5, eps=1e-6)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fused_gdn_prefill_qkvz(*args, **kwargs)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_gdn_prefill_qkvz(*args, **kwargs)
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
    ap.add_argument(
        "--shapes",
        default=_DEFAULT_SHAPES,
        help="comma list of BxS (batch x seqlen), e.g. 1x1024,4x2048",
    )
    ap.add_argument("--num-k-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=_HEAD_DIM)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    arch = get_arch()
    if arch != "gfx950":
        print(f"fused_gdn_prefill_qkvz is gfx950-only; this is {arch}")
        return 0

    peak = _PEAK_BW.get(arch)
    print(f"fused_gdn_prefill_qkvz on {torch.cuda.get_device_name(0)} ({arch}), "
          f"num_k_heads={args.num_k_heads} head_dim={args.head_dim}")
    hdr = f"{'batch':>6}{'seqlen':>8}{'tokens':>8}{'tile':>20}{'us':>10}{'Mtok/s':>9}{'TB/s':>8}{'%pk':>6}"
    print(hdr)
    for spec in args.shapes.split(","):
        b_str, s_str = spec.lower().split("x")
        b, s = int(b_str), int(s_str)
        m = b * s
        tile = _select_tile_key(m, b)
        if tile is None:
            print(f"{b:>6}{s:>8}{m:>8}{'(uncovered)':>20}")
            continue
        us = bench_one(b, s, args.num_k_heads, args.head_dim, args.iters)
        mtoks = m / (us * 1e-6) / 1e6
        nbytes = bytes_moved(m, b, args.num_k_heads, args.head_dim)
        tbs = nbytes / (us * 1e-6) / 1e12
        pct = 100.0 * tbs * 1e12 / peak if peak else float("nan")
        print(
            f"{b:>6}{s:>8}{m:>8}{tile:>20}{us:>10.2f}{mtoks:>9.2f}{tbs:>8.2f}{pct:>5.0f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
