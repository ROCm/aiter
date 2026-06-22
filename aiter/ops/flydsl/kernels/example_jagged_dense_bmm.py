# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Standalone example driving the jdbba_kernel via the jagged_dense_bmm @flyc.jit
# launcher in jagged_dense_bmm.py. It builds jagged/dense/bias inputs, runs the
# kernel, validates against a torch eager reference, and reports runtime + TFLOPs.
#
# Computes, per group b over its packed row slice [s, e):
#     Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
#
# Run inside the project venv:
#     source flydsl_venv/bin/activate
#     python aiter/aiter/ops/flydsl/kernels/example_jagged_dense_bmm.py
#
# This file lives next to jagged_dense_bmm.py and imports it as a sibling module
# so it does not pull in the full aiter package (which has extra JIT build deps).

from __future__ import annotations

import argparse
import sys

import torch

import flydsl.compiler as flyc

# Sibling import (script dir is on sys.path[0]); avoids importing the aiter pkg.
from jagged_dense_bmm import BLOCK_M, K, N, jagged_dense_bmm


def make_seq_offsets(n_groups, max_seq_len, regime, seed, device):
    """Per-group prefix-sum row offsets (int32, length n_groups + 1).

    uniform: every group has exactly max_seq_len rows.
    skew:    max_seq_len * U(0,1)**4 rows, with ~20% empty groups plus one full
             and one near-full group (closer to a real deployment distribution).
    """
    if regime == "uniform":
        return torch.arange(
            0, (n_groups + 1) * max_seq_len, max_seq_len, dtype=torch.int32, device=device
        )
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n_groups, generator=g)
    rows = (max_seq_len * (u**4)).floor().to(torch.int64)
    rows[: max(1, n_groups // 5)] = 0
    rows[-1] = max_seq_len
    if n_groups > 1:
        rows[-2] = int(0.9 * max_seq_len)
    so = torch.zeros(n_groups + 1, dtype=torch.int32)
    for i in range(n_groups):
        so[i + 1] = so[i] + int(rows[i])
    return so.to(device)


def make_inputs(n_groups, max_seq_len, regime, seed, device):
    torch.manual_seed(0)
    seq_offsets = make_seq_offsets(n_groups, max_seq_len, regime, seed, device)
    total_rows = int(seq_offsets[-1].item())
    jagged = torch.randn(max(total_rows, 1), K, dtype=torch.bfloat16, device=device)
    dense = torch.randn(n_groups, K, N, dtype=torch.bfloat16, device=device)
    bias = torch.randn(n_groups, N, dtype=torch.bfloat16, device=device)
    return jagged, dense, bias, seq_offsets, total_rows


def torch_reference(jagged, dense, bias, seq_offsets):
    total_rows = jagged.shape[0]
    out = torch.zeros((total_rows, N), dtype=torch.bfloat16, device=jagged.device)
    for b in range(dense.shape[0]):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            out[s:e] = (
                jagged[s:e].float() @ dense[b].float() + bias[b].float()[None, :]
            ).to(torch.bfloat16)
    return out


def run_flydsl(jagged, dense, bias, seq_offsets, n_groups, max_seq_len):
    # FlyDSL wants Dense as a tall (n_groups * N, K) matrix and a flat bias.
    dense_tall = dense.transpose(1, 2).reshape(n_groups * N, K).contiguous()
    bias_flat = bias.reshape(n_groups * N).contiguous()
    total_rows = jagged.shape[0]
    # Pad output rows by BLOCK_M so any partial tail-tile store stays in-bounds.
    out = torch.zeros(total_rows + BLOCK_M, N, dtype=torch.bfloat16, device="cuda")

    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    def fn():
        jagged_dense_bmm(
            tC, tA, dense_tall, bias_flat, seq_offsets, n_groups, max_seq_len,
            stream=torch.cuda.current_stream(),
        )

    return fn, out[:total_rows]


def benchmark(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per launch


def main(argv=None):
    p = argparse.ArgumentParser(description="jagged_dense_bmm (jdbba) example: validate + benchmark")
    p.add_argument("-b", "--n-groups", type=int, default=64, help="number of groups (batch)")
    p.add_argument("-m", "--max-seq-len", type=int, default=512, help="max rows per group")
    p.add_argument("--regime", choices=["uniform", "skew"], default="uniform")
    p.add_argument("--seed", type=int, default=1234, help="skew RNG seed")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA/ROCm device not available; this example requires a GPU.")
        return 1

    device = "cuda"
    print(f"shape: n_groups={args.n_groups}, max_seq_len={args.max_seq_len}, "
          f"K={K}, N={N}, regime={args.regime}")

    jagged, dense, bias, seq_offsets, total_rows = make_inputs(
        args.n_groups, args.max_seq_len, args.regime, args.seed, device
    )
    print(f"packed rows L = {total_rows}")

    fn, got = run_flydsl(jagged, dense, bias, seq_offsets, args.n_groups, args.max_seq_len)

    # --- Validation ---
    ref = torch_reference(jagged, dense, bias, seq_offsets)
    fn()
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), got.float().flatten(), dim=0
    ).item()
    max_abs = (ref.float() - got.float()).abs().max().item()
    passed = cos > 0.999
    print(f"validation: {'PASS' if passed else 'FAIL'}  cosine={cos:.6f}  max_abs_err={max_abs:.4f}")

    # --- Performance ---
    ms = benchmark(fn, args.warmup, args.iters)
    # FLOPs use the actual packed length L (sum of M_b): per group 2*M_b*K*N.
    flops = 2.0 * total_rows * K * N
    tflops = flops / (ms * 1e-3) / 1e12
    mem = (total_rows * K + args.n_groups * K * N + args.n_groups * N + total_rows * N) * 2
    gbps = mem / (ms * 1e-3) / 1e9
    print(f"performance: {ms:.4f} ms/launch   {tflops:.2f} TFLOP/s   {gbps:.1f} GB/s")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
