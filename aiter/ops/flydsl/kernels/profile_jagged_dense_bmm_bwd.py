# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Profiling / roofline driver for the jagged_dense_bmm BACKWARD kernels.
#
# This is deliberately separate from example_jagged_dense_bmm_bwd.py (the
# correctness harness). It does the *minimum* GPU work needed to place each
# backward kernel on a roofline:
#   * builds inputs and device buffers once,
#   * compiles each launcher once (warmup, JIT cache hit on replays),
#   * then issues only the kernel-under-test in a tight loop.
# No torch reference GEMMs / autograd run here, so the captured dispatches are
# exclusively our FlyDSL kernels -- which keeps rocprof(-compute) output clean
# and keeps every profiler replay cheap.
#
# Two modes:
#   --mode bench    wall-clock timing -> achieved TFLOP/s, GB/s and arithmetic
#                   intensity (FLOP/byte). A quick, profiler-free roofline coord.
#   --mode profile  just runs the warmup + timed loop (no host timing prints);
#                   this is the command you hand to rocprof-compute, e.g.
#                       rocprof-compute profile -n bwd_ddense --roof-only -- \
#                         flydsl_venv/bin/python profile_jagged_dense_bmm_bwd.py \
#                         --mode profile --only ddense --iters 50
#
# Run the app itself in flydsl_venv (torch + flydsl). The rocprof-compute driver
# runs in its own venv; see docs for the wrapper script.

from __future__ import annotations

import argparse
import sys
import time

import torch

import flydsl.compiler as flyc

# Sibling imports (script dir is on sys.path[0]); avoids importing the aiter pkg.
import example_jagged_dense_bmm_bwd as _ex
import jagged_dense_bmm_bwd as _bwd
from example_jagged_dense_bmm_bwd import make_inputs
from jagged_dense_bmm import BLOCK_M, K, N
from jagged_dense_bmm_bwd import SPLIT, grad_dense_bias, grad_jagged

# dDense and dBias share one fused launcher (grad_dense_bias), so they are profiled
# together as the "dense_bias" target (mirrors the bench component of the same name).
GRADS = ("djagged", "dense_bias")

# roctx range markers (torch maps the nvtx API onto roctx on ROCm). Optional:
# used only to annotate hip-trace / timeline captures, guarded so a missing
# backend never breaks profiling.
try:
    _range_push = torch.cuda.nvtx.range_push
    _range_pop = torch.cuda.nvtx.range_pop
except Exception:  # pragma: no cover - depends on build
    def _range_push(_name):
        return None

    def _range_pop():
        return None


def _prep(which, jagged, dense, d_out, seq_offsets, n_groups, max_seq_len):
    """Allocate device buffers once and return a zero-arg launch closure that
    issues exactly the kernel(s) for gradient `which`."""
    device = jagged.device
    total_rows = jagged.shape[0]
    stream = torch.cuda.current_stream()

    tDOut = flyc.from_dlpack(d_out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    if which == "djagged":
        dense_kn = dense.reshape(n_groups * K, N).contiguous()
        d_jagged = torch.zeros(total_rows + BLOCK_M, K, dtype=torch.bfloat16, device=device)
        tDJ = flyc.from_dlpack(d_jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)

        def launch():
            grad_jagged(tDJ, tDOut, dense_kn, seq_offsets, n_groups, max_seq_len, stream=stream)

        return launch

    if which == "dense_bias":
        d_dense = torch.zeros(n_groups, K, N, dtype=torch.bfloat16, device=device)
        d_bias = torch.zeros(n_groups, N, dtype=torch.bfloat16, device=device)
        dense_partials = torch.zeros(n_groups * SPLIT * K, N, dtype=torch.float32, device=device)
        bias_partials = torch.zeros(n_groups * SPLIT, N, dtype=torch.float32, device=device)
        tJagged = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
        d_dense_v = d_dense.view(n_groups * K, N)

        def launch():
            grad_dense_bias(d_dense_v, d_bias, tJagged, tDOut, seq_offsets, dense_partials,
                            bias_partials, n_groups, max_seq_len, stream=stream)

        return launch

    raise ValueError(f"unknown gradient '{which}'")


def _flops_bytes(which, total_rows, n_groups):
    """Analytic FLOP count and minimal DRAM bytes touched (bf16 in/out, fp32
    scratch) for arithmetic-intensity estimation. L = total packed rows."""
    L = total_rows
    if which == "djagged":
        # dJagged[s:e] = dOut[s:e] @ Dense[b].T -> 2*L*N*K MACs.
        flops = 2 * L * N * K
        bytes_ = L * N * 2 + n_groups * K * N * 2 + L * K * 2
    elif which == "dense_bias":
        # Fused dDense (+ dBias). dDense[b] = Jagged[s:e].T @ dOut[s:e] -> 2*L*K*N MACs
        # dominates; the dBias reduction (~L*N adds) now piggybacks on the same dOut
        # reads. partials kernel reads Jagged+dOut, writes dDense+dBias fp32 partials;
        # the two reduce passes read those partials and write bf16 dDense+dBias.
        flops = 2 * L * K * N
        dpart = n_groups * SPLIT * K * N * 4
        bpart = n_groups * SPLIT * N * 4
        bytes_ = (L * K * 2 + L * N * 2 + dpart + bpart) + (
            dpart + n_groups * K * N * 2 + bpart + n_groups * N * 2
        )
    else:
        raise ValueError(which)
    return flops, bytes_


def run(which, jagged, dense, d_out, seq_offsets, n_groups, max_seq_len,
        warmup, iters, mode):
    launch = _prep(which, jagged, dense, d_out, seq_offsets, n_groups, max_seq_len)

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    if mode == "profile":
        for _ in range(iters):
            _range_push(f"bwd_{which}")
            launch()
            _range_pop()
        torch.cuda.synchronize()
        return None

    # bench: wall-clock around the timed loop.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        launch()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    flops, bytes_ = _flops_bytes(which, jagged.shape[0], n_groups)
    tflops = flops / dt / 1e12
    gbps = bytes_ / dt / 1e9
    ai = flops / bytes_
    print(f"  {which:<8} {dt * 1e6:9.2f} us/iter  "
          f"{tflops:8.2f} TFLOP/s  {gbps:8.1f} GB/s  AI={ai:6.2f} FLOP/byte")
    return dt


def main(argv=None):
    p = argparse.ArgumentParser(description="jagged_dense_bmm backward: profiling / roofline driver")
    p.add_argument("-b", "--n-groups", type=int, default=64)
    p.add_argument("-m", "--max-seq-len", type=int, default=512)
    p.add_argument("-d", "--dim", type=int, default=None,
                   help="square dense dim D (= K = N); overrides the compile-time "
                        "constant at runtime so no source edit is needed")
    p.add_argument("--regime", choices=["uniform", "skew"], default="uniform")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--only", default="all", help="comma list of {djagged,dense_bias} or 'all'")
    p.add_argument("--mode", choices=["bench", "profile"], default="bench")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA/ROCm device not available; this driver requires a GPU.")
        return 1

    # Apply the runtime D override BEFORE any kernel launch (FlyDSL snapshots used
    # globals on first compile). Propagate the new K/N/SPLIT to this module and the
    # example module (make_inputs allocates from those snapshotted-by-value names).
    global K, N, SPLIT
    if args.dim is not None:
        _bwd.configure_dim(args.dim)
        K = N = _ex.K = _ex.N = args.dim
        SPLIT = _bwd.SPLIT

    which = GRADS if args.only == "all" else tuple(s.strip() for s in args.only.split(","))
    for w in which:
        if w not in GRADS:
            print(f"unknown gradient '{w}'; choose from {GRADS} or 'all'")
            return 2

    device = "cuda"
    jagged, dense, bias, d_out, seq_offsets, total_rows = make_inputs(
        args.n_groups, args.max_seq_len, args.regime, args.seed, device
    )
    if args.mode == "bench":
        print(f"shape: n_groups={args.n_groups}, max_seq_len={args.max_seq_len}, K={K}, N={N}, "
              f"regime={args.regime}, split={SPLIT}, L={total_rows}, "
              f"warmup={args.warmup}, iters={args.iters}")

    for w in which:
        run(w, jagged, dense, d_out, seq_offsets, args.n_groups, args.max_seq_len,
            args.warmup, args.iters, args.mode)

    return 0


if __name__ == "__main__":
    sys.exit(main())
