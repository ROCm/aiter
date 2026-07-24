# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Microbenchmark for the scattered-pointer Q4_K_M MoE matvec kernel.
# Mirrors the streaming-MoE call pattern: K active experts in K separate
# device buffers, fp32 input, fp32 output.
#
# This benchmark reports kernel-only latency (Triton ``do_bench``) — it does
# not include any host-side cache-miss / SSD-read time, since those are
# orthogonal to kernel performance.

import argparse
import sys

import numpy as np
import torch
import triton

from aiter.ops.triton.moe.moe_op_q4k_streaming import (
    fused_moe_q4k_streaming,
    QK_K,
    BLOCK_BYTES,
)
from op_tests.triton_tests.moe.q4k_pack_reference import build_pattern_expert


def _build_inputs(n_tokens, n_used_per_token, n_unique, n_dim_in, n_dim_out, seed=0):
    rng = np.random.default_rng(seed)
    # Bench cares about memory access pattern, not random byte content. Use
    # the fast pattern builder (one random block replicated to fill each
    # expert's buffer); each unique expert still gets its own VRAM allocation
    # so cache behaviour matches the streaming-MoE call site.
    expert_bufs = [
        build_pattern_expert(rng, n_dim_in, n_dim_out) for _ in range(n_unique)
    ]
    expert_tensors = [
        torch.frombuffer(bytearray(b), dtype=torch.uint8).cuda() for b in expert_bufs
    ]
    expert_ptrs = torch.tensor(
        [t.data_ptr() for t in expert_tensors], dtype=torch.uint64, device="cuda"
    )
    remap_np = rng.integers(
        0, n_unique, size=(n_tokens, n_used_per_token), dtype=np.int32
    )
    remap = torch.from_numpy(remap_np).cuda()
    a = torch.randn(n_tokens, n_dim_in, dtype=torch.float32, device="cuda")
    c = torch.zeros(
        (n_tokens, n_used_per_token, n_dim_out), dtype=torch.float32, device="cuda"
    )
    # Keep refs alive to prevent GC of the underlying device buffers.
    return a, expert_ptrs, remap, c, expert_tensors


# Reference workloads matching real streaming-MoE deployments. All n_dim_in
# values must be multiples of QK_K=256 (a hard Q4_K_M constraint).
WORKLOADS = [
    # name,                     n_tokens, n_used, n_unique, n_dim_in, n_dim_out
    # Mixtral 8x7B: hidden=4096, intermediate=14336, top_k=2
    ("mixtral8x7b_gate_up_decode",      1,      2,        2,     4096,     14336),
    ("mixtral8x7b_down_decode",         1,      2,        2,    14336,      4096),
    # DeepSeek-V3 per-expert: hidden=7168, intermediate=2048, top_k=8 routed
    ("dsv3_gate_up_decode",             1,      8,        8,     7168,      2048),
    ("dsv3_down_decode",                1,      8,        8,     2048,      7168),
    # Qwen3.5-397B-A17B per-expert: hidden=4096, intermediate=1536, top_k=4
    ("qwen35_397b_gate_up_decode",      1,      4,        4,     4096,      1536),
    ("qwen35_397b_down_decode",         1,      4,        4,     1536,      4096),
    # Higher-batch decode (vLLM continuous batching)
    ("mixtral8x7b_gate_up_b8",          8,      2,        8,     4096,     14336),
]


def _bytes_per_dispatch(n_unique, n_dim_in, n_dim_out):
    """Total Q4_K_M bytes read across all unique experts in one dispatch."""
    return n_unique * n_dim_out * (n_dim_in // QK_K) * BLOCK_BYTES


def _flops_per_dispatch(n_tokens, n_used, n_dim_in, n_dim_out):
    """fma count = n_tokens * n_used * n_dim_in * n_dim_out (treat fma = 2 flops)."""
    return 2.0 * n_tokens * n_used * n_dim_in * n_dim_out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--workload",
        default=None,
        help="Run a single workload by name; defaults to all.",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    print(
        f"{'workload':32}  {'tok':>3}  {'used':>4}  {'uniq':>4}  "
        f"{'K':>5}  {'N':>5}  {'ms':>7}  {'GB/s':>8}  {'GFLOPS':>8}"
    )

    for name, n_tokens, n_used, n_unique, n_dim_in, n_dim_out in WORKLOADS:
        if args.workload and args.workload != name:
            continue

        a, expert_ptrs, remap, c, _refs = _build_inputs(
            n_tokens, n_used, n_unique, n_dim_in, n_dim_out
        )

        def run():
            fused_moe_q4k_streaming(a, expert_ptrs, remap, c, n_dim_out=n_dim_out)

        # Warmup + measure
        ms = triton.testing.do_bench(
            run, warmup=args.warmup, rep=args.rep, return_mode="median"
        )

        bytes_total = _bytes_per_dispatch(n_unique, n_dim_in, n_dim_out)
        flops = _flops_per_dispatch(n_tokens, n_used, n_dim_in, n_dim_out)
        gbs = bytes_total / (ms * 1e-3) / 1e9
        gflops = flops / (ms * 1e-3) / 1e9
        print(
            f"{name:32}  {n_tokens:>3}  {n_used:>4}  {n_unique:>4}  "
            f"{n_dim_in:>5}  {n_dim_out:>5}  {ms:>7.3f}  {gbs:>8.1f}  {gflops:>8.1f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
