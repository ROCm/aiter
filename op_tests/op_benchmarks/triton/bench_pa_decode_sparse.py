# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Profiling driver for the sparse paged-decode attention kernels.

Two Triton kernels are exercised:
  _pa_decode_sparse        — split-K main kernel (grid: T × n_head_blocks × KV_SPLITS)
  _pa_decode_sparse_reduce — log-sum-exp combine  (grid: T × H)

Usage examples
--------------
# Default DSv4 shapes, bf16 KV, print table:
  python bench_pa_decode_sparse.py

# FP8 KV cache path:
  python bench_pa_decode_sparse.py --kv_dtype fp8

# Single shape for quick profiling (e.g. before rocprof):
  python bench_pa_decode_sparse.py --T 32 --H 128 --D 576 --kv_len 2048

# Override split parameters:
  python bench_pa_decode_sparse.py --block_k 32 --kv_splits 8

# Profile main kernel only (skip the reduce step):
  python bench_pa_decode_sparse.py --skip_reduce

# rocprof wrapper (profile after warmup):
  rocprof --stats python bench_pa_decode_sparse.py --T 32 --H 128 --D 576 --kv_len 2048

Kernels
-------
The two kernels map to SILOTIGER-652 "Stage 3" sparse MLA decode:
  - _pa_decode_sparse:        inner split-K attention loop over gathered KV tiles
  - _pa_decode_sparse_reduce: log-sum-exp combine of KV_SPLITS partial softmax states
"""

import os
import sys

# Make the script runnable directly (python path/to/bench.py) by putting the
# repo root on sys.path, so `import aiter` / `import op_tests` resolve without
# requiring PYTHONPATH or an installed aiter package.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import sys

import torch
import triton

from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

# ---------------------------------------------------------------------------
# DSv4 representative shapes
# ---------------------------------------------------------------------------
# T:      decode batch (number of active tokens)
# H:      query heads (DSv4 = 128)
# D:      absorbed MLA head dim = kv_lora_rank + rope_rank = 512 + 64 = 576
# kv_len: sparse top-k budget (number of KV slots per token, typical ~2048)

DSV4_SHAPES = [
    # (label,       T,    H,    D,   kv_len)
    ("T1-k2k",      1,  128,  576,   2048),   # single-token, full sparse budget
    ("T4-k2k",      4,  128,  576,   2048),   # small batch
    ("T16-k2k",    16,  128,  576,   2048),   # medium batch
    ("T32-k2k",    32,  128,  576,   2048),   # medium batch
    ("T64-k2k",    64,  128,  576,   2048),   # large batch
    ("T128-k2k",  128,  128,  576,   2048),   # large batch
    ("T256-k2k",  256,  128,  576,   2048),   # max batch
    ("T32-k1k",    32,  128,  576,   1024),   # shorter context
    ("T32-k4k",    32,  128,  576,   4096),   # longer context
    ("T32-k8k",    32,  128,  576,   8192),   # long context
]

_FP8_DTYPE = torch.float8_e4m3fnuz
_FP8_GROUP_SIZE = 64


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def make_inputs_bf16(T, H, D, kv_len, seed=42):
    """Build bf16 inputs for pa_decode_sparse."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    total_pages = T * kv_len

    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=torch.bfloat16, device=device) * 0.5
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)

    # Each token attends to exactly kv_len slots drawn from the pool.
    kv_lens = torch.full((T,), kv_len, dtype=torch.int64, device=device)
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=device)
    indptr[1:] = kv_lens.cumsum(0).to(torch.int32)

    total_indices = int(indptr[-1].item())
    indices = torch.randint(0, total_pages, (total_indices,), dtype=torch.int32, device=device)

    softmax_scale = D ** -0.5
    return q, unified_kv, None, indices, indptr, attn_sink, softmax_scale


def make_inputs_fp8(T, H, D, kv_len, seed=42):
    """Build FP8 KV inputs (block-wise quantized) for pa_decode_sparse."""
    q, unified_kv_bf16, _, indices, indptr, attn_sink, softmax_scale = make_inputs_bf16(
        T, H, D, kv_len, seed
    )
    total_pages = unified_kv_bf16.shape[0]
    num_groups = D // _FP8_GROUP_SIZE

    kv_f32 = unified_kv_bf16.float().view(total_pages, num_groups, _FP8_GROUP_SIZE)
    amax = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    fp8_max = torch.finfo(_FP8_DTYPE).max
    scales = (amax / fp8_max).squeeze(-1).to(torch.float32)  # [total_pages, num_groups]
    kv_fp8 = (kv_f32 / amax * fp8_max).view(total_pages, D).to(_FP8_DTYPE)

    return q, kv_fp8, scales, indices, indptr, attn_sink, softmax_scale


# ---------------------------------------------------------------------------
# Bandwidth helpers
# ---------------------------------------------------------------------------

def _bytes_bf16(T, H, D, kv_len):
    """Approximate HBM traffic for the bf16 main kernel (per invocation)."""
    q_bytes = T * H * D * 2                 # query read
    kv_bytes = T * kv_len * D * 2           # KV pool gather (kv_len slots per token)
    out_bytes = T * H * D * 2               # output write
    idx_bytes = T * kv_len * 4              # kv_indices int32 read
    return q_bytes + kv_bytes + out_bytes + idx_bytes


def _bytes_fp8(T, H, D, kv_len):
    """Approximate HBM traffic for the FP8 main kernel."""
    q_bytes = T * H * D * 2                 # bf16 query read
    kv_bytes = T * kv_len * D * 1           # fp8 KV gather
    scales_bytes = T * kv_len * (D // _FP8_GROUP_SIZE) * 4  # fp32 scale reads
    out_bytes = T * H * D * 2               # bf16 output
    idx_bytes = T * kv_len * 4
    return q_bytes + kv_bytes + scales_bytes + out_bytes + idx_bytes


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

def bench_shape(label, T, H, D, kv_len, kv_dtype, block_k, kv_splits, skip_reduce,
                warmup, rep):
    if kv_dtype == "fp8":
        q, unified_kv, kv_scales, indices, indptr, sink, scale = make_inputs_fp8(
            T, H, D, kv_len
        )
        traffic_fn = _bytes_fp8
    else:
        q, unified_kv, kv_scales, indices, indptr, sink, scale = make_inputs_bf16(
            T, H, D, kv_len
        )
        traffic_fn = _bytes_bf16

    def fn():
        pa_decode_sparse(
            q,
            unified_kv,
            indices,
            indptr,
            sink,
            scale,
            kv_scales=kv_scales,
            block_h=None,        # let wrapper choose (min(H,16) → covers all H=128 heads)
            kv_splits=kv_splits, # None = auto
            has_invalid=False,
            skip_reduce=skip_reduce,
        )

    # Warmup + benchmark
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    bw_bytes = traffic_fn(T, H, D, kv_len)
    bw_gbs = bw_bytes / (ms * 1e-3) * 1e-9

    # Arithmetic: two GEMMs per KV tile
    # GEMM1: q[H,D] × kv^T[D,BLOCK_K] per tile; tiles = T * kv_len / BLOCK_K
    bk = block_k if block_k else (16 if D >= 256 else 32)
    n_tiles = T * triton.cdiv(kv_len, bk)
    flops = 2.0 * n_tiles * H * D * bk        # QK dot
    flops += 2.0 * n_tiles * H * bk           # PV accumulate (D=Dv here, approximate)
    tflops = flops / (ms * 1e-3) * 1e-12

    return ms, bw_gbs, tflops


def run_benchmark(args):
    if args.T and args.H and args.D and args.kv_len:
        shapes = [("custom", args.T, args.H, args.D, args.kv_len)]
    else:
        shapes = DSV4_SHAPES

    kv_dtype = args.kv_dtype
    block_k = args.block_k if args.block_k else None
    kv_splits = args.kv_splits if args.kv_splits else None
    skip_reduce = args.skip_reduce
    warmup = args.warmup
    rep = args.rep

    print(
        f"\n{'Shape':<14} {'T':>4} {'H':>4} {'D':>4} {'kv_len':>7} "
        f"{'ms':>8} {'GB/s':>8} {'TFLOPS':>8}"
    )
    print("-" * 67)

    for label, T, H, D, kv_len in shapes:
        ms, bw, tflops = bench_shape(
            label, T, H, D, kv_len,
            kv_dtype=kv_dtype,
            block_k=block_k,
            kv_splits=kv_splits,
            skip_reduce=skip_reduce,
            warmup=warmup,
            rep=rep,
        )
        print(
            f"{label:<14} {T:>4} {H:>4} {D:>4} {kv_len:>7} "
            f"{ms:>8.3f} {bw:>8.1f} {tflops:>8.3f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark _pa_decode_sparse + _pa_decode_sparse_reduce",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Shape overrides (all must be set together to override the default table)
    p.add_argument("--T", type=int, default=0, help="Number of decode tokens")
    p.add_argument("--H", type=int, default=0, help="Number of query heads")
    p.add_argument("--D", type=int, default=0, help="Head dimension")
    p.add_argument("--kv_len", type=int, default=0, help="KV slots per token (sparse budget)")

    # Kernel tuning
    p.add_argument("--block_k", type=int, default=0,
                   help="Override BLOCK_K (default: 16 for D>=256, else 32)")
    p.add_argument("--kv_splits", type=int, default=0,
                   help="Override KV_SPLITS (default: auto to fill ~256 CTAs)")
    p.add_argument("--skip_reduce", action="store_true",
                   help="Profile main kernel only, skip the reduce step")

    # KV dtype
    p.add_argument("--kv_dtype", choices=["bf16", "fp8"], default="bf16",
                   help="KV cache dtype (default: bf16)")

    # Timing
    p.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    p.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")

    return p.parse_args()


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA/HIP device required", file=sys.stderr)
        return 1
    args = parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
