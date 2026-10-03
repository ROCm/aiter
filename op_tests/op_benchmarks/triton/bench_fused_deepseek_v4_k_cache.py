# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark the DeepSeek-V4 paged K-cache writer and reader.

fused_deepseek_v4_quantize_and_insert_k_cache quantizes bf16 K to UE8M0 FP8
and scatters it into the paged cache; fused_deepseek_v4_dequantize_and_gather_k_cache
is its inverse, gathering a window per request. Both are bandwidth bound, so
the figure to watch is GB/s against the traffic each one must move.

Both record layouts are measured, since they differ in how the scales are
placed and that changes the access pattern:

    --rec 640   aligned: fp8 | scales | pad | bf16, all inside the record
    --rec 584   packed:  fp8 | bf16, scales in a region after the block

    python bench_fused_deepseek_v4_k_cache.py --seqs 1024 --reqs 8
"""
import argparse
import sys

import torch
import triton

from aiter.ops.triton.quant.fused_mxfp8_quant import (
    fused_deepseek_v4_dequantize_and_gather_k_cache,
    fused_deepseek_v4_quantize_and_insert_k_cache,
)

_QK, _NOPE = 512, 448
_PACKED_REC, _ALIGNED_REC = 584, 640


def build(num_reqs, seq_len, rec_bytes, block, device="cuda"):
    """One paged cache plus the block table and slot mapping that address it."""
    torch.manual_seed(0)
    per_req = (seq_len + block - 1) // block
    nb = num_reqs * per_req
    num_tokens = num_reqs * seq_len

    k = torch.randn(num_tokens, _QK, dtype=torch.bfloat16, device=device)
    pages = torch.randperm(nb, device=device)
    block_table = pages.reshape(num_reqs, per_req).to(torch.int32)
    pos = torch.arange(seq_len, device=device)
    blk = block_table[:, pos // block].to(torch.int64)
    slot = (blk * block + (pos % block)).reshape(-1)

    cache = torch.zeros(nb, block, rec_bytes, dtype=torch.uint8, device=device)
    seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)
    out = torch.empty(num_reqs, seq_len, _QK, dtype=torch.bfloat16, device=device)
    return k, cache, slot, block_table, seq_lens, out


def write_bytes(num_tokens):
    """read bf16 K, write fp8 NoPE + bf16 RoPE + the UE8M0 scale group."""
    return num_tokens * (_QK * 2 + _NOPE + (_QK - _NOPE) * 2 + 8)


def read_bytes(num_tokens):
    """read the record, write bf16 out."""
    return num_tokens * (_NOPE + (_QK - _NOPE) * 2 + 8 + _QK * 2)


def main():
    p = argparse.ArgumentParser(
        description="Benchmark the DSv4 paged K-cache writer and reader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reqs", type=int, default=8, help="requests per launch")
    p.add_argument("--seqs", type=int, nargs="+", default=[256, 1024, 4096],
                   help="sequence length per request")
    p.add_argument("--rec", type=int, nargs="+", default=[_ALIGNED_REC],
                   choices=[_PACKED_REC, _ALIGNED_REC],
                   help="record layout in bytes")
    p.add_argument("--block", type=int, default=64, help="paged cache block size")
    p.add_argument("--rep", type=int, default=20,
                   help="do_bench_cudagraph measurement target, in ms")
    a = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")

    print("dsv4 k-cache  reqs=%d block=%d" % (a.reqs, a.block))
    print("%-8s %-7s %-10s %-12s %-10s %-12s"
          % ("rec", "seq", "write(us)", "write(GB/s)", "read(us)", "read(GB/s)"))

    for rec in a.rec:
        for seq in a.seqs:
            k, cache, slot, bt, seq_lens, out = build(a.reqs, seq, rec, a.block)
            ntok = a.reqs * seq

            def do_write():
                fused_deepseek_v4_quantize_and_insert_k_cache(k, cache, slot, a.block)

            def do_read():
                fused_deepseek_v4_dequantize_and_gather_k_cache(
                    out, cache, seq_lens, None, bt, a.block, 0
                )

            do_write()          # compile outside the timed region
            do_read()
            torch.cuda.synchronize()
            wms = triton.testing.do_bench_cudagraph(do_write, rep=a.rep)
            rms = triton.testing.do_bench_cudagraph(do_read, rep=a.rep)

            print("%-8d %-7d %-10.2f %-12.1f %-10.2f %-12.1f"
                  % (rec, seq,
                     wms * 1e3, write_bytes(ntok) / (wms * 1e-3) / 1e9,
                     rms * 1e3, read_bytes(ntok) / (rms * 1e-3) / 1e9))


if __name__ == "__main__":
    main()
