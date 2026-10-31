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

import functools
import sys

import torch
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

from aiter.ops.triton.quant.fused_mxfp8_quant import (
    fused_deepseek_v4_dequantize_and_gather_k_cache,
    fused_deepseek_v4_quantize_and_insert_k_cache,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)

_QK, _NOPE = 512, 448
_PACKED_REC, _ALIGNED_REC = 584, 640

_OUT_VAL_KEYS = ["k", "k_cache", "slot_mapping", "block_table", "seq_lens", "out"]


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
    slot_mapping = (blk * block + (pos % block)).reshape(-1)

    k_cache = torch.zeros(nb, block, rec_bytes, dtype=torch.uint8, device=device)
    seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)
    out = torch.empty(num_reqs, seq_len, _QK, dtype=torch.bfloat16, device=device)
    return k, k_cache, slot_mapping, block_table, seq_lens, out


def _do_write(block_size):
    def _inner(k, k_cache, slot_mapping, **_):
        fused_deepseek_v4_quantize_and_insert_k_cache(
            k, k_cache, slot_mapping, block_size
        )

    return _inner


def _do_read(block_size):
    def _inner(out, k_cache, seq_lens, block_table, **_):
        fused_deepseek_v4_dequantize_and_gather_k_cache(
            out, k_cache, seq_lens, None, block_table, block_size, 0
        )

    return _inner


def write_bytes(num_tokens):
    """read bf16 K, write fp8 NoPE + bf16 RoPE + the UE8M0 scale group."""
    return num_tokens * (_QK * 2 + _NOPE + (_QK - _NOPE) * 2 + 8)


def read_bytes(num_tokens):
    """read the record, write bf16 out."""
    return num_tokens * (_NOPE + (_QK - _NOPE) * 2 + 8 + _QK * 2)


def main():
    p = get_parser(kernel_name="DSv4 paged K-cache writer and reader")
    p.add_argument("--reqs", type=int, default=8, help="requests per launch")
    p.add_argument(
        "--seqs",
        type=int,
        nargs="+",
        default=[256, 1024, 4096],
        help="sequence length per request",
    )
    p.add_argument(
        "--rec",
        type=int,
        nargs="+",
        default=[_ALIGNED_REC],
        choices=[_PACKED_REC, _ALIGNED_REC],
        help="record layout in bytes",
    )
    p.add_argument("--block", type=int, default=64, help="paged cache block size")
    p.add_argument(
        "--rep",
        type=int,
        default=20,
        help="Unused: RotatingBufferBench.bench's replay count is a fixed "
        "constant, not an adaptive rep target. Kept for CLI compatibility.",
    )
    a = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")

    print(f"dsv4 k-cache  reqs={a.reqs} block={a.block}")
    print(
        f"{'rec':<8} {'seq':<7} {'write(us)':<10} "
        f"{'write(GB/s)':<12} {'read(us)':<10} {'read(GB/s)':<12}"
    )

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(a), **get_do_bench_graph_kwargs(a)}
    )

    for rec in a.rec:
        for seq in a.seqs:
            _p_gen = functools.partial(build, a.reqs, seq, rec, a.block)
            bencher.build_pool([_p_gen], [_OUT_VAL_KEYS])
            ntok = a.reqs * seq

            wms = bencher.bench(_do_write(a.block))
            rms = bencher.bench(_do_read(a.block))

            print(
                f"{rec:<8d} {seq:<7d} {wms * 1e3:<10.2f} "
                f"{write_bytes(ntok) / (wms * 1e-3) / 1e9:<12.1f} "
                f"{rms * 1e3:<10.2f} "
                f"{read_bytes(ntok) / (rms * 1e-3) / 1e9:<12.1f}"
            )


if __name__ == "__main__":
    main()
