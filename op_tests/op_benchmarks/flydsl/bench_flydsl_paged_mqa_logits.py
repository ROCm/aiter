# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone microbenchmark for the FlyDSL paged FP8 MQA-logits (decode) kernel.

Runs ONE config of ``flydsl_fp8_paged_mqa_logits`` in isolation (no reference /
Gluon comparison) so the kernel can be profiled directly. Example rocprofv3 run::

    PYTHONPATH=. rocprofv3 --kernel-trace --output-format csv \\
        --output-directory .rocprofv3 -- \\
        python op_tests/op_benchmarks/flydsl/bench_flydsl_paged_mqa_logits.py \\
            --heads 64 --head-dim 128 --kv-len 32768 --batch 16 --next-n 2 \\
            --kv-block-size 64 --preshuffle --iters 200 --no-check

Covers both the block-flat gather (``--kv-block-size N``, default 1) and the
production preshuffle layout (``--preshuffle``; requires KVBlockSize % 16 == 0).
The input build, co-pack byte layout, preshuffle, and torch oracle are imported
from the flydsl correctness test, so the profiled kernel sees exactly the tested
layout (single source of truth).

By default the run is correctness-gated once (torch oracle, exact ``-inf`` mask +
calc_diff < 1e-3) before the timed loop; pass ``--no-check`` to skip it (handy
when profiling). The timed loop launches the kernel ``--iters`` times back to
back -- that is what rocprofv3 traces. ``--time`` additionally reports device
self-time via ``run_perftest``.
"""

import argparse
import random

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits
from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits import DEFAULT_VARIANT
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_common import run_perftest

# Reuse the correctness test's layout/oracle helpers so the profiled kernel sees
# exactly the tested co-pack + preshuffle byte layout.
from op_tests.flydsl_tests.test_flydsl_fp8_paged_mqa_logits import (
    calc_diff,
    kv_cache_cast_to_fp8,
    preshuffle_kv_data,
    ref_fp8_paged_mqa_logits,
)

torch.set_default_device("cuda")

DTYPE_MAP = {"fnuz": get_fp8_e4m3_dtype(), "fn": torch.float8_e4m3fn}


def build_inputs(
    batch,
    next_n,
    heads,
    head_dim,
    kv_len,
    block_size,
    q_dtype,
    var_ratio,
    seed,
    pool_blocks=0,
):
    """Paged decode inputs (block-flat co-packed cache), block_size >= 1.

    ``var_ratio`` controls the per-sequence context-length spread around
    ``kv_len``: 0.0 == exact (every sequence == kv_len); v>0 draws lengths in
    [(1-v), (1+v)]*kv_len. A fixed ``seed`` makes the whole build deterministic
    (stable kernel work across profiling runs).

    The pool holds one distinct physical block per block the batch actually
    needs, so the KV footprint the kernel touches is ``sum(ctx)*index_dim`` --
    what a real decode reads. Sizing the pool from ``max_model_len`` instead
    (as an earlier version did) makes the block table wrap: at batch=16,
    kv_len=32768, KVBlockSize=64 that is a 1024-block pool serving 8192 block
    references, so all 16 sequences share the same blocks and the working set
    collapses from 69 MB to 8.65 MB -- small enough to sit in cache and make
    the kernel look ~13% faster than it is. ``pool_blocks`` overrides the size
    to study cache residency deliberately (values below the demand re-introduce
    wrapping; the padded tail of each row stays block 0 either way).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    fp8_dtype = get_fp8_e4m3_dtype()

    max_model_len = 2 * kv_len

    if var_ratio == 0.0:
        context_lens = torch.full((batch,), kv_len, device="cuda", dtype=torch.int32)
    else:
        lo = max(1, int((1 - var_ratio) * kv_len))
        hi = int((1 + var_ratio) * kv_len) + 1
        context_lens = torch.randint(lo, hi, (batch,)).cuda().to(torch.int32)
    # decode with MTP needs at least next_n tokens of context.
    context_lens = torch.clamp(context_lens, min=next_n)

    # block table: ceil(ctx / block_size) physical blocks per sequence.
    blocks_per_seq = (context_lens.to(torch.int64) + block_size - 1) // block_size
    max_block_len = int(blocks_per_seq.max().item())
    needed_blocks = int(blocks_per_seq.sum().item())
    num_blocks = needed_blocks if pool_blocks <= 0 else max(pool_blocks, max_block_len)

    q = torch.randn((batch, next_n, heads, head_dim), dtype=torch.bfloat16)
    kv_cache = torch.randn((num_blocks, block_size, 1, head_dim), dtype=torch.bfloat16)
    weights = torch.randn((batch * next_n, heads), dtype=torch.float32)

    # Hand blocks out of a shuffled pool in sequence order (so a sequence's
    # blocks are scattered across the pool, as they are after real paged
    # allocation), padding each row's unused tail with block 0.
    pool = list(range(num_blocks))
    random.shuffle(pool)
    pool_t = torch.tensor(pool, device="cuda", dtype=torch.int32)
    starts = torch.cumsum(blocks_per_seq, 0) - blocks_per_seq
    col = torch.arange(max_block_len, device="cuda", dtype=torch.int64)
    in_row = col[None, :] < blocks_per_seq[:, None]
    draw = (starts[:, None] + col[None, :]) % num_blocks
    block_tables = torch.where(
        in_row, pool_t[draw], torch.zeros((), device="cuda", dtype=torch.int32)
    ).to(torch.int32)

    q_fp8 = q.to(q_dtype)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache, fp8_dtype)
    return (
        q,
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    # Single-config by design: profile exactly one kernel launch shape.
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--next-n", type=int, default=2)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument(
        "--kv-block-size", type=int, default=1, help="KVBlockSize (block-flat page)"
    )
    parser.add_argument(
        "--preshuffle",
        action="store_true",
        help="use the shuffle_weight(16x16) KV layout (needs kv-block-size %% 16 == 0)",
    )
    parser.add_argument("--q-dtype", type=str, default="fnuz", choices=["fnuz", "fn"])
    parser.add_argument(
        "--split-kv",
        type=int,
        default=0,
        help="0 == auto (host formula); else override",
    )
    parser.add_argument("--wave-per-eu", type=int, default=2)
    parser.add_argument("--chunk-k", type=int, default=128, help="KV tile width")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=f"paged_w<WPB> tag; None uses the kernel default ({DEFAULT_VARIANT})",
    )
    parser.add_argument("--var-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pool-blocks",
        type=int,
        default=0,
        help="physical blocks in the cache pool; 0 == one per block the batch "
        "needs (no sharing). Smaller values make sequences share blocks, "
        "shrinking the footprint into cache",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="kernel launches in the timed loop"
    )
    parser.add_argument(
        "--num-iters", type=int, default=101, help="run_perftest iters for --time"
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="also report device self-time (run_perftest)",
    )
    parser.add_argument(
        "--no-check", action="store_true", help="skip the one-shot correctness gate"
    )
    args = parser.parse_args()

    if get_gfx() not in ("gfx942", "gfx950"):
        print(f"unsupported gfx {get_gfx()}; skipping")
        return
    if args.preshuffle and args.kv_block_size % 16 != 0:
        raise SystemExit(
            f"--preshuffle requires --kv-block-size divisible by 16; "
            f"got {args.kv_block_size}"
        )

    (
        q,
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
    ) = build_inputs(
        args.batch,
        args.next_n,
        args.heads,
        args.head_dim,
        args.kv_len,
        args.kv_block_size,
        DTYPE_MAP[args.q_dtype],
        args.var_ratio,
        args.seed,
        args.pool_blocks,
    )

    # Oracle reads the unshuffled cache; the kernel gets the preshuffled copy.
    kv_kernel = (
        preshuffle_kv_data(kv_cache_fp8, args.head_dim)
        if args.preshuffle
        else kv_cache_fp8
    )

    out = torch.full(
        (args.batch * args.next_n, max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    split_kv = None if args.split_kv == 0 else args.split_kv

    def run():
        flydsl_fp8_paged_mqa_logits(
            q_fp8,
            kv_kernel,
            weights,
            out,
            context_lens,
            block_tables,
            max_model_len,
            Preshuffle=args.preshuffle,
            KVBlockSize=args.kv_block_size,
            ChunkK=args.chunk_k,
            SplitKV=split_kv,
            WavePerEU=args.wave_per_eu,
            variant=args.variant,
        )

    # Warm / JIT-compile once (excluded from the timed loop and from rocprofv3
    # runs that filter on the steady-state kernel name).
    run()
    torch.cuda.synchronize()

    if not args.no_check:
        ref = ref_fp8_paged_mqa_logits(
            q,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            fp8_dtype,
            block_size=args.kv_block_size,
        )
        neg_inf = float("-inf")
        ref_mask = ref == neg_inf
        diff = float(
            calc_diff(out.masked_fill(out == neg_inf, 0), ref.masked_fill(ref_mask, 0))
        )
        mask_ok = bool(torch.equal(out == neg_inf, ref_mask))
        ok = (diff < 1e-3) and mask_ok
        print(
            f"# correctness: calc_diff={diff:.3e} mask_ok={mask_ok} "
            f"-> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            raise SystemExit("correctness gate failed")

    # Work estimate: QK dot is 2*H*D MACs per (query row, valid KV position).
    total_ctx = int(context_lens.sum().item())
    flops = 2 * args.heads * args.head_dim * args.next_n * total_ctx
    # Bytes the kernel *requests*: every query row walks its whole context, so
    # the next_n rows of a sequence each re-read the same blocks.
    kv_bytes = args.next_n * total_ctx * (args.head_dim + 4)
    # Bytes the kernel *touches*: distinct blocks the table points at. Equal to
    # kv_bytes/next_n unless the pool is small enough that sequences share.
    n_blk_row = (context_lens.to(torch.int64) + args.kv_block_size - 1) // (
        args.kv_block_size
    )
    in_row = (
        torch.arange(block_tables.shape[1], device="cuda")[None, :] < n_blk_row[:, None]
    )
    distinct_blocks = int(torch.unique(block_tables[in_row]).numel())
    kv_unique = distinct_blocks * args.kv_block_size * (args.head_dim + 4)

    print(
        f"# gfx={get_gfx()} B={args.batch} nn={args.next_n} H={args.heads} "
        f"D={args.head_dim} kv_len={args.kv_len} kvb={args.kv_block_size} "
        f"preshuffle={args.preshuffle} chunk_k={args.chunk_k} "
        f"split_kv={args.split_kv or 'auto'} variant={args.variant or DEFAULT_VARIANT} "
        f"q_dtype={args.q_dtype} iters={args.iters}"
    )
    print(
        f"# KV footprint: {kv_unique / 1e6:.1f} MB touched "
        f"({distinct_blocks} blocks), {kv_bytes / 1e6:.1f} MB requested"
    )

    # Timed loop: back-to-back kernel launches -- this is what rocprofv3 traces.
    torch.cuda.synchronize()
    for _ in range(args.iters):
        run()
    torch.cuda.synchronize()

    if args.time:
        _, us = run_perftest(run, num_iters=args.num_iters)
        us = float(us)
        print(
            f"time: {us:.2f} us | {flops / us / 1e6:.1f} TFLOP/s | "
            f"{kv_bytes / us / 1e3:.1f} GB/s requested | "
            f"{kv_unique / us / 1e3:.1f} GB/s touched"
        )


if __name__ == "__main__":
    main()
