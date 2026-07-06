#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lean bench/PMC harness for flydsl_pa_mqa_logits_fp4_prefill VALU tuning.

Builds the bench-case inputs once and launches ONLY the kernel repeatedly
(no torch reference / no ATOM path), so it is cheap enough for back-to-back
wall-time comparisons and for rocprofv3 PMC passes (SQ_INSTS_VALU etc).

Wall time:
    python -m op_tests.bench_flydsl_pa_mqa_logits_fp4_prefill_valu \
        --iters 50 --warmup 15 --reps 4

VALU instruction count (per dispatch), isolated to the kernel:
    printf 'pmc: SQ_INSTS_VALU SQ_INSTS_VALU_MFMA_F6F4 SQ_WAVES\n' > /tmp/pmc.txt
    rocprofv3 -i /tmp/pmc.txt \
        --kernel-include-regex 'pa_mqa_logits_fp4_prefill.*' \
        -d /tmp/rpvalu -o v --output-format csv -- \
        python -m op_tests.bench_flydsl_pa_mqa_logits_fp4_prefill_valu \
            --iters 4 --warmup 2 --reps 1
"""

import argparse

import torch

from op_tests.test_flydsl_pa_mqa_logits_fp4_prefill import (
    indexer_k_fp4_paged_preshuffle,
    quant_q_fp4_preshuffle,
)

dev = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--n_q", type=int, default=8000)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()

    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_prefill
    from aiter.ops.flydsl.kernels.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
    )

    bs, heads, head_dim = args.bs, args.heads, args.head_dim
    kv_block_size, block_k = 64, 256
    ctx = max(kv_block_size, (args.ctx // kv_block_size) * kv_block_size)
    windows = [[ctx] * args.n_q for _ in range(bs)]

    torch.manual_seed(7)
    max_end = ctx
    max_blocks_per_seq = max(
        (max_end + block_k - 1) // block_k * (block_k // kv_block_size),
        block_k // kv_block_size,
    )
    t_max = max_blocks_per_seq * kv_block_size
    max_seq_len = t_max
    num_blocks = max_blocks_per_seq * bs

    kv_bf16 = torch.randn(bs, t_max, head_dim, dtype=torch.bfloat16, device=dev)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        bs, max_blocks_per_seq
    )
    k_flat = kv_bf16.reshape(bs * t_max, head_dim)
    tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
    tt = torch.arange(t_max, device=dev).repeat(bs)
    phys = block_tables[tb, tt // kv_block_size].long()
    slot_mapping = (phys * kv_block_size + (tt % kv_block_size)).to(torch.int32)
    k_tiles = head_dim // 128
    kv_cache = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, 16, dtype=torch.uint8, device=dev
    )
    kv_scale = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev
    )
    indexer_k_fp4_paged_preshuffle(
        k_flat, slot_mapping, kv_cache, kv_scale, kv_block_size
    )

    rb, ls, le = [], [], []
    for b in range(bs):
        for w in windows[b]:
            rb.append(b)
            ls.append(0)
            le.append(w)
    total_tokens = len(rb)
    parallel_unit_num = max(512, total_tokens)
    row_to_batch = torch.tensor(rb, dtype=torch.int32, device=dev)
    local_starts = torch.tensor(ls, dtype=torch.int32, device=dev)
    local_ends = torch.tensor(le, dtype=torch.int32, device=dev)

    q_bf16 = torch.randn(
        total_tokens, heads, head_dim, dtype=torch.bfloat16, device=dev
    )
    weights = (
        torch.randn(total_tokens, heads, dtype=torch.float32, device=dev) * 0.1
    ).to(torch.bfloat16)
    weight_scale = 1.5
    q_fp4, q_scale = quant_q_fp4_preshuffle(q_bf16)

    _, cta_info, n_ctas = compute_prefill_schedule(
        row_to_batch, local_starts, local_ends, block_k, parallel_unit_num, max_seq_len
    )
    out = torch.full(
        (total_tokens, max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )

    def run():
        flydsl_pa_mqa_logits_fp4_prefill(
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            row_to_batch,
            local_starts,
            local_ends,
            max_seq_len,
            weight_scale=weight_scale,
            block_k=block_k,
            kv_block_size=kv_block_size,
            parallel_unit_num=parallel_unit_num,
            out=out,
            cta_info=cta_info,
            n_ctas=n_ctas,
        )

    print(
        f"total_tokens={total_tokens} n_ctas={n_ctas} ctx={ctx} "
        f"max_seq_len={max_seq_len}"
    )
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    best = 1e18
    for _ in range(args.reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run()
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) / args.iters * 1000
        print(f"  rep us = {us:.2f}")
        best = min(best, us)
    print(f"kernel best = {best:.2f} us")


if __name__ == "__main__":
    main()
