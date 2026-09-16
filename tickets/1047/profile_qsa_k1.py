# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Family A FlyDSL K1 vs live AMD select for rocprof (SILOTIGER-1047 2d).

Oracle is not run. Intended wrap::

    HIP_VISIBLE_DEVICES=6 rocprofv3 --kernel-trace --stats -f csv \\
      -d /tmp/qsa_k1_rocprof_8k -- python3 tickets/1047/profile_qsa_k1.py \\
      -b 1 -s 8192 --sleep 0.5
"""

from __future__ import annotations

import argparse
import time

import torch

from aiter import dtypes
from aiter.ops.flydsl.qsa import (
    FAMILY_A_INDEXER,
    pack_paged_cache,
    qsa_k1_family_a_block_ids,
)
from aiter.ops.triton.attention.qsa_vllm_amd import qsa_select_paged_tokens
from op_tests.test_flydsl_qsa import _query_positions


def _event_us(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1e3 / iters


def _build(m: int, seq_len: int, page_size: int, dtype):
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device).contiguous()
    slen = torch.full((1,), seq_len, dtype=torch.int32, device=device)
    token_to_req = torch.zeros(m, dtype=torch.int32, device=device)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(
        k_bar.unsqueeze(1), page_size, generator=gen_i
    )
    index_cache = index_cache.contiguous()
    index_table = index_table.contiguous()
    block_ids = torch.empty((m, idx.block_budget), dtype=torch.int32, device=device)
    token_out = torch.empty((m, idx.index_width), dtype=torch.int32, device=device)

    def k1():
        return qsa_k1_family_a_block_ids(
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
            out=block_ids,
        )

    def hip():
        return qsa_select_paged_tokens(
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
            idx.token_budget,
            idx.compress_ratio,
            out=token_out,
        )

    return k1, hip, n_blocks


def main():
    parser = argparse.ArgumentParser(description="Profile FlyDSL K1 vs HIP select")
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-s", "--seq", type=int, nargs="*", default=[8192, 32768])
    parser.add_argument("-p", "--page-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to sleep after warmup (lets rocprof drop compile)",
    )
    args = parser.parse_args()
    dtype = dtypes.bf16
    if not torch.cuda.is_available():
        raise SystemExit("K1 profile requires a GPU")

    print(
        f"gfx={torch.cuda.get_device_name(0)} m={args.batch} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print("seq_len n_blocks flydsl_k1_us hip_select_us")
    for seq_len in args.seq:
        if args.batch > seq_len:
            print(f"skip m={args.batch} seq_len={seq_len}")
            continue
        k1, hip, n_blocks = _build(args.batch, seq_len, args.page_size, dtype)
        k1_us = _event_us(k1, args.warmup, args.iters)
        hip_us = _event_us(hip, args.warmup, args.iters)
        if args.sleep:
            time.sleep(args.sleep)
        print(f"{seq_len} {n_blocks} {k1_us:.3f} {hip_us:.3f}")


if __name__ == "__main__":
    main()
