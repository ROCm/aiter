# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel time of fused_gdn_verify_qkvz, the Gated DeltaNet speculative-verify step on
ReplaySSM state (gfx1250), in us per call. RotatingBufferBench replays the calls in a
CUDA graph over a pool of input copies, so that no call reads the state from the cache.

    python3 bench_fused_gdn_verify_qkvz.py
    # Qwen3.5-35B-A3B with DP attention: 16 key heads, 32 value heads
    python3 bench_fused_gdn_verify_qkvz.py --num-k-heads 16 --batch 1,16,64,192
"""

import functools

import pandas as pd
import torch

from aiter import logger
from aiter.ops.triton.gated_delta_net.fused_gdn_verify_qkvz import (
    fused_gdn_verify_qkvz,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.gated_delta_net.test_fused_gdn_verify_qkvz import (
    EPS,
    K,
    V,
    commit,
    make_layer,
    make_pools,
    make_step,
    op_args,
)

# The tensor arguments of one call. The layers have no conv bias, so conv_bias is a
# fixed None argument.
_KEYS = [
    "projected_qkvz",
    "projected_ba",
    "conv_state",
    "conv_state_indices",
    "num_accepted_tokens",
    "query_start_loc",
    "checkpoint",
    "record_k",
    "record_u",
    "record_g",
    "write_pos",
    "slot_idx",
    "conv_weight",
    "A_log",
    "dt_bias",
    "norm_weight",
    "out",
]


def make_inputs(num_k_heads, ratio, seqs, window, cap):
    """One call: a layer, a verify step and the state pools after the commit. Each
    sequence has its own slot and the pools have no other slots, so the pool size
    is the data that the call reads."""
    layer = make_layer(num_k_heads, num_k_heads * ratio, seed=1)
    pools = make_pools(layer, window, cap, seed=2, num_slots=seqs)
    step = make_step(layer, seqs, window, seed=3, num_slots=seqs)
    commit(pools, step, cap)
    out = torch.empty((step.n, layer.vh, V), dtype=torch.bfloat16, device="cuda")
    args = op_args(layer, step, pools, out)
    return tuple(args[key] for key in _KEYS)


def bench_one(bencher, num_k_heads, ratio, seqs, window, cap):
    gen = functools.partial(make_inputs, num_k_heads, ratio, seqs, window, cap)
    bencher.build_pool([gen], [_KEYS])
    op = functools.partial(
        fused_gdn_verify_qkvz,
        conv_bias=None,
        max_query_len=window,
        scale=K**-0.5,
        norm_eps=EPS,
    )
    return bencher.bench(op) * 1000.0  # ms to us


def main():
    parser = get_parser(kernel_name="Fused GDN Verify QKVZ")
    parser.add_argument("--batch", default="1,4,16,64", help="sequences per call")
    parser.add_argument("--num-k-heads", type=int, default=4)
    parser.add_argument("--ratio", type=int, default=2, help="value heads per key head")
    parser.add_argument(
        "--window", type=int, default=4, help="verify window: draft tokens + 1"
    )
    parser.add_argument("--cache-len", type=int, default=16, help="records per slot")
    args = parser.parse_args()
    if get_arch() != "gfx1250":
        logger.warning("fused_gdn_verify_qkvz runs on gfx1250 only, got %s", get_arch())
        return
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    results = []
    for seqs in (int(b) for b in args.batch.split(",")):
        us = bench_one(
            bencher, args.num_k_heads, args.ratio, seqs, args.window, args.cache_len
        )
        results.append({"seqs": seqs, "us": round(us, 2)})
    logger.info(
        "fused_gdn_verify_qkvz, %d key heads, ratio %d, window %d, %d records (us):\n%s",
        args.num_k_heads,
        args.ratio,
        args.window,
        args.cache_len,
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
