# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel time of fused_gdn_verify_qkvz, the Gated DeltaNet speculative-verify step on
ReplaySSM state (gfx1250), in us per call. The calls rotate over copies of the state
pools, as the layers of a model do, so that no call reads the state from the cache.

    python3 bench_fused_gdn_verify_qkvz.py
    # Qwen3.5-35B-A3B with DP attention: 16 key heads, 32 value heads
    python3 bench_fused_gdn_verify_qkvz.py --num-k-heads 16 --batch 1,16,64,192
"""

import pandas as pd
import torch

from aiter import logger
from aiter.ops.triton.gated_delta_net.fused_gdn_verify_qkvz import (
    fused_gdn_verify_qkvz,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.test_common import run_perftest
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
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


def bench_one(num_k_heads, ratio, seqs, window, cap, rotate):
    layer = make_layer(num_k_heads, num_k_heads * ratio, seed=1)
    num_slots = 2 * seqs + 16
    pools = make_pools(layer, window, cap, seed=2, num_slots=num_slots)
    step = make_step(layer, seqs, window, seed=3, num_slots=num_slots)
    commit(pools, step, cap)
    out = torch.empty((step.n, layer.vh, V), dtype=torch.bfloat16, device="cuda")
    args = op_args(layer, step, pools, out)
    kwargs = dict(max_query_len=window, scale=K**-0.5, norm_eps=EPS, **args)
    # The first timing of new inputs is too high. Run once and drop it.
    run_perftest(fused_gdn_verify_qkvz, **kwargs, num_rotate_args=rotate)
    _, us = run_perftest(fused_gdn_verify_qkvz, **kwargs, num_rotate_args=rotate)
    return us


def main():
    parser = get_parser(kernel_name="Fused GDN Verify QKVZ")
    parser.add_argument("--batch", default="1,4,16,64", help="sequences per call")
    parser.add_argument("--num-k-heads", type=int, default=4)
    parser.add_argument("--ratio", type=int, default=2, help="value heads per key head")
    parser.add_argument(
        "--window", type=int, default=4, help="verify window: draft tokens + 1"
    )
    parser.add_argument("--cache-len", type=int, default=16, help="records per slot")
    parser.add_argument(
        "--rotate", type=int, default=30, help="copies of the state pools"
    )
    args = parser.parse_args()
    if get_arch() != "gfx1250":
        logger.warning("fused_gdn_verify_qkvz runs on gfx1250 only, got %s", get_arch())
        return
    results = []
    for seqs in (int(b) for b in args.batch.split(",")):
        us = bench_one(
            args.num_k_heads, args.ratio, seqs, args.window, args.cache_len, args.rotate
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
