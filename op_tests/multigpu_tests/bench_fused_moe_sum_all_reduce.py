# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Perf: fused expert-sum + all-reduce vs baseline (moe_sum + all-reduce).

Both paths use the same custom all-reduce, so this isolates the win from
folding the topk expert-sum into the all-reduce input stage (dropping the
standalone moe_sum kernel + its intermediate + the AR input copy).

Real M3 MoE down-proj shapes: hidden=6144, topk=8.

Run (4 GPUs):
    HIP_VISIBLE_DEVICES="0,1,2,7" \
      python op_tests/multigpu_tests/bench_fused_moe_sum_all_reduce.py -t 4
"""

import argparse
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing import Optional

import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_moe_sum_all_reduce,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

set_start_method("spawn", force=True)


def _time_us(func, x, group, iters=200, warmup=40):
    """Steady-state per-call latency (us). Ranks are barrier-aligned before the
    timed window; iterations run back-to-back (no empty_cache) so collectives
    stay in lockstep and the number reflects real throughput."""
    for _ in range(warmup):
        func(x)
    torch.cuda.synchronize()
    if group is not None:
        dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        func(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters

TOPK = 8
HIDDEN = 6144
# decode -> prefill token counts
M_LIST = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]


def _worker(tp_size, pp_size, rank, dtype, distributed_init_method: Optional[str]):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size, rank=rank, distributed_init_method=distributed_init_method
    )
    ensure_model_parallel_initialized(tp_size, pp_size)

    def fused(x):
        return tensor_model_parallel_fused_moe_sum_all_reduce(x)

    def baseline(x):
        out_shape = x.shape[:-2] + x.shape[-1:]
        summed = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        aiter.moe_sum(x, summed)
        return tensor_model_parallel_all_reduce(summed)

    group = get_tp_group().device_group
    results = {}
    try:
        # warm up custom AR / pools with a mid-size input
        _ = fused(torch.randn((256, TOPK, HIDDEN), dtype=dtype, device=device) * 0.1)
        torch.cuda.synchronize()
        for m in M_LIST:
            x = torch.randn((m, TOPK, HIDDEN), dtype=dtype, device=device) * 0.1
            base_us = _time_us(baseline, x, group)
            fused_us = _time_us(fused, x, group)
            results[m] = (float(fused_us), float(base_us))
        return results
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tp", type=int, default=4)
    ap.add_argument("-p", "--pp", type=int, default=1)
    ap.add_argument("-d", "--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()
    dtype = dtypes.d_dtypes[args.dtype]

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49377"
    init_method = get_distributed_init_method(get_ip(), get_open_port())

    with Pool(processes=args.tp) as pool:
        rets = [
            pool.apply_async(
                _worker, args=(args.tp, args.pp, rank, dtype, init_method)
            )
            for rank in range(args.tp)
        ]
        rets = [r.get() for r in rets]

    # Aggregate across ranks: report the worst (max) rank latency per shape.
    rows = []
    for m in M_LIST:
        fused_us = max(r[m][0] for r in rets)
        base_us = max(r[m][1] for r in rets)
        speedup = base_us / fused_us if fused_us else float("nan")
        saved = base_us - fused_us
        rows.append(
            {
                "tp": args.tp,
                "m": m,
                "topk": TOPK,
                "n": HIDDEN,
                "dtype": args.dtype,
                "baseline_us": round(base_us, 2),
                "fused_us": round(fused_us, 2),
                "saved_us": round(saved, 2),
                "speedup": round(speedup, 3),
            }
        )

    try:
        import pandas as pd

        print(pd.DataFrame(rows).to_markdown(index=False))
    except ImportError:
        for r in rows:
            print(r)


if __name__ == "__main__":
    freeze_support()
    main()
