# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import torch
import torch.distributed as dist
from typing import Optional
import argparse
import pandas as pd
from aiter import dtypes

from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip
from aiter.dist.communication_op import tensor_model_parallel_reduce_scatter
from aiter.test_common import perftest
from multiprocessing import set_start_method, Pool, freeze_support
import logging

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def reduce_scatter(
    tp_size,
    pp_size,
    rankID,
    x,
    dim=0,
    use_custom=False,
    distributed_init_method: Optional[str] = None,
):
    """Per-rank worker. Runs reduce_scatter on x with the given dim and
    returns (output, per-call latency in us)."""
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)

    # warmup + barrier so the timing on first call isn't polluted.
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    @perftest()
    def run_ca(x):
        return tensor_model_parallel_reduce_scatter(x, use_custom=use_custom, dim=dim)

    out = run_ca(x)

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def _build_input(shape, dtype, tp_size, rand_seed):
    """Deterministic per-rank input: rand_seed[i] repeats over a chunk so
    each rank ends up with an identical tensor of shape `shape`. With all
    ranks having identical input, sum_across_ranks = tp_size * input — that
    gives us an analytic reference for any scatter dim (see _ref_output)."""
    n = 1
    for s in shape:
        n *= s
    chunk_size = n // tp_size
    return rand_seed.repeat_interleave(chunk_size).reshape(shape).to(dtype).contiguous()


def _ref_output(input_tensor, dim, rank, tp_size):
    """Analytic reference for one rank's reduce_scatter output. Computed in
    fp32 to avoid bf16 accumulation noise on the multiply."""
    ndim = input_tensor.dim()
    if dim < 0:
        dim += ndim
    full_sum = tp_size * input_tensor.float()
    chunk = input_tensor.shape[dim] // tp_size
    out = full_sum.narrow(dim, rank * chunk, chunk).contiguous()
    return out.to(input_tensor.dtype)


def run_reduce_scatter_parallel(
    tp_size,
    pp_size,
    shape,
    dim,
    dtype,
    rand_seed,
    use_custom,
    distributed_init_method,
):
    """Spawn tp_size processes, each running one reduce_scatter call.
    Returns list of (out, us) per rank."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = []
    for i in range(tp_size):
        x = _build_input(shape, dtype, tp_size, rand_seed)
        rets.append(
            pool.apply_async(
                reduce_scatter,
                args=(tp_size, pp_size, i, x, dim, use_custom, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    return [el.get() for el in rets]


def run_case(label, shape, dim, dtype, tp_size, init_method_factory):
    """End-to-end one case: spawn the custom run, compute accuracy against
    the analytic PyTorch reference, collect latency. Returns one row for
    the summary table.

    No external-library comparison — other libs (torch.distributed /
    pynccl) don't support scatter on non-zero dims, so latency-vs-them
    isn't meaningful for the new kernels."""
    rand_seed = torch.randint(1, 16, (tp_size,), dtype=dtype, device="cuda")

    custom_rets = run_reduce_scatter_parallel(
        tp_size,
        1,
        shape,
        dim,
        dtype,
        rand_seed,
        True,
        init_method_factory(),
    )

    # Analytic reference vs each rank's output.
    ref_input = _build_input(shape, dtype, tp_size, rand_seed)
    max_err = 0.0
    mean_err = 0.0
    for rank, (out, _us) in enumerate(custom_rets):
        ref = _ref_output(ref_input, dim, rank, tp_size).cpu()
        diff = (out.cpu().float() - ref.float()).abs()
        max_err = max(max_err, diff.max().item())
        # Use max-over-ranks for mean too, so a single bad rank shows up.
        mean_err = max(mean_err, diff.mean().item())
    custom_us = [us for _, us in custom_rets]

    return {
        "case": label,
        "shape": str(tuple(shape)),
        "dim": dim,
        "dtype": str(dtype).split(".")[-1],
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "min_us": min(custom_us),
        "max_us": max(custom_us),
    }


# Cases designed for tp_size=8 and bf16 (pack_size = 16 // 2 = 8).
# Each case targets a specific kernel branch in dispatchReduceScatter:
#
#   first_dim_vec  : numel % (ngpus * pack_size) == 0  → split_first_dim
#   last_dim_vec   : last % (ngpus * pack_size) == 0   → split_lastdim (vec)
#   last_dim_naive : last % ngpus == 0 but % pack != 0 → split_lastdim_naive
#   mid_dim_vec    : k % pack_size == 0                → split_middim (vec)
#   mid_dim_naive  : k % pack_size != 0                → split_middim_naive
#
# All five satisfy shape[dim] % ngpus == 0 (Python wrapper's hard requirement).
l_cases = [
    ("first_dim_vec", (512, 1024), 0),
    ("last_dim_vec", (256, 512), -1),
    ("last_dim_naive", (256, 40), -1),
    ("mid_dim_vec", (16, 64, 64), 1),
    ("mid_dim_naive", (16, 64, 5), 1),
]

l_dtype = ["bf16"]

parser = argparse.ArgumentParser(description="reduce_scatter accuracy + latency test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    default=None,
    help="data type",
)
parser.add_argument(
    "-c",
    "--case",
    type=str,
    default=None,
    help="run only one case by label, e.g. mid_dim_naive",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        dtypes_to_run = [dtypes.d_dtypes[k] for k in l_dtype]
    else:
        dtypes_to_run = [dtypes.d_dtypes[args.dtype]]
    if args.case is None:
        cases_to_run = l_cases
    else:
        cases_to_run = [c for c in l_cases if c[0] == args.case]
        assert cases_to_run, f"no case named {args.case!r}"

    tp_size = 8

    def init_method_factory():
        return get_distributed_init_method(get_ip(), get_open_port())

    rows = []
    for dtype in dtypes_to_run:
        for label, shape, dim in cases_to_run:
            print(f"\n=== {label}  shape={shape}  dim={dim}  dtype={dtype} ===")
            row = run_case(label, shape, dim, dtype, tp_size, init_method_factory)
            print(
                f"  max_abs_err={row['max_abs_err']:.4g}  "
                f"mean_abs_err={row['mean_abs_err']:.4g}  "
                f"latency={row['min_us']:.2f}-{row['max_us']:.2f}us"
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    print("\n=== reduce_scatter summary ===")
    print(df.to_markdown(index=False, floatfmt=".4g"))
