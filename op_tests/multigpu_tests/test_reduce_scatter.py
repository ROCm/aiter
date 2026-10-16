# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_reduce_scatter
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def barrier_before_teardown():
    """Align all ranks before tearing down the distributed groups.

    Drain this rank's GPU work, then join a barrier so no rank starts freeing
    IPC buffers / destroying process groups while a peer is still inside a
    NCCL / custom-all-reduce collective -- that race intermittently hangs when
    these comm UTs run back-to-back in CI. No-op if dist is uninitialized.
    """
    if not dist.is_initialized():
        return
    torch.cuda.synchronize()
    get_tp_group().barrier()
    torch.cuda.synchronize()


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


def _run_reduce_scatter_case(rankID, tp_size, case_idx, shape, dim, dtype):
    """Run one reduce_scatter case on an initialized rank and compare its
    output against the analytic reference.

    Every rank draws the same ``rand_seed`` from the ``case_idx`` seed, so
    each builds the identical input and its own reference slice locally;
    only the error scalars and latency go back to the parent."""
    gen = torch.Generator(device="cuda").manual_seed(case_idx)
    rand_seed = torch.randint(
        1, 16, (tp_size,), dtype=dtype, device="cuda", generator=gen
    )
    x = _build_input(shape, dtype, tp_size, rand_seed)
    ref = _ref_output(x, dim, rankID, tp_size)

    @perftest()
    def run_ca(x):
        return tensor_model_parallel_reduce_scatter(x, use_custom=True, dim=dim)

    out, us = run_ca(x)
    diff = (out.float() - ref.float()).abs()
    return {
        "us": us,
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
    }


def reduce_scatter_sweep(
    tp_size,
    pp_size,
    rankID,
    cases,
    dtype,
    distributed_init_method: str | None = None,
    force_fallback=False,
):
    """Per-rank worker. Runs every ``(label, shape, dim)`` case inside a
    single distributed init and returns one result dict per case.

    Setting up the TP group dominates a single case (seconds vs. microseconds
    of kernel time), so the group is created and torn down once. Results are
    returned rather than asserted here: every rank must walk the full case
    list so the collectives stay aligned across ranks.

    force_fallback: set AITER_CUSTOM_AR_MAX_SIZE=0 so the custom kernel is
    disabled and every reduce_scatter takes the pynccl fallback path in
    CudaCommunicator.reduce_scatter. This exercises the non-zero-dim fallback
    that used to mis-lay-out its result (movedim direction + discarded
    reshape/movedim); must be set before the group's CustomAllreduce is built,
    so custom and fallback cases cannot share one init."""
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    if force_fallback:
        os.environ["AITER_CUSTOM_AR_MAX_SIZE"] = "0"
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)

    # warmup + barrier so the timing on first call isn't polluted.
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    results = [
        _run_reduce_scatter_case(rankID, tp_size, case_idx, shape, dim, dtype)
        for case_idx, (_label, shape, dim) in enumerate(cases)
    ]

    if dist.is_initialized():
        barrier_before_teardown()
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return results


def run_suite(cases, dtype, tp_size, force_fallback=False):
    """End-to-end one suite: spawn tp_size processes that share one TP group
    and run every case, then fold the per-rank accuracy against the analytic
    PyTorch reference and the latency into one summary row per case.

    force_fallback routes every reduce_scatter through the pynccl fallback
    (custom AR disabled) instead of the custom kernel.

    No external-library comparison — other libs (torch.distributed /
    pynccl) don't support scatter on non-zero dims, so latency-vs-them
    isn't meaningful for the new kernels."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                reduce_scatter_sweep,
                args=(tp_size, 1, rank, cases, dtype, init_method, force_fallback),
            )
            for rank in range(tp_size)
        ]
        per_rank = [el.get() for el in rets]

    rows = []
    for i, (label, shape, dim) in enumerate(cases):
        rank_results = [results[i] for results in per_rank]
        custom_us = [r["us"] for r in rank_results]
        rows.append(
            {
                "case": label,
                "path": "fallback" if force_fallback else "custom",
                "shape": str(tuple(shape)),
                "dim": dim,
                "dtype": str(dtype).split(".")[-1],
                "max_abs_err": max(r["max_abs_err"] for r in rank_results),
                # Use max-over-ranks for mean too, so a single bad rank shows up.
                "mean_abs_err": max(r["mean_abs_err"] for r in rank_results),
                "min_us": min(custom_us),
                "max_us": max(custom_us),
            }
        )
    return rows


def build_cases(tp_size, dtype):
    """Build test cases that target each kernel branch in dispatchReduceScatter.

    Each case is designed so that shape[dim] % tp_size == 0 (hard requirement)
    and the vectorisation / naive path is selected by the alignment of the
    last dimension with pack_size.

    Kernel branches:
      first_dim_vec  : numel % (ngpus * pack_size) == 0  → split_first_dim
      last_dim_vec   : last % (ngpus * pack_size) == 0   → split_lastdim (vec)
      last_dim_naive : last % ngpus == 0 but % pack != 0 → split_lastdim_naive
      mid_dim_vec    : k % pack_size == 0                → split_middim (vec)
      mid_dim_naive  : k % pack_size != 0                → split_middim_naive
    """
    pack_size = 16 // dtype.itemsize

    # first_dim_vec: 2-D, scatter on dim 0
    #   shape[0] % tp_size == 0, numel % (tp_size * pack_size) == 0
    first_rows = 64 * tp_size
    first_cols = pack_size * tp_size
    assert (first_rows * first_cols) % (tp_size * pack_size) == 0

    # last_dim_vec: 2-D, scatter on last dim
    #   shape[-1] % (tp_size * pack_size) == 0
    last_vec_cols = pack_size * tp_size
    last_vec_rows = 256

    # last_dim_naive: 2-D, scatter on last dim
    #   shape[-1] % tp_size == 0 BUT shape[-1] % (tp_size * pack_size) != 0
    last_naive_cols = tp_size * (pack_size - 1) if pack_size > 1 else tp_size
    if last_naive_cols % (tp_size * pack_size) == 0:
        last_naive_cols = tp_size
    last_naive_rows = 256

    # mid_dim_vec: 3-D, scatter on dim 1
    #   shape[1] % tp_size == 0, k (last dim) % pack_size == 0
    mid_vec_m = 16
    mid_vec_n = 8 * tp_size
    mid_vec_k = pack_size * 8

    # mid_dim_naive: 3-D, scatter on dim 1
    #   shape[1] % tp_size == 0, k % pack_size != 0
    mid_naive_m = 16
    mid_naive_n = 8 * tp_size
    mid_naive_k = pack_size + 1 if pack_size > 1 else 3
    if mid_naive_k % pack_size == 0:
        mid_naive_k += 1

    return [
        ("first_dim_vec", (first_rows, first_cols), 0),
        ("last_dim_vec", (last_vec_rows, last_vec_cols), -1),
        ("last_dim_naive", (last_naive_rows, last_naive_cols), -1),
        ("mid_dim_vec", (mid_vec_m, mid_vec_n, mid_vec_k), 1),
        ("mid_dim_naive", (mid_naive_m, mid_naive_n, mid_naive_k), 1),
    ]


def build_fallback_cases(tp_size, dtype):
    """Cases that exercise the pynccl fallback (custom AR disabled), covering
    every scatter axis. dim=1 and dim=2 are the regression targets: the old
    fallback used the wrong movedim direction and discarded its reshape/movedim
    results, so it returned a transposed (garbage) tensor for non-zero dims.

    Only the requirement shape[dim] % tp_size == 0 matters here (no pack/vec
    alignment gates on the fallback), so keep the shapes small."""
    return [
        ("fallback_dim0", (4 * tp_size, 8, 6), 0),
        ("fallback_dim1_mid", (5, 4 * tp_size, 6), 1),
        ("fallback_dim2_last", (5, 8, 4 * tp_size), 2),
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
parser.add_argument(
    "-t",
    "--tp_size",
    type=int,
    choices=[2, 4, 8],
    default=8,
    help="tensor-parallel world size (default: 8)",
)
parser.add_argument(
    "-s",
    "--suite",
    type=str,
    choices=["custom", "fallback", "all"],
    default="all",
    help="which kernel path to test: custom kernel, pynccl fallback, or both",
)

# The fallback path uses int-valued bf16 inputs whose reduced sum is exact, so a
# correct result matches the reference to the bit; any nonzero error means the
# non-zero-dim fallback mis-laid-out its output (the bug this guards against).
FALLBACK_TOL = 1e-6


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        dtypes_to_run = [dtypes.d_dtypes[k] for k in l_dtype]
    else:
        dtypes_to_run = [dtypes.d_dtypes[args.dtype]]

    tp_size = args.tp_size

    rows = []
    failures = []
    for dtype in dtypes_to_run:
        # (case-list, force_fallback) per selected suite; one TP group each.
        suites = []
        if args.suite in ("custom", "all"):
            suites.append((build_cases(tp_size, dtype), False))
        if args.suite in ("fallback", "all"):
            suites.append((build_fallback_cases(tp_size, dtype), True))

        for all_cases, force_fallback in suites:
            if args.case is None:
                cases_to_run = all_cases
            else:
                cases_to_run = [c for c in all_cases if c[0] == args.case]
            if not cases_to_run:
                continue
            path = "fallback" if force_fallback else "custom"
            suite_rows = run_suite(cases_to_run, dtype, tp_size, force_fallback)
            for (label, shape, dim), row in zip(cases_to_run, suite_rows):
                print(
                    f"\n=== [{path}] {label}  shape={shape}  dim={dim}  "
                    f"dtype={dtype}  tp={tp_size} ==="
                )
                print(
                    f"  max_abs_err={row['max_abs_err']:.4g}  "
                    f"mean_abs_err={row['mean_abs_err']:.4g}  "
                    f"latency={row['min_us']:.2f}-{row['max_us']:.2f}us"
                )
                rows.append(row)
                # Fallback cases have an exact reference -> any error is a bug.
                if force_fallback and row["max_abs_err"] > FALLBACK_TOL:
                    failures.append(
                        f"{label} (dim={dim}): max_abs_err={row['max_abs_err']:.4g}"
                    )

    df = pd.DataFrame(rows)
    print("\n=== reduce_scatter summary ===")
    print(df.to_markdown(index=False, floatfmt=".4g"))

    assert not failures, (
        "reduce_scatter fallback produced wrong results (non-zero-dim layout bug):\n  "
        + "\n  ".join(failures)
    )
