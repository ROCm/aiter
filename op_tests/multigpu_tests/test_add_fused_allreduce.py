# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy + latency benchmark for the fused add-then-allreduce path.

    out = allreduce_over_ranks(a_i + b_i)

Two paths are each compared against a CPU fp32 reference (the analytic sum over
ranks of a_i + b_i, cast down to the kernel dtype):
  - fused   : ca_comm.add_fused_allreduce(a, b)   (auto-dispatched kernel)
  - unfused : torch.add(a, b) -> ca_comm.custom_all_reduce(...)
The accuracy table reports the mismatch ratio of each path vs the reference.

Just run it — no arguments. It runs both stages and prints all results at the
end:
  accuracy : fp32-reference mismatch ratio for fused + unfused on named shapes.
  bench    : eager + cuda-graph latency sweep over element counts, one table
             per world size (tp=2, 4, 8).

    python test_add_fused_allreduce.py
"""

import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

KB = 1024
MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Shared per-rank setup
# ---------------------------------------------------------------------------


def _setup_rank(tp_size, pp_size, rankID, distributed_init_method):
    """Init distributed env for this rank and return (ca_comm, device, group)."""
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)

    ca_comm = get_tp_group().device_communicator.ca_comm
    assert ca_comm is not None and not ca_comm.disabled, "custom allreduce unavailable"

    # warmup + barrier so first-call timing isn't polluted.
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()
    return ca_comm, device, group


def _teardown():
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Accuracy mode
# ---------------------------------------------------------------------------


def accuracy_worker(
    tp_size,
    pp_size,
    rankID,
    a,
    b,
    distributed_init_method: Optional[str] = None,
):
    """Per-rank worker. Runs the fused (auto-dispatched) and unfused
    (add + allreduce) paths, returning both outputs for comparison against the
    CPU fp32 reference."""
    ca_comm, device, group = _setup_rank(
        tp_size, pp_size, rankID, distributed_init_method
    )
    a = a.to(device)
    b = b.to(device)

    out_fused = ca_comm.add_fused_allreduce(a, b)
    out_unfused = ca_comm.custom_all_reduce(torch.add(a, b))

    result = {
        "out_fused": out_fused.cpu(),
        "out_unfused": out_unfused.cpu(),
    }
    _teardown()
    return result


def run_accuracy_parallel(tp_size, pp_size, a_list, b_list, distributed_init_method):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = []
    for i in range(tp_size):
        rets.append(
            pool.apply_async(
                accuracy_worker,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    a_list[i],
                    b_list[i],
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    return [el.get() for el in rets]


def run_accuracy_case(label, shape, dtype, tp_size, init_method_factory):
    # Distinct per-rank inputs so the reduction is non-trivial.
    a_list = [torch.randn(shape, dtype=dtype) for _ in range(tp_size)]
    b_list = [torch.randn(shape, dtype=dtype) for _ in range(tp_size)]

    # Analytic fp32 reference: sum over ranks of (a_i + b_i).
    ref = torch.zeros(shape, dtype=torch.float32)
    for a, b in zip(a_list, b_list):
        ref += a.float() + b.float()

    rets = run_accuracy_parallel(tp_size, 1, a_list, b_list, init_method_factory())

    # Compare in fp32: cast the analytic ref down to the kernel dtype and back
    # so rounding matches, then both sides are float32.
    ref_cast = ref.to(dtype).float()

    # Worst rank bounds accuracy: track the largest mismatch ratio against the
    # CPU fp32 reference across all ranks, for each path.
    max_err_fused = 0.0
    max_err_unfused = 0.0
    for rank, r in enumerate(rets):
        e_f = checkAllclose(
            ref_cast,
            r["out_fused"].float(),
            msg=f"{label} rank{rank} fused-vs-ref",
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
        )
        e_u = checkAllclose(
            ref_cast,
            r["out_unfused"].float(),
            msg=f"{label} rank{rank} unfused-vs-ref",
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
        )
        max_err_fused = max(max_err_fused, e_f)
        max_err_unfused = max(max_err_unfused, e_u)

    return {
        "case": label,
        "shape": str(tuple(shape)),
        "dtype": str(dtype).split(".")[-1],
        f"add_fused_allreduce_err_ratio(atol={ACCURACY_ATOL},rtol={ACCURACY_RTOL})": max_err_fused,
        f"add+custom_all_reduce_err_ratio(atol={ACCURACY_ATOL},rtol={ACCURACY_RTOL})": max_err_unfused,
    }


# pack_size for bf16/fp16 = 16//2 = 8; with tp_size=8 the divisibility
# requirement is numel % (8 * 8) == 0. All shapes below satisfy it.
ACCURACY_CASES = [
    ("small", (2, 7168)),
    ("medium", (128, 8192)),
    ("large", (512, 8192)),
]

ACCURACY_DTYPES = ["fp16", "bf16"]
ACCURACY_TP = 8
ACCURACY_ATOL = 1e-2
ACCURACY_RTOL = 1e-2


def run_accuracy():
    """Run all accuracy cases and return a summary DataFrame."""
    dtypes_to_run = [dtypes.d_dtypes[k] for k in ACCURACY_DTYPES]

    def init_method_factory():
        return get_distributed_init_method(get_ip(), get_open_port())

    rows = []
    for dtype in dtypes_to_run:
        for label, shape in ACCURACY_CASES:
            row = run_accuracy_case(
                label, shape, dtype, ACCURACY_TP, init_method_factory
            )
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bench mode
# ---------------------------------------------------------------------------

ELEM_NUMS = [
    1 * KB,
    2 * KB,
    4 * KB,
    8 * KB,
    16 * KB,
    32 * KB,
    64 * KB,
    128 * KB,
    256 * KB,
    512 * KB,
    1 * MB,
    2 * MB,
    4 * MB,
    8 * MB,
    16 * MB,
    32 * MB,
    64 * MB,
]

BENCH_COLUMNS = [
    "eager_fused",
    "eager_unfused",
    "graph_fused",
    "graph_unfused",
]


def elem_label(n):
    if n % MB == 0:
        return f"{n // MB}m"
    return f"{n // KB}k"


def make_shape(n):
    if n < 8 * KB:
        return (n,)
    assert n % 8192 == 0, f"elem_num {n} not divisible by 8192"
    return (n // 8192, 8192)


def _bench_eager(fn):
    """Return kernel latency (us) of `fn` measured in eager mode."""

    @perftest(num_rotate_args=1)
    def _run():
        return fn()

    _, us = _run()
    return us


def _bench_graph(fn):
    """Capture `fn` into a cuda graph and return replay latency (us)."""
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        with torch.cuda.graph(graph, stream=gc.stream):
            fn()

    @perftest(num_rotate_args=1)
    def _replay():
        graph.replay()

    _, us = _replay()
    return us


def _safe(bench_fn, call):
    try:
        return bench_fn(call)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"bench failed: {e}")
        return float("nan")


def bench_worker(
    tp_size,
    pp_size,
    rankID,
    elem_nums,
    dtype,
    distributed_init_method: Optional[str] = None,
):
    ca_comm, device, group = _setup_rank(
        tp_size, pp_size, rankID, distributed_init_method
    )

    results = {}
    for n in elem_nums:
        shape = make_shape(n)
        a = torch.randn(shape, dtype=dtype, device=device)
        b = torch.randn(shape, dtype=dtype, device=device)
        out_f = torch.empty(shape, dtype=dtype, device=device)

        # unfused custom_all_reduce has a size cap (bf16: <= 32M elems); skip
        # (record NaN) when the shape doesn't fit the custom kernel.
        unfused_ok = ca_comm.should_custom_ar(out_f)

        call_fused = lambda: ca_comm.add_fused_allreduce(a, b, out=out_f)  # noqa: E731
        call_unf = lambda: ca_comm.custom_all_reduce(torch.add(a, b))  # noqa: E731

        row = {
            "eager_fused": _safe(_bench_eager, call_fused),
            "eager_unfused": (
                _safe(_bench_eager, call_unf) if unfused_ok else float("nan")
            ),
            "graph_fused": _safe(_bench_graph, call_fused),
            "graph_unfused": (
                _safe(_bench_graph, call_unf) if unfused_ok else float("nan")
            ),
        }
        results[n] = row

        dist.barrier(group=group)
        a = b = out_f = None
        torch.cuda.empty_cache()

    _teardown()
    return results


def run_bench_parallel(tp_size, pp_size, elem_nums, dtype, distributed_init_method):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = []
    for i in range(tp_size):
        rets.append(
            pool.apply_async(
                bench_worker,
                args=(tp_size, pp_size, i, elem_nums, dtype, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    return [el.get() for el in rets]


def bench_one_tp(tp_size, dtype):
    """Run the benchmark for a single world size and return a formatted DataFrame."""

    def init_method_factory():
        return get_distributed_init_method(get_ip(), get_open_port())

    rets = run_bench_parallel(tp_size, 1, ELEM_NUMS, dtype, init_method_factory())

    rows = []
    for n in ELEM_NUMS:
        shape = make_shape(n)
        row = {"elem_num": elem_label(n), "shape": str(tuple(shape))}
        for col in BENCH_COLUMNS:
            # worst rank bounds the collective latency.
            vals = [r[n][col] for r in rets]
            row[col] = max(vals)
        rows.append(row)

    return pd.DataFrame(rows)


BENCH_TP = [8, 4, 2]


def run_bench():
    """Run the latency sweep for every world size; return {tp_size: DataFrame}."""
    dtype = dtypes.bf16
    return {tp_size: bench_one_tp(tp_size, dtype) for tp_size in BENCH_TP}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    freeze_support()

    accuracy_df = run_accuracy()
    bench_dfs = run_bench()

    print("\n\n########## RESULTS ##########")

    print("\n=== add_fused_allreduce accuracy summary ===")
    print(accuracy_df.to_markdown(index=False))

    for tp_size, df in bench_dfs.items():
        print(
            f"\n=== add_fused_allreduce latency (us, max over {tp_size} ranks), dtype=bf16 ==="
        )
        print(df.to_markdown(index=False, floatfmt=".2f"))
