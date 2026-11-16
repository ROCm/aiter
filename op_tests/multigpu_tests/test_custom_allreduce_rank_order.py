# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check that custom AR / reduce-scatter sum ranks in a fixed 0..n-1 order.

Run with: python3 op_tests/multigpu_tests/test_custom_allreduce_rank_order.py
"""

import argparse
import logging
import os
import sys
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

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

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

# The partials below are specific to four ranks.
TP_SIZE = 4

# Captured BF16 partials straddle a rounding midpoint after FP32 summation.
PARTIALS = [
    [1.1496013030409813e-9, 5.966285243630409e-10, 1e-7],
    [-0.01531982421875, -0.00616455078125, 1.0],
    [-0.006622314453125, -0.0029449462890625, -1.0],
    [-0.006072998046875, -0.0007476806640625, -1e-6],
]

# (rows, width, dtype, use_new): row counts and widths that select the LL,
# LL128, 1-stage and 2-stage kernels.
AR_CASES = [
    (1, 7168, torch.bfloat16, True),
    (4, 8192, torch.bfloat16, True),
    (17, 7168, torch.bfloat16, True),
    (97, 7168, torch.bfloat16, True),
    (512, 7168, torch.bfloat16, True),
    (512, 7168, torch.bfloat16, False),
    (1025, 88, torch.bfloat16, True),
    (97, 7168, torch.float16, True),
    (512, 7168, torch.float32, True),
]

# (shape, dim, dtype): vectorized and naive first/last/middle-dim kernels.
RS_CASES = [
    ((8, 7168), 0, torch.bfloat16),
    ((5, 7168), -1, torch.bfloat16),
    ((5, 88), -1, torch.bfloat16),
    ((2, 8, 7168), 1, torch.bfloat16),
    ((2, 8, 7), 1, torch.bfloat16),
    ((8, 7168), 0, torch.float16),
]


def barrier_before_teardown():
    if not dist.is_initialized():
        return
    torch.cuda.synchronize()
    get_tp_group().barrier()
    torch.cuda.synchronize()


def _partials(numel, dtype):
    return torch.tensor(PARTIALS, dtype=dtype).repeat(1, (numel + 2) // 3)[:, :numel]


def _rank_order_sum(partials, dtype):
    reference = partials[0].float()
    for peer in range(1, TP_SIZE):
        reference = reference + partials[peer].float()
    return reference.to(dtype)


def _run_ar_case(rank, ca, rows, width, dtype, use_new, retained):
    partials = _partials(width, dtype)
    x = torch.empty((rows, width), dtype=dtype, device=f"cuda:{rank}")
    x.copy_(partials[rank].expand(rows, width))
    assert ca.should_custom_ar(x)
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        out = ca.custom_all_reduce(x, use_new=use_new)
    # Retain peer IPC storage until every captured graph is finished.
    retained.append((graph, x, out))
    ok = True
    # Change the inputs after capture so replay must read the new values.
    for factor in (1.0, -1.0, 0.5):
        changed = (partials.float() * factor).to(dtype)
        reference = _rank_order_sum(changed, dtype).expand(rows, width)
        x.copy_(changed[rank].expand(rows, width))
        eager = ca.custom_all_reduce(x, use_new=use_new)
        graph.replay()
        torch.cuda.synchronize()
        ok &= torch.equal(eager.cpu(), reference)
        ok &= torch.equal(out.cpu(), reference)
    return ok


def _run_rs_case(rank, ca, shape, dim, dtype):
    numel = 1
    for s in shape:
        numel *= s
    partials = _partials(numel, dtype).reshape(TP_SIZE, *shape)
    x = partials[rank].to(f"cuda:{rank}")
    assert ca.should_custom_rs(x, dim)
    reference = _rank_order_sum(partials, dtype).chunk(TP_SIZE, dim)[rank]
    out = torch.empty(reference.shape, dtype=dtype, device=x.device)
    ca.custom_reduce_scatter(x, out, dim)
    torch.cuda.synchronize()
    return torch.equal(out.cpu(), reference)


def rank_order_worker(rank, distributed_init_method):
    """Run every case on one rank and return one pass flag per case. Every
    rank walks the full list so the collectives stay aligned."""
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=TP_SIZE,
        rank=rank,
        local_rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(TP_SIZE, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled

    retained = []
    results = [_run_ar_case(rank, ca, *case, retained) for case in AR_CASES]
    results += [_run_rs_case(rank, ca, *case) for case in RS_CASES]

    barrier_before_teardown()
    del retained
    destroy_model_parallel()
    destroy_distributed_environment()
    torch.cuda.empty_cache()
    return results


def run_rank_order_test():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=TP_SIZE) as pool:
        rets = [
            pool.apply_async(rank_order_worker, args=(rank, init_method))
            for rank in range(TP_SIZE)
        ]
        per_rank = [el.get() for el in rets]

    labels = [
        f"all_reduce rows={r} width={w} {str(d).split('.')[-1]} use_new={u}"
        for r, w, d, u in AR_CASES
    ] + [
        f"reduce_scatter shape={s} dim={dim} {str(d).split('.')[-1]}"
        for s, dim, d in RS_CASES
    ]
    rows = [
        {
            "case": label,
            "failed_ranks": [r for r in range(TP_SIZE) if not per_rank[r][i]],
        }
        for i, label in enumerate(labels)
    ]
    df = pd.DataFrame(rows)
    print("\n=== custom AR rank-order summary ===")
    print(df.to_markdown(index=False))
    return [row["case"] for row in rows if row["failed_ranks"]]


parser = argparse.ArgumentParser(
    description="custom all-reduce / reduce-scatter fixed rank-order test"
)

if __name__ == "__main__":
    freeze_support()
    parser.parse_args()
    if not torch.version.hip or torch.cuda.device_count() < TP_SIZE:
        print(f"Skipping: requires {TP_SIZE} ROCm GPUs")
        sys.exit(0)
    failures = run_rank_order_test()
    assert (
        not failures
    ), "results differ from the rank-order 0..n-1 sum:\n  " + "\n  ".join(failures)
