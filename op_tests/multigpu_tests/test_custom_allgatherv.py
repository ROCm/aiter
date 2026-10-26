# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`all_gatherv` of several tensors in one custom launch (`custom_all_gatherv`).

The tensors are a DP MoE layer's routed rows: MXFP8 hidden, e8m0 scales, fp32
top-k weights and int32 top-k ids. Checks byte for byte against `torch.cat`
over the ranks:
  eager, every rank the same row count and ranks with different counts
    (a prefill), including counts whose ids / weights rows are not a multiple
    of 16 bytes and a rank holding no rows;
  a captured graph replayed on new inputs;
  a captured graph of several MoE layers, each gathering then reduce-
    scattering through the same registered pool, every layer's rows made in
    the graph from the previous layer's output.
And times one launch against one custom gather per tensor, both captured.

Usage: python test_custom_allgatherv.py [-t TP] [-r ROWS ...]
"""

import argparse
import os
from multiprocessing import Pool, set_start_method

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
from aiter.test_common import run_perftest

set_start_method("spawn", force=True)

HIDDEN, SCALE_COLS, TOPK, LAYERS = 5120, 160, 6, 8
# The per-tensor custom kernels take float dtypes only.
CARRIERS = {1: torch.bfloat16, 4: torch.float32}


def routed_rows(rows, seed, rank):
    """One rank's routed rows for `seed`, the same on every rank that asks."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)

    def ints(low, high, cols, dtype):
        return torch.randint(
            low, high, (rows, cols), dtype=dtype, device="cuda", generator=gen
        )

    return [
        ints(0, 256, HIDDEN, torch.uint8),
        ints(0, 256, SCALE_COLS, torch.uint8),
        torch.rand(rows, TOPK, device="cuda", generator=gen),
        ints(-1, 384, TOPK, torch.int32),
    ]


def reference(sizes, seed):
    per_rank = [routed_rows(n, seed, r) for r, n in enumerate(sizes)]
    return [torch.cat(ts) for ts in zip(*per_rank)]


def bytes_differ(outs, refs):
    return sum(
        int((o.view(torch.uint8) != r.view(torch.uint8)).sum())
        for o, r in zip(outs, refs)
    )


def custom_gatherv(group, inputs, sizes):
    outs = group.device_communicator.ca_comm.custom_all_gatherv(inputs, sizes)
    assert outs is not None, "custom_all_gatherv declined the rows"
    return outs


def check_eager(group, sizes, seed):
    mine = routed_rows(sizes[group.rank_in_group], seed, group.rank_in_group)
    return bytes_differ(custom_gatherv(group, mine, sizes), reference(sizes, seed))


def layer_inputs(base, carry, layer):
    """Layer `layer`'s routed rows, made from `base` and the previous layer's
    reduce-scatter output `carry`, so every layer allocates its own inputs as
    a decode step's MoE layers do."""
    hidden, scale, weights, ids = base
    bump = (carry[:, :1] * 0).to(torch.uint8) + layer
    return [hidden + bump, scale ^ bump, weights * (layer + 1) + carry * 0, ids + layer]


def check_graph(group, rows, seed, graphs):
    """One graph of LAYERS MoE layers, each gathering its rows in one launch
    and reduce-scattering a tensor made from them; replayed on new rows."""
    rank, world = group.rank_in_group, group.world_size
    sizes = [rows] * world
    base = [t.clone() for t in routed_rows(rows, seed, rank)]
    gathered = []

    def step():
        carry = torch.zeros(rows, TOPK, device="cuda")
        for layer in range(LAYERS):
            outs = group.device_communicator.all_gatherv(
                layer_inputs(base, carry, layer), dim=0, sizes=sizes
            )
            gathered.append(outs)
            rs_in = outs[2].repeat(1, 8).to(torch.bfloat16)
            carry = group.reduce_scatter_tensor(rs_in)[:, :TOPK].float() * 0

    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        step()
    graphs.append((graph, base, gathered))

    bad = 0
    for replay in range(1, 4):
        for b, t in zip(base, routed_rows(rows, seed + replay, rank)):
            b.copy_(t)
        torch.cuda.synchronize()
        dist.barrier(group=group.cpu_group)
        graph.replay()
        torch.cuda.synchronize()
        ranks = [routed_rows(rows, seed + replay, r) for r in range(world)]
        zero = torch.zeros(rows, TOPK, device="cuda")
        for layer, outs in enumerate(gathered):
            per_rank = [layer_inputs(b, zero, layer) for b in ranks]
            bad += bytes_differ(outs, [torch.cat(ts) for ts in zip(*per_rank)])
    return bad


def time_graphs(group, rows, seed, graphs):
    """One custom_all_gatherv launch against one custom gather per tensor."""
    static = routed_rows(rows, seed, group.rank_in_group)
    sizes = [rows] * group.world_size
    one, per_tensor = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(one, stream=gc.stream):
        outs = custom_gatherv(group, static, sizes)
    with graph_capture() as gc, torch.cuda.graph(per_tensor, stream=gc.stream):
        singles = [
            group.all_gather(t.view(CARRIERS[t.element_size()]), use_custom=True, dim=0)
            for t in static
        ]
    graphs.append((one, per_tensor, static, outs, singles))
    dist.barrier(group=group.cpu_group)
    _, one_us = run_perftest(one.replay)
    dist.barrier(group=group.cpu_group)
    _, per_tensor_us = run_perftest(per_tensor.replay)
    return one_us, per_tensor_us


def rank_main(rank, world, init_method, row_counts):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(world, 1)
    group = get_tp_group()
    dist.all_reduce(torch.zeros(1, device="cuda"), group=group.device_group)
    torch.cuda.synchronize()

    # Graphs and their buffers outlive the sweep: captured collectives key
    # peer IPC addresses by local pointer (see test_allgather.py).
    graphs = []
    results = []
    for i, rows in enumerate(row_counts):
        ragged = [rows + 5 * r for r in range(world)]
        one_us, per_tensor_us = time_graphs(group, rows, i, graphs)
        results.append(
            {
                "uniform": check_eager(group, [rows] * world, i),
                "ragged": check_eager(group, ragged, i),
                "graph": check_graph(group, rows, 100 + i, graphs),
                "one_us": one_us,
                "per_tensor_us": per_tensor_us,
            }
        )
    empty = check_eager(group, [0] + [3] * (world - 1), 999)

    torch.cuda.synchronize()
    group.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    graphs.clear()
    return results, empty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--tp", type=int, default=2)
    parser.add_argument(
        "-r", "--rows", type=int, nargs="+", default=[1, 3, 16, 64, 97, 384]
    )
    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=args.tp) as pool:
        per_rank = pool.starmap(
            rank_main, [(r, args.tp, init_method, args.rows) for r in range(args.tp)]
        )
    failed = False
    for i, rows in enumerate(args.rows):
        cells = [results[i] for results, _ in per_rank]
        bad = {k: sum(c[k] for c in cells) for k in ("uniform", "ragged", "graph")}
        failed |= any(bad.values())
        print(
            f"rows={rows:4d} bad_bytes={bad} "
            f"one_us={max(c['one_us'] for c in cells):7.2f} "
            f"per_tensor_us={max(c['per_tensor_us'] for c in cells):7.2f}"
        )
    empty = sum(e for _, e in per_rank)
    failed |= empty != 0
    print(f"a rank with no rows: bad_bytes={empty}")
    print("FAILED" if failed else "OK")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
