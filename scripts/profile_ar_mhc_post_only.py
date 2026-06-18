#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark split (AR + mhc_post) vs fused (AR+mhc_post epilogue only)."""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool, set_start_method
from statistics import median
from typing import Callable, Optional

import torch
import torch.distributed as dist

import aiter
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
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
from aiter.ops.custom_all_reduce import launch_fused_allreduce_mhc_post_only

set_start_method("spawn", force=True)

WARMUP = 5
ITERS = 50


def _make_inputs(m: int, hidden_size: int, rank: int, device: torch.device):
    torch.manual_seed(20260617)
    hc_mult = 4
    base_layer_input = torch.randn(m, hidden_size, dtype=aiter.dtypes.bf16, device=device)
    return {
        "layer_input": base_layer_input * float(rank + 1),
        "residual_in": torch.randn(m, hc_mult, hidden_size, dtype=aiter.dtypes.bf16, device=device),
        "post_layer_mix": torch.randn(m, hc_mult, 1, dtype=aiter.dtypes.fp32, device=device),
        "comb_res_mix": torch.randn(m, hc_mult, hc_mult, dtype=aiter.dtypes.fp32, device=device),
    }


def _event_us(fn: Callable[[], None], *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def _bench_graph(fn: Callable[[], None], *, warmup: int, iters: int) -> list[float]:
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        with torch.cuda.graph(graph, stream=gc.stream):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def profile_worker(
    tp_size: int,
    rank_id: int,
    m: int,
    hidden_size: int,
    with_graph: bool,
    init_method: str,
):
    device = torch.device(f"cuda:{rank_id}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank_id,
        distributed_init_method=init_method,
    )
    ensure_model_parallel_initialized(tp_size, 1)
    tensors = _make_inputs(m, hidden_size, rank_id, device)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    ca_comm = get_tp_group().device_communicator.ca_comm
    next_residual = torch.empty_like(tensors["residual_in"])
    post_mix = tensors["post_layer_mix"]
    if post_mix.ndim == 3:
        post_mix = post_mix.squeeze(-1)

    def _reg():
        if ca_comm is None or ca_comm.disabled:
            return 0, 0
        return ca_comm._pool["input"].data_ptr, ca_comm._pool["input"].max_size

    def split_ar_post():
        reduced = tensor_model_parallel_all_reduce(tensors["layer_input"])
        aiter.mhc_post(
            next_residual,
            reduced,
            tensors["residual_in"],
            post_mix,
            tensors["comb_res_mix"],
        )

    def fused_ar_post(*, graph_replay: bool = False):
        if graph_replay:
            reg_ptr, reg_bytes = 0, 0
        else:
            reg_ptr, reg_bytes = _reg()
        launch_fused_allreduce_mhc_post_only(
            ca_comm._ptr,
            tensors["layer_input"],
            tensors["residual_in"],
            tensors["post_layer_mix"],
            tensors["comb_res_mix"],
            reg_ptr=reg_ptr,
            reg_bytes=reg_bytes,
        )

    for _ in range(WARMUP):
        split_ar_post()
        fused_ar_post()
    torch.cuda.synchronize()

    bench = _bench_graph if with_graph else _event_us
    split_us = bench(split_ar_post, warmup=2, iters=ITERS)
    if with_graph:
        fused_us = bench(lambda: fused_ar_post(graph_replay=True), warmup=2, iters=ITERS)
    else:
        fused_us = bench(fused_ar_post, warmup=2, iters=ITERS)

    destroy_model_parallel()
    destroy_distributed_environment()

    return {
        "rank": rank_id,
        "split_median": median(split_us),
        "fused_median": median(fused_us),
    }


def run_profile(
    tp_size: int,
    m: int,
    hidden_size: int,
    with_graph: bool,
    init_method: Optional[str] = None,
):
    if init_method is None:
        init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(
            profile_worker,
            args=(tp_size, r, m, hidden_size, with_graph, init_method),
        )
        for r in range(tp_size)
    ]
    pool.close()
    pool.join()
    rows = [r.get() for r in rets]
    split = max(x["split_median"] for x in rows)
    fused = max(x["fused_median"] for x in rows)
    saved = split - fused
    speedup = (saved / split * 100.0) if split > 0 else 0.0
    return split, fused, saved, speedup


def main():
    parser = argparse.ArgumentParser(description="AR+mhc_post only profile TP=2")
    parser.add_argument("-t", "--tp-size", type=int, default=2)
    parser.add_argument(
        "-s",
        "--shapes",
        type=str,
        default="1,4096 2,4096 4,4096 16,4096 32,4096 128,4096 1024,4096 2048,4096 8192,4096",
    )
    parser.add_argument("-g", "--graph", type=int, default=-1, choices=[-1, 0, 1])
    args = parser.parse_args()

    shapes = []
    for tok in args.shapes.split():
        m_s, h_s = tok.split(",")
        shapes.append((int(m_s), int(h_s)))

    graph_modes = [False, True] if args.graph < 0 else [bool(args.graph)]

    print("# AR + mhc_post only (split vs fused epilogue)")
    print(f"# HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}")
    print(f"# warmup={WARMUP} iters={ITERS} metric=rank-max median (us)")
    print()

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    for with_graph in graph_modes:
        mode = "graph-on" if with_graph else "graph-off"
        print(f"## {mode}")
        print("M\tsplit\tfused\tsaved\tspeedup")
        for m, hidden_size in shapes:
            split, fused, saved, sp = run_profile(
                args.tp_size, m, hidden_size, with_graph, init_method
            )
            print(f"{m}\t{split:.1f}\t{fused:.1f}\t{saved:.1f}\t{sp:+.1f}%")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
