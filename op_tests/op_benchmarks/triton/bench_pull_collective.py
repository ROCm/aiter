# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark of the pull collectives against RCCL at sequence-parallel prefill shapes.

For each token count T, the benchmark times the zero-copy path (the producer data
is already in the IPC buffer, as a GEMM or quant epilogue would leave it) and the
RCCL op on the same shapes. Rank 0 prints one row for each op.

Usage:
    python op_tests/op_benchmarks/triton/bench_pull_collective.py -n 4 --tokens 4096 8192
"""

import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter import logger
from aiter.test_common import ensure_spawn_method

HIDDEN = 6144
ATTN_D = 2048


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _time_us(fn, iters=20, warmup=5, reps=3):
    best = float("inf")
    for _ in range(reps):
        for _ in range(warmup):
            fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters * 1000.0)
    return best


def _bench_tokens(comm, t, world, device):
    rows_out = []
    rows = t // world
    # all-gather of fp8 codes
    a = torch.randint(0, 255, (rows, HIDDEN), dtype=torch.uint8, device=device)
    out = torch.empty((t, HIDDEN), dtype=torch.uint8, device=device)
    ref = torch.empty_like(out)
    comm.buffer(a.shape, torch.uint8).copy_(a)

    def pull_ag():
        comm.acquire()
        comm.all_gather([(0, rows * HIDDEN, out)])

    rows_out.append(
        (
            "all_gather_fp8",
            t,
            _time_us(pull_ag),
            _time_us(lambda: dist.all_gather_into_tensor(ref, a)),
        )
    )
    # reduce-scatter of bf16
    x = torch.randn((t, HIDDEN), dtype=torch.bfloat16, device=device)
    comm.buffer(x.shape, torch.bfloat16).copy_(x)
    rs_out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    rs_ref = torch.empty_like(rs_out)

    def pull_rs():
        comm.acquire()
        comm.reduce_scatter(rs_out, x.numel())

    rows_out.append(
        (
            "reduce_scatter_bf16",
            t,
            _time_us(pull_rs),
            _time_us(lambda: dist.reduce_scatter_tensor(rs_ref, x)),
        )
    )
    # MXFP8 attention-output all-to-all, unpack included
    codes = torch.randint(0, 255, (t, ATTN_D), dtype=torch.uint8, device=device)
    scales = torch.randint(0, 255, (t, ATTN_D // 32), dtype=torch.uint8, device=device)
    comm.buffer(codes.shape, torch.uint8).copy_(codes)
    comm.buffer(scales.shape, torch.uint8, byte_offset=codes.numel()).copy_(scales)
    out_c = torch.empty((rows, world * ATTN_D), dtype=torch.uint8, device=device)
    out_s = torch.empty((rows, world * ATTN_D // 32), dtype=torch.uint8, device=device)
    rc = torch.empty_like(codes)
    rs = torch.empty_like(scales)

    def pull_a2a():
        comm.acquire()
        comm.all_to_all_mxfp8(out_c, out_s, rows, ATTN_D)

    def rccl_a2a():
        dist.all_to_all_single(rc, codes)
        dist.all_to_all_single(rs, scales)
        rc.view(world, rows, ATTN_D).transpose(0, 1).reshape(rows, -1)
        rs.view(world, rows, ATTN_D // 32).transpose(0, 1).reshape(rows, -1)

    rows_out.append(("a2a_mxfp8_unpack", t, _time_us(pull_a2a), _time_us(rccl_a2a)))
    return rows_out


def _worker(rank, world, port, tokens, results):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    gloo = dist.new_group(backend="gloo")
    from aiter.ops.triton.comms.pull_collective import PullCollective

    max_t = max(tokens)
    comm = PullCollective(gloo, max_t * HIDDEN * 2 + 4096, device)
    rows_out = []
    for t in tokens:
        rows_out += _bench_tokens(comm, t, world, device)
    dist.barrier()
    comm.close()
    if rank == 0:
        results.put(rows_out)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--world-size", type=int, default=4)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[2048, 4096, 8192, 16384]
    )
    args = parser.parse_args()
    ensure_spawn_method()
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_worker, args=(r, args.world_size, port, args.tokens, results)
        )
        for r in range(args.world_size)
    ]
    for p in procs:
        p.start()
    rows = results.get(timeout=900)
    for p in procs:
        p.join(timeout=60)
    print(f"{'op':<22}{'tokens':>8}{'pull_us':>10}{'rccl_us':>10}{'ratio':>8}")
    for name, t, pull, rccl in rows:
        print(f"{name:<22}{t:>8}{pull:>10.1f}{rccl:>10.1f}{pull / rccl:>8.2f}")
    logger.info("benchmark done")


if __name__ == "__main__":
    main()
