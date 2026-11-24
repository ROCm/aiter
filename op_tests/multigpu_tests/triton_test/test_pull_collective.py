# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-GPU tests of the pull collectives.

Each case compares the result of ``PullCollective`` with a reference built from
RCCL data movement (all-gather, all-to-all) or from an fp32 rank-order sum
(reduce-scatter). Data movement must be bit-exact. Each op runs many times in a
row with new data, to catch a race on the reused buffer, and the reduce-scatter
also runs from a captured graph.

Usage:
    pytest op_tests/multigpu_tests/triton_test/test_pull_collective.py
    python op_tests/multigpu_tests/triton_test/test_pull_collective.py -n 4
"""

import argparse
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter import logger
from aiter.test_common import ensure_spawn_method

H = 6144
ITERS = 12


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _check_all_gather(comm, rank, world, device, rows):
    n = rows * H
    gen = torch.Generator(device=device)
    # two fields with different sizes, one publish
    for it in range(ITERS):
        gen.manual_seed(100 * it + rank)
        a = torch.randint(
            0, 255, (rows, H), generator=gen, dtype=torch.uint8, device=device
        )
        b = torch.randint(
            0, 255, (rows, H // 32), generator=gen, dtype=torch.uint8, device=device
        )
        comm.acquire()
        comm.buffer((rows, H), torch.uint8).copy_(a)
        comm.buffer((rows, H // 32), torch.uint8, byte_offset=n).copy_(b)
        out_a = torch.empty((world * rows, H), dtype=torch.uint8, device=device)
        out_b = torch.empty((world * rows, H // 32), dtype=torch.uint8, device=device)
        comm.all_gather([(0, n, out_a), (n, n // 32, out_b)])
        ref_a = torch.empty_like(out_a)
        ref_b = torch.empty_like(out_b)
        dist.all_gather_into_tensor(ref_a, a)
        dist.all_gather_into_tensor(ref_b, b)
        assert torch.equal(out_a, ref_a), f"all_gather field a differs, iter {it}"
        assert torch.equal(out_b, ref_b), f"all_gather field b differs, iter {it}"


def _check_all_to_all(comm, rank, world, device, rows):
    gen = torch.Generator(device=device)
    for it in range(ITERS):
        gen.manual_seed(200 * it + rank)
        x = torch.randint(
            0, 255, (world * rows, H), generator=gen, dtype=torch.uint8, device=device
        )
        comm.acquire()
        comm.buffer(x.shape, torch.uint8).copy_(x)
        out = torch.empty_like(x)
        comm.all_to_all([(0, x.numel(), out)])
        ref = torch.empty_like(x)
        dist.all_to_all_single(ref, x)
        assert torch.equal(out, ref), f"all_to_all differs, iter {it}"


def _check_reduce_scatter(comm, rank, world, device, rows):
    gen = torch.Generator(device=device)
    total = world * rows
    out = torch.empty((rows, H), dtype=torch.bfloat16, device=device)
    for it in range(ITERS):
        gen.manual_seed(300 * it + rank)
        x = (
            torch.randn((total, H), generator=gen, dtype=torch.float32, device=device)
            * (1 + rank)
        ).to(torch.bfloat16)
        comm.acquire()
        comm.buffer(x.shape, torch.bfloat16).copy_(x)
        comm.reduce_scatter(out, x.numel())
        full = torch.empty((world, total, H), dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(full.view(-1), x.view(-1))
        acc = torch.zeros((rows, H), dtype=torch.float32, device=device)
        for r in range(world):
            acc += full[r, rank * rows : (rank + 1) * rows].float()
        assert torch.equal(
            out, acc.to(torch.bfloat16)
        ), f"reduce_scatter differs, iter {it}"


def _check_reduce_scatter_graph(comm, rank, world, device, rows):
    total = world * rows
    x = torch.randn((total, H), dtype=torch.bfloat16, device=device)
    buf = comm.buffer(x.shape, torch.bfloat16)
    out = torch.empty((rows, H), dtype=torch.bfloat16, device=device)
    # capture acquire + produce + collective as one graph
    comm.acquire()
    buf.copy_(x)
    comm.reduce_scatter(out, x.numel())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        comm.acquire()
        buf.copy_(x)
        comm.reduce_scatter(out, x.numel())
    for it in range(5):
        x.copy_(torch.randn((total, H), dtype=torch.bfloat16, device=device))
        graph.replay()
        torch.cuda.synchronize()
        full = torch.empty((world, total, H), dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(full.view(-1), x.view(-1))
        acc = torch.zeros((rows, H), dtype=torch.float32, device=device)
        for r in range(world):
            acc += full[r, rank * rows : (rank + 1) * rows].float()
        assert torch.equal(
            out, acc.to(torch.bfloat16)
        ), f"graph replay differs, it {it}"


def _check_all_to_all_mxfp8(comm, rank, world, device, rows, d):
    gen = torch.Generator(device=device)
    total = world * rows
    for it in range(ITERS):
        gen.manual_seed(400 * it + rank)
        codes = torch.randint(
            0, 255, (total, d), generator=gen, dtype=torch.uint8, device=device
        )
        scales = torch.randint(
            0, 255, (total, d // 32), generator=gen, dtype=torch.uint8, device=device
        )
        comm.acquire()
        comm.buffer(codes.shape, torch.uint8).copy_(codes)
        comm.buffer(scales.shape, torch.uint8, byte_offset=codes.numel()).copy_(scales)
        out_c = torch.empty((rows, world * d), dtype=torch.uint8, device=device)
        out_s = torch.empty((rows, world * d // 32), dtype=torch.uint8, device=device)
        comm.all_to_all_mxfp8(out_c, out_s, rows, d)
        rc = torch.empty_like(codes)
        rs = torch.empty_like(scales)
        dist.all_to_all_single(rc, codes)
        dist.all_to_all_single(rs, scales)
        # block p of the exchange -> column block p
        exp_c = rc.view(world, rows, d).transpose(0, 1).reshape(rows, world * d)
        exp_s = (
            rs.view(world, rows, d // 32).transpose(0, 1).reshape(rows, world * d // 32)
        )
        assert torch.equal(out_c, exp_c), f"mxfp8 codes differ, iter {it}"
        assert torch.equal(out_s, exp_s), f"mxfp8 scales differ, iter {it}"


def _check_errors(comm, world, device):
    bad = torch.empty((3, 7), dtype=torch.uint8, device=device)
    with pytest.raises(ValueError):
        comm.all_gather([(4, 8, bad)])  # offset not a multiple of 8
    with pytest.raises(ValueError):
        comm.reduce_scatter(
            torch.empty((8,), dtype=torch.float32, device=device), 8 * world
        )
    with pytest.raises(ValueError):
        comm.all_to_all_mxfp8(bad, bad, 4, 300)  # d is not 256 * 2^k


def _worker(rank, world, port, results):
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
        gloo = dist.new_group(backend="gloo")
        from aiter.ops.triton.comms.pull_collective import PullCollective

        rows = 512
        d = 2048
        comm = PullCollective(gloo, world * rows * H * 2, device)
        _check_all_gather(comm, rank, world, device, rows)
        _check_all_to_all(comm, rank, world, device, rows)
        _check_reduce_scatter(comm, rank, world, device, rows)
        _check_reduce_scatter_graph(comm, rank, world, device, rows)
        _check_all_to_all_mxfp8(comm, rank, world, device, rows, d)
        _check_errors(comm, world, device)
        dist.barrier()
        comm.close()
        dist.destroy_process_group()
        results.put((rank, None))
    except BaseException as exc:
        results.put((rank, repr(exc)))
        raise


def _run(world):
    ensure_spawn_method()
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, world, port, results))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    errors = {}
    for _ in range(world):
        rank, err = results.get(timeout=600)
        if err:
            errors[rank] = err
    for p in procs:
        p.join(timeout=60)
    assert not errors, f"failed ranks: {errors}"


@pytest.mark.parametrize("world", [2, 4])
def test_pull_collective(world):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _run(world)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--world-size", type=int, default=4)
    args = parser.parse_args()
    _run(args.world_size)
    logger.info("pull collective test passed for %d ranks", args.world_size)
