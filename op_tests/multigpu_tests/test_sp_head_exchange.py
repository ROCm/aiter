# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""SP4 head-exchange layout, graph replay, and vector alignment contracts;
the SP4 all-to-all and INT4 quick reduce-scatter."""

import resource
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, port, invalid_source):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from aiter.ops.sp_head_exchange import sp_head_exchange

    # AITER_CHECK rejects invalid arguments with abort(), not a Python exception.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    ca = CustomAllreduce(
        dist.group.WORLD, torch.device("cuda", rank), max_size=64 * 1024**2
    )
    assert not ca.disabled and ca.enable_register_for_capturing
    pool = ca._pool["input"]
    for dtype in (torch.bfloat16, torch.float16, torch.uint8):
        # Offset the source to exercise valid unaligned copy-in, too.
        x = torch.empty(16 * 32 + 1, dtype=dtype, device=rank)[1:].view(16, 32)
        out = torch.empty((4, 128), dtype=dtype, device=rank)

        def exchange(source, stage, pointer=pool.data_ptr, output=out):
            sp_head_exchange(ca._ptr, source, output, pointer, pool.max_size, stage, 4)

        if invalid_source == "staged":
            exchange(x, True, pool.data_ptr + 1)
            raise AssertionError("unaligned staged buffer was accepted")
        if invalid_source == "registered":
            exchange(x, False)
            raise AssertionError("unaligned registered input was accepted")

        registered = x.clone()
        x.fill_(rank)
        exchange(x, True)
        torch.testing.assert_close(
            out.cpu(),
            torch.cat([torch.full((4, 32), r, dtype=dtype) for r in range(4)], dim=1),
            rtol=0,
            atol=0,
        )
        registered.copy_(x)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        # Match the serving path: capture registration exchanges allocation
        # base handles plus tensor offsets before the first replay.
        with ca.capture(), torch.cuda.graph(graph, stream=stream):
            exchange(registered, False)
        torch.cuda.current_stream().wait_stream(stream)
        for generation in range(3):
            x.copy_(
                (
                    torch.arange(x.numel(), device=rank).view_as(x)
                    + rank * 17
                    + generation
                )
                % 127
            )
            peers = [None] * 4
            dist.all_gather_object(peers, x.cpu())
            expected = torch.cat(
                [peer[rank * 4 : (rank + 1) * 4] for peer in peers], dim=1
            )
            exchange(x, True)
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            registered.copy_(x)
            for _ in range(5):
                graph.replay()
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    # The communicator's wrappers: a producer writing the registered pool,
    # a staged input, and a captured input bound in place.
    x = (torch.arange(16 * 32, device=rank).view(16, 32) + rank * 17) % 127
    x = x.to(torch.bfloat16)
    peers = [None] * 4
    dist.all_gather_object(peers, x.cpu())
    expected = torch.cat([peer[rank * 4 : (rank + 1) * 4] for peer in peers], dim=1)
    assert ca.should_head_exchange(x)
    view = ca.registered_input_view(x.shape, x.dtype)
    view.copy_(x)
    for out in (ca.head_exchange(view, registered=True), ca.head_exchange(x)):
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with ca.capture(), torch.cuda.graph(graph, stream=stream):
        out = ca.head_exchange(x)
    graph.replay()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    dist.barrier()
    ca.close()
    dist.destroy_process_group()


def _collectives_worker(rank, port):
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=4, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}"
    )
    ensure_model_parallel_initialized(4, 1)
    comm = get_tp_group().device_communicator

    def part(r, seed, shape):
        gen = torch.Generator(device="cuda").manual_seed(seed * 4 + r)
        return torch.randn(shape, generator=gen, device="cuda").to(torch.bfloat16)

    # all_to_all: block i (3 rows) of rank r's input is block r on rank i.
    def expected(seed):
        parts = [part(r, seed, (12, 37, 5)) for r in range(4)]
        return torch.cat([p[rank * 3 : rank * 3 + 3] for p in parts])

    assert comm.pynccl_comm is not None and not comm.pynccl_comm.disabled
    send = part(rank, 0, (12, 37, 5))
    recv = torch.empty_like(send)
    comm.all_to_all(recv, send)
    torch.testing.assert_close(recv, expected(0), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        comm.all_to_all(recv, send)
    for seed in range(1, 4):
        send.copy_(part(rank, seed, (12, 37, 5)))
        recv.zero_()
        graph.replay()
        torch.testing.assert_close(recv, expected(seed), rtol=0, atol=0)

    qr = comm.qr_comm
    assert qr is not None and not qr.disabled
    assert not qr.should_quick_reduce_scatter(part(rank, 0, (4096, 1024)))
    assert not qr.should_quick_reduce_scatter(part(rank, 0, (4097, 3072)))
    parts = [part(r, 4, (4096, 3072)) for r in range(4)]  # 24 MiB each
    assert qr.should_quick_reduce_scatter(parts[rank])
    out = qr.quick_reduce_scatter(parts[rank])
    ref = sum(p.float() for p in parts)[rank * 1024 : (rank + 1) * 1024]
    # Lossy (INT4 steps of max / 8 per 32 values); a wrong or missing shard is off > 50%.
    assert out.shape == ref.shape
    assert ((out.float() - ref).norm() / ref.norm()).item() < 0.15

    torch.cuda.synchronize()
    destroy_model_parallel()
    destroy_distributed_environment()


def _sp4_port():
    if torch.version.hip is None or torch.cuda.device_count() < 4:
        pytest.skip("requires four ROCm GPUs with peer access")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        pytest.skip("SP4 collectives are validated on gfx950")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_sp_all_to_all_and_quick_reduce_scatter(monkeypatch):
    port = _sp4_port()
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    monkeypatch.setenv("AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "64")
    mp.spawn(_collectives_worker, args=(port,), nprocs=4, join=True)


@pytest.mark.parametrize("invalid_source", [None, "staged", "registered"])
def test_sp_head_exchange(invalid_source, capfd):
    port = _sp4_port()
    if invalid_source is None:
        mp.spawn(_worker, args=(port, invalid_source), nprocs=4, join=True)
    else:
        with pytest.raises(mp.ProcessExitedException, match="SIGABRT"):
            mp.spawn(_worker, args=(port, invalid_source), nprocs=4, join=True)
        message = (
            "staged head exchange buffer requires 16-byte alignment"
            if invalid_source == "staged"
            else "registered head exchange input requires 16-byte alignment"
        )
        assert message in capfd.readouterr().err
