# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""SP4 head-exchange layout, graph replay, and vector alignment contracts."""

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
    torch.cuda.synchronize()
    dist.barrier()
    ca.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("invalid_source", [None, "staged", "registered"])
def test_sp_head_exchange(invalid_source, capfd):
    if torch.version.hip is None or torch.cuda.device_count() < 4:
        pytest.skip("requires four ROCm GPUs with peer access")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        pytest.skip("SP4 head exchange is validated on gfx950")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
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
