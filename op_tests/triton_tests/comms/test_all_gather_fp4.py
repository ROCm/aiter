# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from aiter.ops.triton.comms.all_gather_fp4 import Fp4AllGather, all_gather_fp4


def generate_fp4_gather_inputs(rows, code_bytes, metadata_bytes, device):
    """Byte payloads: every FP4 bit pattern and opaque metadata are legal to copy."""
    codes = torch.randint(0, 256, (rows, code_bytes), dtype=torch.uint8, device=device)
    meta = torch.randint(
        0, 256, (rows, metadata_bytes), dtype=torch.uint8, device=device
    )
    qout = torch.empty((4 * rows, code_bytes), dtype=torch.uint8, device=device)
    mout = torch.empty((4 * rows, metadata_bytes), dtype=torch.uint8, device=device)
    return codes, meta, (qout, mout)


def _fill(tensor, rank, epoch):
    values = torch.arange(tensor.numel(), dtype=torch.int32, device=tensor.device)
    tensor.copy_(
        ((values * 13 + rank * 47 + epoch * 19) % 256).to(torch.uint8).view_as(tensor)
    )


def _check(outputs, epoch):
    for output in outputs:
        for rank, shard in enumerate(output.chunk(4, dim=0)):
            ref = torch.empty_like(shard)
            _fill(ref, rank, epoch)
            torch.testing.assert_close(shard, ref, atol=0, rtol=0)


@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != 4,
    reason="run with torchrun --nproc-per-node=4",
)
def test_all_gather_fp4_tp4():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    try:
        # Both global prefill sizes, plus partial tiles and metadata-free calls.
        for code_bytes, metadata_bytes in [
            (20, 4),
            (2560, 208),
            (3072, 224),
            (3584, 352),
            (3072, 0),
        ]:
            comm = Fp4AllGather(
                dist.group.WORLD, 4096, code_bytes, metadata_bytes, device=device
            )
            try:
                for epoch, rows in enumerate([3, 2048, 4096, 7, 2048]):
                    codes, meta, out = generate_fp4_gather_inputs(
                        rows, code_bytes, metadata_bytes, device
                    )
                    _fill(codes, rank, epoch)
                    _fill(meta, rank, epoch)
                    all_gather_fp4(codes, meta, comm, out=out)
                    _check(out, epoch)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    all_gather_fp4(codes, meta, comm, out=out)
                for epoch in range(10, 20):
                    _fill(codes, rank, epoch)
                    _fill(meta, rank, epoch)
                    graph.replay()
                    _check(out, epoch)
                if code_bytes == 3072 and metadata_bytes == 224:
                    cases = [
                        generate_fp4_gather_inputs(
                            n, code_bytes, metadata_bytes, device
                        )
                        for n in (7, 2048, 3, 4096, 16, 2048, 7)
                    ]
                    for q, meta, result in cases:
                        all_gather_fp4(q, meta, comm, out=result)
                    mixed = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(mixed):
                        for q, meta, result in cases:
                            all_gather_fp4(q, meta, comm, out=result)
                    for epoch in range(20, 23):
                        for i, (q, meta, result) in enumerate(cases):
                            _fill(q, rank, epoch + i)
                            _fill(meta, rank, epoch + i)
                        mixed.replay()
                        for i, (_, _, result) in enumerate(cases):
                            _check(result, epoch + i)
                assert comm.errors.item() == 0, "peer readiness wait timed out"
            finally:
                comm.close()
    finally:
        dist.destroy_process_group()
