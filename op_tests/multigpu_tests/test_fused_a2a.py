# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""World-8 correctness test for the M1 fused Ulysses in-hop transport."""

from __future__ import annotations

import socket

import mori.shmem as ms
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.multiprocessing as mp

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.fused_a2a_intranode_op import FusedA2AIntraNodeOp

_WORLD_SIZE = 8
_CASES = (
    ("small", 8, 17, 128),
    ("deployed", 40, 9419, 128),
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _head_major_input(rank, heads, seq_len, head_dim, device):
    numel = heads * seq_len * head_dim
    values = torch.arange(numel, dtype=torch.int64).view(1, seq_len, heads, head_dim)
    values = values + rank * numel
    # Match the incumbent Ulysses host-side [B,S,H,D] -> [B,H,S,D] reorder.
    return values.to(torch.bfloat16).to(device).permute(0, 2, 1, 3).contiguous()


def _run_rank(rank, world_size, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        cpu_group = dist.new_group(backend="gloo")
        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        ms.shmem_torch_process_group_init("mori")

        for case_name, heads, seq_len, head_dim in _CASES:
            input_tensor = _head_major_input(rank, heads, seq_len, head_dim, device)
            flat = input_tensor.view(-1)
            reference = funcol.wait_tensor(
                funcol.all_to_all_single(flat, None, None, dist.group.WORLD)
            )

            op = FusedA2AIntraNodeOp(
                rank=rank,
                world_size=world_size,
                shape=input_tensor.shape,
                dtype=input_tensor.dtype,
            )
            actual = op(input_tensor)
            torch.cuda.synchronize()
            if not torch.equal(actual, reference):
                mismatch = torch.nonzero(actual != reference, as_tuple=False)[0].item()
                raise AssertionError(
                    f"{case_name} rank {rank}: byte mismatch at flat index {mismatch}: "
                    f"actual={actual[mismatch].item()} reference={reference[mismatch].item()}"
                )

            heads_local = heads // world_size
            result = actual.view(world_size, 1, heads_local, seq_len, head_dim)
            result = result.permute(1, 2, 0, 3, 4).reshape(
                1, heads_local, world_size * seq_len, head_dim
            )
            expected_shape = (1, heads_local, world_size * seq_len, head_dim)
            if tuple(result.shape) != expected_shape:
                raise AssertionError(
                    f"{case_name} rank {rank}: got {tuple(result.shape)}, "
                    f"expected {expected_shape}"
                )
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS {case_name}: input={tuple(input_tensor.shape)} output={expected_shape}"
                )
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


def main():
    if not torch.cuda.is_available():
        print("SKIP: fused_a2a requires ROCm GPUs")
        return 0
    arch = get_gfx_runtime()
    if arch != "gfx950":
        print(f"SKIP: fused_a2a M1 supports gfx950, attached GPU is {arch}")
        return 0
    if torch.cuda.device_count() < _WORLD_SIZE:
        print(
            f"SKIP: fused_a2a requires {_WORLD_SIZE} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
        return 0

    mp.spawn(_run_rank, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)
    print(f"2 passed, 0 skipped on {arch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
