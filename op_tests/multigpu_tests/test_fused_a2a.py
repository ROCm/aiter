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
from aiter.ops.flydsl.kernels.fused_a2a_intranode_op import (
    FusedA2AIntraNodeOp,
    FusedA2AOutIntraNodeOp,
)

_WORLD_SIZE = 8
_CASES = (
    ("small", 8, 17, 128),
    ("deployed", 40, 9419, 128),
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sequence_major_input(rank, heads, seq_len, head_dim, device):
    numel = heads * seq_len * head_dim
    values = torch.arange(numel, dtype=torch.int64).view(1, seq_len, heads, head_dim)
    return (values + rank * numel).to(torch.bfloat16).to(device)


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
            q = _sequence_major_input(rank, heads, seq_len, head_dim, device)
            inputs = (q, q + 1000, q + 2000)
            references = []
            heads_local = heads // world_size
            for input_tensor in inputs:
                packed = input_tensor.permute(0, 2, 1, 3).contiguous()
                gathered = funcol.wait_tensor(
                    funcol.all_to_all_single(
                        packed.view(-1), None, None, dist.group.WORLD
                    )
                )
                reference = gathered.view(world_size, 1, heads_local, seq_len, head_dim)
                references.append(
                    reference.permute(1, 2, 0, 3, 4).reshape(
                        1, heads_local, world_size * seq_len, head_dim
                    )
                )

            op = FusedA2AIntraNodeOp(
                rank=rank,
                world_size=world_size,
                shape=input_tensor.shape,
                dtype=input_tensor.dtype,
            )
            actuals = op(*inputs)
            torch.cuda.synchronize()
            expected_shape = (1, heads_local, world_size * seq_len, head_dim)
            for tensor_name, actual, reference in zip(
                "qkv", actuals, references, strict=True
            ):
                actual = actual.view(reference.shape)
                if not torch.equal(actual, reference):
                    mismatch = torch.nonzero(actual != reference, as_tuple=False)[
                        0
                    ].flatten()
                    raise AssertionError(
                        f"{case_name} {tensor_name} rank {rank}: byte mismatch at "
                        f"index {mismatch.tolist()}: actual={actual[tuple(mismatch)].item()} "
                        f"reference={reference[tuple(mismatch)].item()}"
                    )
                if tuple(actual.shape) != expected_shape:
                    raise AssertionError(
                        f"{case_name} {tensor_name} rank {rank}: got "
                        f"{tuple(actual.shape)}, expected {expected_shape}"
                    )
            out_input = references[0].contiguous()
            out_packed = out_input.permute(2, 0, 1, 3).contiguous()
            out_reference = funcol.wait_tensor(
                funcol.all_to_all_single(
                    out_packed.view(-1), None, None, dist.group.WORLD
                )
            ).view(1, seq_len, heads, head_dim)
            out_op = FusedA2AOutIntraNodeOp(
                rank=rank,
                world_size=world_size,
                shape=out_input.shape,
                dtype=out_input.dtype,
            )
            out_actual = out_op(out_input).view(out_reference.shape)
            torch.cuda.synchronize()
            if not torch.equal(out_actual, out_reference):
                mismatch = torch.nonzero(out_actual != out_reference, as_tuple=False)[
                    0
                ].flatten()
                raise AssertionError(
                    f"{case_name} out rank {rank}: byte mismatch at index "
                    f"{mismatch.tolist()}: actual={out_actual[tuple(mismatch)].item()} "
                    f"reference={out_reference[tuple(mismatch)].item()}"
                )

            dist.barrier()
            if rank == 0:
                print(
                    f"PASS {case_name}: in={tuple(input_tensor.shape)} "
                    f"gathered={expected_shape} out={tuple(out_reference.shape)}"
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
