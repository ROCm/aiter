# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

import aiter as ops
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port


def _rmsnorm_reference(
    allreduced: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = allreduced + residual
    variance = residual_out.float().pow(2).mean(dim=-1, keepdim=True)
    out = residual_out.float() * torch.rsqrt(variance + eps) * weight.float()
    return out.to(allreduced.dtype), residual_out.to(allreduced.dtype)


def _qr_rmsnorm_worker(
    rank: int,
    world_size: int,
    inputs: list[torch.Tensor],
    residuals: list[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    distributed_init_method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    ptr = None
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=distributed_init_method,
            rank=rank,
            world_size=world_size,
        )
        group = dist.new_group(list(range(world_size)), backend="nccl")

        ptr = ops.init_custom_qr(rank, world_size, None)
        handle = ops.qr_get_handle(ptr)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)
        ops.qr_open_handles(ptr, handles)

        inp = inputs[rank].to(device)
        residual = residuals[rank].to(device)
        weight = weight.to(device)
        out = torch.empty_like(inp)
        residual_out = torch.empty_like(residual)

        # Align ranks before launching the custom IPC kernel.
        dist.all_reduce(torch.zeros(1, device=device), group=group)
        torch.cuda.synchronize()

        ops.qr_all_reduce_rmsnorm(
            ptr,
            inp,
            residual,
            residual_out,
            out,
            weight,
            eps,
            inp.shape[-1],
            0,  # QuickReduceRegime.FP
            False,
        )
        torch.cuda.synchronize()
        return out.cpu(), residual_out.cpu()
    finally:
        if ptr is not None:
            ops.qr_destroy(ptr)
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 GPUs")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qr_all_reduce_rmsnorm_matches_torch_reference(dtype: torch.dtype):
    world_size = 2
    hidden_dim = 4096
    tokens = 256
    eps = 1e-6

    generator = torch.Generator().manual_seed(1234)
    inputs = [
        torch.randn((tokens, hidden_dim), dtype=dtype, generator=generator) * 0.1
        for _ in range(world_size)
    ]
    residuals = [
        torch.randn((tokens, hidden_dim), dtype=dtype, generator=generator) * 0.1
        for _ in range(world_size)
    ]
    weight = torch.randn((hidden_dim,), dtype=dtype, generator=generator) * 0.1

    allreduced = torch.stack(inputs).sum(dim=0)
    refs = [
        _rmsnorm_reference(allreduced, residuals[rank], weight, eps)
        for rank in range(world_size)
    ]

    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=world_size) as pool:
        rets = [
            pool.apply_async(
                _qr_rmsnorm_worker,
                args=(
                    rank,
                    world_size,
                    inputs,
                    residuals,
                    weight,
                    eps,
                    distributed_init_method,
                ),
            )
            for rank in range(world_size)
        ]
        results = [ret.get(timeout=120) for ret in rets]

    rtol = 5e-3 if dtype == torch.float16 else 4e-2
    atol = 5e-3 if dtype == torch.float16 else 4e-2
    for (out, residual_out), (ref_out, ref_residual_out) in zip(results, refs):
        torch.testing.assert_close(
            residual_out.float(),
            ref_residual_out.float(),
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            out.float(),
            ref_out.float(),
            rtol=rtol,
            atol=atol,
        )
