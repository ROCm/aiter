# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Eight-rank MoonEP forward validation at the production 8k shape.

Run on one 8x gfx950 node:

    MORI_SHMEM_HEAP_SIZE=17179869184 \
      torchrun --standalone --nproc-per-node=8 \
      op_tests/test_moonep_gfx950_real_shape.py

The weights implement a rectangular identity MLP.  That keeps the production
communication and GEMM shapes while making a full elementwise reference
possible without a second set of multi-gigabyte expert weights.
"""

from __future__ import annotations

import os
import time

import mori.shmem as ms
import pytest
import torch
import torch.distributed as dist

from aiter.ops.flydsl.moonep import MoonEPPlanConfig
from aiter.ops.flydsl.moonep_ep import MoonEPBF16ReferenceEP


S = 8192
H = 7168
K = 8
E = 384
I = 2048
R = 8
B = E // R
TOKEN_PADDING = 128


def _share_shmem_unique_id(rank: int) -> bytes:
    objects = [ms.shmem_get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    unique_id = objects[0]
    assert isinstance(unique_id, bytes) and len(unique_id) == 128
    return unique_id


def _identity_home_weights(device: torch.device):
    experts_per_rank = E // R
    gate = torch.zeros(
        experts_per_rank, H, I, dtype=torch.bfloat16, device=device
    )
    up = torch.zeros_like(gate)
    down = torch.zeros(
        experts_per_rank, I, H, dtype=torch.bfloat16, device=device
    )
    diagonal = torch.arange(I, device=device)
    gate[:, diagonal, diagonal] = 1
    up[:, diagonal, diagonal] = 1
    down[:, diagonal, diagonal] = 1
    return gate, up, down


def _expected_output(
    hidden: torch.Tensor, route_weights: torch.Tensor
) -> torch.Tensor:
    x = hidden[:, :I].to(torch.float32)
    activated = (torch.nn.functional.silu(x) * x).to(torch.bfloat16)
    expected = torch.zeros_like(x)
    for k_idx in range(K):
        expected.add_(
            (activated.to(torch.float32) * route_weights[:, k_idx, None])
            .to(torch.bfloat16)
            .to(torch.float32)
        )
    return expected.to(torch.bfloat16)


def _run_real_shape_forward() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == R

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl")
    status = ms.shmem_init_attr(
        ms.MORI_SHMEM_INIT_WITH_UNIQUEID,
        rank,
        world_size,
        _share_shmem_unique_id(rank),
    )
    assert status == 0

    ep = None
    try:
        config = MoonEPPlanConfig(
            rank=rank,
            world_size=world_size,
            num_tokens=S,
            top_k=K,
            num_experts=E,
            prefetch_slots=B,
            token_padding=TOKEN_PADDING,
        )
        ep = MoonEPBF16ReferenceEP(
            config,
            H,
            I,
            dispatch_block_num=128,
            prefetch_block_num=128,
        )

        # Route through all 48 experts of home rank 0.  The plan must spill
        # their work across the other seven ranks and exercise remote slots.
        token = torch.arange(S, dtype=torch.int64, device=device)[:, None]
        k_idx = torch.arange(K, dtype=torch.int64, device=device)[None, :]
        topk = ((token * K + k_idx + rank * 13) % B).to(torch.int32)
        tokens_per_expert = torch.bincount(
            topk.reshape(-1).to(torch.int64), minlength=E
        ).to(torch.int32)

        raw_weights = (
            k_idx.to(torch.float32)
            + 1.0
            + (token % 7).to(torch.float32) * 0.125
        )
        route_weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).contiguous()

        generator = torch.Generator(device=device).manual_seed(20260804 + rank)
        hidden = torch.randn(
            S, H, dtype=torch.bfloat16, device=device, generator=generator
        )

        home_gate, home_up, home_down = _identity_home_weights(device)
        ep.load_home_weights(home_gate, home_up, home_down)
        torch.cuda.synchronize(device)
        del home_gate, home_up, home_down
        torch.cuda.empty_cache()

        begin = time.perf_counter()
        output, gathered_weights, plan = ep.forward(
            hidden,
            route_weights,
            topk_experts=topk,
            tokens_per_expert=tokens_per_expert,
            zero_copy=False,
        )
        torch.cuda.synchronize(device)
        cold_seconds = time.perf_counter() - begin

        assert gathered_weights is not None
        torch.testing.assert_close(
            gathered_weights, route_weights, rtol=0, atol=0
        )
        local_dynamic_slots = int(
            (plan.experts_to_copy[rank] >= 0).sum().item()
        )
        dynamic_slots = torch.tensor(
            local_dynamic_slots, dtype=torch.int32, device=device
        )
        dist.all_reduce(dynamic_slots, op=dist.ReduceOp.SUM)
        assert int(dynamic_slots.item()) > 0

        expected_first = _expected_output(hidden, route_weights)
        torch.testing.assert_close(
            output[:, :I], expected_first, rtol=0, atol=0
        )
        assert int(torch.count_nonzero(output[:, I:]).item()) == 0

        begin = time.perf_counter()
        reused_output, reused_weights, reused_plan = ep.forward(
            hidden,
            route_weights,
            plan=plan,
            zero_copy=True,
        )
        torch.cuda.synchronize(device)
        reused_seconds = time.perf_counter() - begin

        assert reused_plan is plan
        assert reused_weights is not None
        torch.testing.assert_close(reused_weights, route_weights, rtol=0, atol=0)
        torch.testing.assert_close(reused_output, output, rtol=0, atol=0)

        metrics = torch.tensor(
            [cold_seconds, reused_seconds], dtype=torch.float64, device=device
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        if rank == 0:
            cold_seconds, reused_seconds = metrics.cpu().tolist()
            print(
                "MOONEP_REAL_SHAPE_PASS "
                f"S={S} H={H} K={K} E={E} I={I} R={R} B={B} "
                f"token_padding={TOKEN_PADDING} "
                f"dynamic_slots={int(dynamic_slots.item())} "
                f"cold_s={cold_seconds:.3f} reuse_s={reused_seconds:.3f}",
                flush=True,
            )
    finally:
        if ep is not None:
            ep.close()
        ms.shmem_barrier_all()
        ms.shmem_finalize()
        dist.destroy_process_group()


def test_real_shape_forward() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != R:
        pytest.skip("run with torchrun --nproc-per-node=8 on gfx950")
    _run_real_shape_forward()


if __name__ == "__main__":
    _run_real_shape_forward()
