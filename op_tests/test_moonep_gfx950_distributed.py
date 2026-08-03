# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Two-rank gfx950 smoke test for MoonEP preplanned direct-P2P dispatch.

Run directly on a single two-GPU node:

    torchrun --standalone --nproc-per-node=2 \
        op_tests/test_moonep_gfx950_distributed.py

The normal one-process pytest collection skips this test.
"""

from __future__ import annotations

import os

import mori.shmem as ms
import pytest
import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.moonep_dispatch_op import (
    MoonEPPreplannedDispatchOp,
)
from aiter.ops.flydsl.moonep import (
    MoonEPPlanConfig,
    build_reference_plan,
    hipblaslt_grouped_gemm_reference,
)
from aiter.ops.flydsl.moonep_ep import MoonEPBF16ReferenceEP


def _share_shmem_unique_id(rank: int) -> bytes:
    objects = [ms.shmem_get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    unique_id = objects[0]
    assert isinstance(unique_id, bytes) and len(unique_id) == 128
    return unique_id


def _run_two_rank_direct_scatter() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    unique_id = _share_shmem_unique_id(rank)
    status = ms.shmem_init_attr(
        ms.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world_size, unique_id
    )
    assert status == 0

    op = None
    prefetch_op = None
    ep = None
    try:
        num_tokens = 4
        hidden_dim = 128
        config = MoonEPPlanConfig(
            rank=rank,
            world_size=world_size,
            num_tokens=num_tokens,
            top_k=1,
            num_experts=4,
            prefetch_slots=2,
        )
        # Both sources send tokens 0/1 to expert 0 (rank 0) and tokens 2/3
        # to expert 2 (rank 1).  Each destination receives four exact rows.
        local_topk = torch.tensor(
            [[0], [0], [2], [2]], dtype=torch.int32, device="cuda"
        )
        tokens_per_expert = torch.tensor(
            [[2, 0, 2, 0], [2, 0, 2, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        plan = build_reference_plan(config, local_topk, tokens_per_expert)

        row_values = (
            torch.arange(num_tokens, device="cuda", dtype=torch.float32)
            + rank * 10
        ).to(torch.bfloat16)
        hidden = row_values[:, None].expand(-1, hidden_dim).contiguous()
        route_weights = (
            torch.arange(num_tokens, device="cuda", dtype=torch.float32)
            + rank * 10
            + 0.25
        )[:, None].contiguous()

        op = MoonEPPreplannedDispatchOp(config, hidden_dim, block_num=8)
        recv_hidden, recv_weights = op.dispatch(hidden, route_weights, plan)
        torch.cuda.synchronize()

        if rank == 0:
            expected_values = torch.tensor(
                [0, 1, 10, 11], dtype=torch.bfloat16, device="cuda"
            )
        else:
            expected_values = torch.tensor(
                [2, 3, 12, 13], dtype=torch.bfloat16, device="cuda"
            )
        expected_hidden = expected_values[:, None].expand(-1, hidden_dim)
        expected_weights = expected_values.to(torch.float32) + 0.25
        torch.testing.assert_close(recv_hidden, expected_hidden, rtol=0, atol=0)
        torch.testing.assert_close(recv_weights, expected_weights, rtol=0, atol=0)

        op.close()
        op = None

        # top_k=2 keeps both experts of each token on the same destination.
        # The second entry is negative-encoded, so only its weight crosses the
        # link; the local epilogue must copy the primary hidden row into its
        # expert segment.  token_padding=4 also exercises zero-fill.
        config = MoonEPPlanConfig(
            rank=rank,
            world_size=world_size,
            num_tokens=2,
            top_k=2,
            num_experts=4,
            prefetch_slots=2,
            token_padding=4,
        )
        local_topk = torch.tensor(
            [[0, 1], [2, 3]], dtype=torch.int32, device="cuda"
        )
        tokens_per_expert = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            dtype=torch.int32,
            device="cuda",
        )
        plan = build_reference_plan(config, local_topk, tokens_per_expert)
        row_values = (
            torch.arange(2, device="cuda", dtype=torch.float32) + rank * 10
        ).to(torch.bfloat16)
        hidden = row_values[:, None].expand(-1, hidden_dim).contiguous()
        route_weights = torch.tensor(
            [
                [rank * 10 + 0.25, rank * 10 + 0.50],
                [rank * 10 + 1.25, rank * 10 + 1.50],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        op = MoonEPPreplannedDispatchOp(config, hidden_dim, block_num=8)
        recv_hidden, recv_weights = op.dispatch(hidden, route_weights, plan)
        torch.cuda.synchronize()

        if rank == 0:
            expected_values = [0, 10, 0, 0, 0, 10, 0, 0]
            expected_weights = [0.25, 10.25, 0, 0, 0.50, 10.50, 0, 0]
        else:
            expected_values = [1, 11, 0, 0, 1, 11, 0, 0]
            expected_weights = [1.25, 11.25, 0, 0, 1.50, 11.50, 0, 0]
        expected_hidden = torch.tensor(
            expected_values, dtype=torch.bfloat16, device="cuda"
        )[:, None].expand(-1, hidden_dim)
        expected_weights_tensor = torch.tensor(
            expected_weights, dtype=torch.float32, device="cuda"
        )
        torch.testing.assert_close(
            recv_hidden[:8], expected_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(
            recv_weights[:8], expected_weights_tensor, rtol=0, atol=0
        )
        torch.testing.assert_close(
            op.recv_duplicate_src[:8],
            torch.tensor(
                [-1, -1, -1, -1, 0, 1, -1, -1],
                dtype=torch.int32,
                device="cuda",
            ),
            rtol=0,
            atol=0,
        )

        # Correctness baseline for the variable-M expert compute path: one
        # hipBLASLt GEMM per non-empty padded group.  Group g uses a scaled
        # identity matrix, making the expected output easy to verify exactly.
        group_count = config.num_experts + int(config.prefetch_slots)
        identity = torch.eye(
            hidden_dim, dtype=torch.bfloat16, device="cuda"
        )
        expert_weights = torch.stack(
            [identity * (group + 1) for group in range(group_count)]
        ).contiguous()
        gemm_output = hipblaslt_grouped_gemm_reference(
            recv_hidden,
            expert_weights,
            plan.cu_seqlens,
            output=op.expert_output,
        )
        if rank == 0:
            expected_gemm_values = [0, 10, 0, 0, 0, 20, 0, 0]
        else:
            expected_gemm_values = [3, 33, 0, 0, 4, 44, 0, 0]
        expected_gemm = torch.tensor(
            expected_gemm_values, dtype=torch.bfloat16, device="cuda"
        )[:, None].expand(-1, hidden_dim)
        torch.testing.assert_close(
            gemm_output[:8], expected_gemm, rtol=0, atol=0
        )

        combined_output = op.combine(route_weights, plan)
        torch.cuda.synchronize()
        expert_scales = (local_topk.to(torch.float32) + 1.0)
        expected_combined_values = (
            route_weights * expert_scales * row_values.to(torch.float32)[:, None]
        ).sum(dim=1).to(torch.bfloat16)
        expected_combined = expected_combined_values[:, None].expand(
            -1, hidden_dim
        )
        torch.testing.assert_close(
            combined_output, expected_combined, rtol=0, atol=0
        )

        op.close()
        op = None

        # Force hot expert 0 to spill from rank 0 onto rank 1.  The global
        # plan selects expert 0 for rank 1's first dynamic slot; verify that
        # rank 1 loads the owner rank's BF16 weight bytes directly.
        hot_config = MoonEPPlanConfig(
            rank=rank,
            world_size=world_size,
            num_tokens=4,
            top_k=1,
            num_experts=4,
            prefetch_slots=2,
        )
        hot_topk = torch.zeros((4, 1), dtype=torch.int32, device="cuda")
        hot_tokens_per_expert = torch.tensor(
            [[4, 0, 0, 0], [4, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        hot_plan = build_reference_plan(
            hot_config, hot_topk, hot_tokens_per_expert
        )
        ep = MoonEPBF16ReferenceEP(
            hot_config,
            hidden_dim,
            64,
            dispatch_block_num=8,
            prefetch_block_num=8,
        )
        op = ep.dispatch_op
        gate_base = torch.eye(
            hidden_dim, 64, dtype=torch.bfloat16, device="cuda"
        )
        up_base = gate_base.clone()
        down_base = torch.eye(
            64, hidden_dim, dtype=torch.bfloat16, device="cuda"
        )
        home_gate = torch.stack(
            [gate_base for _ in range(hot_config.experts_per_rank)]
        ).contiguous()
        home_up = torch.stack(
            [up_base for _ in range(hot_config.experts_per_rank)]
        ).contiguous()
        home_down = torch.stack(
            [down_base for _ in range(hot_config.experts_per_rank)]
        ).contiguous()
        ep.load_home_weights(home_gate, home_up, home_down)
        hot_values = (
            torch.arange(4, dtype=torch.float32, device="cuda") + rank * 10
        ).to(torch.bfloat16)
        hot_hidden = hot_values[:, None].expand(-1, hidden_dim).contiguous()
        hot_route_weights = (
            torch.arange(4, dtype=torch.float32, device="cuda")
            + rank * 10
            + 0.125
        )[:, None].contiguous()
        hot_output, returned_hot_plan = ep.forward(
            hot_hidden, hot_route_weights, plan=hot_plan
        )
        torch.cuda.synchronize()
        assert returned_hot_plan is hot_plan
        hot_recv = op.recv_hidden
        hot_recv_weights = op.recv_route_weights
        prefetched_gate = ep.gate_op.prefetched_weights
        expected_hot_values = torch.arange(
            rank * 10,
            rank * 10 + 4,
            dtype=torch.float32,
            device="cuda",
        ).to(torch.bfloat16)
        expected_hot_hidden = expected_hot_values[:, None].expand(
            -1, hidden_dim
        )
        torch.testing.assert_close(
            hot_recv[:4], expected_hot_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(
            hot_recv_weights[:4],
            expected_hot_values.to(torch.float32) + 0.125,
            rtol=0,
            atol=0,
        )
        if rank == 0:
            assert bool((hot_plan.experts_to_copy[rank] == -1).all())
        else:
            assert int(hot_plan.experts_to_copy[rank, 0]) == 0
            torch.testing.assert_close(
                prefetched_gate[0], gate_base, rtol=0, atol=0
            )

        ref_gate = hot_hidden @ gate_base
        ref_up = hot_hidden @ up_base
        ref_act = (
            torch.nn.functional.silu(ref_gate.to(torch.float32))
            * ref_up.to(torch.float32)
        ).to(torch.bfloat16)
        ref_expert = ref_act @ down_base
        expected_hot_output = (
            ref_expert.to(torch.float32) * hot_route_weights[:, 0, None]
        ).to(torch.bfloat16)
        torch.testing.assert_close(
            hot_output, expected_hot_output, rtol=1e-2, atol=2e-2
        )

        reused_hidden = (hot_hidden + 1.0).contiguous()
        reused_route_weights = (hot_route_weights * 0.5).contiguous()
        reused_output, reused_plan = ep.forward(
            reused_hidden,
            reused_route_weights,
            plan=returned_hot_plan,
        )
        torch.cuda.synchronize()
        assert reused_plan is returned_hot_plan
        reused_gate = reused_hidden @ gate_base
        reused_up = reused_hidden @ up_base
        reused_act = (
            torch.nn.functional.silu(reused_gate.to(torch.float32))
            * reused_up.to(torch.float32)
        ).to(torch.bfloat16)
        reused_expert = reused_act @ down_base
        expected_reused = (
            reused_expert.to(torch.float32)
            * reused_route_weights[:, 0, None]
        ).to(torch.bfloat16)
        torch.testing.assert_close(
            reused_output, expected_reused, rtol=1e-2, atol=2e-2
        )
    finally:
        if ep is not None:
            ep.close()
            op = None
        if prefetch_op is not None:
            prefetch_op.close()
        if op is not None:
            op.close()
        ms.shmem_barrier_all()
        ms.shmem_finalize()
        dist.destroy_process_group()


def test_two_rank_direct_scatter() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with torchrun --nproc-per-node=2 on gfx950")
    _run_two_rank_direct_scatter()


if __name__ == "__main__":
    _run_two_rank_direct_scatter()
