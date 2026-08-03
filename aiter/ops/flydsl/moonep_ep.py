# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Correctness-first BF16 MoonEP forward orchestration for gfx950."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.kernels.moonep_dispatch_op import (
    MoonEPPreplannedDispatchOp,
)
from aiter.ops.flydsl.kernels.moonep_weight_prefetch import (
    MoonEPWeightPrefetchOp,
)
from aiter.ops.flydsl.moonep import (
    MoonEPPlanConfig,
    MoonEPReferencePlan,
    build_reference_plan,
    hipblaslt_moonep_mlp_reference,
)


class MoonEPBF16ReferenceEP:
    """End-to-end MoonEP communication plus a gated expert MLP.

    The physical EP path is complete: planning, direct peer dispatch, dynamic
    weight-slot prefetch, variable-M expert GEMM, and direct peer weighted
    combine. Expert compute is ``down(silu(gate(x)) * up(x))`` in BF16.
    """

    def __init__(
        self,
        config: MoonEPPlanConfig,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        dispatch_block_num: int = 128,
        prefetch_block_num: int = 128,
        warp_num_per_block: int = 4,
    ) -> None:
        self.config = config
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dispatch_op = MoonEPPreplannedDispatchOp(
            config,
            hidden_dim,
            block_num=dispatch_block_num,
            warp_num_per_block=warp_num_per_block,
        )
        common = dict(
            rank=config.rank,
            world_size=config.world_size,
            num_experts=config.num_experts,
            prefetch_slots=int(config.prefetch_slots),
            block_num=prefetch_block_num,
            block_threads=warp_num_per_block * 64,
        )
        self.gate_op = MoonEPWeightPrefetchOp(
            weight_shape=(hidden_dim, intermediate_dim), **common
        )
        self.up_op = MoonEPWeightPrefetchOp(
            weight_shape=(hidden_dim, intermediate_dim), **common
        )
        self.down_op = MoonEPWeightPrefetchOp(
            weight_shape=(intermediate_dim, hidden_dim), **common
        )
        self._closed = False

    def load_home_weights(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> None:
        """Publish this rank's resident gate/up/down expert weights."""

        self.gate_op.load_home_weights(gate)
        self.up_op.load_home_weights(up)
        self.down_op.load_home_weights(down)

    def forward(
        self,
        hidden: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        topk_experts: torch.Tensor | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        plan: MoonEPReferencePlan | None = None,
    ) -> tuple[torch.Tensor, MoonEPReferencePlan]:
        """Run one complete correctness-first BF16 EP forward."""

        if self._closed:
            raise RuntimeError("MoonEP EP is closed")
        if plan is None:
            if topk_experts is None or tokens_per_expert is None:
                raise ValueError("routing inputs are required when plan is absent")
            plan = build_reference_plan(
                self.config, topk_experts, tokens_per_expert
            )

        dispatched, _ = self.dispatch_op.dispatch(hidden, route_weights, plan)
        prefetched_gate = self.gate_op.prefetch(plan.experts_to_copy)
        prefetched_up = self.up_op.prefetch(plan.experts_to_copy)
        prefetched_down = self.down_op.prefetch(plan.experts_to_copy)
        hipblaslt_moonep_mlp_reference(
            dispatched,
            self.gate_op.home_weights,
            self.up_op.home_weights,
            self.down_op.home_weights,
            prefetched_gate,
            prefetched_up,
            prefetched_down,
            plan.cu_seqlens,
            plan.group_expert_ids,
            rank=self.config.rank,
            experts_per_rank=self.config.experts_per_rank,
            num_experts=self.config.num_experts,
            output=self.dispatch_op.expert_output,
        )
        return self.dispatch_op.combine(route_weights, plan), plan

    def close(self) -> None:
        """Collectively release symmetric EP buffers in reverse order."""

        if self._closed:
            return
        self.down_op.close()
        self.up_op.close()
        self.gate_op.close()
        self.dispatch_op.close()
        self._closed = True


__all__ = ["MoonEPBF16ReferenceEP"]
