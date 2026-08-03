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
    hipblaslt_moonep_grouped_gemm_reference,
)


class MoonEPBF16ReferenceEP:
    """End-to-end MoonEP communication plus expert-projection baseline.

    The physical EP path is complete: planning, direct peer dispatch, dynamic
    weight-slot prefetch, variable-M expert GEMM, and direct peer weighted
    combine.  Expert compute is deliberately one BF16 ``H x H`` projection so
    communication correctness stays independent of a model-specific gated MLP.
    """

    def __init__(
        self,
        config: MoonEPPlanConfig,
        hidden_dim: int,
        *,
        dispatch_block_num: int = 128,
        prefetch_block_num: int = 128,
        warp_num_per_block: int = 4,
    ) -> None:
        self.config = config
        self.hidden_dim = hidden_dim
        self.dispatch_op = MoonEPPreplannedDispatchOp(
            config,
            hidden_dim,
            block_num=dispatch_block_num,
            warp_num_per_block=warp_num_per_block,
        )
        self.weight_op = MoonEPWeightPrefetchOp(
            rank=config.rank,
            world_size=config.world_size,
            num_experts=config.num_experts,
            prefetch_slots=int(config.prefetch_slots),
            weight_shape=(hidden_dim, hidden_dim),
            block_num=prefetch_block_num,
            block_threads=warp_num_per_block * 64,
        )
        self._closed = False

    def load_home_weights(self, weights: torch.Tensor) -> None:
        """Publish this rank's ``[experts_per_rank,H,H]`` BF16 weights."""

        self.weight_op.load_home_weights(weights)

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

        dispatched, _, prefetched = self.dispatch_op.dispatch_and_prefetch(
            hidden, route_weights, plan, self.weight_op
        )
        hipblaslt_moonep_grouped_gemm_reference(
            dispatched,
            self.weight_op.home_weights,
            prefetched,
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
        self.weight_op.close()
        self.dispatch_op.close()
        self._closed = True


__all__ = ["MoonEPBF16ReferenceEP"]
