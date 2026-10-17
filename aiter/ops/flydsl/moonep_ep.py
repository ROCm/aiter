# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Correctness-first BF16 MoonEP forward orchestration for gfx950."""

from __future__ import annotations

import torch
import torch.distributed as dist

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
    weight-slot prefetch, variable-M expert GEMM, and direct peer combine.
    Route weights are applied to expert rows before MoonEP's unweighted K-sum.
    Expert compute is ``down(silu(gate(x)) * up(x))`` in BF16.
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
        self._unit_route_weights = torch.ones(
            (config.num_tokens, config.top_k),
            dtype=torch.float32,
            device=self.dispatch_op.device,
        )
        self._full_gate_weight: torch.Tensor | None = None
        self._full_up_weight: torch.Tensor | None = None
        self._full_down_weight: torch.Tensor | None = None
        self._closed = False

    @property
    def destroyed(self) -> bool:
        return self._closed

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

    def prefetch_weight(
        self,
        plan: MoonEPReferencePlan,
        *,
        full_gate_weight: torch.Tensor,
        full_up_weight: torch.Tensor,
        full_down_weight: torch.Tensor,
    ) -> None:
        """Fill ``[E:E+B]`` weight slots using MoonEP's forward layout."""

        if self._closed:
            raise RuntimeError("MoonEP EP is closed")
        if plan.config != self.config:
            raise ValueError("plan config does not match this EP")
        e = self.config.num_experts
        b = int(self.config.prefetch_slots)
        expected = {
            "gate": (e + b, self.hidden_dim, self.intermediate_dim),
            "up": (e + b, self.hidden_dim, self.intermediate_dim),
            "down": (e + b, self.intermediate_dim, self.hidden_dim),
        }
        tensors = {
            "gate": full_gate_weight,
            "up": full_up_weight,
            "down": full_down_weight,
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(f"full_{name}_weight has the wrong shape")
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError(f"full_{name}_weight must be contiguous BF16")
            if tensor.device != self.dispatch_op.device:
                raise ValueError(f"full_{name}_weight is on the wrong device")

        begin = self.config.rank * self.config.experts_per_rank
        end = begin + self.config.experts_per_rank
        self.load_home_weights(
            full_gate_weight[begin:end],
            full_up_weight[begin:end],
            full_down_weight[begin:end],
        )
        self._full_gate_weight = full_gate_weight
        self._full_up_weight = full_up_weight
        self._full_down_weight = full_down_weight
        full_gate_weight[e:].copy_(
            self.gate_op.prefetch(plan.experts_to_copy)
        )
        full_up_weight[e:].copy_(
            self.up_op.prefetch(plan.experts_to_copy)
        )
        full_down_weight[e:].copy_(
            self.down_op.prefetch(plan.experts_to_copy)
        )

    def dispatch(
        self,
        hidden_sh: torch.Tensor,
        route_weights_sk: torch.Tensor | None = None,
        topk_experts_sk: torch.Tensor | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        plan: MoonEPReferencePlan | None = None,
        *,
        zero_copy: bool = False,
    ):
        """Synchronous forward dispatch with MoonEP-compatible returns."""

        fresh_plan = plan is None
        if fresh_plan:
            if topk_experts_sk is None or tokens_per_expert is None:
                raise ValueError("routing inputs are required when plan is absent")
            if tokens_per_expert.ndim == 1:
                gathered = [torch.empty_like(tokens_per_expert) for _ in range(
                    self.config.world_size
                )]
                dist.all_gather(gathered, tokens_per_expert)
                tokens_per_expert = torch.stack(gathered)
            plan = build_reference_plan(
                self.config, topk_experts_sk, tokens_per_expert
            )
        assert plan is not None
        dispatch_weights = (
            route_weights_sk
            if route_weights_sk is not None
            else self._unit_route_weights
        )
        hidden_nvsh, weights_nvs = self.dispatch_op.dispatch(
            hidden_sh, dispatch_weights, plan
        )
        if not zero_copy:
            hidden_nvsh = hidden_nvsh.clone()
            weights_nvs = weights_nvs.clone()
        return (
            hidden_nvsh,
            weights_nvs if route_weights_sk is not None else None,
            plan.cu_seqlens if fresh_plan else None,
            plan,
        )

    def combine(
        self,
        plan: MoonEPReferencePlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor | None = None,
        *,
        zero_copy: bool = False,
    ):
        """Synchronous forward combine with MoonEP-compatible returns."""

        if tuple(hidden_nvsh.shape) != tuple(self.dispatch_op.recv_hidden.shape):
            raise ValueError("hidden_nvsh has the wrong shape")
        if zero_copy:
            if hidden_nvsh.data_ptr() != self.dispatch_op.recv_hidden.data_ptr():
                raise ValueError("zero-copy combine requires the dispatch shard view")
            if route_weights_nvs is not None and (
                route_weights_nvs.data_ptr()
                != self.dispatch_op.recv_route_weights.data_ptr()
            ):
                raise ValueError("zero-copy route weights must alias the shard")
        else:
            self.dispatch_op.recv_hidden.copy_(hidden_nvsh)
            if route_weights_nvs is not None:
                self.dispatch_op.recv_route_weights.copy_(route_weights_nvs)

        hidden_sh, gathered = self.dispatch_op.combine(plan)
        return (
            hidden_sh.clone(),
            gathered.clone() if route_weights_nvs is not None else None,
            None,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        route_weights: torch.Tensor | None = None,
        *,
        topk_experts: torch.Tensor | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        plan: MoonEPReferencePlan | None = None,
        zero_copy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, MoonEPReferencePlan]:
        """Run one complete correctness-first BF16 EP forward."""

        if self._closed:
            raise RuntimeError("MoonEP EP is closed")
        dispatched, dispatched_weights, _, plan = self.dispatch(
            hidden,
            route_weights,
            topk_experts,
            tokens_per_expert,
            plan,
            zero_copy=zero_copy,
        )
        has_route_weights = route_weights is not None
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
            full_gate=self._full_gate_weight,
            full_up=self._full_up_weight,
            full_down=self._full_down_weight,
        )
        if has_route_weights:
            self.dispatch_op.expert_output.mul_(
                dispatched_weights[:, None]
            )
        combine_hidden = self.dispatch_op.expert_output
        if zero_copy:
            self.dispatch_op.recv_hidden.copy_(combine_hidden)
            combine_hidden = self.dispatch_op.recv_hidden
        result, gathered_weights, _ = self.combine(
            plan,
            combine_hidden,
            dispatched_weights,
            zero_copy=zero_copy,
        )
        return (
            result,
            gathered_weights if has_route_weights else None,
            plan,
        )

    def close(self) -> None:
        """Collectively release symmetric EP buffers in reverse order."""

        if self._closed:
            return
        self.down_op.close()
        self.up_op.close()
        self.gate_op.close()
        self.dispatch_op.close()
        self._closed = True

    destroy = close


__all__ = ["MoonEPBF16ReferenceEP"]
