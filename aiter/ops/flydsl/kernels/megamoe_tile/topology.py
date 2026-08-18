# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogicalTopology:
    world_size: int
    gpus_per_node: int
    ep_rank_to_shmem_pe: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.gpus_per_node <= 0:
            raise ValueError("world_size and gpus_per_node must be positive")
        if self.world_size % self.gpus_per_node:
            raise ValueError("world_size must be divisible by gpus_per_node")
        if self.ep_rank_to_shmem_pe is not None:
            if len(self.ep_rank_to_shmem_pe) != self.world_size:
                raise ValueError("ep_rank_to_shmem_pe must cover the EP world")
            if len(set(self.ep_rank_to_shmem_pe)) != self.world_size:
                raise ValueError("ep_rank_to_shmem_pe must be one-to-one")

    @property
    def num_nodes(self) -> int:
        return self.world_size // self.gpus_per_node

    def node_of(self, rank: int) -> int:
        self._check_rank(rank)
        return rank // self.gpus_per_node

    def local_rank_of(self, rank: int) -> int:
        self._check_rank(rank)
        return rank % self.gpus_per_node

    def proxy_rank(self, destination_node: int, source_rank: int) -> int:
        """Rank-aligned proxy used by MORI InterNodeV1."""
        if not 0 <= destination_node < self.num_nodes:
            raise ValueError("destination_node is outside topology")
        return destination_node * self.gpus_per_node + self.local_rank_of(source_rank)

    def shmem_pe(self, ep_rank: int) -> int:
        self._check_rank(ep_rank)
        if self.ep_rank_to_shmem_pe is None:
            return ep_rank
        return int(self.ep_rank_to_shmem_pe[ep_rank])

    def proxy_pe(self, destination_node: int, source_rank: int) -> int:
        return self.shmem_pe(self.proxy_rank(destination_node, source_rank))

    def same_node(self, lhs: int, rhs: int) -> bool:
        return self.node_of(lhs) == self.node_of(rhs)

    def ranks_on_node(self, node: int) -> tuple[int, ...]:
        if not 0 <= node < self.num_nodes:
            raise ValueError("node is outside topology")
        base = node * self.gpus_per_node
        return tuple(range(base, base + self.gpus_per_node))

    def _check_rank(self, rank: int) -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside world_size={self.world_size}")


@dataclass(frozen=True)
class RoutePlan:
    """Per-route ownership plus per-token node de-duplication metadata."""

    destination_rank: torch.Tensor  # [M, K] int64
    destination_node: torch.Tensor  # [M, K] int64
    local_expert: torch.Tensor  # [M, K] int64
    node_mask: torch.Tensor  # [M, num_nodes] bool
    node_route_expected: torch.Tensor  # [M, num_nodes] int32

    @property
    def tokens(self) -> int:
        return int(self.destination_rank.shape[0])

    @property
    def topk(self) -> int:
        return int(self.destination_rank.shape[1])


def build_route_plan(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    topology: LogicalTopology,
) -> RoutePlan:
    """Build the accepted-routing snapshot used by dispatch and combine.

    Expert ownership is contiguous: ``owner = expert // experts_per_rank``.
    Invalid/sentinel expert IDs are rejected in the first implementation rather
    than silently dropped, because a dropped route would deadlock ready counters.
    """

    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must be [tokens, topk]")
    if num_experts % topology.world_size:
        raise ValueError("num_experts must be divisible by world_size")
    if topk_ids.numel() and ((topk_ids < 0).any() or (topk_ids >= num_experts).any()):
        raise ValueError("topk_ids contains sentinel/out-of-range experts")

    epr = num_experts // topology.world_size
    ids = topk_ids.to(torch.int64)
    destination_rank = torch.div(ids, epr, rounding_mode="floor")
    local_expert = ids - destination_rank * epr
    destination_node = torch.div(
        destination_rank, topology.gpus_per_node, rounding_mode="floor"
    )

    m = int(ids.shape[0])
    node_mask = torch.zeros(
        (m, topology.num_nodes), dtype=torch.bool, device=ids.device
    )
    node_expected = torch.zeros(
        (m, topology.num_nodes), dtype=torch.int32, device=ids.device
    )
    if ids.numel():
        node_mask.scatter_(1, destination_node, True)
        node_expected.scatter_add_(
            1,
            destination_node,
            torch.ones_like(destination_node, dtype=torch.int32),
        )

    return RoutePlan(
        destination_rank=destination_rank,
        destination_node=destination_node,
        local_expert=local_expert,
        node_mask=node_mask,
        node_route_expected=node_expected,
    )
