# SPDX-License-Identifier: MIT
"""Route logical rail peers over physical CCO rail and LSA connections.

The 4x4 development topology has two physical eight-GPU hosts. CCO's LSA
team therefore still has eight ranks and its GDA rail team still has two
physical nodes. A logical peer on the other host with a different physical
LSA index is reached through the aligned rank, then one local LSA hop.

The same plan also describes four real four-GPU hosts, where every logical
peer is directly rail aligned and no relay is needed. This module only plans
routes; the device protocol owns payload, ready and credit ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransferKind = Literal["lsa", "rail", "rail_relay"]


@dataclass(frozen=True)
class PeerTransfer:
    source_rank: int
    destination_rank: int
    kind: TransferKind
    rail_node: int | None = None
    bridge_rank: int | None = None


@dataclass(frozen=True)
class VirtualNodeTopology:
    world_size: int
    logical_gpus_per_node: int
    physical_gpus_per_node: int

    def __post_init__(self) -> None:
        if self.world_size < 2 or self.logical_gpus_per_node < 1:
            raise ValueError("world_size and logical_gpus_per_node must be positive")
        if self.physical_gpus_per_node < self.logical_gpus_per_node:
            raise ValueError("a physical node must contain a whole logical node")
        if self.world_size % self.physical_gpus_per_node:
            raise ValueError("world_size must divide into physical nodes")
        if self.physical_gpus_per_node % self.logical_gpus_per_node:
            raise ValueError("logical nodes must divide each physical node")

    @property
    def logical_nodes(self) -> int:
        return self.world_size // self.logical_gpus_per_node

    @property
    def physical_nodes(self) -> int:
        return self.world_size // self.physical_gpus_per_node

    def _check_rank(self, rank: int) -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside world_size={self.world_size}")

    def logical_node(self, rank: int) -> int:
        self._check_rank(rank)
        return rank // self.logical_gpus_per_node

    def physical_node(self, rank: int) -> int:
        self._check_rank(rank)
        return rank // self.physical_gpus_per_node

    def physical_lsa_rank(self, rank: int) -> int:
        self._check_rank(rank)
        return rank % self.physical_gpus_per_node

    def aligned_peer(self, source_rank: int, destination_node: int) -> PeerTransfer:
        """Plan a logical rail send to the same lane on ``destination_node``."""
        self._check_rank(source_rank)
        if not 0 <= destination_node < self.logical_nodes:
            raise ValueError(f"logical node {destination_node} is out of range")
        if destination_node == self.logical_node(source_rank):
            raise ValueError("a logical rail send needs another logical node")

        lane = source_rank % self.logical_gpus_per_node
        destination_rank = destination_node * self.logical_gpus_per_node + lane
        source_physical = self.physical_node(source_rank)
        destination_physical = self.physical_node(destination_rank)
        if source_physical == destination_physical:
            return PeerTransfer(source_rank, destination_rank, "lsa")

        source_lsa = self.physical_lsa_rank(source_rank)
        bridge_rank = destination_physical * self.physical_gpus_per_node + source_lsa
        if bridge_rank == destination_rank:
            return PeerTransfer(
                source_rank, destination_rank, "rail", destination_physical
            )
        return PeerTransfer(
            source_rank,
            destination_rank,
            "rail_relay",
            destination_physical,
            bridge_rank,
        )
