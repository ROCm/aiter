# SPDX-License-Identifier: MIT
"""Peer-indexed C2/C3 window slices, omitting the owning logical node.

The RX, ready and credit arrays enumerate source/consumer nodes in ascending
order, excluding self. Source planes in node_accumulator retain absolute node
indices. Parity strides always use the allocation geometry, including padding.
This is address planning; it does not implement posting or buffer lifetimes.
"""

from dataclasses import dataclass, replace


def peer_index(node: int, peer: int, nodes: int) -> int:
    if not 0 <= node < nodes or not 0 <= peer < nodes or peer == node:
        raise ValueError("peer must be another logical node")
    return peer - int(peer > node)


@dataclass(frozen=True)
class PeerRegion:
    region: object
    nodes: int

    def __post_init__(self):
        if self.nodes < 2:
            raise ValueError("at least two logical nodes are required")
        if not self.region.plane_bytes or self.region.plane_bytes % (self.nodes - 1):
            raise ValueError("region must contain equal parity-indexed peer slices")

    @property
    def peer_bytes(self) -> int:
        return self.region.plane_bytes // (self.nodes - 1)

    def delta(self, node: int, peer: int) -> int:
        return peer_index(node, peer, self.nodes) * self.peer_bytes

    def offset(self, node: int, peer: int, parity: int) -> int:
        return self.region.parity_offset(parity) + self.delta(node, peer)


@dataclass(frozen=True)
class PeerProtocolLayout:
    """A stage's payload/ready/credit slices backed by the C1 buffer plan."""

    nodes: int
    payload: PeerRegion
    ready: PeerRegion
    credit: PeerRegion

    @classmethod
    def create(cls, plan, stage: int):
        names = {
            1: ("remote_dispatch_rx", "remote_chunk_ready", "remote_chunk_credit"),
            2: ("remote_partial_rx", "return_group_ready", "return_consumed"),
        }
        if stage not in names:
            raise ValueError("stage must be 1 or 2")
        nodes = plan.world_size // plan.gpus_per_node
        return cls(
            nodes,
            *(PeerRegion(plan.region(stage, name), nodes) for name in names[stage])
        )


def add_bridge_window(plan, buffers):
    """Bounded relay scratch for the window probe, including plan-only mode.

    This is test storage. Keep padding equal to the existing rail windows and
    reject oversized scratch before constructing any communicator/allocation.
    """
    nodes = plan.world_size // plan.gpus_per_node
    width = max(
        PeerProtocolLayout.create(plan, stage).payload.peer_bytes for stage in (1, 2)
    )
    # Eight-GPU logical nodes are the physical RAIL nodes used by the probe:
    # every peer is aligned and the bridge holds generation markers only.
    # The virtual 4x4 topology still needs a full payload relay per peer.
    area = 256 if plan.gpus_per_node == 8 else (width + 8 + 255) // 256 * 256
    size = 2 * 2 * (nodes - 1) * area + 256
    padding = next(w.allocation_bytes for w in plan.windows if w.rdma_register)
    registered = (size + (64 << 20) - 1) // (64 << 20) * (64 << 20)
    if registered > padding:
        raise ValueError("logical-peer bridge exceeds the existing padded rail window")
    region = buffers.BufferRegion(
        0,
        "logical_peer_bridge",
        "peer_bridge",
        0,
        size,
        256,
        (2, size // 2),
        plan.region(1, "dispatch_staging").dtype,
        2,
    )
    bridge = buffers.BufferWindow("peer_bridge", (region,), padding, registered, True)
    return replace(plan, windows=(*plan.windows, bridge)), area
