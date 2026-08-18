# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import HierMegaMoETileConfig


@dataclass
class HierWorkspace:
    dispatch_tx: torch.Tensor
    dispatch_rx: torch.Tensor
    dispatch_ready: torch.Tensor
    dispatch_credit: torch.Tensor
    partial_tx: torch.Tensor
    partial_rx: torch.Tensor
    partial_ready: torch.Tensor
    partial_credit: torch.Tensor

    @classmethod
    def allocate(
        cls,
        config: HierMegaMoETileConfig,
        *,
        device: torch.device | str,
    ) -> "HierWorkspace":
        """Allocate the copy-stub workspace.

        The same shapes are the contract for future symmetric SHMEM allocation.
        Data buffers are byte tensors; control uses int64 generations.
        """

        nodes = config.num_logical_nodes
        ring = config.ring_depth
        chunk = config.chunk_bytes
        data_shape = (nodes, ring, chunk)
        ctrl_shape = (nodes, ring)
        return cls(
            dispatch_tx=torch.empty(data_shape, dtype=torch.uint8, device=device),
            dispatch_rx=torch.empty(data_shape, dtype=torch.uint8, device=device),
            dispatch_ready=torch.zeros(ctrl_shape, dtype=torch.int64, device=device),
            dispatch_credit=torch.zeros(ctrl_shape, dtype=torch.int64, device=device),
            partial_tx=torch.empty(data_shape, dtype=torch.uint8, device=device),
            partial_rx=torch.empty(data_shape, dtype=torch.uint8, device=device),
            partial_ready=torch.zeros(ctrl_shape, dtype=torch.int64, device=device),
            partial_credit=torch.zeros(ctrl_shape, dtype=torch.int64, device=device),
        )

    def nbytes(self) -> int:
        tensors = (
            self.dispatch_tx,
            self.dispatch_rx,
            self.dispatch_ready,
            self.dispatch_credit,
            self.partial_tx,
            self.partial_rx,
            self.partial_ready,
            self.partial_credit,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @classmethod
    def allocate_symmetric(cls, config: HierMegaMoETileConfig) -> "HierWorkspace":
        """Allocate the real MORI data/control plane.

        Every SHMEM PE must call this method in the same order with identical
        shapes.  Ordinary ``torch.empty`` is intentionally not accepted by the
        address-based RDMA API because it has no symmetric heap lkey/offset.
        """

        from mori.shmem.tensor_utils import mori_shmem_create_tensor

        nodes = config.num_logical_nodes
        ring = config.ring_depth
        chunk = config.chunk_bytes
        data_shape = (nodes, ring, chunk)
        ctrl_shape = (nodes, ring)

        def alloc(shape, dtype):
            return mori_shmem_create_tensor(shape, dtype)

        return cls(
            dispatch_tx=alloc(data_shape, torch.uint8),
            dispatch_rx=alloc(data_shape, torch.uint8),
            dispatch_ready=alloc(ctrl_shape, torch.int64).zero_(),
            dispatch_credit=alloc(ctrl_shape, torch.int64).zero_(),
            partial_tx=alloc(data_shape, torch.uint8),
            partial_rx=alloc(data_shape, torch.uint8),
            partial_ready=alloc(ctrl_shape, torch.int64).zero_(),
            partial_credit=alloc(ctrl_shape, torch.int64).zero_(),
        )


@dataclass
class PersistentH1Workspace:
    """Local control plane for one persistent H1 launch geometry.

    ``entry_count`` remains monotonic across launches. Ticket 0 resets the
    sharded work heads and publishes ``epoch_gate`` on device, so normal reuse
    requires no host memset. A workspace must not be shared concurrently or
    across different worker-grid sizes.
    """

    entry_count: torch.Tensor
    epoch_gate: torch.Tensor
    work_head: torch.Tensor
    work_shards: int
    worker_blocks: int
    bound_stream_id: int | None = None
    last_generation: int | None = None

    @classmethod
    def allocate(
        cls,
        *,
        work_shards: int = 8,
        worker_blocks: int = 192,
        device: torch.device | str,
    ) -> "PersistentH1Workspace":
        if work_shards not in (1, 2, 4, 8):
            raise ValueError("work_shards must be one of 1, 2, 4, 8")
        if worker_blocks < work_shards:
            raise ValueError(
                "worker_blocks must be >= work_shards so every ticket shard "
                "has a claimant"
            )
        dev = torch.device(device)
        return cls(
            entry_count=torch.zeros(1, dtype=torch.int64, device=dev),
            epoch_gate=torch.zeros(1, dtype=torch.int32, device=dev),
            work_head=torch.zeros(
                work_shards * 16, dtype=torch.int32, device=dev
            ),
            work_shards=work_shards,
            worker_blocks=worker_blocks,
        )

    def validate_launch(self, kernel, *, generation: int, stream) -> None:
        """Bind geometry/stream and reject epoch contracts that would corrupt tickets."""

        if int(kernel.work_shards) != self.work_shards:
            raise ValueError(
                f"kernel WORK_SHARDS={kernel.work_shards} does not match "
                f"workspace work_shards={self.work_shards}"
            )
        if self.worker_blocks < int(kernel.min_worker_blocks):
            raise ValueError(
                f"worker_blocks={self.worker_blocks} is below kernel minimum "
                f"{kernel.min_worker_blocks}"
            )
        stream_id = int(getattr(stream, "cuda_stream", id(stream)))
        if self.bound_stream_id is None:
            self.bound_stream_id = stream_id
        elif self.bound_stream_id != stream_id:
            raise RuntimeError(
                "PersistentH1Workspace is already bound to another stream; "
                "synchronize and call release_stream_binding() before reuse"
            )
        generation = int(generation)
        if self.last_generation is not None and generation <= self.last_generation:
            raise ValueError(
                "persistent H1 signal generation must increase monotonically"
            )
        self.last_generation = generation

    def release_stream_binding(self) -> None:
        """Allow rebinding after the caller has synchronized prior GPU work."""

        self.bound_stream_id = None
