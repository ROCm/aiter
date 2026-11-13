# SPDX-License-Identifier: MIT
"""Separate MegaMoE's local payload from bounded CCO rail registrations.

The plan preserves each region's shape and parity stride, but assigns a new
window and offset. Kernels using it must pass those handles and offsets;
legacy kernels taking one arena pointer cannot consume this plan directly.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, replace
from typing import Any

from typing_extensions import Self

logger = logging.getLogger(__name__)

# Conservative guard for the tested CCO dmabuf/IOVA-zero path. This is not
# an Ionic hardware limit or a bound on MORI shmem's ordinary-address MRs.
MAX_CCO_RAIL_MR_BYTES = 1 << 30
CCO_ALLOCATION_ALIGNMENT = 64 << 20
STAGE1_RAIL_REGIONS = frozenset(
    (
        "dispatch_staging",
        "remote_dispatch_rx",
        "remote_chunk_ready",
        "remote_chunk_credit",
    )
)
STAGE2_RAIL_REGIONS = frozenset(
    (
        "node_accumulator",
        "remote_partial_rx",
        "return_group_ready",
        "return_consumed",
        "rail_idx",
    )
)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class BufferRegion:
    stage: int
    name: str
    window: str
    offset: int
    nbytes: int
    alignment: int
    shape: tuple[int, ...]
    dtype: Any
    parity_depth: int

    @property
    def plane_bytes(self) -> int:
        """Zero stride for persistent state, otherwise one parity plane."""
        if self.shape and self.shape[0] == self.parity_depth:
            return self.nbytes // self.parity_depth
        return 0

    def parity_offset(self, parity: int) -> int:
        if not 0 <= parity < self.parity_depth:
            raise ValueError("parity is outside parity_depth")
        if not self.shape or self.shape[0] != self.parity_depth:
            raise ValueError(f"{self.name} is not parity indexed")
        return self.offset + parity * (self.nbytes // self.parity_depth)


@dataclass(frozen=True)
class BufferWindow:
    name: str
    regions: tuple[BufferRegion, ...]
    allocation_bytes: int
    registered_bytes: int
    rdma_register: bool


@dataclass(frozen=True)
class CcoBufferPlan:
    world_size: int
    gpus_per_node: int
    num_qp: int
    windows: tuple[BufferWindow, ...]

    @property
    def regions(self) -> tuple[BufferRegion, ...]:
        return tuple(region for window in self.windows for region in window.regions)

    def region_index(self, stage: int, name: str) -> int:
        for index, region in enumerate(self.regions):
            if region.stage == stage and region.name == name:
                return index
        raise KeyError((stage, name))

    @classmethod
    def create(
        cls,
        stage1,
        stage2,
        *,
        max_mr_bytes: int = MAX_CCO_RAIL_MR_BYTES,
    ) -> CcoBufferPlan:
        """Pack whole regions; reject regions that need finer MR chunking.

        Rail allocations have equal padded sizes, while each NIC registers
        only its occupied prefix. Local and rail windows require distinct
        communicators/VA spaces, as measured on 47+51.
        """
        for field in ("world_size", "gpus_per_node", "parity_depth", "num_qp"):
            if getattr(stage1, field) != getattr(stage2, field):
                raise ValueError(f"Stage1/Stage2 {field} must agree")
        if not CCO_ALLOCATION_ALIGNMENT <= max_mr_bytes <= MAX_CCO_RAIL_MR_BYTES:
            raise ValueError(
                "max_mr_bytes must be between 64 MiB and the 1 GiB CCO guard"
            )
        mr_limit = max_mr_bytes // CCO_ALLOCATION_ALIGNMENT * CCO_ALLOCATION_ALIGNMENT
        local_regions = []
        local_end = 0
        rail_windows = []
        for stage, layout, rail_names in (
            (1, stage1, STAGE1_RAIL_REGIONS),
            (2, stage2, STAGE2_RAIL_REGIONS),
        ):
            missing = rail_names - {r.name for r in layout.regions}
            if missing:
                raise ValueError(
                    f"Stage{stage} missing rail regions: {sorted(missing)}"
                )
            rail_regions = []
            rail_end = 0
            index = 0
            for region in layout.regions:
                if stage == 1 and region.name == "stage2_embed":
                    continue  # Stage2 is packed separately, never counted twice.
                if region.name in rail_names:
                    offset = _align_up(rail_end, region.alignment)
                    if offset + region.nbytes > mr_limit and rail_regions:
                        rail_windows.append(
                            BufferWindow(
                                f"s{stage}_rail{index}",
                                tuple(rail_regions),
                                0,
                                _align_up(rail_end, CCO_ALLOCATION_ALIGNMENT),
                                True,
                            )
                        )
                        index += 1
                        rail_regions = []
                        rail_end = offset = 0
                    if offset + region.nbytes > mr_limit:
                        raise ValueError(
                            f"Stage{stage}.{region.name} needs {region.nbytes} bytes, "
                            f"exceeding the {mr_limit}-byte MR bound; split this region into chunks"
                        )
                    window = f"s{stage}_rail{index}"
                else:
                    offset = _align_up(local_end, region.alignment)
                    window = "local"
                placed = BufferRegion(
                    stage,
                    region.name,
                    window,
                    offset,
                    region.nbytes,
                    region.alignment,
                    region.shape,
                    region.dtype,
                    layout.parity_depth,
                )
                if window == "local":
                    local_regions.append(placed)
                    local_end = offset + region.nbytes
                else:
                    rail_regions.append(placed)
                    rail_end = offset + region.nbytes
            if rail_regions:
                rail_windows.append(
                    BufferWindow(
                        f"s{stage}_rail{index}",
                        tuple(rail_regions),
                        0,
                        _align_up(rail_end, CCO_ALLOCATION_ALIGNMENT),
                        True,
                    )
                )
        padding = max(w.registered_bytes for w in rail_windows)
        local_bytes = _align_up(local_end, CCO_ALLOCATION_ALIGNMENT)
        local = BufferWindow(
            "local", tuple(local_regions), local_bytes, local_bytes, False
        )
        return cls(
            stage1.world_size,
            stage1.gpus_per_node,
            stage1.num_qp,
            (local, *(replace(w, allocation_bytes=padding) for w in rail_windows)),
        )

    def region(self, stage: int, name: str) -> BufferRegion:
        for window in self.windows:
            for region in window.regions:
                if region.stage == stage and region.name == name:
                    return region
        raise KeyError((stage, name))

    @property
    def allocation_bytes(self) -> int:
        return sum(w.allocation_bytes for w in self.windows)

    def per_rank_vmm(self, *, rail: bool) -> int:
        return max(
            128 << 20,
            sum(w.allocation_bytes for w in self.windows if w.rdma_register == rail),
        )


class CcoBufferSet:
    """Own a plan's allocations/windows, borrowing two caller-owned comms.

    Creation/registration is collective in plan order. The caller initializes
    and destroys communicators collectively, and keeps this set alive while
    any kernels or graphs refer to its pointers. ``close`` synchronizes the
    owning device and attempts every release, propagating the first error.
    """

    def __init__(
        self,
        plan: CcoBufferPlan,
        local_communicator,
        rail_communicator,
        *,
        gda_context_count: int = 8,
        sdma_queue_count: int = 0,
    ):
        import torch
        from mori.cco import (
            GDA_CONNECTION_CROSSNODE,
            GDA_CONNECTION_NONE,
            GDA_CONNECTION_RAIL,
            CCODevCommRequirements,
        )

        if local_communicator is rail_communicator:
            raise ValueError(
                "local and rail buffers require distinct CCO communicators"
            )
        if gda_context_count < plan.num_qp:
            raise ValueError("GDA contexts must cover all Stage1 QPs")
        if sdma_queue_count not in (0, 1):
            raise ValueError("separated buffers support zero or one SDMA queue")
        if (
            "rdma_register"
            not in inspect.signature(local_communicator.register_window).parameters
        ):
            raise RuntimeError(
                "MORI CCO register_window(rdma_register=False) is required"
            )
        for comm in (local_communicator, rail_communicator):
            if comm.nranks != plan.world_size or comm.rank != local_communicator.rank:
                raise ValueError(
                    "buffer plan and communicators must have matching ranks/world size"
                )
        self.plan = plan
        self.gda_context_count = gda_context_count
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.memories = {}
        self.windows = {}
        self._resources = []
        self._closed = False
        self._device_table = None
        self.lsa_peer_base = 0
        self.gda_peer_stride = 1
        self.gda_peer_lane = 0
        self.gda_physical_width = 0
        try:
            for spec in plan.windows:
                comm = rail_communicator if spec.rdma_register else local_communicator
                mem = comm.alloc_mem(spec.allocation_bytes)
                self._resources.append(mem)
                self.memories[spec.name] = mem
                win = comm.register_window(
                    mem.ptr, spec.registered_bytes, rdma_register=spec.rdma_register
                )
                self._resources.append(win)
                self.windows[spec.name] = win
            for attr, comm, connection in (
                ("local_dev_comm", local_communicator, GDA_CONNECTION_NONE),
                ("rail_dev_comm", rail_communicator, GDA_CONNECTION_RAIL),
            ):
                if attr == "rail_dev_comm" and self.gda_peer_stride != 1:
                    connection = GDA_CONNECTION_CROSSNODE
                req = CCODevCommRequirements()
                req.gda_connection_type = connection
                req.gda_context_count = (
                    gda_context_count if connection != GDA_CONNECTION_NONE else 1
                )
                req.gda_signal_count = req.gda_counter_count = 0
                if attr == "local_dev_comm" and sdma_queue_count:
                    req.sdma_queue_count = sdma_queue_count
                req.lsa_barrier_count = req.rail_gda_barrier_count = (
                    req.barrier_count
                ) = 0
                dev_comm = comm.create_dev_comm(req)
                self._resources.append(dev_comm)
                setattr(self, attr, dev_comm)
                if attr == "local_dev_comm":
                    if (
                        sdma_queue_count
                        and int(getattr(dev_comm, "sdma_num_queue", 0) or 0)
                        < sdma_queue_count
                    ):
                        raise RuntimeError(
                            "MORI did not provide the requested local SDMA queue"
                        )
                    physical_width = dev_comm.lsa_size
                    logical_width = plan.gpus_per_node
                    if physical_width < logical_width or physical_width % logical_width:
                        raise ValueError(
                            "logical nodes must partition the physical CCO LSA team"
                        )
                    if dev_comm.lsa_rank != comm.rank % physical_width:
                        raise ValueError(
                            "logical nodes require contiguous CCO LSA rank groups"
                        )
                    self.lsa_peer_base = (
                        dev_comm.lsa_rank // logical_width
                    ) * logical_width
                    if physical_width != logical_width:
                        self.gda_peer_stride = logical_width
                        self.gda_peer_lane = comm.rank % logical_width
                        self.gda_physical_width = physical_width
                elif (dev_comm.lsa_size, dev_comm.lsa_rank) != (
                    self.local_dev_comm.lsa_size,
                    self.local_dev_comm.lsa_rank,
                ):
                    raise ValueError(
                        "local and rail communicators must share the physical LSA topology"
                    )
            # Stable runtime bindings for kernels and captured graphs. Each
            # row is [local region pointer, window handle, window offset,
            # parity stride]; all pointers refer to parity zero. Handles and
            # offsets come from the owning window, never a combined arena.
            rows = [
                (
                    self.windows[region.window].local_ptr + region.offset,
                    self.windows[region.window].handle,
                    region.offset,
                    region.plane_bytes,
                )
                for region in plan.regions
            ]
            if sdma_queue_count:
                # Preserve every region's four-word ABI. The trailing metadata
                # binds SDMA to the communicator that owns the local windows.
                rows.append((self.local_dev_comm.ptr, 0, 0, 0))
            self._device_table = torch.tensor(
                rows, dtype=torch.int64, device=self.device
            )
            # Consumers receive raw pointers and may launch on another stream.
            # Finish the one-time table upload before publishing those pointers.
            torch.cuda.synchronize(self.device)
        except Exception:
            try:
                self.close()
            except Exception:
                logger.exception("CCO buffer cleanup failed after allocation error")
            raise

    def address(self, stage: int, name: str, *, parity: int | None = None) -> int:
        if self._closed:
            raise RuntimeError("CCO buffer set is closed")
        region = self.plan.region(stage, name)
        offset = region.offset if parity is None else region.parity_offset(parity)
        return self.windows[region.window].local_ptr + offset

    @property
    def device_table_ptr(self) -> int:
        if self._closed:
            raise RuntimeError("CCO buffer set is closed")
        return self._device_table.data_ptr()

    def close(self) -> None:
        if self._closed:
            return
        import torch

        self._closed = True
        error = None
        try:
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            logger.exception("CCO buffer device synchronization failed")
            error = exc
        for resource in reversed(self._resources):
            try:
                resource.close()
            except Exception as exc:
                logger.exception("CCO buffer release failed")
                if error is None:
                    error = exc
        self._resources.clear()
        self._device_table = None
        self.memories.clear()
        self.windows.clear()
        if error is not None:
            raise error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_):
        self.close()
