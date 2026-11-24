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

from .layout import STAGE1_RAIL_REGIONS, STAGE2_RAIL_REGIONS, _align_up

logger = logging.getLogger(__name__)

# Upper bound of one rail MR (CCO dmabuf/IOVA-zero path). Very large single
# MRs hard-hung whole nodes; rail regions are packed into windows of at most
# this size instead.
MAX_CCO_RAIL_MR_BYTES = 1 << 30
CCO_ALLOCATION_ALIGNMENT = 64 << 20
# The local (LSA-only) window's allocation size. A second operator on the same
# pooled communicator places it at a nonzero slot offset, where hipMemSetAccess
# failed with invalid argument for 192 and 320 MiB but passed for 256 and 512
# MiB (gfx950, MORI b80cbe4 and #996, GDR on or off).
LOCAL_ALLOCATION_ALIGNMENT = 256 << 20
# Read and written only by this GPU (GMM1 output, GEMM2 input): ordinary device
# memory outside every CCO window, so it is neither LSA-mapped nor registered.
PRIVATE_REGIONS = frozenset(("h1_output_q", "h1_output_scale"))


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
    # False: a plain device allocation (PRIVATE_REGIONS), not a CCO window.
    cco: bool = True


@dataclass(frozen=True)
class CcoBufferPlan:
    world_size: int
    gpus_per_node: int
    num_qp: int
    windows: tuple[BufferWindow, ...]

    @property
    def regions(self) -> tuple[BufferRegion, ...]:
        return tuple(region for window in self.windows for region in window.regions)

    @classmethod
    def create(cls, stage1, stage2) -> CcoBufferPlan:
        """Pack whole regions; reject regions that need finer MR chunking.

        Rail allocations have equal padded sizes, while each NIC registers
        only its occupied prefix. Local and rail windows require distinct
        communicators/VA spaces.
        """
        for field in ("world_size", "gpus_per_node", "parity_depth", "num_qp"):
            if getattr(stage1, field) != getattr(stage2, field):
                raise ValueError(f"Stage1/Stage2 {field} must agree")
        mr_limit = (
            MAX_CCO_RAIL_MR_BYTES // CCO_ALLOCATION_ALIGNMENT * CCO_ALLOCATION_ALIGNMENT
        )
        local_regions = []
        local_end = 0
        private_regions = []
        private_end = 0
        rail_windows = []
        for stage, layout, rail_names in (
            (1, stage1, STAGE1_RAIL_REGIONS),
            (2, stage2, STAGE2_RAIL_REGIONS),
        ):
            missing = set(rail_names) - {r.name for r in layout.regions}
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
                            f"exceeding the {mr_limit >> 20} MiB rail MR bound; lower max_tok_per_rank "
                            f"or set max_recv_routes_per_rank"
                        )
                    window = f"s{stage}_rail{index}"
                elif stage == 1 and region.name in PRIVATE_REGIONS:
                    offset = _align_up(private_end, region.alignment)
                    window = "private"
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
                elif window == "private":
                    private_regions.append(placed)
                    private_end = offset + region.nbytes
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
        local_bytes = _align_up(local_end, LOCAL_ALLOCATION_ALIGNMENT)
        local = BufferWindow(
            "local", tuple(local_regions), local_bytes, local_bytes, False
        )
        private = (
            (
                BufferWindow(
                    "private", tuple(private_regions), private_end, 0, False, False
                ),
            )
            if private_regions
            else ()
        )
        return cls(
            stage1.world_size,
            stage1.gpus_per_node,
            stage1.num_qp,
            (
                local,
                *(replace(w, allocation_bytes=padding) for w in rail_windows),
                *private,
            ),
        )

    def region(self, stage: int, name: str) -> BufferRegion:
        for window in self.windows:
            for region in window.regions:
                if region.stage == stage and region.name == name:
                    return region
        raise KeyError((stage, name))

    def per_rank_vmm(self, *, rail: bool) -> int:
        return max(
            128 << 20,
            sum(
                w.allocation_bytes
                for w in self.windows
                if w.cco and w.rdma_register == rail
            ),
        )


class _PrivateWindow:
    """A plain device allocation in a window slot: local pointer only.

    Its table rows carry handle 0 and offset 0; only region_local_address may
    reach these regions (no LSA or rail access exists for them)."""

    handle = 0

    def __init__(self, tensor):
        self.tensor = tensor
        self.local_ptr = tensor.data_ptr()


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
        # Stock MORI has no local-only registration: on a cross-node communicator
        # every window gets an MR, so the local window is registered whole too.
        local_only = (
            "rdma_register"
            in inspect.signature(local_communicator.register_window).parameters
        )
        for comm in (local_communicator, rail_communicator):
            if comm.nranks != plan.world_size or comm.rank != local_communicator.rank:
                raise ValueError(
                    "buffer plan and communicators must have matching ranks/world size"
                )
        self.plan = plan
        self.gda_context_count = gda_context_count
        self.device = torch.device("cuda", torch.cuda.current_device())
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
                if not spec.cco:
                    self.windows[spec.name] = _PrivateWindow(
                        torch.empty(
                            spec.allocation_bytes, dtype=torch.uint8, device=self.device
                        )
                    )
                    continue
                comm = rail_communicator if spec.rdma_register else local_communicator
                mem = comm.alloc_mem(spec.allocation_bytes)
                self._resources.append(mem)
                win = (
                    comm.register_window(
                        mem.ptr, spec.registered_bytes, rdma_register=spec.rdma_register
                    )
                    if local_only
                    else comm.register_window(mem.ptr, spec.registered_bytes)
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
            private = {w.name for w in plan.windows if not w.cco}
            rows = [
                (
                    self.windows[region.window].local_ptr + region.offset,
                    self.windows[region.window].handle,
                    0 if region.window in private else region.offset,
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
        self.windows.clear()
        if error is not None:
            raise error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_):
        self.close()
