# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared assets for the IPC collective host wrappers (all-reduce, all-to-all)."""

import ctypes
import logging
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.collectives_shared import ATOMS, clamp_grid_cap
from .kernels.tensor_shim import _preload_compiled, _run_compiled
from .quick_allreduce_ipc import UncachedIpcHeap

logger = logging.getLogger("aiter")

_SUPPORTED_ARCHS = ("gfx942", "gfx950")

# How the IPC inbox is allocated. The wire protocol is identical in every
# mode; only the memory type changes.
INBOX_MEMORY_MODES = ("auto", "uncached", "finegrained", "default")


def _cuda_index(device) -> int:
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"QuickAllReduceInt4 requires a CUDA device, got {device}")
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)
    return int(device)


def _resolve_inbox_flags(mode: str, world_size: int) -> tuple[int, str]:
    """(hipExtMallocWithFlags mode, resolved name) for an ``inbox_memory``.

    ``"auto"`` is ``uncached`` on xGMI and ``finegrained`` on PCIe, except at
    TP2, where it is ``uncached`` on PCIe too. The PCIe rule exists because
    uncached peer writes serialize per destination and collapse as the fanout
    widens. At TP2 every schedule rites to a single remote peer,
    so there is nothing to collapse, and an uncached inbox skips the L2 writeback
    a cacheable one pays at every publish.
    """
    if mode not in INBOX_MEMORY_MODES:
        raise ValueError(
            f"inbox_memory must be one of {INBOX_MEMORY_MODES}, got {mode!r}"
        )
    if mode == "auto":
        single_peer = int(world_size) == 2
        mode = "uncached" if single_peer or has_xgmi_peer_links() else "finegrained"
    flags = {
        "uncached": UncachedIpcHeap._HIP_DEVICE_MALLOC_UNCACHED,
        "finegrained": UncachedIpcHeap._HIP_DEVICE_MALLOC_FINEGRAINED,
        "default": UncachedIpcHeap._HIP_DEVICE_MALLOC_DEFAULT,
    }[mode]
    return flags, mode


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(
        self,
        *,
        spec,
        group,
        rank: int,
        world_size: int,
        inbox_flags: int,
    ):
        self.spec = spec
        self.launch = spec["launch"]
        self.super_tile = spec["super_tile"]
        self.grid = spec["grid"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self.block = spec["block"]
        self.skip_self = spec.get("skip_self", False)
        self._peer_bases = [None] * world_size
        self._buf_ptr = None
        self._meta_ptr = None
        self._gpu_peer_ptrs = None
        self._colors = None
        # Split-build exchange state inside the meta block; 0 when there is none.
        self._xchg = 0
        try:
            # The inbox is the only allocation peers write into, so it is the
            # only one whose memory type matters for fabric throughput.
            self._buf_ptr = UncachedIpcHeap.alloc(self.buf_bytes, inbox_flags)
            my_handle = UncachedIpcHeap.get_mem_handle_bytes(self._buf_ptr)
            all_meta = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, (my_handle, 0)
            )

            peer_ptrs = [0] * world_size
            for r in range(world_size):
                handle, off = all_meta[r]
                if r == rank:
                    peer_ptrs[r] = self._buf_ptr + off
                else:
                    base = int(UncachedIpcHeap.open_mem_handle(bytes(handle)))
                    self._peer_bases[r] = base
                    peer_ptrs[r] = base + off

            peer_bytes = world_size * 8
            color_bytes = self.grid * 4
            # A split fused one-shot's exchange words and per-workgroup ``prev``
            # values, line-aligned after the colours. Like them, only this rank's
            # own kernel touches it; ``alloc`` zeroes it, and zero words with
            # zero ``prev`` are a consistent start.
            xchg_bytes = int(spec.get("xchg_bytes", 0))
            xchg_off = (peer_bytes + color_bytes + 127) // 128 * 128
            meta_bytes = (
                xchg_off + xchg_bytes if xchg_bytes else peer_bytes + color_bytes
            )
            # Peer-pointer table and per-block colours: written by the host once
            # and by this rank's own kernel, never by a peer. Stays uncached in
            # every mode -- no cross-GPU visibility question, and it is a few
            # KiB.
            self._meta_ptr = UncachedIpcHeap.alloc_uncached(meta_bytes)
            self._gpu_peer_ptrs = self._meta_ptr
            self._colors = self._meta_ptr + peer_bytes
            UncachedIpcHeap.copy_host_to_device(
                self._gpu_peer_ptrs,
                (ctypes.c_int64 * world_size)(*peer_ptrs),
                peer_bytes,
            )
            UncachedIpcHeap.copy_host_to_device(
                self._colors,
                (ctypes.c_int32 * self.grid)(*([1] * self.grid)),
                color_bytes,
            )
            if xchg_bytes:
                self._xchg = self._meta_ptr + xchg_off
        except Exception:
            self.close()
            raise

    def close(self):
        for b in self._peer_bases:
            if b is not None:
                try:
                    UncachedIpcHeap.close_mem_handle(int(b))
                except RuntimeError:
                    pass
        self._peer_bases = []
        if self._meta_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._meta_ptr)
            except RuntimeError:
                pass
            self._meta_ptr = None
            self._gpu_peer_ptrs = None
            self._colors = None
            self._xchg = 0
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:
                pass
            self._buf_ptr = None


def _validate_ipc_process_group(group, *, rank: int) -> None:
    """Reject groups that cannot exchange HIP IPC handles or CPU-side metadata."""
    # Keep parallel_state lazy: this module is imported while aiter's AOT setup
    # is still initializing the top-level package.
    from aiter.dist.parallel_state import in_the_same_node_as

    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        raise ValueError(
            f"QuickAllReduceInt4 does not support NCCL process groups (got "
            f"{backend!r} on group rank {rank}): IPC handle exchange requires "
            "CPU-side broadcast_object_list."
        )

    same_node = in_the_same_node_as(group, source_rank=0)
    if not all(same_node):
        off_node = [r for r, ok in enumerate(same_node) if not ok]
        raise RuntimeError(
            "QuickAllReduceInt4 does not support multi-node process groups: HIP "
            f"IPC handles are node-local (ranks not on rank 0's node: {off_node})."
        )


# KFD io-link type for xGMI, from include/uapi/linux/kfd_sysfs.h. PCIe is 2.
_HSA_IOLINK_TYPE_XGMI = 11
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def has_xgmi_peer_links() -> bool:
    """Whether any GPU-to-GPU link on this host is xGMI rather than PCIe.

    Arch is not enough to make this call: an MI350X (xGMI) and an MI350P
    (PCIe-only) both report ``gfx950``, and the right inbox memory type is
    opposite on the two. KFD exposes the real link type per peer pair, so read
    that instead of guessing from the SKU.

    Both ``io_links`` and ``p2p_links`` have to be scanned. KFD only populates
    ``p2p_links`` for peers reachable indirectly (through a host bridge), so on
    a directly-connected mesh it holds nothing but the PCIe links to the CPU
    nodes and the xGMI peers appear solely under ``io_links``. Reading
    ``p2p_links`` alone reports "PCIe" on an 8-GPU all-xGMI MI350X, which flips
    ``inbox_memory="auto"`` to the fine-grained heap and silently costs the
    uncached fanout the kernel was designed around.

    Returns True when the topology cannot be read, which keeps the historical
    uncached allocation on any host we cannot classify -- the failure mode of
    guessing "PCIe" on an xGMI box is a silent perf regression on hardware
    where the current design is already optimal.

    The answer is per host, not per process group: one xGMI link anywhere
    classifies every group on the node as xGMI. That assumes a node is uniformly
    xGMI or uniformly PCIe, which holds for the single-node systems this targets.
    On a mixed host, a group of PCIe-only peers would get the xGMI policy and an
    uncached inbox; fixing that means classifying only the links between the
    group's own devices.
    """
    try:
        for subdir in ("io_links", "p2p_links"):
            for props in _KFD_NODES.glob(f"*/{subdir}/*/properties"):
                for line in props.read_text().splitlines():
                    field, _, value = line.partition(" ")
                    if field == "type" and int(value) == _HSA_IOLINK_TYPE_XGMI:
                        return True
        return False
    except (OSError, ValueError):
        logger.debug(
            "QuickAllReduceInt4: cannot read KFD topology; assuming xGMI",
            exc_info=True,
        )
        return True


def kernel_symbol(launch) -> str:
    """The JIT symbol a kernel factory stamped on its launch wrapper."""
    name = getattr(getattr(launch, "func", None), "__name__", None)
    if not name:
        return "?"
    return name.removeprefix("launch_")


def payload_probes(floors, lo: int, hi: int) -> tuple[int, ...]:
    """Payload sizes that between them select every config a ladder with rung
    *floors* assigns to ``lo..hi`` bytes (inclusive); ``()`` if none fit.

    Within one rung the choice is monotone in the payload -- a fixed config, or
    a super-tile taken once the tile count reaches a threshold -- so the two
    ends of each rung's slice of the range reach everything in it. Payloads are
    whole multiples of 16 B, so the probes are too.
    """
    step = 16
    lo = max(step, -(-int(lo) // step) * step)
    hi = int(hi) // step * step
    if lo > hi:
        return ()
    probes = {lo, hi}
    for floor in floors:
        first = -(-int(floor) // step) * step
        if lo < first <= hi:
            probes.update((first - step, first))
    return tuple(sorted(probes))


class _LadderedIpcOp:
    """A set of IPC engines, one per ladder rung, and the launch that picks one.

    Everything an IPC collective does *around* its kernel: one ``_StEngine``
    (inbox + compiled binary) per distinct ``(super_tile, block, skip_self)``,
    selection by payload size, the persistent-grid launch, preload and cleanup.
    Subclasses validate their payload, call :meth:`_init_engines` once, and
    launch through :meth:`_launch_eng`. A subclass whose kernel sees something
    other than the whole payload overrides :meth:`_kernel_nbytes`.
    """

    def _init_engines(
        self,
        *,
        group,
        device_index: int,
        rank: int,
        world_size: int,
        ladder: tuple,
        laddered: bool,
        build: Callable[..., dict],
        inbox_flags: int,
        arch: str,
        batch_publishes: bool,
        min_batch_blocks: int,
        grid_cap: int,
        label: str,
    ) -> None:
        """Build every engine the *ladder* rungs need.

        *ladder* is ``((min_bytes, super_tile, grid_cap, block, skip_self),
        ...)``, ascending; *laddered* says whether to select among the rungs
        by payload (``False`` pins the first). *build* is called with
        ``super_tile``, ``grid``, ``block`` and ``skip_self`` and returns a
        kernel spec.

        One engine per distinct ``(super_tile, block, skip_self)``, keyed that
        way in ``_by_cfg``. Each ``(block, skip_self)`` also gets an ST=1
        engine: ``_pick_cfg`` falls back to it when a payload has fewer tiles
        than the chosen super-tile, and the fallback has to share the rung's
        block -- a different tile size would change the tile count it was
        picked for. Engines are built in a fixed order because each does its
        own IPC handle exchange, which is a collective -- ranks disagreeing on
        the order would deadlock.

        The rungs go in first so an ST=1 that the ladder *sites* keeps its own
        cap. Only then is the fallback filled in, and at the smallest cap among
        its rungs rather than at the global default: the fallback fires only
        when a payload has fewer tiles than the super-tile it would otherwise
        take, so ``_grid_x`` there is bounded by that super-tile (<= 32) with a
        release fence, and by the chosen rung's own cap without one -- under
        128 either way. Seeding it with the 1216 default instead built a 194
        MiB inbox to launch at most 32 blocks into, and did it on every object
        ever constructed.
        """
        self.group = group
        self._device_index = int(device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._grid = int(grid_cap)
        self._batch_publishes = bool(batch_publishes)
        self._min_batch_blocks = int(min_batch_blocks)
        self._has_launched = False
        self._ladder = ladder if laddered else ()
        self._primary = ladder[0][1:2] + ladder[0][3:]
        self._by_cfg = {}

        caps = {}
        for _floor, st, rung_cap, b, ss in ladder:
            caps.setdefault((st, b, ss), rung_cap)
        for b, ss in {(b, ss) for _st, b, ss in list(caps)}:
            caps.setdefault(
                (1, b, ss),
                min(c for (_st, cb, css), c in caps.items() if (cb, css) == (b, ss)),
            )
        cu_count = int(
            torch.cuda.get_device_properties(self._device_index).multi_processor_count
        )
        lds_capacity = get_lds_capacity_bytes(arch)
        try:
            with torch.cuda.device(self._device_index):
                for key in sorted(caps):
                    st, b, ss = key
                    # A persistent kernel deadlocks if it launches more workgroups
                    # than fit, and the ranks have to agree on the number: take the
                    # minimum across the group so a heterogeneous node converges.
                    grid = clamp_grid_cap(
                        caps[key],
                        arch=arch,
                        world_size=self.world_size,
                        super_tile=st,
                        cu_count=cu_count,
                        block=b,
                    )
                    shared_grid = torch.tensor(grid, dtype=torch.int64)
                    dist.all_reduce(shared_grid, op=dist.ReduceOp.MIN, group=group)
                    spec = build(
                        super_tile=st,
                        grid=int(shared_grid.item()),
                        block=b,
                        skip_self=ss,
                    )
                    if spec["lds_bytes"] > lds_capacity:
                        raise ValueError(
                            f"{label} at block={b} needs {spec['lds_bytes']} B of "
                            f"LDS, over the {lds_capacity} B {arch} has"
                        )
                    self._by_cfg[key] = _StEngine(
                        spec=spec,
                        group=self.group,
                        rank=self.rank,
                        world_size=self.world_size,
                        inbox_flags=inbox_flags,
                    )
        except Exception:
            self.close()
            raise

        primary = self._by_cfg[self._primary]
        self.super_tile, self.block, self.skip_self = self._primary
        self.buf_bytes = primary.buf_bytes
        self.lds_bytes = primary.lds_bytes
        self.tile_bytes = primary.tile_bytes
        self.tile_fp16 = primary.tile_fp16
        self.rank_tile_bytes = primary.rank_tile_bytes
        self.wire_tile_bytes = primary.wire_tile_bytes

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on *this* rank, across every rung.

        ``buf_bytes`` is the primary engine's alone, which understates a
        ladder-driven object by however many rungs it built. The total is what
        actually has to fit, and a sweep holding several tuning variants live
        at once is the realistic way to exhaust a device.
        """
        return sum(eng.buf_bytes for eng in self._by_cfg.values())

    def _kernel_nbytes(self, live_bytes: int) -> int:
        """Bytes the kernel's ``nbytes`` argument covers for a *live_bytes*
        payload: the whole payload here."""
        return int(live_bytes)

    def _ladder_cfg(self, live_bytes: int) -> tuple[int, int, bool]:
        """``(super_tile, block, skip_self)`` the ladder assigns to *live_bytes*.

        Publishes per rank fall as the super-tile grows, and on a cacheable
        inbox each costs a full L2 writeback, so a bigger payload wants a bigger
        ST -- but ST also divides the block count, so it cannot simply be
        maximised. The rungs are the measured trade.
        """
        cfg = self._primary
        for floor, st, _cap, b, ss in self._ladder:
            if live_bytes >= floor:
                cfg = (st, b, ss)
        return cfg

    @staticmethod
    def _num_tiles(live_bytes: int, block: int) -> int:
        """Tiles a *live_bytes* payload spans.

        The same for every schedule: an all-reduce tile is ``ATOMS`` atoms of
        the payload, an all-to-all tile is ``ATOMS / N`` atoms of each of the
        ``N`` chunks.
        """
        tile_bytes = int(block) * ATOMS * 16
        return max(1, (live_bytes + tile_bytes - 1) // tile_bytes)

    def _pick_cfg(self, live_bytes: int) -> tuple[tuple[int, int, bool], int]:
        """Engine key for a *live_bytes* payload, and its tile count.

        The ladder chooses the rung, which fixes the block and so the tile
        count; the tile count then only has to confirm there is a whole
        super-tile to take, falling back to the same block's ST=1 engine when
        there is not.

        Without a release fence a publish is nearly free, so the only reason to
        batch tiles is when there are more of them than blocks -- prefer ST=1
        and the parallelism it buys.

        With one, that trade inverts: every publish costs a full L2 writeback,
        and ST=1 pays one per tile per phase. Take a super-tile as soon as
        there is a whole one to take. Measured on MI350P at 1024x7168, TP4:
        577.71 us at ST=1 against 269.01 at ST=8.
        """
        want, b, ss = self._ladder_cfg(live_bytes)
        num_tiles = self._num_tiles(live_bytes, b)
        if want == 1:
            st = 1
        elif self._batch_publishes:
            st = want if num_tiles >= want else 1
        else:
            st = want if num_tiles > self._by_cfg[(want, b, ss)].grid else 1
        return (st, b, ss), num_tiles

    def _grid_x(self, num_tiles: int, super_tile: int, grid: int | None = None) -> int:
        """Blocks to launch for *num_tiles* tiles under *super_tile*.

        *grid* is the compile-time cap of the engine that will run, which is
        per-super-tile once a ladder is in play -- the wire buffer scales with
        ``ST * grid``, so a high rung pairs a large ST with a small cap.

        Batching publishes only pays if a block actually owns a super-tile's
        worth of work: ST=8 across 448 blocks holding one tile each still
        publishes per tile. Hand each block a full super-tile instead, which
        cuts publishes to ``num_tiles / ST`` per phase.

        Bounded below by ``min_batch_blocks``, because that trade inverts at
        small sizes: 14 tiles over 2 blocks saves a handful of fences and gives
        up the whole machine to do it. Measured on MI350P at 32x7168, TP4,
        61.95 us unbounded against 23.98 with the grid left alone.
        """
        if self._batch_publishes and super_tile != 1:
            batched = max(-(-num_tiles // super_tile), self._min_batch_blocks)
            num_tiles = min(num_tiles, batched)
        return max(1, min(num_tiles, self._grid if grid is None else grid))

    def _launch_args(
        self, eng: _StEngine, inp_ptr, out_ptr, stream, *, live_bytes, num_tiles
    ):
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        return (
            Int32(self.rank),
            Int64(self._kernel_nbytes(live_bytes)),
            Int32(num_tiles),
            Int64(inp_ptr),
            Int64(out_ptr),
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            Int32(self._grid_x(num_tiles, eng.super_tile, eng.grid)),
            stream,
        )

    def _launch_eng(self, eng: _StEngine, inp, out, stream, *, live_bytes: int) -> None:
        kernel_bytes = self._kernel_nbytes(live_bytes)
        num_tiles = max(1, (kernel_bytes + eng.tile_bytes - 1) // eng.tile_bytes)
        args = self._launch_args(
            eng,
            int(inp.data_ptr()),
            int(out.data_ptr()),
            stream,
            live_bytes=live_bytes,
            num_tiles=num_tiles,
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch, *args)

    def cfgs_for(self, lo: int, hi: int) -> list[tuple[int, int, bool]]:
        """``_by_cfg`` keys a payload of ``lo..hi`` bytes (inclusive) can
        select, in build order."""
        floors = [rung[0] for rung in self._ladder]
        picked = {self._pick_cfg(n)[0] for n in payload_probes(floors, lo, hi)}
        return [key for key in self._by_cfg if key in picked]

    def preload(self, *, payload_range=None) -> None:
        """JIT-compile engine binaries without launching any of them.

        ``payload_range=(lo, hi)`` takes only the binaries a payload of
        ``lo..hi`` bytes (inclusive) would run; ``None`` takes every one. The
        ladder builds engines for the whole size range, while a dispatcher
        routes only its own window here, so the rest never run.

        Local to this rank, not a collective. Dispatchers call it once at init,
        via ``preload_fly_engines``, to keep JIT compiles out of CUDA graph
        capture; the benchmark and the op tests call it before timing or
        correctness checks begin. The HIP module load still happens on each
        binary's first launch.
        """
        keys = self._by_cfg if payload_range is None else self.cfgs_for(*payload_range)
        for key in keys:
            eng = self._by_cfg[key]
            args = self._launch_args(eng, 0, 0, None, live_bytes=0, num_tiles=0)
            _preload_compiled(eng.launch, *args)

    def close(self):
        engines = getattr(self, "_by_cfg", None)
        if not engines:
            return
        with torch.cuda.device(self._device_index):
            if getattr(self, "_has_launched", False):
                torch.cuda.synchronize(self._device_index)
                self._has_launched = False
            for eng in engines.values():
                eng.close()
            engines.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            # Destructors must not raise, especially during interpreter shutdown.
            return

    def variant(self, nbytes: int) -> str:
        """Identity of the binary an *nbytes* payload would actually run."""
        cfg, num_tiles = self._pick_cfg(int(nbytes))
        eng = self._by_cfg[cfg]
        grid_x = self._grid_x(num_tiles, eng.super_tile, eng.grid)
        return f"{kernel_symbol(eng.launch)}/grid_x{grid_x}"
