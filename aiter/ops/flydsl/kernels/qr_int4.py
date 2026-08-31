# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4 two-shot all-reduce.

Public type ``QRInt4``. Super-tile ST∈{1,8}; ST=1 when ``num_tiles ≤ grid_cap``.
INT4 nibble + group-16 E4M3. Payload HBM is bf16.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import flydsl.compiler as flyc
import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.dist.parallel_state import in_the_same_node_as
from aiter.jit.utils.chip_info import get_gfx_runtime

from .qr_int4_ipc import UncachedIpcHeap
from .qr_int4_kernel import (
    DEFAULT_GRID_CAP,
    SUPER_TILES,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
    has_release_fence,
    make_qr_int4_kernel,
)

logger = logging.getLogger("aiter")

_SUPPORTED_ARCHS = ("gfx942", "gfx950")

# How the IPC inbox is allocated. The wire protocol is identical in every
# mode; only the memory type changes. See docs/qr_int4_mi350p.md.
INBOX_MEMORY_MODES = ("auto", "uncached", "finegrained")

# Smallest payload sent through this kernel, in bytes.
#
# This is an accuracy policy, not a performance guard. On a fine-grained inbox
# the kernel is faster than ``cross_device_reduce`` at every size measured on
# MI350P at TP4 -- 13.4 us against 22.2 at 14 KiB, 15.0 against 54.3 at 112 KiB
# -- so there is no size at which using it costs throughput. What it always
# costs is ~36 dB of SQNR (~19 dB against the exact kernels' ~55 dB).
#
# The default skips the decode tail, where the absolute saving is a few
# microseconds on a collective that is not the bottleneck and quantizing the
# whole activation is a poor trade. Raise it to be more conservative, or pass
# ``min_bytes=0`` to use the kernel at every size.
#
# The value comes from the post-fix sweep in docs/qr_int4_mi350p.md, not from
# the pre-fix numbers, which had the kernel merely at par at these sizes.
MIN_PAYLOAD_BYTES = 128 << 10

# Floor on the block count when batching publishes into super-tiles; see
# ``QRInt4._grid_x``. Shrinking the grid trades parallelism for fewer release
# fences, which is only a good trade once there are enough fences to matter.
# 32 is a quarter of an MI350P's 128 CUs: low enough that the 14 MiB case still
# reaches its measured optimum of 56 blocks, high enough that the sub-MiB cases
# keep the parallelism they actually need.
_MIN_BATCH_BLOCKS = 32

# KFD io-link type for xGMI, from include/uapi/linux/kfd_sysfs.h. PCIe is 2.
_HSA_IOLINK_TYPE_XGMI = 11
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def has_xgmi_peer_links() -> bool:
    """Whether any GPU-to-GPU link on this host is xGMI rather than PCIe.

    Arch is not enough to make this call: an MI350X (xGMI) and an MI350P
    (PCIe-only) both report ``gfx950``, and the right inbox memory type is
    opposite on the two. KFD exposes the real link type per peer pair, so read
    that instead of guessing from the SKU.

    Returns True when the topology cannot be read, which keeps the historical
    uncached allocation on any host we cannot classify -- the failure mode of
    guessing "PCIe" on an xGMI box is a silent perf regression on hardware
    where the current design is already optimal.
    """
    try:
        for props in _KFD_NODES.glob("*/p2p_links/*/properties"):
            for line in props.read_text().splitlines():
                field, _, value = line.partition(" ")
                if field == "type" and int(value) == _HSA_IOLINK_TYPE_XGMI:
                    return True
        return False
    except (OSError, ValueError):
        logger.debug("QRInt4: cannot read KFD topology; assuming xGMI", exc_info=True)
        return True


def _resolve_inbox_flags(mode: str) -> tuple[int, str]:
    """(hipExtMallocWithFlags mode, resolved name) for an ``inbox_memory``."""
    if mode not in INBOX_MEMORY_MODES:
        raise ValueError(
            f"inbox_memory must be one of {INBOX_MEMORY_MODES}, got {mode!r}"
        )
    if mode == "auto":
        mode = "uncached" if has_xgmi_peer_links() else "finegrained"
    flags = (
        UncachedIpcHeap._HIP_DEVICE_MALLOC_UNCACHED
        if mode == "uncached"
        else UncachedIpcHeap._HIP_DEVICE_MALLOC_FINEGRAINED
    )
    return flags, mode


def _cuda_index(device) -> int:
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"QRInt4 requires a CUDA device, got {device}")
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)
    return int(device)


def _validate_ipc_process_group(group, *, rank: int) -> None:
    """Reject groups that cannot exchange HIP IPC handles or CPU-side metadata."""
    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        raise ValueError(
            f"QRInt4 does not support NCCL process groups (got {backend!r} on "
            f"group rank {rank}): IPC handle exchange requires CPU-side "
            "broadcast_object_list."
        )

    same_node = in_the_same_node_as(group, source_rank=0)
    if not all(same_node):
        off_node = [r for r, ok in enumerate(same_node) if not ok]
        raise RuntimeError(
            "QRInt4 does not support multi-node process groups: HIP IPC "
            f"handles are node-local (ranks not on rank 0's node: {off_node})."
        )


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(self, *, spec, group, rank: int, world_size: int, inbox_flags: int):
        self.spec = spec
        self.launch = spec["launch"]
        self.compiled = None
        self.super_tile = spec["super_tile"]
        self.grid = spec["grid"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self._peer_bases = [None] * world_size
        # The inbox is the only allocation peers write into, so it is the only
        # one whose memory type matters for fabric throughput.
        self._buf_ptr = UncachedIpcHeap.alloc(self.buf_bytes, inbox_flags)
        self._meta_ptr = None
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
        # Peer-pointer table and per-block colours: written by the host once and
        # by this rank's own kernel, never by a peer. Stays uncached in every
        # mode -- no cross-GPU visibility question, and it is a few KiB.
        self._meta_ptr = UncachedIpcHeap.alloc_uncached(peer_bytes + color_bytes)
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
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:
                pass
            self._buf_ptr = None


class QRInt4:
    """IPC inbox + flag buffer and launch wrapper for ``qr_int4``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.

    ``inbox_memory`` selects how the IPC inbox is allocated:

    * ``"auto"`` (default) -- ``uncached`` on hosts with xGMI peer links,
      ``finegrained`` on PCIe-attached hosts. Decided from the KFD topology,
      not from the arch string: MI350X and MI350P both report ``gfx950`` and
      want opposite answers.
    * ``"uncached"`` -- the historical behaviour; correct everywhere, but peer
      writes collapse on PCIe (1.44 GB/s to 3 peers on MI350P, against
      33.5 GB/s fine-grained).
    * ``"finegrained"`` -- device-coherent, full PCIe rate. Cacheable, so the
      peer stores are emitted ``sc0 sc1`` to write through rather than parking
      in the writer's L2; the wire protocol is unchanged.

    ``min_bytes`` is the payload below which ``allreduce`` refuses to run,
    defaulting to ``MIN_PAYLOAD_BYTES``. ``compile`` is deliberately not gated:
    its warmup tensor is allowed to be small.

    See ``docs/qr_int4_mi350p.md``.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int = WORLD,
        super_tile: int = 8,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        min_bytes: int | None = None,
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        if super_tile not in SUPER_TILES:
            raise ValueError(
                f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}"
            )
        group_world = dist.get_world_size(group=group)
        group_rank = dist.get_rank(group=group)
        if group_world != int(world_size):
            raise ValueError(
                f"world_size={world_size} does not match group size {group_world}"
            )
        if group_rank != int(rank):
            raise ValueError(f"rank={rank} does not match group rank {group_rank}")
        _validate_ipc_process_group(group, rank=int(rank))
        arch = get_gfx_runtime()
        if arch not in _SUPPORTED_ARCHS:
            raise RuntimeError(
                f"QRInt4 supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
            )
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")
        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory)
        torch.cuda.set_device(device)
        self.group = group
        self.device = device
        self._device_index = _cuda_index(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.super_tile = int(super_tile)
        self._grid = cap
        self.inbox_memory = resolved_inbox
        self._batch_publishes = has_release_fence(resolved_inbox)
        self.min_bytes = MIN_PAYLOAD_BYTES if min_bytes is None else int(min_bytes)
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be non-negative, got {self.min_bytes}")

        sts = [1]
        if self.super_tile != 1:
            sts.append(self.super_tile)
        self._by_st = {}
        for st in sts:
            spec = make_qr_int4_kernel(
                world_size=self.world_size,
                super_tile=st,
                grid=self._grid,
                inbox_memory=resolved_inbox,
            )
            self._by_st[st] = _StEngine(
                spec=spec,
                group=self.group,
                rank=self.rank,
                world_size=self.world_size,
                inbox_flags=inbox_flags,
            )

        primary = self._by_st[self.super_tile]
        self.buf_bytes = primary.buf_bytes
        self.lds_bytes = primary.lds_bytes
        self.tile_bytes = primary.tile_bytes
        self.tile_fp16 = primary.tile_fp16
        self.rank_tile_bytes = primary.rank_tile_bytes
        self.wire_tile_bytes = primary.wire_tile_bytes

    def _pick_st(self, num_tiles: int) -> int:
        """Super-tile for a payload of *num_tiles* tiles.

        Without a release fence a publish is nearly free, so the only reason to
        batch tiles is when there are more of them than blocks -- prefer ST=1
        and the parallelism it buys.

        With one, that trade inverts: every publish costs a full L2 writeback,
        and ST=1 pays one per tile per phase. Take a super-tile as soon as
        there is a whole one to take. Measured on MI350P at 1024x7168, TP4:
        577.71 us at ST=1 against 269.01 at ST=8.
        """
        if self.super_tile == 1:
            return 1
        if self._batch_publishes:
            return self.super_tile if num_tiles >= self.super_tile else 1
        return self.super_tile if num_tiles > self._grid else 1

    def _grid_x(self, num_tiles: int, super_tile: int) -> int:
        """Blocks to launch for *num_tiles* tiles under *super_tile*.

        Batching publishes only pays if a block actually owns a super-tile's
        worth of work: ST=8 across 448 blocks holding one tile each still
        publishes per tile. Hand each block a full super-tile instead, which
        cuts publishes to ``num_tiles / ST`` per phase.

        Bounded below by ``_MIN_BATCH_BLOCKS``, because that trade inverts at
        small sizes: 14 tiles over 2 blocks saves a handful of fences and gives
        up the whole machine to do it. Measured on MI350P at 32x7168, TP4,
        61.95 us unbounded against 23.98 with the grid left alone.
        """
        if self._batch_publishes and super_tile != 1:
            batched = max(-(-num_tiles // super_tile), _MIN_BATCH_BLOCKS)
            num_tiles = min(num_tiles, batched)
        return max(1, min(num_tiles, self._grid))

    def _check_payload(self, inp, out) -> int:
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            raise TypeError("QRInt4 requires torch.Tensor input/output")
        if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("QRInt4 supports bf16 input/output")
        if not inp.is_cuda or not out.is_cuda:
            raise ValueError("QRInt4 requires CUDA tensors")
        if (
            inp.device.index != self._device_index
            or out.device.index != self._device_index
        ):
            raise ValueError(
                f"inp/out must be on cuda:{self._device_index}, "
                f"got {inp.device} / {out.device}"
            )
        if not inp.is_contiguous() or not out.is_contiguous():
            raise ValueError("QRInt4 requires contiguous input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        return live_bytes

    def _launch_eng(self, eng: _StEngine, inp, out, stream) -> None:
        live_bytes = int(inp.numel()) * int(inp.element_size())
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        grid_x = self._grid_x(num_tiles, eng.super_tile)
        if stream is None:
            stream = Stream(None)
        args = (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            Int32(grid_x),
            stream,
        )
        if eng.compiled is None:
            eng.compiled = flyc.compile(eng.launch, *args)
        else:
            eng.compiled(*args)

    def compile(self, inp, out, stream=None) -> None:
        """Eager-JIT every ST binary. Optional: first ``allreduce`` JIT-compiles the picked ST.

        Default ST=8 also builds an ST=1 engine for ``num_tiles ≤ grid_cap``.
        Skipping this method is correct for a single size class: that
        ``allreduce`` calls ``flyc.compile`` for the chosen ST only, and a
        later size that picks the other ST JIT-compiles then.

        ``flyc.compile`` also launches, so this is a real collective: every
        rank must call it with the same ``inp``/``out`` shape. The warmup
        tensor may be small; we still launch every engine so a later
        prefill-sized ``allreduce`` does not JIT mid-collective. ``out`` is
        overwritten.
        """
        self._check_payload(inp, out)
        for eng in self._by_st.values():
            self._launch_eng(eng, inp, out, stream)

    def close(self):
        for eng in self._by_st.values():
            eng.close()
        self._by_st = {}

    def is_beneficial(self, nbytes: int) -> bool:
        """Whether *nbytes* is large enough for this kernel to be worth using.

        Callers with a fallback should route anything smaller to it; see
        ``MIN_PAYLOAD_BYTES``. ``allreduce`` refuses payloads below the
        threshold rather than silently running them slowly.
        """
        return int(nbytes) >= self.min_bytes

    def allreduce(self, inp, out, stream=None):
        live_bytes = self._check_payload(inp, out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"QRInt4.allreduce got a {live_bytes} B payload, below the "
                f"{self.min_bytes} B floor: at decode sizes this kernel saves a "
                "few microseconds on a collective that is not the bottleneck, "
                "and charges ~36 dB of SQNR for them. Route small messages to "
                "an exact all-reduce, or pass min_bytes=0 to override."
            )
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        st = self._pick_st(num_tiles)
        self._launch_eng(self._by_st[st], inp, out, stream)
