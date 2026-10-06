# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for the gfx942/gfx950 TP∈{2,4,8} FlyDSL all-to-all.

Public type ``FlyQuickAllToAll``: an equal-split all-to-all, the semantics of
``torch.distributed.all_to_all_single`` without split sizes. The input is cut
into ``N`` equal chunks; chunk ``j`` goes to rank ``j`` and output chunk ``i``
comes from rank ``i``. Two schedules:

* ``"mesh"`` (default) -- every tile goes to all ``N-1`` peers at once.
* ``"ring"`` -- shifted pairwise: ``N-1`` steps, each to a single peer, so
  every store has one destination.

Which one is faster is a function of the fabric, the world size and the
payload; ``alltoall_policy`` owns that choice for the dispatcher.

Wire formats: ``None`` (register-direct, byte-exact, any dtype), ``"int4"``
and ``"int6"`` (group-16 E4M3-scaled, bf16 only). ``"fp16"`` is the lossless
LDS-staged path, for testing the transport.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter.jit.utils.chip_info import get_gfx_runtime

from .allreduce_shared import (
    _SUPPORTED_ARCHS,
    _cuda_index,
    _LadderedIpcOp,
    _resolve_inbox_flags,
    _validate_ipc_process_group,
    has_xgmi_peer_links,
)
from .kernels.collectives_shared import (
    BLOCK,
    DEFAULT_GRID_CAP,
    SUPPORTED_WORLDS,
    has_release_fence,
)
from .kernels.quick_allreduce_codec import SUPPORTED_BLOCKS
from .kernels.quick_alltoall_mesh import (
    A2A_CODECS,
    A2A_DIRECT_ORDERS,
    A2A_MESH_SUPER_TILES,
    A2A_QUANT_CODECS,
    a2a_mesh_st_ladder,
    make_quick_alltoall_mesh_kernel,
)
from .kernels.quick_alltoall_ring import (
    A2A_RING_SUPER_TILES,
    a2a_ring_st_ladder,
    make_quick_alltoall_ring_kernel,
)

logger = logging.getLogger("aiter")

# Floor on the block count when batching publishes into super-tiles; see
# ``_LadderedIpcOp._grid_x``.
_MIN_BATCH_BLOCKS = 32

# Dtypes the quantizing codecs read and write; the codec math is packed fp16.
_QUANT_DTYPES = (torch.bfloat16,)


@dataclass(frozen=True)
class _A2AAlgorithm:
    name: str
    build: Callable[..., dict]
    super_tiles: tuple[int, ...]
    default_super_tile: int
    st_ladder: Callable[[int, str], tuple]


ALGORITHMS = {
    "mesh": _A2AAlgorithm(
        name="mesh",
        build=make_quick_alltoall_mesh_kernel,
        super_tiles=A2A_MESH_SUPER_TILES,
        default_super_tile=1,
        st_ladder=a2a_mesh_st_ladder,
    ),
    "ring": _A2AAlgorithm(
        name="ring",
        build=make_quick_alltoall_ring_kernel,
        super_tiles=A2A_RING_SUPER_TILES,
        default_super_tile=8,
        st_ladder=a2a_ring_st_ladder,
    ),
}
CODEC_CHOICES = (None, *A2A_CODECS)


def resolve_codec(codec: str | None) -> str:
    """``None`` and ``"none"`` both name the register-direct, unquantized wire."""
    codec = "none" if codec is None else str(codec).lower()
    if codec not in A2A_CODECS:
        raise ValueError(f"codec must be one of {CODEC_CHOICES}, got {codec!r}")
    return codec


class FlyQuickAllToAll(_LadderedIpcOp):
    """IPC inbox and launch wrapper for the FlyDSL equal-split all-to-all.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.

    ``algorithm`` -- ``"mesh"`` (default) or ``"ring"``. ``codec`` -- ``None`` (lossless, register-direct),
    ``"int4"`` or ``"int6"``; ``"fp16"`` is the lossless LDS-staged transport.

    ``inbox_memory``, ``link``, ``grid_cap``, ``block`` and ``super_tile`` mean
    what they mean for ``FlyQuickAllReduce``: pinning ``super_tile`` collapses
    the size ladder to one rung, ``grid_cap`` bounds every rung. ``order``
    picks the register-direct store order of the mesh (``"peer"``: one
    destination's atoms back to back; ``"atom"``: interleaved).

    ``min_bytes`` is the payload below which :meth:`all_to_all` refuses to run;
    0 by default -- unlike the all-reduce, a lossless all-to-all has no
    accuracy reason to refuse small payloads, only a speed one, which is the
    dispatcher's call.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int,
        algorithm: str = "mesh",
        codec: str | None = None,
        super_tile: int | None = None,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        batch_publishes: bool | None = None,
        link: str | None = None,
        block: int | None = None,
        order: str = "peer",
        min_bytes: int = 0,
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        if link is None:
            link = "xgmi" if has_xgmi_peer_links() else "pcie"
        if link not in ("pcie", "xgmi"):
            raise ValueError(f"link must be 'pcie' or 'xgmi', got {link!r}")
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {tuple(ALGORITHMS)}, got {algorithm!r}"
            )
        algo = ALGORITHMS[algorithm]
        codec = resolve_codec(codec)
        if block is not None and block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
        if order not in A2A_DIRECT_ORDERS:
            raise ValueError(f"order must be one of {A2A_DIRECT_ORDERS}, got {order!r}")
        pinned_st = super_tile is not None
        if super_tile is None:
            super_tile = algo.default_super_tile
        if super_tile not in algo.super_tiles:
            raise ValueError(
                f"super_tile must be one of {algo.super_tiles} for "
                f"algorithm={algorithm!r}, got {super_tile!r}"
            )
        if int(min_bytes) < 0:
            raise ValueError(f"min_bytes must be non-negative, got {min_bytes}")
        if dist.get_world_size(group=group) != int(world_size):
            raise ValueError(
                f"world_size={world_size} does not match group size "
                f"{dist.get_world_size(group=group)}"
            )
        if dist.get_rank(group=group) != int(rank):
            raise ValueError(
                f"rank={rank} does not match group rank {dist.get_rank(group=group)}"
            )
        _validate_ipc_process_group(group, rank=int(rank))
        arch = get_gfx_runtime()
        if arch not in _SUPPORTED_ARCHS:
            raise RuntimeError(
                f"FlyQuickAllToAll supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
            )
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")

        world_ladder = algo.st_ladder(int(world_size), link)
        laddered = bool(world_ladder) and not pinned_st
        if laddered:
            ladder = tuple(
                (floor, st, min(rung_cap, cap), int(b if block is None else block))
                for floor, st, rung_cap, b in world_ladder
            )
        else:
            ladder = (
                (0, int(super_tile), cap, int(BLOCK if block is None else block)),
            )
        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory, world_size)

        self.device = torch.device("cuda", _cuda_index(device))
        self.link = link
        self.algorithm = algorithm
        self.codec = codec
        self.order = order
        self.inbox_memory = resolved_inbox
        self.min_bytes = int(min_bytes)

        def _build(*, super_tile, grid, block):
            return algo.build(
                world_size=int(world_size),
                rank=int(rank),
                super_tile=super_tile,
                grid=grid,
                inbox_memory=resolved_inbox,
                codec=codec,
                block=block,
                order=order,
            )

        self._init_engines(
            group=group,
            device_index=self.device.index,
            rank=rank,
            world_size=world_size,
            ladder=ladder,
            laddered=laddered,
            build=_build,
            inbox_flags=inbox_flags,
            arch=arch,
            batch_publishes=(
                has_release_fence(resolved_inbox) or algorithm == "ring"
                if batch_publishes is None
                else batch_publishes
            ),
            min_batch_blocks=_MIN_BATCH_BLOCKS,
            grid_cap=cap,
            label=f"all-to-all {algorithm} {codec}",
        )

    def _kernel_nbytes(self, live_bytes: int) -> int:
        """The kernel sees one chunk."""
        return int(live_bytes) // self.world_size

    def check_payload(self, inp: torch.Tensor, out: torch.Tensor | None = None) -> int:
        """Validate *inp* (and *out*, when given); return the payload's bytes.

        Raises on anything the kernel cannot serve. :meth:`supports` is the
        non-raising form for dispatchers, which ask before allocating *out*.
        """
        tensors = (inp,) if out is None else (inp, out)
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("FlyQuickAllToAll requires torch.Tensor input/output")
        if self.codec in A2A_QUANT_CODECS and any(
            t.dtype not in _QUANT_DTYPES for t in tensors
        ):
            raise ValueError(
                f"codec={self.codec!r} needs {_QUANT_DTYPES} input/output, got "
                f"{[t.dtype for t in tensors]}"
            )
        for t in tensors:
            if not t.is_cuda or t.device.index != self._device_index:
                raise ValueError(
                    f"FlyQuickAllToAll requires tensors on cuda:{self._device_index}, "
                    f"got {t.device}"
                )
            if not t.is_contiguous():
                raise ValueError("FlyQuickAllToAll requires contiguous input/output")
            if int(t.data_ptr()) % 16:
                raise ValueError(
                    "FlyQuickAllToAll requires 16-byte-aligned input/output"
                )
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes % (16 * self.world_size):
            raise ValueError(
                f"byte size {live_bytes} must split into {self.world_size} chunks "
                "of a whole number of 16 B"
            )
        if live_bytes // self.world_size > 0xFFFFFFFF:
            raise ValueError("FlyQuickAllToAll chunks must not exceed 4 GiB")
        if out is not None:
            if int(out.numel()) * int(out.element_size()) != live_bytes:
                raise ValueError("inp/out byte size mismatch")
            inp_ptr, out_ptr = int(inp.data_ptr()), int(out.data_ptr())
            if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
                raise ValueError(
                    "FlyQuickAllToAll requires non-overlapping input/output"
                )
        return live_bytes

    def supports(self, inp: torch.Tensor, out: torch.Tensor | None = None) -> bool:
        """Whether :meth:`all_to_all` would accept *inp* (and *out*). Never raises."""
        try:
            nbytes = self.check_payload(inp, out)
        except (TypeError, ValueError):
            return False
        return self.is_beneficial(nbytes)

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) >= self.min_bytes

    def all_to_all(self, inp: torch.Tensor, out: torch.Tensor, stream=None) -> None:
        """Equal-split all-to-all from *inp* into *out*.

        ``stream=None`` uses the current PyTorch stream on this device.
        """
        live_bytes = self.check_payload(inp, out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"FlyQuickAllToAll.all_to_all got a {live_bytes} B payload, below "
                f"the {self.min_bytes} B floor"
            )
        cfg, _num_tiles = self._pick_cfg(live_bytes)
        self._launch_eng(self._by_cfg[cfg], inp, out, stream, live_bytes=live_bytes)
