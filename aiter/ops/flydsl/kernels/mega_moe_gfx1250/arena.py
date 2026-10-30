# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The one cco symmetric allocation a MegaMoE makes: its regions, their
layout, and what cco rounds it up to."""

from dataclasses import dataclass

import torch

from .types import _from_gpu_ptr


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class SymmetricArena:
    """One cco symmetric allocation, registered as a window and cut into the
    named regions of an ``_ArenaLayout`` (or a [(name, nbytes)] list, laid down
    in its own order)."""

    _ALIGNMENT = 256

    def __init__(self, communicator, layout):
        if not isinstance(layout, _ArenaLayout):
            layout = _ArenaLayout.of(layout)
        self._communicator = communicator
        self.layout = layout
        self._offsets = dict(layout.offsets)
        self._sizes = dict(layout.regions)
        self._total_bytes = layout.total_bytes
        self._memory = communicator.alloc_mem(self._total_bytes)
        try:
            self._window = communicator.register_window(
                self._memory.ptr, self._total_bytes
            )
        except BaseException:
            self._memory.close()
            raise

    @property
    def handle(self) -> int:
        return self._window.handle

    def offset(self, name: str) -> int:
        return self._offsets[name]

    def local_ptr(self, name: str) -> int:
        return self._window.local_ptr + self._offsets[name]

    def size(self, name: str) -> int:
        """A region's bytes; mori's plans check theirs against it at bind."""
        return self._sizes[name]

    def zero(self, name: str | None = None):
        if name is None:
            pointer, size = self._window.local_ptr, self._total_bytes
        else:
            pointer = self.local_ptr(name)
            size = self._sizes[name]
        _from_gpu_ptr(pointer, (size,), torch.int8).zero_()

    def close(self):
        try:
            self._window.close()
        finally:
            self._memory.close()


# The order the regions are laid down in: fixed, so a region's offset does not
# depend on which stage declares it.
_ARENA_REGION_ORDER = (
    "tok_off",
    "recv_num",
    "recv_to_src_token",
    "out_idx",
    "out_wts",
    "disp_out",
    "cross_device_barrier",
    "disp_out_scales",
    "ep_rowmap",
    "compact_hist",
    "compact_done",
    "comb_inp",
)
# cco rounds each allocation up to its VMM granularity -- 2 MiB on the parts
# MegaMoE runs on -- and, past 2 GiB, to 2 MiB again for the MR registration.
_CCO_VMM_GRANULARITY = 2 << 20
_CCO_MR_PAD_THRESHOLD = 2 << 30
_CCO_MR_PAD_ALIGN = 2 << 20


@dataclass(frozen=True)
class _ArenaLayout:
    """The arena's named regions, their offsets, and the bytes the whole takes."""

    regions: tuple  # ((name, nbytes), ...), in layout order
    offsets: dict
    total_bytes: int

    @classmethod
    def of(cls, regions) -> "_ArenaLayout":
        """Lay ``regions`` down in the order given, each 256 B aligned."""
        offsets, offset = {}, 0
        for name, size in regions:
            offset = _align_up(offset, SymmetricArena._ALIGNMENT)
            offsets[name] = offset
            offset += size
        total = max(
            _align_up(offset, SymmetricArena._ALIGNMENT), SymmetricArena._ALIGNMENT
        )
        return cls(tuple((name, int(size)) for name, size in regions), offsets, total)

    @classmethod
    def merge(cls, *requests) -> "_ArenaLayout":
        """The regions every request asks for, each at the largest size asked,
        in _ARENA_REGION_ORDER."""
        need = {}
        for request in requests:
            for name, size in request.items():
                need[name] = max(need.get(name, 0), int(size))
        unknown = sorted(set(need) - set(_ARENA_REGION_ORDER))
        if unknown:
            raise ValueError(f"unknown arena region(s) {unknown}")
        return cls.of([(n, need[n]) for n in _ARENA_REGION_ORDER if need.get(n)])


def _cco_alloc_bytes(nbytes: int, granularity: int) -> int:
    """What ccoMemAlloc takes out of per_rank_vmm for an ``nbytes`` allocation."""
    aligned = _align_up(nbytes, granularity)
    if nbytes > _CCO_MR_PAD_THRESHOLD:
        aligned = _align_up(aligned, _CCO_MR_PAD_ALIGN)
    return aligned
