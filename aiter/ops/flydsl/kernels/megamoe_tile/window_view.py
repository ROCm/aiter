# SPDX-License-Identifier: MIT
"""Host-side reads and writes of a CCO registered window.

A CCO window is ordinary device memory, so inspecting it needs neither a
hipMemcpy wrapper nor a clear kernel -- wrapping the local VA in a torch tensor
through ``__cuda_array_interface__`` is enough. This is the same approach
``mega_moe_gfx1250``'s ``SymmetricArena`` uses for its arena, and it replaces
the hand-rolled ``cco/host.py`` helpers this operator used to carry.
"""

from __future__ import annotations

import torch

__all__ = [
    "read_window_bytes",
    "read_window_u32",
    "read_window_u64",
    "window_tensor",
    "write_window_u64",
    "zero_window",
]

_TYPESTR = {
    torch.int8: "|i1",
    torch.uint8: "|u1",
    torch.int32: "<i4",
    torch.int64: "<i8",
}


class _GpuPointerView:
    """Minimal ``__cuda_array_interface__`` so torch can wrap a raw device VA."""

    def __init__(self, pointer: int, shape, typestr: str):
        self.__cuda_array_interface__ = {
            "data": (int(pointer), False),
            "shape": tuple(shape),
            "strides": None,
            "typestr": typestr,
            "version": 3,
        }


def window_tensor(pointer: int, count: int, dtype: torch.dtype) -> torch.Tensor:
    """Zero-copy torch view of ``count`` ``dtype`` elements at a window VA."""
    view = _GpuPointerView(pointer, (int(count),), _TYPESTR[dtype])
    return torch.as_tensor(view, device=f"cuda:{torch.cuda.current_device()}")


def zero_window(pointer: int, nbytes: int) -> None:
    window_tensor(pointer, nbytes, torch.int8).zero_()


def read_window_u32(pointer: int, count: int) -> tuple[int, ...]:
    # torch has no uint32; mask back to the unsigned value callers expect.
    return tuple(
        int(v) & 0xFFFFFFFF
        for v in window_tensor(pointer, count, torch.int32).tolist()
    )


def read_window_u64(pointer: int, count: int) -> tuple[int, ...]:
    return tuple(
        int(v) & 0xFFFFFFFFFFFFFFFF
        for v in window_tensor(pointer, count, torch.int64).tolist()
    )


def read_window_bytes(pointer: int, nbytes: int) -> bytes:
    return bytes(window_tensor(pointer, nbytes, torch.uint8).cpu().tolist())


def write_window_u64(pointer: int, values) -> None:
    values = [int(v) for v in values]
    window_tensor(pointer, len(values), torch.int64).copy_(
        torch.tensor(
            [v - (1 << 64) if v >= (1 << 63) else v for v in values],
            dtype=torch.int64,
        )
    )
