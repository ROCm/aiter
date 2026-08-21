# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Private host-side types for the gfx1250 MegaMoE pipeline."""

from dataclasses import dataclass
from math import prod

import torch

_DTYPE_INFO = {
    torch.int8: ("|i1", 1, None),
    torch.uint8: ("|u1", 1, None),
    torch.int16: ("<i2", 2, None),
    torch.int32: ("<i4", 4, None),
    torch.float32: ("<f4", 4, None),
    torch.bfloat16: ("<u1", 2, torch.bfloat16),
}


class GpuPointerView:
    def __init__(self, pointer: int, shape, typestr: str):
        self.__cuda_array_interface__ = {
            "data": (pointer, False),
            "shape": tuple(shape),
            "strides": None,
            "typestr": typestr,
            "version": 3,
        }


def _from_gpu_ptr(pointer: int, shape, dtype: torch.dtype) -> torch.Tensor:
    try:
        typestr, element_size, reinterpret_dtype = _DTYPE_INFO[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported GPU pointer dtype: {dtype}") from error

    device = torch.cuda.current_device()
    if reinterpret_dtype is not None:
        byte_view = GpuPointerView(pointer, (prod(shape) * element_size,), typestr)
        raw = torch.as_tensor(byte_view, device=f"cuda:{device}")
        return raw.view(reinterpret_dtype).reshape(shape)
    view = GpuPointerView(pointer, shape, typestr)
    return torch.as_tensor(view, device=f"cuda:{device}")


@dataclass(frozen=True, slots=True)
class Stage2ScatterContext:
    """Resources used by the GEMM2 P2P scatter epilogue.

    This object stays in Python. ``fused_moe`` unpacks it into schema-supported
    integers and a tensor before crossing the torch custom-op boundary.
    """

    arena_handle: int
    combine_input_offset: int
    slot_stride_bytes: int
    max_tokens_per_rank: int
    world_size: int
    source_token_map: torch.Tensor

    def __post_init__(self):
        if self.arena_handle < 0:
            raise ValueError("arena_handle must be non-negative")
        if self.combine_input_offset < 0:
            raise ValueError("combine_input_offset must be non-negative")
        if self.slot_stride_bytes <= 0 or (
            self.slot_stride_bytes & (self.slot_stride_bytes - 1)
        ):
            raise ValueError("slot_stride_bytes must be a positive power of two")
        if self.max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if (
            self.source_token_map.dtype != torch.int32
            or not self.source_token_map.is_contiguous()
        ):
            raise ValueError("source_token_map must be contiguous int32")


@dataclass(frozen=True, slots=True)
class FusedGatherContext:
    """Resources the persistent GEMM's fused gather prologue reads.

    Carries what the standalone ``moe_wire_gather_preshuffle`` launch would
    have taken, so folding that pass into the GEMM is a matter of handing this
    object over instead of running a kernel.

    The destinations are not listed here: the prologue scatters into the GEMM's
    own ``a`` and ``a_scales``, writing the scales through the very argument the
    mainloop reads them from, because two pointer arguments onto one buffer
    would be noalias to each other.

    ``num_valid_routes`` must be a real (1,) int32 tensor or an empty one --
    the prologue reads a null data pointer as "no dead tail to skip", so a
    stand-in buffer would be mistaken for a survivor count. ``grid_bar`` is the
    zeroed grid-barrier counter, which the launch path resets every time
    because the barrier leaves it at its end value.
    """

    wire: torch.Tensor
    topids_to_rows: torch.Tensor
    num_valid_routes: torch.Tensor
    grid_bar: torch.Tensor
    numel: int
    feat_dim: int
    wmma_rep: int
    source_topk: int = 0
    row_starts: torch.Tensor | None = None
    route_max_m: int = 0

    def __post_init__(self):
        if self.numel <= 0:
            raise ValueError("numel must be positive")
        if self.feat_dim <= 0 or self.feat_dim % 32:
            raise ValueError("feat_dim must be a positive multiple of 32")
        if self.wmma_rep < 1:
            raise ValueError("wmma_rep must be >= 1")
        if self.source_topk < 0:
            raise ValueError("source_topk must be non-negative")
        if self.num_valid_routes.dtype != torch.int32:
            raise ValueError("num_valid_routes must be int32")
        if self.grid_bar.dtype != torch.int32 or self.grid_bar.numel() < 1:
            raise ValueError("grid_bar must be a non-empty int32 counter")
        if self.remap_rows and self.route_max_m <= 0:
            raise ValueError("route_max_m must be positive when remapping rows")

    @property
    def remap_rows(self) -> bool:
        return self.row_starts is not None
