# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Private host-side types for the gfx1250 MegaMoE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable

import torch

# The MX combine wire, shared by the gemm2 scatter epilogue that writes it
# (mxfp4_preshuffle_gfx1250_tdm.py) and the reduce that reads it (combine.py).
COMBINE_SCALE_BLOCK = 32

_DTYPE_INFO = {
    torch.int8: ("|i1", 1, None),
    torch.int16: ("<i2", 2, None),
    torch.int32: ("<i4", 4, None),
    torch.float32: ("<f4", 4, None),
    torch.bfloat16: ("<u1", 2, torch.bfloat16),
    # The quantizing wires: an fp8 payload views as fp8, an fp4 one as raw bytes
    # (its row width is in BYTES, not features), and both e8m0 scale rows are
    # bytes. Same byte-view-then-reinterpret shape as bf16, one byte per element.
    torch.uint8: ("|u1", 1, None),
    torch.float8_e4m3fn: ("<u1", 1, torch.float8_e4m3fn),
    torch.float8_e4m3fnuz: ("<u1", 1, torch.float8_e4m3fnuz),
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
    compact_layout: bool = False
    compact_masked_m: torch.Tensor | None = None
    compact_psum: torch.Tensor | None = None
    compact_ep_rowmap: torch.Tensor | None = None
    compact_wire_row_stride: int = 0
    # This step's shape, not the arena's. The compact rows a forward actually
    # holds depend on what dispatch delivered, while the arena is sized once for
    # the worst case; a GEMM keyed off the arena runs the largest tuned tile and
    # a grid to match on every decode step. All three are python ints so a
    # captured graph keeps a static grid.
    #   recv_bound: recv-token upper bound, i.e. the CSV token bucket.
    #   align_m:    per-expert row alignment the plan wrote; a GEMM tile_m must
    #               divide it or a tile would straddle two experts.
    #   rows:       row upper bound for this step, bounding the GEMM grid.
    compact_recv_bound: int = 0
    compact_align_m: int = 0
    compact_rows: int = 0
    stage1_mega: Stage1MegaContext | None = None

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
class Stage1MegaContext:
    """Live tensors and protocol state for the private single-kernel stage 1.

    The context is passed through the existing grouped-MoE thread-local rather
    than the public custom-op schema. Tensor fields intentionally retain the
    sender-side quant buffers until the fused kernel has consumed them.
    """

    source_payload: torch.Tensor
    source_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    token_destination_map: torch.Tensor
    rank: int
    arena_handle: int
    output_offset: int
    rowmap_offset: int
    tile_state_offset: int
    wire_row_stride: int
    generation_addr: int
    plan_slot: int
    tile_m: int
    tile_count: int
    compact_cap: int
    tile_state_addr: int
    fallback_dispatch: Callable[[], None]

    def __post_init__(self):
        if self.plan_slot not in (0, 1):
            raise ValueError("stage1 mega plan_slot must be 0 or 1")
        if self.rank < 0:
            raise ValueError("stage1 mega rank must be non-negative")
        if self.tile_m <= 0 or self.tile_count <= 0 or self.compact_cap <= 0:
            raise ValueError("stage1 mega tile geometry must be positive")
        for name in (
            "generation_addr",
            "tile_state_addr",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive device address")
        if self.arena_handle < 0:
            raise ValueError("stage1 mega arena_handle must be non-negative")
        for name in ("output_offset", "rowmap_offset", "tile_state_offset"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.wire_row_stride <= 0:
            raise ValueError("stage1 mega wire_row_stride must be positive")
        if not callable(self.fallback_dispatch):
            raise ValueError("stage1 mega fallback_dispatch must be callable")
