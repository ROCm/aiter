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
class Stage1PrequantContext:
    """Says that ``hidden_states`` is a dispatch wire buffer, not bf16 rows.

    Quantizing before dispatch instead of after it roughly halves what crosses
    the fabric, and costs nothing in accuracy: a token is quantized once either
    way, on the same values. What it does cost is the layout. The MX scales that
    gemm1 wants are interleaved across ``wmma_rep*16`` consecutive destination
    rows, and a sender cannot know a token's destination row -- the receiver
    assigns it only after sorting every peer's routes by expert. So the wire
    carries plain row-major scales appended to each payload row, and the
    receiver's gather applies the preshuffle on the way into the GEMM layout.

    ``stride_bytes`` is the whole wire row; the payload occupies the first
    ``payload_bytes`` and the e8m0 scales the rest.
    """

    stride_bytes: int
    payload_bytes: int

    def __post_init__(self):
        if self.payload_bytes <= 0:
            raise ValueError("payload_bytes must be positive")
        scale_bytes = self.stride_bytes - self.payload_bytes
        if scale_bytes <= 0:
            raise ValueError(
                f"wire row of {self.stride_bytes}B leaves no room for the scales "
                f"of a {self.payload_bytes}B payload"
            )
        if self.payload_bytes % 32:
            raise ValueError("an MX payload row must be a whole number of blocks")
        if scale_bytes < self.payload_bytes // 32:
            raise ValueError(
                f"{scale_bytes}B of scales is short of the "
                f"{self.payload_bytes // 32} e8m0 bytes the payload needs"
            )


@dataclass(frozen=True, slots=True)
class Stage1DispatchContext:
    """Resources for the gfx950-style dispatch fused into gfx1250 GEMM1.

    The sender-side wire remains row-major ``[fp8 payload | e8m0 scales]``.
    Dispatch producers use the destination-owned compact plan to write each
    route directly into the receiver's final contiguous-M payload/scale/rowmap
    slots. GEMM1 therefore consumes ``payload`` directly; it does not gather
    from recv slots.

    Arena offsets name the same regions on every rank. ``workspace`` and all
    output tensors are local views of those regions, passed explicitly so the
    custom-op boundary records the buffers the fused kernel mutates.
    """

    arena_handle: int
    workspace_offset: int
    payload_offset: int
    row_scale_offset: int
    scale_offset: int
    rowmap_offset: int
    m_tile_map_offset: int
    num_valid_offset: int
    rank: int
    world_size: int
    experts_per_rank: int
    max_tokens_per_rank: int
    max_rows: int
    wire_stride_bytes: int
    payload_bytes: int
    wire: torch.Tensor
    workspace: torch.Tensor
    payload: torch.Tensor
    scale: torch.Tensor
    rowmap: torch.Tensor
    num_valid: torch.Tensor
    m_tile_map: torch.Tensor

    def __post_init__(self):
        if self.arena_handle < 0:
            raise ValueError("arena_handle must be non-negative")
        if min(
            self.workspace_offset,
            self.payload_offset,
            self.row_scale_offset,
            self.scale_offset,
            self.rowmap_offset,
            self.m_tile_map_offset,
            self.num_valid_offset,
        ) < 0:
            raise ValueError("stage1 dispatch arena offsets must be non-negative")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("stage1 dispatch rank must be in [0, world_size)")
        if min(
            self.world_size,
            self.experts_per_rank,
            self.max_tokens_per_rank,
            self.max_rows,
            self.wire_stride_bytes,
            self.payload_bytes,
        ) <= 0:
            raise ValueError("stage1 dispatch dimensions must be positive")
        if self.wire_stride_bytes <= self.payload_bytes:
            raise ValueError("stage1 dispatch wire must append row-major scales")
        for name, tensor, dtype in (
            ("wire", self.wire, torch.uint8),
            ("workspace", self.workspace, torch.uint8),
            ("payload", self.payload, torch.uint8),
            ("scale", self.scale, torch.uint8),
            ("rowmap", self.rowmap, torch.int32),
            ("num_valid", self.num_valid, torch.int32),
            ("m_tile_map", self.m_tile_map, torch.int32),
        ):
            if tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous {dtype}")


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
