# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Private host-side types for the gfx1250 MegaMoE pipeline."""

from dataclasses import dataclass
from math import prod

import torch

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
