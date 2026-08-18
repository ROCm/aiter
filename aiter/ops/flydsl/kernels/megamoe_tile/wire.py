# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class K3DispatchWireLayout:
    hidden_bytes: int = 3584 // 2
    scale_bytes: int = 3584 // 32
    topk: int = 16

    @property
    def ids_bytes(self) -> int:
        return self.topk * 4

    @property
    def weights_bytes(self) -> int:
        return self.topk * 4

    @property
    def source_offset(self) -> int:
        return self.hidden_bytes + self.scale_bytes + self.ids_bytes + self.weights_bytes

    @property
    def route_mask_offset(self) -> int:
        return self.source_offset + 8  # source u32 + four bytes reserved

    @property
    def record_bytes(self) -> int:
        # Preserve the K3/small-shape 2-KiB ABI, then grow at the transport's
        # 256-byte alignment. H=7168/topk16 has a 3952-byte raw record and
        # therefore rounds to 4096 bytes.
        raw = self.route_mask_offset + 8
        return max(2048, (raw + 255) // 256 * 256)

    @property
    def records_per_64k(self) -> int:
        return (64 * 1024) // self.record_bytes


@dataclass(frozen=True)
class K3PartialWireLayout:
    hidden: int = 3584

    @property
    def payload_bytes(self) -> int:
        return self.hidden * 2

    @property
    def record_bytes(self) -> int:
        raw = self.payload_bytes + 4
        return (raw + 255) // 256 * 256

    @property
    def records_per_64k(self) -> int:
        return (64 * 1024) // self.record_bytes


def _row_bytes(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8).view(rows, -1)


def pack_dispatch_records(
    hidden_fp4: torch.Tensor,
    scales_e8m0: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    source_flat_token: torch.Tensor,
    destination_route_mask: torch.Tensor,
    *,
    layout: K3DispatchWireLayout = K3DispatchWireLayout(),
) -> torch.Tensor:
    """Byte-exact host/reference packer for the future FlyDSL pack role."""

    rows = int(hidden_fp4.shape[0])
    if topk_ids.shape != (rows, layout.topk) or topk_weights.shape != (
        rows,
        layout.topk,
    ):
        raise ValueError("topk tensors do not match K3 wire layout")
    out = torch.zeros(
        (rows, layout.record_bytes), dtype=torch.uint8, device=hidden_fp4.device
    )
    off = 0
    for value, width in (
        (hidden_fp4, layout.hidden_bytes),
        (scales_e8m0, layout.scale_bytes),
        (topk_ids.to(torch.int32), layout.ids_bytes),
        (topk_weights.to(torch.float32), layout.weights_bytes),
    ):
        raw = _row_bytes(value, rows)
        if raw.shape[1] != width:
            raise ValueError(f"wire field expected {width} bytes, got {raw.shape[1]}")
        out[:, off : off + width].copy_(raw)
        off += width
    out[:, layout.source_offset : layout.source_offset + 4].copy_(
        _row_bytes(source_flat_token.to(torch.int32), rows)
    )
    out[:, layout.route_mask_offset : layout.route_mask_offset + 8].copy_(
        _row_bytes(destination_route_mask.to(torch.int64), rows)
    )
    return out


def unpack_dispatch_records(
    records: torch.Tensor,
    *,
    layout: K3DispatchWireLayout = K3DispatchWireLayout(),
) -> dict[str, torch.Tensor]:
    if records.ndim != 2 or records.shape[1] != layout.record_bytes:
        raise ValueError("records has the wrong stride")
    rows = records.shape[0]
    off = 0
    hidden = records[:, off : off + layout.hidden_bytes]
    off += layout.hidden_bytes
    scale = records[:, off : off + layout.scale_bytes]
    off += layout.scale_bytes
    ids = (
        records[:, off : off + layout.ids_bytes]
        .contiguous()
        .view(torch.int32)
        .view(rows, layout.topk)
    )
    off += layout.ids_bytes
    weights = (
        records[:, off : off + layout.weights_bytes]
        .contiguous()
        .view(torch.float32)
        .view(rows, layout.topk)
    )
    source = (
        records[:, layout.source_offset : layout.source_offset + 4]
        .contiguous()
        .view(torch.int32)
        .view(rows)
    )
    route_mask = (
        records[:, layout.route_mask_offset : layout.route_mask_offset + 8]
        .contiguous()
        .view(torch.int64)
        .view(rows)
    )
    return {
        "hidden_fp4_u8": hidden,
        "scales_e8m0_u8": scale,
        "topk_ids": ids,
        "topk_weights": weights,
        "source_flat_token": source,
        "destination_route_mask": route_mask,
    }
