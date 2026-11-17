# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Receive and staging layouts of the pre-attention monokernel per Attention A2A profile.

For one rank of a Ulysses group, each role's receive buffers are the A2A op's packed
buffers for the profile, byte for byte and with its sizes (``packed_buffer_sizes``), so the
MHA-v4 launcher reads them unchanged. A role's layout also lists the byte ranges this rank
fills in every destination's buffers. The rank quantizes into a staging image holding just
those ranges (destination-major) and copies each range with one SDMA put.
"""

import math
from dataclasses import dataclass, field

from aiter.ops.mha_v4 import AttentionPack
from aiter.ops.mha_v4_quant import (
    MHA_V4_MXFP6_V_PACKED_ROW_BYTES,
    MHA_V4_MXFP6_V_SCALE_TILE_BYTES,
)

from .attention_a2a_intranode import packed_buffer_sizes
from .attention_a2a_profiles import attention_a2a_profile

HEAD_DIM = 128
TILE_TOKENS = 128  # K/V tile of the MHA's packed K and V layouts

# Payload bits per value.
_BITS = {
    "e4m3": 8,
    "int8": 8,
    "mxfp8": 8,
    "e4m3_pc": 8,
    "mxfp6": 6,
    "mxfp6_p": 6,
    "mxfp4": 4,
}


@dataclass
class Range:
    region: str  # "payload" or "scale"
    recv: int  # byte offset in the destination's region
    stage: int  # byte offset in this rank's staging for that destination
    length: int


@dataclass
class RoleLayout:
    role: int
    codec: str
    scale: str  # "tensor", "channel" or "block"
    sequence: (
        int  # tokens of the role in the receive buffers (valid_kv_len for trimmed K/V)
    )
    payload_bytes: int
    scale_bytes: int  # scale storage, with the consumer's padding
    scale_view_bytes: int  # the part the packed result exposes as the role's scale
    multiplier: float  # folded into Q before MX quantization
    ranges: list = field(default_factory=list)
    stage_bytes: int = 0
    trimmed: bool = (
        False  # tokens at or past `sequence` are masked, as the op's direct K/V
    )
    # Tiled K/V: tiles per head in the receive buffers, this rank's first tile, and the
    # staging bytes per head; FP6 K stages one record before its first tile.
    tiles: int = 0
    first_tile: int = 0
    head_stage: int = 0
    # The profile's V pack; V_FOR_FP6_P permutes V tokens within 64-token frames.
    v_pack: AttentionPack = AttentionPack.DEFAULT


def role_layouts(
    profile, rank, world, seq_local, heads, valid_kv_len=None, softmax_scale=None
):
    """The three RoleLayouts of ``profile`` (a name or an ``AttentionA2AProfile``) on
    ``rank``. ``valid_kv_len`` (global) trims K/V as the op's direct K/V output does; it
    applies only to profiles with ``direct_kv_output``."""
    if isinstance(profile, str):
        profile = attention_a2a_profile(profile)
    codecs = profile.codecs
    seq_full = seq_local * world
    trimmed = (
        profile.direct_kv_output
        and valid_kv_len is not None
        and valid_kv_len < seq_full
    )
    kv = valid_kv_len if trimmed else seq_full
    sizes = packed_buffer_sizes(codecs, world, seq_local, heads, kv, profile.block_num)
    heads_local = heads // world
    softmax_scale = HEAD_DIM**-0.5 if softmax_scale is None else softmax_scale
    q_multiplier = (
        softmax_scale * math.log2(math.e) if codecs[0] not in ("int8", "e4m3") else 1.0
    )
    first = rank * seq_local
    layouts = []
    for role, codec in enumerate(codecs):
        layout = RoleLayout(
            role=role,
            codec=codec,
            scale=profile.scale_kinds[role],
            sequence=sizes.sequence[role],
            payload_bytes=sizes.payload[role],
            scale_bytes=sizes.scale_storage[role],
            scale_view_bytes=sizes.scale[role],
            multiplier=q_multiplier if role == 0 else 1.0,
            trimmed=trimmed and role > 0,
            v_pack=profile.v_pack if role == 2 else AttentionPack.DEFAULT,
        )
        last = min(first + seq_local, layout.sequence)
        tiled = (role == 1 and codec in ("mxfp4", "mxfp6")) or (
            role == 2 and codec in ("mxfp4", "mxfp6_p")
        )
        stage = 0
        if not tiled:
            # Token-major payload: this rank's tokens are one contiguous range.
            row = heads_local * HEAD_DIM * _BITS[codec] // 8
            layout.ranges.append(
                Range("payload", first * row, stage, (last - first) * row)
            )
            stage += (last - first) * row
            if layout.scale == "block":
                # One E8M0 per 32 channels, token-major.
                row = heads_local * HEAD_DIM // 32
                layout.ranges.append(
                    Range("scale", first * row, stage, (last - first) * row)
                )
                stage += (last - first) * row
        else:
            # Per-head records of 128-token tiles; ranks own whole tiles.
            if seq_local % TILE_TOKENS:
                raise ValueError(
                    f"tiled {codec} {'QKV'[role]} needs rows per rank divisible by "
                    f"{TILE_TOKENS}"
                )
            tiles = (layout.sequence + TILE_TOKENS - 1) // TILE_TOKENS
            t0, t1 = first // TILE_TOKENS, (last + TILE_TOKENS - 1) // TILE_TOKENS
            record, scale_record = _tile_records(role, codec)
            layout.tiles, layout.first_tile = tiles, t0
            if role == 1 and codec == "mxfp6":
                # A record's last byte (the last field of its scale image B) is the next
                # tile's first scale. So a rank fills the byte before its first tile,
                # staged as the last byte of an extra record ahead of its own, and leaves
                # its last byte to the next rank.
                lead, tail = int(t0 > 0), int(t1 < tiles)
                layout.head_stage = (t1 - t0 + 1) * record
                for head in range(heads_local):
                    layout.ranges.append(
                        Range(
                            "payload",
                            (head * tiles + t0) * record - lead,
                            stage + record - lead,
                            (t1 - t0) * record + lead - tail,
                        )
                    )
                    stage += layout.head_stage
            else:
                layout.head_stage = (t1 - t0) * record
                for head in range(heads_local):
                    layout.ranges.append(
                        Range(
                            "payload",
                            (head * tiles + t0) * record,
                            stage,
                            (t1 - t0) * record,
                        )
                    )
                    stage += (t1 - t0) * record
            if role == 1:
                # MXFP4 and MXFP6 K keep token-major scales beside the tiled payload.
                row = heads_local * HEAD_DIM // 32
                layout.ranges.append(
                    Range("scale", first * row, stage, (last - first) * row)
                )
                stage += (last - first) * row
            elif scale_record:
                for head in range(heads_local):
                    layout.ranges.append(
                        Range(
                            "scale",
                            (head * tiles + t0) * scale_record,
                            stage,
                            (t1 - t0) * scale_record,
                        )
                    )
                    stage += (t1 - t0) * scale_record
        layout.stage_bytes = (stage + 255) // 256 * 256
        layouts.append(layout)
    return layouts


def _tile_records(role, codec):
    """Payload and scale bytes per (head, 128-token tile) of a tiled K/V layout."""
    if role == 1 and codec == "mxfp6":
        return 17408, 0  # scale images inside the record, token-major scales beside it
    if role == 1 and codec == "mxfp4":
        return 8192, 0
    if role == 2 and codec == "mxfp4":
        return 8192, 512
    if role == 2 and codec == "mxfp6_p":
        return (
            TILE_TOKENS * MHA_V4_MXFP6_V_PACKED_ROW_BYTES,
            MHA_V4_MXFP6_V_SCALE_TILE_BYTES,
        )
    raise ValueError(f"no tiled layout for {codec} {'QKV'[role]}")
