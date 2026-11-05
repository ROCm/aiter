# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""LL128 layout of a bf16 combine slot: every 128 B line carries its writer's
generation, so the reduce can trust a line on its own tag and the combine needs
no cross-device barrier between gemm2 and the reduce.

A slot row is cut into segments of ``SEG_COLS`` columns. A segment's 32
16-byte granules (8 bf16 each) go 7 to a line, the line's last 16 bytes hold
the tag (the generation in all four dwords), so a segment takes ``SEG_LINES``
lines. Granules stay whole, so the gemm2 epilogue's b128 LDS stores and the
reduce's b128 LDS loads keep their width; only their offsets move.

The tag is only as good as the line write is atomic: a reader must never see a
128 B line half old and half new (op_tests/multigpu_tests/bench_ll128_tdm.py
checks it on the target).
"""

SEG_COLS = 256
LINE_B = 128
GRANULE_B = 16
GRANULE_ELEMS = GRANULE_B // 2
LINE_GRANULES = LINE_B // GRANULE_B - 1
TAG_OFF = LINE_GRANULES * GRANULE_B
SEG_GRANULES = SEG_COLS // GRANULE_ELEMS
SEG_LINES = (SEG_GRANULES + LINE_GRANULES - 1) // LINE_GRANULES
SEG_B = SEG_LINES * LINE_B


def row_bytes(cols: int) -> int:
    """Bytes of ``cols`` bf16 columns (whole segments) in the LL128 layout."""
    if cols % SEG_COLS:
        raise ValueError(
            f"LL128 rows hold whole {SEG_COLS}-column segments, got {cols}"
        )
    return cols // SEG_COLS * SEG_B


def fits(hidden_dim: int, slot_stride_bytes: int) -> bool:
    """Whether a hidden_dim row in the LL128 layout fits one combine slot."""
    return hidden_dim % SEG_COLS == 0 and row_bytes(hidden_dim) <= slot_stride_bytes


def granule_offset(g):
    """Byte offset of segment granule ``g`` from its segment's start. ``g`` may
    be a python int or a kernel value."""
    return (g // LINE_GRANULES) * LINE_B + (g % LINE_GRANULES) * GRANULE_B


def col_offset(col):
    """Byte offset of bf16 column ``col`` (a multiple of 8) from the row's start."""
    return (col // SEG_COLS) * SEG_B + granule_offset((col % SEG_COLS) // GRANULE_ELEMS)


def tag_offset(line):
    """Byte offset of line ``line``'s tag from the row's start (segments are
    whole lines, so line ``l`` of a row is simply at ``l * LINE_B``)."""
    return line * LINE_B + TAG_OFF
