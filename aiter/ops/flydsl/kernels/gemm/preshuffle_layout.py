# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Byte layouts for the cooperative A DMA used by preshuffle GEMMs."""

import flydsl.expr as fx


def make_preshuffle_dma_layouts(
    tile_m, tile_k_bytes, row_stride, num_threads, *, swizzle=True
):
    """Map (byte, lane, wave, round) to global A and (byte, wave, round) to LDS.

    The DMA adds lane * 16 to its LDS address in hardware. Its destination
    layout therefore contains only the uniform wave/round offsets. Applying
    the LDS swizzle to the source makes the physical DMA stores contiguous.
    """
    rounds = tile_m * tile_k_bytes // (num_threads * 16)
    waves = num_threads // 64
    blocks_per_row = tile_k_bytes // 16
    rows_per_wave = 64 // blocks_per_row
    shape = (16, (blocks_per_row, rows_per_wave), waves, rounds)

    # Preserve row and K-block coordinates. Flattening them first makes the
    # pitched source layout lower to a div/mul pair in every unrolled round.
    dma_to_coord = fx.make_layout(
        shape,
        (
            fx.E(1),
            (16 * fx.E(1), fx.E(0)),
            rows_per_wave * fx.E(0),
            rows_per_wave * waves * fx.E(0),
        ),
    )
    if swizzle:
        bits = blocks_per_row.bit_length() - 1
        coord_swizzle = fx.static(fx.CoordSwizzleType.get(bits, 0, [0], 4, [1]))
        dma_to_coord = fx.make_composed_layout(coord_swizzle, dma_to_coord)
    source = fx.make_composed_layout(
        fx.make_layout((tile_m, tile_k_bytes), (row_stride, 1)), dma_to_coord
    )
    destination = fx.make_layout((16, waves, rounds), (1, 1024, num_threads * 16))
    return source, destination


def preshuffle_dma_lane_coord(lane, tile_k_bytes):
    """Coordinate for the lane mode returned by ``make_preshuffle_dma_layouts``."""
    blocks_per_row = tile_k_bytes // 16
    return lane % blocks_per_row, lane // blocks_per_row
