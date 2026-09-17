# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc.

"""How the implicit-GEMM conv3d is spread over blocks, and what each block owns.

Three things share one arithmetic here, which is why they live together: the
launch's grid dimensions, the asserts about what a grid dimension can hold,
and the decode every block runs to turn its block ids back into an (M, N, K)
tile. Deriving the decode from anything but the grid it was launched with is
how a block ends up owning a tile nobody sized for.

Four independent things are folded onto three grid axes:

- M over ``grid.x``, spilling into ``grid.z`` as "M chunks" when the tile
  count passes what one axis holds;
- N over ``grid.y``, over-provisioned to ``groups * tiles_per_group`` so a
  grouped conv's N tail is per group rather than global;
- split-K over the rest of ``grid.z``;
- and, when M fits one axis, a WGM swizzle over x/y that walks WGM rows of M
  before moving on in N, so concurrent blocks share B tiles.
"""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu

# A grid dimension is 32-bit in blocks on x and 16-bit on y/z; x is further
# capped so that block_id.x * block_threads stays inside 32 bits.
MAX_GRID_YZ = 65535


class LaunchGrid(NamedTuple):
    """The grid one compiled conv3d launches on, and what a block decodes with.

    A NamedTuple for the same reason the other plans are: only tuples and
    scalars reach FlyDSL's cache key.
    """

    # The launch itself.
    grid_x: int
    grid_y: int
    grid_z: int
    block_threads: int

    # M: how many tiles there are, and how they fold onto x and z.
    grid_m: int
    m_chunks: int
    tile_m: int
    row_chk: bool
    wgm: int

    # N: per-group tiling, and whether the last tile of a group is partial.
    tile_n: int
    tiles_per_group: int
    n_tail: bool
    groups: int
    kg: int
    cgp: int

    # K: the split, in whole tiles.
    tile_k: int
    tiles_per_split: int
    splitk: int
    use_splitk: bool


def make_launch_grid(param, geom, *, tile_m, tile_n, tile_k, block_threads):
    """The LaunchGrid for one problem and launch config, or an assertion."""
    k, groups = param.k, param.groups
    kg = k // groups
    npq = geom.npq

    tiles_per_group = (kg + tile_n - 1) // tile_n
    n_tail = kg % tile_n != 0
    grid_y = groups * tiles_per_group

    k_tiles = (geom.crs + tile_k - 1) // tile_k
    splitk = max(1, min(param.splitk, k_tiles))
    tiles_per_split = k_tiles // splitk

    grid_m = (npq + tile_m - 1) // tile_m
    max_grid_x = 0xFFFFFFFF // block_threads
    grid_x = min(grid_m, max_grid_x)
    m_chunks = (grid_m + grid_x - 1) // grid_x

    assert grid_y <= MAX_GRID_YZ, (
        f"grid.y = {grid_y} exceeds the {MAX_GRID_YZ}-block limit"
    )
    assert m_chunks * splitk <= MAX_GRID_YZ, (
        f"grid.z = {m_chunks} M-chunks x {splitk} splits exceeds the {MAX_GRID_YZ}-block limit"
    )

    return LaunchGrid(
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=m_chunks * splitk,
        block_threads=block_threads,
        grid_m=grid_m,
        m_chunks=m_chunks,
        tile_m=tile_m,
        # The last M tile is partial, or chunking over-provisioned the x axis:
        # either way some block owns rows past npq and must not write them.
        row_chk=(npq % tile_m != 0) or (grid_x * m_chunks > grid_m),
        # Chunked M already uses z, so a swizzle over x/y would reorder blocks
        # that are no longer adjacent in M. WGM only applies to the flat case.
        wgm=1 if m_chunks > 1 else max(1, int(param.wgm)),
        tile_n=tile_n,
        tiles_per_group=tiles_per_group,
        n_tail=n_tail,
        groups=groups,
        kg=kg,
        cgp=geom.cgp,
        tile_k=tile_k,
        tiles_per_split=tiles_per_split,
        splitk=splitk,
        use_splitk=splitk > 1,
    )


class BlockCoords(NamedTuple):
    """Where in the GEMM this block's tile sits. Device values, not constants."""

    m_offset: object
    n_offset: object
    n_local: object
    ch_base: object
    k_off: object


def block_coords(grid):
    """Decode this block's ids into the tile it owns.

    ``n_local`` is the column within the group and ``n_offset`` the global
    one; they differ only for a grouped conv, where the N grid is per group.
    ``ch_base`` is the group's first input channel, which only the gather
    needs, and is None when there is one group.
    """
    if const_expr(grid.m_chunks > 1):
        m_chunk = fx.Int64(gpu.block_id("z")) % fx.Int64(grid.m_chunks)
        m_offset = (
            fx.Int64(gpu.block_id("x")) + m_chunk * fx.Int64(grid.grid_x)
        ) * grid.tile_m
        n_tile = fx.Int32(gpu.block_id("y"))
    elif const_expr(grid.wgm > 1):
        pid = fx.Int64(gpu.block_id("x")) + fx.Int64(gpu.block_id("y")) * fx.Int64(
            grid.grid_m
        )
        blocks_per_swizzle = fx.Int64(grid.wgm * grid.grid_y)
        swizzle_id = pid // blocks_per_swizzle
        first_m = swizzle_id * fx.Int64(grid.wgm)
        # The last swizzle group is short when grid_m is not a multiple of WGM.
        swizzle_rows = fx.min(fx.Int64(grid.grid_m) - first_m, fx.Int64(grid.wgm))
        local = pid % blocks_per_swizzle
        m_offset = (first_m + (local % swizzle_rows)) * grid.tile_m
        n_tile = local // swizzle_rows
    else:
        m_offset = fx.Int32(gpu.block_id("x")) * grid.tile_m
        n_tile = fx.Int32(gpu.block_id("y"))

    if const_expr(grid.groups > 1):
        gi = n_tile // grid.tiles_per_group
        n_local = (n_tile % grid.tiles_per_group) * grid.tile_n
        n_offset = gi * grid.kg + n_local
        ch_base = gi * grid.cgp
    else:
        n_offset = n_tile * grid.tile_n
        n_local = n_offset
        ch_base = None

    if const_expr(grid.use_splitk):
        if const_expr(grid.m_chunks > 1):
            split_idx = fx.Int64(gpu.block_id("z")) // fx.Int64(grid.m_chunks)
        else:
            split_idx = fx.Int64(gpu.block_id("z"))
        k_off = split_idx * (grid.tiles_per_split * grid.tile_k)
    else:
        k_off = 0

    return BlockCoords(
        m_offset=m_offset,
        n_offset=n_offset,
        n_local=n_local,
        ch_base=ch_base,
        k_off=k_off,
    )
