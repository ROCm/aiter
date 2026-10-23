# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Production owner/visit map for the Q256 producer's fixed ABI128 geometry.

Eight programs write all 1024 int32 slots. Only the compact 8192-token layout
with exactly 32 trailing shared physical tiles uses cost-ranked scheduling;
other layouts retain the identity fallback, including over-capacity layouts.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Producer ABI geometry: changing these requires a matching producer change.
ABI_TOKENS = tl.constexpr(8192)
ABI_TILE_ROWS = tl.constexpr(256)
ABI_SHARED_EXPERT = tl.constexpr(512)
ABI_SHARED_ROWS = tl.constexpr(128)
ABI_SHARED_TILES = ABI_TOKENS // ABI_TILE_ROWS
ABI_SHARED_JOBS = ABI_TOKENS // ABI_SHARED_ROWS
ABI_XCDS = tl.constexpr(8)
ABI_OWNERS_PER_XCD = tl.constexpr(32)
ABI_VISITS = tl.constexpr(4)
ABI_MAP_ENTRIES = ABI_XCDS * ABI_OWNERS_PER_XCD * ABI_VISITS
ABI_SLOTS_PER_XCD = ABI_OWNERS_PER_XCD * ABI_VISITS
ABI_FRAGMENT_ROWS = tl.constexpr(32)

# Ranking estimates, not tensor dimensions or producer ABI fields.
ROUTED_BASE_COST = tl.constexpr(410)
ROUTED_FRAGMENT_COST = tl.constexpr(22)
SHARED_JOB_COST = tl.constexpr(280)

_make_job_map_repr = make_kernel_repr("q256_make_job_map_abi128", [])


@triton.jit(repr=_make_job_map_repr)
def make_job_map(Ids, Experts, Counts, Map):
    xcd = tl.program_id(0)
    pos = tl.arange(0, ABI_SLOTS_PER_XCD)
    ordinal = xcd + pos * ABI_XCDS
    physical_tiles = tl.load(Counts) // ABI_TILE_ROWS
    tokens = tl.load(Counts + 1)
    boundary = physical_tiles - ABI_SHARED_TILES
    suffix = tl.arange(0, ABI_SHARED_TILES)
    suffix_ids = tl.load(Experts + boundary + suffix, mask=boundary >= 1, other=-1)
    preceding = tl.load(
        Experts + boundary - 1, mask=boundary >= 1, other=ABI_SHARED_EXPERT
    )
    qualifies = (
        (tokens == ABI_TOKENS)
        & (boundary >= 1)
        & (preceding != ABI_SHARED_EXPERT)
        & (
            tl.sum((suffix_ids == ABI_SHARED_EXPERT).to(tl.int32), 0)
            == ABI_SHARED_TILES
        )
    )
    jobs = tl.where(qualifies, boundary + ABI_SHARED_JOBS, physical_tiles)
    valid = ordinal < jobs
    routed = ordinal < boundary
    # Transpose complete 8x8 routed groups; leave the boundary tail unchanged.
    tile = (ordinal & -64) + (ordinal % 8) * 8 + ((ordinal // 8) % 8)
    tile = tl.where(ordinal < (boundary // 64) * 64, tile, ordinal)
    tile = tl.where(
        qualifies & ((boundary % ABI_TILE_ROWS) != 0) & (tile < ABI_TILE_ROWS),
        tile ^ 1,
        tile,
    )
    fragments = tl.full((ABI_SLOTS_PER_XCD,), 1, tl.int32)
    for probe in range(1, ABI_TILE_ROWS // ABI_FRAGMENT_ROWS):
        token = (
            tl.load(
                Ids + tile * ABI_TILE_ROWS + probe * ABI_FRAGMENT_ROWS,
                mask=valid & routed & qualifies,
                other=ABI_TOKENS,
            )
            & 0xFFFFFF
        )
        fragments += (token < tokens).to(tl.int32)
    cost = tl.where(
        routed, ROUTED_BASE_COST + fragments * ROUTED_FRAGMENT_COST, SHARED_JOB_COST
    )
    key = tl.where(valid, (cost << 16) | (65535 - ordinal), 0).to(tl.uint32)
    ordered = tl.sort(key, descending=True)
    mapped = (65535 - (ordered & 65535)).to(tl.int32)
    mapped = tl.where(ordered != 0, mapped, -1)
    visit = pos // ABI_OWNERS_PER_XCD
    local_owner = tl.where(
        visit % 2 == 0,
        pos % ABI_OWNERS_PER_XCD,
        ABI_OWNERS_PER_XCD - 1 - (pos % ABI_OWNERS_PER_XCD),
    )
    slot = visit * (ABI_OWNERS_PER_XCD * ABI_XCDS) + local_owner * ABI_XCDS + xcd
    identity = tl.where(slot < jobs, slot, -1)
    mapped = tl.where(qualifies & (jobs <= ABI_MAP_ENTRIES), mapped, identity)
    tl.store(Map + slot, mapped)
