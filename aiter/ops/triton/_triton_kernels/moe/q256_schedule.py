"""Production GPU job map for the Qwen producer's ABI128 map pointer.

Launch eight programs to write the 1024-entry int32 owner/visit map. Shared
geometry must match the producer: SHARED_ROWS=0 retains adaptive M32/M64;
explicit SHARED_ROWS=32, 64, or 128 selects 8192 // SHARED_ROWS shared jobs.
BALANCE_SHARED retains the optional least-accumulated-work shared assignment.
"""

import triton
import triton.language as tl


@triton.jit
def make_job_map(
    Ids,
    Experts,
    Counts,
    Map,
    STYLE: tl.constexpr,
    SHARED_ROWS: tl.constexpr = 128,
    BALANCE_SHARED: tl.constexpr = False,
):
    xcd = tl.program_id(0)
    pos = tl.arange(0, 128)
    ordinal = xcd + pos * 8
    physical_tiles = tl.load(Counts) // 256
    tokens = tl.load(Counts + 1)
    boundary = physical_tiles - 32
    suffix = tl.arange(0, 32)
    suffix_ids = tl.load(Experts + boundary + suffix, mask=boundary >= 1, other=-1)
    preceding = tl.load(Experts + boundary - 1, mask=boundary >= 1, other=512)
    qualifies = (
        (tokens == 8192)
        & (boundary >= 1)
        & (preceding != 512)
        & (tl.sum((suffix_ids == 512).to(tl.int32), 0) == 32)
    )
    if SHARED_ROWS:
        shared_jobs = 8192 // SHARED_ROWS
    else:
        shared_jobs = tl.where((boundary % 256) == 0, 256, 128)
    jobs = tl.where(qualifies, boundary + shared_jobs, physical_tiles)
    valid = ordinal < jobs
    routed = ordinal < boundary
    tile = (ordinal & -64) + (ordinal % 8) * 8 + ((ordinal // 8) % 8)
    tile = tl.where(ordinal < (boundary // 64) * 64, tile, ordinal)
    tile = tl.where(qualifies & ((boundary % 256) != 0) & (tile < 256), tile ^ 1, tile)
    fragments = tl.full((128,), 1, tl.int32)
    for probe in range(1, 8):
        token = (
            tl.load(
                Ids + tile * 256 + probe * 32,
                mask=valid & routed & qualifies,
                other=8192,
            )
            & 0xFFFFFF
        )
        fragments += (token < tokens).to(tl.int32)
    cost = tl.where(
        routed, 410 + fragments * 22, tl.where(shared_jobs == 256, 200, 280)
    )
    key = tl.where(valid, (cost << 16) | (65535 - ordinal), 0).to(tl.uint32)
    ordered = tl.sort(key, descending=True)
    mapped = (65535 - (ordered & 65535)).to(tl.int32)
    mapped = tl.where(ordered != 0, mapped, -1)
    visit = pos // 32
    local_owner = tl.where(visit % 2 == 0, pos % 32, 31 - (pos % 32))
    slot = visit * 256 + local_owner * 8 + xcd
    if BALANCE_SHARED:
        # Preserve routed owner/visit assignments exactly. Shared jobs follow
        # the least accumulated routed/assigned-shared estimate on this XCD.
        owners = tl.arange(0, 32)
        routed_mask = (mapped >= 0) & (mapped < boundary)
        routed_cost = (ordered >> 16).to(tl.int32)
        same_owner = local_owner[:, None] == owners[None, :]
        work = tl.sum(
            tl.where(same_owner & routed_mask[:, None], routed_cost[:, None], 0), 0
        )
        visits = tl.sum((same_owner & routed_mask[:, None]).to(tl.int32), 0)
        assigned = tl.where(routed_mask, mapped, -1)
        shared_first = boundary + ((xcd + 8 - boundary % 8) % 8)
        shared_count = (jobs - shared_first + 7) // 8
        shared_cost = tl.where(shared_jobs == 256, 200, 280)
        for index in range(shared_count):
            owner_key = tl.where(visits < 4, work * 32 + owners, 2147483647)
            selected = tl.min(owner_key, 0) % 32
            chosen = owners == selected
            owner_visit = tl.sum(tl.where(chosen, visits, 0), 0)
            destination = owner_visit * 256 + selected * 8 + xcd
            assigned = tl.where(slot == destination, shared_first + index * 8, assigned)
            work += chosen.to(tl.int32) * shared_cost
            visits += chosen.to(tl.int32)
        # Fallback matches the production planner and producer geometry.
        identity = tl.where(slot < physical_tiles, slot, -1)
        assigned = tl.where(qualifies & (jobs <= 1024), assigned, identity)
        tl.store(Map + slot, assigned)
    else:
        identity = tl.where(slot < jobs, slot, -1)
        mapped = tl.where(qualifies & (jobs <= 1024), mapped, identity)
        tl.store(Map + slot, mapped)
