# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure Python inputs and reference job map for the ABI128 scheduler."""


def make_inputs(boundary=512, tokens=8192, suffix=32, preceding_shared=False):
    """Padded 256-row tiles with varied fragment occupancy and tagged IDs."""
    physical_tiles = boundary + suffix
    ids = [tokens] * (physical_tiles * 256)
    experts = [tile % 512 for tile in range(boundary)] + [512] * suffix
    if preceding_shared and boundary:
        experts[boundary - 1] = 512
    for tile in range(boundary):
        for fragment in range(1 + tile % 8):
            for row in range(32):
                ids[tile * 256 + fragment * 32 + row] = (3 << 24) | row
    return ids, experts, [len(ids), tokens]


def reference_job_map(sorted_ids, sorted_experts, counts):
    """Plan tasks as Python records, without packed keys or a sorting network.

    Each XCD receives every eighth logical job. Routed jobs rank by occupied
    32-row fragments, then ordinal; shared jobs follow in ordinal order. The
    ranked queue is dealt to 32 owners in alternating forward/reverse visits.
    """
    physical_tiles = counts[0] // 256
    boundary = physical_tiles - 32
    qualifies = (
        counts[1] == 8192
        and boundary >= 1
        and sorted_experts[boundary - 1] != 512
        and sorted_experts[boundary:physical_tiles] == [512] * 32
    )
    jobs = boundary + 64 if qualifies else physical_tiles
    if not qualifies or jobs > 1024:
        return [slot if slot < jobs else -1 for slot in range(1024)]

    queues = [[] for _ in range(8)]
    for ordinal in range(jobs):
        if ordinal < boundary:
            tile = ordinal
            if ordinal < boundary // 64 * 64:
                group, cell = divmod(ordinal, 64)
                row, column = divmod(cell, 8)
                tile = group * 64 + column * 8 + row
            if boundary % 256 and tile < 256:
                tile ^= 1
            fragments = 1 + sum(
                (sorted_ids[tile * 256 + fragment * 32] & 0xFFFFFF) < counts[1]
                for fragment in range(1, 8)
            )
            priority = (0, -fragments, ordinal)
        else:
            priority = (1, 0, ordinal)
        queues[ordinal % 8].append((priority, ordinal))

    result = [-1] * 1024
    for xcd, queue in enumerate(queues):
        for rank, (_, ordinal) in enumerate(sorted(queue)):
            visit, owner = divmod(rank, 32)
            if visit % 2:
                owner = 31 - owner
            result[visit * 256 + owner * 8 + xcd] = ordinal
    return result
