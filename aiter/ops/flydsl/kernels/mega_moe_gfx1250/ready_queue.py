# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Word offsets for the optional S1F queue of completed A tiles.

The queue follows the existing affinity heads in addr_runs. Each entry stores
M+1 (zero means unpublished) and exposes NT independently claimable N tasks.
Compact-plan clears all words before each S1F launch on the same stream.
"""

AFFINITY_WORDS = 256 * 32
HEAD = 0
TAIL = 32
COUNTS = 64


def storage_words(flag_tile_cap):
    """Return header, completion counters and append-only queue capacity."""
    return COUNTS + 2 * flag_tile_cap
