# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Blocked GDN prefill gate constants, kept import-free so the AOT collector can
read them on hosts without a GPU driver."""

# Target number of blocks per sequence; the per-sequence block size is the
# sequence's chunk count divided by this.
_GDN_K5_TARGET_SEGMENTS = 8
# The blocked path needs at least this many chunks in the longest sequence.
# Blocked loses on balanced two-sequence splits under ~96 chunks ([4064,4065],
# [3000,5129]) and wins from 97.
_BLOCKED_MIN_CHUNKS = 96
# Upper bound on (blocks x heads) per longest-sequence chunk. Serial K5 costs
# ~1.6 us/chunk of the longest sequence; blocked ~0.18 us/block/head plus a
# ~25-chunk carry pass. gfx950 break-even is work ratio ~9; 6 leaves margin and
# bounds scratch.
_BLOCKED_MAX_WORK_PER_CHUNK = 6
