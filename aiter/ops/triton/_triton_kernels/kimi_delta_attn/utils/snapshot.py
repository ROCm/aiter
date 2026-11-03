# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side placement of FlashKDA state snapshots on the K1 chunk grid.

Pure Python, so the placement can be tested without a GPU.
"""

__all__ = ["snapshot_chunk_rows"]


def snapshot_chunk_rows(
    seqs: tuple[tuple[int, int], ...],
    chunk: int,
    targets: tuple[tuple[int, int], ...],
) -> list[int]:
    """The ``snapshot_out`` row of each global chunk, or -1 where there is none.

    Chunks are numbered as the K1 workspace numbers them: sequence by sequence,
    ``ceil(length / chunk)`` per sequence, for any segmentation. A target
    ``(n, off)`` asks for the state of sequence ``n`` after its first ``off``
    tokens, which is the state that enters chunk ``off // chunk``. So ``off``
    must be a multiple of ``chunk`` strictly inside the sequence. At 0 the state
    is the initial state, and at the end it is the final state.

    A sequence can have any number of targets, in any order. Target ``i`` goes
    to row ``i``. Two targets at the same position are an error.
    """
    starts, total = [], 0
    for bos, eos in seqs:
        starts.append(total)
        total += -(-(eos - bos) // chunk)
    rows = [-1] * total
    for i, (n, off) in enumerate(targets):
        if not 0 <= n < len(seqs):
            raise ValueError(f"snapshot target {i} names sequence {n} of {len(seqs)}")
        length = seqs[n][1] - seqs[n][0]
        if not 0 < off < length or off % chunk:
            raise ValueError(
                f"snapshot target {i} at token {off} is not an interior multiple "
                f"of the {chunk}-token chunk of a {length}-token sequence"
            )
        c = starts[n] + off // chunk
        if rows[c] >= 0:
            raise ValueError(
                f"snapshot targets {rows[c]} and {i} are the same position "
                f"(sequence {n}, token {off})"
            )
        rows[c] = i
    return rows
