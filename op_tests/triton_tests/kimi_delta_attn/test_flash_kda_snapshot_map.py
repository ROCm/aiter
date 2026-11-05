# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests for where FlashKDA puts its state snapshots.

`snapshot_chunk_rows` maps each target ``(n, off)`` to one chunk of the K1
grid. Pass C stores row ``rows[chunk_base + j]`` when it enters chunk ``j`` of
a segment. These tests walk that loop on the CPU, with the reference
recurrence moving the state forward one chunk at a time. Then they compare each
stored state with the reference run over the first ``off`` tokens only. They
launch no kernel, so they test the placement apart from it. They still need
an importable aiter, as on the CI runners: importing the helper runs the
``kimi_delta_attn`` package, whose ``flash_kda`` reads the GPU architecture at
import.
"""

import random

import pytest
import torch

from aiter.ops.triton._triton_kernels.kimi_delta_attn.utils.snapshot import (
    snapshot_chunk_rows,
)
from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref, kda_gate_ref, l2norm_ref

# FLASH_KDA_CHUNK. The module that defines it needs a GPU to import.
C = 32
H, K, V = 2, 8, 8
LOWER_BOUND = -5.0


def _inputs(lens, seed=0):
    """Packed ``[1, T, ...]`` inputs, with the gate, l2norm and beta sigmoid applied.

    They are applied once to the whole batch, so a slice of a sequence sees the
    same values as the whole sequence does.
    """
    gen = torch.Generator().manual_seed(seed)
    T = sum(lens)

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    A_log, dt_bias = randn(H).abs() * 0.5, randn(H * K) * 0.1
    return {
        "q": l2norm_ref(randn(1, T, H, K)),
        "k": l2norm_ref(randn(1, T, H, K)),
        "v": randn(1, T, H, V),
        "g": kda_gate_ref(randn(1, T, H, K) * 0.1, A_log, dt_bias, LOWER_BOUND),
        "beta": torch.sigmoid(randn(1, T, H)),
    }


def _bounds(lens):
    out, bos = [], 0
    for length in lens:
        out.append((bos, bos + length))
        bos += length
    return tuple(out)


def _state(x, bos, eos, h0=None):
    """The reference state after tokens ``[bos, eos)``, from ``h0`` (zero if None)."""
    _, state = chunk_kda_ref(
        **{name: t[:, bos:eos] for name, t in x.items()},
        scale=K**-0.5,
        initial_state=h0,
        output_final_state=True,
    )
    return state


def _segments(seqs, chunks_per_seg):
    """``(n, chunk_base, nchunks, tok_base, tok_end)`` per segment.

    This is the layout of `_build_segments`. The GPU suite checks that the two
    agree.
    """
    out, g_chunk = [], 0
    for n, (bos, eos) in enumerate(seqs):
        nch = -(-(eos - bos) // C)
        for s in range(max(1, -(-nch // chunks_per_seg))):
            c0 = s * chunks_per_seg
            m = min(chunks_per_seg, nch - c0)
            end = min(bos + (c0 + m) * C, eos)
            out.append((n, g_chunk + c0, m, bos + c0 * C, end))
        g_chunk += nch
    return out


def _walk(x, seqs, rows, chunks_per_seg, n_rows):
    """Pass C on the CPU.

    Each segment starts from its true incoming state. When it enters chunk
    ``j``, it stores the state in row ``rows[chunk_base + j]``.
    """
    snaps = [None] * n_rows
    carried = {}
    for n, base, nchunks, tok, end in _segments(seqs, chunks_per_seg):
        state = carried.get(n)
        for j in range(nchunks):
            row = rows[base + j]
            if row >= 0:
                snaps[row] = state.clone()
            t0 = tok + j * C
            state = _state(x, t0, min(t0 + C, end), state)
        carried[n] = state
    return snaps


def _interior(seqs, every=1):
    """Every ``every``-th interior chunk boundary of each sequence, as targets."""
    return [
        (n, off)
        for n, (bos, eos) in enumerate(seqs)
        for off in range(C, eos - bos, C)[::every]
    ]


@pytest.mark.parametrize("chunks_per_seg", [1, 2, 3, 1 << 20])
@pytest.mark.parametrize(
    "lens",
    [
        [160],  # one sequence, four interior boundaries
        [100, 64, 33],  # several sequences, all ending mid-chunk or on a boundary
        [0, 130, 31, 97],  # an empty one and one with no interior boundary
    ],
)
def test_every_snapshot_is_the_state_of_its_prefix(lens, chunks_per_seg):
    seqs = _bounds(lens)
    x = _inputs(lens)
    targets = _interior(seqs)
    random.Random(0).shuffle(targets)
    rows = snapshot_chunk_rows(seqs, C, tuple(targets))
    assert len(rows) == sum(-(-length // C) for length in lens)
    assert sorted(r for r in rows if r >= 0) == list(range(len(targets)))

    snaps = _walk(x, seqs, rows, chunks_per_seg, len(targets))
    for i, (n, off) in enumerate(targets):
        bos = seqs[n][0]
        assert torch.equal(snaps[i], _state(x, bos, bos + off)), (n, off)


def test_rows_follow_the_order_of_the_targets():
    seqs = _bounds([200, 96])
    x = _inputs([200, 96], seed=1)
    targets = list(reversed(_interior(seqs, every=2)))
    rows = snapshot_chunk_rows(seqs, C, tuple(targets))
    snaps = _walk(x, seqs, rows, 2, len(targets))
    for i, (n, off) in enumerate(targets):
        bos = seqs[n][0]
        assert torch.equal(snaps[i], _state(x, bos, bos + off))


def test_no_targets_store_nothing():
    seqs = _bounds([96, 40])
    rows = snapshot_chunk_rows(seqs, C, ())
    assert rows == [-1] * (3 + 2)
    assert _walk(_inputs([96, 40]), seqs, rows, 1, 0) == []


@pytest.mark.parametrize(
    "targets",
    [
        [(0, 0)],  # the initial state
        [(0, 96)],  # the final state of the 96-token sequence
        [(0, 33)],  # off the chunk grid
        [(0, 31)],  # off the chunk grid, before the first boundary
        [(0, 128)],  # past the end
        [(1, 32)],  # a 20-token sequence has no interior boundary
        [(2, 32)],  # no such sequence
        [(-1, 32)],
        [(0, 32), (0, 64), (0, 32)],  # the same position twice
    ],
)
def test_rejects_a_position_it_cannot_store(targets):
    with pytest.raises(ValueError):
        snapshot_chunk_rows(_bounds([96, 20]), C, tuple(targets))
