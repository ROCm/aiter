# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode in-kernel routing (``decode_id_routing``): canonical 16-row expert blocks.

A one-wave probe kernel runs ``wave_expert`` / ``wave_block_rows`` for every canonical block of
a ``topk_ids`` tensor and writes (expert, rank of the block's first row, block count, 16 packed
rows) back. The oracle is plain torch/Python on the same ids: experts in ascending order, each
padded to a multiple of 16 rows, rows in (token, slot) pair order, padding rows = token ``m``.

Cases: every row count of the decode range (8..128, including non-multiples of 16 and the
``pairs`` capacity boundaries 64/128/256/512), uniform / skewed / all-one-expert id patterns,
and ids outside ``[0, E)`` (they must be ignored, not scribble on the routing scratch).
Single GPU, gfx950.
"""

import pytest
import torch

pytest.importorskip("flydsl")
if (
    not torch.cuda.is_available()
    or "gfx95" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("needs a gfx950 GPU", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx

from aiter.ops.flydsl.kernels.mega_moe_tp.decode_id_routing import (
    TOPK,
    WAVE_BYTES,
    E,
    wave_block_rows,
    wave_expert,
)
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_tile_primitives import (
    id_pairs_for,
    max_sorted_rows,
)

# the most canonical blocks any decode row count can produce (m = 128)
NBLOCKS = max_sorted_rows(128, E, TOPK, 16) // 16


def _make_probe(pairs):
    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, WAVE_BYTES, 16]

    @flyc.kernel(known_block_size=[64, 1, 1])
    def probe(IDS: fx.Tensor, OUT: fx.Tensor, m: fx.Int32):
        lane = fx.Int32(fx.thread_idx.x)
        block = fx.Int32(fx.block_idx.x)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        hb = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        expert, first, total, vals = wave_expert(IDS, m, block, lane, hb, pairs)
        # OUT row of a block: [expert, first, total, 0, 16 packed rows]; -1 = block unused
        base = block * 20
        if block < total:
            packed = wave_block_rows(vals, expert, first, lane, hb)
            if lane < 16:
                OUT[base + 4 + lane] = packed
            if lane == 0:
                OUT[base] = expert
                OUT[base + 1] = first
                OUT[base + 2] = total
        if (block >= total) & (lane == 0):
            OUT[base + 2] = total

    @flyc.jit
    def launch(IDS: fx.Tensor, OUT: fx.Tensor, m: fx.Int32, stream: fx.Stream):
        probe(IDS, OUT, m).launch(grid=(NBLOCKS, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


_PROBES = {}


def _run(ids):
    m = ids.shape[0]
    pairs = id_pairs_for(m)
    if pairs not in _PROBES:
        _PROBES[pairs] = _make_probe(pairs)
    out = torch.full((NBLOCKS, 20), -1, dtype=torch.int32, device=ids.device)
    _PROBES[pairs](ids.contiguous(), out.view(-1), m, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu()


def _reference(ids):
    m = ids.shape[0]
    flat = ids.flatten().tolist()
    valid = [e for e in range(E) if any(v == e for v in flat)]
    blocks = []  # (expert, first, rows)
    for e in valid:
        pairs = [q for q, v in enumerate(flat) if v == e]
        for first in range(0, len(pairs), 16):
            rows = [(q >> 2) | ((q & 3) << 24) for q in pairs[first : first + 16]]
            rows += [m] * (16 - len(rows))
            blocks.append((e, first, rows))
    return blocks


def _check(ids):
    out = _run(ids)
    ref = _reference(ids)
    for b, (e, first, rows) in enumerate(ref):
        got = out[b].tolist()
        assert got[2] == len(ref), (b, got[2], len(ref))
        assert (got[0], got[1]) == (e, first), (b, got[:2], (e, first))
        assert got[4:] == rows, (b, got[4:], rows)
    # past the last block the kernel only reports the block count
    for b in range(len(ref), out.shape[0]):
        assert out[b][2].item() == len(ref)


def _ids(kind, m, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":  # four distinct experts per token
        return torch.stack(
            [torch.randperm(E, generator=g)[:TOPK] for _ in range(m)]
        ).to(torch.int32)
    if kind == "skewed":  # a few hot experts take most of the pairs
        hot = torch.randint(0, 6, (m, TOPK), generator=g)
        return hot.to(torch.int32)
    if kind == "one":  # every token routes to the same four experts: >= 16 rows deep
        return torch.tensor([[0, 1, 2, 3]] * m, dtype=torch.int32)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["uniform", "skewed", "one"])
@pytest.mark.parametrize("m", [8, 15, 16, 17, 24, 32, 33, 64, 65, 96, 97, 128])
def test_canonical_blocks(m, kind):
    _check(_ids(kind, m, seed=m).cuda())


@pytest.mark.parametrize("m", [16, 64, 96])
def test_ids_outside_expert_range_are_ignored(m):
    """Dropped-token markers (-1) and ids >= E must not reach the histogram: the oracle treats
    them as absent, and a stray write past a wave's scratch slice would corrupt its neighbour.
    """
    ids = _ids("uniform", m, seed=1000 + m)
    g = torch.Generator().manual_seed(m)
    bad = torch.rand(ids.shape, generator=g) < 0.2
    # ids 256 + r alias the wave's row table slot r (the padding value of a short block), 336+
    # lands past the wave's scratch slice
    junk = torch.tensor(
        [-1, E, E + 8, 336, 1 << 20] + [256 + r for r in range(16)], dtype=torch.int32
    )
    ids[bad] = junk[torch.randint(0, junk.numel(), (int(bad.sum()),), generator=g)]
    _check(ids.cuda())


def test_pairs_capacity_covers_every_row():
    """The row counts where ``id_pairs_for`` steps (16/32/64/96 -> 64/128/256/512 pair slots)
    must keep the last token's pairs in range."""
    for m in (16, 17, 32, 33, 64, 65, 96, 128):
        assert id_pairs_for(m) >= m * TOPK
