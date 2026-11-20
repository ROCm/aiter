# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""iq2r_glm53_unpack is the exact inverse of iq2r_glm53_pack (CPU only).

Canonical IQ2R-v2 buffers come from the format's own writers with random
indices, signs, levels, d and gains, so every bit of the records is exercised,
for 512- and 1,024-entry codebooks.
"""

import pytest
import torch

from aiter.iq2r_glm53 import (
    HIDDEN,
    iq2r_glm53_pack,
    iq2r_glm53_slice_down_aux,
    iq2r_glm53_slice_gate_aux,
    iq2r_glm53_unpack,
)
from aiter.ops.iq2r_format import IQ2RMetadata, iq2r_slice_input_data
from aiter.ops.iq2r_v2 import (
    IQ2R_V2_CODEBOOK_MAX_BYTE,
    IQ2R_V2_CODEBOOK_MIN_BYTE,
    IQ2R_V2_LEVELS_C4,
    iq2r_v2_materialize,
    iq2r_v2_write_auxiliary,
    iq2r_v2_write_data,
)

LEVELS = IQ2R_V2_LEVELS_C4


def _canonical(n: int, k: int, generator: torch.Generator, entries: int):
    indices = torch.randint(0, entries, (n, k // 8), generator=generator)
    negative = torch.rand((n, k), generator=generator) < 0.5
    levels = torch.randint(0, 16, (n, k // 32), generator=generator)
    data = iq2r_v2_write_data(indices, negative, levels, LEVELS, entries)
    book = torch.randint(
        IQ2R_V2_CODEBOOK_MIN_BYTE,
        IQ2R_V2_CODEBOOK_MAX_BYTE + 1,
        (1, entries, 8),
        generator=generator,
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    d = torch.randint(100, 140, (n, k // 256), generator=generator, dtype=torch.uint8)
    gains = tuple(float(g) for g in 0.5 + torch.rand(2, generator=generator))
    return data, iq2r_v2_write_auxiliary(book.float(), d, gains)


def _layer(intermediate: int, experts: int, seed: int, entries: int = 512):
    generator = torch.Generator().manual_seed(seed)
    gate = [
        _canonical(2 * intermediate, HIDDEN, generator, entries) for _ in range(experts)
    ]
    down = [
        _canonical(HIDDEN, intermediate, generator, entries) for _ in range(experts)
    ]

    def stack(parts, i):
        return torch.stack([p[i] for p in parts])

    return stack(gate, 0), stack(gate, 1), stack(down, 0), stack(down, 1)


@pytest.mark.parametrize("entries", [512, 1024])
@pytest.mark.parametrize("intermediate", [256, 512, 2048])
def test_unpack_inverts_pack(intermediate, entries):
    canonical = _layer(intermediate, 2, intermediate, entries)
    packed = iq2r_glm53_pack(*canonical, intermediate_size=intermediate, levels=LEVELS)
    unpacked = iq2r_glm53_unpack(*packed, intermediate_size=intermediate, levels=LEVELS)
    # Canonical bytes, including the zero padding blocks of the gate/up records.
    for got, want in zip(unpacked, canonical):
        assert torch.equal(got, want)
    repacked = iq2r_glm53_pack(*unpacked, intermediate_size=intermediate, levels=LEVELS)
    for got, want in zip(repacked, packed):
        assert torch.equal(got, want)


@pytest.mark.parametrize("entries", [512, 1024])
def test_unpack_of_tp_slices(entries):
    """Unpacking a TP shard sliced from a packed full layer equals slicing the canonical weights."""
    full, shard, rank = 1024, 256, 2
    canonical = _layer(full, 1, 7, entries)
    packed = iq2r_glm53_pack(*canonical, intermediate_size=full, levels=LEVELS)
    from aiter.iq2r_glm53 import iq2r_glm53_slice_gate

    rows = (2 * shard * rank, 2 * shard)
    sliced = (
        iq2r_glm53_slice_gate(packed[0], *rows, entries),
        iq2r_glm53_slice_gate_aux(packed[1], *rows, 2 * full, LEVELS.classes, entries),
        iq2r_slice_input_data(
            packed[2], IQ2RMetadata(HIDDEN, full, entries), shard * rank, shard
        ),
        iq2r_glm53_slice_down_aux(
            packed[3], shard * rank, shard, full, LEVELS.classes, entries
        ),
    )
    gd, ga, dd, da = iq2r_glm53_unpack(*sliced, intermediate_size=shard, levels=LEVELS)

    def dense(data, aux, n, k):
        return iq2r_v2_materialize(
            data, aux, n, k, LEVELS, apply_gain=False, codebook_entries=entries
        )

    dense_gate = dense(canonical[0][0], canonical[1][0], 2 * full, HIDDEN)
    dense_down = dense(canonical[2][0], canonical[3][0], HIDDEN, full)
    got_gate = dense(gd[0], ga[0], 2 * shard, HIDDEN)
    got_down = dense(dd[0], da[0], HIDDEN, shard)
    assert torch.equal(got_gate, dense_gate[rows[0] : rows[0] + rows[1]])
    assert torch.equal(got_down, dense_down[:, shard * rank : shard * (rank + 1)])
