# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Codec-level tests for the QRInt4 wire formats.

Single-GPU and no IPC: these check the *codec*, not the collective. The
schedule tests in ``test_flydsl_qr_int4.py`` cover the wire protocol.

Two things are worth checking here that a collective test cannot.

First, packing is checked on the **words**, not on the reconstruction. A
half-swap between the lane pair that shares an INT6 ``hi2`` slot reconstructs
plausibly -- the values are still in range, still roughly the right magnitude --
and would survive a reconstruction-only check while quietly costing accuracy at
every hop. Comparing the packed i32 against a host reference catches it.

Second, the error model that sizes the codec is asserted rather than trusted.
``qr_codec_ref`` predicts ~21 dB for a TP8 ring on an INT6 reduce-scatter lap
against ~15.7 on INT4, and those numbers are why INT6 is the TP8 default; if
the model drifts, the default should be revisited rather than silently kept.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from op_tests.flydsl_tests.qr_codec_ref import (
    CODECS_REF,
    GROUP,
    INT4_REF,
    INT6_REF,
    allreduce_mesh,
    allreduce_ring,
    pack_int4,
    pack_int6,
    quantize,
    roundtrip,
    sqnr_db,
    unpack_int4,
    unpack_int6,
)

# Values per thread-atom, and threads per block -- one "row" of the reference
# packing is one rank-tile's worth of threads.
ATOM_VALUES = 8
BLOCK = 256


def _rows(n_rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n_rows, BLOCK, ATOM_VALUES, generator=g, dtype=torch.float32) * 0.1
    )


@pytest.mark.parametrize("codec_name", ("int4", "int6"))
def test_pack_unpack_roundtrip_is_lossless(codec_name):
    """The packing itself must lose nothing -- only the quantizer may."""
    codec = CODECS_REF[codec_name]
    x = _rows(4, seed=7)
    q, _ = quantize(x.reshape(4, -1), codec)
    q = q.reshape(4, BLOCK, ATOM_VALUES)
    if codec is INT4_REF:
        back = unpack_int4(pack_int4(q))
    else:
        lo4, hi2 = pack_int6(q)
        back = unpack_int6(lo4, hi2)
    assert torch.equal(back, q)


def test_int6_low_plane_matches_int4_layout():
    """INT6's nibble plane is byte-identical in layout to INT4's.

    Not an incidental property: it is why ``_fanout_to_next`` needs nothing but
    a different sector count, and why the 1024 B region keeps its offsets.
    """
    q = torch.randint(0, 64, (2, BLOCK, ATOM_VALUES), dtype=torch.int32)
    lo4, _ = pack_int6(q)
    assert torch.equal(lo4, pack_int4(q & 0xF))


def test_int6_hi2_plane_is_half_the_width():
    """512 B per rank-tile: two threads to an i32, 16 dense bits each."""
    q = torch.randint(0, 64, (3, BLOCK, ATOM_VALUES), dtype=torch.int32)
    lo4, hi2 = pack_int6(q)
    assert lo4.shape == (3, BLOCK)
    assert hi2.shape == (3, BLOCK // 2)


def test_int6_hi2_pairing_is_not_swapped():
    """The even thread of a pair owns the *low* half of the shared i32.

    Written as its own case because a swap here is the one packing bug that
    reconstructs plausibly: every value stays in range and only the top two
    bits move between two neighbouring threads.
    """
    q = torch.zeros(1, BLOCK, ATOM_VALUES, dtype=torch.int32)
    # Thread 0 gets the top 2 bits set on element 0; thread 1 gets nothing.
    q[0, 0, 0] = 0x30
    _, hi2 = pack_int6(q)
    assert hi2[0, 0] & 0xFFFF == 0x3, hex(int(hi2[0, 0]))
    assert hi2[0, 0] >> 16 == 0

    q = torch.zeros(1, BLOCK, ATOM_VALUES, dtype=torch.int32)
    q[0, 1, 0] = 0x30  # odd thread -> high half
    _, hi2 = pack_int6(q)
    assert hi2[0, 0] & 0xFFFF == 0
    assert hi2[0, 0] >> 16 == 0x3, hex(int(hi2[0, 0]))


@pytest.mark.parametrize("codec_name,min_db", (("int4", 21.0), ("int6", 32.0)))
def test_single_roundtrip_sqnr(codec_name, min_db):
    """Two extra bits are worth ~12 dB on one quantization."""
    codec = CODECS_REF[codec_name]
    x = _rows(8, seed=11).reshape(8, -1)
    got = roundtrip(x, codec).to(torch.float32)
    assert sqnr_db(got, x) >= min_db


def test_degenerate_groups_do_not_produce_nan():
    """Zero and sub-2^-7 groups drive the encode reciprocal to its ceiling.

    The kernel materialises ``1/d`` as fp16, so an unclamped reciprocal becomes
    Inf and ``0 * Inf`` is NaN. INT6 has a quarter of INT4's headroom here, so
    both codecs are checked.
    """
    x = _rows(4, seed=13).reshape(4, -1)
    x[0] = 0.0
    x[1] *= 1e-8
    x[2, : 4 * GROUP] = 0.0
    for codec in (INT4_REF, INT6_REF):
        got = roundtrip(x, codec)
        assert torch.isfinite(got).all(), codec.name


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_ring_int6_beats_the_18db_floor(world_size):
    """The reason INT6 is the TP8 default, asserted rather than assumed.

    INT4 is *not* asserted against the floor here: it is under it at TP8 by
    construction, which is the whole point.
    """
    xs = [_rows(2, seed=100 + r).reshape(2, -1) for r in range(world_size)]
    ref = torch.stack(xs).sum(0)
    got = allreduce_ring(xs, INT6_REF, INT4_REF)
    assert sqnr_db(got, ref) >= 18.0


def test_ring_int4_at_tp8_is_the_configuration_int6_replaces():
    """Pins the ~15 dB that motivated the change, so a drift is visible.

    If this starts passing 18.0, the INT6 default is no longer buying what it
    was introduced to buy and ``_RS_INT6_MIN_WORLD`` should be revisited.
    """
    xs = [_rows(2, seed=100 + r).reshape(2, -1) for r in range(8)]
    ref = torch.stack(xs).sum(0)
    db = sqnr_db(allreduce_ring(xs, INT4_REF, INT4_REF), ref)
    assert 14.0 <= db <= 17.0, db


def _resolve(monkeypatch, algorithm, world_size, env=None, rs=None, ag=None):
    from aiter.ops.flydsl.kernels import qr_int4

    # Patch the parsed value rather than os.environ: the variable is read once
    # at import, which is the behaviour under test everywhere else.
    monkeypatch.setattr(qr_int4, "AITER_ALL_REDUCE_CODEC", env)
    monkeypatch.setattr(qr_int4, "_warned_codecs", set())
    return qr_int4._resolve_codecs(qr_int4.ALGORITHMS[algorithm], world_size, rs, ag)


@pytest.mark.parametrize(
    "world_size,expected",
    ((2, ("int4", "int4")), (4, ("int4", "int4")), (8, ("int6", "int4"))),
)
def test_ring_codec_defaults_widen_only_at_tp8(monkeypatch, world_size, expected):
    """TP8 is the only world size where INT4 misses the floor."""
    assert _resolve(monkeypatch, "ring", world_size) == expected


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_mesh_is_int4_at_every_world_size(monkeypatch, world_size):
    """The mesh has no separable lap, so the per-N default must not leak into it."""
    assert _resolve(monkeypatch, "mesh", world_size) == ("int4", "int4")


@pytest.mark.parametrize("env,expected", (("int4", "int4"), ("int6", "int6")))
def test_env_override_sets_both_laps(monkeypatch, env, expected):
    """One variable, both laps -- including the all-gather lap, which has no
    other way to reach INT6."""
    assert _resolve(monkeypatch, "ring", 8, env=env) == (expected, expected)


def test_explicit_argument_outranks_the_environment(monkeypatch):
    got = _resolve(monkeypatch, "ring", 8, env="int6", rs="int4")
    assert got == ("int4", "int6")


def test_env_that_the_schedule_cannot_build_falls_back(monkeypatch):
    """A process-wide variable must not break an unrelated call site."""
    assert _resolve(monkeypatch, "mesh", 8, env="int6") == ("int4", "int4")


def test_explicit_codec_the_schedule_cannot_build_raises(monkeypatch):
    """Unlike the environment: naming it in code is a programming error."""
    with pytest.raises(ValueError, match="rs_codec"):
        _resolve(monkeypatch, "mesh", 8, rs="int6")


def test_mesh_sqnr_is_world_size_independent():
    """The mesh quantizes twice regardless of N, so its SQNR does not move.

    This is the baseline the ring is compared against, and the reason the ring
    needed a per-N answer where the mesh did not.
    """
    seen = []
    for world_size in (2, 4, 8):
        xs = [_rows(2, seed=200 + r).reshape(2, -1) for r in range(world_size)]
        ref = torch.stack(xs).sum(0)
        seen.append(sqnr_db(allreduce_mesh(xs, INT4_REF), ref))
    assert max(seen) - min(seen) < 0.5, seen
