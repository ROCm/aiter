# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The INT4 sequence-parallel router decides and pads on the host. CPU only: the
engine is a stand-in that checks the rules the real one enforces (whole tiles,
aligned contiguous input, zero padding rows) and computes a layout-revealing
result; shapes too large to allocate are meta tensors, which the router only
reads shapes from."""

import pickle
from types import SimpleNamespace

import pytest
import torch
from conftest import requires_built_aiter

requires_built_aiter()
pytest.importorskip("flydsl")

from aiter.dist.device_communicators import sp_collectives as sp
from aiter.ops.flydsl import allreduce_policy as policy

BF16 = torch.bfloat16
GFX950_TP2 = {"arch": "gfx950", "link": "xgmi", "world_size": 2}


class FakeQuick:
    """The part of ``FlyQuickAllReduce`` the router uses. A call's result is
    this rank's own block (reduce-scatter, the other ranks contributing zeros)
    or block ``r`` equal to the input plus ``r`` (all-gather), so a wrong block
    offset or unpadding shows in the values."""

    def __init__(self, world_size=2, rank=0, link="xgmi", fail=None):
        self.world_size, self.rank, self.link = world_size, rank, link
        self.device = torch.device("cpu")
        self.group = object()
        self.fail = fail
        self.compiled = []
        self.launched = []
        self.real_rows = None  # rows of a block that are not padding

    def compile_sp(self, full, shard):
        self.compiled.append((full.numel() * 2, shard.numel() * 2))
        if self.fail is not None:
            raise self.fail

    def _check(self, full, shard):
        w = self.world_size
        assert full.is_contiguous() and shard.is_contiguous()
        assert full.data_ptr() % 16 == 0 and shard.data_ptr() % 16 == 0
        assert full.numel() == w * shard.numel()
        assert full.numel() * 2 % sp.TILE_BYTES == 0

    def _padding_is_zero(self, blocks):
        """*blocks* is ``[blocks, rows, H]``."""
        if self.real_rows is not None and blocks.device.type != "meta":
            assert not blocks[:, self.real_rows :].any()

    def reduce_scatter(self, inp, out):
        self._check(inp, out)
        self.launched.append(("rs", tuple(inp.shape)))
        if inp.device.type == "meta":
            return
        blocks = inp.view(self.world_size, *out.shape)
        self._padding_is_zero(blocks)
        out.copy_(blocks[self.rank])

    def all_gather(self, inp, out):
        self._check(out, inp)
        self.launched.append(("ag", tuple(inp.shape)))
        if inp.device.type == "meta":
            return
        self._padding_is_zero(inp.unsqueeze(0))
        for r, block in enumerate(out.view(self.world_size, *inp.shape)):
            block.copy_(inp + r)


def _router(world_size=2, rank=0, window=None, **kw):
    quick = FakeQuick(world_size, rank, **kw)
    window = policy.Window(0, None) if window is None else window
    return sp.SequenceParallelCollectives(quick, window), quick


def _meta(*shape, dtype=BF16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _served(router, quick, op, x):
    before = len(quick.launched)
    result = getattr(router, op)(x)
    assert (result is not None) == (len(quick.launched) == before + 1)
    return result is not None


@pytest.fixture(autouse=True)
def peers(monkeypatch):
    """The other ranks of the factory's vote: by default they send this rank's
    vote; set ``peers.others`` to what they send instead. ``peers.calls``
    lists the groups the votes ran over and ``peers.votes`` what this rank sent."""
    state = SimpleNamespace(calls=[], votes=[], others="same")

    def all_gather_object(out, obj, group=None):
        state.calls.append(group)
        state.votes.append(obj)
        other = obj if state.others == "same" else state.others
        out[:] = [obj] + [other] * (len(out) - 1)

    monkeypatch.setattr(sp.dist, "all_gather_object", all_gather_object)
    return state


@pytest.fixture(autouse=True)
def poison_uninitialized_memory():
    """``empty`` returns NaN, so a padding row the router forgot to zero shows."""
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def test_tile_is_32_kib():
    assert sp.TILE_BYTES == 32768


@pytest.mark.parametrize(
    ("world_size", "hidden", "unit"),
    [
        (2, 2560, 16),
        (4, 2560, 8),
        (8, 2560, 4),
        (2, 2048, 4),
        (2, 4096, 2),
        (2, 8192, 1),
        (2, 2568, 1024),  # H % 8 == 0 but H % 16 != 0
        (2, 1, 8192),
    ],
)
def test_block_row_unit(world_size, hidden, unit):
    assert sp.block_row_unit(world_size, hidden) == unit


def test_policy_has_the_one_measured_row(monkeypatch):
    assert policy.sp_int4_window(**GFX950_TP2) == policy.Window(8 << 20, None)
    # no environment override: the all-reduce policy's variables do not move it
    monkeypatch.setenv("AITER_FLY_AR_MESH_MIN_BYTES", "0")
    monkeypatch.setenv("AITER_FLY_AR_MESH_MAX_BYTES", "1")
    assert policy.sp_int4_window(**GFX950_TP2) == policy.Window(8 << 20, None)
    for other in (
        {"arch": "gfx950", "link": "xgmi", "world_size": 4},
        {"arch": "gfx950", "link": "xgmi", "world_size": 8},
        {"arch": "gfx950", "link": "pcie", "world_size": 2},
        {"arch": "gfx942", "link": "xgmi", "world_size": 2},
        {"arch": None, "link": "xgmi", "world_size": 2},
    ):
        assert policy.sp_int4_window(**other) is None


# (T, H, served): hand-derived. The window counts the bytes sent, T * H * 2, not
# the padding. H=2560 (5120 B per row), TP2: 8 MiB is 1638.4 rows, so T=1638 (8,386,560 B)
# is short and T=1640 (8,396,800 B; 820 rows per rank, padded to 832) is served.
# H=2048: T=2048 is exactly 8 MiB, T=2046 is 8,380,416 B. H=2568 has a 1024-row unit:
# T=2 would pad one row per rank to 10.5 MB, which must not make it a served payload.
@pytest.mark.parametrize(
    ("rows", "hidden", "served"),
    [
        (1638, 2560, False),
        (1640, 2560, True),
        (1634, 2560, False),  # its padded payload, 8,519,680 B, is above 8 MiB
        (1633, 2560, False),  # odd: no equal blocks
        (16385, 2560, False),  # odd, and its floor would be inside the window
        (1000, 2560, False),
        (16384, 2560, True),
        (2048, 2048, True),
        (2046, 2048, False),
        (2040, 2048, False),
        (2, 2568, False),
        (3302, 2568, True),
    ],
)
def test_reduce_scatter_window(rows, hidden, served):
    window = policy.sp_int4_window(**GFX950_TP2)
    router, quick = _router(window=window)
    assert _served(router, quick, "reduce_scatter", _meta(rows, hidden)) == served


@pytest.mark.parametrize(
    ("rows", "hidden", "served"),
    [
        (819, 2560, False),  # 8,386,560 B; padded to 832 rows it would be above 8 MiB
        (820, 2560, True),
        (1024, 2048, True),
        (1020, 2048, False),
        (1, 2568, False),
    ],
)
def test_all_gather_window(rows, hidden, served):
    router, quick = _router(window=policy.sp_int4_window(**GFX950_TP2))
    assert _served(router, quick, "all_gather", _meta(rows, hidden)) == served


@pytest.mark.parametrize("op", ["reduce_scatter", "all_gather"])
def test_nothing_is_served_past_the_4_gib_window(op):
    router, quick = _router(window=policy.sp_int4_window(**GFX950_TP2))
    # 5.12 GB (reduce-scatter) and 10.24 GB (all-gather) of full tensor are
    # refused, 512 MB and 1.02 GB are served
    assert not _served(router, quick, op, _meta(1_000_000, 2560))
    assert _served(router, quick, op, _meta(100_000, 2560))


def test_the_4_gib_cap_counts_the_padded_bytes():
    router, quick = _router(window=policy.sp_int4_window(**GFX950_TP2))
    # At hidden 2561 a block is padded to a multiple of 8192 rows: 418,000 rows
    # (4,281,992,000 B sent, inside the cap) become 425,984 rows (4,363,780,096 B
    # moved, above it). 417,792 rows are 51 * 8192: no padding, 4,279,861,248 B.
    assert not _served(router, quick, "reduce_scatter", _meta(2 * 418_000, 2561))
    assert not _served(router, quick, "all_gather", _meta(418_000, 2561))
    assert _served(router, quick, "reduce_scatter", _meta(2 * 417_792, 2561))
    assert _served(router, quick, "all_gather", _meta(417_792, 2561))


@pytest.mark.parametrize("op", ["reduce_scatter", "all_gather"])
@pytest.mark.parametrize(
    "x",
    [
        _meta(16384, 2560, dtype=torch.float16),
        _meta(16384, 2560, dtype=torch.float32),
        _meta(8, 2048, 2560),
        _meta(16384 * 2560),
    ],
    ids=["fp16", "fp32", "3d", "1d"],
)
def test_other_dtypes_and_ranks_are_not_served(op, x):
    router, quick = _router(window=policy.sp_int4_window(**GFX950_TP2))
    assert not _served(router, quick, op, x)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("rows", [5, 16, 17])  # padded, whole unit, padded again
def test_reduce_scatter_returns_this_ranks_unpadded_block(rank, rows):
    router, quick = _router(rank=rank)
    quick.real_rows = rows
    x = torch.arange(2 * rows * 2560).reshape(2 * rows, 2560).to(BF16)
    out = router.reduce_scatter(x)
    assert out.shape == (rows, 2560) and out.dtype == BF16 and out.is_contiguous()
    assert torch.equal(out, x[rank * rows : (rank + 1) * rows])


@pytest.mark.parametrize("rows", [5, 16, 17])
def test_all_gather_returns_every_ranks_unpadded_block_in_order(rows):
    router, quick = _router(world_size=4, rank=1)
    quick.real_rows = rows
    x = torch.arange(rows * 2568).reshape(rows, 2568).to(BF16) % 100
    out = router.all_gather(x)
    assert out.shape == (4 * rows, 2568) and out.is_contiguous()
    assert torch.equal(out, torch.cat([x + r for r in range(4)]))


@pytest.mark.parametrize("op", ["reduce_scatter", "all_gather"])
def test_misaligned_input_is_copied_not_refused(op):
    router, _ = _router()
    rows = 16  # a whole unit: no padding, so the input itself would reach the engine
    shape = (2 * rows if op == "reduce_scatter" else rows, 2560)
    base = torch.zeros(shape[0] * shape[1] + 1, dtype=BF16)
    x = base[1:].view(shape)
    assert x.data_ptr() % 16 != 0
    assert getattr(router, op)(x) is not None  # FakeQuick asserts the alignment


def test_non_contiguous_input_is_served():
    router, _ = _router(rank=1)
    rows = 16
    x = torch.arange(2560 * 2 * rows).reshape(2560, 2 * rows).to(BF16).t()
    assert not x.is_contiguous()
    out = router.reduce_scatter(x)
    assert torch.equal(out, x[rows : 2 * rows])


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(policy, "detect_arch", lambda: "gfx950")


def test_factory_compiles_one_tile_and_returns_the_router(gfx950, peers):
    quick = FakeQuick()
    router = sp.create_sp_collectives(quick)
    assert isinstance(router, sp.SequenceParallelCollectives)
    assert quick.compiled == [(sp.TILE_BYTES, sp.TILE_BYTES // 2)]
    assert peers.calls == [quick.group]


@pytest.mark.parametrize(
    ("arch", "link", "world_size"),
    [("gfx942", "xgmi", 2), ("gfx950", "pcie", 2), ("gfx950", "xgmi", 4)],
)
def test_factory_without_a_window_returns_none_after_the_vote_without_compiling(
    monkeypatch, peers, arch, link, world_size
):
    monkeypatch.setattr(policy, "detect_arch", lambda: arch)
    quick = FakeQuick(world_size=world_size, link=link)
    assert sp.create_sp_collectives(quick) is None
    assert quick.compiled == []
    # a rank without a window still votes: its peers would wait for it
    assert peers.calls == [quick.group]


@pytest.mark.parametrize("others", [None, (1 << 30, None)])
def test_factory_returns_none_when_a_peer_has_another_window(gfx950, peers, others):
    peers.others = others
    quick = FakeQuick()
    assert sp.create_sp_collectives(quick) is None
    assert quick.compiled == []
    assert peers.calls == [quick.group]


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
def test_the_vote_is_plain_data(monkeypatch, peers, arch):
    # A peer unpickles every vote: a Window would import FlyDSL on a rank that has
    # none, so the vote must not name a class of aiter, with a window or without.
    monkeypatch.setattr(policy, "detect_arch", lambda: arch)
    sp.create_sp_collectives(FakeQuick())
    (vote,) = peers.votes
    assert vote == ((8 << 20, None) if arch == "gfx950" else None)
    assert b"aiter" not in pickle.dumps(vote)


def test_factory_returns_none_when_the_kernels_cannot_run(gfx950, monkeypatch):
    warnings = []
    monkeypatch.setattr(sp.logger, "warning", lambda *a: warnings.append(a))
    quick = FakeQuick(fail=RuntimeError("single-phase preflight failed"))
    assert sp.create_sp_collectives(quick) is None
    assert len(quick.compiled) == 1 and len(warnings) == 1


def test_factory_returns_none_without_flydsl_after_the_vote(monkeypatch, peers):
    monkeypatch.setattr(sp, "fly_policy", None)
    quick = FakeQuick()
    assert sp.create_sp_collectives(quick) is None
    assert quick.compiled == []
    assert peers.calls == [quick.group]
