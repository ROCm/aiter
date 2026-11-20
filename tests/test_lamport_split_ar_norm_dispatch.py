# SPDX-License-Identifier: MIT
"""The size dispatch of fused_allreduce_rmsnorm to the Lamport two-shot
all-reduce + split norm (gfx1250, TP4)."""

from types import SimpleNamespace

import pytest
import torch

from aiter.dist.device_communicators import communicator_cuda
from aiter.dist.device_communicators.communicator_cuda import (
    _use_lamport_split_ar_norm,
)

MIB = 1 << 20
HIDDEN = 6144


def _comm(world_size=4, cap=8 * MIB):
    def should_lamport_ar(inp, two_shot=False):
        nbytes = inp.numel() * inp.element_size()
        return inp.dtype == torch.bfloat16 and 0 < nbytes <= cap * (
            world_size if two_shot else 1
        )

    ca_comm = SimpleNamespace(
        _lamport_max_bytes=cap, should_lamport_ar=should_lamport_ar
    )
    return SimpleNamespace(ca_comm=ca_comm, world_size=world_size)


def _args(rows, hidden=HIDDEN, dtype=torch.bfloat16):
    x = torch.empty((rows, hidden), dtype=dtype)
    return x, torch.empty_like(x), torch.empty(hidden, dtype=dtype), hidden


@pytest.mark.parametrize(
    ("rows", "want"),
    [
        (24, False),  # decode: the Gluon / Lamport fused one-shot
        (170, False),  # 2 MiB and under
        (171, True),  # (2, 4] MiB
        (341, True),
        (342, False),  # (4, 6] MiB: the fused one-shot wins
        (512, False),
        (513, True),  # (6, 8] MiB
        (682, True),
        (683, True),  # above the one-shot cap, up to 4 x the cap
        (2730, True),
        (2731, False),  # above the two-shot's range
    ],
)
def test_hidden_6144_ranges(rows, want):
    assert _use_lamport_split_ar_norm(_comm(), *_args(rows)) is want


def test_other_hidden_only_above_the_cap():
    # 4096 x 2 bytes: 384 rows is 3 MiB, 1100 rows is 8.6 MiB.
    assert not _use_lamport_split_ar_norm(_comm(), *_args(384, hidden=4096))
    assert _use_lamport_split_ar_norm(_comm(), *_args(1100, hidden=4096))


def test_off_switch_world_size_and_operands(monkeypatch):
    assert not _use_lamport_split_ar_norm(_comm(world_size=2), *_args(1024))
    assert not _use_lamport_split_ar_norm(
        SimpleNamespace(ca_comm=None, world_size=4), *_args(1024)
    )
    x, res, w, n = _args(1024)
    assert not _use_lamport_split_ar_norm(_comm(), x, res[:512], w, n)
    assert not _use_lamport_split_ar_norm(_comm(), x, res.float(), w, n)
    assert not _use_lamport_split_ar_norm(_comm(), x, res, w, n + 256)
    assert not _use_lamport_split_ar_norm(_comm(), x[None], res[None], w, n)
    monkeypatch.setenv("AITER_LAMPORT_SPLIT_AR_NORM", "0")
    assert not _use_lamport_split_ar_norm(_comm(), *_args(1024))


def test_ranges_stay_under_the_cap():
    for ranges in communicator_cuda._LAMPORT_SPLIT_AR_NORM_RANGES.values():
        assert all(0 < low < high <= 8 * MIB for low, high in ranges)
