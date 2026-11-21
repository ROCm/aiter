# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""When topk_gating sends a call to the Triton 384-expert sqrtsoftplus top-k.

The dispatch reads only tensor metadata, so CPU tensors stand in for the GPU
ones; the arch is patched, and the two kernels are replaced by recorders.
Standard Tests run op_tests files as scripts, so the file also runs pytest on
itself.
"""

import sys

import pytest
import torch

pytest.importorskip("triton")
topk_ops = pytest.importorskip("aiter.ops.topk")
from aiter.ops.triton.moe.moe_routing import topk_gating as topk_gating_launchers
from aiter.ops.triton.moe.moe_routing.topk_gating import (
    SQRTSOFTPLUS_384_ARCHS,
)

EXPERTS = 384
_GFX = min(SQRTSOFTPLUS_384_ARCHS)


def _args(rows, topk=6, experts=EXPERTS, dtype=torch.bfloat16, bias="fp32"):
    logits = torch.empty((rows, experts), dtype=dtype)
    weights = torch.empty((rows, topk), dtype=torch.float32)
    ids = torch.empty((rows, topk), dtype=torch.int32)
    biases = {
        "none": torch.empty(0, dtype=torch.float32),
        "fp32": torch.zeros(experts, dtype=torch.float32),
        "bf16": torch.zeros(experts, dtype=torch.bfloat16),
    }
    return weights, ids, logits, biases[bias]


def _patch(monkeypatch, gfx):
    """Run as on gfx, from the state before the first topk_gating call."""
    monkeypatch.setattr(topk_ops, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(topk_ops, "_sqrtsoftplus_384_arch", topk_ops._UNSET)
    monkeypatch.setattr(topk_ops, "_sqrtsoftplus_384_launch", None)


def _can_use(args, gfx=_GFX):
    arch = SQRTSOFTPLUS_384_ARCHS.get(gfx)
    return topk_ops._can_use_sqrtsoftplus_384_topk(arch, *args)


def _record_kernels(monkeypatch):
    """Replace both kernels; return the list of the kernels that ran.

    The Triton launcher is replaced where topk_gating binds it on the first
    call, so call _patch first."""
    ran = []
    monkeypatch.setattr(topk_ops, "topk_gating_fwd", lambda *a: ran.append("hip"))
    monkeypatch.setattr(
        topk_gating_launchers,
        "_sqrtsoftplus_384_topk_launch",
        lambda *a: ran.append("triton"),
    )
    if topk_ops._sqrtsoftplus_384_arch is not topk_ops._UNSET:
        monkeypatch.setattr(
            topk_ops, "_sqrtsoftplus_384_launch", lambda *a: ran.append("triton")
        )
    return ran


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_rows(monkeypatch, gfx):
    _patch(monkeypatch, gfx)
    max_rows = SQRTSOFTPLUS_384_ARCHS[gfx].max_rows
    for rows, expected in (
        (0, False),
        (1, True),
        (max_rows, True),
        (max_rows + 1, False),
    ):
        assert _can_use(_args(rows), gfx) is expected, rows


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_each_contract_term(monkeypatch, gfx):
    _patch(monkeypatch, gfx)
    assert _can_use(_args(8), gfx)
    assert _can_use(_args(8, bias="none"))
    assert _can_use(_args(8, topk=1))
    assert _can_use(_args(8, topk=8))
    assert not _can_use(_args(8, dtype=torch.float16))
    assert not _can_use(_args(8, bias="bf16"))
    assert not _can_use(_args(8, topk=9))
    weights, ids, logits, bias = _args(8)
    strided = torch.empty((8, 2 * EXPERTS), dtype=torch.bfloat16)[:, ::2]
    assert not _can_use((weights, ids, strided, bias))
    assert not _can_use((weights, ids, logits.view(1, 8, EXPERTS), bias))
    assert not _can_use((weights[:4], ids, logits, bias))
    assert not _can_use((weights, ids.long(), logits, bias))


def test_an_arch_outside_the_table(monkeypatch):
    assert not _can_use(_args(1), "gfx942")


def test_the_arch_is_looked_up_once(monkeypatch):
    """The first call resolves the entry and the launcher; later calls read
    them from module-level values."""
    calls = []
    gfx = min(SQRTSOFTPLUS_384_ARCHS)
    _patch(monkeypatch, gfx)
    monkeypatch.setattr(topk_ops, "get_gfx_runtime", lambda: calls.append(1) or gfx)
    ran = _record_kernels(monkeypatch)
    for _ in range(3):
        _call(_args(1))
    assert ran == ["triton"] * 3 and len(calls) == 1
    assert topk_ops._sqrtsoftplus_384_arch == SQRTSOFTPLUS_384_ARCHS[gfx]


def _call(args, need_renorm=True, score_func="sqrtsoftplus", shared=0):
    weights, ids, logits, bias = args
    topk_ops.topk_gating(
        weights, ids, logits, bias, need_renorm, 1.5, score_func, shared
    )


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_a_call_in_range_runs_triton(monkeypatch, gfx):
    _patch(monkeypatch, gfx)
    ran = _record_kernels(monkeypatch)
    weights, ids, logits, bias = _args(8)
    _call((weights, ids, logits, bias))
    _call((weights, ids, logits, None))
    _call(_args(8, bias="none"))
    assert ran == ["triton"] * 3


def _strided(rows, cols, dtype):
    return torch.empty((rows, 2 * cols), dtype=dtype)[:, ::2]


# Each case breaks one term of the entry's guard; the call must reach HIP.
_OUTSIDE = {
    "no_renorm": lambda: {"args": _args(8), "need_renorm": False},
    "sigmoid": lambda: {"args": _args(8), "score_func": "sigmoid"},
    "softmax": lambda: {"args": _args(8), "score_func": "softmax"},
    "shared": lambda: {"args": _args(8), "shared": 1},
    "no_rows": lambda: {"args": _args(0)},
    "experts_256": lambda: {"args": _args(8, experts=256, bias="none")},
    "experts_512": lambda: {"args": _args(8, experts=512, bias="none")},
    "logits_fp16": lambda: {"args": _args(8, dtype=torch.float16)},
    "bias_bf16": lambda: {"args": _args(8, bias="bf16")},
    "bias_383": lambda: {
        "args": _args(8)[:3] + (torch.zeros(383, dtype=torch.float32),)
    },
    "bias_192": lambda: {
        "args": _args(8)[:3] + (torch.zeros(192, dtype=torch.float32),)
    },
    "bias_stride_2": lambda: {
        "args": _args(8)[:3] + (torch.zeros(768, dtype=torch.float32)[::2],)
    },
    "topk_0": lambda: {"args": _args(8, topk=0)},
    "topk_9": lambda: {"args": _args(8, topk=9)},
    "logits_col_stride": lambda: {
        "args": _args(8)[:2] + (_strided(8, EXPERTS, torch.bfloat16), _args(8)[3])
    },
    # [8, 1, 384] with unit stride on dim 1: every other term holds.
    "logits_3d": lambda: {
        "args": _args(8)[:2]
        + (
            torch.empty((8, EXPERTS), dtype=torch.bfloat16).as_strided(
                (8, 1, EXPERTS), (EXPERTS, 1, 1)
            ),
            _args(8)[3],
        )
    },
    "weights_col_stride": lambda: {
        "args": (_strided(8, 6, torch.float32),) + _args(8)[1:]
    },
    "ids_col_stride": lambda: {
        "args": _args(8)[:1] + (_strided(8, 6, torch.int32),) + _args(8)[2:]
    },
    "weights_fewer_rows": lambda: {
        "args": (torch.empty((4, 6), dtype=torch.float32),) + _args(8)[1:]
    },
    "ids_fewer_rows": lambda: {
        "args": _args(8)[:1] + (torch.empty((4, 6), dtype=torch.int32),) + _args(8)[2:]
    },
    "weights_rows_overlap": lambda: {
        "args": (torch.empty((1, 6), dtype=torch.float32).expand(8, 6),) + _args(8)[1:]
    },
    "ids_rows_overlap": lambda: {
        "args": _args(8)[:1]
        + (torch.empty((1, 6), dtype=torch.int32).expand(8, 6),)
        + _args(8)[2:]
    },
    "outputs_fewer_rows": lambda: {
        "args": (
            torch.empty((4, 6), dtype=torch.float32),
            torch.empty((4, 6), dtype=torch.int32),
        )
        + _args(8)[2:]
    },
    "weights_fp16": lambda: {
        "args": (torch.empty((8, 6), dtype=torch.float16),) + _args(8)[1:]
    },
    "ids_1d": lambda: {
        "args": _args(8)[:1] + (torch.empty(8, dtype=torch.int32),) + _args(8)[2:]
    },
    "ids_int64": lambda: {
        "args": _args(8)[:1] + (torch.empty((8, 6), dtype=torch.int64),) + _args(8)[2:]
    },
    "ids_wider": lambda: {
        "args": _args(8)[:1] + (torch.empty((8, 8), dtype=torch.int32),) + _args(8)[2:]
    },
}


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
@pytest.mark.parametrize("case", sorted(_OUTSIDE))
def test_a_call_outside_the_limits_runs_hip(monkeypatch, gfx, case):
    _patch(monkeypatch, gfx)
    ran = _record_kernels(monkeypatch)
    _call(**_OUTSIDE[case]())
    assert ran == ["hip"], case


def _no_check(*args, **kwargs):
    pytest.fail("the call reached _can_use_sqrtsoftplus_384_topk")


# Calls that the cheap tests in topk_gating reject before the tensor checks.
_CANNOT_MATCH = (
    "no_renorm",
    "sigmoid",
    "softmax",
    "shared",
    "experts_256",
    "experts_512",
)


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
@pytest.mark.parametrize("case", _CANNOT_MATCH)
def test_a_call_that_cannot_match_skips_the_tensor_checks(monkeypatch, gfx, case):
    _patch(monkeypatch, gfx)
    monkeypatch.setattr(topk_ops, "_sqrtsoftplus_384_arch", SQRTSOFTPLUS_384_ARCHS[gfx])
    monkeypatch.setattr(topk_ops, "_can_use_sqrtsoftplus_384_topk", _no_check)
    ran = _record_kernels(monkeypatch)
    _call(**_OUTSIDE[case]())
    assert ran == ["hip"], case


@pytest.mark.parametrize("first", ["experts_256", "softmax"])
def test_an_arch_outside_the_table_stops_at_the_first_test(monkeypatch, first):
    """The first call resolves the entry whatever its shape, so a later
    DSV4.1-shaped call off the table reads only the cached None."""
    _patch(monkeypatch, "gfx942")
    calls = []
    monkeypatch.setattr(
        topk_ops, "get_gfx_runtime", lambda: calls.append(1) or "gfx942"
    )
    monkeypatch.setattr(topk_ops, "_can_use_sqrtsoftplus_384_topk", _no_check)
    ran = _record_kernels(monkeypatch)
    _call(**_OUTSIDE[first]())
    assert topk_ops._sqrtsoftplus_384_arch is None and calls == [1]
    _call(_args(8))
    assert ran == ["hip", "hip"] and calls == [1]


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_a_first_call_of_another_shape_resolves_the_entry(monkeypatch, gfx):
    _patch(monkeypatch, gfx)
    ran = _record_kernels(monkeypatch)
    _call(**_OUTSIDE["experts_256"]())
    assert topk_ops._sqrtsoftplus_384_arch == SQRTSOFTPLUS_384_ARCHS[gfx]
    _call(_args(8))
    assert ran == ["hip", "triton"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
