#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routing tests for ``aiter.top_k_per_row_decode``.

The op picks between the FlyDSL tiered decode kernel and the HIP one-block kernel.
What is worth pinning down is that the choice is made from host values only -- a
gate that touched ``seqLens`` would sync the device every decode step -- and that
every rejection lands on HIP rather than raising.

The gate tests patch ``_FLYDSL_TOPK_DECODE_GATES`` instead of setting env, because
the env overrides are read once at import. They need no GPU. The correctness tests
do, and they check that a shape which flips the branch still returns the same set of
indices as ``torch.topk``, including when the caller hands over a buffer padded well
past the real sequence length -- which is what a serving stack does, since it sizes
its score buffer to the model's max context rather than the request's.
"""

from __future__ import annotations

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import topk as topk_mod

ARCH = "gfx950"
MIN_WIDTH, MAX_ROWS = 32768, 16
K = 512
KS = frozenset({512, 2048})


@pytest.fixture
def gates(monkeypatch):
    """Pin the gate table so the assertions do not move with the host's arch.

    These are deliberately generic values that exercise the gate *machinery*, not
    the shipped gfx950 thresholds -- those are asserted directly in
    test_shipped_gfx950_gate below, which reads the real table.
    """
    monkeypatch.setattr(
        topk_mod,
        "_FLYDSL_TOPK_DECODE_GATES",
        {ARCH: topk_mod._DecodeGate(MIN_WIDTH, MAX_ROWS, KS)},
    )
    monkeypatch.setattr(topk_mod, "_FLYDSL_TOPK_DECODE_ARCHS", frozenset({ARCH}))
    monkeypatch.setattr(topk_mod, "_FLYDSL_TOPK_DECODE_DISABLED", False)
    monkeypatch.setattr(topk_mod, "get_gfx", lambda: ARCH)
    monkeypatch.setattr(topk_mod, "is_flydsl_available", lambda: True)


class FakeLogits:
    """Stands in for a logits tensor so the gate tests need no device."""

    def __init__(
        self, rows: int, width: int, stride0: int | None = None, ndim: int = 2
    ):
        self.ndim = ndim
        self.shape = (rows, width)
        self._stride0 = width if stride0 is None else stride0

    def stride(self, dim: int) -> int:
        return self._stride0 if dim == 0 else 1


def routed(
    rows=1,
    width=MIN_WIDTH,
    k=K,
    next_n=1,
    stride0=None,
    stride1=1,
    ndim=2,
    arg_stride0=None,
):
    """Ask the gate about one call. ``stride0`` shapes the tensor, ``arg_stride0``
    is what the caller claims -- separately, so a mismatch can be expressed."""
    logits = FakeLogits(rows, width, stride0, ndim)
    s0 = logits.stride(0) if arg_stride0 is None else arg_stride0
    return topk_mod._should_use_flydsl_decode(logits, next_n, rows, s0, stride1, k)


@pytest.mark.parametrize(
    "width, expected",
    [(MIN_WIDTH - 1, False), (MIN_WIDTH, True), (MIN_WIDTH * 4, True)],
)
def test_width_gate(gates, width, expected):
    assert routed(width=width) is expected


@pytest.mark.parametrize(
    "rows, expected", [(1, True), (MAX_ROWS, True), (MAX_ROWS + 1, False)]
)
def test_rows_gate(gates, rows, expected):
    assert routed(rows=rows) is expected


@pytest.mark.parametrize(
    "k, expected", [(512, True), (2048, True), (256, False), (1024, False)]
)
def test_k_gate(gates, k, expected):
    assert routed(k=k) is expected


def test_excluded_rows_fall_back(gates, monkeypatch):
    """A row inside [1, max_rows] but listed as excluded must route HIP."""
    monkeypatch.setattr(
        topk_mod,
        "_FLYDSL_TOPK_DECODE_GATES",
        {ARCH: topk_mod._DecodeGate(MIN_WIDTH, MAX_ROWS, KS, frozenset({2}))},
    )
    assert routed(rows=1) is True
    assert routed(rows=2) is False  # carved out
    assert routed(rows=4) is True


def test_shipped_gfx950_gate():
    """Pin the real gfx950 window to the SILOTIGER-699 conclusion so a silent edit
    to the table trips here. This reads the shipped constants, not the fixture."""
    gate = topk_mod._FLYDSL_TOPK_DECODE_GATES["gfx950"]
    assert gate.min_width == 131072
    assert gate.max_rows == 16
    assert gate.ks == frozenset({2048})
    assert gate.excluded_rows == frozenset({2})


def test_shipped_gfx942_gate():
    """Same pin for gfx942, set from the MI300X sweep. Narrower in rows than gfx950
    because that arch runs the frozen kernel config, wider in k because all four
    AOT-precompiled values measured alike."""
    gate = topk_mod._FLYDSL_TOPK_DECODE_GATES["gfx942"]
    assert gate.min_width == 131072
    assert gate.max_rows == 8
    assert gate.ks == frozenset({256, 512, 1024, 2048})
    assert gate.excluded_rows == frozenset()


def test_unlisted_arch_falls_back(gates, monkeypatch):
    monkeypatch.setattr(topk_mod, "get_gfx", lambda: "gfx1100")
    assert routed() is False


def test_arch_override_cannot_enable_an_unmeasured_arch(gates, monkeypatch):
    """Listing an arch in the env narrows the table; it never adds a row to it."""
    monkeypatch.setattr(
        topk_mod, "_FLYDSL_TOPK_DECODE_ARCHS", frozenset({ARCH, "gfx1100"})
    )
    monkeypatch.setattr(topk_mod, "get_gfx", lambda: "gfx1100")
    assert routed() is False


def test_disable_flag_forces_hip(gates, monkeypatch):
    monkeypatch.setattr(topk_mod, "_FLYDSL_TOPK_DECODE_DISABLED", True)
    assert routed() is False


def test_missing_flydsl_falls_back(gates, monkeypatch):
    monkeypatch.setattr(topk_mod, "is_flydsl_available", lambda: False)
    assert routed() is False


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"stride1": 2}, id="stride1_not_contiguous"),
        pytest.param(
            {"arg_stride0": MIN_WIDTH + 7}, id="stride0_disagrees_with_tensor"
        ),
        pytest.param({"arg_stride0": MIN_WIDTH - 7}, id="stride0_below_tensor"),
        pytest.param({"next_n": 0}, id="next_n_below_one"),
        pytest.param({"ndim": 3}, id="logits_not_2d"),
    ],
)
def test_contract_violations_fall_back(gates, kwargs):
    """FlyDSL raises on these; the HIP kernel takes them, so route there quietly."""
    assert routed(**kwargs) is False


def test_matching_stride0_passes(gates):
    """The same value that just failed as a mismatch must pass when it is honest."""
    assert routed(stride0=MIN_WIDTH + 7, arg_stride0=MIN_WIDTH + 7) is True


# --- GPU ---------------------------------------------------------------------

pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm device"
)


def make_inputs(rows: int, width: int, seq_len: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    logits = torch.randn((rows, width), dtype=torch.float32, device="cuda")
    if seq_len < width:
        # A real caller fills the tail with -inf; poison it instead so a kernel
        # that ignores seqLens fails loudly rather than accidentally passing.
        logits[:, seq_len:] = 1e30
    seq_lens = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    return logits, seq_lens, indices


@pytestmark_gpu
@pytest.mark.parametrize(
    "rows, width, seq_len, k",
    [
        (1, 32768, 32768, 512),
        (1, 32767, 32767, 512),  # one short of the gate -> HIP
        (16, 65536, 65536, 512),
        (17, 65536, 65536, 512),  # one over the row cap -> HIP
        (1, 65536, 65536, 2048),
        (1, 65536, 65536, 1024),  # outside every arch's window -> HIP
        (4, 131072, 70000, 512),  # padded buffer, poisoned tail
        (1, 8192, 8192, 512),
    ],
)
def test_both_branches_match_torch_topk(rows, width, seq_len, k):
    logits, seq_lens, indices = make_inputs(rows, width, seq_len, k)
    indices.fill_(-1)
    topk_mod.top_k_per_row_decode(
        logits, 1, seq_lens, indices, rows, logits.stride(0), logits.stride(1), k
    )
    torch.cuda.synchronize()

    assert bool(((indices >= 0) & (indices < seq_len)).all())
    expected = torch.topk(logits[:, :seq_len], k, dim=1).indices
    assert torch.equal(indices.long().sort(dim=1).values, expected.sort(dim=1).values)


@pytestmark_gpu
def test_hip_and_flydsl_agree_on_a_gated_shape():
    """The two kernels must return the same index set for an identical in-gate
    call -- not merely each match torch. Forces each branch directly rather than
    letting the gate pick, so a divergence between them cannot hide behind routing.
    """
    if get_gfx_runtime() not in topk_mod._FLYDSL_TOPK_DECODE_GATES:
        pytest.skip("no FlyDSL gates for this arch")
    from aiter.ops.flydsl.topk_per_row_decode import flydsl_top_k_per_row_decode

    rows, width, seq_len, k = 4, 131072, 70000, 2048
    logits, seq_lens, indices = make_inputs(rows, width, seq_len, k)
    s0, s1 = logits.stride(0), logits.stride(1)

    hip_idx = torch.empty_like(indices).fill_(-1)
    topk_mod._top_k_per_row_decode(logits, 1, seq_lens, hip_idx, rows, s0, s1, k, None)
    fly_idx = torch.empty_like(indices).fill_(-1)
    flydsl_top_k_per_row_decode(
        logits, 1, seq_lens, fly_idx, rows, s0, s1, k, ordered=False, workspace=None
    )
    torch.cuda.synchronize()

    assert torch.equal(
        hip_idx.long().sort(dim=1).values, fly_idx.long().sort(dim=1).values
    )


@pytestmark_gpu
def test_workspace_cache_reuses_one_buffer_per_shape():
    if get_gfx_runtime() not in topk_mod._FLYDSL_TOPK_DECODE_GATES:
        pytest.skip("no FlyDSL gates for this arch")
    topk_mod.clear_flydsl_topk_decode_workspace_cache()
    device = torch.device("cuda:0")

    first = topk_mod._get_flydsl_topk_workspace(device, 1, 65536)
    again = topk_mod._get_flydsl_topk_workspace(device, 1, 65536)
    assert first is again
    assert first.dtype is torch.int32
    assert first.numel() & (first.numel() - 1) == 0  # bucketed to a power of two

    wider = topk_mod._get_flydsl_topk_workspace(device, 16, 65536)
    assert wider is not first
    assert wider.numel() > first.numel()

    topk_mod.clear_flydsl_topk_decode_workspace_cache()
    assert topk_mod._get_flydsl_topk_workspace(device, 1, 65536) is not first


@pytestmark_gpu
def test_caller_workspace_bypasses_the_cache():
    if get_gfx_runtime() not in topk_mod._FLYDSL_TOPK_DECODE_GATES:
        pytest.skip("no FlyDSL gates for this arch")
    rows, width, k = 1, 65536, 512
    logits, seq_lens, indices = make_inputs(rows, width, width, k)
    from aiter.ops.flydsl.topk_per_row_decode import (
        flydsl_top_k_per_row_decode_workspace_size,
    )

    slots = flydsl_top_k_per_row_decode_workspace_size(rows, width)
    workspace = torch.zeros(slots, dtype=torch.int32, device="cuda")
    topk_mod.clear_flydsl_topk_decode_workspace_cache()

    topk_mod.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        indices,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        workspace,
    )
    torch.cuda.synchronize()

    assert topk_mod._get_flydsl_topk_workspace_keyed.cache_info().currsize == 0
    expected = torch.topk(logits, k, dim=1).indices
    assert torch.equal(indices.long().sort(dim=1).values, expected.sort(dim=1).values)


@pytestmark_gpu
def test_hip_fallback_allocates_no_workspace():
    """A shape that falls back must not leave a FlyDSL buffer cached behind it."""
    topk_mod.clear_flydsl_topk_decode_workspace_cache()
    rows, width, k = 1, 8192, 512
    logits, seq_lens, indices = make_inputs(rows, width, width, k)
    topk_mod.top_k_per_row_decode(
        logits, 1, seq_lens, indices, rows, logits.stride(0), logits.stride(1), k
    )
    torch.cuda.synchronize()
    assert topk_mod._get_flydsl_topk_workspace_keyed.cache_info().currsize == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
