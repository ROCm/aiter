#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routing tests for ``aiter.top_k_per_row_decode``.

The op picks between the FlyDSL tiered decode kernel and the HIP one-block kernel.
The choice is made from host values only -- a gate that touched ``seqLens`` would
sync the device every decode step -- and every rejection has to land on HIP rather
than raising.

The cases below straddle each gate boundary and check the observable result: a
shape that flips the branch still returns the same set of indices as
``torch.topk``, including when the caller hands over a buffer padded well past the
real sequence length, which is what a serving stack does since it sizes its score
buffer to the model's max context rather than the request's.
"""

from __future__ import annotations

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import topk as topk_mod

# AITER_FLYDSL_TOPK_MIN_WIDTH / _MAX_ROWS rewrite the shipped table at import
# time, which is a legitimate thing to do while benchmarking. Pinning the shipped
# values is only meaningful against the shipped table, so stand down when the
# window has deliberately been moved.
gate_is_overridden = pytest.mark.skipif(
    bool(topk_mod._FLYDSL_TOPK_MIN_WIDTH_ENV or topk_mod._FLYDSL_TOPK_MAX_ROWS_ENV),
    reason="decode gate overridden via AITER_FLYDSL_TOPK_MIN_WIDTH/_MAX_ROWS",
)


@gate_is_overridden
def test_shipped_gfx950_gate():
    """Pin the real gfx950 window to the SILOTIGER-699 conclusion so a silent edit
    to the table trips here."""
    gate = topk_mod._FLYDSL_TOPK_DECODE_GATES["gfx950"]
    assert gate.min_width == 131072
    assert gate.max_rows == 16
    assert gate.ks == frozenset({2048})
    assert gate.excluded_rows == frozenset({2})


@gate_is_overridden
def test_shipped_gfx942_gate():
    """Same pin for gfx942, set from the MI300X sweep. Narrower in rows than gfx950
    because that arch runs the frozen kernel config, wider in k because all four
    AOT-precompiled values measured alike."""
    gate = topk_mod._FLYDSL_TOPK_DECODE_GATES["gfx942"]
    assert gate.min_width == 131072
    assert gate.max_rows == 8
    assert gate.ks == frozenset({256, 512, 1024, 2048})
    assert gate.excluded_rows == frozenset()


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


ROUTING_CASES = [
    (1, 32768, 32768, 512),
    (1, 32767, 32767, 512),  # one short of the gate -> HIP
    (16, 65536, 65536, 512),
    (17, 65536, 65536, 512),  # one over the row cap -> HIP
    (1, 65536, 65536, 2048),
    (1, 65536, 65536, 1024),  # outside every arch's window -> HIP
    (4, 131072, 70000, 512),  # padded buffer, poisoned tail
    (1, 8192, 8192, 512),
    # gfx950 admits only k=2048 at width >= 131072, so every case above lands on
    # HIP there and the FlyDSL branch of the arch this PR targets would go
    # untested. These three sit inside both shipped windows.
    (1, 131072, 131072, 2048),  # -> FlyDSL on gfx950 and gfx942
    (1, 131072, 70000, 2048),  # same, with the poisoned tail
    (2, 131072, 131072, 2048),  # gfx950 carves out rows==2 -> HIP; gfx942 FlyDSL
]


@gate_is_overridden
@pytest.mark.parametrize("arch", sorted(topk_mod._FLYDSL_TOPK_DECODE_GATES))
def test_routing_cases_reach_flydsl_on_every_shipped_arch(arch):
    """Every arch with a gate needs at least one case above that the gate admits.

    Nothing else notices when it stops being true: the cases below all return the
    right indices from HIP, so an arch whose window drifts away from the matrix
    keeps a green suite while never running the FlyDSL kernel at all. gfx950 was
    in exactly that state -- its window is k==2048 above width 131072, and no case
    held both at once. This runs on the gate tuple alone, so it fails on the arch
    it describes rather than only on hardware that happens to be present.
    """
    gate = topk_mod._FLYDSL_TOPK_DECODE_GATES[arch]
    admitted = [
        (rows, width, seq_len, k)
        for rows, width, seq_len, k in ROUTING_CASES
        if rows <= gate.max_rows
        and rows not in gate.excluded_rows
        and k in gate.ks
        and width >= gate.min_width
    ]
    assert admitted, f"no routing case lands inside the {arch} window: {gate}"


@pytestmark_gpu
@pytest.mark.parametrize("rows, width, seq_len, k", ROUTING_CASES)
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
@pytest.mark.parametrize("claimed_stride1", [2, 1], ids=["honest", "claims_one"])
def test_column_strided_input_falls_back_instead_of_raising(claimed_stride1):
    """A column-strided buffer sits inside the gate's width window, so the stride1
    screen is the only thing keeping it away from FlyDSL, which raises on it. HIP
    ignores stride1, so the call has to simply come back either way -- including
    when the caller claims 1, which HIP accepts and the gate must not take at its
    word.

    k has to be 2048: it is the only value inside every shipped window, and under
    any other one gfx950 rejects on k before the stride checks are ever reached,
    which would leave the screen this test is about unexercised there."""
    if get_gfx_runtime() not in topk_mod._FLYDSL_TOPK_DECODE_GATES:
        pytest.skip("no FlyDSL gates for this arch")
    rows, width, k = 1, 131072, 2048
    wide = torch.randn((rows, width * 2), dtype=torch.float32, device="cuda")
    logits = wide[:, ::2]
    assert logits.stride(1) == 2 and logits.shape[1] == width
    seq_lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, k), dtype=torch.int32, device="cuda")

    topk_mod.top_k_per_row_decode(
        logits, 1, seq_lens, indices, rows, logits.stride(0), claimed_stride1, k
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
