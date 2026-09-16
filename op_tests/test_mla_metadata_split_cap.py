# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`max_split_per_batch` must actually bound the metadata allocation.

`get_mla_metadata_info_v1` sizes the reduce scratch from a fast_mode estimate
that saturates at `((max_splits - 1) * 2) * tiles_per_batch` and does not grow
with `tile_cnt`. Two things follow:

  * a supplied `max_split_per_batch` can only *tighten* the sizing, and the
    previous `max()` let the uncapped estimate always win, so the cap had no
    effect on the allocation at all;
  * at large batch the fast-mode estimate can sit *below* `tile_cnt +
    per_tile_cap`, and taking the min keeps the smaller. That the smaller is
    still sufficient is established by measurement, not assumed -- see
    `test_mla_metadata_split_cap_fill.py`, where at batch 512 the planner
    writes at most ~510 partials against a bound of ~2040.

Sizes here are derived, never hardcoded: they depend on the CU count, so a
literal from one part fails on another.

These are pure sizing queries -- no kernel is launched and no GPU work is done.
"""

import pytest

import aiter
from aiter import dtypes

NHEAD = 128
MAX_SEQLEN_QO = 4

# Large enough that no cap can constrain the schedule, AND that the saturated
# fast-mode estimate sits below `tile_cnt + max_splits` so the min() picks the
# fast-mode side. Both premises scale with max_splits, which tracks the CU count
# and differs across parts -- 512 satisfies them on gfx950 (max_splits 256) but
# NOT on gfx942 (304), where the capped bound comes out below the no-cap one and
# the invariant below would fail. So derive it instead of hardcoding.
_MAX_SPLITS = aiter.get_mla_decode_fwd_max_splits(
    NHEAD, MAX_SEQLEN_QO, dtypes.fp8, dtypes.fp8
)
_LARGE_BATCH = 4 * _MAX_SPLITS


def _reduce_partial_map_size(batch_size, max_split_per_batch, fast_mode=True):
    """Element count of the reduce_partial_map buffer, the one that dominates.

    aiter's mla_decode_fwd sizes its fp32 `logits` from this, so it is what
    turns a loose bound into gigabytes.
    """
    sizes = aiter.get_mla_metadata_info_v1(
        batch_size,
        MAX_SEQLEN_QO,
        NHEAD,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=fast_mode,
        max_split_per_batch=max_split_per_batch,
    )
    # (work_meta_data, work_indptr, work_info_set, reduce_indptr,
    #  reduce_final_map, reduce_partial_map)
    return sizes[5][0]


@pytest.mark.parametrize("batch_size", [1, 8, 64])
def test_a_cap_never_enlarges_the_allocation(batch_size):
    """Where the cap is active it must not grow the sizing.

    Excludes batch 512 deliberately: there `tile_cnt + max_splits` exceeds the
    saturated fast-mode estimate, so the cap-aware path is legitimately larger
    than the no-cap path. See test_a_non_constraining_cap_does_not_change_the_size.
    """
    uncapped = _reduce_partial_map_size(batch_size, max_split_per_batch=-1)
    capped = _reduce_partial_map_size(batch_size, max_split_per_batch=1)
    assert capped <= uncapped, (
        f"batch_size={batch_size}: capping splits grew the allocation "
        f"{uncapped} -> {capped}"
    )


@pytest.mark.parametrize("batch_size", [1, 8, 64])
def test_a_tight_cap_actually_shrinks_the_allocation(batch_size):
    """The regression this guards. Under max() the cap is inert and these are
    equal; under min() the capped sizing is far smaller.

    Measured on gfx950, reduce_partial_map entries at nhead=128, qo_len=4:

        batch   uncapped   cap=1   cap=256
            1       1024       5       260
            8       1052      40       288
           64       1276     320       512
    """
    uncapped = _reduce_partial_map_size(batch_size, max_split_per_batch=-1)
    capped = _reduce_partial_map_size(batch_size, max_split_per_batch=1)
    assert capped < uncapped, (
        f"batch_size={batch_size}: max_split_per_batch had no effect on the "
        f"allocation ({uncapped} both ways) -- the cap is being ignored"
    )


def test_a_non_constraining_cap_does_not_change_the_size():
    """A cap that cannot constrain the schedule must not change the reservation.

    At large batch `per_tile_cap` is `min(max_splits, cap * batch)` = max_splits
    for every cap >= 1, so no cap constrains anything and the sizing must match
    the no-cap path. Applying the sum bound only inside the cap branch produced
    exactly that split -- the same cluster count getting two different
    reservations depending on whether a dummy cap was passed. The batch used
    here is derived (4 * max_splits), so no literal is quoted: it differs per
    part, which is the bug the derivation fixes.
    """
    no_cap = _reduce_partial_map_size(_LARGE_BATCH, max_split_per_batch=-1)
    for cap in (1, 4, 32, 256):
        assert _reduce_partial_map_size(_LARGE_BATCH, cap) == no_cap, (
            f"cap={cap} changed the size at batch {_LARGE_BATCH} "
            f"(no-cap {no_cap}) even though it cannot constrain the schedule"
        )


def test_a_larger_cap_is_never_smaller_than_a_tighter_one():
    """Monotonicity: relaxing the cap cannot shrink the bound."""
    sizes = [_reduce_partial_map_size(64, cap) for cap in (1, 4, 16, 64)]
    assert sizes == sorted(sizes), f"non-monotonic in the cap: {sizes}"


def test_no_cap_is_unchanged():
    """-1 means 'no cap' and must not enter the branch, so the sizing has to
    match the historical value exactly.

    Deliberately asserts -1 only, NOT 0. The two are equivalent to this sizing
    helper -- both fail the `> 0` guard -- but they are not equivalent to the
    planner, so treating them as interchangeable would advertise a contract the
    library does not honour: v1.2 computes
    `num_splits = min(num_clusters, max_split_per_batch * num_batches)` and
    `auto_split = (max_split_per_batch < 0)`, so 0 gives num_splits == 0 with
    auto_split false, and mla_v12_effective_splits returns that 0 unclamped (the
    `max(1, ...)` applies only on the auto path). 0 therefore means ZERO splits
    to the fill, not 'unlimited'. Use -1.
    """
    uncapped = _reduce_partial_map_size(64, -1)
    assert _reduce_partial_map_size(64, -2) == uncapped


@pytest.mark.parametrize("batch_size", [1, 8, 64])
def test_non_fast_mode_ignores_the_cap(batch_size):
    """fast_mode=False must be byte-identical with and without a cap.

    That path dispatches to get_mla_metadata_v1_1 (csrc/kernels/mla/metadata.cu),
    whose device entry point takes no max_split_per_batch -- compare
    get_mla_metadata_v1_0_device directly above it, which does. The planner is
    therefore uncapped there, and shrinking the sizing would undersize
    reduce_partial_map, which faults the GPU rather than raising.
    """
    uncapped = _reduce_partial_map_size(batch_size, -1, fast_mode=False)
    for cap in (1, 4, 256):
        assert _reduce_partial_map_size(batch_size, cap, fast_mode=False) == uncapped, (
            f"batch_size={batch_size} cap={cap}: the cap changed the non-fast_mode "
            f"sizing, but get_mla_metadata_v1_1 does not honour a cap -- this "
            f"undersizes reduce_partial_map and faults the GPU"
        )


def _check_native_gate_without_pytest():
    """main() has no monkeypatch fixture; drive _NATIVE_GATE_CASES by hand.

    CI runs this file with python3, so anything left pytest-only is never
    executed there -- which is how the gfx1250 qlen check got missed once.
    """
    import os as _os

    import aiter.ops.attention as attention_ops

    saved = (
        attention_ops.get_gfx,
        attention_ops.is_experimental_enabled,
        _os.environ.get("AITER_MLA_DECODE_PS1_FLYDSL"),
    )
    try:
        attention_ops.is_experimental_enabled = lambda: False
        for arch, qlen, nhead, env, expected in _NATIVE_GATE_CASES:
            attention_ops.get_gfx = lambda _a=arch: _a
            if env is None:
                _os.environ.pop("AITER_MLA_DECODE_PS1_FLYDSL", None)
            else:
                _os.environ["AITER_MLA_DECODE_PS1_FLYDSL"] = env
            _assert_native_gate(attention_ops, arch, qlen, nhead, env, expected)
    finally:
        attention_ops.get_gfx, attention_ops.is_experimental_enabled, env0 = saved
        if env0 is None:
            _os.environ.pop("AITER_MLA_DECODE_PS1_FLYDSL", None)
        else:
            _os.environ["AITER_MLA_DECODE_PS1_FLYDSL"] = env0


def main():
    """aiter's CI runs each op_tests module with `python3`, not pytest."""
    for batch_size in (1, 8, 64):
        test_a_cap_never_enlarges_the_allocation(batch_size)
        test_a_tight_cap_actually_shrinks_the_allocation(batch_size)
        test_non_fast_mode_ignores_the_cap(batch_size)
    test_a_non_constraining_cap_does_not_change_the_size()
    test_a_larger_cap_is_never_smaller_than_a_tighter_one()
    test_no_cap_is_unchanged()
    _check_native_gate_without_pytest()
    aiter.logger.info("mla metadata split-cap sizing tests: all passed")


# What v1_2_device.cuh:910-921 serves natively, at fp8/fp8 with experimental off.
# Anything else with nhead % 16 == 0 is folded to 16 heads and its batch count
# scaled by qk_batch_ratio before the cap is applied, so a false "native" here
# drops the ratio and under-sizes reduce_partial_map -- which faults the GPU.
#
#   gfx950: 32/64/128 any qlen, 96 at qlen <= 6, 16 always
#   gfx942: 128 only (plus 64 at qlen == 1, and 16 always)
#   gfx1250: 32/64/128 at qlen == 1, and ONLY with AITER_MLA_DECODE_PS1_FLYDSL=1
#
# `env` is the AITER_MLA_DECODE_PS1_FLYDSL value; None means leave it unset.
# "false" is covered deliberately: a truthiness test would read it as enabled and
# take the unsafe branch. The C++ uses atoi() != 0, which also rejects it.
_NATIVE_GATE_CASES = [
    ("gfx950", 4, 32, None, True),
    ("gfx950", 4, 64, None, True),
    ("gfx950", 4, 96, None, True),
    ("gfx950", 4, 128, None, True),
    ("gfx950", 4, 48, None, False),
    ("gfx950", 8, 96, None, False),  # the 96 clause is qlen <= 6
    ("gfx942", 4, 128, None, True),
    ("gfx942", 4, 32, None, False),  # the 32/64/128 clause is gfx950-only
    ("gfx942", 4, 64, None, False),
    ("gfx942", 1, 64, None, True),  # ... except 64 at qlen == 1, which both serve
    ("gfx942", 4, 48, None, False),
    ("gfx942", 4, 96, None, False),
    ("gfx1250", 1, 32, "1", True),
    ("gfx1250", 1, 64, "1", True),
    ("gfx1250", 1, 128, "1", True),
    ("gfx1250", 1, 128, "0", False),
    ("gfx1250", 1, 128, "false", False),
    ("gfx1250", 1, 128, None, False),
    ("gfx1250", 4, 128, "1", False),  # PS1 is qlen == 1 only
]


def _assert_native_gate(attention_ops, arch, qlen, nhead, env, expected):
    """One case of _NATIVE_GATE_CASES. Shared by the pytest test and main()."""
    got = attention_ops._mla_v12_natively_supported(nhead, qlen, dtypes.fp8, dtypes.fp8)
    assert got is expected, (
        f"{arch} nhead={nhead} qlen={qlen} env={env!r}: native={got}, expected "
        f"{expected}. A false positive drops qk_batch_ratio and under-sizes "
        f"reduce_partial_map, which faults the GPU"
    )


@pytest.mark.parametrize("arch,qlen,nhead,env,expected", _NATIVE_GATE_CASES)
def test_the_native_gate_matches_the_kernel(
    monkeypatch, arch, qlen, nhead, env, expected
):
    """The fold is applied iff the planner folds, per v1_2_device.cuh:910-921.

    Parameterised over the arch rather than skipped off gfx950, so MI300X's much
    narrower native set and gfx1250's env-gated one are both checked on any
    runner. hk_mtp_experimental would make nhead*qlen == 128 native everywhere,
    so it is pinned off and these expectations describe the base gate.
    """
    import aiter.ops.attention as attention_ops

    monkeypatch.setattr(attention_ops, "get_gfx", lambda: arch)
    monkeypatch.setattr(attention_ops, "is_experimental_enabled", lambda: False)
    if env is None:
        monkeypatch.delenv("AITER_MLA_DECODE_PS1_FLYDSL", raising=False)
    else:
        monkeypatch.setenv("AITER_MLA_DECODE_PS1_FLYDSL", env)
    _assert_native_gate(attention_ops, arch, qlen, nhead, env, expected)


if __name__ == "__main__":
    main()
