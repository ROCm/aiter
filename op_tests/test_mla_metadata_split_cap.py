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
    than the no-cap path. See test_large_batch_uses_the_larger_safe_bound.
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
            1       1024      12       260
            8       1052      96       288
           64       1276     512       512
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
    exactly that split (no-cap 2040, any cap 2304) -- the same cluster count
    getting two different reservations depending on whether a dummy cap was
    passed.
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
    """max_split_per_batch <= 0 means 'no cap' and must not enter the branch,
    so the sizing has to match the historical value exactly."""
    assert _reduce_partial_map_size(64, -1) == _reduce_partial_map_size(64, 0)


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


def main():
    """aiter's CI runs each op_tests module with `python3`, not pytest."""
    for batch_size in (1, 8, 64):
        test_a_cap_never_enlarges_the_allocation(batch_size)
        test_a_tight_cap_actually_shrinks_the_allocation(batch_size)
        test_non_fast_mode_ignores_the_cap(batch_size)
    test_a_non_constraining_cap_does_not_change_the_size()
    test_a_larger_cap_is_never_smaller_than_a_tighter_one()
    test_no_cap_is_unchanged()
    aiter.logger.info("mla metadata split-cap sizing tests: all passed")


if __name__ == "__main__":
    main()
