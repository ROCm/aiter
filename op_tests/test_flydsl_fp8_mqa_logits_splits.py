# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the occupancy-aware fp8_mqa_logits split heuristic.

Pure-function tests: the cost model and the split target need no GPU, and
arch-dependent inputs (CU count, blocks per CU) are supplied explicitly.
"""

import dataclasses
import math

import pytest

from aiter.ops.flydsl import fp8_mqa_logits_kernels as knl
from aiter.ops.flydsl.fp8_mqa_logits_kernels import (
    _MAX_SPLITS,
    _SPLIT_POLICIES,
    _auto_num_splits,
    _splits_by_wave_cost,
)

# gfx950 with occupancy unknown: the fixed-oversubscription fallback.
_GFX950_FIXED = dataclasses.replace(
    _SPLIT_POLICIES["gfx950"], block_overhead_tiles=None
)


def _cost(splits, grid_x, effective_cus, window_tiles, overhead):
    waves = math.ceil(grid_x * splits / effective_cus)
    return waves * (math.ceil(window_tiles / splits) + overhead)


class TestWaveCostModel:
    @pytest.mark.parametrize(
        "grid_x,effective_cus,window_tiles,max_splits",
        [
            (512, 1520, 1024, 128),
            (256, 608, 256, 32),
            (1024, 2432, 512, 64),
            (8192, 2432, 128, 16),
            (1, 1520, 1024, 128),
        ],
    )
    def test_returns_the_argmin(self, grid_x, effective_cus, window_tiles, max_splits):
        """A true minimizer over the admissible range."""
        overhead = 3
        chosen = _splits_by_wave_cost(
            grid_x, effective_cus, window_tiles, max_splits, overhead
        )
        best = min(
            _cost(s, grid_x, effective_cus, window_tiles, overhead)
            for s in range(1, min(max_splits, _MAX_SPLITS) + 1)
        )
        assert _cost(chosen, grid_x, effective_cus, window_tiles, overhead) == best

    def test_respects_max_splits(self):
        assert _splits_by_wave_cost(8, 4096, 8192, 4, 3) <= 4

    def test_search_is_bounded(self):
        """Host cost stays flat as seq_len_kv grows."""
        assert _splits_by_wave_cost(1, 10**6, 10**6, 10**6, 0) <= _MAX_SPLITS

    def test_result_is_always_positive(self):
        assert _splits_by_wave_cost(10**6, 1, 1, 1, 0) >= 1

    def test_ties_prefer_fewer_splits(self):
        """Equal predicted cost should not buy extra KV traffic."""
        assert _splits_by_wave_cost(1520, 1520, 1024, 8, 0) == 1

    def test_saturated_grid_does_not_split(self):
        """A row grid that already oversubscribes stays at one split."""
        assert _splits_by_wave_cost(100_000, 1520, 1024, 64, 3) == 1

    def test_occupancy_changes_the_answer(self):
        """The property the change rests on: same shape, different occupancy,
        different split count. A fixed ``cu_oversub`` cannot express this."""
        chosen = {
            _splits_by_wave_cost(512, 304 * occupancy, 1024, 64, 3)
            for occupancy in (2, 3, 5, 8)
        }
        assert len(chosen) > 1

    def test_wave_alignment_may_lower_splits_at_higher_occupancy(self):
        """Non-monotone by design: at occupancy 2, 64*9=576 blocks fit one wave
        of 608 effective CUs and beat the 14 splits occupancy 1 picks. Asserting
        monotonicity here would encode a property the hardware lacks."""
        assert _splits_by_wave_cost(64, 304 * 1, 1024, 16, 3) == 14
        assert _splits_by_wave_cost(64, 304 * 2, 1024, 16, 3) == 9

    def test_never_exceeds_one_block_per_effective_cu_when_splitting(self):
        """Splitting stops once every effective CU has work."""
        for occupancy in (2, 5, 8):
            effective_cus = 304 * occupancy
            for grid_x in (64, 256, 512, 2048):
                splits = _splits_by_wave_cost(grid_x, effective_cus, 1024, 128, 3)
                assert grid_x * splits <= effective_cus * 128


class TestSplitPolicy:
    def test_gfx950_is_occupancy_aware(self):
        """gfx950 uses gfx942's wave-cost model; its gates are its own."""
        policy = _SPLIT_POLICIES["gfx950"]
        assert policy.block_overhead_tiles == 8
        assert policy.cu_oversub == 4
        assert policy.min_seq_len_kv == 0
        assert policy.min_tiles_per_split == 2
        assert policy.fallback_cu == 256

    def test_gfx942_is_occupancy_aware(self):
        policy = _SPLIT_POLICIES["gfx942"]
        assert policy.block_overhead_tiles is not None
        assert policy.block_overhead_tiles > 0
        # The gates the occupancy path still honours.
        assert policy.min_seq_len_kv == 4096
        assert policy.min_tiles_per_split == 8

    def test_gfx950_streaming_is_occupancy_derived(self):
        """The streaming target is grounded in measured occupancy, not a magic
        constant; the fixed fallback is retained for rollback."""
        policy = _SPLIT_POLICIES["gfx950"]
        assert policy.streaming_lowh_residency == 4
        assert policy.streaming_lowh_cu_oversub == 12


class TestStreamingSplitTarget:
    """``_auto_num_splits`` on the gfx950 fixed-oversubscription fallback
    (``blocks_per_cu`` unknown, or a policy without ``block_overhead_tiles``).

    The streaming (aggressive) target is ``residency * blocks_per_cu *
    cu_count``; without ``blocks_per_cu`` it falls back to
    ``streaming_lowh_cu_oversub``, and the common non-streaming launch keeps
    ``cu_oversub * cu_count``.
    """

    # A streaming small-M / large-N shape: seq_len and rows_per_block are set so
    # grid_x is a small, exact number, and seq_len_kv large enough that
    # max_splits never binds.
    SEQ_LEN = 16  # grid_x = 16 // 8 = 2
    RPB = 8
    BKV = 64
    SEQ_LEN_KV = 1 << 20  # max_splits = (2**20//64)//2 = 8192, non-binding
    CU = 256

    def _call(self, monkeypatch, *, policy, blocks_per_cu, aggressive):
        monkeypatch.setattr(knl, "_split_policy", lambda: policy)
        monkeypatch.setattr(knl, "_device_cu_count", lambda _idx: self.CU)
        return _auto_num_splits(
            self.SEQ_LEN,
            self.SEQ_LEN_KV,
            self.RPB,
            self.BKV,
            0,
            blocks_per_cu,
            aggressive=aggressive,
        )

    def test_aggressive_target_is_residency_times_occ_times_cu(self, monkeypatch):
        policy = _GFX950_FIXED  # residency=4
        grid_x = self.SEQ_LEN // self.RPB  # 2
        splits = self._call(
            monkeypatch, policy=policy, blocks_per_cu=3, aggressive=True
        )
        assert splits == math.ceil(4 * 3 * self.CU / grid_x)

    def test_residency_at_occ3_reproduces_the_legacy_twelve(self, monkeypatch):
        """Behaviour-preserving on the ticket: residency=4 * occ=3 == 12x."""
        grid_x = self.SEQ_LEN // self.RPB
        legacy = dataclasses.replace(
            _GFX950_FIXED, streaming_lowh_residency=0
        )  # falls back to streaming_lowh_cu_oversub=12
        occ_derived = self._call(
            monkeypatch,
            policy=_GFX950_FIXED,
            blocks_per_cu=3,
            aggressive=True,
        )
        fixed = self._call(
            monkeypatch, policy=legacy, blocks_per_cu=99, aggressive=True
        )
        assert occ_derived == fixed == math.ceil(12 * self.CU / grid_x)

    def test_unknown_blocks_per_cu_falls_back_to_fixed_streaming(self, monkeypatch):
        """First call under graph capture has no ``blocks_per_cu``."""
        grid_x = self.SEQ_LEN // self.RPB
        splits = self._call(
            monkeypatch,
            policy=_SPLIT_POLICIES["gfx950"],
            blocks_per_cu=None,
            aggressive=True,
        )
        assert splits == math.ceil(12 * self.CU / grid_x)

    def test_non_aggressive_keeps_cu_oversub(self, monkeypatch):
        grid_x = self.SEQ_LEN // self.RPB
        splits = self._call(
            monkeypatch,
            policy=_GFX950_FIXED,
            blocks_per_cu=3,
            aggressive=False,
        )
        assert splits == math.ceil(4 * self.CU / grid_x)

    def test_streaming_target_never_below_the_conservative_target(self, monkeypatch):
        """A tiny residency*occ must not undercut cu_oversub*cu_count."""
        grid_x = self.SEQ_LEN // self.RPB
        thin = dataclasses.replace(
            _GFX950_FIXED, streaming_lowh_residency=1
        )  # 1 * occ(1) * cu = 256 < cu_oversub(4)*cu = 1024
        splits = self._call(monkeypatch, policy=thin, blocks_per_cu=1, aggressive=True)
        assert splits == math.ceil(4 * self.CU / grid_x)


class TestGfx950WaveCostSplitTarget:
    """``_auto_num_splits`` on gfx950 with occupancy known: the wave-cost
    model under gfx950's own gates (no ``min_seq_len_kv``, 2-tile chunks)."""

    def test_uses_the_wave_cost_model(self, monkeypatch):
        policy = _SPLIT_POLICIES["gfx950"]
        monkeypatch.setattr(knl, "_split_policy", lambda: policy)
        monkeypatch.setattr(knl, "_device_cu_count", lambda _idx: 256)
        got = _auto_num_splits(128, 8192, 1, 64, 0, 5, aggressive=True)
        max_splits = (8192 // 64) // policy.min_tiles_per_split
        expected = _splits_by_wave_cost(
            128, 256 * 5, 8192 // 64, max_splits, policy.block_overhead_tiles
        )
        assert got == expected


class TestOccupancyAwareSplitTarget:
    """``_auto_num_splits`` on the gfx942 occupancy-aware path."""

    def _call(self, monkeypatch, *, blocks_per_cu, aggressive=False):
        monkeypatch.setattr(knl, "_split_policy", lambda: _SPLIT_POLICIES["gfx942"])
        monkeypatch.setattr(knl, "_device_cu_count", lambda _idx: 304)
        return _auto_num_splits(
            512, 1 << 17, 4, 128, 0, blocks_per_cu, aggressive=aggressive
        )

    def test_uses_the_wave_cost_model(self, monkeypatch):
        policy = _SPLIT_POLICIES["gfx942"]
        max_splits = ((1 << 17) // 128) // policy.min_tiles_per_split
        expected = _splits_by_wave_cost(
            512 // 4,
            304 * 2,
            (1 << 17) // 128,
            max_splits,
            policy.block_overhead_tiles,
        )
        assert self._call(monkeypatch, blocks_per_cu=2) == expected

    def test_streaming_flag_does_not_change_the_cost_model(self, monkeypatch):
        """gfx942 has no streaming targets; ``aggressive`` must be inert."""
        assert self._call(monkeypatch, blocks_per_cu=2, aggressive=True) == self._call(
            monkeypatch, blocks_per_cu=2
        )

    def test_unknown_blocks_per_cu_keeps_cu_oversub(self, monkeypatch):
        grid_x = 512 // 4
        assert self._call(monkeypatch, blocks_per_cu=None) == math.ceil(
            4 * 304 / grid_x
        )

