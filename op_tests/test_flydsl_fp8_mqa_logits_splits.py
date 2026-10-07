# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the occupancy-aware fp8_mqa_logits split heuristic.

Pure-function tests: the cost model and the lookup's resolution order need no
GPU, and arch-dependent inputs are supplied explicitly.
"""

import dataclasses
import math

import pytest

from aiter.ops.flydsl import fp8_mqa_logits_kernels as knl
from aiter.ops.flydsl import kernel_occupancy as occ_mod
from aiter.ops.flydsl.fp8_mqa_logits_kernels import (
    _MAX_SPLIT_SEARCH,
    _SPLIT_POLICIES,
    _auto_num_splits,
    _splits_by_wave_cost,
)
from aiter.ops.flydsl.kernel_occupancy import DEFAULT_OCCUPANCY, kernel_occupancy


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
            for s in range(1, min(max_splits, _MAX_SPLIT_SEARCH) + 1)
        )
        assert _cost(chosen, grid_x, effective_cus, window_tiles, overhead) == best

    def test_respects_max_splits(self):
        assert _splits_by_wave_cost(8, 4096, 8192, 4, 3) <= 4

    def test_search_is_bounded(self):
        """Host cost stays flat as seq_len_kv grows."""
        assert _splits_by_wave_cost(1, 10**6, 10**6, 10**6, 0) <= _MAX_SPLIT_SEARCH

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


class TestOccupancyLookup:
    def test_prefers_artifact_metadata(self, monkeypatch):
        """A readable artifact wins over the table."""
        monkeypatch.setattr(occ_mod, "_artifact_metadata", lambda _: {"stub": 1})
        monkeypatch.setattr(occ_mod, "_occupancy_from_metadata", lambda *a: 7)
        assert (
            kernel_occupancy(
                object(),
                arch="gfx942",
                variant="mfma_r2_w4",
                num_heads=32,
                head_size=128,
                device_index=0,
            )
            == 7
        )

    def test_falls_back_to_measured_table(self, monkeypatch):
        """First launch of a shape has no artifact, so the table answers."""
        monkeypatch.setattr(occ_mod, "_artifact_metadata", lambda _: None)
        expected = occ_mod._MEASURED_OCCUPANCY["gfx942"][("mfma_r2_w4", 32, 128)]
        assert (
            kernel_occupancy(
                None,
                arch="gfx942",
                variant="mfma_r2_w4",
                num_heads=32,
                head_size=128,
                device_index=0,
            )
            == expected
        )

    def test_falls_back_to_default_for_unknown_key(self, monkeypatch):
        monkeypatch.setattr(occ_mod, "_artifact_metadata", lambda _: None)
        assert (
            kernel_occupancy(
                None,
                arch="gfx942",
                variant="mfma_r9_w9",
                num_heads=7,
                head_size=3,
                device_index=0,
            )
            == DEFAULT_OCCUPANCY
        )

    def test_unknown_arch_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(occ_mod, "_artifact_metadata", lambda _: None)
        assert (
            kernel_occupancy(
                None,
                arch="gfx1",
                variant="mfma_r2_w4",
                num_heads=32,
                head_size=128,
                device_index=0,
            )
            == DEFAULT_OCCUPANCY
        )

    def test_metadata_parse_tolerates_missing_fields(self):
        """Malformed metadata must degrade, not raise."""
        assert occ_mod._occupancy_from_metadata({}, "gfx942", 0) is None

    def test_measured_table_values_are_sane(self):
        for arch, table in occ_mod._MEASURED_OCCUPANCY.items():
            assert table, arch
            for key, value in table.items():
                assert isinstance(value, int) and value >= 1, (arch, key)

    def test_every_arch_with_limits_has_a_table(self):
        """A table-less arch would silently degrade to DEFAULT_OCCUPANCY."""
        assert set(occ_mod._ARCH_LIMITS) <= set(occ_mod._MEASURED_OCCUPANCY)

    @pytest.mark.parametrize("head_size", [64, 128])
    def test_gfx950_streaming_swp_variant_is_in_the_table(self, head_size):
        """The SILOTIGER-1134 streaming path routes H<=32 to the ``_swp``
        variant; a missing table row would make the first-launch occupancy
        estimate fall to DEFAULT_OCCUPANCY and under-split. H=32 is the only
        head count the 32x32-MFMA ``_swp`` family admits (H16 has no atom)."""
        key = ("mfma32x32x64_bkv64_r2_w4_lds3_swp", 32, head_size)
        assert key in occ_mod._MEASURED_OCCUPANCY["gfx950"]

    def test_metadata_defers_to_table_when_shared_mem_per_cu_unavailable(self):
        """Some ROCm torch builds expose shared_memory_per_block but not the
        per-CU figure. For an LDS kernel that means the binding LDS limit can't
        be evaluated, so the model must return None (defer to the measured
        table) rather than overestimate by dropping the constraint."""

        class _Props:
            warp_size = 64
            max_threads_per_multi_processor = 2048
            # deliberately no shared_memory_per_multiprocessor

        import torch as _torch

        orig = _torch.cuda.get_device_properties
        _torch.cuda.get_device_properties = lambda _i: _Props()
        try:
            fields = {
                "threads_per_block": 256,
                "vgpr_count": 146,
                "agpr_count": 0,
                "group_segment_fixed_size": 24576,
            }
            assert occ_mod._occupancy_from_metadata(fields, "gfx950", 0) is None
        finally:
            _torch.cuda.get_device_properties = orig


class TestArchRegisterFileRules:
    """CDNA3 and CDNA4 allocate AGPRs differently; the model must not blur it."""

    @staticmethod
    def _fields(vgpr, agpr, threads=128, lds=0):
        return {
            "vgpr_count": vgpr,
            "agpr_count": agpr,
            "threads_per_block": threads,
            "group_segment_fixed_size": lds,
        }

    def test_gfx942_is_unified_and_gfx950_is_split(self):
        assert occ_mod._ARCH_LIMITS["gfx942"].unified_agpr is True
        assert occ_mod._ARCH_LIMITS["gfx950"].unified_agpr is False

    def test_split_file_rule_changes_the_answer_for_high_agpr(self):
        """gfx950's bkv256 variants reach vgpr=452/agpr=200: summed, that
        overflows the 512-VGPR file and predicts 1 where hardware reports 2.

        LDS is 0 so the register term is the one under test -- LDS capacity
        differs by arch (64 vs 160 KiB) and would otherwise bind first.
        """
        fields = self._fields(452, 200, threads=128, lds=0)
        gfx950 = occ_mod._occupancy_from_metadata(fields, "gfx950", 0)
        gfx942 = occ_mod._occupancy_from_metadata(fields, "gfx942", 0)
        if gfx950 is None or gfx942 is None:
            pytest.skip("needs a CUDA/HIP device for the device-property query")
        # max(452, 200) = 452 -> 456 granules -> 1 wave/SIMD -> 2 blocks/CU.
        assert gfx950 == 2
        # 452 + 200 = 652 > 512 -> no wave fits -> clamped to 1.
        assert gfx942 == 1

    def test_rules_agree_when_no_agprs_are_used(self):
        """Every gfx942 kernel in this op reports agpr=0, so nothing shifts."""
        fields = self._fields(86, 0, threads=256)
        a = occ_mod._occupancy_from_metadata(fields, "gfx942", 0)
        b = occ_mod._occupancy_from_metadata(fields, "gfx950", 0)
        if a is None or b is None:
            pytest.skip("needs a CUDA/HIP device for the device-property query")
        assert a == b


class TestSplitPolicy:
    def test_gfx950_is_unchanged(self):
        """gfx950 stays on the fixed-oversubscription path."""
        policy = _SPLIT_POLICIES["gfx950"]
        assert policy.occupancy_aware is False
        assert policy.cu_oversub == 4
        assert policy.min_seq_len_kv == 0
        assert policy.min_tiles_per_split == 2
        assert policy.fallback_cu == 256

    def test_gfx942_is_occupancy_aware(self):
        policy = _SPLIT_POLICIES["gfx942"]
        assert policy.occupancy_aware is True
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
    """``_auto_num_splits`` on the gfx950 fixed-oversubscription path.

    The streaming (aggressive) target is ``residency * occupancy * cu_count``;
    the common non-streaming launch keeps ``cu_oversub * cu_count`` and never
    queries occupancy.
    """

    # A streaming small-M / large-N shape: grid_x = 32768 // (8*2) ... we set
    # rows_per_block and seq_len_padded so grid_x is a small, exact number, and
    # seq_len_kv large enough that max_splits never binds.
    SEQ_LEN_PADDED = 16  # grid_x = 16 // 8 = 2
    RPB = 8
    BKV = 64
    SEQ_LEN_KV = 1 << 20  # max_splits = (2**20//64)//2 = 8192, non-binding
    CU = 256

    def _call(self, monkeypatch, *, policy, occ, aggressive, occ_calls=None):
        monkeypatch.setattr(knl, "_split_policy", lambda: policy)
        monkeypatch.setattr(knl, "_device_cu_count", lambda _idx: self.CU)

        def _occ(*_a, **_k):
            if occ_calls is not None:
                occ_calls.append(1)
            return occ

        monkeypatch.setattr(knl, "kernel_occupancy", _occ)
        return _auto_num_splits(
            self.SEQ_LEN_PADDED,
            self.SEQ_LEN_KV,
            self.RPB,
            self.BKV,
            device_index=0,
            variant="mfma32x32x64_bkv64_r2_w4_lds3_swp",
            num_heads=32,
            head_size=128,
            aggressive=aggressive,
        )

    def test_aggressive_target_is_residency_times_occ_times_cu(self, monkeypatch):
        policy = _SPLIT_POLICIES["gfx950"]  # residency=4
        grid_x = self.SEQ_LEN_PADDED // self.RPB  # 2
        splits = self._call(monkeypatch, policy=policy, occ=3, aggressive=True)
        assert splits == math.ceil(4 * 3 * self.CU / grid_x)

    def test_residency_at_occ3_reproduces_the_legacy_twelve(self, monkeypatch):
        """Behaviour-preserving on the ticket: residency=4 * occ=3 == 12x."""
        grid_x = self.SEQ_LEN_PADDED // self.RPB
        legacy = dataclasses.replace(
            _SPLIT_POLICIES["gfx950"], streaming_lowh_residency=0
        )  # falls back to streaming_lowh_cu_oversub=12
        occ_derived = self._call(
            monkeypatch, policy=_SPLIT_POLICIES["gfx950"], occ=3, aggressive=True
        )
        fixed = self._call(monkeypatch, policy=legacy, occ=99, aggressive=True)
        assert occ_derived == fixed == math.ceil(12 * self.CU / grid_x)

    def test_residency_zero_does_not_query_occupancy(self, monkeypatch):
        """Legacy fallback path must not pay for the occupancy query."""
        legacy = dataclasses.replace(
            _SPLIT_POLICIES["gfx950"], streaming_lowh_residency=0
        )
        calls = []
        self._call(
            monkeypatch, policy=legacy, occ=3, aggressive=True, occ_calls=calls
        )
        assert calls == []

    def test_non_aggressive_keeps_cu_oversub_and_skips_occupancy(self, monkeypatch):
        grid_x = self.SEQ_LEN_PADDED // self.RPB
        calls = []
        splits = self._call(
            monkeypatch,
            policy=_SPLIT_POLICIES["gfx950"],
            occ=3,
            aggressive=False,
            occ_calls=calls,
        )
        assert splits == math.ceil(4 * self.CU / grid_x)
        assert calls == []

    def test_streaming_target_never_below_the_conservative_target(self, monkeypatch):
        """A tiny residency*occ must not undercut cu_oversub*cu_count."""
        grid_x = self.SEQ_LEN_PADDED // self.RPB
        thin = dataclasses.replace(
            _SPLIT_POLICIES["gfx950"], streaming_lowh_residency=1
        )  # 1 * occ(1) * cu = 256 < cu_oversub(4)*cu = 1024
        splits = self._call(monkeypatch, policy=thin, occ=1, aggressive=True)
        assert splits == math.ceil(4 * self.CU / grid_x)
