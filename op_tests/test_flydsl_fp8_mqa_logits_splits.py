# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.flydsl import fp8_mqa_logits_kernels as kernels


@pytest.mark.parametrize(
    "grid_x,effective_cus,window_tiles,max_splits,overhead,expected",
    [
        (100_000, 1520, 1024, 64, 8, 1),  # row grid already fills the device
        (8, 4096, 8192, 4, 8, 4),  # capped by max_splits
        (1, 10**6, 10**6, 10**6, 0, 64),  # capped by the search bound
        (64, 304, 1024, 16, 3, 14),
        (64, 608, 1024, 16, 3, 9),  # more occupancy, fewer splits: one wave
    ],
)
def test_splits_by_wave_cost(
    grid_x, effective_cus, window_tiles, max_splits, overhead, expected
):
    splits = kernels._splits_by_wave_cost(
        grid_x, effective_cus, window_tiles, max_splits, overhead
    )
    assert splits == expected


def test_occupancy_is_known_on_first_call(monkeypatch):
    if not torch.cuda.is_available() or kernels._ARCH != "gfx942":
        pytest.skip("occupancy-aware splits are gfx942 only")
    kernels.compile_fp8_mqa_logits.cache_clear()
    seen = []
    real = kernels._auto_num_splits
    monkeypatch.setattr(
        kernels, "_auto_num_splits", lambda *a: seen.append(a[-1]) or real(*a)
    )
    seq_len, seq_len_kv, num_heads, head_size = 128, 8192, 64, 128
    fp8 = torch.float8_e4m3fnuz
    kernels.flydsl_fp8_mqa_logits(
        torch.randn(seq_len, num_heads, head_size, device="cuda").to(fp8),
        torch.randn(seq_len_kv, head_size, device="cuda").to(fp8),
        torch.ones(seq_len_kv, device="cuda"),
        torch.ones(seq_len, num_heads, device="cuda"),
        torch.zeros(seq_len, dtype=torch.int32, device="cuda"),
        torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device="cuda"),
    )
    assert seen[0] >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
