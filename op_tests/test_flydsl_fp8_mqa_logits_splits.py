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


def test_occupancy_is_known_on_first_call():
    """FlyDSL reports occupancy for a kernel that has not launched yet."""
    if not torch.cuda.is_available() or kernels._ARCH != "gfx942":
        pytest.skip("occupancy-aware splits are gfx942 only")
    seq_len, seq_len_kv, num_heads, head_size = 128, 8192, 64, 128
    variant = kernels._resolve_variant(None, seq_len, seq_len_kv, num_heads)
    kernels.compile_fp8_mqa_logits.cache_clear()
    launcher = kernels.compile_fp8_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=kernels._parse_variant(variant)[0],
        variant=variant,
    )
    fp8 = torch.float8_e4m3fnuz
    args = (
        torch.empty(seq_len, num_heads, head_size, device="cuda", dtype=fp8),
        torch.empty(seq_len_kv, head_size, device="cuda", dtype=fp8),
        torch.empty(seq_len_kv, device="cuda"),
        torch.empty(seq_len, num_heads, device="cuda"),
        torch.empty(seq_len, dtype=torch.int32, device="cuda"),
        torch.empty(seq_len, dtype=torch.int32, device="cuda"),
        torch.empty(seq_len, seq_len_kv, device="cuda"),
        seq_len,
        seq_len_kv,
        seq_len_kv,
        1,
        torch.cuda.current_stream(),
    )
    blocks = kernels._max_blocks_per_cu(launcher, args)
    assert isinstance(blocks, int) and blocks >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
