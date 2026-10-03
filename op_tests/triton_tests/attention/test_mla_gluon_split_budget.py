# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the bh16 KV-split budget in mla_gluon.

The expected values below are measured split counts, not the model's own
output, so coefficients that stop tracking the hardware fail here rather than
quietly agreeing with themselves.
"""
import pytest
import torch

from aiter.ops.triton.gluon.mla_gluon import _bh16_num_kv_splits, mla_gluon
from aiter.ops.triton.utils._triton import arch_info

BLOCK_N = 128
CKV, KPE = 512, 64

# MI355X, 18,688-token context, nhead=96 (six 16-head blocks), NUM_KV_SPLITS
# swept 1..8 at 100 iterations x 3 repeats, microseconds. "rows" is batch x
# qlen. Coefficients were fitted on rows in {8,16,24,32,40,48,64}; the rest
# were held out and measured afterwards.
MEASURED_US = {
    8: {1: 324.7, 2: 167.3, 3: 115.6, 4: 90.4, 5: 81.0, 6: 121.8, 7: 105.4, 8: 97.6},
    12: {
        1: 325.7,
        2: 168.9,
        3: 119.4,
        4: 173.7,
        5: 146.9,
        6: 125.6,
        7: 116.2,
        8: 144.8,
    },
    16: {
        1: 327.4,
        2: 169.9,
        3: 225.9,
        4: 176.0,
        5: 149.1,
        6: 183.8,
        7: 159.6,
        8: 146.8,
    },
    20: {
        1: 329.1,
        2: 173.5,
        3: 230.4,
        4: 179.9,
        5: 221.3,
        6: 189.6,
        7: 225.6,
        8: 196.3,
    },
    24: {
        1: 329.3,
        2: 331.4,
        3: 232.9,
        4: 262.9,
        5: 223.3,
        6: 248.7,
        7: 218.2,
        8: 244.5,
    },
    28: {
        1: 330.5,
        2: 336.9,
        3: 239.1,
        4: 268.9,
        5: 311.9,
        6: 259.9,
        7: 288.5,
        8: 298.1,
    },
    32: {
        1: 331.5,
        2: 337.4,
        3: 345.9,
        4: 275.1,
        5: 309.3,
        6: 321.4,
        7: 336.3,
        8: 341.8,
    },
    36: {
        1: 332.6,
        2: 340.3,
        3: 358.5,
        4: 360.1,
        5: 395.4,
        6: 395.4,
        7: 382.4,
        8: 394.7,
    },
    40: {
        1: 332.3,
        2: 341.9,
        3: 359.6,
        4: 386.5,
        5: 411.0,
        6: 433.8,
        7: 437.6,
        8: 446.9,
    },
    48: {
        1: 650.0,
        2: 505.3,
        3: 476.4,
        4: 487.1,
        5: 519.5,
        6: 532.1,
        7: 519.9,
        8: 526.7,
    },
    56: {
        1: 655.5,
        2: 514.7,
        3: 520.9,
        4: 600.4,
        5: 619.4,
        6: 613.7,
        7: 633.5,
        8: 634.3,
    },
    64: {
        1: 654.9,
        2: 542.6,
        3: 646.2,
        4: 667.3,
        5: 708.9,
        6: 697.4,
        7: 710.8,
        8: 712.1,
    },
    80: {
        1: 664.1,
        2: 747.3,
        3: 832.2,
        4: 865.3,
        5: 890.6,
        6: 904.8,
        7: 906.0,
        8: 894.1,
    },
    96: {
        1: 983.3,
        2: 943.5,
        3: 1007.9,
        4: 1006.6,
        5: 1071.5,
        6: 1068.1,
        7: 1086.3,
        8: 1096.1,
    },
}
MEASURED_KV_LEN = 18688
MEASURED_M_BLOCKS = 6
# rows=96 is the one shape where the model over-splits: it picks 3 where 2 was
# fastest, costing 2.5% against the shipped budget. The bound is set just above
# that so a regression anywhere else still trips the test.
MAX_LOSS_VS_SHIPPED = 1.03
MAX_LOSS_VS_BEST = 1.10


def _shipped(base_grid):
    return max(1, 256 // base_grid)


@pytest.mark.parametrize("base_grid", [1, 6, 48, 144, 256, 257, 576, 4096])
def test_no_hint_is_the_shipped_budget(base_grid):
    """Callers that say nothing about their context must not move at all."""
    assert _bh16_num_kv_splits(base_grid, BLOCK_N) == _shipped(base_grid)
    assert _bh16_num_kv_splits(base_grid, BLOCK_N, None) == _shipped(base_grid)


@pytest.mark.parametrize("rows", sorted(MEASURED_US))
def test_hinted_pick_tracks_the_measured_optimum(rows):
    base_grid = rows * MEASURED_M_BLOCKS
    times = MEASURED_US[rows]
    picked = _bh16_num_kv_splits(base_grid, BLOCK_N, MEASURED_KV_LEN)
    assert picked in times, f"pick s={picked} is outside the measured sweep"
    best = min(times.values())
    shipped = times[_shipped(base_grid)]
    assert times[picked] <= best * MAX_LOSS_VS_BEST, (
        f"rows={rows}: picked s={picked} at {times[picked]} us against a "
        f"measured best of {best} us"
    )
    assert times[picked] <= shipped * MAX_LOSS_VS_SHIPPED, (
        f"rows={rows}: picked s={picked} at {times[picked]} us, slower than "
        f"the shipped budget's {shipped} us"
    )


def test_recovers_the_shapes_the_shipped_budget_collapses():
    """Where the grid exceeds one wave the shipped budget floors to a single
    split and every workgroup walks the whole context; these are the shapes
    the hint exists for."""
    for rows, want_speedup in ((24, 1.4), (32, 1.15), (48, 1.3), (64, 1.15)):
        base_grid = rows * MEASURED_M_BLOCKS
        times = MEASURED_US[rows]
        assert _shipped(base_grid) == 1
        picked = _bh16_num_kv_splits(base_grid, BLOCK_N, MEASURED_KV_LEN)
        got = times[1] / times[picked]
        assert got >= want_speedup, f"rows={rows}: only {got:.2f}x"


def test_no_split_when_the_grid_already_fills_a_wave():
    """rows=40 x 6 head blocks is 240 of 256 workgroups: there is no idle
    capacity to buy, so splitting could only add stage-2 work."""
    assert _bh16_num_kv_splits(40 * MEASURED_M_BLOCKS, BLOCK_N, MEASURED_KV_LEN) == 1


@pytest.mark.parametrize("rows", [1, 4, 8, 24, 64])
def test_single_block_context_is_never_split(rows):
    """One BLOCK_N of KV cannot be divided, so splitting is pure overhead."""
    assert _bh16_num_kv_splits(rows * MEASURED_M_BLOCKS, BLOCK_N, BLOCK_N) == 1


# A second shape family, measured after the coefficients were fixed: 12 heads
# zero-padded to 16 (ONE head block, where DCP gives six), DSpark verify
# qlen=5, no DCP so each rank walks the whole context. base_grid is conc*5.
# This is the shape sgl-project/sglang#41388 routes onto the asm path.
# {(kv_len, base_grid): {splits: us}}
MEASURED_QH16 = {
    (32768, 5): {
        1: 658,
        2: 334,
        3: 228,
        4: 172,
        6: 119,
        8: 91,
        12: 66,
        16: 51,
        24: 51,
        32: 51,
        43: 50,
        48: 50,
        51: 50,
        64: 50,
        96: 50,
        128: 51,
        192: 53,
        256: 51,
    },
    (32768, 40): {
        1: 666,
        2: 339,
        3: 234,
        4: 178,
        6: 136,
        8: 184,
        12: 142,
        16: 153,
        24: 148,
        32: 147,
        48: 176,
        64: 172,
        96: 208,
    },
    (32768, 80): {
        1: 670,
        2: 343,
        3: 256,
        4: 352,
        6: 264,
        8: 280,
        12: 270,
        16: 265,
        24: 294,
        32: 288,
        48: 327,
    },
    (32768, 320): {1: 1356, 2: 1042, 3: 1001, 4: 973, 6: 1025, 8: 1001, 12: 1044},
    (131072, 5): {
        1: 2623,
        2: 1315,
        3: 888,
        4: 667,
        6: 451,
        8: 341,
        12: 232,
        16: 177,
        24: 122,
        32: 94,
        48: 74,
        49: 82,
        51: 73,
        64: 99,
        96: 79,
        128: 87,
        192: 97,
        256: 95,
    },
    (131072, 40): {
        1: 2634,
        2: 1321,
        3: 893,
        4: 675,
        6: 497,
        8: 685,
        12: 503,
        16: 532,
        24: 511,
        32: 496,
        48: 533,
        64: 518,
        96: 556,
    },
    (131072, 80): {
        1: 2639,
        2: 1329,
        3: 976,
        4: 1362,
        6: 985,
        8: 1042,
        12: 991,
        16: 967,
        24: 1006,
        32: 989,
        48: 1027,
    },
    (131072, 320): {1: 6000, 2: 4585, 3: 4059, 4: 3826, 6: 4128, 8: 3852, 12: 3890},
}


@pytest.mark.parametrize("key", sorted(MEASURED_QH16))
def test_pick_generalizes_to_the_one_head_block_shape(key):
    """The coefficients were fitted at nhead=96 and a single context length.
    They have to hold at one head block and two other contexts without refit,
    or they are describing one benchmark rather than the hardware."""
    kv_len, base_grid = key
    times = MEASURED_QH16[key]
    picked = _bh16_num_kv_splits(base_grid, BLOCK_N, kv_len)
    assert picked in times, f"pick s={picked} is outside the measured sweep"
    best = min(times.values())
    shipped = times[_shipped(base_grid)]
    assert times[picked] <= best * MAX_LOSS_VS_BEST
    assert times[picked] <= shipped * MAX_LOSS_VS_SHIPPED


@pytest.mark.parametrize("kv_len", [32768, 131072])
def test_recovers_the_high_concurrency_collapse_at_one_head_block(kv_len):
    """conc 64 x qlen 5 is 320 workgroups, so the shipped budget floors to one
    split and the whole context is walked by every workgroup."""
    times = MEASURED_QH16[(kv_len, 320)]
    assert _shipped(320) == 1
    picked = _bh16_num_kv_splits(320, BLOCK_N, kv_len)
    assert times[1] / times[picked] >= 1.35


def test_margin_keeps_the_shipped_pick_on_a_modelled_tie():
    """base_grid 5 at 128k is the case that forced the margin: the model saw a
    1.01x gain in moving 51 -> 49 and the measured curve is jagged enough there
    that 49 reads 11% slower than 51. A predicted tie is not worth the risk."""
    times = MEASURED_QH16[(131072, 5)]
    picked = _bh16_num_kv_splits(5, BLOCK_N, 131072)
    assert picked == _shipped(5) == 51
    assert times[49] / times[51] > 1.10  # what the margin is protecting against


@pytest.mark.parametrize("base_grid", [1, 5, 10, 16, 40])
@pytest.mark.parametrize("kv_len", [32768, 131072])
def test_small_grids_keep_their_many_splits(base_grid, kv_len):
    """A single query position of 16 heads is one workgroup, so filling the
    machine takes ~256 of them. The hint must not pull such a launch down to a
    handful of splits -- that is the shape this regime was written for."""
    shipped = _shipped(base_grid)
    picked = _bh16_num_kv_splits(base_grid, BLOCK_N, kv_len)
    assert (
        picked >= shipped // 2
    ), f"base_grid={base_grid}: hint drops {shipped} splits to {picked}"


@pytest.mark.parametrize("rows", [8, 16, 24, 32, 48, 64])
def test_pick_is_monotone_in_context_length(rows):
    """At a fixed grid a longer context can only justify more splits."""
    base_grid = rows * MEASURED_M_BLOCKS
    picks = [
        _bh16_num_kv_splits(base_grid, BLOCK_N, kv)
        for kv in (512, 2048, 8192, 32768, 131072)
    ]
    assert picks == sorted(picks), f"rows={rows}: {picks}"


@pytest.mark.skipif(
    arch_info.get_arch() != "gfx950", reason="bh16bn128 fp8 regime is gfx950 only"
)
@pytest.mark.parametrize("rows", [16, 24, 32])
def test_hint_changes_the_schedule_not_the_result(rows):
    kv_len, nhead, page, pool = MEASURED_KV_LEN, 96, 64, 1 << 20
    assert _bh16_num_kv_splits(rows * 6, BLOCK_N, kv_len) != _shipped(
        rows * 6
    ), "this shape would not exercise a different split count"

    dev = "cuda"
    torch.manual_seed(rows)
    starts = (
        torch.randint(
            0, pool // page, (rows, kv_len // page), device=dev, dtype=torch.int32
        )
        * page
    )
    table = (
        starts[:, :, None] + torch.arange(page, device=dev, dtype=torch.int32)
    ).reshape(rows, kv_len)
    seqlens = torch.full((rows,), kv_len, device=dev, dtype=torch.int32)
    kv = torch.randn(pool, CKV + KPE, device=dev, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    q = torch.randn(rows, nhead, CKV + KPE, device=dev, dtype=torch.bfloat16)
    q_nope, q_pe = torch.split(q, [CKV, KPE], dim=-1)

    outs, lses = [], []
    for hint in (None, kv_len):
        o = torch.empty(rows, nhead, CKV, device=dev, dtype=torch.bfloat16)
        _, lse = mla_gluon(
            q_nope,
            q_pe,
            kv,
            o,
            table,
            seqlens,
            1.0 / (CKV + KPE) ** 0.5,
            k_pe=None,
            kv_pe_offset=CKV,
            use_2d_view=True,
            kv_scale=1.0,
            min_kv_seq_len=1,
            return_lse=True,
            kv_len_hint=hint,
        )
        outs.append(o.float())
        lses.append(lse.float())

    torch.testing.assert_close(outs[0], outs[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lses[0], lses[1], atol=2e-2, rtol=2e-2)
