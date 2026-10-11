# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression for get_block_size_M ranking tile heights by rounds alone.

When no tuned config exists, get_2stage_cfgs takes block_m from
get_block_size_M (in fused_moe and in fused_moe_dp_shared_expert, which share
one helper). A decode-sized call has only a few real rows per expert, so most
of a large M tile is padding. Ranking candidates by rounds of tiles prices a
64-row tile like a 32-row one, and the idle-CU tiebreak then picks the larger,
mostly empty block: 32 tokens x top-8 over 32 experts schedule 2304 rows for 256
real ones at block_m=64, against 1280 at block_m=32. The tiebreak also counted a
last round that the tiles fill exactly as a whole idle round.

The choice depends only on the shape and the CU count, so these tests pin the
CU count and need no GPU.
"""

import pytest

from aiter import fused_moe, fused_moe_dp_shared_expert
from aiter.fused_moe import get_block_size_M

CU_NUM = 256
BLOCKS = (32, 64, 128)


@pytest.fixture(autouse=True)
def fixed_cu_count(monkeypatch):
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: CU_NUM)
    get_block_size_M.cache_clear()
    yield
    get_block_size_M.cache_clear()


def padded_rows(token, topk, expert, block_m):
    """Rows the sorted MoE GEMM schedules at this block size, padding included."""
    max_num_tokens = token * topk + expert * block_m - topk
    return -(-max_num_tokens // block_m) * block_m


def previous_choice(token, topk, expert, inter_dim):
    """The ranking before this fix: rounds of tiles, then idle CUs."""
    tg_n = (inter_dim + 127) // 128
    ranked = []
    for el in BLOCKS:
        tg_num = tg_n * (token * topk + expert * el - topk + el - 1) // el
        ranked.append(((tg_num + CU_NUM - 1) // CU_NUM, CU_NUM - tg_num % CU_NUM, el))
    return min(ranked, key=lambda x: x[:2])[-1]


def test_decode_shape_triggers_mostly_padding_block():
    # 32 tokens x top-8 over 32 experts: 8 real rows per expert.
    shape = (32, 8, 32, 4096)
    assert previous_choice(*shape) == 64
    assert padded_rows(*shape[:3], 64) == 2304
    assert padded_rows(*shape[:3], 32) == 1280
    assert get_block_size_M(*shape) == 32


def test_exact_wave_counts_no_idle_cu():
    # At block_m=32 the tiles fill the last round exactly (8960 = 35 x 256), and
    # 32 and 64 tie on weighted rounds; that full round must count 0 idle CUs.
    shape = (382, 8, 64, 7168)
    tg_n = (shape[3] + 127) // 128
    assert tg_n * padded_rows(*shape[:3], 32) // 32 == 35 * CU_NUM
    assert padded_rows(*shape[:3], 32) < padded_rows(*shape[:3], 64)
    assert get_block_size_M(*shape) == 32


def test_dp_shared_expert_uses_the_same_ranking():
    # fused_moe_dp_shared_expert's untuned fallback kept its own copy of the
    # rounds-only ranking; it now shares this helper.
    assert fused_moe_dp_shared_expert.get_block_size_M is get_block_size_M
    shape = (32, 8, 8, 2048)
    assert previous_choice(*shape) == 64
    assert fused_moe_dp_shared_expert.get_block_size_M(*shape) == 32


@pytest.mark.parametrize(
    "token, topk, expert, inter_dim",
    [(32, 8, 32, 4096), (64, 8, 64, 2048), (8, 8, 16, 7168), (128, 8, 256, 1536)],
)
def test_decode_shapes_pick_the_smaller_block(token, topk, expert, inter_dim):
    old = previous_choice(token, topk, expert, inter_dim)
    new = get_block_size_M(token, topk, expert, inter_dim)
    assert new == 32
    assert padded_rows(token, topk, expert, new) < padded_rows(token, topk, expert, old)


@pytest.mark.parametrize("expert", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("inter_dim", [512, 1536, 4096, 7168])
def test_decode_never_pads_more_than_before(expert, topk, inter_dim):
    # Every token count with at most 64 real rows per expert.
    for token in range(1, 64 * expert // topk + 1):
        new = get_block_size_M(token, topk, expert, inter_dim)
        old = previous_choice(token, topk, expert, inter_dim)
        assert new in BLOCKS
        assert padded_rows(token, topk, expert, new) <= padded_rows(
            token, topk, expert, old
        )


@pytest.mark.parametrize(
    "token, topk, expert, inter_dim",
    [(4096, 8, 32, 4096), (16384, 8, 128, 2048), (32768, 8, 256, 1536)],
)
def test_prefill_shapes_keep_their_choice(token, topk, expert, inter_dim):
    # 1024 real rows per expert: padding is negligible for every candidate.
    assert get_block_size_M(token, topk, expert, inter_dim) == 128
    assert previous_choice(token, topk, expert, inter_dim) == 128
