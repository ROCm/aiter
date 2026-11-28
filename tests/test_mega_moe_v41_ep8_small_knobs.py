# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP8 DeepSeek-V4.1-Flash small-input configs: the per-bucket knob table -- no GPU needed.

Every bucket of the 48-expert EP8 layout is _tune_v41_flash_ep8_small plus exactly its
_V41_FLASH_EP8_SMALL_KNOBS entry, the table keeps one launch protocol, and every en_order it
sets is buildable. That the table leaves the outputs bit for bit as they are without it is
checked on 8 GPUs: op_tests/multigpu_tests/test_mega_moe_v2.py --suite routes
--network v41_flash --compare-untuned.
"""

from contextlib import contextmanager
from dataclasses import fields

import pytest

pytest.importorskip("flydsl", reason="MegaMoE is built with FlyDSL")

import aiter.ops.flydsl.kernels.mega_moe.mega_moe_config as cfg
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    FUSED_TOKEN_BUCKETS,
    select_fused_config,
)

V41_FLASH_EP8_SHAPE = {
    "world_size": 8,
    "experts_per_rank": 48,
    "model_dim": 5120,
    "inter_dim": 2304,
}
OTHER_SHAPES = {
    "v4_pro_ep8": {},
    "v41_flash_ep4": dict(V41_FLASH_EP8_SHAPE, world_size=4, experts_per_rank=96),
    "v41_flash_ep2": dict(V41_FLASH_EP8_SHAPE, world_size=2, experts_per_rank=192),
}
SMALL_MTPRS = (256, 512)
_GROUPS = ("stage1", "stage2", "fused_stage1", "fused_stage2")


@contextmanager
def _without_table():
    """select_fused_config with an empty _V41_FLASH_EP8_SMALL_KNOBS (the tune the table builds on)."""
    table = cfg._V41_FLASH_EP8_SMALL_KNOBS
    cfg._V41_FLASH_EP8_SMALL_KNOBS = {}
    cfg._fused_config.cache_clear()
    try:
        yield
    finally:
        cfg._V41_FLASH_EP8_SMALL_KNOBS = table
        cfg._fused_config.cache_clear()


def _buckets(mtpr):
    return [b for b in FUSED_TOKEN_BUCKETS if b <= mtpr]


def _protocol(k):
    return (
        k.dedup_rows,
        k.plan_local,
        bool(k.direct_done),
        k.tok_counters,
        k.bf16_head_tokens,
    )


def test_table_buckets_are_fused_buckets():
    assert set(cfg._V41_FLASH_EP8_SMALL_KNOBS) <= set(FUSED_TOKEN_BUCKETS)
    for bucket, knobs in cfg._V41_FLASH_EP8_SMALL_KNOBS.items():
        assert set(knobs) <= set(_GROUPS), bucket


@pytest.mark.parametrize("mtpr", SMALL_MTPRS)
def test_each_bucket_is_the_tune_plus_its_entry(mtpr):
    with _without_table():
        base = {
            b: select_fused_config(b, mtpr, **V41_FLASH_EP8_SHAPE)
            for b in _buckets(mtpr)
        }
    for bucket in _buckets(mtpr):
        tuned = select_fused_config(bucket, mtpr, **V41_FLASH_EP8_SHAPE)
        knobs = cfg._V41_FLASH_EP8_SMALL_KNOBS.get(bucket, {})
        assert tuned.p2p_quant == base[bucket].p2p_quant
        assert tuned.stage2.block_m == tuned.stage1.sort_block_m
        for group in _GROUPS:
            want = knobs.get(group, {})
            mine, before = getattr(tuned, group), getattr(base[bucket], group)
            for f in fields(mine):
                expected = want.get(f.name, getattr(before, f.name))
                assert getattr(mine, f.name) == expected, (bucket, group, f.name)


@pytest.mark.parametrize("mtpr", SMALL_MTPRS)
def test_table_keeps_the_launch_protocol(mtpr):
    """The ranks of a launch may pick different buckets: the table sets no knob they exchange through."""
    with _without_table():
        base = {
            _protocol(select_fused_config(b, mtpr, **V41_FLASH_EP8_SHAPE).fused_stage1)
            for b in _buckets(mtpr)
        }
    tuned = {
        _protocol(select_fused_config(b, mtpr, **V41_FLASH_EP8_SHAPE).fused_stage1)
        for b in _buckets(mtpr)
    }
    assert len(base) == 1 and tuned == base
    tok_done_block_n = set()
    for bucket in _buckets(mtpr):
        config = select_fused_config(bucket, mtpr, **V41_FLASH_EP8_SHAPE)
        k = config.fused_stage1
        assert not k.tok_done or k.tok_counters, bucket
        if k.tok_done:
            tok_done_block_n.add(config.stage2.block_n)
    # A tok_done reducer counts the senders' GEMM2 n-blocks: every tok_done bucket splits model_dim alike.
    assert len(tok_done_block_n) <= 1, tok_done_block_n


@pytest.mark.parametrize("mtpr", SMALL_MTPRS)
def test_en_order_is_buildable(mtpr):
    """en_order reads the local plan, and its GEMM1 order runs on m-tile units (not with pair_tiles)."""
    for bucket in _buckets(mtpr):
        k1 = select_fused_config(bucket, mtpr, **V41_FLASH_EP8_SHAPE).fused_stage1
        assert 0 <= k1.en_order < 8, bucket
        if k1.en_order:
            assert k1.plan_local, bucket
        if k1.en_order & 1:
            assert not k1.pair_tiles, bucket


@pytest.mark.parametrize("name", sorted(OTHER_SHAPES))
@pytest.mark.parametrize("mtpr", SMALL_MTPRS)
def test_other_layouts_unchanged(name, mtpr):
    """The table is the 48-expert EP8 layout's: the other fused layouts select the same configs without it."""
    shape = OTHER_SHAPES[name]
    tuned = [select_fused_config(b, mtpr, **shape) for b in _buckets(mtpr)]
    with _without_table():
        base = [select_fused_config(b, mtpr, **shape) for b in _buckets(mtpr)]
    assert tuned == base
    assert all(c.fused_stage1.en_order == 0 for c in tuned)
