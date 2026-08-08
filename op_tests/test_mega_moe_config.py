# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import pytest

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    MAX_MTPR_CLASS,
    TOKEN_BUCKETS,
    expert_config_class,
    mtpr_config_class,
    nearest_token_bucket,
    select_mega_moe_config,
)


@pytest.mark.parametrize(
    "tokens,bucket",
    [
        (2, 1),
        (3, 4),
        (6, 8),
        (16300, 16384),
        (16400, 16384),
        (24576, 32768),
        (65536, 32768),
    ],
)
def test_nearest_token_bucket_prefers_larger_on_ties(tokens, bucket):
    assert nearest_token_bucket(tokens) == bucket


@pytest.mark.parametrize("mtpr", [2048, 4096, 8192, 16384, 32768, 65536])
def test_large_mtpr_uses_one_config_class(mtpr):
    assert mtpr_config_class(mtpr) == MAX_MTPR_CLASS


@pytest.mark.parametrize("tokens", [1, 8, 32, 128, 256, 512, 1024, 2048])
def test_large_mtpr_configs_are_capacity_invariant(tokens):
    reference = select_mega_moe_config(tokens, 2048)
    for mtpr in (4096, 8192, 16384, 32768):
        if tokens <= mtpr:
            assert select_mega_moe_config(tokens, mtpr) is reference


@pytest.mark.parametrize(
    "tokens,sbm,dispatch_cu,work_shards,persist_cu",
    [
        (1, 32, 224, 1, 240),
        (32, 32, 64, 1, 240),
        (128, 32, 192, 4, 240),
        (256, 64, 160, 4, 240),
        (512, 64, 64, 4, 240),
        (1024, 64, 64, 4, 224),
        (2048, 64, 64, 8, 256),
        (4096, 128, 64, 4, 240),
        (8192, 128, 96, 4, 240),
        (16384, 128, 32, 4, 192),
        (32768, 128, 32, 4, 240),
    ],
)
def test_large_mtpr_profiles_follow_geometry_rules(
    tokens, sbm, dispatch_cu, work_shards, persist_cu
):
    config = select_mega_moe_config(tokens, max(2048, tokens))
    stage1, stage2 = config.stage1, config.stage2

    assert stage1.sort_block_m == sbm
    assert stage1.num_dispatch_cu == dispatch_cu
    assert stage1.work_shards == work_shards
    assert stage1.grid_mult == 1
    assert stage1.use_tile_resource
    assert stage1.payload_chunk_rows == 384
    assert stage1.payload_tile_ready
    assert stage2.block_m == (64 if sbm == 128 else 32)
    assert stage2.persist_cu == persist_cu
    assert stage2.skew_cu == (96 if tokens >= 512 else 0)
    assert config.p2p_quant == "fp8_blockwise_1x32"


def test_fixed_and_bounded_compact_profiles_remain_specialized():
    fixed = select_mega_moe_config(128, 128)
    bounded = select_mega_moe_config(512, 512)

    assert (
        fixed.stage1.tile_n,
        fixed.stage1.num_waves,
        fixed.stage1.num_dispatch_cu,
    ) == (128, 4, 224)
    assert not fixed.stage1.payload_tile_ready and fixed.p2p_quant == "none"
    assert (bounded.stage1.sort_block_m, bounded.stage1.grid_mult) == (64, 2)
    assert bounded.stage1.num_dispatch_cu == 128
    assert not bounded.stage1.payload_tile_ready and bounded.p2p_quant == "none"


def test_large_mtpr_protocol_is_rank_invariant_across_token_buckets():
    configs = [select_mega_moe_config(tokens, 32768) for tokens in TOKEN_BUCKETS]

    assert {config.p2p_quant for config in configs} == {"fp8_blockwise_1x32"}
    assert {config.stage1.payload_chunk_rows for config in configs} == {384}
    assert {config.stage1.payload_tile_ready for config in configs} == {True}


@pytest.mark.parametrize("experts_per_rank", [48, 52, 56, 64])
def test_redundant_experts_share_one_wave_geometry(experts_per_rank):
    base = select_mega_moe_config(8192, 32768, experts_per_rank=48)
    redundant = select_mega_moe_config(8192, 32768, experts_per_rank=experts_per_rank)

    assert expert_config_class(experts_per_rank) == 64
    assert redundant is base


def test_multiple_expert_waves_scale_payload_producers():
    base = select_mega_moe_config(4096, 32768, experts_per_rank=48)
    wide = select_mega_moe_config(4096, 32768, experts_per_rank=80)

    assert wide.stage1.num_dispatch_cu == 2 * base.stage1.num_dispatch_cu
    assert wide.stage2 == base.stage2


def test_model_geometry_selects_tile_widths():
    config = select_mega_moe_config(8192, 32768, model_dim=3584, inter_dim=1536)

    assert config.stage1.tile_n == 256
    assert config.stage2.block_n == 128


def test_nearby_tokens_share_the_bucket_config():
    assert select_mega_moe_config(500, 8192) is select_mega_moe_config(512, 32768)


@pytest.mark.parametrize(
    "tokens,mtpr,kwargs",
    [
        (0, 16, {}),
        (17, 16, {}),
        (1, 0, {}),
        (1, 24, {}),
        (1, 16, {"experts_per_rank": 0}),
        (1, 16, {"model_dim": 0}),
    ],
)
def test_invalid_shape_is_rejected(tokens, mtpr, kwargs):
    with pytest.raises(ValueError):
        select_mega_moe_config(tokens, mtpr, **kwargs)
