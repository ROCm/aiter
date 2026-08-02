# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import pytest

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    TOKEN_BUCKETS,
    nearest_token_bucket,
    select_mega_moe_config,
)

_STANDARD_PROFILES = {
    1: (32, 256, 4, 1, 64, 0, 1, 2, 32, 256, 0, 0, 0, "none"),
    4: (32, 256, 4, 1, 128, 0, 1, 2, 32, 256, 0, 0, 0, "none"),
    8: (32, 256, 4, 2, 128, 0, 1, 2, 32, 128, 0, 0, 0, "none"),
    16: (32, 128, 4, 4, 96, 0, 1, 1, 32, 128, 0, 0, 0, "none"),
    32: (32, 128, 4, 3, 128, 0, 0, 2, 32, 128, 0, 0, 0, "none"),
    64: (32, 128, 4, 3, 208, 0, 0, 2, 32, 256, 0, 0, 0, "none"),
    128: (32, 128, 4, 3, 224, 0, 0, 2, 32, 128, 1, 240, 0, "none"),
    256: (64, 512, 8, 1, 160, 1, 1, 2, 32, 128, 1, 128, 0, "none"),
    512: (64, 512, 8, 2, 128, 1, 0, 2, 32, 128, 1, 240, 1, "none"),
    1024: (64, 512, 8, 2, 128, 1, 0, 2, 32, 256, 1, 240, 1, "none"),
    2048: (64, 512, 8, 1, 32, 1, 1, 2, 32, 256, 1, 240, 1, "fp8_blockwise_1x32"),
    4096: (128, 512, 8, 1, 32, 1, 0, 2, 64, 256, 1, 256, 0, "fp8_blockwise_1x32"),
    8192: (128, 512, 8, 1, 32, 1, 0, 2, 64, 256, 1, 240, 0, "fp8_blockwise_1x32"),
    16384: (128, 512, 8, 1, 32, 1, 1, 2, 64, 256, 1, 256, 0, "fp8_blockwise_1x32"),
    32768: (128, 512, 8, 1, 32, 1, 1, 2, 64, 256, 1, 240, 0, "fp8_blockwise_1x32"),
}


def _profile(config):
    stage1 = config.stage1
    stage2 = config.stage2
    return (
        stage1.sort_block_m,
        stage1.tile_n,
        stage1.num_waves,
        stage1.grid_mult,
        stage1.num_dispatch_cu,
        int(stage1.mfma_amajor),
        int(stage1.use_tile_resource),
        stage1.waves_per_eu_hint,
        stage2.block_m,
        stage2.block_n,
        int(stage2.persist),
        stage2.persist_cu,
        int(stage2.persist_strided),
        config.p2p_quant,
    )


@pytest.mark.parametrize("tokens,expected", _STANDARD_PROFILES.items())
def test_standard_profiles_match_tuned_artifacts(tokens, expected):
    config = select_mega_moe_config(tokens, max(16, tokens))
    stage1 = config.stage1
    stage2 = config.stage2

    assert _profile(config) == expected
    assert stage1.async_a_copy == (tokens >= 256 and tokens != 2048)
    assert stage1.b_nt == (0 if tokens == 1 or tokens >= 1024 else 3)
    assert stage1.work_shards == (4 if tokens >= 8192 else 8)
    assert stage1.external_grouping == (tokens >= 2048)
    assert stage1.external_counting == (tokens >= 8192)
    assert stage1.pipe_weights and stage1.swizzle_a
    assert stage2.use_nt == (tokens <= 128)
    assert stage2.b_hoist and stage2.ascale_prefetch
    assert stage2.spatial_partition == 402 and not stage2.bf16_lds


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


def test_mtpr_selects_fixed_or_compact_configs():
    fixed = select_mega_moe_config(128, 128)
    compact = select_mega_moe_config(128, 8192)

    assert (
        fixed.stage1.tile_n,
        fixed.stage1.num_waves,
        fixed.stage1.num_dispatch_cu,
    ) == (128, 4, 224)
    assert (
        compact.stage1.tile_n,
        compact.stage1.num_waves,
        compact.stage1.num_dispatch_cu,
    ) == (512, 8, 192)
    for tokens in (8, 16, 32):
        assert select_mega_moe_config(tokens, 128).stage2.block_n == 128
        assert select_mega_moe_config(tokens, 8192).stage2.block_n == 256


@pytest.mark.parametrize(
    "tokens,mtpr,stage1,stage2",
    [
        (8, 8192, (32, 1, 192, False, 3, 1), (32, 256, 240, False)),
        (256, 8192, (64, 1, 160, True, 3, 4), (32, 128, 240, False)),
        (512, 8192, (64, 1, 64, True, 0, 4), (32, 256, 240, True)),
        (1024, 32768, (64, 1, 64, True, 0, 4), (32, 256, 224, True)),
        (2048, 16384, (64, 1, 64, True, 0, 4), (32, 256, 256, True)),
        (4096, 8192, (128, 1, 64, False, 0, 4), (64, 256, 240, False)),
    ],
)
def test_oversized_capacity_profiles_match_tuned_rules(tokens, mtpr, stage1, stage2):
    config = select_mega_moe_config(tokens, mtpr)
    s1 = config.stage1
    s2 = config.stage2

    assert (
        s1.sort_block_m,
        s1.grid_mult,
        s1.num_dispatch_cu,
        s1.use_tile_resource,
        s1.b_nt,
        s1.work_shards,
    ) == stage1
    assert (s2.block_m, s2.block_n, s2.persist_cu, s2.persist_strided) == stage2
    assert s2.persist


@pytest.mark.parametrize("mtpr", [16384, 32768])
@pytest.mark.parametrize("tokens", [1, 128, 512, 1024, 2048, 4096])
def test_large_capacity_uses_safe_tile_resource_addressing(tokens, mtpr):
    assert select_mega_moe_config(tokens, mtpr).stage1.use_tile_resource


@pytest.mark.parametrize("mtpr", [2048, 4096, 8192, 16384, 32768])
def test_requested_oversized_capacity_matrix_is_valid(mtpr):
    for tokens in (bucket for bucket in TOKEN_BUCKETS if bucket <= mtpr // 2):
        config = select_mega_moe_config(tokens, mtpr)

        assert config.stage2.block_m <= config.stage1.sort_block_m
        assert config.stage1.sort_block_m % config.stage2.block_m == 0
        assert config.p2p_quant == "fp8_blockwise_1x32"


@pytest.mark.parametrize(
    "tokens,mtpr,expected",
    [
        (4, 2048, (128, 8, False, False)),
        (1, 4096, (224, 1, False, False)),
        (64, 4096, (160, 1, False, False)),
        (1, 8192, (224, 1, False, False)),
        (8, 8192, (192, 1, False, False)),
        (64, 8192, (160, 4, False, False)),
        (4, 16384, (224, 8, False, False)),
        (32, 16384, (192, 4, False, False)),
        (16, 32768, (64, 1, False, False)),
        (32, 32768, (64, 2, False, False)),
    ],
)
def test_oversized_small_tail_matches_tuned_rules(tokens, mtpr, expected):
    stage1 = select_mega_moe_config(tokens, mtpr).stage1

    assert (
        stage1.num_dispatch_cu,
        stage1.work_shards,
        stage1.external_grouping,
        stage1.external_counting,
    ) == expected


@pytest.mark.parametrize(
    "mtpr,expected",
    [
        (128, "none"),
        (1024, "none"),
        (2048, "fp8_blockwise_1x32"),
        (8192, "fp8_blockwise_1x32"),
    ],
)
def test_p2p_quant_is_rank_invariant_for_an_mtpr(mtpr, expected):
    configs = [
        select_mega_moe_config(tokens, mtpr)
        for tokens in TOKEN_BUCKETS
        if tokens <= mtpr
    ]

    assert {config.p2p_quant for config in configs} == {expected}


def test_nearby_tokens_share_the_bucket_config():
    assert select_mega_moe_config(500, 512) is select_mega_moe_config(512, 512)


@pytest.mark.parametrize("tokens,mtpr", [(0, 16), (17, 16), (1, 0), (1, 24)])
def test_invalid_shape_is_rejected(tokens, mtpr):
    with pytest.raises(ValueError):
        select_mega_moe_config(tokens, mtpr)
