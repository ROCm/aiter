# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import pytest

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import nearest_token_bucket, select_mega_moe_config

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
    assert not stage1.active_expert_producer and not stage1.cooperative_payload_copy
    assert stage2.use_nt == (tokens <= 128)
    assert stage2.b_hoist and stage2.ascale_prefetch
    assert stage2.spatial_partition == 402 and not stage2.bf16_lds


@pytest.mark.parametrize(
    "tokens,bucket",
    [(2, 1), (3, 4), (6, 8), (16300, 16384), (16400, 16384), (24576, 32768), (65536, 32768)],
)
def test_nearest_token_bucket_prefers_larger_on_ties(tokens, bucket):
    assert nearest_token_bucket(tokens) == bucket


def test_mtpr_selects_fixed_or_compact_configs():
    fixed = select_mega_moe_config(128, 128)
    compact = select_mega_moe_config(128, 8192)

    assert (fixed.stage1.tile_n, fixed.stage1.num_waves, fixed.stage1.num_dispatch_cu) == (128, 4, 224)
    assert (compact.stage1.tile_n, compact.stage1.num_waves, compact.stage1.num_dispatch_cu) == (512, 8, 128)
    for tokens in (8, 16, 32):
        assert select_mega_moe_config(tokens, 128).stage2.block_n == 128
        assert select_mega_moe_config(tokens, 8192).stage2.block_n == 256


def test_nearby_tokens_share_the_bucket_config():
    assert select_mega_moe_config(500, 512) is select_mega_moe_config(512, 512)


@pytest.mark.parametrize("tokens,mtpr", [(0, 16), (17, 16), (1, 0), (1, 24)])
def test_invalid_shape_is_rejected(tokens, mtpr):
    with pytest.raises(ValueError):
        select_mega_moe_config(tokens, mtpr)
