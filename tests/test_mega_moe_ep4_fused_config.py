"""Fused-route tile selection and the protocol shared by uneven EP ranks."""

from dataclasses import replace

import pytest

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    FUSED_TOKEN_BUCKETS,
    select_fused_config,
    tune_large_m_launch_config,
)


FLASH_EP4 = dict(world_size=4, experts_per_rank=96, model_dim=5120, inter_dim=2304)


@pytest.mark.parametrize("capacity, tokens", [(256, 192), (256, 256), (512, 256), (512, 383)])
def test_ep4_256_bucket_pairs_the_smaller_row_tile_in_both_gemms(capacity, tokens):
    config = select_fused_config(tokens, capacity, **FLASH_EP4)
    assert (config.stage1.sort_block_m, config.stage1.tile_n) == (32, 256)
    assert config.stage2.block_m == 32


@pytest.mark.parametrize("tokens", [384, 512])
def test_512_bucket_keeps_its_wider_row_tile(tokens):
    config = select_fused_config(tokens, 512, **FLASH_EP4)
    assert (config.stage1.sort_block_m, config.stage1.tile_n) == (64, 128)
    assert config.stage2.block_m == 64


def test_ep8_profile_and_large_input_routes_keep_their_tiles():
    ep8 = select_fused_config(256, 512)
    assert (ep8.stage1.sort_block_m, ep8.stage2.block_m) == (64, 64)
    for capacity in (8192, 16384):
        large = select_fused_config(256, capacity, **FLASH_EP4)
        assert (large.stage1.sort_block_m, large.stage2.block_m) == (128, 64)


@pytest.mark.parametrize("capacity", [256, 512])
def test_mixed_token_buckets_keep_one_small_input_wire_protocol(capacity):
    configs = [
        select_fused_config(tokens, capacity, **FLASH_EP4)
        for tokens in FUSED_TOKEN_BUCKETS
        if tokens <= capacity
    ]
    protocols = {
        (k.dedup_rows, k.plan_local, bool(k.direct_done), k.tok_counters, k.bf16_head_tokens)
        for k in (config.fused_stage1 for config in configs)
    }
    assert len(protocols) == 1
    assert all(config.stage1.sort_block_m == config.stage2.block_m for config in configs)


@pytest.mark.parametrize("capacity", [8192, 16384])
def test_large_prefill_local_tune_preserves_the_complete_wire_config(capacity):
    base = select_fused_config(capacity, capacity, **FLASH_EP4)
    for tokens in (0, 384, 4096, 8191, 8192, capacity):
        config = tune_large_m_launch_config(base, tokens, **FLASH_EP4)
        if tokens < 8192:
            assert config == base
        else:
            assert (config.stage1.b_nt, config.fused_stage2.m_group) == (0, 16)
            restored = replace(
                config,
                stage1=replace(config.stage1, b_nt=base.stage1.b_nt),
                fused_stage2=replace(config.fused_stage2, m_group=base.fused_stage2.m_group),
            )
            assert restored == base


def test_large_prefill_local_tune_leaves_other_layouts_unchanged():
    layout = dict(FLASH_EP4, world_size=8, experts_per_rank=48)
    base = select_fused_config(16384, 16384, **layout)
    assert tune_large_m_launch_config(base, 16384, **layout) is base
