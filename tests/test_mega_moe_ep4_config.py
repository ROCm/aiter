"""EP4 dispatch policy and the wire invariants of its mixed token buckets."""

import pytest

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    build_mega_moe_bundle_plan,
    select_mega_moe_config,
)

FLASH_EP4 = {
    "experts_per_rank": 96,
    "model_dim": 5120,
    "inter_dim": 2304,
    "world_size": 4,
}


@pytest.mark.parametrize("tokens", [192, 256, 383])
def test_atom_ep4_reduces_padding_in_the_256_bucket(tokens):
    config = select_mega_moe_config(tokens, 16384, **FLASH_EP4)
    assert config.stage1.sort_block_m == 32
    assert config.stage2.block_m == 32


@pytest.mark.parametrize("tokens", [384, 512, 1024])
def test_neighboring_buckets_keep_wider_tiles(tokens):
    config = select_mega_moe_config(tokens, 16384, **FLASH_EP4)
    assert config.stage1.sort_block_m == 64


@pytest.mark.parametrize(
    "capacity, overrides",
    [
        (256, {}),
        (8192, {}),
        (32768, {}),
        (16384, {"a_dtype": "fp4"}),
        (16384, {"world_size": 2}),
        (16384, {"experts_per_rank": 97}),
        (16384, {"model_dim": 4096}),
    ],
)
def test_unmeasured_capacities_and_profiles_keep_their_tiles(capacity, overrides):
    config = select_mega_moe_config(256, capacity, **(FLASH_EP4 | overrides))
    assert config.stage1.sort_block_m == 64


def test_different_token_buckets_keep_one_wire_protocol():
    plan = build_mega_moe_bundle_plan(16384, **FLASH_EP4)
    assert {entry.config.p2p_quant for entry in plan.entries} == {"fp8_blockwise_1x32"}
    assert {entry.config.stage1.payload_chunk_rows for entry in plan.entries} == {1536}
    for entry in plan.entries:
        stage2 = plan.stage2_variants[entry.stage2_variant_id]
        assert stage2.sbm == entry.config.stage1.sort_block_m
        assert stage2.sbm % stage2.config.block_m == 0
    assert plan.entry_for_tokens(0).token_bucket == 1
