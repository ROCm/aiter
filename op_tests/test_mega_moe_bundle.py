# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import pytest

os.environ.setdefault("AITER_AOT_IMPORT", "1")

from aiter.aot.flydsl.mega_moe import default_jobs
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    TOKEN_BUCKETS,
    Stage2BundleKey,
    Stage2Config,
    build_mega_moe_bundle_plan,
    select_mega_moe_config,
)


def test_mtpr8192_bundle_deduplicates_expected_variants():
    plan = build_mega_moe_bundle_plan(8192)

    assert len(plan.entries) == 13
    assert len(plan.stage1_variants) == 11
    assert len(plan.stage2_variants) == 6
    assert [entry.pair_id for entry in plan.entries] == list(range(13))


def test_aot_jobs_cover_all_large_mtpr_profiles_ranks_and_stages():
    jobs = default_jobs()
    identities = {
        (job["mtpr"], job["experts_per_rank"], job["rank"], job["stage"])
        for job in jobs
    }
    assert len(jobs) == len(identities) == 3 * 3 * 8 * 2
    assert {job["experts_per_rank"] for job in jobs} == {48, 52, 56}


def test_aot_jobs_can_cover_r0_r32_r64_expert_profiles():
    jobs = default_jobs((8192,), (48, 52, 56))
    identities = {(job["experts_per_rank"], job["rank"], job["stage"]) for job in jobs}

    assert len(jobs) == len(identities) == 3 * 8 * 2
    assert {job["experts_per_rank"] for job in jobs} == {48, 52, 56}


def test_bundle_selection_matches_production_config_for_every_token():
    plan = build_mega_moe_bundle_plan(8192)

    for tokens in range(1, 8193):
        entry = plan.entry_for_tokens(tokens)
        assert entry.config == select_mega_moe_config(tokens, 8192)
        assert plan.stage1_variants[entry.stage1_variant_id] == entry.config.stage1
        stage2_key = plan.stage2_variants[entry.stage2_variant_id]
        assert stage2_key.config == entry.config.stage2
        assert stage2_key.sbm == entry.config.stage1.sort_block_m
        assert stage2_key.p2p_quant == entry.config.p2p_quant


def test_large_mtpr_profiles_share_configs_for_common_buckets():
    plans = [build_mega_moe_bundle_plan(mtpr) for mtpr in (8192, 16384, 32768)]

    for tokens in range(1, 8193):
        configs = [plan.entry_for_tokens(tokens).config for plan in plans]
        assert configs[1:] == configs[:-1]


def test_role_retirement_is_not_a_configurable_stage1_variant():
    plan = build_mega_moe_bundle_plan(8192)

    for bucket in TOKEN_BUCKETS:
        if bucket > 8192:
            continue
        stage1 = plan.entry_for_tokens(bucket).config.stage1
        assert not hasattr(stage1, "retire_control_ctas")
        assert stage1.payload_tile_ready


def test_default_prefill_keeps_payload_deduplication_disabled():
    plan = build_mega_moe_bundle_plan(8192)
    rank_tokens = (1, 8, 32, 128, 256, 512, 4096, 8192)
    configs = tuple(
        plan.entry_for_tokens(tokens).config.stage1 for tokens in rank_tokens
    )

    assert all(not config.deduplicate_payload for config in configs)


@pytest.mark.parametrize("mtpr", [8192, 16384, 32768])
@pytest.mark.parametrize("experts_per_rank", [48, 52, 56])
def test_every_deployment_bucket_maps_to_its_exact_production_pair(
    mtpr, experts_per_rank
):
    plan = build_mega_moe_bundle_plan(mtpr, experts_per_rank=experts_per_rank)

    for bucket in (value for value in TOKEN_BUCKETS if value <= mtpr):
        entry = plan.entry_for_tokens(bucket)
        expected = select_mega_moe_config(
            bucket, mtpr, experts_per_rank=experts_per_rank
        )
        assert entry.config == expected
        assert plan.stage1_variants[entry.stage1_variant_id] == expected.stage1
        assert plan.stage2_variants[entry.stage2_variant_id] == Stage2BundleKey(
            expected.stage2,
            expected.stage1.sort_block_m,
            expected.p2p_quant,
            False,
        )


@pytest.mark.parametrize(
    ("mtpr", "entry_count"), [(8192, 13), (16384, 14), (32768, 15)]
)
def test_large_mtpr_profile_covers_every_bucket(mtpr, entry_count):
    plan = build_mega_moe_bundle_plan(mtpr)
    assert len(plan.entries) == entry_count
    assert plan.entries[-1].token_bucket == mtpr


def test_stage2_bundle_identity_includes_stage1_sbm():
    config = Stage2Config(
        block_m=32,
        block_n=256,
        persist=True,
        persist_cu=240,
        use_nt=False,
    )

    key64 = Stage2BundleKey(config, 64, "fp8_blockwise_1x32", False)
    key128 = Stage2BundleKey(config, 128, "fp8_blockwise_1x32", False)
    assert key64 != key128


def test_stage2_bundle_rejects_incompatible_sbm():
    config = Stage2Config(
        block_m=64,
        block_n=256,
        persist=True,
        persist_cu=240,
        use_nt=False,
    )

    with pytest.raises(ValueError, match="must divide bundle SBM"):
        Stage2BundleKey(config, 32, "fp8_blockwise_1x32", False)


def test_small_mtpr_bundle_keeps_stage1_and_stage2_in_fixed_slot_mode():
    plan = build_mega_moe_bundle_plan(128)

    assert plan.fixed_slot_dispatch
    assert all(key.fixed_slot_dispatch for key in plan.stage2_variants)


def test_large_mtpr_bundle_keeps_stage1_and_stage2_in_compact_mode():
    plan = build_mega_moe_bundle_plan(8192)

    assert not plan.fixed_slot_dispatch
    assert all(not key.fixed_slot_dispatch for key in plan.stage2_variants)


@pytest.mark.parametrize("tokens", [0, 8193])
def test_bundle_rejects_out_of_range_tokens(tokens):
    with pytest.raises(ValueError, match="must be in"):
        build_mega_moe_bundle_plan(8192).entry_for_tokens(tokens)
