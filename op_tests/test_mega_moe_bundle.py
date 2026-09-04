# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import pytest
import torch

os.environ.setdefault("AITER_AOT_IMPORT", "1")

from aiter.aot.flydsl.mega_moe import _group_major_geometry, default_jobs
from aiter.ops.flydsl.kernels import tensor_shim
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    TOKEN_BUCKETS,
    Stage2BundleKey,
    Stage2Config,
    build_mega_moe_bundle_plan,
    fixed_stage1_epoch_slot,
    fixed_stage1_epoch_slot_count,
    select_mega_moe_config,
    stage1_bundle_identity,
)


def test_preload_compiled_prefers_native_no_dispatch_api(monkeypatch):
    events = []

    class FakeLaunch:
        def preload(self, *args):
            events.append(("preload", args))
            return "native-artifact"

    monkeypatch.setattr(
        tensor_shim.flyc,
        "compile",
        lambda *_args: pytest.fail("native preload must not call flyc.compile"),
    )

    assert tensor_shim._preload_compiled(FakeLaunch(), 1, 2) == "native-artifact"
    assert events == [("preload", (1, 2))]


def test_zero_token_quantize_does_not_launch_a_kernel(monkeypatch):
    pytest.importorskip("mori.shmem", reason="MegaMoEV2 requires MORI")
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2 import MegaMoEV2

    moe = object.__new__(MegaMoEV2)
    moe._s1_quant_x = torch.empty((1, 32), dtype=torch.uint8)
    moe._s1_quant_scale = torch.empty((1, 1), dtype=torch.uint8)
    x = torch.empty((0, 32), dtype=torch.bfloat16)

    monkeypatch.setattr(
        "aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2.per_1x32_mx_quant",
        lambda *_args, **_kwargs: pytest.fail("zero-token quant must not launch"),
    )

    quant, scale = moe.quantize(x)
    assert quant.shape == (0, 32)
    assert scale.shape == (0, 1)


def test_fixed_stage1_epoch_slot_includes_dispatch_geometry():
    num_cu = 256
    slots = {}
    for tokens in (1, 4, 8, 16, 32, 64, 128):
        stage1 = select_mega_moe_config(tokens, 128).stage1
        geometry = stage1.grid_mult, stage1.num_dispatch_cu
        slot = fixed_stage1_epoch_slot(*geometry, num_cu)
        assert slot < fixed_stage1_epoch_slot_count(num_cu)
        assert slots.setdefault(slot, geometry) == geometry

    one = select_mega_moe_config(1, 128).stage1
    four = select_mega_moe_config(4, 128).stage1
    assert one.grid_mult == four.grid_mult
    assert fixed_stage1_epoch_slot(
        one.grid_mult, one.num_dispatch_cu, num_cu
    ) != fixed_stage1_epoch_slot(four.grid_mult, four.num_dispatch_cu, num_cu)


@pytest.mark.parametrize("slice_output", [False, True])
def test_aligned_pair_stage2_forwards_slice_output(monkeypatch, slice_output):
    pytest.importorskip("mori.shmem", reason="MegaMoEV2 requires MORI")
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2 import MegaMoEV2

    marker = object()
    observed = []

    def fake_candidate(_self, _run_tokens, _config, _stream, **kwargs):
        observed.append(kwargs["slice_output"])
        return marker

    monkeypatch.setattr(MegaMoEV2, "_run_aligned_pair_stage2_candidate", fake_candidate)
    moe = object.__new__(MegaMoEV2)
    config = select_mega_moe_config(8192, 8192)
    assert config.stage2.aligned_pair
    assert moe._run_stage2(8192, None, slice_output, config) is marker
    assert observed == [slice_output]


def test_compact_fanout_accepts_kimi_k3_ep8_expert_count():
    config = select_mega_moe_config(
        4096,
        32768,
        experts_per_rank=112,
        model_dim=3584,
        inter_dim=512,
    )

    assert config.stage1.num_dispatch_cu == 32


def test_compact_fanout_rejects_segment_overflow():
    with pytest.raises(ValueError, match="1032 segments"):
        select_mega_moe_config(4096, 32768, experts_per_rank=128)


def test_compact_fanout_rejects_pair_id_overflow():
    with pytest.raises(ValueError, match="at most 256 experts per rank"):
        select_mega_moe_config(4096, 32768, experts_per_rank=257, world_size=1)


@pytest.mark.parametrize("experts_per_rank", [56, 112])
def test_non_reference_expert_profiles_use_compact_small_mtpr(experts_per_rank):
    plan = build_mega_moe_bundle_plan(
        128,
        experts_per_rank=experts_per_rank,
        model_dim=3584,
        inter_dim=512,
    )

    assert not plan.fixed_slot_dispatch
    assert all(not key.fixed_slot_dispatch for key in plan.stage2_variants)


@pytest.mark.parametrize("old_value", [None, "0"])
def test_preload_compiled_legacy_fallback_never_dispatches(monkeypatch, old_value):
    launch = object()
    if old_value is None:
        monkeypatch.delenv("COMPILE_ONLY", raising=False)
    else:
        monkeypatch.setenv("COMPILE_ONLY", old_value)

    def compile_only(exe, *args):
        assert exe is launch
        assert args == (3, 4)
        assert os.environ["COMPILE_ONLY"] == "1"
        return "legacy-artifact"

    monkeypatch.setattr(tensor_shim.flyc, "compile", compile_only)

    assert tensor_shim._preload_compiled(launch, 3, 4) == "legacy-artifact"
    assert os.environ.get("COMPILE_ONLY") == old_value


def test_mtpr8192_bundle_deduplicates_expected_variants():
    plan = build_mega_moe_bundle_plan(8192)

    assert len(plan.entries) == 13
    assert len(plan.stage1_variants) == 8
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


def test_aot_jobs_can_describe_kimi_k3_ep8_profile():
    jobs = default_jobs(
        (8192,),
        (112,),
        world_size=8,
        topk=16,
        model_dim=3584,
        inter_dim=512,
    )

    assert len(jobs) == 8 * 2
    assert {job["topk"] for job in jobs} == {16}
    assert {job["experts_per_rank"] for job in jobs} == {112}
    assert all("_w8_k16_d3584_i512" in job["kernel_name"] for job in jobs)


def test_aot_geometry_matches_small_mtpr_runtime_layouts():
    assert _group_major_geometry(
        1,
        112,
        world_size=8,
        topk=16,
        fixed_slot_dispatch=False,
    ) == (128, 14720, 460)
    assert _group_major_geometry(
        128,
        48,
        world_size=8,
        topk=6,
        fixed_slot_dispatch=True,
    ) == (1024, 49408, 1544)


def test_bundle_selection_matches_production_config_for_every_token():
    plan = build_mega_moe_bundle_plan(8192)

    for tokens in range(1, 8193):
        entry = plan.entry_for_tokens(tokens)
        assert entry.config == select_mega_moe_config(tokens, 8192)
        assert stage1_bundle_identity(
            plan.stage1_variants[entry.stage1_variant_id]
        ) == stage1_bundle_identity(entry.config.stage1)
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
        assert stage1_bundle_identity(
            plan.stage1_variants[entry.stage1_variant_id]
        ) == stage1_bundle_identity(expected.stage1)
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


@pytest.mark.parametrize("mtpr", [8192, 16384, 32768])
def test_stage1_bundle_deduplicates_prepare_only_variants(mtpr):
    plan = build_mega_moe_bundle_plan(mtpr)
    identities = tuple(stage1_bundle_identity(v) for v in plan.stage1_variants)

    assert len(identities) == len(set(identities))
    for entry in plan.entries:
        variant = plan.stage1_variants[entry.stage1_variant_id]
        assert stage1_bundle_identity(variant) == stage1_bundle_identity(
            entry.config.stage1
        )


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


def test_empty_rank_uses_smallest_collective_bundle_entry():
    plan = build_mega_moe_bundle_plan(8192)

    assert plan.entry_for_tokens(0) == plan.entries[0]


@pytest.mark.parametrize("tokens", [-1, 8193])
def test_bundle_rejects_out_of_range_tokens(tokens):
    with pytest.raises(ValueError, match="must be in"):
        build_mega_moe_bundle_plan(8192).entry_for_tokens(tokens)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
