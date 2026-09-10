# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

from aiter.aot.flydsl.mega_moe import _combine_aot_identities, default_aot_jobs
from aiter.ops.flydsl.kernels.communication_ops_utils import GeometryTuningTable
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    build_mega_moe_bundle_plan,
    select_mega_moe_config,
)


def test_glm52_ep8_decode_specialization():
    expected_stage1 = {
        1: (224, 256, 4),
        4: (128, 512, 8),
        6: (192, 512, 8),
        12: (128, 512, 8),
        24: (128, 512, 8),
        48: (160, 512, 8),
        96: (96, 512, 8),
        256: (64, 512, 8),
    }
    for tokens, expected in expected_stage1.items():
        config = select_mega_moe_config(
            tokens,
            256,
            experts_per_rank=32,
            model_dim=6144,
            inter_dim=2048,
            topk=8,
        )
        assert (
            config.stage1.num_dispatch_cu,
            config.stage1.tile_n,
            config.stage1.num_waves,
        ) == expected
        assert config.stage2.block_n == 128
        assert config.stage2.persist
        assert config.stage2.persist_cu == (192 if tokens <= 128 else 128)


def test_glm52_policy_is_shape_ep_and_topk_specific():
    ep4 = select_mega_moe_config(
        24,
        256,
        experts_per_rank=64,
        model_dim=6144,
        inter_dim=2048,
        world_size=4,
        topk=8,
    )
    other_shape = select_mega_moe_config(
        24,
        256,
        experts_per_rank=32,
        model_dim=7168,
        inter_dim=3072,
        topk=8,
    )
    other_mtpr = select_mega_moe_config(
        24,
        512,
        experts_per_rank=32,
        model_dim=6144,
        inter_dim=2048,
        topk=8,
    )
    other_topk = select_mega_moe_config(
        24,
        256,
        experts_per_rank=32,
        model_dim=6144,
        inter_dim=2048,
        topk=6,
    )
    assert ep4.stage1.num_dispatch_cu == 32
    assert other_shape.stage1.num_dispatch_cu == 32
    assert other_mtpr.stage1.num_dispatch_cu == 32
    assert other_topk.stage1.num_dispatch_cu == 32
    assert ep4.stage2.block_n == 256
    assert other_shape.stage2.block_n == 256
    assert other_mtpr.stage2.block_n == 256
    assert other_topk.stage2.block_n == 256


def test_glm52_bundle_supports_zero_token_rank():
    plan = build_mega_moe_bundle_plan(
        256,
        experts_per_rank=32,
        model_dim=6144,
        inter_dim=2048,
        topk=8,
    )
    assert not plan.fixed_slot_dispatch
    assert plan.entry_for_tokens(0) == plan.entry_for_tokens(1)
    assert plan.entry_for_tokens(96).config.stage2.persist_cu == 192
    assert plan.entry_for_tokens(256).config.stage1.num_dispatch_cu == 64


def test_glm52_ep8_combine_geometry_table():
    root = Path(__file__).resolve().parents[2]
    table = GeometryTuningTable.from_tuning_file(
        root / "aiter/configs/tuned_dispatch_combine_intranode.csv",
        ep_size=8,
        gfx="gfx950",
        gpu_model="mi355x",
        dtype="fp8_ocp",
        hidden_dim=6144,
        zero_copy=False,
        topk=8,
        local_expert_num=32,
        combine_dtype="bf16",
    )
    assert table.dispatch == {}
    assert table.lookup("combine", 1) == (64, 4)
    assert table.lookup("combine", 2) == (96, 4)
    assert table.lookup("combine", 6) == (64, 8)
    assert table.lookup("combine", 48) == (96, 16)
    assert table.lookup("combine", 96) == (96, 16)
    plan = build_mega_moe_bundle_plan(
        256,
        experts_per_rank=32,
        model_dim=6144,
        inter_dim=2048,
        topk=8,
    )
    assert set(_combine_aot_identities(plan, table, 256)) == {
        (64, 4, False),
        (96, 4, False),
        (64, 8, False),
        (96, 8, False),
        (96, 16, False),
        (128, 16, False),
    }


def test_glm52_ep8_is_in_default_aot_jobs():
    all_jobs = default_aot_jobs()
    assert len({job["kernel_name"] for job in all_jobs}) == len(all_jobs)
    jobs = [
        job
        for job in all_jobs
        if (
            job["mtpr"],
            job["experts_per_rank"],
            job["world_size"],
            job["topk"],
            job["model_dim"],
            job["inter_dim"],
            job["swiglu_limit"],
        )
        == (256, 32, 8, 8, 6144, 2048, 0.0)
    ]
    assert len(jobs) == 16
    assert {job["stage"] for job in jobs} == {1, 2}
    assert {job["rank"] for job in jobs} == set(range(8))
    assert all("_w8_k8_d6144_i2048" in job["kernel_name"] for job in jobs)
