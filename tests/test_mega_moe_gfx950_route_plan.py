# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 on gfx950: route selection, one plan picks the route of every call -- no GPU needed.

Every EP rank must take the same route for a call, so the plan's choice may
depend only on EP-wide inputs (the plan and the call's max_tokens). The
multi-GPU behaviour of each route (EP8 DeepSeek-V4-Pro, EP4 DeepSeek-V4.1-Flash)
lives in op_tests/multigpu_tests/test_mega_moe_v2.py: --suite size for the
pinned staged and large-input routes, --suite routes for the instance that
serves all three.
"""

import pytest

pytest.importorskip("flydsl", reason="MegaMoE is built with FlyDSL")
pytest.importorskip("mori.shmem", reason="MegaMoE's runtime imports mori")

import hashlib  # noqa: E402
from dataclasses import astuple, fields  # noqa: E402

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (  # noqa: E402
    DEFAULT_SMALL_M_MTPRS,
    FUSED_TOKEN_BUCKETS,
    LARGE_M_MIN_TOKENS,
    LARGE_M_MTPRS,
    ROUTE_LARGE_M,
    ROUTE_SMALL_M,
    ROUTE_STAGED,
    TOKEN_BUCKETS,
    FusedStage1Knobs,
    FusedStage2Knobs,
    build_mega_moe_bundle_plan,
    fused_layout_supported,
    large_m_mtprs_for,
    select_fused_config,
    small_m_cu_supported,
    small_m_mtprs_for,
)

V4_PRO = dict(
    arch="gfx950:sramecc+:xnack-",
    world_size=8,
    model_dim=7168,
    inter_dim=3072,
    experts=384,
    topk=6,
    quant="a8w4",
    swiglu_limit=10.0,
)
# EP4 DeepSeek-V4.1-Flash: 96 experts a rank.
V41_FLASH_EP4 = dict(V4_PRO, world_size=4, model_dim=5120, inter_dim=2304, experts=384)
V41_FLASH_EP4_SHAPE = dict(world_size=4, experts_per_rank=96, model_dim=5120, inter_dim=2304)
MTPRS = tuple(b for b in TOKEN_BUCKETS if b >= 16)


def _plan(mtpr, *, small=DEFAULT_SMALL_M_MTPRS, large=None):
    return build_mega_moe_bundle_plan(
        mtpr,
        small_m_mtprs=small,
        large_m=mtpr in LARGE_M_MTPRS if large is None else large,
    )


def test_fused_layout_gate():
    assert fused_layout_supported(**V4_PRO)
    assert fused_layout_supported(**V41_FLASH_EP4)
    # Each layout is supported whole: no mix of the two shapes.
    assert not fused_layout_supported(**{**V41_FLASH_EP4, "world_size": 8})
    assert not fused_layout_supported(**{**V41_FLASH_EP4, "experts": 768})
    assert not fused_layout_supported(**{**V41_FLASH_EP4, "inter_dim": 3072})
    for field, value in (
        ("arch", "gfx942"),
        ("world_size", 4),
        ("model_dim", 6144),
        ("inter_dim", 2048),
        ("experts", 416),
        ("topk", 8),
        ("quant", "a4w4"),
        ("swiglu_limit", 0.0),
    ):
        assert not fused_layout_supported(**{**V4_PRO, field: value}), field
    assert not fused_layout_supported(**V4_PRO, act="situv2")
    assert not fused_layout_supported(**V4_PRO, fanout_masks=(3,) * 8)
    assert small_m_cu_supported(256) and not small_m_cu_supported(128)


def test_small_m_capacities_pruned_to_reachable():
    assert small_m_mtprs_for(8192) == (256, 512)
    assert small_m_mtprs_for(512) == (256, 512)
    assert small_m_mtprs_for(256) == (256,)
    assert small_m_mtprs_for(128) == (256,)
    assert small_m_mtprs_for(128, (128, 256, 512)) == (128,)
    with pytest.raises(ValueError):
        small_m_mtprs_for(8192, (1024,))
    with pytest.raises(ValueError):
        small_m_mtprs_for(8192, (384,))


@pytest.mark.parametrize("mtpr", MTPRS)
def test_route_for_every_max_tokens(mtpr):
    plan = _plan(mtpr)
    small = plan.small_m_mtprs
    for max_tokens in sorted(
        {0, 1, 2, 3, 255, 256, 257, 511, 512, 513, 1023, 1024, mtpr - 1, mtpr}
    ):
        if not 0 <= max_tokens <= mtpr:
            continue
        route, capacity = plan.route_for(max_tokens)
        if small and max_tokens <= small[-1]:
            assert route == ROUTE_SMALL_M
            assert capacity == min(m for m in small if m >= max_tokens)
        elif plan.large_m and max_tokens >= LARGE_M_MIN_TOKENS:
            assert (route, capacity) == (ROUTE_LARGE_M, mtpr)
        else:
            assert (route, capacity) == (ROUTE_STAGED, mtpr)
    with pytest.raises(ValueError):
        plan.route_for(mtpr + 1)
    with pytest.raises(ValueError):
        plan.route_for(-1)


def test_route_exclusions():
    plan = _plan(8192)
    assert plan.route_for(100) == (ROUTE_SMALL_M, 256)
    assert plan.route_for(100, small_m=False) == (ROUTE_STAGED, 8192)
    assert plan.route_for(4096) == (ROUTE_LARGE_M, 8192)
    assert plan.route_for(4096, large_m=False) == (ROUTE_STAGED, 8192)
    # MegaMoEV2 itself takes the staged route when a call passes no max_tokens.
    assert plan.route_for(plan.mtpr) == (ROUTE_LARGE_M, 8192)
    staged_only = _plan(8192, small=(), large=False)
    assert all(
        staged_only.route_for(t) == (ROUTE_STAGED, 8192) for t in (0, 1, 512, 8192)
    )


def test_large_m_capacities_per_layout():
    shape = dict(world_size=8, model_dim=7168, inter_dim=3072)
    assert large_m_mtprs_for(**shape) == LARGE_M_MTPRS
    # EP4 V4.1-Flash: the large-input kernel only at the MTPR its tune was measured at.
    assert large_m_mtprs_for(world_size=4, model_dim=5120, inter_dim=2304) == (8192,)
    select_fused_config(8192, 8192, **V41_FLASH_EP4_SHAPE)
    for mtpr in (1024, 2048, 4096, 16384):
        with pytest.raises(ValueError):
            select_fused_config(mtpr, mtpr, **V41_FLASH_EP4_SHAPE)


def test_large_m_capacity_validated():
    for mtpr in MTPRS:
        if mtpr in LARGE_M_MTPRS:
            assert _plan(mtpr).large_m
        else:
            with pytest.raises(ValueError):
                _plan(mtpr, large=True)


# Every fused launch config, as recorded before the fused selector moved into
# mega_moe_config: the move must not change one of them. On a mismatch the
# assertion shows the current table.
_FUSED_CASES = (
    (256, 0),
    (256, 1),
    (256, 2),
    (256, 3),
    (256, 5),
    (256, 16),
    (256, 17),
    (256, 100),
    (256, 128),
    (256, 200),
    (256, 255),
    (256, 256),
    (512, 0),
    (512, 1),
    (512, 2),
    (512, 3),
    (512, 5),
    (512, 16),
    (512, 17),
    (512, 100),
    (512, 128),
    (512, 200),
    (512, 255),
    (512, 256),
    (512, 300),
    (512, 511),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
)
_FUSED_DIGEST = "f8cb680aa7ec3e668f98ed9a71a485c0bdbe703735cbf6477622cb4ae36b0666"


def test_fused_configs_unchanged():
    table = {}
    for mtpr, tokens in _FUSED_CASES:
        config = select_fused_config(tokens, mtpr)
        table[(mtpr, tokens)] = (
            astuple(config.stage1),
            astuple(config.stage2),
            config.p2p_quant,
        )
    digest = hashlib.sha256(repr(sorted(table.items())).encode()).hexdigest()
    assert digest == _FUSED_DIGEST, table


@pytest.mark.parametrize("mtpr", MTPRS)
def test_no_hint_takes_staged(mtpr):
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2 import MegaMoEV2

    moe = object.__new__(MegaMoEV2)  # _route needs only the plan and the capacity
    moe.mtpr, moe._bundle_plan, moe._large_m_choice = mtpr, _plan(mtpr), None
    for rows in sorted({0, 1, min(128, mtpr), mtpr}):
        assert moe._route(rows, None) == (ROUTE_STAGED, mtpr)
        assert moe._route(rows, None, small_m=False) == (ROUTE_STAGED, mtpr)


def test_set_large_m_enabled_keeps_mains_interface():
    """Callers of main's interface (SGLang) switch the large-input route per call."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2 import MegaMoEV2

    moe = object.__new__(MegaMoEV2)
    moe.mtpr, moe._bundle_plan, moe._large_m_choice = 8192, _plan(8192), None
    moe._large_m = object()
    assert moe._route(7, None) == (ROUTE_STAGED, 8192)
    moe.set_large_m_enabled(True)
    assert moe._route(7, None) == (ROUTE_LARGE_M, 8192)
    assert moe._route(7, None, small_m=False) == (ROUTE_LARGE_M, 8192)
    assert moe._route(7, 300) == (ROUTE_SMALL_M, 512)
    moe.set_large_m_enabled(False)
    assert moe._route(7, None) == (ROUTE_STAGED, 8192)
    assert moe._route(4096, 4096) == (ROUTE_STAGED, 8192)


def test_ep8_fused_configs_keep_fused_knobs_off():
    """The EP8 V4-Pro fused kernels are the ones before the EP4 tune: every fused knob off."""
    for mtpr, tokens in _FUSED_CASES:
        config = select_fused_config(tokens, mtpr)
        assert config.fused_stage1 == FusedStage1Knobs(), (mtpr, tokens)
        assert config.fused_stage2 == FusedStage2Knobs(), (mtpr, tokens)


def _knob_table(config):
    return (
        astuple(config.stage1),
        astuple(config.stage2),
        config.p2p_quant,
        tuple((f.name, getattr(config.fused_stage1, f.name)) for f in fields(FusedStage1Knobs)),
        tuple((f.name, getattr(config.fused_stage2, f.name)) for f in fields(FusedStage2Knobs)),
    )


# The EP4 V4.1-Flash fused configs, as ported from 28ee7fcd (its _tune_v41_flash_ep4 on the post-processed
# small-input configs and on the large-input config at MTPR 8192). On a mismatch the assertion shows the
# current table.
_EP4_FUSED_DIGEST = "46b5425e10dcb8ecf68983dee099df3fdb83a2ec4c0008bf44c4e083dafab45e"


def test_ep4_fused_configs_unchanged():
    table = {}
    for mtpr, tokens in _FUSED_CASES:
        if mtpr in (256, 512, 8192):
            table[(mtpr, tokens)] = _knob_table(select_fused_config(tokens, mtpr, **V41_FLASH_EP4_SHAPE))
    digest = hashlib.sha256(repr(sorted(table.items())).encode()).hexdigest()
    assert digest == _EP4_FUSED_DIGEST, table


@pytest.mark.parametrize("mtpr", (256, 512))
def test_ep4_small_m_buckets_share_one_protocol(mtpr):
    """The ranks of a launch may pick different buckets: the knobs that change what ranks and launches
    exchange are the same in every bucket (SmallMRuntime asserts it too)."""
    protocols = set()
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        k = select_fused_config(bucket, mtpr, **V41_FLASH_EP4_SHAPE).fused_stage1
        protocols.add((k.dedup_rows, k.plan_local, bool(k.direct_done), k.tok_counters, k.bf16_head_tokens))
        # tok_done adds to the per-token counters every bucket clears.
        assert not k.tok_done or k.tok_counters
        # The runtime's tok_counters also double-buffer the expert counters, which the plan_local combine zeroes.
        assert not k.tok_counters or k.plan_local
        # ... and the srcmap / weight slots, so the payload must go to the source-indexed dedup token rows.
        assert not k.tok_counters or k.dedup_rows
    assert protocols == {(True, True, True, True, 4)}
    # A tok_done bucket (1 and 4 tokens) and one without it run on the same counters.
    assert select_fused_config(1, mtpr, **V41_FLASH_EP4_SHAPE).fused_stage1.tok_done
    assert not select_fused_config(16, mtpr, **V41_FLASH_EP4_SHAPE).fused_stage1.tok_done
    # A tok_done reducer waits for valid routes x its own GEMM2 n-blocks, which every tok_done sender counts.
    tok_done_block_n = {
        config.stage2.block_n
        for config in (select_fused_config(b, mtpr, **V41_FLASH_EP4_SHAPE) for b in FUSED_TOKEN_BUCKETS if b <= mtpr)
        if config.fused_stage1.tok_done
    }
    assert tok_done_block_n == {256}


def test_ep4_dispatch_cus_cover_two_expert_waves():
    """96 experts a rank are two 64-expert waves: twice the EP8 producers, capped at 224."""
    for tokens in (1, 4, 8, 16, 32, 64, 128, 256, 512):
        ep8 = select_fused_config(tokens, 512).stage1.num_dispatch_cu
        ep4 = select_fused_config(tokens, 512, **V41_FLASH_EP4_SHAPE).stage1.num_dispatch_cu
        assert ep4 == min(224, 2 * ep8), tokens
    assert select_fused_config(8192, 8192, **V41_FLASH_EP4_SHAPE).stage1.num_dispatch_cu == 192


@pytest.mark.parametrize("mtpr", (256, 512))
def test_small_m_token_rows_live_in_the_payload_arena(mtpr):
    """dedup_rows token rows sit past the fixed slots of the payload arena, inside u32 buffer offsets."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import (
        _TOKEN_REGION_END,
        SmallMRuntime,
    )

    def rows(world_size, model_dim, inter_dim):
        return SmallMRuntime.payload_capacity(
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=384,
            topk=6,
            max_tok_per_rank=mtpr,
        )

    ep8 = rows(8, 7168, 3072)
    ep4 = rows(4, 5120, 2304)
    # EP8 keeps the plain fixed-slot payload; EP4 adds one row per (source rank, token).
    assert ep8[1] == 48 * (8 * mtpr) + 256
    assert ep4[1] == 96 * (4 * mtpr) + 256 + 4 * mtpr
    assert ep4[1] * ep4[2] <= _TOKEN_REGION_END


def test_route_payload_sharing():
    """The EP8 fused runtimes end with an all-rank barrier, so they share the instance's payload arena; the EP4
    small- (tok_done) and large-input (comb_tok_wait) runtimes can end before their peers, so they keep their own."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import LargeMRuntime
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import SmallMRuntime

    ep8 = dict(world_size=8, model_dim=7168, inter_dim=3072, experts=384, topk=6)
    ep4 = dict(world_size=4, model_dim=5120, inter_dim=2304, experts=384, topk=6)
    for mtpr in DEFAULT_SMALL_M_MTPRS:
        assert SmallMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8)
        assert not SmallMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep4)
    for mtpr in large_m_mtprs_for(world_size=8, model_dim=7168, inter_dim=3072):
        assert LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8)
    for mtpr in large_m_mtprs_for(world_size=4, model_dim=5120, inter_dim=2304):
        assert not LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep4)
