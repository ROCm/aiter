# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP8 DeepSeek-V4.1-Flash DSpark draft MoE (16 experts a rank, topk 3) on the small-input route -- no GPU needed.

The layout is a fused layout without a large-input kernel: calls up to SMALL_M_MAX_MTPR rows a rank take the
small-input route, larger ones the staged route. Every route returns BF16 rows (BF16_RETURN_LAYOUTS): the
small-input configs are the 48-expert EP8 tune with a BF16 GEMM2 return on every rank, and the staged route's
p2p_quant "auto" resolves to "none" at every capacity. The draft MoE on 1, 2 or 4 ranks (128 / 64 / 32 experts a
rank) has no fused route; its staged route returns BF16 rows the same way. The target-model layouts keep their
returns. The multi-GPU check (outputs against the staged route and the fp32 reference) is
op_tests/multigpu_tests/test_mega_moe_v2.py --suite routes --network v41_flash_draft --staged-rtol 0.005 on 8
ranks (both routes) or 4 / 2 ranks (the staged route).
"""

from contextlib import contextmanager
from dataclasses import fields, replace

import pytest

pytest.importorskip("flydsl", reason="MegaMoE is built with FlyDSL")

import aiter.ops.flydsl.kernels.mega_moe.mega_moe_config as cfg
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    FUSED_TOKEN_BUCKETS,
    LARGE_M_MIN_TOKENS,
    LARGE_M_MTPRS,
    P2P_FP8_MIN_MTPR,
    ROUTE_SMALL_M,
    ROUTE_STAGED,
    SMALL_M_MAX_MTPR,
    TOKEN_BUCKETS,
    build_mega_moe_bundle_plan,
    fused_layout_supported,
    large_m_mtprs_for,
    select_fused_config,
    select_mega_moe_config,
)

DRAFT = {
    "arch": "gfx950:sramecc+:xnack-",
    "world_size": 8,
    "model_dim": 5120,
    "inter_dim": 2304,
    "experts": 128,
    "topk": 3,
    "quant": "a8w4",
    "swiglu_limit": 10.0,
}
DRAFT_SHAPE = {
    "world_size": 8,
    "experts_per_rank": 16,
    "model_dim": 5120,
    "inter_dim": 2304,
}
V41_FLASH_EP8_SHAPE = dict(DRAFT_SHAPE, experts_per_rank=48)
SMALL_MTPRS = (256, 512)
# The draft MoE (128 experts) on 1, 2 and 4 ranks: staged route only.
STAGED_DRAFT_SHAPES = [
    dict(DRAFT_SHAPE, world_size=world, experts_per_rank=128 // world)
    for world in (1, 2, 4)
]
# The target model (384 experts) on 8, 4 and 2 ranks, V4-Pro, and V4.1-Flash EP8 with EPLB redundant experts.
TARGET_SHAPES = [
    {},
    V41_FLASH_EP8_SHAPE,
    dict(V41_FLASH_EP8_SHAPE, world_size=4, experts_per_rank=96),
    dict(V41_FLASH_EP8_SHAPE, world_size=2, experts_per_rank=192),
    dict(V41_FLASH_EP8_SHAPE, experts_per_rank=49),
]


def _buckets(mtpr):
    return [b for b in FUSED_TOKEN_BUCKETS if b <= mtpr]


@contextmanager
def _without_ep8_table():
    table = cfg._V41_FLASH_EP8_SMALL_KNOBS
    cfg._V41_FLASH_EP8_SMALL_KNOBS = {}
    cfg._fused_config.cache_clear()
    try:
        yield
    finally:
        cfg._V41_FLASH_EP8_SMALL_KNOBS = table
        cfg._fused_config.cache_clear()


def test_layout_gate():
    assert fused_layout_supported(**DRAFT)
    for field, value in (
        ("experts", 136),
        ("topk", 4),
        ("world_size", 4),
        ("model_dim", 7168),
        ("quant", "a4w4"),
    ):
        assert not fused_layout_supported(**{**DRAFT, field: value}), field


def test_no_large_m_capacity():
    assert large_m_mtprs_for(**DRAFT_SHAPE) == ()
    # The 48-expert layout of the same dims keeps every large-input capacity.
    assert large_m_mtprs_for(**V41_FLASH_EP8_SHAPE) == LARGE_M_MTPRS
    assert (
        large_m_mtprs_for(world_size=8, model_dim=5120, inter_dim=2304) == LARGE_M_MTPRS
    )
    for mtpr in LARGE_M_MTPRS:
        with pytest.raises(ValueError):
            select_fused_config(1, mtpr, **DRAFT_SHAPE)


@pytest.mark.parametrize("mtpr", [b for b in TOKEN_BUCKETS if b >= 16])
def test_route_for_every_max_tokens(mtpr):
    """As MegaMoEV2 builds it: small-input route up to SMALL_M_MAX_MTPR rows a rank, staged above."""
    large_mtprs = large_m_mtprs_for(**DRAFT_SHAPE)
    plan = build_mega_moe_bundle_plan(
        mtpr,
        small_m_mtprs=(256, 512),
        large_m=mtpr in large_mtprs,
        **DRAFT_SHAPE,
    )
    assert not plan.large_m
    for max_tokens in sorted(
        {0, 1, 3, 255, 256, 257, 511, 512, 513, LARGE_M_MIN_TOKENS, mtpr - 1, mtpr}
    ):
        if not 0 <= max_tokens <= mtpr:
            continue
        route, capacity = plan.route_for(max_tokens)
        if max_tokens <= SMALL_M_MAX_MTPR and max_tokens <= max(plan.small_m_mtprs):
            assert route == ROUTE_SMALL_M, max_tokens
            assert capacity == min(m for m in plan.small_m_mtprs if m >= max_tokens)
        else:
            assert route == ROUTE_STAGED, max_tokens


@pytest.mark.parametrize("mtpr", SMALL_MTPRS)
def test_small_m_configs_are_the_ep8_tune_with_bf16_rows(mtpr):
    """Each bucket: the 48-expert EP8 tune (without its per-bucket table) with every rank returning BF16 rows."""
    with _without_ep8_table():
        base = {
            b: select_fused_config(b, mtpr, **V41_FLASH_EP8_SHAPE)
            for b in _buckets(mtpr)
        }
    for bucket in _buckets(mtpr):
        draft = select_fused_config(bucket, mtpr, **DRAFT_SHAPE)
        ref = base[bucket]
        k1 = replace(ref.fused_stage1, bf16_rows_always=True)
        assert draft.stage1 == ref.stage1, bucket
        assert draft.stage2 == ref.stage2, bucket
        assert draft.p2p_quant == ref.p2p_quant, bucket
        assert draft.fused_stage2 == ref.fused_stage2, bucket
        assert draft.fused_stage1 == k1, bucket


@pytest.mark.parametrize("mtpr", SMALL_MTPRS)
def test_small_m_buckets_share_one_protocol(mtpr):
    protocols = set()
    for bucket in _buckets(mtpr):
        k = select_fused_config(bucket, mtpr, **DRAFT_SHAPE).fused_stage1
        assert k.bf16_rows_always, bucket
        assert k.en_order == 0, bucket
        protocols.add(
            (
                k.dedup_rows,
                k.plan_local,
                bool(k.direct_done),
                k.tok_counters,
                k.bf16_head_tokens,
                k.bf16_rows_always,
            )
        )
    assert len(protocols) == 1


@pytest.mark.parametrize("mtpr", [b for b in TOKEN_BUCKETS if b >= 16])
def test_staged_route_returns_bf16_at_every_capacity(mtpr):
    """The draft layout's staged route sends BF16 rows back (p2p_quant "none") for every bucket of every
    capacity, also from P2P_FP8_MIN_MTPR on, where "auto" picks FP8 rows for other layouts; the 48-expert layout
    keeps them."""
    plan = build_mega_moe_bundle_plan(mtpr, **DRAFT_SHAPE)
    assert {entry.config.p2p_quant for entry in plan.entries} == {"none"}
    assert {key.p2p_quant for key in plan.stage2_variants} == {"none"}
    assert not any(entry.config.stage2.aligned_pair for entry in plan.entries)
    main = build_mega_moe_bundle_plan(mtpr, **V41_FLASH_EP8_SHAPE)
    expect = "fp8_blockwise_1x32" if mtpr >= P2P_FP8_MIN_MTPR else "none"
    assert {entry.config.p2p_quant for entry in main.entries} == {expect}
    for entry, main_entry in zip(plan.entries, main.entries):
        bucket = entry.token_bucket
        # Only the return format (and the FP8-only aligned-pair Stage2 it rules out) differs from the base config
        # of the same dims: the draft layout has no staged tune of its own.
        base = select_mega_moe_config(
            bucket, mtpr, p2p_quant="none", **DRAFT_SHAPE
        )
        assert entry.config == base, bucket
        assert main_entry.token_bucket == bucket


def test_staged_route_explicit_p2p_quant_kept():
    """Only "auto" resolves to BF16 rows: an explicit p2p_quant is the caller's."""
    config = select_mega_moe_config(
        64, 16384, p2p_quant="fp8_blockwise_1x32", **DRAFT_SHAPE
    )
    assert config.p2p_quant == "fp8_blockwise_1x32"


@pytest.mark.parametrize("mtpr", SMALL_MTPRS)
def test_other_layouts_keep_their_returns(mtpr):
    """bf16_rows_always and the BF16 staged return are the draft layout's: every other fused layout keeps the
    rows-received choice and the "auto" staged return."""
    assert cfg.BF16_RETURN_LAYOUTS == frozenset(
        {(8, 16, 5120, 2304), (4, 32, 5120, 2304), (2, 64, 5120, 2304), (1, 128, 5120, 2304)}
    )
    for shape in TARGET_SHAPES:
        for bucket in _buckets(mtpr):
            assert not select_fused_config(
                bucket, mtpr, **shape
            ).fused_stage1.bf16_rows_always
        for big in (1024, 16384):
            assert (
                select_mega_moe_config(64, big, **shape).p2p_quant
                == "fp8_blockwise_1x32"
            )


@pytest.mark.parametrize("shape", STAGED_DRAFT_SHAPES, ids=lambda s: f"ep{s['world_size']}")
def test_staged_draft_layouts_have_no_fused_route(shape):
    layout = dict(DRAFT, world_size=shape["world_size"])
    assert not fused_layout_supported(**layout)
    for mtpr in (256, 512, 16384):
        plan = build_mega_moe_bundle_plan(mtpr, **shape)
        assert plan.small_m_mtprs == () and not plan.large_m
        assert plan.route_for(min(10, mtpr))[0] == ROUTE_STAGED


@pytest.mark.parametrize("shape", STAGED_DRAFT_SHAPES, ids=lambda s: f"ep{s['world_size']}")
@pytest.mark.parametrize("mtpr", [b for b in TOKEN_BUCKETS if b >= 16])
def test_staged_draft_layouts_return_bf16_at_every_capacity(shape, mtpr):
    """The draft MoE on 1 / 2 / 4 ranks (the DEP4 / DEP2 draft instance has a 16384-token capacity, where "auto"
    picked FP8 rows) returns BF16 rows on its staged route; only the return format and the FP8-only aligned-pair
    Stage2 differ from what "auto" picked before."""
    plan = build_mega_moe_bundle_plan(mtpr, **shape)
    assert {entry.config.p2p_quant for entry in plan.entries} == {"none"}
    assert {key.p2p_quant for key in plan.stage2_variants} == {"none"}
    assert not any(entry.config.stage2.aligned_pair for entry in plan.entries)
    for entry in plan.entries:
        bucket = entry.token_bucket
        assert entry.config == select_mega_moe_config(
            bucket, mtpr, p2p_quant="none", **shape
        ), bucket
        fp8 = select_mega_moe_config(
            bucket, mtpr, p2p_quant="fp8_blockwise_1x32", **shape
        )
        assert entry.config.stage1 == fp8.stage1, bucket
        if not fp8.stage2.aligned_pair:
            assert entry.config.stage2 == fp8.stage2, bucket
        # An explicit p2p_quant is the caller's.
        assert fp8.p2p_quant == "fp8_blockwise_1x32"


def test_knob_defaults_off():
    k = {f.name: f.default for f in fields(cfg.FusedStage1Knobs)}
    assert k["bf16_rows_always"] is False
    assert "fp8_rows_always" not in k


@pytest.mark.parametrize("mtpr", SMALL_MTPRS)
def test_direct_fixed_slot(mtpr):
    """The small-input kernel dispatches the draft layout into fixed slots, as the 48-expert one."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import (
        _use_direct_fixed_slot,
    )

    cap = 8 * mtpr
    assert _use_direct_fixed_slot(8, 16, mtpr, cap, 32)
    assert _use_direct_fixed_slot(8, 48, mtpr, cap, 32)
    assert not _use_direct_fixed_slot(8, 17, mtpr, cap, 32)
