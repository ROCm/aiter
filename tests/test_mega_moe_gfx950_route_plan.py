# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 on gfx950: route selection, one plan picks the route of every call -- no GPU needed.

Every EP rank must take the same route for a call, so the plan's choice may
depend only on EP-wide inputs (the plan and the call's max_tokens). The
multi-GPU behaviour of each route (EP8 DeepSeek-V4-Pro, EP8 / EP4 / EP2 DeepSeek-V4.1-Flash)
lives in op_tests/multigpu_tests/test_mega_moe_v2.py: --suite size for the
pinned staged and large-input routes, --suite routes for the instance that
serves all three.
"""

import pytest

pytest.importorskip("flydsl", reason="MegaMoE is built with FlyDSL")
pytest.importorskip("mori.shmem", reason="MegaMoE's runtime imports mori")

import hashlib
from dataclasses import astuple, fields

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    DEFAULT_SMALL_M_MTPRS,
    FUSED_QUANTS,
    FUSED_TOKEN_BUCKETS,
    LARGE_M_MIN_TOKENS,
    LARGE_M_MTPRS,
    LARGE_M_QUANTS,
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

V4_PRO = {
    "arch": "gfx950:sramecc+:xnack-",
    "world_size": 8,
    "model_dim": 7168,
    "inter_dim": 3072,
    "experts": 384,
    "topk": 6,
    "quant": "a8w4",
    "swiglu_limit": 10.0,
}
# EP4 DeepSeek-V4.1-Flash: 96 experts a rank.
V41_FLASH_EP4 = dict(V4_PRO, world_size=4, model_dim=5120, inter_dim=2304, experts=384)
V41_FLASH_EP4_SHAPE = {
    "world_size": 4,
    "experts_per_rank": 96,
    "model_dim": 5120,
    "inter_dim": 2304,
}
# EP2 DeepSeek-V4.1-Flash: 192 experts a rank.
V41_FLASH_EP2 = dict(V41_FLASH_EP4, world_size=2)
V41_FLASH_EP2_SHAPE = {
    "world_size": 2,
    "experts_per_rank": 192,
    "model_dim": 5120,
    "inter_dim": 2304,
}
# EP8 DeepSeek-V4.1-Flash: V4-Pro's 48 experts a rank with V4.1-Flash's dims.
V41_FLASH_EP8 = dict(V4_PRO, model_dim=5120, inter_dim=2304)
V41_FLASH_EP8_SHAPE = {
    "world_size": 8,
    "experts_per_rank": 48,
    "model_dim": 5120,
    "inter_dim": 2304,
}
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
    assert fused_layout_supported(**V41_FLASH_EP2)
    assert fused_layout_supported(**V41_FLASH_EP8)
    assert not fused_layout_supported(**{**V41_FLASH_EP2, "experts": 192})
    # Each layout is supported whole: no mix of the shapes.
    assert not fused_layout_supported(**{**V41_FLASH_EP4, "experts": 768})
    assert not fused_layout_supported(**{**V41_FLASH_EP8, "experts": 768})
    assert not fused_layout_supported(**{**V41_FLASH_EP4, "inter_dim": 3072})
    assert not fused_layout_supported(**{**V41_FLASH_EP8, "inter_dim": 3072})
    assert not fused_layout_supported(**{**V41_FLASH_EP8, "model_dim": 7168})
    # A4W4 takes the small-input kernel on every layout; the large-input kernel stays A8W4.
    for layout in (V4_PRO, V41_FLASH_EP4, V41_FLASH_EP2, V41_FLASH_EP8):
        assert fused_layout_supported(**{**layout, "quant": "a4w4"})
    assert set(FUSED_QUANTS) == {"a8w4", "a4w4"} and LARGE_M_QUANTS == ("a8w4",)
    for field, value in (
        ("arch", "gfx942"),
        ("world_size", 4),
        ("model_dim", 6144),
        ("inter_dim", 2048),
        ("experts", 416),
        ("topk", 8),
        ("quant", "a8w8"),
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
    shape = {"world_size": 8, "model_dim": 7168, "inter_dim": 3072}
    assert large_m_mtprs_for(**shape) == LARGE_M_MTPRS
    # EP4 V4.1-Flash: 8192 and 16384 (it faults at 1024..4096), both with the tune measured at 8192.
    assert large_m_mtprs_for(world_size=4, model_dim=5120, inter_dim=2304) == (
        8192,
        16384,
    )
    tuned = _knob_table(select_fused_config(8192, 8192, **V41_FLASH_EP4_SHAPE))
    for mtpr in large_m_mtprs_for(world_size=4, model_dim=5120, inter_dim=2304):
        for tokens in (0, 1, LARGE_M_MIN_TOKENS, mtpr):
            assert (
                _knob_table(select_fused_config(tokens, mtpr, **V41_FLASH_EP4_SHAPE))
                == tuned
            ), (mtpr, tokens)
    with pytest.raises(ValueError):
        select_fused_config(32768, 32768, **V41_FLASH_EP4_SHAPE)
    # EP2 V4.1-Flash: 8192 and 16384, one tune.
    assert large_m_mtprs_for(world_size=2, model_dim=5120, inter_dim=2304) == (
        8192,
        16384,
    )
    tuned = _knob_table(select_fused_config(8192, 8192, **V41_FLASH_EP2_SHAPE))
    for mtpr in (8192, 16384):
        for tokens in (0, 1, LARGE_M_MIN_TOKENS, mtpr):
            assert (
                _knob_table(select_fused_config(tokens, mtpr, **V41_FLASH_EP2_SHAPE))
                == tuned
            ), (mtpr, tokens)
    for mtpr in (1024, 2048, 4096, 32768):
        with pytest.raises(ValueError):
            select_fused_config(mtpr, mtpr, **V41_FLASH_EP2_SHAPE)


@pytest.mark.parametrize("mtpr", MTPRS)
def test_ep4_route_for_every_max_tokens(mtpr):
    """EP4 V4.1-Flash: small-input up to 512 rows a rank, large-input from 513 at its large capacities."""
    plan = build_mega_moe_bundle_plan(
        mtpr,
        experts_per_rank=96,
        model_dim=5120,
        inter_dim=2304,
        world_size=4,
        small_m_mtprs=DEFAULT_SMALL_M_MTPRS,
        large_m=mtpr in large_m_mtprs_for(world_size=4, model_dim=5120, inter_dim=2304),
    )
    assert plan.large_m == (mtpr in (8192, 16384))
    for max_tokens in sorted({0, 1, 256, 257, 512, 513, 768, 1024, mtpr - 1, mtpr}):
        if not 0 <= max_tokens <= mtpr:
            continue
        route, capacity = plan.route_for(max_tokens)
        if max_tokens <= plan.small_m_mtprs[-1]:
            assert route == ROUTE_SMALL_M, max_tokens
        elif plan.large_m:
            assert (route, capacity) == (ROUTE_LARGE_M, mtpr), max_tokens
        else:
            assert (route, capacity) == (ROUTE_STAGED, mtpr), max_tokens
        assert plan.route_for(max_tokens, large_m=False)[0] != ROUTE_LARGE_M


@pytest.mark.parametrize("layout", ("ep8", "ep4", "ep2", "ep8f"))
def test_large_m_buffers_fit_32bit_offsets(layout):
    """Every large-input capacity keeps its buffer-resource views inside 32-bit offsets and its encodings in
    their bit fields (the kernel and LargeMRuntime raise past them; this pins the margin).
    """
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import (
        LargeMRuntime,
        large_m_kernel_specs,
    )

    world_size, epr, model_dim, inter_dim = {
        "ep8": (8, 48, 7168, 3072),
        "ep4": (4, 96, 5120, 2304),
        "ep2": (2, 192, 5120, 2304),
        "ep8f": (8, 48, 5120, 2304),
    }[layout]
    for mtpr in large_m_mtprs_for(
        world_size=world_size, model_dim=model_dim, inter_dim=inter_dim
    ):
        config = select_fused_config(
            mtpr,
            mtpr,
            world_size=world_size,
            experts_per_rank=epr,
            model_dim=model_dim,
            inter_dim=inter_dim,
        )
        num_valid_max, payload_rows, _, _ = LargeMRuntime.payload_capacity(
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=world_size * epr,
            topk=6,
            max_tok_per_rank=mtpr,
        )
        # dedup_rows writes and GEMM1 reads payload rows [0, world_size * mtpr) only: one row a source token.
        if config.fused_stage1.dedup_rows:
            assert payload_rows == world_size * mtpr + 1, (layout, mtpr)
            assert world_size * mtpr * model_dim < 1 << 31, (layout, mtpr)
        else:
            assert payload_rows == num_valid_max, (layout, mtpr)
        s2_spec, _ = large_m_kernel_specs(
            config,
            model_dim=model_dim,
            inter_dim=inter_dim,
            topk=6,
            rank=0,
            npes=world_size,
            mtpr=mtpr,
            experts_per_rank=epr,
            num_valid_max=num_valid_max,
        )
        spec = dict(s2_spec)
        assert spec["comb_inp_nbytes"] < 1 << 31, (layout, mtpr)
        assert world_size * mtpr <= 1 << 24, (layout, mtpr)
        k1, k2 = config.fused_stage1, config.fused_stage2
        if k1.tag_counts:
            assert mtpr < 1 << 16
        if k1.bucket_pairs:
            assert world_size * epr * mtpr * 4 < 1 << 31, (layout, mtpr)
        if k2.scale_stage:
            nbytes = (
                spec["max_g2_m_blocks"] * (model_dim // spec["BN"]) * spec["BM"] * 16
            )
            assert nbytes < 1 << 31, (layout, mtpr)


@pytest.mark.parametrize("mtpr", MTPRS)
def test_v41ep8_route_for_every_max_tokens(mtpr):
    """EP8 V4.1-Flash: small-input up to 512 rows a rank, large-input from 513 at its large capacities."""
    large = large_m_mtprs_for(world_size=8, model_dim=5120, inter_dim=2304)
    plan = build_mega_moe_bundle_plan(
        mtpr,
        experts_per_rank=48,
        model_dim=5120,
        inter_dim=2304,
        world_size=8,
        small_m_mtprs=DEFAULT_SMALL_M_MTPRS,
        large_m=mtpr in large,
    )
    assert plan.large_m == (mtpr in large)
    for max_tokens in sorted({0, 1, 256, 257, 512, 513, 768, 1024, mtpr - 1, mtpr}):
        if not 0 <= max_tokens <= mtpr:
            continue
        route, capacity = plan.route_for(max_tokens)
        if max_tokens <= plan.small_m_mtprs[-1]:
            assert route == ROUTE_SMALL_M, max_tokens
        elif plan.large_m:
            assert (route, capacity) == (ROUTE_LARGE_M, mtpr), max_tokens
        else:
            assert (route, capacity) == (ROUTE_STAGED, mtpr), max_tokens


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


# Fused knobs added with the EP2 V4.1-Flash tune and after it: they enter the table only when set, so a digest
# pinned before them changes only where a config sets one.
_EP2_KNOBS = ("stream_w", "pair_tiles")


def _knob_table(config):
    return (
        astuple(config.stage1),
        astuple(config.stage2),
        config.p2p_quant,
        tuple(
            (f.name, getattr(config.fused_stage1, f.name))
            for f in fields(FusedStage1Knobs)
            if f.name not in _EP2_KNOBS
            or getattr(config.fused_stage1, f.name) != f.default
        ),
        tuple(
            (f.name, getattr(config.fused_stage2, f.name))
            for f in fields(FusedStage2Knobs)
        ),
    )


# The EP4 V4.1-Flash fused configs, as ported from 28ee7fcd (its _tune_v41_flash_ep4 on the post-processed
# small-input configs and on the large-input config at MTPR 8192). On a mismatch the assertion shows the current
# table.
_EP4_FUSED_DIGEST = "bc90461aa6bf712787a59363f7f54b6d9cc79b9b20027ffad260c2d8ea4ec7f7"


def test_ep4_fused_configs_unchanged():
    table = {}
    for mtpr, tokens in _FUSED_CASES:
        if mtpr in (256, 512, 8192):
            table[(mtpr, tokens)] = _knob_table(
                select_fused_config(tokens, mtpr, **V41_FLASH_EP4_SHAPE)
            )
    digest = hashlib.sha256(repr(sorted(table.items())).encode()).hexdigest()
    assert digest == _EP4_FUSED_DIGEST, table


@pytest.mark.parametrize("mtpr", (256, 512))
def test_ep4_small_m_buckets_share_one_protocol(mtpr):
    """The ranks of a launch may pick different buckets: the knobs that change what ranks and launches
    exchange are the same in every bucket (SmallMRuntime asserts it too)."""
    protocols = set()
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        k = select_fused_config(bucket, mtpr, **V41_FLASH_EP4_SHAPE).fused_stage1
        protocols.add(
            (
                k.dedup_rows,
                k.plan_local,
                bool(k.direct_done),
                k.tok_counters,
                k.bf16_head_tokens,
            )
        )
        # tok_done adds to the per-token counters every bucket clears.
        assert not k.tok_done or k.tok_counters
        # The runtime's tok_counters also double-buffer the expert counters, which the plan_local combine zeroes.
        assert not k.tok_counters or k.plan_local
        # ... and the srcmap / weight slots, so the payload must go to the source-indexed dedup token rows.
        assert not k.tok_counters or k.dedup_rows
    assert protocols == {(True, True, True, True, 4)}
    # A tok_done bucket (1 and 4 tokens) and one without it run on the same counters.
    assert select_fused_config(1, mtpr, **V41_FLASH_EP4_SHAPE).fused_stage1.tok_done
    assert not select_fused_config(
        16, mtpr, **V41_FLASH_EP4_SHAPE
    ).fused_stage1.tok_done
    # A tok_done reducer waits for valid routes x its own GEMM2 n-blocks, which every tok_done sender counts.
    tok_done_block_n = {
        config.stage2.block_n
        for config in (
            select_fused_config(b, mtpr, **V41_FLASH_EP4_SHAPE)
            for b in FUSED_TOKEN_BUCKETS
            if b <= mtpr
        )
        if config.fused_stage1.tok_done
    }
    assert tok_done_block_n == {256}


# The EP8 V4.1-Flash fused configs (_tune_v41_flash_ep8_small / _large). On a mismatch the assertion shows the
# current table. Pinned with the 256-wide GEMM1 / GEMM2 n-tiles at 16 and 32 tokens and pair_tiles at 64 and 128
# tokens.
_V41EP8_FUSED_DIGEST = (
    "66f44ffe68a7668c3d75c54543796e00df694daec7f8795f12ec778cd75a5127"
)


def test_v41ep8_fused_configs_unchanged():
    table = {}
    for mtpr, tokens in _FUSED_CASES:
        if mtpr in (1024, 2048, 4096):
            continue
        table[(mtpr, tokens)] = _knob_table(
            select_fused_config(tokens, mtpr, **V41_FLASH_EP8_SHAPE)
        )
    digest = hashlib.sha256(repr(sorted(table.items())).encode()).hexdigest()
    assert digest == _V41EP8_FUSED_DIGEST, table


def test_v41ep8_large_m_capacities():
    """EP8 V4.1-Flash: every large-input capacity, each with the tune measured at 16384."""
    assert large_m_mtprs_for(world_size=8, model_dim=5120, inter_dim=2304) == (
        1024,
        2048,
        4096,
        8192,
        16384,
    )
    tuned = _knob_table(select_fused_config(16384, 16384, **V41_FLASH_EP8_SHAPE))
    with pytest.raises(ValueError):
        select_fused_config(32768, 32768, **V41_FLASH_EP8_SHAPE)
    for mtpr in (1024, 2048, 4096, 8192, 16384):
        for tokens in (0, 1, LARGE_M_MIN_TOKENS, mtpr):
            assert (
                _knob_table(select_fused_config(tokens, mtpr, **V41_FLASH_EP8_SHAPE))
                == tuned
            ), (mtpr, tokens)


@pytest.mark.parametrize("mtpr", (256, 512))
def test_v41ep8_small_m_buckets_share_one_protocol(mtpr):
    """As at EP4: the knobs that change what ranks and launches exchange are the same in every bucket."""
    protocols = set()
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        config = select_fused_config(bucket, mtpr, **V41_FLASH_EP8_SHAPE)
        k = config.fused_stage1
        protocols.add(
            (
                k.dedup_rows,
                k.plan_local,
                bool(k.direct_done),
                k.tok_counters,
                k.bf16_head_tokens,
            )
        )
        assert not k.tok_done or (k.tok_counters and config.stage2.block_n == 256)
        assert not k.tok_counters or (k.plan_local and k.dedup_rows)
        # The unrolled A-ahead GEMM1 hung at grid_mult 2 (8 tokens) on gfx950: only grids of one CTA a CU take it.
        assert k.g1_a_ahead == 1 or config.stage1.grid_mult == 1
    assert protocols == {(True, True, True, True, 4)}


def test_ep4_dispatch_cus_cover_two_expert_waves():
    """96 experts a rank are two 64-expert waves: twice the EP8 producers, capped at 224."""
    for tokens in (1, 4, 8, 16, 32, 64, 128, 256, 512):
        ep8 = select_fused_config(tokens, 512).stage1.num_dispatch_cu
        ep4 = select_fused_config(
            tokens, 512, **V41_FLASH_EP4_SHAPE
        ).stage1.num_dispatch_cu
        assert ep4 == min(224, 2 * ep8), tokens
    assert (
        select_fused_config(8192, 8192, **V41_FLASH_EP4_SHAPE).stage1.num_dispatch_cu
        == 192
    )


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
    ep8f = rows(8, 5120, 2304)
    # EP8 V4.1-Flash takes EP4's dedup rows: one more row per (source rank, token).
    assert ep8f[1] == 48 * (8 * mtpr) + 256 + 8 * mtpr
    assert ep8f[1] * ep8f[2] <= _TOKEN_REGION_END


def test_route_payload_sharing():
    """The EP8 fused runtimes end with an all-rank barrier, so they share the instance's payload arena; the EP4
    small- (tok_done) and large-input (comb_tok_wait) runtimes can end before their peers, so they keep their own.
    """
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import LargeMRuntime
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import SmallMRuntime

    ep8 = {
        "world_size": 8,
        "model_dim": 7168,
        "inter_dim": 3072,
        "experts": 384,
        "topk": 6,
    }
    ep4 = {
        "world_size": 4,
        "model_dim": 5120,
        "inter_dim": 2304,
        "experts": 384,
        "topk": 6,
    }
    for mtpr in DEFAULT_SMALL_M_MTPRS:
        assert SmallMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8)
        assert not SmallMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep4)
    for mtpr in large_m_mtprs_for(world_size=8, model_dim=7168, inter_dim=3072):
        assert LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8)
    for mtpr in large_m_mtprs_for(world_size=4, model_dim=5120, inter_dim=2304):
        assert not LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep4)
    # EP8 V4.1-Flash takes EP4's tok_done (its small-input runtime keeps its own payload) but not its
    # comb_tok_wait (the large-input runtime shares the instance's payload arena).
    ep8f = dict(ep8, model_dim=5120, inter_dim=2304)
    for mtpr in DEFAULT_SMALL_M_MTPRS:
        assert not SmallMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8f)
    for mtpr in large_m_mtprs_for(world_size=8, model_dim=5120, inter_dim=2304):
        assert LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8f)


@pytest.mark.parametrize("mtpr", (256, 512))
def test_ep2_small_m_buckets_share_one_protocol(mtpr):
    """EP2 V4.1-Flash: one launch protocol over every bucket, 32-row expert tiles everywhere (8-16 rows an expert
    at 256-512 tokens), and one GEMM2 block_n over the tok_done buckets."""
    protocols, tok_done_block_n = set(), set()
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        config = select_fused_config(bucket, mtpr, **V41_FLASH_EP2_SHAPE)
        k = config.fused_stage1
        protocols.add(
            (
                k.dedup_rows,
                k.plan_local,
                bool(k.direct_done),
                k.tok_counters,
                k.bf16_head_tokens,
            )
        )
        assert not k.tok_done or k.tok_counters
        assert not k.tok_counters or (k.plan_local and k.dedup_rows)
        assert config.stage1.sort_block_m == config.stage2.block_m == 32, bucket
        # Producers are split evenly over the 2 destinations and leave consumers.
        assert 0 < config.stage1.num_dispatch_cu < 256, bucket
        assert config.stage1.num_dispatch_cu % 2 == 0, bucket
        if k.tok_done:
            tok_done_block_n.add(config.stage2.block_n)
    assert protocols == {(True, True, True, True, 4)}
    assert len(tok_done_block_n) == 1


def test_ep2_large_m_config():
    """EP2 V4.1-Flash large-input kernel: producers below the CU count and split over the 2 destinations, and a
    batched dispatch, whose token-scale reads are bounded by the scale buffer (they fault otherwise).
    """
    for mtpr in (8192, 16384):
        config = select_fused_config(mtpr, mtpr, **V41_FLASH_EP2_SHAPE)
        assert 0 < config.stage1.num_dispatch_cu < 256
        assert config.stage1.num_dispatch_cu % 2 == 0
        assert config.fused_stage1.batch_rows
    # EP4 and EP8 V4.1-Flash (whose tune builds on EP4's) batch their rows too, so their reads are bounded as well
    # (batched reads always are); EP8 V4-Pro does not batch them.
    for shape in (V41_FLASH_EP4_SHAPE, V41_FLASH_EP8_SHAPE):
        for mtpr in (8192, 16384):
            knobs = select_fused_config(mtpr, mtpr, **shape).fused_stage1
            assert knobs.batch_rows
    assert not select_fused_config(8192, 8192).fused_stage1.batch_rows


@pytest.mark.parametrize("mtpr", (256, 512))
def test_ep2_small_m_token_rows_live_in_the_payload_arena(mtpr):
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import (
        _TOKEN_REGION_END,
        SmallMRuntime,
    )

    _nvm, rows, row_bytes, _ = SmallMRuntime.payload_capacity(
        world_size=2,
        model_dim=5120,
        inter_dim=2304,
        experts=384,
        topk=6,
        max_tok_per_rank=mtpr,
    )
    assert rows == 192 * (2 * mtpr) + 256 + 2 * mtpr
    assert rows * row_bytes <= _TOKEN_REGION_END


def test_ep2_routes_keep_their_own_payload():
    """The EP2 small- (tok_done) and large-input (comb_tok_wait) runtimes can end before their peer."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import LargeMRuntime
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import SmallMRuntime

    ep2 = {
        "world_size": 2,
        "model_dim": 5120,
        "inter_dim": 2304,
        "experts": 384,
        "topk": 6,
    }
    for mtpr in DEFAULT_SMALL_M_MTPRS:
        assert not SmallMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep2)
    for mtpr in large_m_mtprs_for(world_size=2, model_dim=5120, inter_dim=2304):
        assert not LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep2)


@pytest.mark.parametrize("mtpr", MTPRS)
def test_ep2_route_for_every_max_tokens(mtpr):
    """EP2 V4.1-Flash: small-input up to 512 rows a rank, large-input from 513 at 8192 and 16384."""
    plan = build_mega_moe_bundle_plan(
        mtpr,
        experts_per_rank=192,
        model_dim=5120,
        inter_dim=2304,
        world_size=2,
        small_m_mtprs=DEFAULT_SMALL_M_MTPRS,
        large_m=mtpr in large_m_mtprs_for(world_size=2, model_dim=5120, inter_dim=2304),
    )
    assert plan.large_m == (mtpr in (8192, 16384))
    for max_tokens in sorted({0, 1, 256, 257, 512, 513, 1024, mtpr - 1, mtpr}):
        if not 0 <= max_tokens <= mtpr:
            continue
        route, capacity = plan.route_for(max_tokens)
        if max_tokens <= plan.small_m_mtprs[-1]:
            assert route == ROUTE_SMALL_M, max_tokens
        elif plan.large_m:
            assert (route, capacity) == (ROUTE_LARGE_M, mtpr), max_tokens
        else:
            assert (route, capacity) == (ROUTE_STAGED, mtpr), max_tokens


# The EP2 V4.1-Flash fused configs (_tune_v41_flash_ep2_small / _large). On a mismatch the assertion shows the
# current table.
_EP2_FUSED_DIGEST = "654385c77002fe9ae5fbbab22e2d4a9e4638fcf4cb534af0120810a447de51b7"


def test_ep2_fused_configs_unchanged():
    table = {}
    for mtpr in (256, 512):
        for tokens in sorted(
            {t for m, t in _FUSED_CASES if m == mtpr}
            | {b for b in FUSED_TOKEN_BUCKETS if b <= mtpr}
        ):
            table[(mtpr, tokens)] = _knob_table(
                select_fused_config(tokens, mtpr, **V41_FLASH_EP2_SHAPE)
            )
    for mtpr in (8192, 16384):
        table[(mtpr, mtpr)] = _knob_table(
            select_fused_config(mtpr, mtpr, **V41_FLASH_EP2_SHAPE)
        )
    digest = hashlib.sha256(repr(sorted(table.items())).encode()).hexdigest()
    assert digest == _EP2_FUSED_DIGEST, table


# (world_size, experts_per_rank, model_dim, inter_dim) of the fused layouts.
_ROUTED_LAYOUTS = {
    "v4_pro_ep8": (8, 48, 7168, 3072),
    "v41_flash_ep8": (8, 48, 5120, 2304),
    "v41_flash_ep4": (4, 96, 5120, 2304),
    "v41_flash_ep2": (2, 192, 5120, 2304),
}


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize("layout", sorted(_ROUTED_LAYOUTS))
def test_routed_instance_routes(layout, quant):
    """A routed instance (built offline, as the AOT replay does) serves up to 512 rows a rank through the
    small-input route for both quants; above that A8W4 takes the large-input route and A4W4 the staged one.
    The A4W4 small-input route sends FP4 payload rows and writes an FP8 GEMM1 output, which the shared GEMM1-output
    arena must hold."""
    import torch

    from aiter.aot.flydsl import mega_moe as aot
    from aiter.aot.flydsl.dry_run import dry_run
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1 import stage1_output_nbytes

    world, epr, model_dim, inter_dim = _ROUTED_LAYOUTS[layout]
    mtpr = 8192
    job = {"experts_per_rank": epr, "model_dim": model_dim, "inter_dim": inter_dim}
    device = torch.device("cuda", 0)
    with dry_run(aot.ARCH, aot.NUM_CU, name=aot.DEVICE_NAME), aot._fake_shmem(device):
        from aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2 import MegaMoEV2

        moe = MegaMoEV2(
            rank=0,
            world_size=world,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=world * epr,
            topk=6,
            quant=quant,
            max_tok_per_rank=mtpr,
            swiglu_limit=10.0,
            routed=True,
            **aot._weights(job, device),
        )
    plan = moe._bundle_plan
    assert plan.small_m_mtprs == DEFAULT_SMALL_M_MTPRS
    assert plan.large_m == (quant == "a8w4") == (moe._large_m is not None)
    row_bytes = model_dim // 2 if quant == "a4w4" else model_dim
    for runtime in moe._small_m_runtimes:
        assert runtime._s1_op.row_bytes == row_bytes
        assert runtime._s1_rx.shape[1] == row_bytes
        assert (
            stage1_output_nbytes(runtime._s1_nvm, inter_dim)[0]
            <= moe._s1_output_arena[0].numel()
        )
    for max_tokens in (0, 1, 256, 257, 512, 513, 4096, mtpr):
        route, capacity = moe._route(min(max_tokens, 1), max_tokens)
        if max_tokens <= 512:
            assert route == ROUTE_SMALL_M, max_tokens
            assert capacity == (256 if max_tokens <= 256 else 512)
        elif quant == "a8w4":
            assert (route, capacity) == (ROUTE_LARGE_M, mtpr), max_tokens
        else:
            assert (route, capacity) == (ROUTE_STAGED, mtpr), max_tokens
