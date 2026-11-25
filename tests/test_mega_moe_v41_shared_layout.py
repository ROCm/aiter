# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 on gfx950, EP8 DeepSeek-V4.1-Flash with its shared expert fused: 49 experts a rank
(the 48 routed experts and the rank's own shared expert), topk 7 -- no GPU needed.

The layout is its own: the fused small- and large-input routes, fixed-slot dispatch at the small
capacities and its own tunes, never another layout's by accident. The multi-GPU behaviour lives in
op_tests/multigpu_tests/test_mega_moe_v2.py (--network v41_flash_shared).
"""

from dataclasses import fields

import pytest

pytest.importorskip("flydsl", reason="MegaMoE is built with FlyDSL")
pytest.importorskip("mori.shmem", reason="MegaMoE's runtime imports mori")

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    DEFAULT_SMALL_M_MTPRS,
    FIXED_SLOT_EXPERTS_PER_RANK,
    FIXED_SLOT_MAX_MTPR,
    FUSED_TOKEN_BUCKETS,
    LARGE_M_MIN_TOKENS,
    LARGE_M_MTPRS,
    ROUTE_LARGE_M,
    ROUTE_SMALL_M,
    ROUTE_STAGED,
    TOKEN_BUCKETS,
    V41_FLASH_EP8_LAYOUT,
    V41_FLASH_EP8_SHARED_LAYOUT,
    V41_FLASH_SHARED_TOPK,
    build_mega_moe_bundle_plan,
    fused_layout_supported,
    large_m_mtprs_for,
    select_fused_config,
    select_mega_moe_config,
)

SHARED = {
    "arch": "gfx950:sramecc+:xnack-",
    "world_size": 8,
    "model_dim": 5120,
    "inter_dim": 2304,
    "experts": 8 * 49,
    "topk": 7,
    "quant": "a8w4",
    "swiglu_limit": 10.0,
}
SHAPE = {"world_size": 8, "experts_per_rank": 49, "model_dim": 5120, "inter_dim": 2304}
ROUTED_SHAPE = dict(SHAPE, experts_per_rank=48)
MTPRS = tuple(b for b in TOKEN_BUCKETS if b >= 16)


def test_layout_constants():
    assert V41_FLASH_EP8_SHARED_LAYOUT == (8, 49, 5120, 2304)
    assert V41_FLASH_SHARED_TOPK == 7
    assert V41_FLASH_EP8_SHARED_LAYOUT[2:] == V41_FLASH_EP8_LAYOUT[2:]
    assert set(FIXED_SLOT_EXPERTS_PER_RANK) == {48, 49}


def test_fused_layout_gate():
    assert fused_layout_supported(**SHARED)
    # The 49-expert table always carries the shared slot: topk 7 with 392 experts, nothing else.
    assert not fused_layout_supported(**{**SHARED, "topk": 6})
    assert not fused_layout_supported(**{**SHARED, "experts": 384})
    assert not fused_layout_supported(**{**SHARED, "experts": 8 * 50, "topk": 7})
    assert not fused_layout_supported(**{**SHARED, "world_size": 4, "experts": 4 * 49})
    assert not fused_layout_supported(**{**SHARED, "quant": "a4w4"})
    assert not fused_layout_supported(**{**SHARED, "swiglu_limit": 0.0})


def test_large_m_capacities():
    """The large-input capacities of the 48-expert EP8 layout: every large-input capacity."""
    assert (
        large_m_mtprs_for(world_size=8, model_dim=5120, inter_dim=2304) == LARGE_M_MTPRS
    )
    with pytest.raises(ValueError):
        select_fused_config(32768, 32768, **SHAPE)
    for mtpr in LARGE_M_MTPRS:
        tuned = select_fused_config(mtpr, mtpr, **SHAPE)
        for tokens in (0, 1, LARGE_M_MIN_TOKENS, mtpr - 1, mtpr):
            assert select_fused_config(tokens, mtpr, **SHAPE) == tuned, (mtpr, tokens)


@pytest.mark.parametrize("mtpr", MTPRS)
def test_route_for_every_max_tokens(mtpr):
    """Small-input up to 512 rows a rank, large-input from 513 at the large capacities, staged otherwise --
    including each capacity's own boundary."""
    large = large_m_mtprs_for(world_size=8, model_dim=5120, inter_dim=2304)
    plan = build_mega_moe_bundle_plan(
        mtpr,
        **SHAPE,
        small_m_mtprs=DEFAULT_SMALL_M_MTPRS,
        large_m=mtpr in large,
    )
    assert plan.large_m == (mtpr in large)
    for max_tokens in sorted(
        {0, 1, 126, 128, 129, 190, 256, 257, 512, 513, mtpr - 1, mtpr}
    ):
        if not 0 <= max_tokens <= mtpr:
            continue
        route, capacity = plan.route_for(max_tokens)
        if max_tokens <= plan.small_m_mtprs[-1]:
            assert route == ROUTE_SMALL_M, max_tokens
            assert capacity >= max_tokens
        elif plan.large_m:
            assert (route, capacity) == (ROUTE_LARGE_M, mtpr), max_tokens
        else:
            assert (route, capacity) == (ROUTE_STAGED, mtpr), max_tokens


@pytest.mark.parametrize("mtpr", [m for m in TOKEN_BUCKETS if m <= 2048])
def test_fixed_slot_dispatch_at_the_small_capacities(mtpr):
    """Fixed-slot dispatch exactly up to FIXED_SLOT_MAX_MTPR, as the 48-expert layout; every bucket's
    config is built for it (select_mega_moe_config refuses a fixed-slot bucket past the bound).
    """
    shared = build_mega_moe_bundle_plan(mtpr, **SHAPE)
    routed = build_mega_moe_bundle_plan(mtpr, **ROUTED_SHAPE)
    assert shared.fixed_slot_dispatch == (mtpr <= FIXED_SLOT_MAX_MTPR), mtpr
    assert shared.fixed_slot_dispatch == routed.fixed_slot_dispatch, mtpr
    for bucket in (b for b in TOKEN_BUCKETS if b <= mtpr):
        select_mega_moe_config(bucket, mtpr, **SHAPE)
    for other in (47, 50, 52):
        plan = build_mega_moe_bundle_plan(128, **dict(SHAPE, experts_per_rank=other))
        assert not plan.fixed_slot_dispatch, other


def test_fixed_slot_capacity_validated():
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import (
        _use_direct_fixed_slot,
    )
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1 import (
        _validate_fixed_slot_geometry,
    )

    for mtpr in (1, 2, 4, 8, 16, 32, 64, 128):
        for tile_m in (32, 64):
            cap = (8 * mtpr + tile_m - 1) // tile_m * tile_m
            _validate_fixed_slot_geometry(8, 49, mtpr, cap, tile_m)
            assert _use_direct_fixed_slot(8, 49, mtpr, cap, tile_m)
            # A receive capacity one tile short or long of one row a source token and EP8 rank is refused.
            for bad in (cap - tile_m, cap + tile_m):
                with pytest.raises(ValueError):
                    _validate_fixed_slot_geometry(8, 49, mtpr, bad, tile_m)
                assert not _use_direct_fixed_slot(8, 49, mtpr, bad, tile_m)
    with pytest.raises(ValueError):
        _validate_fixed_slot_geometry(8, 50, 128, 1024, 32)
    with pytest.raises(ValueError):
        _validate_fixed_slot_geometry(4, 49, 128, 512, 32)


@pytest.mark.parametrize("mtpr", LARGE_M_MTPRS)
def test_large_m_buffers_fit_32bit_offsets(mtpr):
    """Every route row of the seven-slot table stays inside the large-input runtime's 32-bit views."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import (
        LargeMRuntime,
        large_m_kernel_specs,
    )

    config = select_fused_config(mtpr, mtpr, **SHAPE)
    num_valid_max, payload_rows, _, _ = LargeMRuntime.payload_capacity(
        world_size=8,
        model_dim=5120,
        inter_dim=2304,
        experts=8 * 49,
        topk=V41_FLASH_SHARED_TOPK,
        max_tok_per_rank=mtpr,
    )
    if config.fused_stage1.dedup_rows:
        assert payload_rows == 8 * mtpr + 1
    else:
        assert payload_rows == num_valid_max
    s2_spec, _ = large_m_kernel_specs(
        config,
        model_dim=5120,
        inter_dim=2304,
        topk=V41_FLASH_SHARED_TOPK,
        rank=0,
        npes=8,
        mtpr=mtpr,
        experts_per_rank=49,
        num_valid_max=num_valid_max,
    )
    spec = dict(s2_spec)
    assert spec["comb_inp_nbytes"] < 1 << 31
    assert 8 * mtpr <= 1 << 24
    if config.fused_stage1.bucket_pairs:
        assert 8 * 49 * mtpr * 4 < 1 << 31


@pytest.mark.parametrize("mtpr", (256, 512))
def test_small_m_buckets_share_one_protocol(mtpr):
    """The knobs that change what ranks and launches exchange are the same in every bucket."""
    protocols = set()
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        k = select_fused_config(bucket, mtpr, **SHAPE).fused_stage1
        protocols.add(
            (
                k.dedup_rows,
                k.plan_local,
                bool(k.direct_done),
                k.tok_counters,
                k.bf16_head_tokens,
            )
        )
    assert len(protocols) == 1


def test_own_tunes_selected():
    """The 49-expert layout reaches its own tuners (they start from the 48-expert ones)."""
    import aiter.ops.flydsl.kernels.mega_moe.mega_moe_config as c

    calls = []
    small, large = (
        c._tune_v41_flash_ep8_shared_small,
        c._tune_v41_flash_ep8_shared_large,
    )
    try:
        c._tune_v41_flash_ep8_shared_small = lambda cfg, b: calls.append(
            ("small", b)
        ) or small(cfg, b)
        c._tune_v41_flash_ep8_shared_large = lambda cfg: calls.append(
            ("large",)
        ) or large(cfg)
        c._fused_config.cache_clear()
        select_fused_config(128, 256, **SHAPE)
        select_fused_config(16384, 16384, **SHAPE)
        select_fused_config(128, 256, **ROUTED_SHAPE)
    finally:
        c._tune_v41_flash_ep8_shared_small, c._tune_v41_flash_ep8_shared_large = (
            small,
            large,
        )
        c._fused_config.cache_clear()
    assert calls == [("small", 128), ("large",)]


@pytest.mark.parametrize(
    "mtpr,tokens",
    [(1024, t) for t in (LARGE_M_MIN_TOKENS, 1024)]
    + [(8192, t) for t in (LARGE_M_MIN_TOKENS, 8192)]
    + [(16384, t) for t in (LARGE_M_MIN_TOKENS, 11000, 16384)],
)
def test_large_m_tunes(mtpr, tokens):
    """The large-input sweep kept the 48-expert EP8 knobs, with the per-token combine wait off (its own
    route-major payload costs too much memory)."""
    shared = select_fused_config(tokens, mtpr, **SHAPE)
    assert shared == select_fused_config(tokens, mtpr, **ROUTED_SHAPE)
    assert not shared.fused_stage2.comb_tok_wait


_GROUPS = ("stage1", "stage2", "fused_stage1", "fused_stage2")


@pytest.mark.parametrize("mtpr", (256, 512))
def test_small_m_tunes(mtpr):
    """Each bucket is the 48-expert EP8 tune with exactly the knobs of its _V41_FLASH_EP8_SHARED_SMALL entry."""
    import aiter.ops.flydsl.kernels.mega_moe.mega_moe_config as c

    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        shared = select_fused_config(bucket, mtpr, **SHAPE)
        routed = select_fused_config(bucket, mtpr, **ROUTED_SHAPE)
        knobs = c._V41_FLASH_EP8_SHARED_SMALL.get(bucket, {})
        assert shared.p2p_quant == routed.p2p_quant
        assert shared.stage2.block_m == shared.stage1.sort_block_m
        for group in _GROUPS:
            want = knobs.get(group, {})
            mine, base = getattr(shared, group), getattr(routed, group)
            for f in fields(mine):
                expected = want.get(f.name, getattr(base, f.name))
                assert getattr(mine, f.name) == expected, (bucket, group, f.name)


@pytest.mark.parametrize("mtpr", (256, 512))
@pytest.mark.parametrize("shape", (SHAPE, ROUTED_SHAPE), ids=("epr49", "epr48"))
def test_small_m_en_order_is_buildable(mtpr, shape):
    """en_order reads the local plan, and its GEMM1 order runs on m-tile units (not with pair_tiles)."""
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        k1 = select_fused_config(bucket, mtpr, **shape).fused_stage1
        assert 0 <= k1.en_order < 8, bucket
        if k1.en_order:
            assert k1.plan_local, bucket
        if k1.en_order & 1:
            assert not k1.pair_tiles, bucket
