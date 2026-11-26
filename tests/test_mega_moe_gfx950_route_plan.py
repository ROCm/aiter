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
    large_m_buckets,
    nearest_token_bucket,
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
    # A4W4 takes both fused kernels (small- and large-input) on every layout, like A8W4.
    for layout in (V4_PRO, V41_FLASH_EP4, V41_FLASH_EP2, V41_FLASH_EP8):
        assert fused_layout_supported(**{**layout, "quant": "a4w4"})
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
        elif plan.large_m:
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
    # A call the small-input route cannot take goes to the large-input route, whatever its rows.
    assert plan.route_for(100, small_m=False) == (ROUTE_LARGE_M, 8192)
    assert plan.route_for(0, small_m=False) == (ROUTE_LARGE_M, 8192)
    assert _plan(8192, small=(256,)).route_for(300) == (ROUTE_LARGE_M, 8192)
    assert _plan(8192, small=()).route_for(1) == (ROUTE_LARGE_M, 8192)
    assert plan.route_for(4096) == (ROUTE_LARGE_M, 8192)
    # MegaMoEV2 itself takes the staged route when a call passes no max_tokens.
    assert plan.route_for(plan.mtpr) == (ROUTE_LARGE_M, 8192)
    staged_only = _plan(8192, small=(), large=False)
    assert all(
        staged_only.route_for(t) == (ROUTE_STAGED, 8192) for t in (0, 1, 512, 8192)
    )


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize(
    "layout", ("v4_pro_ep8", "v41_flash_ep8", "v41_flash_ep4", "v41_flash_ep2")
)
def test_large_m_capacities_share_one_tune(layout, quant):
    """Every layout builds the large-input kernel at every LARGE_M_MTPRS capacity, each launching at a token bucket
    the config the capacity of 16384 launches there (its tunes measured at 8192 / 16384 carry over unchanged).
    """
    shape = _ROUTED_LAYOUTS[layout]
    for mtpr in LARGE_M_MTPRS:
        for bucket in large_m_buckets(mtpr):
            assert select_fused_config(
                bucket, mtpr, quant=quant, **shape
            ) == select_fused_config(bucket, 16384, quant=quant, **shape), (
                mtpr,
                bucket,
            )
    with pytest.raises(ValueError):
        select_fused_config(32768, 32768, quant=quant, **shape)


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize(
    "layout", ("v4_pro_ep8", "v41_flash_ep8", "v41_flash_ep4", "v41_flash_ep2")
)
def test_large_m_small_calls_take_the_min_tokens_config(layout, quant):
    """A large-input call of fewer than LARGE_M_MIN_TOKENS rows launches the config of LARGE_M_MIN_TOKENS, one the
    runtime builds (large_m_buckets); every large-input tune keeps that config up to 1024 rows, so it is the one of
    the call's own bucket too."""
    shape = _ROUTED_LAYOUTS[layout]
    for mtpr in LARGE_M_MTPRS:
        floor = select_fused_config(LARGE_M_MIN_TOKENS, mtpr, quant=quant, **shape)
        assert (
            nearest_token_bucket(LARGE_M_MIN_TOKENS, FUSED_TOKEN_BUCKETS)
            == large_m_buckets(mtpr)[0]
        )
        for tokens in (0, 1, 2, 3, 100, 256, 511, 512, 700, 1024):
            if tokens <= mtpr:
                assert (
                    select_fused_config(tokens, mtpr, quant=quant, **shape) == floor
                ), (mtpr, tokens)


@pytest.mark.parametrize("mtpr", MTPRS)
def test_ep4_route_for_every_max_tokens(mtpr):
    """EP4 V4.1-Flash: small-input up to 512 rows a rank, large-input beyond at every large capacity."""
    plan = build_mega_moe_bundle_plan(
        mtpr,
        experts_per_rank=96,
        model_dim=5120,
        inter_dim=2304,
        world_size=4,
        small_m_mtprs=DEFAULT_SMALL_M_MTPRS,
        large_m=mtpr in LARGE_M_MTPRS,
    )
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


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize("layout", ("ep8", "ep4", "ep2", "ep8f"))
def test_large_m_buffers_fit_32bit_offsets(layout, quant):
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
    a_dtype = "fp4" if quant == "a4w4" else "fp8"
    # Packed FP4 payload rows hold two values a byte.
    row_bytes = model_dim // 2 if a_dtype == "fp4" else model_dim
    for mtpr in LARGE_M_MTPRS:
        config = select_fused_config(
            mtpr,
            mtpr,
            world_size=world_size,
            experts_per_rank=epr,
            model_dim=model_dim,
            inter_dim=inter_dim,
            quant=quant,
        )
        num_valid_max, payload_rows, payload_row_bytes, _ = (
            LargeMRuntime.payload_capacity(
                world_size=world_size,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=world_size * epr,
                topk=6,
                max_tok_per_rank=mtpr,
                a_dtype=a_dtype,
            )
        )
        assert payload_row_bytes == row_bytes, (layout, mtpr)
        # dedup_rows writes and GEMM1 reads payload rows [0, world_size * mtpr) only: one row a source token.
        if config.fused_stage1.dedup_rows:
            assert payload_rows == world_size * mtpr + 1, (layout, mtpr)
            assert world_size * mtpr * row_bytes < 1 << 31, (layout, mtpr)
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
    """EP8 V4.1-Flash: small-input up to 512 rows a rank, large-input beyond at every large capacity."""
    plan = build_mega_moe_bundle_plan(
        mtpr,
        experts_per_rank=48,
        model_dim=5120,
        inter_dim=2304,
        world_size=8,
        small_m_mtprs=DEFAULT_SMALL_M_MTPRS,
        large_m=mtpr in LARGE_M_MTPRS,
    )
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
_FUSED_DIGEST = "579ba7f2e5c54b2169d3aefa502f80441c1a1b9c1720ca406803b03a9c31afc1"


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
    moe.mtpr, moe._bundle_plan = mtpr, _plan(mtpr)
    for rows in sorted({0, 1, min(128, mtpr), mtpr}):
        assert moe._route(rows, None) == (ROUTE_STAGED, mtpr)
        assert moe._route(rows, None, small_m=False) == (ROUTE_STAGED, mtpr)


def test_max_tokens_is_the_route_switch():
    """A call of a routed instance without max_tokens keeps main's staged route; with it, the fused route of route_for."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2 import MegaMoEV2

    moe = object.__new__(MegaMoEV2)
    moe.mtpr, moe._bundle_plan = 8192, _plan(8192)
    assert moe._route(7, None) == (ROUTE_STAGED, 8192)
    assert moe._route(7, None, small_m=False) == (ROUTE_STAGED, 8192)
    assert moe._route(7, 300) == (ROUTE_SMALL_M, 512)
    assert moe._route(7, 300, small_m=False) == (ROUTE_LARGE_M, 8192)
    assert moe._route(4096, 4096) == (ROUTE_LARGE_M, 8192)


def test_ep8_fused_configs_keep_fused_knobs_off():
    """The EP8 V4-Pro fused kernels are the ones before the EP4 tune: every fused knob off."""
    for mtpr, tokens in _FUSED_CASES:
        config = select_fused_config(tokens, mtpr)
        assert config.fused_stage1 == FusedStage1Knobs(), (mtpr, tokens)
        assert config.fused_stage2 == FusedStage2Knobs(), (mtpr, tokens)


# Fused knobs added after the digests below were pinned (stream_w with the EP2 V4.1-Flash tune): they enter the
# table only when set, so a pinned digest changes only where a config sets one.
_LATE_KNOBS = ("stream_w",)


def _knob_table(config):
    return (
        astuple(config.stage1),
        astuple(config.stage2),
        config.p2p_quant,
        tuple(
            (f.name, getattr(config.fused_stage1, f.name))
            for f in fields(FusedStage1Knobs)
            if f.name not in _LATE_KNOBS
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
_EP4_FUSED_DIGEST = "8099b23f9592c0eaecdeecac707a4dacff777d9b95ad112ac10f2c2a67aa2fab"


def test_ep4_fused_configs_unchanged():
    table = {}
    for mtpr, tokens in _FUSED_CASES:
        if mtpr in (256, 512, 8192):
            table[(mtpr, tokens)] = _knob_table(
                select_fused_config(tokens, mtpr, **V41_FLASH_EP4_SHAPE)
            )
    digest = hashlib.sha256(repr(sorted(table.items())).encode()).hexdigest()
    assert digest == _EP4_FUSED_DIGEST, table


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize("mtpr", (256, 512))
def test_ep4_small_m_buckets_share_one_protocol(mtpr, quant):
    """The ranks of a launch may pick different buckets: the knobs that change what ranks and launches
    exchange are the same in every bucket (SmallMRuntime asserts it too)."""
    protocols = set()
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        k = select_fused_config(
            bucket, mtpr, quant=quant, **V41_FLASH_EP4_SHAPE
        ).fused_stage1
        protocols.add(
            (
                bool(k.direct_done),
                k.tok_counters,
                k.bf16_head_tokens,
            )
        )
        # tok_done adds to the per-token counters every bucket clears.
        assert not k.tok_done or k.tok_counters
    assert protocols == {(True, True, 4)}
    # A tok_done bucket (1 and 4 tokens) and one without it run on the same counters.
    assert select_fused_config(
        1, mtpr, quant=quant, **V41_FLASH_EP4_SHAPE
    ).fused_stage1.tok_done
    assert not select_fused_config(
        16, mtpr, quant=quant, **V41_FLASH_EP4_SHAPE
    ).fused_stage1.tok_done
    # A tok_done reducer waits for valid routes x its own GEMM2 n-blocks, which every tok_done sender counts.
    tok_done_block_n = {
        config.stage2.block_n
        for config in (
            select_fused_config(b, mtpr, quant=quant, **V41_FLASH_EP4_SHAPE)
            for b in FUSED_TOKEN_BUCKETS
            if b <= mtpr
        )
        if config.fused_stage1.tok_done
    }
    assert tok_done_block_n == {256}


# The EP8 V4.1-Flash fused configs (_tune_v41_flash_ep8_small / _large). On a mismatch the assertion shows the
# current table. Pinned with the ring GEMM cores at every small-input bucket.
_V41EP8_FUSED_DIGEST = (
    "5fec03b5026cd86721330d7e016843a704d5d0543fe2039fed7beda35fcf4b90"
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


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize("mtpr", (256, 512))
def test_v41ep8_small_m_buckets_share_one_protocol(mtpr, quant):
    """As at EP4: the knobs that change what ranks and launches exchange are the same in every bucket."""
    protocols = set()
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        config = select_fused_config(bucket, mtpr, quant=quant, **V41_FLASH_EP8_SHAPE)
        k = config.fused_stage1
        protocols.add(
            (
                bool(k.direct_done),
                k.tok_counters,
                k.bf16_head_tokens,
            )
        )
        assert not k.tok_done or (k.tok_counters and config.stage2.block_n == 256)
    assert protocols == {(True, True, 4)}


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize(
    "shape",
    ({}, V41_FLASH_EP8_SHAPE, V41_FLASH_EP4_SHAPE, V41_FLASH_EP2_SHAPE),
    ids=("v4_pro_ep8", "v41_flash_ep8", "v41_flash_ep4", "v41_flash_ep2"),
)
def test_small_m_ring_geometry(shape, quant):
    """Every small-input kernel runs the ring GEMM cores (32-row m-tiles up to 128 tokens an EP group, else 64;
    256-wide GEMM1 units and GEMM2 tiles) on a co-resident grid: two CTAs a CU need two waves a SIMD.
    """
    world_size = shape.get("world_size", 8)
    for mtpr in (256, 512):
        for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
            config = select_fused_config(bucket, mtpr, quant=quant, **shape)
            s1, s2 = config.stage1, config.stage2
            block_m = 32 if bucket * world_size <= 128 else 64
            assert s1.sort_block_m == s2.block_m == block_m, (mtpr, bucket)
            assert s1.tile_n == 64 * s1.num_waves and s2.block_n == 256, (mtpr, bucket)
            grid_mult = min(s1.grid_mult, 2)
            assert grid_mult == 1 or s1.waves_per_eu_hint >= 2, (mtpr, bucket)


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
    """The token rows sit past the fixed slots of the payload arena, inside u32 buffer offsets."""
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
    # Every layout adds one row per (source rank, token) to the fixed slots.
    assert ep8[1] == 48 * (8 * mtpr) + 256 + 8 * mtpr
    assert ep8[1] * ep8[2] <= _TOKEN_REGION_END
    assert ep4[1] == 96 * (4 * mtpr) + 256 + 4 * mtpr
    assert ep4[1] * ep4[2] <= _TOKEN_REGION_END
    ep8f = rows(8, 5120, 2304)
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
    for mtpr in LARGE_M_MTPRS:
        assert LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8)
        assert not LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep4)
    # EP8 V4.1-Flash takes EP4's tok_done (its small-input runtime keeps its own payload) but not its
    # comb_tok_wait (the large-input runtime shares the instance's payload arena).
    ep8f = dict(ep8, model_dim=5120, inter_dim=2304)
    for mtpr in DEFAULT_SMALL_M_MTPRS:
        assert not SmallMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8f)
    for mtpr in LARGE_M_MTPRS:
        assert LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep8f)


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize("mtpr", (256, 512))
def test_ep2_small_m_buckets_share_one_protocol(mtpr, quant):
    """EP2 V4.1-Flash: one launch protocol over every bucket and one GEMM2 block_n over the tok_done buckets."""
    protocols, tok_done_block_n = set(), set()
    for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
        config = select_fused_config(bucket, mtpr, quant=quant, **V41_FLASH_EP2_SHAPE)
        k = config.fused_stage1
        protocols.add(
            (
                bool(k.direct_done),
                k.tok_counters,
                k.bf16_head_tokens,
            )
        )
        assert not k.tok_done or k.tok_counters
        # Producers are split evenly over the 2 destinations and leave consumers.
        assert 0 < config.stage1.num_dispatch_cu < 256, bucket
        assert config.stage1.num_dispatch_cu % 2 == 0, bucket
        if k.tok_done:
            tok_done_block_n.add(config.stage2.block_n)
    assert protocols == {(True, True, 4)}
    assert len(tok_done_block_n) == 1


def test_ep2_large_m_config():
    """EP2 V4.1-Flash large-input kernel: producers below the CU count and split over the 2 destinations, and a
    batched dispatch, whose token-scale reads are bounded by the scale buffer (they fault otherwise).
    """
    for mtpr in LARGE_M_MTPRS:
        config = select_fused_config(mtpr, mtpr, **V41_FLASH_EP2_SHAPE)
        assert 0 < config.stage1.num_dispatch_cu < 256
        assert config.stage1.num_dispatch_cu % 2 == 0
        assert config.fused_stage1.batch_rows
    # EP4 and EP8 V4.1-Flash (whose tune builds on EP4's) batch their rows too, so their reads are bounded as well
    # (batched reads always are); EP8 V4-Pro does not batch them.
    for shape in (V41_FLASH_EP4_SHAPE, V41_FLASH_EP8_SHAPE):
        for mtpr in LARGE_M_MTPRS:
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
    for mtpr in LARGE_M_MTPRS:
        assert not LargeMRuntime.ends_all_ranks(max_tok_per_rank=mtpr, **ep2)


@pytest.mark.parametrize("mtpr", MTPRS)
def test_ep2_route_for_every_max_tokens(mtpr):
    """EP2 V4.1-Flash: small-input up to 512 rows a rank, large-input beyond at every large capacity."""
    plan = build_mega_moe_bundle_plan(
        mtpr,
        experts_per_rank=192,
        model_dim=5120,
        inter_dim=2304,
        world_size=2,
        small_m_mtprs=DEFAULT_SMALL_M_MTPRS,
        large_m=mtpr in LARGE_M_MTPRS,
    )
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
_EP2_FUSED_DIGEST = "ee6447ea4a394e0fa9ed96677afa2661cbd427ac2a7251bfe5bc7bfd949d818f"


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


_ROUTED_LAYOUTS = {
    "v4_pro_ep8": {
        "world_size": 8,
        "experts_per_rank": 48,
        "model_dim": 7168,
        "inter_dim": 3072,
    },
    "v41_flash_ep8": V41_FLASH_EP8_SHAPE,
    "v41_flash_ep4": V41_FLASH_EP4_SHAPE,
    "v41_flash_ep2": V41_FLASH_EP2_SHAPE,
}


@pytest.mark.parametrize("mtpr", (128, 1024, 4096, 8192, 16384))
@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize("layout", sorted(_ROUTED_LAYOUTS))
def test_routed_instance_routes(layout, quant, mtpr):
    """A routed instance (built offline, as the AOT replay does) serves up to 512 rows a rank through the
    small-input route and, at a large-input capacity, every other call through the large-input route, for both
    quants. The A4W4 fused routes send FP4 payload rows and write an FP8 GEMM1 output, which the shared
    GEMM1-output arena must hold (the staged A4W4 route writes FP4)."""
    import torch

    from aiter.aot.flydsl import mega_moe as aot
    from aiter.aot.flydsl.dry_run import dry_run
    from aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_op import (
        group_major_capacity,
    )
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import LargeMRuntime
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m import SmallMRuntime
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1 import stage1_output_nbytes

    shape = _ROUTED_LAYOUTS[layout]
    world, epr = shape["world_size"], shape["experts_per_rank"]
    model_dim, inter_dim = shape["model_dim"], shape["inter_dim"]
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
    small = small_m_mtprs_for(mtpr)
    assert plan.small_m_mtprs == small
    large = mtpr in LARGE_M_MTPRS
    assert plan.large_m == large == (moe._large_m is not None)
    assert large == (mtpr != 128)
    row_bytes = model_dim // 2 if quant == "a4w4" else model_dim
    a_dtype = "fp4" if quant == "a4w4" else "fp8"
    if large:
        runtime = moe._large_m
        assert runtime._a_dtype == a_dtype
        assert runtime._s1_op.row_bytes == row_bytes
        assert runtime._s1_rx.shape[1] == row_bytes
        # The large-input route quantizes into the staged route's buffers, as bytes for FP4.
        assert runtime._quant_x.shape == (mtpr, row_bytes)
        assert runtime._quant_x.dtype == (
            torch.uint8 if quant == "a4w4" else torch.float8_e4m3fn
        )
        assert (
            stage1_output_nbytes(runtime._s1_nvm, inter_dim, "fp8")[0]
            <= moe._s1_output_arena[0].numel()
        )
    for runtime in moe._small_m_runtimes:
        assert runtime._s1_op.row_bytes == row_bytes
        assert runtime._s1_rx.shape[1] == row_bytes
        assert (
            stage1_output_nbytes(runtime._s1_nvm, inter_dim, "fp8")[0]
            <= moe._s1_output_arena[0].numel()
        )
    # The shared payload arena holds exactly the largest of the routes that share it.
    cfg = moe.comb_cfg
    staged = group_major_capacity(
        world_size=world,
        hidden_dim=model_dim,
        max_tok_per_rank=mtpr,
        experts_per_rank=epr,
        topk=6,
        data_type=cfg.dispatch_dtype,
        unit_size=cfg.gm_unit_size,
        scale_dim=cfg.scale_dim,
        scale_type_size=cfg.scale_type_size,
        compact=cfg.gm_compact,
        indexed_payload=cfg.gm_indexed_payload,
    )
    assert staged[2] == row_bytes
    fused = {
        "world_size": world,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": world * epr,
        "topk": 6,
    }
    shared = [staged] + [
        SmallMRuntime.payload_capacity(max_tok_per_rank=m, a_dtype=a_dtype, **fused)
        for m in moe._small_m_shared
    ]
    if moe._large_m_shares_payload:
        shared.append(
            LargeMRuntime.payload_capacity(
                max_tok_per_rank=mtpr, a_dtype=a_dtype, **fused
            )
        )
    assert moe._payload_arena.rx_bytes == max(rows * rb for _, rows, rb, _ in shared)
    for max_tokens in (0, 1, 128, 129, 256, 257, 512, 513, 4096, mtpr):
        if max_tokens > mtpr:
            continue
        route, capacity = moe._route(min(max_tokens, 1), max_tokens)
        if max_tokens <= small[-1]:
            assert route == ROUTE_SMALL_M, max_tokens
            assert capacity == min(m for m in small if m >= max_tokens)
        else:
            assert (route, capacity) == (ROUTE_LARGE_M, mtpr), max_tokens
        if large:
            assert moe._route(min(max_tokens, 1), max_tokens, small_m=False) == (
                ROUTE_LARGE_M,
                mtpr,
            )


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize("layout", sorted(_ROUTED_LAYOUTS))
def test_routed_calls_never_stage(layout, quant):
    """On a fused layout every call of a routed instance (both fused routes built) at a power-of-two capacity up to
    16384 that passes max_tokens takes a fused route: the small-input route up to 512 rows a rank, the large-input
    route for the rest, launching a config its runtime builds. Without the small-input route (prequantized input, or
    a part without 256 CUs) the large-input route takes every call from capacity 1024; below it such a call stages.
    """
    shape = _ROUTED_LAYOUTS[layout]
    for log2 in range(15):
        mtpr = 1 << log2
        plan = build_mega_moe_bundle_plan(
            mtpr,
            a_dtype="fp4" if quant == "a4w4" else "fp8",
            small_m_mtprs=DEFAULT_SMALL_M_MTPRS,
            large_m=mtpr in LARGE_M_MTPRS,
            **shape,
        )
        no_small = build_mega_moe_bundle_plan(
            mtpr,
            a_dtype="fp4" if quant == "a4w4" else "fp8",
            large_m=mtpr in LARGE_M_MTPRS,
            **shape,
        )
        built = large_m_buckets(mtpr)
        for max_tokens in range(mtpr + 1):
            route, _ = plan.route_for(max_tokens)
            assert route != ROUTE_STAGED, (mtpr, max_tokens)
            if route == ROUTE_LARGE_M:
                bucket = nearest_token_bucket(
                    max(max_tokens, LARGE_M_MIN_TOKENS), FUSED_TOKEN_BUCKETS
                )
                assert bucket in built, (mtpr, max_tokens)
            for other in (
                plan.route_for(max_tokens, small_m=False),
                no_small.route_for(max_tokens),
            ):
                assert other == (
                    (ROUTE_LARGE_M, mtpr)
                    if mtpr in LARGE_M_MTPRS
                    else (ROUTE_STAGED, mtpr)
                ), (mtpr, max_tokens)


@pytest.mark.parametrize("mtpr", LARGE_M_MTPRS)
@pytest.mark.parametrize("layout", sorted(_ROUTED_LAYOUTS))
def test_large_m_a4w4_config_geometry(layout, mtpr):
    """The A4W4 large-input kernel's config (its layout's A8W4 config unless tuned for A4W4): a GEMM2 reading FP8,
    whole-wave FP4 A copies, and the A8W4 payload geometry other than the row bytes while the A4W4 tune keeps the
    knobs that size the runtime (large_m_runtime_knobs).
    """
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import large_m_runtime_knobs
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import (
        LargeMRuntime,
        large_m_kernel_specs,
    )

    shape = _ROUTED_LAYOUTS[layout]
    world, epr = shape["world_size"], shape["experts_per_rank"]
    model_dim, inter_dim = shape["model_dim"], shape["inter_dim"]
    fused = {
        "world_size": world,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": world * epr,
        "topk": 6,
        "max_tok_per_rank": mtpr,
    }
    fp8 = LargeMRuntime.payload_capacity(a_dtype="fp8", **fused)
    fp4 = LargeMRuntime.payload_capacity(a_dtype="fp4", **fused)
    assert fp8[2] == model_dim and fp4[2] == model_dim // 2
    config = select_fused_config(mtpr, mtpr, quant="a4w4", **shape)
    if large_m_runtime_knobs(config) == large_m_runtime_knobs(
        select_fused_config(mtpr, mtpr, **shape)
    ):
        assert (fp4[0], fp4[1], fp4[3]) == (fp8[0], fp8[1], fp8[3])
    s2_spec, _ = large_m_kernel_specs(
        config,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=6,
        rank=0,
        npes=world,
        mtpr=mtpr,
        experts_per_rank=epr,
        num_valid_max=fp4[0],
    )
    # GEMM2 reads the FP8 GEMM1 output for both quants.
    assert dict(s2_spec)["a_dtype"] == "fp8"
    # The dispatch copies a token row in 64-lane column blocks of 1024 values: a dwordx4 a lane for FP8, a dwordx2
    # for FP4, so the row must tile by 1024 values.
    assert model_dim % 1024 == 0
    for unit_i32, row_bytes in ((4, model_dim), (2, model_dim // 2)):
        assert (model_dim // 1024) * 64 * unit_i32 * 4 == row_bytes
    # The GEMM1 tiles copy whole 64-chunk waves of 16-byte A chunks (FP4 K-steps are 128 bytes).
    k1 = config.fused_stage1
    sbm = config.stage1.sort_block_m
    rows = [sbm]
    if k1.short_tail_rows:
        rows.append(k1.short_tail_rows)
    if k1.mid_tail_rows:
        rows.append(k1.mid_tail_rows)
    if k1.fold_tail:
        rows.append(sbm + k1.short_tail_rows)
    for tile_rows in rows:
        assert tile_rows * (128 // 16) % 64 == 0, tile_rows


def test_fused_config_quant():
    with pytest.raises(ValueError, match="quant"):
        select_fused_config(1, 512, quant="a16w4")
    # Every fused route and layout serves A4W4 with a config of its own selection.
    for shape in ({}, V41_FLASH_EP4_SHAPE, V41_FLASH_EP2_SHAPE, V41_FLASH_EP8_SHAPE):
        for mtpr in (512, 16384):
            config = select_fused_config(1, mtpr, quant="a4w4", **shape)
            assert config.p2p_quant == select_fused_config(1, mtpr, **shape).p2p_quant


_FUSED_SHAPES = ({}, V41_FLASH_EP4_SHAPE, V41_FLASH_EP2_SHAPE, V41_FLASH_EP8_SHAPE)


def test_large_m_buckets():
    assert large_m_buckets(1024) == (512, 1024)
    assert large_m_buckets(16384) == (512, 1024, 2048, 4096, 8192, 16384)


@pytest.mark.parametrize("quant", ("a8w4", "a4w4"))
@pytest.mark.parametrize("shape", _FUSED_SHAPES)
def test_large_m_bucket_configs_share_the_runtime_knobs(shape, quant):
    """A large-input call launches the config of its EP-wide max_tokens; the runtime's buffers and the routes'
    shared storage follow the capacity's config, so every bucket keeps its runtime knobs. A8W4 runs one config
    per capacity."""
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import large_m_runtime_knobs

    for mtpr in LARGE_M_MTPRS:
        capacity = select_fused_config(mtpr, mtpr, quant=quant, **shape)
        for bucket in large_m_buckets(mtpr):
            config = select_fused_config(bucket, mtpr, quant=quant, **shape)
            assert large_m_runtime_knobs(config) == large_m_runtime_knobs(capacity)
            if quant == "a8w4":
                assert config == capacity, (mtpr, bucket)


# The A4W4 fused configs (_tune_a4w4_small / _tune_a4w4_large on each layout's A8W4 tune): every small-input bucket
# at MTPR 256 and 512, and every large-input capacity pinned with them (_A4W4_DIGEST_LARGE_MTPRS; the others launch
# the configs of 16384, test_large_m_capacities_share_one_tune) at each of its token buckets. On a mismatch the
# assertion shows the current table.
_A4W4_DIGEST_LARGE_MTPRS = {
    "v4_pro_ep8": LARGE_M_MTPRS,
    "v41_flash_ep8": LARGE_M_MTPRS,
    "v41_flash_ep4": (8192, 16384),
    "v41_flash_ep2": (8192, 16384),
}
_A4W4_FUSED_DIGESTS = {
    "v4_pro_ep8": "02cb1e09de48c0c07e5107d26a03dbef213d680cdfc7b1adc7e6be39c3dd5f52",
    "v41_flash_ep8": "3f693ea052bc694190e5f415b2a2d4f4afd88e080c3896d28ab48073156972be",
    "v41_flash_ep4": "02348338ca7fe0329a2c642875c3b21d892c177e9f3233ff6d1d33e979bea89e",
    "v41_flash_ep2": "602205aa4ad05cd1e4642ff40755921a1321f07302f955d571831c4aa4c61fe4",
}


@pytest.mark.parametrize("layout", sorted(_A4W4_FUSED_DIGESTS))
def test_a4w4_fused_configs_unchanged(layout):
    shape = _ROUTED_LAYOUTS[layout]
    table = {}
    for mtpr in (256, 512):
        for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= mtpr):
            table[(mtpr, bucket)] = _knob_table(
                select_fused_config(bucket, mtpr, quant="a4w4", **shape)
            )
    for mtpr in _A4W4_DIGEST_LARGE_MTPRS[layout]:
        for bucket in large_m_buckets(mtpr):
            table[(mtpr, bucket)] = _knob_table(
                select_fused_config(bucket, mtpr, quant="a4w4", **shape)
            )
    digest = hashlib.sha256(repr(sorted(table.items())).encode()).hexdigest()
    assert digest == _A4W4_FUSED_DIGESTS[layout], table
