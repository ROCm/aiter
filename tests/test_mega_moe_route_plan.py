# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 route selection: one plan picks the route of every call -- no GPU needed.

Every EP rank must take the same route for a call, so the plan's choice may
depend only on EP-wide inputs (the plan and the call's max_tokens). The 8-GPU
behaviour of each route lives in op_tests/multigpu_tests/test_mega_moe_v2.py and
test_mega_moe_small_m.py.
"""

import pytest

pytest.importorskip("flydsl", reason="MegaMoE is built with FlyDSL")
pytest.importorskip("mori.shmem", reason="MegaMoE's runtime imports mori")

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (  # noqa: E402
    DEFAULT_SMALL_M_MTPRS,
    LARGE_M_MIN_TOKENS,
    LARGE_M_MTPRS,
    ROUTE_LARGE_M,
    ROUTE_SMALL_M,
    ROUTE_STAGED,
    TOKEN_BUCKETS,
    build_mega_moe_bundle_plan,
    fused_layout_supported,
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
MTPRS = tuple(b for b in TOKEN_BUCKETS if b >= 16)


def _plan(mtpr, *, small=DEFAULT_SMALL_M_MTPRS, large=None):
    return build_mega_moe_bundle_plan(
        mtpr,
        small_m_mtprs=small,
        large_m=mtpr in LARGE_M_MTPRS if large is None else large,
    )


def test_fused_layout_gate():
    assert fused_layout_supported(**V4_PRO)
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
    # Without a hint the instance plans for its capacity.
    assert plan.route_for(plan.mtpr) == (ROUTE_LARGE_M, 8192)
    staged_only = _plan(8192, small=(), large=False)
    assert all(
        staged_only.route_for(t) == (ROUTE_STAGED, 8192) for t in (0, 1, 512, 8192)
    )


def test_large_m_capacity_validated():
    for mtpr in MTPRS:
        if mtpr in LARGE_M_MTPRS:
            assert _plan(mtpr).large_m
        else:
            with pytest.raises(ValueError):
                _plan(mtpr, large=True)
