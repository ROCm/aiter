"""Shared-expert layouts use SmallM while retaining staged prefill."""

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    build_mega_moe_bundle_plan,
    fused_layout_supported,
    large_m_mtprs_for,
    select_fused_config,
)


LAYOUT = dict(world_size=4, model_dim=5120, inter_dim=2304)


def test_only_the_target_shared_layout_is_added():
    args = dict(LAYOUT, arch="gfx950", experts=388, topk=7, quant="a8w4", swiglu_limit=10.0)
    assert fused_layout_supported(**args)
    for patch in ({"experts": 392}, {"topk": 6}, {"quant": "a4w4"}, {"fanout_masks": (1, 2, 4, 8)}):
        assert not fused_layout_supported(**dict(args, **patch))


def test_shared_layout_has_no_large_input_capacity():
    assert large_m_mtprs_for(**LAYOUT, experts_per_rank=97) == ()
    assert large_m_mtprs_for(**LAYOUT, experts_per_rank=96) == (8192, 16384)
    config = select_fused_config(256, 512, **LAYOUT, experts_per_rank=97)
    assert config.stage1.sort_block_m == config.stage2.block_m == 32
    assert config.fused_stage1.dedup_rows and config.fused_stage1.plan_local


def test_shared_row_hint_routes_and_no_hint_staged_plan():
    plan = build_mega_moe_bundle_plan(
        16384, **LAYOUT, experts_per_rank=97, small_m_mtprs=(256, 512), large_m=False
    )
    assert plan.route_for(1) == ("small_m", 256)
    assert plan.route_for(384) == ("small_m", 512)
    assert plan.route_for(512) == ("small_m", 512)
    assert plan.route_for(513) == ("staged", 16384)
    assert plan.route_for(1024) == ("staged", 16384)
    assert plan.route_for(256, small_m=False) == ("staged", 16384)
