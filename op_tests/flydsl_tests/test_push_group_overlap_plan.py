"""CPU-only layout/config tests for the fused dispatch->GEMM1 overlap stage-1.

These cover the pieces that must be right before a single line of device code
runs: which symmetric regions exist, how big they are, and how the mega grid is
split into producer / planner / consumer CTAs.
"""
import pytest
import torch

from aiter.ops.flydsl.dispatch_combine_v2 import dispatch_combine_op as dc_op
from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
    EpDispatchCombineConfig,
    _push_group_overlap_regions,
    _validate_overlap_ctas,
)


def _cfg(*, world_size=4, epr=8, overlap=True, tile_m=64, cap=128, topk=6, tokens=128):
    return EpDispatchCombineConfig(
        rank=0,
        world_size=world_size,
        hidden_dim=7168,
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        data_type=torch.bfloat16,
        dispatch_data_type=torch.float8_e4m3fn,
        combine_data_type=torch.bfloat16,
        scale_dim=7168 // 32,
        scale_type_size=1,
        combine_mode="scatter_fused",
        push_group=True,
        push_group_overlap=overlap,
        push_group_scale_wmma_rep=tile_m // 16,
        cap_per_expert=cap,
    )


def test_overlap_regions_are_absent_when_disabled():
    assert _push_group_overlap_regions(_cfg(overlap=False)) == []


def test_overlap_regions_cover_counts_and_tiles():
    cfg = _cfg(world_size=4, epr=8, tile_m=64, cap=128)
    regions = dict(_push_group_overlap_regions(cfg))
    e_total = 4 * 8
    assert regions["pg_ov_count"] == 2 * 4 * e_total * 4
    assert regions["pg_ov_count_done"] == 4 * 4
    # cap=128, tile_m=64 -> 2 tiles per expert
    assert regions["pg_ov_tile_arrived"] == 8 * 2 * 4
    assert regions["pg_ov_tile_expected"] == 8 * 2 * 4
    assert regions["pg_ov_entry"] == 4
    # Protocol barriers and debug scratch, then the work-stealing heads, then the
    # plan-ready replicas -- each group spaced out so it owns its own lines.
    assert regions["pg_ov_bar"] == dc_op.PG_OV_BAR_SLOTS * 4
    assert dc_op.PG_OV_BAR_SLOTS >= (
        dc_op.PG_OV_PLAN_SLOT
        + dc_op.PG_OV_PLAN_FANOUT_MAX * dc_op.PG_OV_PLAN_STRIDE
    )
    assert dc_op.PG_OV_PLAN_SLOT >= (
        dc_op.PG_OV_WORK_SLOT + dc_op.PG_OV_WORK_SHARDS * dc_op.PG_OV_WORK_STRIDE
    )
    assert regions["pg_ov_plan_ready"] == 4
    assert regions["pg_ov_my_base"] == e_total * 4
    assert regions["pg_ov_hist"] == e_total * 4
    assert regions["pg_ov_route_slot"] == 128 * 6 * 4


def test_tiles_per_expert_tracks_the_scale_preshuffle_geometry():
    cfg = _cfg(tile_m=16, cap=128)
    assert cfg.push_group_overlap_tile_m == 16
    assert cfg.push_group_overlap_tiles_per_expert == 8
    regions = dict(_push_group_overlap_regions(cfg))
    assert regions["pg_ov_tile_arrived"] == 8 * 8 * 4


def test_overlap_requires_push_group():
    with pytest.raises(ValueError, match="requires push_group"):
        cfg = _cfg()
        cfg.push_group = False
        cfg.__post_init__()


def test_overlap_requires_send_side_quant():
    with pytest.raises(ValueError, match="send-side quant"):
        cfg = _cfg()
        cfg.push_group_scale_wmma_rep = 0
        cfg.__post_init__()


def test_cta_split_rejects_degenerate_grids():
    # dispatch_ctas covers the planner (role 0) plus the producers, matching the
    # partition the kernel derives from a ticket.
    with pytest.raises(ValueError, match="no producer after the planner"):
        _validate_overlap_ctas(grid_ctas=64, dispatch_ctas=1)
    # A work-queue shard nobody pulls from is a set of tiles nobody computes.
    with pytest.raises(ValueError, match="consumer CTAs .one per work shard."):
        _validate_overlap_ctas(grid_ctas=64, dispatch_ctas=64)
    assert _validate_overlap_ctas(grid_ctas=64, dispatch_ctas=3) == (2, 1, 61)
