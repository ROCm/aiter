# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU guards for the interleaved a8w4 stage-1 prefill launch rules.

These are pure host tests: they do not compile or launch a kernel, so they run
on every CI node.  They pin the three things the launch change relies on:

* the 256-wide tile is taken only on the measured DeepSeek-V4 prefill family
  -- four ``(model_dim, inter_dim)`` pairs, E 384/385, topk 6/7, a 64/128-row
  block and a bf16/fp8 epilogue -- and only when ``2 * inter_dim`` is
  256-aligned;
* a caller-supplied ``xcd_swizzle`` is never re-gridded by the tile rule;
* the XCD rule refuses a launch whose ``tile2_pad != 0`` (the open
  aiter#8512 / #6598 grid mismatch) or whose stock remap would drop a block a
  routing can fill (aiter#7567);
* the CShuffle half-budget is used only for the measured family's wide
  interleaved tile, so every other config compiles to the same kernel as before;
* the rules are gated to the measured gfx950 arch and decline bias / sorted
  weights.
"""

import pytest

from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage_common import (
    flydsl_stage1_cshuffle_n,
)
from aiter.ops.flydsl.moe_common import flydsl_stage1_measured_prefill
from aiter.ops.flydsl.moe_kernels import (
    flydsl_xcd_remap_loses_blocks,
    pick_flydsl_stage1_prefill_tile_n,
    pick_flydsl_stage1_prefill_xcd,
    resolve_flydsl_stage1_tile_n,
)

_TILE_BASE = {
    "gate_up_interleave": True,
    "a_dtype": "fp8",
    "b_dtype": "fp4",
    "out_dtype": "bf16",
    "k_batch": 1,
    "tile_k": 256,
    "tile_m": 128,
    "has_bias": False,
    "has_sorted_weights": False,
    "model_dim_pad": 0,
    "inter_dim_pad": 0,
    "act": "silu",
    "xcd_swizzle": 0,
    "gpu_arch": "gfx950",
}

_XCD_BASE = {
    "token_num": 16384,
    "inter_dim": 1152,
    "inter_dim_pad": 0,
    "model_dim": 5120,
    "tile_n": 128,
    "tile_k": 256,
    "grid_y": 2178,
    "persist_m": 1,
    "tile_m": 64,
    "experts": 385,
    "topk": 7,
    "gate_up_interleave": True,
    "a_dtype": "fp8",
    "b_dtype": "fp4",
    "out_dtype": "bf16",
    "k_batch": 1,
    "xcd_swizzle": 0,
    "act": "silu",
    "has_bias": False,
    "has_sorted_weights": False,
    "model_dim_pad": 0,
    "gpu_arch": "gfx950",
}


def _tile(**kw):
    args = dict(_TILE_BASE)
    args.update(kw)
    return pick_flydsl_stage1_prefill_tile_n(**args)


def _xcd(**kw):
    args = dict(_XCD_BASE)
    args.update(kw)
    return pick_flydsl_stage1_prefill_xcd(**args)


def test_stock_resolver_still_forces_128():
    # The historical resolver is unchanged: on the non-interleaved layout a
    # 256-wide tile on a non-256-aligned inter_dim is still illegal, and the
    # interleaved prefill picker is what re-opens 256 on the measured family.
    assert resolve_flydsl_stage1_tile_n(1152, 256) == 128
    assert resolve_flydsl_stage1_tile_n(640, 256) == 128
    assert resolve_flydsl_stage1_tile_n(384, 256) == 128
    assert resolve_flydsl_stage1_tile_n(1152, 128) == 128


def test_stock_resolver_warning_is_scoped_to_the_path(caplog):
    # On the interleaved path the resolver's 128 is not the final launch: the
    # prefill picker may re-open 256, so the warning must not say "runs 128".
    import aiter

    caplog.handler.setLevel(0)
    aiter.logger.addHandler(caplog.handler)
    try:
        assert resolve_flydsl_stage1_tile_n(1152, 256, interleaved=True) == 128
    finally:
        aiter.logger.removeHandler(caplog.handler)
    assert "may re-open tile_n=256" in caplog.text
    assert "actually runs 128" not in caplog.text


def test_wide_tile_on_measured_prefill_sizes():
    assert (
        _tile(
            token_num=8192,
            inter_dim=1152,
            model_dim=5120,
            experts=385,
            topk=7,
            tile_n=128,
        )
        == 256
    )
    assert (
        _tile(
            token_num=4096,
            inter_dim=640,
            model_dim=5120,
            experts=384,
            topk=6,
            tile_n=128,
        )
        == 256
    )
    assert (
        _tile(
            token_num=16384,
            inter_dim=640,
            model_dim=5120,
            experts=384,
            topk=6,
            tile_m=64,
            tile_n=128,
        )
        == 256
    )
    assert (
        _tile(
            token_num=32768,
            inter_dim=640,
            model_dim=5120,
            experts=384,
            topk=6,
            tile_m=64,
            tile_n=128,
        )
        == 256
    )
    # 16K+ K=7168/N=384: measured on the 64-row block.
    assert (
        _tile(
            token_num=16384,
            inter_dim=384,
            model_dim=7168,
            experts=385,
            topk=7,
            tile_m=64,
            tile_n=128,
        )
        == 256
    )
    # 16K+ K=5120/N=640 was also measured on the 128-row block (TP8 M16384).
    assert (
        _tile(
            token_num=16384,
            inter_dim=640,
            model_dim=5120,
            experts=384,
            topk=6,
            tile_m=128,
            tile_n=128,
        )
        == 256
    )
    # 16K+ on a wide gate/up axis keeps 128; the XCD rule gets its own test.
    assert (
        _tile(
            token_num=16384,
            inter_dim=1152,
            model_dim=5120,
            experts=385,
            topk=7,
            tile_n=128,
        )
        == 128
    )
    assert (
        _tile(
            token_num=32768,
            inter_dim=1152,
            model_dim=5120,
            experts=385,
            topk=7,
            tile_n=128,
        )
        == 128
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"token_num": 2048},  # below the 4K boundary
        {"token_num": 8193},  # between the two boundaries
        {"token_num": 16383},
        # 16K+ 128-row block was timed only for K5120/N640.
        {"token_num": 16384, "tile_m": 128, "model_dim": 7168, "inter_dim": 384},
        {"tile_n": 64},  # only widen a 128 tile
        {"tile_n": 256},  # already wide
        {"xcd_swizzle": 4},  # never re-grid a caller's XCD choice
        {"inter_dim_pad": 64},  # production's TP4 padded layer
        {"model_dim_pad": 128},
        {"act": "situv2"},  # Kimi-K3
        {"topk": 8},  # only topk 6/7 timed
        {"topk": 16},
        {"experts": 128},  # E128 DSV4 variant
        {"experts": 256},  # not timed
        {"experts": 512},  # not timed
        {"experts": 896},  # Kimi-K3 E896
        {"model_dim": 6144},  # MiMo / GLM / Mistral widths
        {"model_dim": 7168, "inter_dim": 640},  # not timed
        {"model_dim": 7168, "inter_dim": 1152},  # not timed
        {"out_dtype": "f16"},  # only bf16/fp8 timed
        {"out_dtype": "fp4"},  # fp4 epilogue not timed
        {"gpu_arch": "gfx942"},  # rules are gfx950-only
        {"gpu_arch": "gfx1250"},
        {"inter_dim": 512},
        {"inter_dim": 2304},
        {"has_bias": True},
        {"has_sorted_weights": True},
        {"k_batch": 2},
        {"a_dtype": "bf16"},
        {"b_dtype": "fp8"},
        # E256 is outside the measured E 384/385 family.
        {"experts": 256, "inter_dim": 384},
    ],
)
def test_rule_declines_everything_else(kw):
    args = {
        "token_num": 8192,
        "inter_dim": 1152,
        "model_dim": 5120,
        "experts": 385,
        "topk": 7,
        "tile_n": 128,
    }
    args.update(kw)
    assert _tile(**args) == args["tile_n"]


def test_xcd_grouping_only_where_safe():
    # 18 x 2178 workgroups: the four dropped ids are padding blocks.
    assert _xcd(token_num=16384) == 4
    assert _xcd(token_num=8192) == 0
    assert _xcd(xcd_swizzle=4) == 4
    # The grouping was measured on the 64-row block; a 128-row block loses.
    assert _xcd(tile_m=128) == 0
    assert _xcd(inter_dim_pad=64) == 0
    # inter_dim 640, tile_k 384: tile2_pad = 192 - 640 % 192 = 128 != 0.
    assert _xcd(inter_dim=640, tile_k=384) == 0
    # 3 x 3 workgroups drops the workgroup of M block 2, which the routing fills.
    assert _xcd(inter_dim=384, tile_n=256, grid_y=3, tile_m=64) == 0
    assert _xcd(model_dim=6144) == 0
    assert _xcd(act="situv2") == 0
    assert _xcd(model_dim_pad=128) == 0
    assert _xcd(has_bias=True) == 0
    assert _xcd(has_sorted_weights=True) == 0


def test_xcd_remap_loses_blocks():
    # Whole rounds over the 8 XCDs: nothing is dropped.
    assert not flydsl_xcd_remap_loses_blocks(16, 2, 4, 1, 32)
    # 18 x 2178 drops four workgroups; they are past 2,170 routed blocks.
    assert not flydsl_xcd_remap_loses_blocks(18, 2178, 4, 1, 2170)
    # 3 x 3 drops the workgroup of block 2, which a full routing fills.
    assert flydsl_xcd_remap_loses_blocks(3, 3, 4, 1, 9)


def test_measured_family_predicate():
    # The record shapes are in the measured family; decode and other models are not.
    assert flydsl_stage1_measured_prefill(
        model_dim=5120,
        inter_dim=1152,
        experts=385,
        topk=7,
        act="silu",
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        k_batch=1,
        gate_up_interleave=True,
        tile_m=64,
        inter_dim_pad=0,
        model_dim_pad=0,
        gpu_arch="gfx950",
    )
    assert flydsl_stage1_measured_prefill(
        model_dim=7168,
        inter_dim=384,
        experts=385,
        topk=7,
        act="silu",
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        k_batch=1,
        gate_up_interleave=True,
        tile_m=64,
        inter_dim_pad=0,
        model_dim_pad=0,
        gpu_arch="gfx950",
    )
    for kw in (
        {"model_dim": 6144},  # MiMo / GLM width
        {"inter_dim": 512},
        {"experts": 256},  # not timed
        {"experts": 512},  # not timed
        {"experts": 896},  # Kimi-K3
        {"topk": 8},  # not timed
        {"topk": 16},
        {"tile_m": 32},  # not timed
        {"out_dtype": "f16"},  # not timed
        {"out_dtype": "fp4"},  # not timed
        {"model_dim": 7168, "inter_dim": 640},  # not timed
        {"model_dim": 7168, "inter_dim": 1152},  # not timed
        {"act": "situv2"},
        {"inter_dim_pad": 64},  # production TP4 padded layer
        {"model_dim_pad": 128},
        {"gate_up_interleave": False},
        {"k_batch": 2},
        {"has_bias": True},
        {"has_sorted_weights": True},
        {"gpu_arch": "gfx942"},
        {"gpu_arch": "gfx1250"},
    ):
        args = {
            "model_dim": 5120,
            "inter_dim": 1152,
            "experts": 385,
            "topk": 7,
            "act": "silu",
            "a_dtype": "fp8",
            "b_dtype": "fp4",
            "out_dtype": "bf16",
            "k_batch": 1,
            "gate_up_interleave": True,
            "tile_m": 64,
            "inter_dim_pad": 0,
            "model_dim_pad": 0,
            "gpu_arch": "gfx950",
        }
        args.update(kw)
        assert not flydsl_stage1_measured_prefill(**args), kw


def test_cshuffle_budget_is_measured_wide_tile_only():
    # Existing 128/64-wide launches keep their historical kernel (and LDS use).
    assert flydsl_stage1_cshuffle_n(64, True, False, True) == 64
    assert flydsl_stage1_cshuffle_n(128, True, False, True) == 128
    # The measured family's interleaved non-split-K epilogue holds only the
    # activated half.
    assert flydsl_stage1_cshuffle_n(256, True, False, True) == 128
    # Outside the measured family the stock full-width budget is kept, even
    # for a 256-wide interleaved tile.
    assert flydsl_stage1_cshuffle_n(256, True, False, False) == 256
    # Split-K and separated epilogues write the full tile.
    assert flydsl_stage1_cshuffle_n(256, True, True, True) == 256
    assert flydsl_stage1_cshuffle_n(256, False, False, True) == 256


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
