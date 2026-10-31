# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks for MegaMoEGfx1250.required_vmm_bytes().

The arena the compact path allocates, including the stage-1 flag region when
``AITER_MEGA_S1F=1`` and the shape can build that kernel, has to fit in the
reservation. No GPU and no communicator.
"""

import pytest

from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import arena, transport
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.mega_moe import MegaMoEGfx1250
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.recipe import (
    CombineKind,
    DispatchKind,
    MegaMoEConfig,
    MegaMoEGeometry,
    MegaMoERecipe,
)

_GIB = 1 << 30


def _config(*, ws, hidden, mtpr, experts, topk, wire, fused, inter=2048):
    recipe = MegaMoERecipe(
        dispatch=DispatchKind.COMPACT if fused else DispatchKind.MORI,
        combine=CombineKind.FUSED_SCATTER,
        dispatch_wire=wire,
    )
    geometry = MegaMoEGeometry(
        rank=0,
        world_size=ws,
        model_dim=hidden,
        inter_dim=inter,
        experts=experts,
        topk=topk,
        max_tokens_per_rank=mtpr,
    )
    return MegaMoEConfig(recipe=recipe, geometry=geometry)


def _layout_bytes(config) -> int:
    layout = transport._arena_layout(config, transport._capabilities_for(config.recipe))
    return layout.total_bytes


def test_compact_reserve_matches_the_arena(monkeypatch):
    monkeypatch.delenv("AITER_MEGA_S1F", raising=False)
    monkeypatch.setenv("AITER_TDM_TILE_M", "64")
    config = _config(
        ws=4, hidden=7168, mtpr=1024, experts=256, topk=8, wire="fp4", fused=True
    )
    layout = transport._arena_layout(config, None)
    assert "tile_flag" not in dict(layout.regions)
    need = MegaMoEGfx1250.required_vmm_bytes(config.recipe, config.geometry)
    assert need == arena._cco_alloc_bytes(layout.total_bytes, 2 << 20)


def test_s1f_reserves_tile_flags(monkeypatch):
    monkeypatch.setenv("AITER_MEGA_S1F", "1")
    monkeypatch.setenv("CU_NUM", "256")
    monkeypatch.setenv("AITER_TDM_TILE_M", "64")
    get_cu_num.cache_clear()
    try:
        off = _config(
            ws=4, hidden=7168, mtpr=1024, experts=256, topk=8, wire="fp4", fused=True
        )
        # Same shape with the switch off, so the only added region is the flags.
        monkeypatch.setenv("AITER_MEGA_S1F", "0")
        bare = _layout_bytes(off)
        monkeypatch.setenv("AITER_MEGA_S1F", "1")
        config = _config(
            ws=4, hidden=7168, mtpr=1024, experts=256, topk=8, wire="fp4", fused=True
        )
        layout = transport._arena_layout(config, None)
        assert "tile_flag" in dict(layout.regions)
        assert layout.total_bytes > bare
        need = MegaMoEGfx1250.required_vmm_bytes(config.recipe, config.geometry)
        assert need == arena._cco_alloc_bytes(layout.total_bytes, 2 << 20)
        bound = MegaMoEGfx1250.required_vmm_bytes(
            world_size=4,
            hidden_dim=7168,
            max_tokens_per_rank=1024,
            experts=256,
            topk=8,
            stage1_fused=True,
        )
        assert bound >= need
    finally:
        get_cu_num.cache_clear()


def test_review_config_fits(monkeypatch):
    # ROCm/ATOM#2384 P1: EP4, fp8 wire, compact stage1 overflowed a 4 GiB slot.
    monkeypatch.delenv("AITER_MEGA_S1F", raising=False)
    monkeypatch.setenv("AITER_TDM_TILE_M", "256")
    config = _config(
        ws=4, hidden=7168, mtpr=16384, experts=384, topk=6, wire="fp8", fused=True
    )
    actual = _layout_bytes(config)
    reserve = MegaMoEGfx1250.required_vmm_bytes(config.recipe, config.geometry)
    assert actual > 4 * _GIB
    assert reserve >= actual


def test_fused_reserves_route_rows(monkeypatch):
    monkeypatch.delenv("AITER_MEGA_S1F", raising=False)
    monkeypatch.setenv("AITER_TDM_TILE_M", "256")
    wide = _layout_bytes(
        _config(
            ws=4, hidden=7168, mtpr=16384, experts=384, topk=6, wire="fp4", fused=True
        )
    )
    monkeypatch.setenv("AITER_TDM_TILE_M", "16")
    narrow = _layout_bytes(
        _config(
            ws=4, hidden=7168, mtpr=16384, experts=384, topk=6, wire="fp4", fused=True
        )
    )
    # Compact rows are padded per expert to the GEMM tile, so the wider tile
    # reserves more of the arena than the narrow one.
    assert wide > narrow


def test_rejects_indivisible_experts():
    with pytest.raises(ValueError):
        MegaMoEGfx1250.required_vmm_bytes(
            world_size=3,
            hidden_dim=7168,
            max_tokens_per_rank=1024,
            experts=385,
            topk=6,
            stage1_fused=False,
        )
