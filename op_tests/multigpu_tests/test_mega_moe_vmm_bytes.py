# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks for MegaMoEGfx1250.required_vmm_bytes().

Rebuilds the arena exactly as _initialize_pipeline lays it out for every
dispatch wire, compact tile and backend, and asserts the bf16-pitch bound that
required_vmm_bytes() reserves covers it. No GPU and no communicator needed.
"""

import itertools

import pytest

from aiter.ops.flydsl.kernels.mega_moe_gfx1250.compact_plan import (
    compact_hist_stride,
)
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.mega_moe import (
    _MAX_COMPACT_TILE_M,
    _VMM_SLACK_BYTES,
    MegaMoEConfig,
    MegaMoEGfx1250,
    _align_up,
    _arena_bound_nbytes,
    _arena_nbytes,
    _arena_regions,
)

_GIB = 1 << 30


def _mori_available() -> bool:
    try:
        from mori.ops.dispatch_combine_v2.hip_backend import (  # noqa: F401
            scale_stride_bytes,
        )
    except ImportError:
        return False
    return True


def _actual_arena_bytes(
    *, ws, hidden, mtpr, experts, topk, wire, fused, tile_m, backend
) -> int:
    epr = experts // ws
    config = MegaMoEConfig(
        rank=0,
        world_size=ws,
        hidden_dim=hidden,
        max_tokens_per_rank=mtpr,
        experts_per_rank=epr,
        topk=topk,
        dispatch_wire=wire,
        dispatch_backend=backend,
        stage1_fused=fused,
    )
    recv_rows = config.compact_row_cap(tile_m) if fused else config.max_recv
    wire_row = (
        _align_up(config.dispatch_token_nbytes + config.dispatch_scale_nbytes, 128)
        if fused
        else config.dispatch_token_nbytes
    )
    hist_stride = compact_hist_stride(
        npes=ws, experts_per_rank=epr, max_routes=mtpr * topk
    )
    return _arena_nbytes(
        _arena_regions(
            config,
            compact=fused,
            recv_rows=recv_rows,
            wire_row=wire_row,
            scale_row=config.dispatch_scale_dst_nbytes,
            hist_stride=hist_stride,
        )
    )


def _cases():
    backends = ["flydsl"] + (["mori"] if _mori_available() else [])
    for ws, topk, hidden, mtpr, experts in itertools.product(
        (2, 4, 8), (4, 6, 8), (4096, 7168), (1024, 16384), (256, 384)
    ):
        for wire, fused, tile_m, backend in itertools.product(
            ("bf16", "fp8", "fp4"), (False, True), (16, 32, 64, 128, 256), backends
        ):
            if fused and (wire == "bf16" or backend != "flydsl"):
                continue
            if not fused and tile_m != 64:
                continue
            yield {
                "ws": ws,
                "hidden": hidden,
                "mtpr": mtpr,
                "experts": experts,
                "topk": topk,
                "wire": wire,
                "fused": fused,
                "tile_m": tile_m,
                "backend": backend,
            }


def test_bound_covers_every_arena():
    for case in _cases():
        actual = _actual_arena_bytes(**case)
        bound = _arena_bound_nbytes(
            world_size=case["ws"],
            hidden_dim=case["hidden"],
            max_tokens_per_rank=case["mtpr"],
            experts_per_rank=case["experts"] // case["ws"],
            topk=case["topk"],
            stage1_fused=case["fused"],
        )
        reserve = MegaMoEGfx1250.required_vmm_bytes(
            world_size=case["ws"],
            hidden_dim=case["hidden"],
            max_tokens_per_rank=case["mtpr"],
            experts=case["experts"],
            topk=case["topk"],
            stage1_fused=case["fused"],
        )
        assert actual <= bound, (case, actual, bound)
        assert reserve >= bound + _VMM_SLACK_BYTES, (case, reserve, bound)


def test_review_config_fits():
    # ROCm/ATOM#2384 P1: EP4, fp8 wire, compact stage1 overflowed a 4 GiB slot.
    geom = {"ws": 4, "hidden": 7168, "mtpr": 16384, "experts": 384, "topk": 6}
    reserve = MegaMoEGfx1250.required_vmm_bytes(
        world_size=4,
        hidden_dim=7168,
        max_tokens_per_rank=16384,
        experts=384,
        topk=6,
        stage1_fused=True,
    )
    for tile_m in (16, 32, 64, 128, _MAX_COMPACT_TILE_M):
        actual = _actual_arena_bytes(
            **geom, wire="fp8", fused=True, tile_m=tile_m, backend="flydsl"
        )
        assert actual > 4 * _GIB
        assert reserve >= actual + _VMM_SLACK_BYTES


def test_fused_reserves_route_rows():
    kwargs = {
        "world_size": 4,
        "hidden_dim": 7168,
        "max_tokens_per_rank": 16384,
        "experts": 384,
        "topk": 6,
    }
    fused = MegaMoEGfx1250.required_vmm_bytes(**kwargs, stage1_fused=True)
    token_major = MegaMoEGfx1250.required_vmm_bytes(**kwargs, stage1_fused=False)
    assert fused > token_major


def test_rejects_indivisible_experts():
    with pytest.raises(ValueError):
        MegaMoEGfx1250.required_vmm_bytes(
            world_size=3,
            hidden_dim=7168,
            max_tokens_per_rank=1024,
            experts=384 + 1,
            topk=6,
            stage1_fused=False,
        )


if __name__ == "__main__":
    test_bound_covers_every_arena()
    test_review_config_fits()
    test_fused_reserves_route_rows()
    test_rejects_indivisible_experts()
    print("ok")
