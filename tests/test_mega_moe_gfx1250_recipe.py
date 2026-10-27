# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEGfx1250's recipe, its checks, and the arena it sizes -- no GPU needed.

The multi-GPU behaviour lives in op_tests/multigpu_tests/test_mega_moe_gfx1250.py;
this covers what is pure: how the legacy flags map to a recipe, what is
rejected and with which words, and that the one arena layout matches the
region list ROCm/aiter#6430 built inline, across a grid of geometries and
against a mori with and without Plan.describe().
"""

import dataclasses
import itertools
from types import SimpleNamespace

import pytest

try:
    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import mega_moe as MM
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.compact_plan import (
        compact_done_nbytes,
        compact_hist_stride,
    )
except Exception as error:  # noqa: BLE001  (no flydsl, no aiter build, ...)
    pytest.skip(f"MegaMoE does not import here: {error}", allow_module_level=True)

COMPACT_TILE = 64


# -- a mori, faked: the helpers MegaMoE reads, with mori's own rules -----------
def _align(n, a):
    return (n + a - 1) // a * a


def _push_wire(ns):
    return _align(ns.hidden_dim // 2 + ns.hidden_dim // 32, 128)


def _push_land_rows(ns):
    return ns.max_num_inp_token_per_rank * min(ns.world_size, ns.num_experts_per_token)


def _lookup(ws, hidden, topk, dtype, experts_per_rank, quant_type="none"):
    return {
        "schedule": [(32, 64, 4, 64, 8), (None, 128, 8, 128, 16)],
        "dispatch_block_num": 64,
        "warp_num_per_block": 8,
        "combine_block_num": 64,
        "combine_warp_num_per_block": 8,
    }


class _DescribedPlan:
    """EpDispatchPlan / EpCombinePlan.describe() after ROCm/mori#844, mirroring
    the EpRegion* sizes in ep_cfg.hpp."""

    dispatch = True

    @classmethod
    def describe(cls, *, arch=None, **kw):
        assert arch == "gfx1250"
        ws, mr, topk = kw["world_size"], kw["max_recv"], kw["num_expert_per_token"]
        stride = _align(kw.get("scale_bytes", 0), 128)
        info = {
            "region.tokOff": 4,
            "region.recvNum": 8 * ws,
            "region.recvToSrc": mr * 4,
            "region.outIdx": mr * topk * 4,
            "region.outWts": mr * topk * 4,
            "region.xdb": 8 * ws,
        }
        if cls.dispatch:
            elem = 2 if str(kw["dtype"]) == "torch.bfloat16" else 1
            info["region.dispOut"] = mr * kw["hidden_dim"] * elem
            info["region.outScales"] = mr * stride
            info["stagingBytes"] = 4096
            return info
        out = mr * kw["hidden_dim"] * 2
        if kw.get("combine_fp4"):
            ns = SimpleNamespace(
                hidden_dim=kw["hidden_dim"],
                max_num_inp_token_per_rank=kw["max_tok_per_rank"],
                world_size=ws,
                num_experts_per_token=topk,
            )
            rows = _push_land_rows(ns)
            out += rows * _push_wire(ns)
            # EpCombinePushOvlAny: no overlapped reduce on a wide EP.
            if 4 <= ws <= 32:
                out += rows * 8
        info["region.outTok"] = out
        return info


class _DescribedCombine(_DescribedPlan):
    dispatch = False


def _arena_regions(*infos, region_names=None):
    need = {}
    for info in infos:
        for k, v in info.items():
            if k.startswith("region."):
                need[k[7:]] = max(need.get(k[7:], 0), v)
    return [(region_names.get(k, k), n) for k, n in need.items() if n]


def _fake_mori(describe):
    ep_plans = SimpleNamespace(
        EpDispatchPlan=_DescribedPlan if describe else object,
        EpCombinePlan=_DescribedCombine if describe else object,
        arena_regions=_arena_regions,
    )
    return MM._MoriCapabilities(
        ep_plans=ep_plans,
        hip_backend=SimpleNamespace(_XDB_FLAG_SLOTS=256),
        lookup=_lookup,
        scale_stride_bytes=lambda n: _align(n, 128),
        push_wire_nbytes=_push_wire,
        push_land_rows=_push_land_rows,
        ep_staging_bytes=lambda ws, mr, sb, mtpr: 4096,
        tokoff_wanted=lambda ws: False,
        self_first_enabled=None,
        plan_describe=describe,
    )


@pytest.fixture(params=[False, True], ids=["formulas", "describe"])
def mori(request, monkeypatch):
    caps = _fake_mori(request.param)
    monkeypatch.setattr(MM, "_mori_capabilities", lambda: caps)
    monkeypatch.setenv("AITER_TDM_TILE_M", str(COMPACT_TILE))
    monkeypatch.delenv("AITER_TDM_TILE_M2", raising=False)
    return caps


# -- ROCm/aiter#6430's region list, verbatim but for its inputs ------------------
def _pr6430_regions(config, mori, tile_m):
    max_recv = config.max_recv
    compact_plan = config.stage1_fused
    compact_cap = config.compact_row_cap(tile_m) if compact_plan else max_recv
    recv_rows = compact_cap if compact_plan else max_recv
    compact_wire_row = (
        _align(config.dispatch_token_nbytes + config.dispatch_scale_nbytes, 128)
        if compact_plan
        else config.dispatch_token_nbytes
    )
    hist_stride = compact_hist_stride(
        npes=config.world_size,
        experts_per_rank=config.experts_per_rank,
        max_routes=config.max_tokens_per_rank * config.topk,
    )
    arena_regions = [
        ("tok_off", 4),
        ("recv_num", 2 * config.world_size * 4),
        ("recv_to_src_token", max_recv * 4),
        ("out_idx", max_recv * config.topk * 4),
        ("out_wts", max_recv * config.topk * 4),
        ("disp_out", recv_rows * compact_wire_row),
        ("cross_device_barrier", config.world_size * 8),
    ]
    if config.dispatch_scale_dst_nbytes and not compact_plan:
        arena_regions.append(
            ("disp_out_scales", recv_rows * config.dispatch_scale_dst_nbytes)
        )
    if compact_plan:
        arena_regions.extend(
            [
                ("ep_rowmap", (recv_rows + 1) * 8),
                ("compact_hist", 2 * hist_stride * 4),
                ("compact_done", compact_done_nbytes()),
            ]
        )
    if config.stage2_fused:
        comb_inp_nbytes = (
            config.max_tokens_per_rank * config.topk * config.combine_slot_stride_bytes
        )
    else:
        comb_inp_nbytes = max_recv * config.hidden_dim * 2
        if config.combine_quant_bits:
            landing_rows = mori.push_landing_rows(config)
            if landing_rows is None:
                landing_rows = config.world_size * max_recv
            comb_inp_nbytes += landing_rows * mori.push_wire_bytes(config)
            if config.world_size >= 4:
                comb_inp_nbytes += landing_rows * 8
    arena_regions.append(("comb_inp", comb_inp_nbytes))
    return arena_regions


PAIRINGS = [
    # (stage1_fused, stage2_fused, dispatch_wire, combine_quant)
    *[
        (False, True, w, q)
        for w in ("bf16", "fp8", "fp4")
        for q in MM._COMBINE_QUANT_MODES
    ],
    *[(False, False, w, q) for w in ("bf16", "fp8", "fp4") for q in ("none", "mxfp4")],
    *[(True, True, w, q) for w in ("fp8", "fp4") for q in MM._COMBINE_QUANT_MODES],
]
GEOMETRIES = list(
    itertools.product(
        (2, 4, 8, 64),  # world_size
        (2048, 7168),  # hidden
        (4, 8),  # topk
        (1, 16, 256),  # max_tokens_per_rank
        (8, 32),  # experts_per_rank
    )
)


def _config(s1, s2, wire, cq, ws, hidden, topk, mtpr, epr, rank=0):
    recipe = MM.MegaMoERecipe.from_legacy(
        stage1_fused=s1, stage2_fused=s2, dispatch_wire=wire, combine_quant=cq
    )
    geometry = MM.MegaMoEGeometry(
        rank=rank,
        world_size=ws,
        model_dim=hidden,
        inter_dim=1024,
        experts=ws * epr,
        topk=topk,
        max_tokens_per_rank=mtpr,
    )
    return MM.MegaMoEConfig(recipe=recipe, geometry=geometry)


def _cases():
    for pairing, geometry in itertools.product(PAIRINGS, GEOMETRIES):
        try:
            yield pairing, geometry, _config(*pairing, *geometry)
        except ValueError:
            continue  # a shape this pairing refuses, e.g. mori's fp4 at hidden 2048


def test_layout_matches_pr6430(mori):
    checked = 0
    for (s1, s2, _, cq), geometry, config in _cases():
        compact = MM._CompactGeometry.of(config) if s1 else None
        got = MM._arena_layout(config, mori, compact)
        want = MM._ArenaLayout.of(_pr6430_regions(config, mori, COMPACT_TILE))
        ws = geometry[0]
        if mori.plan_describe and not s2 and cq == "mxfp4" and ws > 32:
            # mori sizes no overlapped-reduce flags on a wide EP; #6430 did.
            assert got.total_bytes < want.total_bytes
            continue
        assert got == want, (s1, s2, cq, geometry)
        checked += 1
    assert checked > 500


def test_required_vmm_bytes(mori):
    for (s1, s2, wire, cq), geometry, config in _cases():
        ws, hidden, topk, mtpr, epr = geometry
        need = MM.MegaMoEGfx1250.required_vmm_bytes(config.recipe, config.geometry)
        layout = MM._arena_layout(config, None if s1 else mori)
        assert need == MM._cco_alloc_bytes(layout.total_bytes, 2 << 20)
        assert need % (2 << 20) == 0 and need >= layout.total_bytes
        # The keyword form names neither the wires nor the combine.
        bound = MM.MegaMoEGfx1250.required_vmm_bytes(
            world_size=ws,
            hidden_dim=hidden,
            max_tokens_per_rank=mtpr,
            experts=ws * epr,
            topk=topk,
            stage1_fused=s1,
        )
        assert bound >= need, ((s1, s2, wire, cq), geometry)


def test_cco_rounding():
    g = 2 << 20
    assert MM._cco_alloc_bytes(1, g) == g
    assert MM._cco_alloc_bytes(g, g) == g
    assert MM._cco_alloc_bytes(g + 1, g) == 2 * g
    big = (2 << 30) + 1
    assert MM._cco_alloc_bytes(big, 1 << 20) == _align(big, 2 << 20)


def test_from_legacy():
    R, D, C = MM.MegaMoERecipe, MM.DispatchKind, MM.CombineKind
    assert R.from_legacy(dispatch_wire="fp4") == R(dispatch_wire="fp4")
    r = R.from_legacy(stage1_fused=True, dispatch_wire="fp8", combine_quant="mxfp4")
    assert (r.dispatch, r.combine, r.combine_wires) == (
        D.COMPACT,
        C.FUSED_SCATTER,
        ("bf16", "fp4"),
    )
    r = R.from_legacy(stage2_fused=False, dispatch_backend="mori", dispatch_wire="bf16")
    assert (r.dispatch, r.combine, r.combine_wires) == (D.MORI, C.MORI, ("bf16",))
    assert r.combine_quant == "none"


def test_from_legacy_reads_env(monkeypatch):
    monkeypatch.setenv("MEGA_DISPATCH_WIRE", "fp8")
    monkeypatch.setenv("MEGA_COMBINE_QUANT", "mxfp8")
    r = MM.MegaMoERecipe.from_legacy()
    assert r.dispatch_wire == "fp8" and r.combine_wires == ("bf16", "fp8")
    # Named arguments win over the environment.
    r = MM.MegaMoERecipe.from_legacy(dispatch_wire="bf16", combine_quant="none")
    assert r.dispatch_wire == "bf16" and r.combine_wires == ("bf16",)


def test_recipe_is_a_cache_key():
    R = MM.MegaMoERecipe
    a = R(dispatch_wire="fp4", combine_wires=("fp4", "bf16"))
    b = R(
        dispatch="mori",
        combine="fused_scatter",
        dispatch_wire="fp4",
        combine_wires=["bf16", "fp4"],
        gate_mode="interleave",
    )
    assert a == b and hash(a) == hash(b) and a.combine_wires == ("bf16", "fp4")
    assert len({a, b, R(dispatch_wire="fp8")}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.dispatch_wire = "bf16"


REJECTED = [
    (
        {"stage1_fused": True, "stage2_fused": False, "dispatch_wire": "fp4"},
        (
            "stage1_fused requires stage2_fused: the compact plan groups recv rows per "
            "expert, and only the gemm2 scatter maps them back to their tokens"
        ),
    ),
    (
        {"stage1_fused": True, "dispatch_wire": "bf16"},
        "stage1_fused requires an fp8 or fp4 dispatch wire",
    ),
    (
        {"stage2_fused": False, "combine_quant": "mxfp8"},
        (
            "stage2_fused=False combines through mori, whose wires are ('none', "
            "'mxfp4'); got combine_quant='mxfp8'"
        ),
    ),
    (
        {"stage1_fused": False, "dispatch_backend": "flydsl"},
        (
            "dispatch_backend='flydsl' does not match stage1_fused=False: the flydsl "
            "dispatch runs only as the compact plan (stage1_fused=True), and "
            "stage1_fused=False dispatches through mori"
        ),
    ),
    (
        {"dispatch_backend": "tdm"},
        "dispatch_backend must be 'flydsl' or 'mori', got 'tdm'",
    ),
    (
        {"combine_quant": "fp4"},
        "combine_quant must be one of ('none', 'mxfp8', 'mxfp4'), got 'fp4'",
    ),
    (
        {"dispatch_wire": "int8"},
        "dispatch_wire must be one of ('bf16', 'fp8', 'fp4'), got 'int8'",
    ),
    (
        {"gate_mode": "separated"},
        "MegaMoE fused stage2 scatter requires gate_mode=INTERLEAVE",
    ),
    (
        {"quant_type": QuantType.per_Token},
        "MegaMoE fused stage2 scatter requires quant_type=per_1x32",
    ),
    (
        {"activation": ActivationType.Gelu},
        "MegaMoE fused stage2 scatter does not support activation=",
    ),
]


@pytest.mark.parametrize("flags,message", REJECTED)
def test_rejected(flags, message):
    with pytest.raises(ValueError) as error:
        MM.MegaMoERecipe.from_legacy(**flags)
    assert message in str(error.value)


def test_rejected_by_geometry(mori):
    with pytest.raises(ValueError, match="mori's fp4 combine needs hidden_dim % 1024"):
        _config(False, False, "bf16", "mxfp4", 4, 1536, 8, 16, 8)
    with pytest.raises(ValueError, match=r"rank=4 must be in \[0, 4\)"):
        _config(False, True, "bf16", "none", 4, 7168, 8, 16, 8, rank=4)
    with pytest.raises(ValueError, match="must be divisible by world_size"):
        MM.MegaMoEGeometry(
            rank=0,
            world_size=3,
            model_dim=7168,
            inter_dim=1024,
            experts=8,
            topk=2,
            max_tokens_per_rank=4,
        )
