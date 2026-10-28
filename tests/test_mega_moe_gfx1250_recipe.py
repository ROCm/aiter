# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEGfx1250's recipe, its checks, and the arena it sizes -- no GPU needed.

The multi-GPU behaviour lives in op_tests/multigpu_tests/test_mega_moe_gfx1250.py;
this covers what is pure: the recipe and how the legacy flags map to it, what is
rejected and with which words, the arena layout over a grid of geometries
against a mori faked from what its plans describe(), required_vmm_bytes, and
the recv-token bound both dispatches check.
"""

import dataclasses
import itertools
import sys
import types

import pytest
import torch

pytest.importorskip("flydsl", reason="MegaMoE is built with FlyDSL")

from aiter import ActivationType, QuantType
from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import (
    arena,
    mori_caps,
    recipe,
    transport,
)
from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import mega_moe as MM
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.compact_plan import (
    compact_done_nbytes,
    compact_hist_stride,
    compact_row_capacity,
)

D, C = MM.DispatchKind, MM.CombineKind
COMPACT_TILE = 64


# -- a mori, faked: what its plans describe(), by the EpRegion* rules ----------
def _align(n, a):
    return (n + a - 1) // a * a


def _push_land_rows(ws, mtpr, topk):
    return mtpr * min(ws, topk)


def _push_wire(hidden):
    return _align(hidden // 2 + hidden // 32, 128)


class _DescribedPlan:
    dispatch = True

    @classmethod
    def describe(cls, *, arch=None, **kw):
        assert arch == "gfx1250"
        ws, mr, topk = kw["world_size"], kw["max_recv"], kw["num_expert_per_token"]
        info = {
            "region.tokOff": 4,
            "region.recvNum": 8 * ws,
            "region.recvToSrc": mr * 4,
            "region.outIdx": mr * topk * 4,
            "region.outWts": mr * topk * 4,
            "region.xdb": 8 * ws,
        }
        if cls.dispatch:
            stride = _align(kw["scale_bytes"], 128)
            elem = 2 if kw["dtype"] == torch.bfloat16 else 1
            info["region.dispOut"] = mr * kw["hidden_dim"] * elem
            info["region.outScales"] = mr * stride
            info["stagingBytes"] = 4096
            info["scaleStride"] = stride
            return info
        out = mr * kw["hidden_dim"] * 2
        if kw["combine_fp4"]:
            rows = _push_land_rows(ws, kw["max_tok_per_rank"], topk)
            out += rows * _push_wire(kw["hidden_dim"])
            # No overlapped-reduce flags on a wide EP.
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


def _lookup(ws, hidden, topk, dtype, experts_per_rank, quant_type="none"):
    return {
        "schedule": [(32, 64, 4, 64, 8), (None, 128, 8, 128, 16)],
        "dispatch_block_num": 64,
        "warp_num_per_block": 8,
        "combine_block_num": 64,
        "combine_warp_num_per_block": 8,
    }


def _install_mori(monkeypatch, describe=True):
    """Fake mori's EP modules, so _mori_capabilities() probes them."""
    plans = types.SimpleNamespace(
        EpDispatchPlan=_DescribedPlan if describe else type("Plan", (), {}),
        EpCombinePlan=_DescribedCombine if describe else type("Plan", (), {}),
    )
    if describe:
        plans.arena_regions = _arena_regions
    backend = types.SimpleNamespace(
        _XDB_FLAG_SLOTS=256,
        TokOffExt=types.SimpleNamespace(wanted=lambda world: False),
    )
    package = types.ModuleType("mori.ops.dispatch_combine_v2")
    package.ep_plans, package.hip_backend = plans, backend
    modules = {
        "mori": types.ModuleType("mori"),
        "mori.ops": types.ModuleType("mori.ops"),
        "mori.ops.dispatch_combine_v2": package,
        "mori.ops.dispatch_combine_v2.ep_plans": plans,
        "mori.ops.dispatch_combine_v2.hip_backend": backend,
        "mori.ops.dispatch_combine_v2.hip_tuning_configs": types.SimpleNamespace(
            lookup=_lookup
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    mori_caps._mori_capabilities.cache_clear()


@pytest.fixture
def mori(monkeypatch):
    _install_mori(monkeypatch)
    monkeypatch.setenv("AITER_TDM_TILE_M", str(COMPACT_TILE))
    monkeypatch.delenv("AITER_TDM_TILE_M2", raising=False)
    yield mori_caps._mori_capabilities()
    mori_caps._mori_capabilities.cache_clear()


# -- the layout, region by region, in the arena's order --------------------------
def _reference_regions(config, tile_m):
    """Every region the pairing lays out, sized by hand."""
    recipe, ws = config.recipe, config.world_size
    max_recv, topk = config.max_recv, config.topk
    compact = recipe.dispatch is D.COMPACT
    if compact:
        rows = compact_row_capacity(
            max_recv=max_recv,
            topk=topk,
            experts_per_rank=config.experts_per_rank,
            tile_m=tile_m,
        )
        wire_row = _align(
            config.dispatch_token_nbytes + config.dispatch_scale_nbytes, 128
        )
    else:
        rows, wire_row = max_recv, config.dispatch_token_nbytes
    regions = [
        ("tok_off", 4),
        ("recv_num", 2 * ws * 4),
        ("recv_to_src_token", max_recv * 4),
        ("out_idx", max_recv * topk * 4),
        ("out_wts", max_recv * topk * 4),
        ("disp_out", rows * wire_row),
        ("cross_device_barrier", ws * 8),
    ]
    if config.is_quant_dispatch_wire and not compact:
        regions.append(
            ("disp_out_scales", rows * _align(config.dispatch_scale_nbytes, 128))
        )
    if compact:
        hist = compact_hist_stride(
            npes=ws,
            experts_per_rank=config.experts_per_rank,
            max_routes=config.max_tokens_per_rank * topk,
        )
        regions += [
            ("ep_rowmap", (rows + 1) * 8),
            ("compact_hist", 2 * hist * 4),
            ("compact_done", compact_done_nbytes()),
        ]
    if recipe.combine is C.FUSED_SCATTER:
        comb_inp = config.max_tokens_per_rank * topk * config.combine_slot_stride_bytes
    else:
        comb_inp = max_recv * config.hidden_dim * 2
        if config.combine_quant_bits:
            land = _push_land_rows(ws, config.max_tokens_per_rank, topk)
            comb_inp += land * _push_wire(config.hidden_dim)
            if 4 <= ws <= 32:
                comb_inp += land * 8
    regions.append(("comb_inp", comb_inp))
    return regions


PAIRINGS = [
    # (dispatch, combine, dispatch_wire, combine_wires)
    *[
        (D.MORI, C.FUSED_SCATTER, w, cw)
        for w in ("bf16", "fp8", "fp4")
        for cw in (("bf16",), ("bf16", "fp8"), ("bf16", "fp4"))
    ],
    *[
        (D.MORI, C.MORI, w, cw)
        for w in ("bf16", "fp8", "fp4")
        for cw in (("bf16",), ("bf16", "fp4"))
    ],
    *[
        (D.COMPACT, C.FUSED_SCATTER, w, cw)
        for w in ("fp8", "fp4")
        for cw in (("bf16",), ("bf16", "fp8"), ("bf16", "fp4"))
    ],
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


def _geometry(ws, hidden, topk, mtpr, epr, rank=0):
    return MM.MegaMoEGeometry(
        rank=rank,
        world_size=ws,
        model_dim=hidden,
        inter_dim=1024,
        experts=ws * epr,
        topk=topk,
        max_tokens_per_rank=mtpr,
    )


def _config(dispatch, combine, wire, wires, *geometry, rank=0):
    recipe = MM.MegaMoERecipe(
        dispatch=dispatch, combine=combine, dispatch_wire=wire, combine_wires=wires
    )
    return MM.MegaMoEConfig(recipe=recipe, geometry=_geometry(*geometry, rank=rank))


def _cases():
    for pairing, geometry in itertools.product(PAIRINGS, GEOMETRIES):
        try:
            yield pairing, geometry, _config(*pairing, *geometry)
        except ValueError:
            continue  # a shape this pairing refuses, e.g. mori's fp4 at hidden 2048


def test_layout(mori):
    checked = 0
    for pairing, geometry, config in _cases():
        got = transport._arena_layout(
            config, transport._capabilities_for(config.recipe)
        )
        want = arena._ArenaLayout.of(_reference_regions(config, COMPACT_TILE))
        assert got == want, (pairing, geometry)
        checked += 1
    assert checked > 500


def test_required_vmm_bytes(mori):
    for pairing, geometry, config in _cases():
        ws, hidden, topk, mtpr, epr = geometry
        need = MM.MegaMoEGfx1250.required_vmm_bytes(config.recipe, config.geometry)
        layout = transport._arena_layout(
            config, transport._capabilities_for(config.recipe)
        )
        assert need == arena._cco_alloc_bytes(layout.total_bytes, 2 << 20)
        assert need % (2 << 20) == 0 and need >= layout.total_bytes
        # The keyword form names neither the wires nor the combine.
        bound = MM.MegaMoEGfx1250.required_vmm_bytes(
            world_size=ws,
            hidden_dim=hidden,
            max_tokens_per_rank=mtpr,
            experts=ws * epr,
            topk=topk,
            stage1_fused=config.recipe.dispatch is D.COMPACT,
        )
        assert bound >= need, (pairing, geometry)


@pytest.mark.parametrize("stage1_fused", [False, True])
def test_keyword_bound_refuses_a_shape_no_recipe_runs(mori, stage1_fused):
    # hidden 100: no quantizing wire (100 % 32), and bf16 rows are not 16 B aligned.
    with pytest.raises(ValueError, match="no recipe with stage1_fused="):
        MM.MegaMoEGfx1250.required_vmm_bytes(
            world_size=4,
            hidden_dim=100,
            max_tokens_per_rank=16,
            experts=32,
            topk=8,
            stage1_fused=stage1_fused,
        )


def test_mori_without_describe_is_refused(monkeypatch):
    _install_mori(monkeypatch, describe=False)
    try:
        with pytest.raises(RuntimeError, match=r"ROCm/mori#844"):
            mori_caps._mori_capabilities()
    finally:
        mori_caps._mori_capabilities.cache_clear()


def test_cco_rounding():
    g = 2 << 20
    assert arena._cco_alloc_bytes(1, g) == g
    assert arena._cco_alloc_bytes(g, g) == g
    assert arena._cco_alloc_bytes(g + 1, g) == 2 * g
    big = (2 << 30) + 1
    assert arena._cco_alloc_bytes(big, 1 << 20) == _align(big, 2 << 20)


# -- the recipe ------------------------------------------------------------------
def test_from_legacy():
    R = MM.MegaMoERecipe
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
    r = R.from_legacy(swiglu_limit=10, hidden_pad=64)
    assert r.compute == MM.ExpertComputeConfig(swiglu_limit=10.0, hidden_pad=64)


def test_from_legacy_reads_env(monkeypatch):
    monkeypatch.setenv("MEGA_DISPATCH_WIRE", "fp8")
    monkeypatch.setenv("MEGA_COMBINE_QUANT", "mxfp8")
    r = MM.MegaMoERecipe.from_legacy()
    assert r.dispatch_wire == "fp8" and r.combine_wires == ("bf16", "fp8")
    # Named arguments win over the environment.
    r = MM.MegaMoERecipe.from_legacy(dispatch_wire="bf16", combine_quant="none")
    assert r.dispatch_wire == "bf16" and r.combine_wires == ("bf16",)


def test_recipe_is_a_cache_key():
    R, X = MM.MegaMoERecipe, MM.ExpertComputeConfig
    a = R(dispatch_wire="fp4", combine_wires=("fp4", "bf16"))
    b = R(
        dispatch="mori",
        combine="fused_scatter",
        dispatch_wire="fp4",
        combine_wires=["bf16", "fp4"],
        compute=X(gate_mode="interleave", swiglu_limit=0),
    )
    assert a == b and hash(a) == hash(b) and a.combine_wires == ("bf16", "fp4")
    assert len({a, b, R(dispatch_wire="fp8")}) == 2
    assert R(compute=X(swiglu_limit=7)) != R()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.dispatch_wire = "bf16"
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.compute.swiglu_limit = 1.0


REJECTED = [
    (
        {"dispatch": "compact", "combine": "mori", "dispatch_wire": "fp4"},
        (
            "combine=mori does not accept the expert_grouped rows "
            "dispatch=compact delivers; combine=fused_scatter does"
        ),
    ),
    (
        {"dispatch": "compact", "dispatch_wire": "bf16"},
        "dispatch=compact carries an fp8 or fp4 dispatch_wire, got 'bf16'",
    ),
    (
        {"combine": "mori", "combine_wires": ("bf16", "fp8")},
        (
            "combine=mori offers the ('bf16', 'fp4') wires, "
            "got combine_wires=('bf16', 'fp8')"
        ),
    ),
    (
        {"combine_wires": ("fp4",)},
        "combine_wires must be ('bf16',) or bf16 plus one of fp8/fp4",
    ),
    (
        {"dispatch_wire": "int8"},
        "dispatch_wire must be one of ('bf16', 'fp8', 'fp4'), got 'int8'",
    ),
]


@pytest.mark.parametrize("fields,message", REJECTED)
def test_recipe_rejected(fields, message):
    with pytest.raises(ValueError) as error:
        MM.MegaMoERecipe(**fields)
    assert message in str(error.value)


COMPUTE_REJECTED = [
    ({"gate_mode": "separated"}, "requires gate_mode=INTERLEAVE"),
    ({"quant_type": QuantType.per_Token}, "requires quant_type=per_1x32"),
    ({"activation": ActivationType.Gelu}, "does not support activation="),
    ({"hidden_pad": -1}, "padding must be non-negative"),
    ({"swiglu_limit": -1.0}, "swiglu_limit must be non-negative"),
]


@pytest.mark.parametrize("fields,message", COMPUTE_REJECTED)
def test_compute_rejected(fields, message):
    with pytest.raises(ValueError, match=message):
        MM.ExpertComputeConfig(**fields)


LEGACY_REJECTED = [
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
    # The flags become a recipe, which is checked as any recipe is.
    (
        {"stage1_fused": True, "stage2_fused": False, "dispatch_wire": "fp4"},
        "combine=mori does not accept the expert_grouped rows",
    ),
]


@pytest.mark.parametrize("flags,message", LEGACY_REJECTED)
def test_legacy_rejected(flags, message):
    with pytest.raises(ValueError) as error:
        MM.MegaMoERecipe.from_legacy(**flags)
    assert message in str(error.value)


def test_rejected_by_geometry(mori):
    with pytest.raises(ValueError, match="mori's fp4 combine needs hidden_dim % 1024"):
        _config(D.MORI, C.MORI, "bf16", ("bf16", "fp4"), 4, 1536, 8, 16, 8)
    with pytest.raises(ValueError, match="the combine reduce tiles hidden in"):
        _config(D.MORI, C.FUSED_SCATTER, "bf16", ("bf16",), 4, 7200, 8, 16, 8)
    with pytest.raises(ValueError, match=r"rank=4 must be in \[0, 4\)"):
        _config(D.MORI, C.FUSED_SCATTER, "bf16", ("bf16",), 4, 7168, 8, 16, 8, rank=4)
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


# -- the recv-token bound: both dispatches refuse what the arena cannot hold -----
@pytest.mark.parametrize(
    "dispatch,wire", [(D.MORI, "bf16"), (D.MORI, "fp4"), (D.COMPACT, "fp4")]
)
def test_recv_token_bound_out_of_range(mori, dispatch, wire):
    config = _config(dispatch, C.FUSED_SCATTER, wire, ("bf16",), 4, 7168, 8, 16, 8)
    stage_cls = recipe._stage_classes(config.recipe)[0]
    stage = stage_cls(config, transport._capabilities_for(config.recipe))
    x = torch.zeros(4, 7168, dtype=torch.bfloat16)
    ids = torch.zeros(4, 8, dtype=torch.int32)
    wts = torch.zeros(4, 8, dtype=torch.float32)
    over = config.max_recv + 1
    message = rf"recv_token_bound must be in \(0, {config.max_recv}\], got"
    with pytest.raises(ValueError, match="^" + message):
        stage.dispatch(x, wts, ids, recv_token_bound=over)
    with pytest.raises(ValueError, match="^" + message):
        stage.dispatch(x, wts, ids, recv_token_bound=0)
    with pytest.raises(ValueError, match="^next_" + message):
        stage.dispatch(x, wts, ids, next_topk_ids=ids, next_recv_token_bound=over)
    with pytest.raises(ValueError, match="^" + message):
        stage.prefetch(ids, over)
    with pytest.raises(ValueError, match="^" + message):
        stage.warmup(over)


def test_compact_plan_limits():
    # A wave per peer in one block of at most 32 waves.
    with pytest.raises(ValueError, match="runs a wave per peer"):
        _config(D.COMPACT, C.FUSED_SCATTER, "fp4", ("bf16",), 64, 7168, 8, 16, 8)
    # An LDS histogram of at most 1024 (rank, expert) segments.
    with pytest.raises(ValueError, match="at most 1024"):
        _config(D.COMPACT, C.FUSED_SCATTER, "fp4", ("bf16",), 32, 7168, 8, 16, 64)


def test_compact_default_bound_is_the_capacity(mori):
    config = _config(D.COMPACT, C.FUSED_SCATTER, "fp4", ("bf16",), 4, 7168, 8, 16, 8)
    stage = recipe._stage_classes(config.recipe)[0](config, None)
    # What a rank receives depends on its peers' token counts, not its own.
    assert stage._recv_bound(None) == config.max_recv
    assert stage._recv_bound(5) == 5


def test_compact_refuses_a_plan_started_for_another_alignment(mori):
    config = _config(D.COMPACT, C.FUSED_SCATTER, "fp4", ("bf16",), 4, 7168, 8, 16, 8)
    stage = recipe._stage_classes(config.recipe)[0](config, None)
    stage._plan_pending, stage._pending_align = True, COMPACT_TILE // 2
    x = torch.zeros(4, 7168, dtype=torch.bfloat16)
    ids = torch.zeros(4, 8, dtype=torch.int32)
    wts = torch.zeros(4, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="the compact plan already started"):
        stage.dispatch(x, wts, ids, recv_token_bound=8)
