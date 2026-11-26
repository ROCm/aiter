# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1 TP4 decode MoE (E128 top-4, H6144 / I768, MXFP4 routed experts + MXFP8 shared expert) through the
public ``fused_moe`` interface (``shared_folded=True``, the ``flydsl_m3decode`` whole-graph row of
``tuned_shared_fmoe.csv``) and the router entry ``moe_route_quant``.

* tests: the integrated path is bit-identical to calling the chain ops directly (``route_quant_small`` +
  ``decode_expert_gemms`` + ``moe_slots_shared_reduce``) for every row count and for the tiling configs the tuner
  enumerates; the re-laid scales match a logical reference; a missing row and an out-of-contract request fail closed.
* ``python op_tests/test_fused_moe_m3_decode.py``: the layer timing table (CUDA-graph replay median, cold weights,
  ABAB) of the public path (gate GEMM + topk_gating + 2-stage fused_moe + shared MXFP8 MLP), the integrated path with
  the public router, with the fused router entry, and the chain ops called directly.
"""

import argparse
import statistics

import pytest
import torch

pytest.importorskip("flydsl")
from aiter.jit.utils.chip_info import get_gfx_runtime

if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
    pytest.skip("needs a gfx950 GPU", allow_module_level=True)

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe, moe_route_quant
from aiter.ops.flydsl import fused_moe_m3_decode as m3
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_bench import (
    DEV as dev,
)
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_bench import (
    ROWS,
    SCALE,
    E,
    H,
    I,
    _u8,
    chain_direct,
    graph_median_us,
    integrated,
    integrated_public_router,
    integrated_router,
    make_layer,
    make_router,
    make_x,
    public_path,
    shared_kw,
)
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_cfg import (
    expert_candidates,
    router_candidates,
)
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_chain import route_quant_small
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import router_line_layout

# ---------------------------------------------------------------- tests


def _tuned_router_cfg(n):
    from aiter.fused_moe import _decode_route_cfg_names, get_padded_M
    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_cfg import RouterCfg

    names = _decode_route_cfg_names(AITER_CONFIGS.AITER_CONFIG_SHARED_FMOE_FILE)
    name = names.get(("gfx950", get_cu_num(), get_padded_M(n), H, I, E, 4))
    cfg = None if name is None else RouterCfg.parse(name)
    return cfg if cfg is not None and cfg.fits(n) else None


@pytest.mark.parametrize("n", ROWS + (9, 24, 100))
def test_integrated_matches_chain(n):
    L, (gate, bias), x = make_layer(), make_router(), make_x(n)
    lines = router_line_layout(gate)
    ref, w, ids = chain_direct(x, L, gate, lines, bias)
    # the router entry runs the tuned row's router config (another split-K sums in another order: compare with
    # the chain ops on that same config)
    ref_r, _, _ = chain_direct(x, L, gate, lines, bias, rcfg=_tuned_router_cfg(n))
    # bf16 rows: the row-quant runs inside the whole-graph implementation
    assert torch.equal(integrated(x, L, w, ids), ref)
    # the router entry in front: bit-identical routes and output
    assert torch.equal(integrated_router(x, L, gate, bias), ref_r)


@pytest.mark.parametrize("n", ROWS)
def test_expert_configs_are_bit_identical(n):
    """Every tiling the tuner may pick keeps the K order: same bits as the default tiling."""
    L, (gate, bias), x = make_layer(), make_router(), make_x(n)
    lines = router_line_layout(gate)
    ref, _, _ = chain_direct(x, L, gate, lines, bias)
    bad, checked = [], 0
    for cfg in expert_candidates(n):
        # every candidate is a legal tiling: an exception is a failure, not a skipped iteration
        out, _, _ = chain_direct(x, L, gate, lines, bias, ecfg=cfg)
        if not torch.equal(out, ref):
            bad.append(cfg.name())
        checked += 1
    assert checked == len(expert_candidates(n)) and not bad, bad


@pytest.mark.parametrize("n", ROWS)
def test_router_configs_pick_the_same_routes(n):
    """Every router config the tuner may pick routes like the default: same experts; weights within fp32
    rounding of a different split-K sum order."""
    (gate, bias), x = make_router(), make_x(n)
    lines = router_line_layout(gate)
    _, _, w0, ids0 = route_quant_small(x, gate, lines, bias, SCALE)
    for cfg in router_candidates(n):
        print(f"router n={n} {cfg.name()}", flush=True)
        r = route_quant_small(x, gate, lines, bias, SCALE, cfg)
        torch.cuda.synchronize()
        if r is None:
            continue
        (_xq, _), (_q8, _), w, ids = r
        assert torch.equal(ids.sort(-1).values, ids0.sort(-1).values), cfg.name()
        torch.testing.assert_close(
            w.sort(-1).values, w0.sort(-1).values, rtol=1e-4, atol=1e-5
        )


def test_scale_relayout_matches_logical():
    """The routed w13 scale re-layout equals shuffling the logical scales with ``is_guinterleave``."""
    from aiter.ops.shuffle import shuffle_scale

    g = torch.Generator(device=dev).manual_seed(5)
    raw = _u8((E, 2 * I, H // 32), g, 121, 124)
    atom_layout = shuffle_scale(raw.view(E * 2 * I, H // 32)).view(E, 2 * I, H // 32)
    want = shuffle_scale(
        raw.view(E * 2 * I, H // 32), E, is_guinterleave=True, gate_up=True
    ).view(torch.uint8)
    assert torch.equal(m3.routed_w13_scale_layout(atom_layout), want)


def test_fails_closed():
    L, (gate, bias), x = make_layer(), make_router(), make_x(4)
    _, _, w, ids = route_quant_small(x, gate, router_line_layout(gate), bias, SCALE)
    with pytest.raises(NotImplementedError):  # 4 rows: no row, no fallback
        integrated(x, L, w, ids)
    x = make_x(16)
    _, w, ids = chain_direct(x, L, gate, router_line_layout(gate), bias)
    with pytest.raises(NotImplementedError):  # a bias is outside the kernels' contract
        fused_moe(
            x,
            L["w1"],
            L["w2"],
            w,
            ids,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=L["w1_scale"],
            w2_scale=L["w2_scale"],
            bias1=torch.zeros(E, 2 * I, device=dev),
            **shared_kw(L),
        )


@pytest.mark.parametrize(
    "kw",
    [
        {"a2_scale": torch.ones(1)},
        {"w1_global_scale": torch.ones(1)},
        {"w2_global_scale": torch.ones(1)},
        {"linear_beta": 1.0},
        {"splitk": 2},
    ],
)
def test_ignored_arguments_fail_closed(kw):
    """An argument the decode chain does not implement is refused, not silently dropped."""
    L, (gate, bias), x = make_layer(), make_router(), make_x(16)
    _, w, ids = chain_direct(x, L, gate, router_line_layout(gate), bias)
    with pytest.raises(NotImplementedError):
        fused_moe(
            x,
            L["w1"],
            L["w2"],
            w,
            ids,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=L["w1_scale"],
            w2_scale=L["w2_scale"],
            **kw,
            **shared_kw(L),
        )


def _fmoe16(L, w, ids, x, **kw):
    return fused_moe(
        x,
        L["w1"],
        L["w2"],
        w,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=L["w1_scale"],
        w2_scale=L["w2_scale"],
        **{**shared_kw(L), **kw},
    )


def test_activation_overrides_fail_closed():
    """The row quantizes to the weights' dtype: another activation dtype / quant type is refused, the weights'
    own dtype is the same request as no override."""
    from aiter import dtypes

    L, (gate, bias), x = make_layer(), make_router(), make_x(16)
    _, w, ids = chain_direct(x, L, gate, router_line_layout(gate), bias)
    for kw in (
        {"quant_dtype_a": dtypes.fp8},
        {"quant_dtype_a2": dtypes.fp4x2},
        {"quant_type_a": QuantType.per_Token.value},
    ):
        with pytest.raises(NotImplementedError):
            _fmoe16(L, w, ids, x, **kw)
    assert torch.equal(
        _fmoe16(L, w, ids, x, quant_dtype_a=L["w1"].dtype), _fmoe16(L, w, ids, x)
    )


def test_shared_weights_must_be_preshuffled_e4m3():
    """The kernels hard-code the preshuffled E4M3FN operand layout: a row-major or E5M2 shared weight of the right
    size is refused, not reinterpreted."""
    L, (gate, bias), x = make_layer(), make_router(), make_x(16)
    _, w, ids = chain_direct(x, L, gate, router_line_layout(gate), bias)
    plain = dict(L, shared_w1=L["shared_w1"].clone())  # no is_shuffled marker
    plain["shared_w1"].is_shuffled = False
    e5m2 = dict(L, shared_w2=L["shared_w2"].view(torch.uint8).view(torch.float8_e5m2))
    e5m2["shared_w2"].is_shuffled = True
    for bad in (plain, e5m2):
        with pytest.raises(NotImplementedError):
            _fmoe16(bad, w, ids, x)


def test_shared_scale_layouts_are_cached_independently():
    """Replacing only the down scales re-lays only ``ss2``; the gate / up layout stays the cached one."""
    L = make_layer()
    _, ss13, ss2 = m3.prepare_decode_weights(
        L["w1_scale"], L["shared_w1_scale"], L["shared_w2_scale"]
    )
    new_down = L["shared_w2_scale"].view(torch.uint8).clone()
    new_down[0, 0] ^= 1
    _, ss13b, ss2b = m3.prepare_decode_weights(
        L["w1_scale"], L["shared_w1_scale"], new_down
    )
    assert ss13b is ss13
    assert not torch.equal(ss2b, ss2)
    want = m3.shared_scale_layouts(L["shared_w1_scale"], new_down)[1]
    assert torch.equal(ss2b, want)


@pytest.mark.parametrize("case", ["all_dropped", "mixed"])
def test_dropped_routes_contribute_nothing(case):
    """An id outside [0, E) is a dropped route: its slot adds zero (the same output as that slot's weight being 0),
    not uninitialized GEMM2 buffer contents."""
    n = 16
    L, x = make_layer(), make_x(n)
    ids = torch.randint(0, E, (n, 4), dtype=torch.int32, device=dev)
    w = torch.rand((n, 4), device=dev)
    dropped = ids.clone()
    zeroed = w.clone()
    if case == "all_dropped":
        dropped.fill_(-1)
        zeroed.zero_()
    else:
        dropped[:, 1] = -1
        dropped[2, :] = E
        zeroed[:, 1] = 0
        zeroed[2, :] = 0
    # poison the allocator so an uninitialized slot buffer would read NaN
    junk = torch.full((n, 4, H), float("nan"), dtype=torch.bfloat16, device=dev)
    del junk
    got = integrated(x, L, w, dropped)
    want = integrated(x, L, zeroed, ids)
    assert torch.isfinite(got.float()).all()
    assert torch.equal(got, want)


def test_route_quant_rejects_other_gates_without_building_layouts():
    gate, bias = make_router()
    x = make_x(16)
    n0 = len(m3._DERIVED)
    assert moe_route_quant(x, gate.half(), bias) is None  # fp16 gate
    assert moe_route_quant(x, gate[:64].contiguous(), bias) is None  # wrong shape
    assert len(m3._DERIVED) == n0


def test_tuner_row_selection_follows_the_environment():
    """``_TempRow`` really switches the shared table (``get_config_file`` is lru-cached on the env *name*) and
    restores the previous one."""
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS
    from csrc.ck_gemm_moe_2stages_codegen.decode_chain_tune import (
        _TempRow,
        row_key,
    )

    before = AITER_CONFIGS.AITER_CONFIG_SHARED_FMOE_FILE  # primes the cache
    row = _TempRow(row_key(16, "gfx950", 256), "k1", "k2")
    with row:
        assert AITER_CONFIGS.AITER_CONFIG_SHARED_FMOE_FILE == row.path
    assert AITER_CONFIGS.AITER_CONFIG_SHARED_FMOE_FILE == before
    assert fm.cfg_2stages_by_file == {} or row.path not in fm.cfg_2stages_by_file


# ---------------------------------------------------------------- timing


def run_timing(rows, nsets, abab, ecfgs=None):
    gate, bias = make_router()
    gate_bf16 = gate.bfloat16()
    lines = router_line_layout(gate)
    layers = [make_layer(100 + i) for i in range(nsets)]
    print("| rows | path | us/layer (median of ABAB rounds) |")
    print("|---|---|---|")
    for n in rows:
        x = make_x(n)

        def pub(L, x=x):
            public_path(x, L, gate_bf16, bias)

        def intp(L, x=x):
            integrated_public_router(x, L, gate_bf16, bias)

        def intr(L, x=x):
            integrated_router(x, L, gate, bias)

        def chain(L, x=x):
            chain_direct(x, L, gate, lines, bias)

        paths = {
            "public": pub,
            "integrated (public router)": intp,
            "integrated (router entry)": intr,
            "chain ops direct": chain,
        }
        res = {k: [] for k in paths}
        for _ in range(abab):
            for k, f in paths.items():
                res[k].append(graph_median_us(f, layers))
        for k, v in res.items():
            print(
                f"| {n} | {k} | {statistics.median(v):.2f} ({', '.join(f'{t:.2f}' for t in v)}) |"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    p.add_argument(
        "--sets", type=int, default=8, help="rotated weight sets (cold weights)"
    )
    p.add_argument("--abab", type=int, default=2)
    a = p.parse_args()
    run_timing(a.rows, a.sets, a.abab)


if __name__ == "__main__":
    main()
