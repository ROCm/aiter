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
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl import fused_moe_m3_decode as m3
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_bench import (
    DEV as dev,
)
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_bench import (
    E,
    H,
    I,
    ROWS,
    SCALE,
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
    bad = []
    for cfg in expert_candidates(n):
        try:
            out, _, _ = chain_direct(x, L, gate, lines, bias, ecfg=cfg)
        except (
            Exception
        ) as exc:  # a candidate the kernels cannot build is dropped by the tuner
            print(f"n={n} {cfg.name()}: {type(exc).__name__}")
            continue
        if not torch.equal(out, ref):
            bad.append(cfg.name())
    assert not bad, bad


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
        (xq, _), (q8, _), w, ids = r
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

        def pub(L):
            public_path(x, L, gate_bf16, bias)

        def intp(L):
            integrated_public_router(x, L, gate_bf16, bias)

        def intr(L):
            integrated_router(x, L, gate, bias)

        def chain(L):
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
