#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Speed + accuracy compare: FlyDSL grouped MoE vs gluon two-stage MoE.

Both paths are built from ONE shared GGUU logical case (the gluon e2e test's
``_make_case``) and evaluated against the SINGLE shared reference
(``test_moe_gemm_a8w4_e2e._ref_from_case`` -> FlyDSL's ``_torch_moe_ref``), so
the two implementations are fed byte-identical logical weights/scales/bias/
routing and judged against the same fp32 baseline. Each is then timed with the
same ``run_perftest``.

This module is intentionally a THIN shell: every FlyDSL-side helper is imported
from ``test_flydsl_grouped_gemm_gfx1250`` and every gluon-side helper from
``test_moe_gemm_a8w4_e2e``, so there is exactly one copy of the reference and
the data builders. The only logic that lives here is ``_fused_case_from_logical``
(feed a shared logical case to ``fused_moe``) and the ``_compare`` driver.

Run (gfx1250 only):
    python op_tests/test_flydsl_grouped_gemm_gfx1250_compare.py --scenario compare \
        --experts 8 --tokens 8 --topk 2 --model-dim 512 --inter-dim 512 \
        --warmup 3 --iters 25
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure THIS repo's aiter/op_tests win over any installed copy when run as a
# script (a script puts op_tests/ on sys.path[0], not the repo root).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from aiter import ActivationType  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility import dtypes  # noqa: E402

# Official production router (the same one bench_moe_gemm_a8w4.py and ATOM use).
from aiter.ops.triton.moe.moe_routing.routing import routing  # noqa: E402
from aiter.ops.triton.moe.moe_routing.topk import topk  # noqa: E402

# FlyDSL-side: reuse the grouped path's builders/runner + the shared reference.
from op_tests.test_flydsl_grouped_gemm_gfx1250 import (  # noqa: E402
    _require_gfx1250,
    _grouped_scale,
    _gguu_to_gugu_rows,
    _invoke_grouped_fused_moe,
    _rel_l2,
    VERIFY_TOL_A4W4,
    VERIFY_TOL_A8W4,
)

# gluon-side: the two-stage MoE under test + the shared reference/case builders.
from op_tests.triton_tests.moe.test_moe_gemm_a8w4_e2e import (  # noqa: E402
    _make_case,
    _ref_from_case,
    _prepare_gluon_inputs,
    _run_gluon_stages,
    _config_from_args,
)


# ---------------------------------------------------------------------------
# Build fused_moe inputs from an explicit GGUU logical case (the shared case),
# so the FlyDSL grouped path consumes byte-identical weights/scales/bias/routing
# to what the gluon path and the reference see.
# ---------------------------------------------------------------------------
def _fused_case_from_logical(
    *,
    hidden: torch.Tensor,  # (T, K) bf16
    w1_logical: torch.Tensor,  # (E, 2*I, K_pack) uint8 GGUU
    w1_scale_raw: torch.Tensor,  # (E, 2*I, K//32) uint8 e8m0
    bias1: torch.Tensor,  # (E, 2*I) fp32
    w2_logical: torch.Tensor,  # (E, K, I_pack) uint8
    w2_scale_raw: torch.Tensor,  # (E, K, I//32) uint8 e8m0
    bias2: torch.Tensor,  # (E, K) fp32
    topk_id: torch.Tensor,  # (T, topk) int
    topk_w: torch.Tensor,  # (T, topk) float/bf16
    experts: int,
    model_dim: int,
    inter_dim: int,
    data_format: str,
    layout: str,
    activation: ActivationType,
    swiglu_limit: float,
) -> dict:
    K = model_dim
    inter = inter_dim

    if layout == "gugu":
        w1_phys = _gguu_to_gugu_rows(w1_logical)
        w1_scale_phys = _gguu_to_gugu_rows(w1_scale_raw)
        bias1_phys = _gguu_to_gugu_rows(bias1)
        gate_mode = GateMode.INTERLEAVE
    else:
        w1_phys = w1_logical
        w1_scale_phys = w1_scale_raw
        bias1_phys = bias1
        gate_mode = GateMode.SEPARATED

    w1_grouped = shuffle_weight(w1_phys, layout=(16, 16)).cuda()
    w2_grouped = shuffle_weight(w2_logical, layout=(16, 16)).cuda()
    w1_scale = _grouped_scale(w1_scale_phys, experts=experts, rows=2 * inter, k_dim=K)
    w2_scale = _grouped_scale(w2_scale_raw, experts=experts, rows=K, k_dim=inter)

    if data_format == "a4w4":
        w1_arg = w1_grouped.view(dtypes.fp4x2)
        w2_arg = w2_grouped.view(dtypes.fp4x2)
    else:
        w1_arg = w1_grouped
        w2_arg = w2_grouped

    return {
        "hidden_states": hidden.cuda(),
        "w1": w1_arg,
        "w2": w2_arg,
        "topk_weight": topk_w.to(torch.bfloat16).cuda(),
        "topk_ids": topk_id.cuda(),
        "activation": activation,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "bias1": bias1_phys.float().cuda(),
        "bias2": bias2.float().cuda(),
        "gate_mode": gate_mode.value,
        "swiglu_limit": swiglu_limit,
    }


# ---------------------------------------------------------------------------
# Compare driver
# ---------------------------------------------------------------------------
def _compare(args: argparse.Namespace) -> None:
    from aiter.test_common import run_perftest

    _require_gfx1250()

    activation = ActivationType.Swiglu if args.act == "swiglu" else ActivationType.Silu
    if activation != ActivationType.Swiglu:
        raise SystemExit("compare only supports --act swiglu (gluon path is swiglu)")

    print(
        f"[compare] data_format={args.data_format} layout={args.layout} "
        f"E={args.experts} T={args.tokens} topk={args.topk} "
        f"K={args.model_dim} I={args.inter_dim} "
        f"warmup={args.warmup} iters={args.iters}",
        flush=True,
    )

    # One shared logical case (weights / hidden / bias). Its balanced-topk
    # routing fields are overwritten below with the OFFICIAL router's output so
    # both paths and the reference consume the exact same expert assignment +
    # softmax gate weights that a real deployment (bench/ATOM) would produce.
    case = _make_case(
        experts=args.experts,
        tokens=args.tokens,
        topk=args.topk,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        use_bias=not args.no_bias,
    )

    # Shared router logits. topk() gives the (T, topk) expert ids + softmax
    # weights that feed fused_moe and the reference; routing() gives the
    # RoutingData + gather/scatter the gluon kernel consumes. Both come from the
    # SAME logits, so the two paths are routed identically (verified: hist match,
    # gate_scal == expt_scal as a multiset).
    torch.manual_seed(0)
    logits = torch.randn(
        args.tokens, args.experts, dtype=torch.float16, device="cuda"
    )
    expt_scal, expt_indx, _ = topk(logits, args.topk, apply_softmax=True)
    case["topk_id"] = expt_indx.to(torch.int32)
    case["topk_w"] = expt_scal.to(torch.float32)

    # Shared reference (decodes the GGUU logical weights at fp32, mxfp8 act
    # quant matched to dtype_max=448 / e4m3fn), now on the official routing.
    ref = _ref_from_case(case).to(torch.bfloat16)

    # ---- FlyDSL grouped path ----
    fused_case = _fused_case_from_logical(
        hidden=case["hidden"],
        w1_logical=case["w1_packed"],
        w1_scale_raw=case["w1_scale_raw"],
        bias1=case["bias1"],
        w2_logical=case["w2_packed"],
        w2_scale_raw=case["w2_scale_raw"],
        bias2=case["bias2"],
        topk_id=case["topk_id"].cuda(),
        topk_w=case["topk_w"].cuda(),
        experts=args.experts,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        data_format=args.data_format,
        layout=args.layout,
        activation=activation,
        swiglu_limit=args.swiglu_limit,
    )
    saved = os.environ.get("AITER_USE_GROUPED_GEMM")
    os.environ["AITER_USE_GROUPED_GEMM"] = "1"
    try:
        flydsl_out = _invoke_grouped_fused_moe(fused_case).to(torch.bfloat16)
        torch.cuda.synchronize()

        def _flydsl_thunk():
            return _invoke_grouped_fused_moe(fused_case)

        _, flydsl_us = run_perftest(
            _flydsl_thunk, num_warmup=args.warmup, num_iters=args.iters
        )
    finally:
        if saved is None:
            os.environ.pop("AITER_USE_GROUPED_GEMM", None)
        else:
            os.environ["AITER_USE_GROUPED_GEMM"] = saved

    # ---- gluon path ----
    # Weight prep (transpose / scale swizzle / H2D) is one-time setup, kept OUT
    # of the timed region. The OFFICIAL routing() call is kept INSIDE the timed
    # thunk so the gluon path is charged for routing the same way fused_moe is
    # (its route-map build is internal and unavoidably timed). This mirrors
    # bench_moe_gemm_a8w4.py / ATOM, which call routing() per forward.
    gluon_cfg = _config_from_args(args)
    topk_id_dev = case["topk_id"].cuda()
    topk_w_dev = case["topk_w"].cuda()

    def _gluon_prep_with(routing_out):
        return _prepare_gluon_inputs(
            hidden=case["hidden"],
            w1_packed=case["w1_packed"],
            w1_scale_raw=case["w1_scale_raw"],
            bias1=case["bias1"],
            w2_packed=case["w2_packed"],
            w2_scale_raw=case["w2_scale_raw"],
            bias2=case["bias2"],
            topk_id=topk_id_dev,
            topk_w=topk_w_dev,
            experts=args.experts,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            block_m_override=args.block_m,
            routing_override=routing_out,
        )

    # Pre-build the weight-side prep once with a throwaway routing, then reuse
    # only its weight tensors; routing is recomputed per-iter inside the thunk.
    base_prep = _gluon_prep_with(routing(logits, args.topk))

    def _gluon_thunk():
        rdata, gather_indx, scatter_indx = routing(logits, args.topk)
        prep = dict(base_prep)
        prep["rdata"] = rdata
        prep["gather_indx"] = gather_indx
        prep["scatter_indx"] = scatter_indx
        return _run_gluon_stages(prep, gluon_cfg)

    gluon_out = _gluon_thunk().to(torch.bfloat16)
    torch.cuda.synchronize()
    _, gluon_us = run_perftest(
        _gluon_thunk, num_warmup=args.warmup, num_iters=args.iters
    )

    flydsl_rel = _rel_l2(flydsl_out, ref)
    gluon_rel = _rel_l2(gluon_out, ref)
    tol = VERIFY_TOL_A8W4 if args.data_format == "a8w4" else VERIFY_TOL_A4W4

    print("\n| path   | rel_l2 vs ref | tol    | latency (us) | speedup |", flush=True)
    print("|--------|---------------|--------|--------------|---------|", flush=True)
    print(
        f"| flydsl | {flydsl_rel:.4e}    | {tol:.3f}  | "
        f"{flydsl_us:11.2f}  | {gluon_us / flydsl_us:6.2f}x |",
        flush=True,
    )
    print(
        f"| gluon  | {gluon_rel:.4e}    | {tol:.3f}  | "
        f"{gluon_us:11.2f}  | {flydsl_us / gluon_us:6.2f}x |",
        flush=True,
    )

    ok = flydsl_rel < tol and gluon_rel < tol
    print(
        f"\n[compare] accuracy {'OK' if ok else 'FAILED'}: "
        f"flydsl_rel={flydsl_rel:.4e} gluon_rel={gluon_rel:.4e} (tol={tol})",
        flush=True,
    )
    if not ok:
        raise SystemExit("compare FAILED: one or both paths exceeded tol")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("compare",), default="compare")
    parser.add_argument("--data-format", choices=("a4w4", "a8w4"), default="a8w4")
    parser.add_argument(
        "--layout",
        choices=("gguu", "gugu"),
        default="gguu",
        help="stage1 weight physical layout. gguu pairs with "
        "GateMode.SEPARATED (default), gugu with INTERLEAVE.",
    )
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument(
        "--act",
        choices=("silu", "swiglu"),
        default="swiglu",
        help="stage1 activation (compare supports swiglu only).",
    )
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="run with zero stage1/stage2 bias tensors",
    )
    # gluon kernel-config overrides (read by the gluon path's
    # _config_from_args / _prepare_from_case). None -> gluon defaults.
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    args = parser.parse_args()
    _compare(args)


if __name__ == "__main__":
    main()
