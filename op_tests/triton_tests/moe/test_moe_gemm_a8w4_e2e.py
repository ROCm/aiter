#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 gluon a8w4 grouped-MoE GEMM, aligned with the FlyDSL reference.

This test drives ``aiter.ops.triton.moe.moe_gemm_a8w4`` as a **full two-stage
MoE** (gemm1 gate/up + swiglu -> MXFP8 1x32 re-quant of the intermediate ->
gemm2 down-proj -> topk-weighted scatter/reduce) so it can be speed- and
correctness-compared apples-to-apples with the FlyDSL grouped path in
``op_tests/test_flydsl_grouped_gemm_gfx1250.py``.

Data format is **a8w4** (MXFP8 activations x MXFP4 weights) with **per-block
1x32** quantization. The reference is FlyDSL's own ``_torch_moe_ref`` called
verbatim, and both the kernel-under-test and the reference are driven from the
same balanced deterministic routing, so the only divergence is MXFP8 activation
round-off (target rel_l2 <= 0.02).

The two-stage wiring mirrors ATOM's ``aiter_a8w4_fused_experts``
(``/data/chenyangguo/ATOM/atom/model_ops/fused_moe_triton.py``) and the mx8
quant chain in ``op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4.py``.

Pytest runs a small correctness case. Direct execution
(``python op_tests/triton_tests/moe/test_moe_gemm_a8w4.py``) runs a
DeepSeek-style perf bench (``--scenario bench``) or the correctness check
(``--scenario verify``) at the FlyDSL bench shape.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure THIS repo's aiter wins over any installed copy when run as a script
# (running as a script puts op_tests/ on sys.path[0], not the repo root).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

# matmul / quant / routing utilities
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
    moe_gemm_a8w4,
    swizzle_scales_gfx1250,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.moe.moe_routing.routing import (
    RoutingData,
    compute_expt_data_torch,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter import ActivationType

# Reuse the FlyDSL test's reference + data builders verbatim so the two paths
# share identical reference math. The flydsl test imports cleanly on any arch.
from op_tests.test_flydsl_grouped_gemm_gfx1250 import (  # noqa: E402
    _torch_moe_ref,
    _balanced_topk,
    _pattern_packed,
    _full_scale,
    _gguu_to_gugu_rows,
    _rel_l2,
    SCALE_BLOCK,
    VERIFY_TOL_A8W4,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

# gpt-oss swiglu constants -- must match aiter.fused_moe.swiglu (alpha=1.702,
# limit=7.0, +1 linear residual), which is what FlyDSL's reference evaluates.
_SWIGLU_ALPHA = 1.702
_SWIGLU_LIMIT = 7.0
_SWIZZLE = "GFX1250_SCALE"


def _fp8_dtype() -> torch.dtype:
    # gfx1250 uses OCP e4m3 (not the fnuz variant gfx942 uses).
    return torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Routing: reconstruct RoutingData + gather/scatter from fixed balanced ids.
# Mirrors routing_torch() in moe_routing/routing.py, but seeded from explicit
# (token, rank) -> expert ids instead of logits, and WITHOUT the softmax on the
# gate weights (we want the uniform 1/topk weights the reference uses).
# ---------------------------------------------------------------------------
def _build_routing_from_ids(
    topk_id: torch.Tensor,  # (T, topk) int
    topk_w: torch.Tensor,  # (T, topk) float
    n_expts_tot: int,
    block_m_override: int | None = None,
):
    import triton

    device = topk_id.device
    T, topk = topk_id.shape
    n_gates = T * topk

    # Sort each token's selections by expert so experts are contiguous, exactly
    # like routing_torch (this keeps gate weights paired with their expert).
    expt_indx, sort_idx = torch.sort(topk_id, dim=1)
    expt_scal = torch.gather(topk_w, 1, sort_idx)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)
    expt_scal = expt_scal.reshape(-1).to(torch.float32)

    # Expert-sorted permutation and its inverse.
    gather_indx = torch.argsort(expt_indx, stable=True).to(torch.int32)
    scatter_indx = torch.argsort(gather_indx.long(), stable=True).to(torch.int32)
    gate_scal = expt_scal[gather_indx.long()]

    hist = torch.histc(
        expt_indx.float(), bins=n_expts_tot, min=0, max=n_expts_tot - 1
    ).to(torch.int32)

    if block_m_override is not None:
        block_m = int(block_m_override)
    else:
        tokens_per_expt = max(1, n_gates // n_expts_tot)
        block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    # block_m drives the launch grid and the decode/prefill dispatch inside
    # moe_gemm_a8w4, plus token_offs_pad / block_pid_map here -- so recompute
    # expt_data with the (possibly overridden) block_m.
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates, block_m)

    rdata = RoutingData(
        block_m, gate_scal, hist, n_expts_tot, topk, expt_data
    )
    return rdata, gather_indx.to(device), scatter_indx.to(device)


# ---------------------------------------------------------------------------
# Kernel-config override.
#
# IMPORTANT: moe_gemm_a8w4 reads ``block_m`` from ``routing_data.block_m`` (not
# the config dict) for BOTH the launch grid and the decode/prefill dispatch
# (``if use_gluon and block_m == 16`` -> decode kernel, else prefill). The
# config dict's block_m only sets the kernel BLOCK_M constexpr. So a block_m
# override MUST be applied to ``rdata.block_m`` (done in _prepare, see
# ``block_m_override``) AND mirrored into the config dict here, or the grid and
# tile geometry disagree. This override patches get_kernel_config_gluon for the
# tile knobs; block_m is handled at prepare time so the two stay consistent and
# the dispatch actually switches to the prefill kernel.
# ---------------------------------------------------------------------------
class _config_override:
    def __init__(self, overrides: dict | None):
        self._overrides = overrides
        self._saved = None

    def __enter__(self):
        if not self._overrides:
            return self
        import aiter.ops.triton.moe.moe_op_gemm_a8w4 as _mod

        self._mod = _mod
        self._saved = _mod.get_kernel_config_gluon
        base = _mod.get_kernel_config_gluon
        ov = self._overrides

        def _patched(m, n, k, routing_data):
            cfg = base(m, n, k, routing_data)
            # block_m comes from routing_data.block_m (set at prepare time); keep
            # the config dict in sync so BLOCK_M matches the dispatched kernel.
            cfg["block_m"] = routing_data.block_m
            for key in (
                "block_n",
                "block_k",
                "num_warps",
                "num_stages",
                "waves_per_eu",
            ):
                if ov.get(key) is not None:
                    cfg[key] = ov[key]
            return cfg

        _mod.get_kernel_config_gluon = _patched
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            self._mod.get_kernel_config_gluon = self._saved


# ---------------------------------------------------------------------------
# Weight prep: take the GGUU logical mxfp4 bytes + e8m0 scale the reference uses
# and arrange them into the (E, K_pack, N) col-major layout the kernel consumes,
# swizzling scales. For w1 we additionally interleave gate/up rows to GUGU so
# the kernel's even/odd swiglu split lines up with the GGUU reference.
# ---------------------------------------------------------------------------
def _prep_w1_kernel(w1_packed, w1_scale_raw, bias1, *, gugu: bool):
    # w1_packed: (E, 2*inter, K_pack) uint8 ; scale: (E, 2*inter, K//32) ; bias: (E, 2*inter)
    if gugu:
        w1_packed = _gguu_to_gugu_rows(w1_packed)
        w1_scale_raw = _gguu_to_gugu_rows(w1_scale_raw)
        bias1 = _gguu_to_gugu_rows(bias1)
    # (E, 2*inter, K_pack) -> (E, K_pack, 2*inter) col-major (stride(-2)==1).
    w1_k = w1_packed.cuda().transpose(1, 2)
    w1_scale_k = swizzle_scales_gfx1250(w1_scale_raw.cuda().transpose(1, 2).contiguous())
    return w1_k, w1_scale_k, bias1.float().cuda()


def _prep_w2_kernel(w2_packed, w2_scale_raw, bias2):
    # w2_packed: (E, model_dim, inter_pack) ; scale: (E, model_dim, inter//32)
    w2_k = w2_packed.cuda().transpose(1, 2)
    w2_scale_k = swizzle_scales_gfx1250(w2_scale_raw.cuda().transpose(1, 2).contiguous())
    return w2_k, w2_scale_k, bias2.float().cuda()


# ---------------------------------------------------------------------------
# Two-stage MoE under test, built on moe_gemm_a8w4 (mx8 act x mxfp4 weight).
#
# Split into a one-time prepare (H2D copies, weight transpose, scale swizzle,
# routing reconstruction) and the hot two-stage compute. ONLY the latter is
# timed: routing/swizzle/H2D are setup that a real serving path computes once
# (weights resident, routing from the live router), so folding them into the
# bench loop would understate gluon's true GEMM throughput. The timed region is
# the real per-call pipeline: act-quant -> gemm1 -> intermediate requant ->
# gemm2 (mirrors ATOM's aiter_a8w4_fused_experts, mx8 variant).
# ---------------------------------------------------------------------------
def _prepare_gluon_inputs(
    *,
    hidden,  # (T, K) bf16
    w1_packed,
    w1_scale_raw,
    bias1,
    w2_packed,
    w2_scale_raw,
    bias2,
    topk_id,  # (T, topk) int, on device
    topk_w,  # (T, topk) float, on device
    experts,
    model_dim,
    inter_dim,
    block_m_override: int | None = None,
) -> dict:
    """One-time, NOT timed: everything that a serving path would precompute."""
    rdata, gather_indx, scatter_indx = _build_routing_from_ids(
        topk_id, topk_w, experts, block_m_override=block_m_override
    )
    w1_k, w1_scale_k, bias1_k = _prep_w1_kernel(
        w1_packed, w1_scale_raw, bias1, gugu=True
    )
    w2_k, w2_scale_k, bias2_k = _prep_w2_kernel(w2_packed, w2_scale_raw, bias2)
    return {
        "hidden_dev": hidden.cuda(),  # H2D once; act-quant stays in the hot loop
        "w1_k": w1_k,
        "w1_scale_k": w1_scale_k,
        "bias1_k": bias1_k,
        "w2_k": w2_k,
        "w2_scale_k": w2_scale_k,
        "bias2_k": bias2_k,
        "rdata": rdata,
        "gather_indx": gather_indx,
        "scatter_indx": scatter_indx,
    }


def _run_gluon_stages(prep: dict, config: dict | None = None) -> torch.Tensor:
    """Timed region: act-quant -> gemm1 -> requant -> gemm2."""
    fp8 = _fp8_dtype()
    rdata = prep["rdata"]

    with _config_override(config):
        # Stage-1 activations: MXFP8 per-1x32 (block=32 along K).
        x_q, x_scale = downcast_to_mxfp(prep["hidden_dev"], fp8, axis=-1)

        # gemm1: gate/up + fused swiglu, gather tokens into expert-sorted order.
        intermediate = moe_gemm_a8w4(
            x_q,
            prep["w1_k"],
            x_scale,
            prep["w1_scale_k"],
            None,  # x_static_scale
            None,  # quant_static_scale
            prep["bias1_k"],
            rdata,
            gather_indx=prep["gather_indx"],
            scatter_indx=None,
            gammas=None,
            swizzle_mx_scale=_SWIZZLE,
            out_dtype=torch.bfloat16,
            apply_swiglu=True,
            alpha=_SWIGLU_ALPHA,
            limit=_SWIGLU_LIMIT,
            swiglu_add_residual=True,
        )

        # Re-quant intermediate to MXFP8 per-1x32 for gemm2.
        inter_q, inter_scale = downcast_to_mxfp(intermediate, fp8, axis=-1)

        # gemm2: down-proj, topk-weighted scatter/reduce back to token order.
        out = moe_gemm_a8w4(
            inter_q,
            prep["w2_k"],
            inter_scale,
            prep["w2_scale_k"],
            None,
            None,
            prep["bias2_k"],
            rdata,
            gather_indx=None,
            scatter_indx=prep["scatter_indx"],
            gammas=rdata.gate_scal,
            swizzle_mx_scale=_SWIZZLE,
            out_dtype=torch.bfloat16,
            apply_swiglu=False,
        )
    return out


# ---------------------------------------------------------------------------
# Build a case: GGUU logical weights/scales/bias + balanced routing, identical
# to the data the FlyDSL reference consumes.
# ---------------------------------------------------------------------------
def _make_case(
    *, experts, tokens, topk, model_dim, inter_dim, use_bias=True, seed=0
):
    K = model_dim
    inter = inter_dim
    K_pack = K // 2
    inter_pack = inter // 2

    w1_packed = _pattern_packed(experts, 2 * inter, K_pack, seed=seed + 17)
    w2_packed = _pattern_packed(experts, K, inter_pack, seed=seed + 47)
    w1_scale_raw = _full_scale(experts, 2 * inter, K // SCALE_BLOCK)
    w2_scale_raw = _full_scale(experts, K, inter // SCALE_BLOCK)
    if use_bias:
        bg = torch.Generator(device="cpu").manual_seed(seed + 91)
        bias1 = (torch.randn((experts, 2 * inter), generator=bg) * 1e-3).float()
        bias2 = (torch.randn((experts, K), generator=bg) * 1e-3).float()
    else:
        bias1 = torch.zeros((experts, 2 * inter))
        bias2 = torch.zeros((experts, K))

    hg = torch.Generator(device="cpu").manual_seed(seed + 123)
    hidden = (torch.randn((tokens, K), generator=hg) * 0.5).to(torch.bfloat16)

    topk_id, topk_w = _balanced_topk(tokens, topk, experts)
    topk_w = topk_w.to(torch.float32)
    return {
        "hidden": hidden,
        "w1_packed": w1_packed,
        "w1_scale_raw": w1_scale_raw,
        "bias1": bias1,
        "w2_packed": w2_packed,
        "w2_scale_raw": w2_scale_raw,
        "bias2": bias2,
        "topk_id": topk_id,
        "topk_w": topk_w,
        "experts": experts,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
    }


def _ref_from_case(case) -> torch.Tensor:
    return _torch_moe_ref(
        case["hidden"].cuda(),
        case["w1_packed"],
        case["w1_scale_raw"],
        case["bias1"].cuda(),
        case["w2_packed"],
        case["w2_scale_raw"],
        case["bias2"].cuda(),
        case["topk_w"].to(torch.bfloat16).cuda(),
        case["topk_id"].to(torch.int32).cuda(),
        data_format="a8w4",
        activation=ActivationType.Swiglu,
        swiglu_limit=_SWIGLU_LIMIT,
    )


def _prepare_from_case(case, block_m: int | None = None) -> dict:
    return _prepare_gluon_inputs(
        hidden=case["hidden"],
        w1_packed=case["w1_packed"],
        w1_scale_raw=case["w1_scale_raw"],
        bias1=case["bias1"],
        w2_packed=case["w2_packed"],
        w2_scale_raw=case["w2_scale_raw"],
        bias2=case["bias2"],
        topk_id=case["topk_id"].to(torch.int32).cuda(),
        topk_w=case["topk_w"].cuda(),
        experts=case["experts"],
        model_dim=case["model_dim"],
        inter_dim=case["inter_dim"],
        block_m_override=block_m,
    )


def _gluon_from_case(
    case, config: dict | None = None, block_m: int | None = None
) -> torch.Tensor:
    return _run_gluon_stages(_prepare_from_case(case, block_m), config)


# ---------------------------------------------------------------------------
# Pytest correctness
# ---------------------------------------------------------------------------
def test_grouped_a8w4_matches_torch_ref():
    if get_arch() != "gfx1250":
        pytest.skip("gluon a8w4 grouped MoE requires gfx1250")

    torch.manual_seed(0)
    case = _make_case(
        experts=8, tokens=8, topk=2, model_dim=512, inter_dim=512
    )
    out = _gluon_from_case(case).to(torch.bfloat16)
    ref = _ref_from_case(case).to(out.dtype)
    rel = _rel_l2(out, ref)
    print(
        f"[gluon a8w4] rel_l2 vs flydsl ref = {rel:.4e} "
        f"(out_norm={float(out.float().norm()):.4e} "
        f"ref_norm={float(ref.float().norm()):.4e})",
        flush=True,
    )
    assert rel < VERIFY_TOL_A8W4, f"rel_l2={rel:.4f} > tol={VERIFY_TOL_A8W4}"


# ---------------------------------------------------------------------------
# CLI: bench / verify at the FlyDSL bench shape
# ---------------------------------------------------------------------------
def _config_from_args(args) -> dict | None:
    # block_m is NOT here: it must be applied to rdata.block_m at prepare time
    # (drives grid + decode/prefill dispatch), passed separately as block_m.
    keys = {
        "block_n": args.block_n,
        "block_k": args.block_k,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "waves_per_eu": args.waves_per_eu,
    }
    cfg = {k: v for k, v in keys.items() if v is not None}
    return cfg or None


def _verify(args) -> None:
    if get_arch() != "gfx1250":
        raise SystemExit("gluon a8w4 grouped MoE requires gfx1250")
    torch.manual_seed(0)
    case = _make_case(
        experts=args.experts,
        tokens=args.tokens,
        topk=args.topk,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
    )
    out = _gluon_from_case(case, _config_from_args(args), args.block_m).to(
        torch.bfloat16
    )
    ref = _ref_from_case(case).to(out.dtype)
    rel = _rel_l2(out, ref)
    print(
        f"[verify a8w4] E={args.experts} T={args.tokens} topk={args.topk} "
        f"K={args.model_dim} I={args.inter_dim} rel_l2={rel:.4e} "
        f"(tol={VERIFY_TOL_A8W4})",
        flush=True,
    )
    if rel >= VERIFY_TOL_A8W4:
        raise SystemExit(f"FAILED: rel_l2={rel:.4f} >= tol={VERIFY_TOL_A8W4}")


def _bench(args) -> None:
    from aiter.test_common import run_perftest

    if get_arch() != "gfx1250":
        raise SystemExit("gluon a8w4 grouped MoE requires gfx1250")
    torch.manual_seed(0)
    case = _make_case(
        experts=args.experts,
        tokens=args.tokens,
        topk=args.topk,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
    )
    config = _config_from_args(args)
    print(
        f"[bench] gluon a8w4 E={args.experts} T={args.tokens} topk={args.topk} "
        f"K={args.model_dim} I={args.inter_dim} block_m={args.block_m} "
        f"config={config} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )

    # One-time setup OUTSIDE the timed region: H2D weight copies, transpose,
    # scale swizzle, routing reconstruction. The timed thunk is the real
    # per-call pipeline only (act-quant -> gemm1 -> requant -> gemm2).
    prep = _prepare_from_case(case, args.block_m)

    def _thunk():
        return _run_gluon_stages(prep, config)

    _thunk()  # warmup / JIT
    torch.cuda.synchronize()
    _, us = run_perftest(_thunk, num_warmup=args.warmup, num_iters=args.iters)
    print(
        f"[bench] gluon a8w4 (act-quant + 2 GEMM) us = {us:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("bench", "verify"), default="bench")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    # Kernel-config overrides (None -> use get_kernel_config_gluon default).
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    args = parser.parse_args()

    if args.scenario == "verify":
        _verify(args)
    else:
        _bench(args)


if __name__ == "__main__":
    main()
