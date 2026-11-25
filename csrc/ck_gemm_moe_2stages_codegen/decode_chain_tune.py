# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Candidate search of the decode-shape MoE chain (``flydsl_m3decode`` whole-graph row + ``flydsl_m3route`` router
config) for ``gemm_moe_tune.py --decode-chain``.

For one row count ``n`` (E128 top-4, H6144 / I768, MXFP4 routed experts + MXFP8 shared expert):

* router front (``kernelName2``): every ``router_candidates(n)`` config through ``route_quant_small``, timed on rotated
  gate copies (graph replay), gated on the routes of the default config (same experts, weights within fp32 rounding of
  a different split-K sum order);
* expert chain (``kernelName1``): every ``expert_candidates(n)`` config *through the public ``fused_moe``* (a one-row
  temporary ``tuned_shared_fmoe.csv`` selected with ``AITER_CONFIG_SHARED_FMOE``), cold rotated weights, graph replay,
  gated bit-identical to the default tiling;
* the public path of the same layer (gate GEMM + topk_gating + the tuned 2-stage ``fused_moe`` + shared MXFP8 MLP) as
  the bar: a row is written only when the chain beats it.
"""

import os
import tempfile

import torch

from aiter.fused_moe import fused_moe, get_2stage_cfgs
from aiter.ops.flydsl.kernels.mega_moe_tp import decode_moe_bench as B
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_cfg import (
    ExpertCfg,
    RouterCfg,
    expert_candidates,
    impl_kernel_name,
    router_candidates,
)
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_chain import route_quant_small
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import router_line_layout

SHARED_COLUMNS = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,shared_expert_id,act_type,dtype,q_dtype_a,q_dtype_w,q_type,"
    "use_g1u1,doweight_stage1,hidden_pad,intermediate_pad,gate_mode,block_m,ksplit,us1,kernelName1,err1,us2,"
    "kernelName2,err2,us"
).split(",")


def row_key(n, gfx, cu_num):
    return dict(
        gfx=gfx,
        cu_num=cu_num,
        token=n,
        model_dim=B.H,
        inter_dim=B.I,
        expert=B.E,
        topk=B.TOPK,
        shared_expert_id=-1,
        act_type="ActivationType.Swiglu",
        dtype="torch.bfloat16",
        q_dtype_a="torch.float4_e2m1fn_x2",
        q_dtype_w="torch.float4_e2m1fn_x2",
        q_type="QuantType.per_1x32",
        use_g1u1=1,
        doweight_stage1=0,
        hidden_pad=0,
        intermediate_pad=0,
        gate_mode="GateMode.SEPARATED",
    )


_ROW_FILE = os.path.join(
    tempfile.gettempdir(), f"aiter_decode_chain_rows_{os.getpid()}.csv"
)


class _TempRow:
    """A one-row ``tuned_shared_fmoe.csv`` selected through the environment, so a candidate is timed through the
    public ``fused_moe`` dispatch exactly as a production row is. One file per process, rewritten per candidate:
    the config machinery remembers the path it first saw."""

    def __init__(self, key, kernel1, kernel2):
        row = dict(
            key,
            block_m=16,
            ksplit=0,
            us1="",
            kernelName1=kernel1,
            err1="",
            us2="",
            kernelName2=kernel2,
            err2="",
            us="",
        )
        self.path = _ROW_FILE
        with open(self.path, "w") as f:
            f.write(",".join(SHARED_COLUMNS) + "\n")
            f.write(",".join(str(row[c]) for c in SHARED_COLUMNS) + "\n")

    def __enter__(self):
        import aiter.fused_moe as fm

        self.old = os.environ.get("AITER_CONFIG_SHARED_FMOE")
        os.environ["AITER_CONFIG_SHARED_FMOE"] = self.path
        get_2stage_cfgs.cache_clear()
        fm.cfg_2stages_by_file.clear()
        fm._decode_route_cfg_names.cache_clear()
        return self

    def __exit__(self, *exc):
        import aiter.fused_moe as fm

        if self.old is None:
            os.environ.pop("AITER_CONFIG_SHARED_FMOE", None)
        else:
            os.environ["AITER_CONFIG_SHARED_FMOE"] = self.old
        get_2stage_cfgs.cache_clear()
        fm.cfg_2stages_by_file.clear()
        fm._decode_route_cfg_names.cache_clear()


def _prequant_call(xq, xs, q8, s8, w, ids):
    def call(L):
        return fused_moe(
            xq.view(B.dtypes.fp4x2),
            L["w1"],
            L["w2"],
            w,
            ids,
            activation=B.ActivationType.Swiglu,
            quant_type=B.QuantType.per_1x32,
            w1_scale=L["w1_scale"],
            w2_scale=L["w2_scale"],
            a1_scale=xs,
            swiglu_limit=7.0,
            shared_a1=(q8, s8),
            **B.shared_kw(L),
        )

    return call


def tune_shape(n, gfx, cu_num, nsets=8, reps=30, full=False, log=print):
    """Best (expert, router) configs for ``n`` rows and the times; None when no chain config beats the public path."""
    layers = [B.make_layer(100 + i) for i in range(nsets)]
    gates = [B.make_router(200 + i) for i in range(nsets)]
    lines = [router_line_layout(g) for g, _ in gates]
    x = B.make_x(n)
    gate0, bias0 = gates[0]

    # ---- router front: the routes of the default config are the reference
    default_r = RouterCfg.default(n)
    ref = route_quant_small(x, gate0, lines[0], bias0, B.SCALE, default_r)
    (xq, xs), (q8, s8), w, ids = ref
    best_r, router_us = None, {}
    for cfg in router_candidates(n, full):
        try:
            r = route_quant_small(x, gate0, lines[0], bias0, B.SCALE, cfg)
            if r is None:
                continue
            ok = (
                torch.equal(r[3].sort(-1).values, ids.sort(-1).values)
                and torch.allclose(
                    r[2].sort(-1).values, w.sort(-1).values, rtol=1e-4, atol=1e-5
                )
                and torch.equal(r[0][0], xq)
                and torch.equal(r[1][0], q8)
            )
            if not ok:
                log(
                    f"[decode-chain] n={n} router {cfg.name()}: routes differ from the default, dropped"
                )
                continue
            sets = list(range(nsets))
            us = B.graph_median_us(
                lambda i: route_quant_small(
                    x, gates[i][0], lines[i], gates[i][1], B.SCALE, cfg
                ),
                sets,
                reps,
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a candidate the kernels cannot build is not a result
            log(
                f"[decode-chain] n={n} router {cfg.name()}: {type(exc).__name__}: {str(exc)[:100]}"
            )
            continue
        router_us[cfg.name()] = us
        log(f"[decode-chain] n={n} router {cfg.name()}: {us:.2f} us")
        if best_r is None or us < router_us[best_r.name()]:
            best_r = cfg

    # ---- expert chain through fused_moe, prequantized rows as the router entry hands them over
    key = row_key(n, gfx, cu_num)
    call = _prequant_call(xq, xs, q8, s8, w, ids)
    ref_out, best_e, expert_us = None, None, {}
    for cfg in expert_candidates(n, full):
        try:
            with _TempRow(key, impl_kernel_name(cfg), default_r.name()):
                out = call(layers[0])
                if ref_out is None:
                    ref_out = out.clone()
                elif not torch.equal(out, ref_out):
                    log(
                        f"[decode-chain] n={n} expert {cfg.name()}: output differs from the default tiling, dropped"
                    )
                    continue
                us = B.graph_median_us(call, layers, reps)
        except Exception as exc:  # noqa: BLE001
            log(
                f"[decode-chain] n={n} expert {cfg.name()}: {type(exc).__name__}: {str(exc)[:100]}"
            )
            continue
        expert_us[cfg.name()] = us
        log(f"[decode-chain] n={n} expert {cfg.name()}: {us:.2f} us")
        if best_e is None or us < expert_us[best_e.name()]:
            best_e = cfg
    if best_e is None or best_r is None:
        return None

    # ---- the bar: the public path of the same layer
    gate_bf16, bias = gate0.bfloat16(), bias0
    public_us = B.graph_median_us(
        lambda L: B.public_path(x, L, gate_bf16, bias), layers, reps
    )
    us1, us2 = expert_us[best_e.name()], router_us[best_r.name()]
    log(
        f"[decode-chain] n={n}: chain {us1:.2f} (experts) + {us2:.2f} (router) = {us1 + us2:.2f} us vs public {public_us:.2f} us"
    )
    if us1 + us2 >= public_us:
        return None
    return dict(
        key,
        block_m=16,
        ksplit=0,
        us1=round(us1, 4),
        kernelName1=impl_kernel_name(best_e),
        err1="0.0%",
        us2=round(us2, 4),
        kernelName2=best_r.name(),
        err2="0.0%",
        us=round(us1 + us2, 4),
        public_us=round(public_us, 4),
    )


__all__ = ["tune_shape", "row_key", "SHARED_COLUMNS", "ExpertCfg"]
