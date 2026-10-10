# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused TP MegaMoE layer (a4w4, MXFP4, gfx950) vs torch: accuracy + perf sweep, feature
cases and SpRsNorm, each ending in a markdown summary table::

    torchrun --nproc_per_node=4 op_tests/multigpu_tests/test_mega_moe_tp.py \\
        --models m3 glm5 -t 256 512 1024 2048

Run with plain ``python3`` it relaunches itself under torchrun on up to 8 GPUs, or
skips without >= 2 gfx950 GPUs. Exits 1 if any case failed.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mega_moe_tp.sp_rs_norm import SpRsNorm
from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig
from aiter.ops.flydsl.moe_common import DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

SUPPORTED_GFX = ("gfx950",)
SEED = 123
# rel_l2 gate vs the torch reference: fp8 / bf16 comm, MXFP8 all-reduce second hop
RTOL = {"fp8": 0.045, "bf16": 0.01, "ag8": 0.05}
GRAPH_CALLS, PERF_ITERS, PERF_WARMUP = 10, 20, 3

QUANT_TYPE = aiter.QuantType.per_1x32
FP4 = dtypes.fp4x2
MODES = ("ag_rs", "ar")


@dataclass(frozen=True)
class ModelShape:
    name: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    act: aiter.ActivationType


MODELS = {
    "glm5": ModelShape("glm5", 6144, 2048, 257, 9, aiter.ActivationType.Silu),
    "dsv3": ModelShape("dsv3", 7168, 2048, 256, 8, aiter.ActivationType.Silu),
    "dsv4": ModelShape("dsv4", 7168, 3072, 384, 6, aiter.ActivationType.Silu),
    "kimi3": ModelShape("kimi3", 3584, 3072, 896, 16, aiter.ActivationType.Situv2),
    "m3": ModelShape("m3", 6144, 3072, 129, 5, aiter.ActivationType.Swiglu),
}


def situ(shape):
    if shape.act == aiter.ActivationType.Situv2:
        return DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA
    return None


class Ctx:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", self.rank))
        torch.cuda.set_device(local)
        self.device = torch.device("cuda", local)
        dist.init_process_group("nccl", device_id=self.device)

    def all_ok(self, ok: bool) -> bool:
        t = torch.tensor([int(ok)], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())


def rel_l2(actual, expected) -> float:
    d = actual.float() - expected.float()
    t = torch.stack([(d * d).sum(), (expected.float() ** 2).sum()])
    dist.all_reduce(t)
    return float((t[0] / t[1]).sqrt()) if t[1] > 0 else float("nan")


def identical_across_ranks(t: torch.Tensor) -> bool:
    ref = t.clone()
    dist.broadcast(ref, src=0)
    ok = torch.tensor([int(torch.equal(ref, t))], device=t.device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


@dataclass
class Weights:
    inter: int
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w1_ref: torch.Tensor
    w1_scale_ref: torch.Tensor
    w2_ref: torch.Tensor
    w2_scale_ref: torch.Tensor


def _quant_experts(experts, rows, cols, magnitude, seed, device, chunk_bytes=512 << 20):
    quant = aiter.get_torch_quant(QUANT_TYPE)
    chunk = max(1, min(experts, chunk_bytes // max(rows * cols * 2, 1)))
    qt = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scale = None
    gen = torch.Generator(device=device)
    for start in range(0, experts, chunk):
        n = min(chunk, experts - start)
        gen.manual_seed(seed + start)
        w = torch.randn(
            (n, rows, cols), dtype=dtypes.bf16, device=device, generator=gen
        )
        q, s = quant(w.mul_(magnitude), quant_dtype=FP4)
        qt[start : start + n] = q.view(n, rows, cols // 2).view(torch.uint8)
        if scale is None:
            scale = torch.empty(
                (experts * rows, s.shape[-1]), dtype=s.dtype, device=device
            )
        scale[start * rows : (start + n) * rows] = s.view(n * rows, -1)
    return qt.view(FP4), scale


def build_weights(shape: ModelShape, ctx: Ctx, seed: int) -> Weights:
    inter = shape.inter_dim // ctx.world
    H, E = shape.model_dim, shape.experts
    base = seed + 1_000_000 * ctx.rank
    w1, w1s = _quant_experts(E, 2 * inter, H, H**-0.25, base, ctx.device)
    w2, w2s = _quant_experts(E, H, inter, inter**-0.25, base + 7, ctx.device)
    return Weights(
        inter,
        shuffle_weight(w1, layout=(16, 16)),
        fp4_utils.e8m0_shuffle(w1s),
        shuffle_weight(w2, layout=(16, 16)),
        fp4_utils.e8m0_shuffle(w2s),
        w1,
        w1s,
        w2,
        w2s,
    )


def route(rows, shape: ModelShape, kind: str, gen: torch.Generator, device):
    E, K = shape.experts, shape.topk
    if kind == "balanced":
        start = int(torch.randint(0, E, (1,), device=device, generator=gen))
        ids = (start + torch.arange(rows * K, device=device)) % E
        score = torch.full((rows, E), -1e4, device=device)
        score.scatter_(1, ids.view(rows, K), 0.0)
        score += 1e-3 * torch.randn((rows, E), device=device, generator=gen)
    else:
        score = torch.randn((rows, E), device=device, generator=gen)
        if kind == "hot":
            score[:, E - K :] += 100.0
        elif kind == "subset":
            # ~55% of the experts: the LB row-block count lands in its mixed split
            score[:, E * 35 // 64 :] = -1e4
    w, ids = torch.softmax(score, dim=-1).topk(K, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return w.float().contiguous(), ids.to(torch.int32).contiguous()


@torch.no_grad()
def torch_partial(shape, wt: Weights, x, w, ids):
    beta = situ(shape) or (1.0, 1.0)
    quant = aiter.get_torch_quant(QUANT_TYPE)
    a1, a1s = quant(x, quant_dtype=FP4)
    out1 = torch_moe_stage1(
        a1,
        wt.w1_ref,
        wt.w2_ref,
        w,
        ids,
        dtype=dtypes.bf16,
        activation=shape.act,
        quant_type=QUANT_TYPE,
        a1_scale=a1s,
        w1_scale=wt.w1_scale_ref,
        w1_bias=None,
        doweight=False,
        swiglu_limit=None,
        situ_beta=beta[0],
        situ_linear_beta=beta[1],
    )
    a2, a2s = quant(out1, quant_dtype=FP4)
    return torch_moe_stage2(
        a2.view(x.shape[0], shape.topk, -1),
        wt.w1_ref,
        wt.w2_ref,
        w,
        ids,
        dtype=dtypes.bf16,
        quant_type=QUANT_TYPE,
        w2_scale=wt.w2_scale_ref,
        a2_scale=a2s,
        w2_bias=None,
        doweight=True,
    ).float()


@dataclass
class Case:
    x: torch.Tensor
    w: torch.Tensor
    ids: torch.Tensor
    ref: torch.Tensor
    ids_all: torch.Tensor | None = None


def make_case(shape, wt, ctx, mode, tokens, kind, seed, mask=0.0) -> Case:
    m = tokens // ctx.world
    if mode == "ag_rs":
        gen = torch.Generator(device=ctx.device).manual_seed(seed + 7919 * ctx.rank)
        x = torch.randn(
            (m, shape.model_dim), dtype=dtypes.bf16, device=ctx.device, generator=gen
        )
        w, ids = route(m, shape, kind, gen, ctx.device)
    else:
        gen = torch.Generator(device=ctx.device).manual_seed(seed)
        x = torch.randn(
            (tokens, shape.model_dim),
            dtype=dtypes.bf16,
            device=ctx.device,
            generator=gen,
        )
        w, ids = route(tokens, shape, kind, gen, ctx.device)
        for t in (x, w, ids):
            dist.broadcast(t, src=0)
    w_ref, ids_ref = w, ids
    if mask:
        sel = torch.rand(ids.shape, device=ctx.device, generator=gen)
        if mode != "ag_rs":
            dist.broadcast(sel, src=0)
        bad = sel < mask
        ids = torch.where(sel < mask / 2, -1, torch.where(bad, shape.experts + 3, ids))
        ids = ids.to(torch.int32).contiguous()
        w_ref = torch.where(bad, 0.0, w)
        ids_ref = torch.where(bad, 0, ids_ref).to(torch.int32)
    if mode == "ag_rs":
        parts = []
        for t in (x, w_ref, ids_ref):
            g = [torch.empty_like(t) for _ in range(ctx.world)]
            dist.all_gather(g, t)
            parts.append(torch.cat(g))
        full = torch_partial(shape, wt, *parts)
        dist.all_reduce(full)
        ref = full[ctx.rank * m : (ctx.rank + 1) * m]
        ids_all = parts[2]
    else:
        ref = torch_partial(shape, wt, x, w_ref, ids_ref)
        dist.all_reduce(ref)
        ids_all = ids_ref
    return Case(x.contiguous(), w, ids, ref, ids_all)


class CaseFailure(AssertionError):
    pass


def new_layer(
    shape, wt, ctx, mode, max_local_tokens, comm_dtype="fp8", ar_gather="bf16", **kw
):
    beta = situ(shape)
    cfg = MegaMoeTPConfig(
        rank=ctx.rank,
        world_size=ctx.world,
        model_dim=shape.model_dim,
        inter_dim=wt.inter,
        experts=shape.experts,
        topk=shape.topk,
        max_local_tokens=max_local_tokens,
        activation=shape.act,
        beta=beta[0] if beta else None,
        linear_beta=beta[1] if beta else None,
        comm_mode=mode,
        comm_dtype=comm_dtype,
        ar_gather=ar_gather,
        **kw,
    )
    return MegaMoeTP(
        cfg,
        w1=wt.w1,
        w1_scale=wt.w1_scale,
        w2=wt.w2,
        w2_scale=wt.w2_scale,
        device=ctx.device,
    )


def call(layer, c: Case, out=None):
    layer.clear_errors()
    y = layer(c.x, c.w, c.ids, out=out)
    torch.cuda.synchronize()
    flags = torch.tensor(
        [layer.poll_errors(), int(torch.isnan(y).any())], device=y.device
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    if flags[0]:
        raise CaseFailure(f"watchdog error code (max over ranks) {int(flags[0])}")
    if flags[1]:
        raise CaseFailure("NaN in output")
    return y


def check(y, c: Case, rtol, what="fused vs torch"):
    e = rel_l2(y, c.ref)
    if not e < rtol:
        raise CaseFailure(f"{what}: rel_l2={e:.4f} >= {rtol}")
    return e


def case_varying_m(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    worst = 0.0
    for m in (8, 1, 32, 1, min(64, S.max_local), 8):
        c = make_case(shape, wt, ctx, mode, m * ctx.world, "random", SEED + m)
        worst = max(worst, check(call(layer, c).clone(), c, RTOL["fp8"], f"m={m}"))
    return worst


def case_layers(shape, wt, ctx, S, mode):
    layers = [new_layer(shape, wt, ctx, mode, S.max_local) for _ in range(3)]
    worst = 0.0
    for rnd in range(2):
        for i, layer in enumerate(layers):
            c = make_case(shape, wt, ctx, mode, 8 * ctx.world, "random", 100 * rnd + i)
            worst = max(
                worst, check(call(layer, c).clone(), c, RTOL["fp8"], f"layer {i}")
            )
        del layers[:2]
        gc.collect()
        layers += [new_layer(shape, wt, ctx, mode, S.max_local) for _ in range(2)]
    return worst


def case_out(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    tokens = 16 * ctx.world
    c1 = make_case(shape, wt, ctx, mode, tokens, "random", 1)
    c2 = make_case(shape, wt, ctx, mode, tokens, "random", 2)
    out = torch.empty_like(c1.ref, dtype=dtypes.bf16)
    y = call(layer, c1, out=out)
    if y.data_ptr() != out.data_ptr():
        raise CaseFailure("out= was not returned")
    keep = out.clone()
    call(layer, c2)
    if not ctx.all_ok(torch.equal(out, keep)):
        raise CaseFailure("out= of call N changed by call N+1")
    return check(keep, c1, RTOL["fp8"])


def case_graph(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    cases = [
        make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", s) for s in (3, 4, 5)
    ]
    eager = [call(layer, c).clone() for c in cases]
    c = Case(cases[0].x.clone(), cases[0].w.clone(), cases[0].ids.clone(), None)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            layer(c.x, c.w, c.ids)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = layer(c.x, c.w, c.ids)
    same = True
    for i in range(12):
        # every replay on new inputs: stale peer data would show
        k = i % len(cases)
        for dst, src in ((c.x, cases[k].x), (c.w, cases[k].w), (c.ids, cases[k].ids)):
            dst.copy_(src)
        g.replay()
        same = same and torch.equal(out, eager[k])
    torch.cuda.synchronize()
    flags = torch.tensor([layer.poll_errors(), int(not same)], device=out.device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    if flags[0]:
        raise CaseFailure(f"watchdog error during graph replay: {int(flags[0])}")
    if flags[1]:
        raise CaseFailure("graph replay != eager")


def case_dynamic(shape, wt, ctx, S, mode, lb):
    # the dynamic schedule up to 256 tokens (lb: the large-batch one from 1 token up, at
    # its smallest row tile, which takes the mixed GEMM2 split); odd local token
    # counts and a narrow routing
    kw = {"lb_min": 1, "lb_mt": 3, "lb_mt_small": 3, "lb_npp": 2, "lb_q": 3}
    layer = new_layer(
        shape, wt, ctx, mode, S.max_local, **(kw if lb else {"lb_min": 0})
    )
    worst = 0.0
    tokens = sorted({ctx.world, 3 * ctx.world, 33 * ctx.world, S.max_tokens})
    for t in (t for t in tokens if t <= S.max_local * ctx.world):
        for kind in ("random", "subset"):
            c = make_case(shape, wt, ctx, mode, t, kind, SEED + 3 * t)
            y = call(layer, c).clone()
            worst = max(worst, check(y, c, RTOL["fp8"], f"M={t} {kind}"))
    return worst


def ref_tail(y, res, nw, eps):
    # res_out = y + res; this rank's FP8 rows + per-row scales of GemmaRMSNorm(res_out)
    r = y.float() + res.float()
    xn = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * (nw.float() + 1.0)
    scale = xn.to(dtypes.bf16).float().abs().amax(-1).clamp_min(1e-10) / 448.0
    return r.to(dtypes.bf16), xn, scale


def case_tail(shape, wt, ctx, S):
    # fused tail, in place as ATOM calls it: tail=(res, res, w)
    layer = new_layer(shape, wt, ctx, "ag_rs", S.max_local, lb_min=1)
    eng = layer.engine
    t = S.max_tokens
    if not layer.tail_ok(t // ctx.world):
        return "skipped (no tail at this size)"
    c = make_case(shape, wt, ctx, "ag_rs", t, "random", 21)
    gen = torch.Generator(device=ctx.device).manual_seed(77 + ctx.rank)
    res = torch.randn(c.x.shape, generator=gen, device=ctx.device).to(dtypes.bf16)
    nw = (0.1 * torch.randn((shape.model_dim,), device=ctx.device)).to(dtypes.bf16)
    dist.broadcast(nw, src=0)
    y0 = call(layer, c).clone()
    r_ref, xn, s_ref = ref_tail(y0, res, nw, eng.tn_eps)
    gather = []
    for t_ in (xn, s_ref):
        g = [torch.empty_like(t_) for _ in range(ctx.world)]
        dist.all_gather(g, t_)
        gather.append(torch.cat(g))
    xn, s_ref = gather
    fp8 = torch.float8_e4m3fn
    q_ref = (xn.to(dtypes.bf16).float() / s_ref[:, None]).clamp(-448, 448).to(fp8)
    e_ref = float((q_ref.float() * s_ref[:, None] - xn).norm() / xn.norm())
    for bf16 in (False, True):
        r = res.clone()
        outs = layer(c.x, c.w, c.ids, tail=(r, r, nw), bf16=bf16)
        torch.cuda.synchronize()
        y, q, s = outs[:3]
        e_q = float((q.view(fp8).float() * s[:, None] - xn).norm() / xn.norm())
        bad = [
            what
            for what, ok in (
                ("watchdog", not layer.poll_errors()),
                ("y", torch.equal(y, y0)),
                ("res_out", torch.equal(r, r_ref)),
                (f"q rel {e_q:.4f} (fp8 {e_ref:.4f})", e_q < 1.05 * e_ref),
                ("scale", float(((s - s_ref).abs() / s_ref).max()) < 1e-2),
                ("bf16 rows", not bf16 or rel_l2(outs[3], xn) < 0.01),
            )
            if not ok
        ]
        if not ctx.all_ok(not bad):
            raise CaseFailure(f"tail bf16={bf16}: {bad or 'other rank'}")
    return e_q


SPRS_EPS = 1e-6
# SpRsNorm's fused router: experts, top-k, routed scaling, shared-expert weight
SPRS_ROUTER = (128, 4, 2.0, 0.5)


def sp_rs_norm_inputs(hidden, tokens, router, seed):
    """Partial sums, residual and norm weight (+ the router's gate / bias / outputs)."""
    ctx = S.ctx
    E, K = SPRS_ROUTER[:2]
    m = tokens // ctx.world
    gen = torch.Generator(device=ctx.device).manual_seed(7)
    wg = torch.randn((E, hidden), generator=gen, device=ctx.device) * hidden**-0.5 * 3
    bias = (0.05 * torch.randn((E,), generator=gen, device=ctx.device)).float()
    w = (0.1 * torch.randn((hidden,), generator=gen, device=ctx.device)).to(dtypes.bf16)
    g = torch.Generator(device=ctx.device).manual_seed(seed + ctx.rank + tokens)
    part = torch.randn((tokens, hidden), generator=g, device=ctx.device).to(dtypes.bf16)
    res = torch.randn((tokens, hidden), generator=g, device=ctx.device).to(dtypes.bf16)
    rt = None
    if router:
        ids = torch.empty((m, K + 1), dtype=torch.int32, device=ctx.device)
        tw = torch.empty((m, K + 1), dtype=torch.float32, device=ctx.device)
        rt = (wg.to(dtypes.bf16), bias, ids, tw)
    return part, res, w, rt


def check_sp_rs_norm(op, inputs, out, res_out):
    """rel_l2 of (res_out, out) vs torch (+ the router's experts / weights); raises on a
    mismatch. Returns (rel_l2, torch normed rows of this rank)."""
    ctx = S.ctx
    part, res, w, rt = inputs
    E, K, scale, shared_w = SPRS_ROUTER
    m = part.shape[0] // ctx.world
    tot = part.float()
    dist.all_reduce(tot)
    rows = slice(ctx.rank * m, (ctx.rank + 1) * m)
    h = tot[rows] + res[rows].float()
    ref = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + SPRS_EPS) * (w.float() + 1)
    e = max(rel_l2(res_out[rows], h), rel_l2(out[rows], ref))
    bad, info = op.poll_errors() or not e < 0.01, f"rel_l2={e:.4f}"
    if rt is not None:
        wg, bias, ids, tw = rt
        logits = (ref.to(dtypes.bf16).float() @ wg.float().t()).to(dtypes.bf16)
        rid = torch.empty((m, K), dtype=torch.int32, device=ctx.device)
        rtw = torch.empty((m, K), dtype=torch.float32, device=ctx.device)
        aiter.topk_gating(rtw, rid, logits, bias, True, scale, score_func="sigmoid")
        same = (ids[:, :K].sort(1).values == rid.sort(1).values).all(1)
        mine = torch.gather(tw[:, :K], 1, ids[:, :K].argsort(1))
        want = torch.gather(rtw, 1, rid.argsort(1))
        e_w = ((mine - want).abs() / want.abs().clamp_min(1e-6))[same]
        # bf16 logits: near-ties may pick another expert
        st = torch.tensor([float(same.sum()), m], device=ctx.device)
        dist.all_reduce(st)
        e_w = float(e_w.max()) if e_w.numel() else 0.0
        info += f" same experts {float(st[0] / st[1]):.3f} weight err {e_w:.1e}"
        bad = bad or float(st[0] / st[1]) < 0.9 or e_w > 0.02
        bad = bad or not bool((ids[:, K] == E).all() and (tw[:, K] == shared_w).all())
    if not ctx.all_ok(not bad):
        raise CaseFailure(f"M={part.shape[0]} router={rt is not None}: {info}")
    return e, ref


def case_masked(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    c = make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", 11, mask=0.2)
    e = check(call(layer, c).clone(), c, RTOL["fp8"], "masked ids")
    c2 = make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", 12)
    check(call(layer, c2).clone(), c2, RTOL["fp8"], "next clean call")
    return e


def case_empty(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    c = make_case(shape, wt, ctx, mode, 4 * ctx.world, "random", 13)
    y = call(layer, Case(c.x[:0], c.w[:0], c.ids[:0], c.ref[:0]))
    if tuple(y.shape) != (0, shape.model_dim):
        raise CaseFailure(f"m=0 returned shape {tuple(y.shape)}")
    check(call(layer, c).clone(), c, RTOL["fp8"], "call after m=0")


def expect_raise(fn, what, errors=(ValueError, TypeError)):
    raised = 0
    try:
        fn()
        torch.cuda.synchronize()
    except errors:
        raised = 1
    t = torch.tensor([raised], device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if not t.item():
        raise CaseFailure(f"{what} was accepted (expected a host-side error)")


def case_validation(shape, wt, ctx, S):
    expect_raise(
        lambda: new_layer(shape, wt, ctx, "ag_rs", 4096), "max_local_tokens=4096"
    )
    layer = new_layer(shape, wt, ctx, "ag_rs", S.max_local)
    c = make_case(shape, wt, ctx, "ag_rs", 8 * ctx.world, "random", 5)
    x, w, ids = c.x, c.w, c.ids
    expect_raise(lambda: layer(x.float(), w, ids), "fp32 hidden states")
    expect_raise(
        lambda: layer(x[:, : shape.model_dim // 2].contiguous(), w, ids), "hidden size"
    )
    expect_raise(
        lambda: layer(x, w[:, :-1].contiguous(), ids[:, :-1].contiguous()), "topk"
    )
    expect_raise(lambda: layer(x, w, ids.to(torch.int16)), "int16 topk_ids")
    expect_raise(lambda: layer(x, w, ids.cpu()), "topk_ids on the cpu")
    expect_raise(
        lambda: layer(x, w, ids, out=torch.empty_like(x[:1])), "out of the wrong shape"
    )


def case_checked(shape, wt, ctx, S):
    os.environ["AITER_MEGAMOE_TP_CHECK"] = "1"
    try:
        layer = new_layer(shape, wt, ctx, "ag_rs", S.max_local)
        m = 16 if ctx.rank == 0 else 8
        c = make_case(shape, wt, ctx, "ag_rs", 16 * ctx.world, "random", 14)
        expect_raise(lambda: layer(c.x[:m], c.w[:m], c.ids[:m]), "ag_rs with uneven m")
        layer = new_layer(shape, wt, ctx, "ar", S.max_local)
        c = make_case(shape, wt, ctx, "ar", 8 * ctx.world, "random", 15)
        ids = c.ids.clone()
        if ctx.rank == 1:
            ids[0, 0] = (ids[0, 0] + 1) % shape.experts
        expect_raise(
            lambda: layer(c.x, c.w, ids), "ar with routing that differs across ranks"
        )
        check(call(layer, c).clone(), c, RTOL["fp8"], "checked call")
    finally:
        os.environ.pop("AITER_MEGAMOE_TP_CHECK", None)


# Distributed state lives here, out of the @benchmark signatures, so the summary
# tables hold only the sweep axes.
@dataclass
class Session:
    ctx: Ctx
    max_local: int
    max_tokens: int
    weights: dict
    layers: dict


S: Session | None = None


def _rank_max(v: float) -> float:
    t = torch.tensor([float(v)], device=S.ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _layer(model: str, mode: str, comm_dtype: str, ar_gather: str):
    key = (model, mode, comm_dtype, ar_gather)
    if key not in S.layers:
        S.layers[key] = new_layer(
            MODELS[model],
            S.weights[model],
            S.ctx,
            mode,
            S.max_local,
            comm_dtype,
            ar_gather,
        )
    return S.layers[key]


def _graph_us(layer, c: Case, out) -> float:
    """us per call of GRAPH_CALLS back-to-back calls replayed in one CUDA graph (the
    way ATOM runs it), the slowest rank's time."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            layer(c.x, c.w, c.ids, out=out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(GRAPH_CALLS):
            layer(c.x, c.w, c.ids, out=out)
    torch.cuda.synchronize()
    dist.barrier()
    _, us = run_perftest(
        g.replay, num_iters=PERF_ITERS, num_warmup=PERF_WARMUP, use_cuda_event=True
    )
    dist.barrier()
    return _rank_max(us / GRAPH_CALLS)


def _work(shape, wt, c: Case, tokens: int) -> tuple[int, int]:
    """Cluster-wide GEMM FLOPs and algorithmic bytes (input, the active experts' MXFP4
    weights, routing, output) of one layer call."""
    world, H, I, K = S.ctx.world, shape.model_dim, wt.inter, shape.topk
    flops = 6 * tokens * K * H * I * world
    ids = c.ids_all.flatten()
    active = int(torch.unique(ids[(ids >= 0) & (ids < shape.experts)]).numel())
    per_rank = (
        c.x.numel() * 2
        + active * (2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32))
        + c.ids.numel() * 8
        + c.ref.numel() * 2
    )
    return flops, per_rank * world


@benchmark()
def test_mega_moe_tp(model: str, mode: str, tokens: int, route: str, seed: int):
    """MegaMoE TP vs the torch reference (not timed): rel_l2 gate, checkAllclose err,
    CUDA-graph latency and roofline numbers per candidate."""
    ctx, shape, wt = S.ctx, MODELS[model], S.weights[model]
    c = make_case(shape, wt, ctx, mode, tokens, route, SEED + 1000 * seed + tokens)
    # name -> (comm_dtype, ar_gather, rel_l2 gate)
    candidates = {
        "megamoe_tp": ("fp8", "bf16", RTOL["fp8"]),
        "megamoe_tp bf16comm": ("bf16", "bf16", RTOL["bf16"]),
    }
    if mode == "ar":  # the MXFP8 all-gather second hop exists only for ar
        candidates["megamoe_tp ag8"] = ("fp8", "fp8", RTOL["ag8"])
    flops, nbytes = _work(shape, wt, c, tokens)
    ret = {"gfx": get_gfx()}
    for name, (comm, agather, rtol) in candidates.items():
        layer = _layer(model, mode, comm, agather)
        out = torch.empty_like(c.ref, dtype=dtypes.bf16)
        y = call(layer, c, out=out).clone()
        e = rel_l2(y, c.ref)
        ref = c.ref.to(dtypes.fp32)
        err = _rank_max(
            checkAllclose(
                ref,
                y.to(dtypes.fp32),
                rtol=0.1,
                atol=0.1 * float(ref.pow(2).mean().sqrt()),
                msg=f"{name}: ",
                printLog=ctx.rank == 0,
            )
        )
        if not e < rtol:
            raise CaseFailure(f"{name}: rel_l2={e:.4f} >= {rtol}")
        if mode == "ar" and not identical_across_ranks(y):
            raise CaseFailure(f"{name}: all-reduce output differs across ranks")
        us = _graph_us(layer, c, out)
        cfg = layer.launch_config(tokens // ctx.world)
        ret[f"{name} cfg"] = (
            f"mt{cfg.mt}{' lb' if cfg.lb else ''}{' ll' if cfg.ll else ''}"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} rel_l2"] = e
    return ret


CASES = {
    "varying m": case_varying_m,
    "new / freed layers": case_layers,
    "out=": case_out,
    "cuda graph": case_graph,
    "dynamic": lambda *a: case_dynamic(*a, lb=False),
    "dynamic LB": lambda *a: case_dynamic(*a, lb=True),
    "masked ids": case_masked,
    "m=0": case_empty,
}
# model-level cases (fused tail: ag_rs only; the others build their own layers)
MODEL_CASES = {
    "fused tail": case_tail,
    "host validation": case_validation,
    "cross-rank checks": case_checked,
}


@benchmark()
def test_mega_moe_tp_case(model: str, mode: str, case: str):
    """One feature case (varying m, graph replay on new inputs, LB, masked ids, ...)."""
    shape, wt = MODELS[model], S.weights[model]
    if case in CASES:
        out = CASES[case](shape, wt, S.ctx, S, mode)
    else:
        out = MODEL_CASES[case](shape, wt, S.ctx, S)
    return {
        "gfx": get_gfx(),
        "rel_l2": out if isinstance(out, float) else None,
        "note": out if isinstance(out, str) else "",
    }


def _sp_rs_norm_op(hidden):
    key = ("sp_rs_norm", hidden)
    if key not in S.layers:
        S.layers[key] = SpRsNorm(
            hidden, S.max_tokens, SPRS_EPS, device=S.ctx.device, router=SPRS_ROUTER
        )
    return S.layers[key]


@benchmark()
def test_sp_rs_norm(hidden: int, tokens: int, router: bool):
    """SpRsNorm (+ the fused router) vs torch (not timed), twice back to back on new
    inputs, then timed on the second."""
    ctx = S.ctx
    op = _sp_rs_norm_op(hidden)
    if router and not op.routes(tokens):
        raise CaseFailure(f"router does not run at {tokens} tokens")
    for seed in (1000, 2000):
        inputs = sp_rs_norm_inputs(hidden, tokens, router, seed)
        out, res_out = op(*inputs[:3], router=inputs[3])
        torch.cuda.synchronize()
        e, ref = check_sp_rs_norm(op, inputs, out, res_out)
    m = tokens // ctx.world
    rows = slice(ctx.rank * m, (ctx.rank + 1) * m)
    err = _rank_max(
        checkAllclose(
            ref,
            out[rows].to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg="sp_rs_norm: ",
            printLog=ctx.rank == 0,
        )
    )
    dist.barrier()
    _, us = run_perftest(
        op,
        *inputs[:3],
        router=inputs[3],
        num_iters=PERF_ITERS,
        num_warmup=PERF_WARMUP,
        use_cuda_event=True,
    )
    dist.barrier()
    us = _rank_max(us)
    E = SPRS_ROUTER[0]
    # cluster-wide: the router GEMM; bytes per rank: partials in, own rows of the
    # residual in, res_out + out (+ the gate weight) out
    flops = 2 * tokens * E * hidden if router else 0
    nbytes = ctx.world * 2 * hidden * (tokens + 3 * m + (E if router else 0))
    return {
        "gfx": get_gfx(),
        "sp_rs_norm us": us,
        "sp_rs_norm TFLOPS": flops / us / 1e6,
        "sp_rs_norm TB/s": nbytes / us / 1e6,
        "sp_rs_norm err": err,
        "sp_rs_norm rel_l2": e,
    }


def _run(fn, *args) -> dict:
    """fn (a @benchmark test); a failure on any rank is recorded in the row's status."""
    ok, msg, row = True, "", None
    try:
        row = fn(*args)
    except CaseFailure as exc:
        ok, msg = False, str(exc)
    except Exception as exc:  # noqa: BLE001
        ok, msg = False, f"{type(exc).__name__}: {exc}"
        if S.ctx.rank == 0:
            traceback.print_exc()
    ok = S.ctx.all_ok(ok)
    row = row if row is not None else {"case": fn.__name__, "args": args}
    row["status"] = "PASS" if ok else f"FAIL {msg}".strip()
    return row


def summarize(name: str, rows: list) -> None:
    if rows and S.ctx.rank == 0:
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            name,
            pd.DataFrame(rows)
            .pipe(lambda df: df[[c for c in df.columns if c != "status"] + ["status"]])
            .to_markdown(index=False),
        )


def main() -> int:
    global S
    # whole-op arch gate before anything launches (positive allow-list)
    if get_gfx() not in SUPPORTED_GFX:
        if int(os.environ.get("RANK", "0")) == 0:
            aiter.logger.warning("MegaMoE TP unsupported on %s; skipping", get_gfx())
        return 0
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["m3", "glm5"],
        choices=list(MODELS),
        help="Models.\n    e.g.: --models m3",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=list(MODES),
        choices=list(MODES),
        help="comm_mode.\n    e.g.: --modes ag_rs",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[8, 64, 256],
        help="GLOBAL tokens per forward (multiples of tp).\n    e.g.: -t 256 2048",
    )
    parser.add_argument(
        "-r",
        "--routes",
        nargs="*",
        default=["balanced", "random", "hot"],
        choices=["balanced", "random", "hot", "subset"],
        help="Routing distributions.\n    e.g.: -r random",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1],
        help="Input / routing seeds.\n    e.g.: --seeds 0",
    )
    args = parser.parse_args()

    ctx = Ctx()
    if ctx.rank != 0:
        aiter.logger.setLevel("WARNING")
    bad = [t for t in args.tokens if t <= 0 or t % ctx.world]
    if bad:
        raise ValueError(f"tokens {bad} must be positive multiples of tp={ctx.world}")
    max_tokens = max(args.tokens)
    S = Session(ctx, max(max_tokens // ctx.world, 128), max_tokens, {}, {})

    rows = {"accuracy + perf": [], "feature cases": [], "SpRsNorm": []}
    for model in args.models:
        S.weights = {model: build_weights(MODELS[model], ctx, SEED)}
        for mode, tokens, route, seed in itertools.product(
            args.modes, args.tokens, args.routes, args.seeds
        ):
            rows["accuracy + perf"].append(
                _run(test_mega_moe_tp, model, mode, tokens, route, seed)
            )
        for mode, case in itertools.product(args.modes, CASES):
            rows["feature cases"].append(_run(test_mega_moe_tp_case, model, mode, case))
        for case in MODEL_CASES:
            if case != "fused tail" or "ag_rs" in args.modes:
                rows["feature cases"].append(
                    _run(test_mega_moe_tp_case, model, "ag_rs", case)
                )
        S.layers, S.weights = {}, {}
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    routed = max_tokens // (16 * ctx.world) * 16 * ctx.world
    for tokens, router in ((routed, True), (max_tokens, False)):
        if tokens:
            rows["SpRsNorm"].append(_run(test_sp_rs_norm, 6144, tokens, router))
    S.layers = {}

    for name, r in rows.items():
        summarize(f"MegaMoE TP tp{ctx.world} {name}", r)
    failed = [r for rs in rows.values() for r in rs if r["status"] != "PASS"]
    total = sum(len(rs) for rs in rows.values())
    if ctx.rank == 0:
        aiter.logger.info("MegaMoE TP: %d/%d passed", total - len(failed), total)
        for r in failed:
            aiter.logger.error(
                "FAIL %s", {k: v for k, v in r.items() if " " not in k and k != "gfx"}
            )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if failed else 0


def _relaunch() -> int:
    """Plain ``python3`` run (CI): torchrun on up to 8 GPUs, or skip."""
    from aiter.ops.flydsl.mega_moe_tp import mega_moe_tp_supported

    n = torch.cuda.device_count()
    if n < 2 or not mega_moe_tp_supported():
        print("test_mega_moe_tp: skipped (needs >= 2 gfx950 GPUs)", flush=True)
        return 0
    n = 8 if n >= 8 else 4 if n >= 4 else 2
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    cmd += [f"--nproc_per_node={n}", os.path.abspath(__file__), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main() if "RANK" in os.environ else _relaunch())
