# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness test of the fused TP MegaMoE layer (a4w4, MXFP4, gfx950), torchrun only::

    torchrun --nproc_per_node=4 op_tests/multigpu_tests/test_mega_moe_TP.py --models glm5
    torchrun --nproc_per_node=8 op_tests/multigpu_tests/test_mega_moe_TP.py --tokens 8 64

Reference: torch_moe_stage1 / stage2 on each rank's weight shard, combined with
torch.distributed collectives. Modes (as a transformer layer uses them):

* ``ag_rs``: each rank its own tokens and routing; all-gather, MoE, reduce-scatter.
* ``rs``: every rank the same tokens and routing; the output is reduce-scattered.
* ``ar``: every rank the same tokens and routing; the output is all-reduced
  (identical on every rank).
* ``ar_ar``: as ``ar`` but the input is each rank's partial (summed in the layer).

Every case starts on a new layer and checks its first call (stale state from
an earlier layer would show there), the watchdog code on every rank, NaN, and
the accuracy against the reference. Also: random / skewed routing over several
seeds, token counts changing on one layer, layers created / freed / recreated,
output ownership with ``out=``, CUDA graph replay == eager, masked expert ids,
empty batches, host-side validation, and inputs that must agree across ranks
(``AITER_MEGAMOE_TP_CHECK=1``). Benchmark vs the split path:
op_tests/op_benchmarks/flydsl/bench_mega_moe_TP.py.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from dataclasses import dataclass

import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2
from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig
from aiter.ops.flydsl.moe_common import DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

QUANT_TYPE = aiter.QuantType.per_1x32
FP4 = dtypes.fp4x2
MODES = ("ag_rs", "rs", "ar", "ar_ar")


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

    def log(self, msg: str):
        if self.rank == 0:
            print(msg, flush=True)


def rel_l2(actual, expected) -> float:
    """rel L2 over all ranks."""
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
    """Each rank its own inter_dim shard."""
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
    """Router stand-in: scores -> softmax -> topk -> renormalize (int32 / fp32)."""
    E, K = shape.experts, shape.topk
    if kind == "balanced":
        start = int(torch.randint(0, E, (1,), device=device, generator=gen))
        ids = (start + torch.arange(rows * K, device=device)) % E
        score = torch.full((rows, E), -1e4, device=device)
        score.scatter_(1, ids.view(rows, K), 0.0)
        score += 1e-3 * torch.randn((rows, E), device=device, generator=gen)
    else:
        score = torch.randn((rows, E), device=device, generator=gen)
        if kind == "hot":  # every token prefers the same K experts
            score[:, E - K :] += 100.0
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
    """One call's inputs and the reference of its output on this rank."""

    x: torch.Tensor
    w: torch.Tensor
    ids: torch.Tensor
    ref: torch.Tensor


def make_case(shape, wt, ctx, mode, tokens, kind, seed, mask=0.0) -> Case:
    """tokens: global token count (tp * m). mask: share of routes with ids -1 / >= E."""
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
    else:
        full = torch_partial(shape, wt, x, w_ref, ids_ref)
        dist.all_reduce(full)
        ref = full[ctx.rank * m : (ctx.rank + 1) * m] if mode == "rs" else full
    if mode == "ar_ar":
        # each rank a share of x: x / tp plus noise that sums to 0 over the ranks
        gp = torch.Generator(device=ctx.device).manual_seed(seed + 31)
        noise = torch.randn((ctx.world, *x.shape), device=ctx.device, generator=gp)
        noise -= noise.mean(0, keepdim=True)
        x = (x.float() / ctx.world + 0.1 * noise[ctx.rank]).to(dtypes.bf16)
        xs = x.float()
        dist.all_reduce(xs)
        ref = torch_partial(shape, wt, xs.to(dtypes.bf16), w_ref, ids_ref)
        dist.all_reduce(ref)
    return Case(x.contiguous(), w, ids, ref)


class CaseFailure(AssertionError):
    pass


def new_layer(
    shape, wt, ctx, mode, max_local_tokens, comm_dtype="fp8", ar_gather="bf16"
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
    """One collective call; errors are reduced over the ranks so they all fail together."""
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


def rtol_of(args, comm_dtype, ar_gather="bf16"):
    if ar_gather == "fp8":
        return args.rtol_ag8
    return args.rtol if comm_dtype == "fp8" else args.rtol_bf16


def case_accuracy(
    shape, wt, ctx, args, mode, tokens, kind, comm_dtype="fp8", ar_gather="bf16"
):
    layer = new_layer(
        shape, wt, ctx, mode, args.max_local_tokens, comm_dtype, ar_gather
    )
    worst = 0.0
    for s in range(args.seeds if kind != "balanced" else 1):
        c = make_case(shape, wt, ctx, mode, tokens, kind, args.seed + 1000 * s + tokens)
        y = call(layer, c).clone()
        worst = max(worst, check(y, c, rtol_of(args, comm_dtype, ar_gather)))
        if mode in ("ar", "ar_ar") and not identical_across_ranks(y):
            raise CaseFailure("all-reduce output differs across ranks")
    return worst


def case_varying_m(shape, wt, ctx, args, mode):
    """One layer, the token count changing between calls (decode <-> prefill)."""
    layer = new_layer(shape, wt, ctx, mode, args.max_local_tokens)
    worst = 0.0
    for m in (8, 1, 32, 1, min(64, args.max_local_tokens), 8):
        c = make_case(shape, wt, ctx, mode, m * ctx.world, "random", args.seed + m)
        worst = max(worst, check(call(layer, c).clone(), c, args.rtol, f"m={m}"))
    return worst


def case_layers(shape, wt, ctx, args, mode):
    """Several live layers called in turn; free some, create new ones and check
    their first calls (stale state of a freed layer must not leak)."""
    layers = [new_layer(shape, wt, ctx, mode, args.max_local_tokens) for _ in range(3)]
    worst = 0.0
    for rnd in range(2):
        for i, layer in enumerate(layers):
            c = make_case(shape, wt, ctx, mode, 8 * ctx.world, "random", 100 * rnd + i)
            worst = max(
                worst, check(call(layer, c).clone(), c, args.rtol, f"layer {i}")
            )
        del layers[:2]
        gc.collect()
        layers += [
            new_layer(shape, wt, ctx, mode, args.max_local_tokens) for _ in range(2)
        ]
    return worst


def case_out(shape, wt, ctx, args, mode):
    """With out=, the result is the caller's tensor: the next call leaves it alone."""
    layer = new_layer(shape, wt, ctx, mode, args.max_local_tokens)
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
    return check(keep, c1, args.rtol)


def case_graph(shape, wt, ctx, args, mode):
    layer = new_layer(shape, wt, ctx, mode, args.max_local_tokens)
    c = make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", 3)
    eager = call(layer, c).clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            layer(c.x, c.w, c.ids)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):  # a capture failure fails the case
        out = layer(c.x, c.w, c.ids)
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    flags = torch.tensor(
        [layer.poll_errors(), int(not torch.equal(out, eager))], device=out.device
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    if flags[0]:
        raise CaseFailure(f"watchdog error during graph replay: {int(flags[0])}")
    if flags[1]:
        raise CaseFailure("graph replay != eager")


def case_masked(shape, wt, ctx, args, mode):
    """topk_ids -1 / >= E are masked (add nothing); the next clean call is unaffected."""
    layer = new_layer(shape, wt, ctx, mode, args.max_local_tokens)
    c = make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", 11, mask=0.2)
    e = check(call(layer, c).clone(), c, args.rtol, "masked ids")
    c2 = make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", 12)
    check(call(layer, c2).clone(), c2, args.rtol, "next clean call")
    return e


def case_empty(shape, wt, ctx, args, mode):
    layer = new_layer(shape, wt, ctx, mode, args.max_local_tokens)
    c = make_case(shape, wt, ctx, mode, 4 * ctx.world, "random", 13)
    y = call(layer, Case(c.x[:0], c.w[:0], c.ids[:0], c.ref[:0]))
    if tuple(y.shape) != (0, shape.model_dim):
        raise CaseFailure(f"m=0 returned shape {tuple(y.shape)}")
    check(call(layer, c).clone(), c, args.rtol, "call after m=0")


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


def case_validation(shape, wt, ctx, args):
    """Host-side checks that fire before any collective or kernel launch."""
    expect_raise(
        lambda: new_layer(shape, wt, ctx, "ag_rs", 4096), "max_local_tokens=4096"
    )
    layer = new_layer(shape, wt, ctx, "ag_rs", args.max_local_tokens)
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


def case_checked(shape, wt, ctx, args):
    """AITER_MEGAMOE_TP_CHECK=1: inputs that must agree across ranks are verified."""
    os.environ["AITER_MEGAMOE_TP_CHECK"] = "1"
    try:
        layer = new_layer(shape, wt, ctx, "ag_rs", args.max_local_tokens)
        m = 16 if ctx.rank == 0 else 8
        c = make_case(shape, wt, ctx, "ag_rs", 16 * ctx.world, "random", 14)
        expect_raise(lambda: layer(c.x[:m], c.w[:m], c.ids[:m]), "ag_rs with uneven m")
        layer = new_layer(shape, wt, ctx, "ar", args.max_local_tokens)
        c = make_case(shape, wt, ctx, "ar", 8 * ctx.world, "random", 15)
        ids = c.ids.clone()
        if ctx.rank == 1:
            ids[0, 0] = (ids[0, 0] + 1) % shape.experts
        expect_raise(
            lambda: layer(c.x, c.w, ids), "ar with routing that differs across ranks"
        )
        check(call(layer, c).clone(), c, args.rtol, "checked call")
    finally:
        os.environ.pop("AITER_MEGAMOE_TP_CHECK", None)


def run(name, ctx, results, fn, *fargs):
    ok, info = True, ""
    try:
        out = fn(*fargs)
        info = f"{out:.4f}" if isinstance(out, float) else (out or "")
    except CaseFailure as exc:
        ok, info = False, str(exc)
    except Exception as exc:  # noqa: BLE001 - any other error fails the case
        ok, info = False, f"{type(exc).__name__}: {exc}"
        if ctx.rank == 0:
            traceback.print_exc()
    ok = ctx.all_ok(ok)
    results.append((name, ok, info))
    ctx.log(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")


def run_model(name, ctx, args, results):
    shape = MODELS[name]
    ctx.log(
        f"\n=== {name} tp{ctx.world} h{shape.model_dim} i{shape.inter_dim}/{ctx.world} "
        f"e{shape.experts} k{shape.topk} ==="
    )
    wt = build_weights(shape, ctx, args.seed)
    common = (shape, wt, ctx, args)
    for mode in args.modes:
        for tokens in args.tokens:
            for kind in args.routes:
                run(
                    f"{name} {mode} M={tokens} {kind}",
                    ctx,
                    results,
                    case_accuracy,
                    *common,
                    mode,
                    tokens,
                    kind,
                )
        for cd in args.comm_dtypes:
            if cd != "fp8":
                tokens = args.tokens[-1]
                run(
                    f"{name} {mode} M={tokens} random comm {cd}",
                    ctx,
                    results,
                    case_accuracy,
                    *common,
                    mode,
                    tokens,
                    "random",
                    cd,
                )
        if mode in ("ar", "ar_ar"):
            tokens = args.tokens[-1]
            run(
                f"{name} {mode} M={tokens} random ar_gather fp8",
                ctx,
                results,
                case_accuracy,
                *common,
                mode,
                tokens,
                "random",
                "fp8",
                "fp8",
            )
        for what, fn in (
            ("varying m", case_varying_m),
            ("new / freed layers", case_layers),
            ("out=", case_out),
            ("cuda graph", case_graph),
            ("masked ids", case_masked),
            ("m=0", case_empty),
        ):
            run(f"{name} {mode} {what}", ctx, results, fn, *common, mode)
    run(f"{name} host validation", ctx, results, case_validation, *common)
    run(f"{name} cross-rank checks", ctx, results, case_checked, *common)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="*", default=["glm5", "m3"], choices=list(MODELS))
    p.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    p.add_argument(
        "--tokens", type=int, nargs="*", default=[8, 64, 256], help="global tokens"
    )
    p.add_argument("--routes", nargs="*", default=["balanced", "random", "hot"])
    p.add_argument("--seeds", type=int, default=3, help="seeds per random / hot case")
    p.add_argument("--comm-dtypes", nargs="*", default=["fp8", "bf16"])
    p.add_argument("--max-local-tokens", type=int, default=128)
    p.add_argument("--rtol", type=float, default=0.045, help="fp8 comm: fused vs torch")
    p.add_argument(
        "--rtol-bf16", type=float, default=0.01, help="bf16 comm: fused vs torch"
    )
    p.add_argument(
        "--rtol-ag8", type=float, default=0.05, help="ar_gather fp8: fused vs torch"
    )
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args(argv)

    ctx = Ctx()
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        ctx.log(f"[SKIP] needs gfx950, found {get_gfx()}")
        dist.destroy_process_group()
        return 0
    bad = [
        t
        for t in args.tokens
        if t % ctx.world or t // ctx.world > args.max_local_tokens
    ]
    if bad:
        raise ValueError(
            f"tokens {bad} must be multiples of tp, <= max_local_tokens * tp"
        )

    results: list = []
    for name in args.models:
        run_model(name, ctx, args, results)
        gc.collect()
        torch.cuda.empty_cache()

    failed = [r for r in results if not r[1]]
    ctx.log(f"\n{len(results) - len(failed)}/{len(results)} passed")
    for n, _, info in failed:
        ctx.log(f"  FAIL {n}: {info}")
    dist.barrier()
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
