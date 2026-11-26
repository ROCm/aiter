# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-FP8 MoE forward (power-of-two 1x128 activation / 128x128 weight scales).

Candidates: the FlyDSL persistent 8-wave ping-pong kernels (``moe_bs_forward``)
and AITER's per_1x128 ``fused_moe``. Graded against an FP32 reference on the
dequantized weights with ideal (unquantized) activations: relative L2 < 0.06,
the bar of SILOTIGER-1763 / PR 6742. Shapes: D 4096 / I 1024, 288 routed
experts (+1 shared expert, selected by every token, for top-k 9).
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
ROUTED_EXPERTS = 288
TOL = 0.06


def requant_pow2(w):
    """[E, n, k] float -> fp8 with power-of-two 128x128 block scales [E, n/128, k/128]."""
    E, n, k = w.shape
    b = w.float().view(E, n // 128, 128, k // 128, 128)
    s = torch.exp2(
        torch.ceil(
            torch.log2(b.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0)
        )
    )
    return (b / s).to(dtypes.fp8).view(E, n, k), s.view(E, n // 128, k // 128)


def dequant(q, s):
    E, n, k = q.shape
    b = q.float().view(E, n // 128, 128, k // 128, 128)
    return (b * s.view(E, n // 128, 1, k // 128, 1)).view(E, n, k)


def make_routing(T, E, topk, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, 4096, dtype=dtypes.bf16, generator=gen)
    routed = E - int(topk == 9)
    tw, ids = torch.topk(
        torch.randn(T, routed, generator=gen).softmax(-1), 8, dim=-1, sorted=False
    )
    if topk == 9:
        ids = torch.cat([ids, torch.full_like(ids[:, :1], E - 1)], dim=1)
        tw = torch.cat([tw, torch.ones_like(tw[:, :1])], dim=1)
    return x, ids.to(dtypes.i32), tw


def run_torch(x, W13, W2, tw, ids, limit):
    """FP32 oracle: sum_k tw_k * W2[e_k] act(W13[e_k] x); ids outside [0, E) contribute nothing."""
    out = torch.zeros(x.shape[0], W2.shape[1])
    for e in torch.unique(ids).tolist():
        if e < 0 or e >= W13.shape[0]:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        g, u = (x[tok].float() @ W13[e].t()).chunk(2, dim=-1)
        if limit > 0:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        act = torch.nn.functional.silu(g) * u
        out.index_add_(0, tok, (act @ W2[e].t()) * tw[tok, slot, None])
    return out


_WEIGHTS = {}


def weights(E, inter, dim):
    if E not in _WEIGHTS:
        gen = torch.Generator(device="cuda").manual_seed(0)
        w13, s13 = requant_pow2(torch.randn(E, 2 * inter, dim, generator=gen) * 0.02)
        w2, s2 = requant_pow2(torch.randn(E, dim, inter, generator=gen) * 0.02)
        _WEIGHTS.clear()
        _WEIGHTS[E] = (w13, s13, w2, s2)
    return _WEIGHTS[E]


def rel_l2(out, ref):
    return ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()


@benchmark()
def test_moe_blockscale(tokens, topk, swiglu_limit, routing):
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_blockscale_pp import e8m0_from_pow2, moe_bs_forward

    dim, inter = 4096, 1024
    E = ROUTED_EXPERTS + int(topk == 9)
    w13, s13, w2, s2 = weights(E, inter, dim)
    w13s, w2s = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    e13, e2 = e8m0_from_pow2(s13), e8m0_from_pow2(s2)
    x, ids, tw = make_routing(tokens, E, topk, seed=tokens * 10 + topk)
    if routing == "invalid":
        # Out-of-range and negative experts contribute nothing; a whole token may be dropped.
        bad = torch.rand(ids.shape) < 0.2
        ids = torch.where(bad, torch.where(torch.rand(ids.shape) < 0.5, -1, E + 7), ids)
        ids[0] = -1
    elif routing == "hot":
        # Every token on expert 0: one expert spans many 256-row tiles. Rotating
        # each row keeps its experts distinct (moe_sorting keeps one route of a
        # token that selects an expert twice; top-k routers never do).
        routed = E - int(topk == 9)
        ids[:, :8] = (ids[:, :8] - ids[:, :1]) % routed
    rows = torch.linspace(0, tokens - 1, min(tokens, 256)).long()
    ref = run_torch(
        x[rows], dequant(w13, s13), dequant(w2, s2), tw[rows], ids[rows], swiglu_limit
    )

    candidates = {}
    out = torch.full_like(x, float("nan"))  # unwritten rows must fail the check
    candidates["flydsl_pp"] = lambda: moe_bs_forward(
        x, ids, tw, w13s, e13, w2s, e2, out=out, swiglu_limit=swiglu_limit
    )
    if swiglu_limit == 0 and routing == "random":
        # What fused_moe's per_1x128 dispatch runs for this shape: the
        # flydsl_bspp backend from the T=4096 bucket up (tuned rows), CK / asm
        # below. Timed for the plain routing only.
        w13s.is_shuffled = w2s.is_shuffled = True
        out_aiter = torch.empty_like(x)
        candidates["fused_moe"] = lambda: fused_moe(
            x,
            w13s,
            w2s,
            tw,
            ids,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_1x128,
            w1_scale=s13,
            w2_scale=s2,
            dtype=dtypes.bf16,
            output=out_aiter,
        )

    routed_rows = int((ids >= 0).logical_and(ids < E).sum())
    flops = routed_rows * 2 * (dim * 2 * inter + inter * dim)
    nbytes = (
        E * (2 * inter * dim + dim * inter)  # every expert's weights once
        + tokens * dim * 2 * 2  # bf16 x in, out
    )
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        res, us = run_perftest(fn, num_iters=20, num_warmup=3)
        err = rel_l2(res[rows], ref)
        assert torch.isfinite(res).all(), f"{name}: non-finite output"
        assert err < TOL, f"{name}: rel L2 {err:.4f} >= {TOL}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    if swiglu_limit > 0:
        # The limit must change the result, or the clamp was not exercised.
        unclamped = run_torch(
            x[rows], dequant(w13, s13), dequant(w2, s2), tw[rows], ids[rows], 0.0
        )
        assert swiglu_limit > 5 or rel_l2(unclamped, ref) > 0.12, "clamp not exercised"
    # Bitwise determinism (no atomics in the FlyDSL path).
    a = candidates["flydsl_pp"]().clone()
    b = candidates["flydsl_pp"]()
    assert torch.equal(a, b), "flydsl_pp: nondeterministic output"
    return ret


@benchmark()
def test_moe_blockscale_graph(tokens, topk):
    """One HIP graph replayed on new routing and new x (the tile count is read on device)."""
    from aiter.ops.flydsl.moe_blockscale_pp import e8m0_from_pow2, moe_bs_forward

    E = ROUTED_EXPERTS + int(topk == 9)
    w13, s13, w2, s2 = weights(E, 1024, 4096)
    w13s, w2s = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    e13, e2 = e8m0_from_pow2(s13), e8m0_from_pow2(s2)
    x, ids, tw = make_routing(tokens, E, topk, seed=1)
    out = torch.empty_like(x)
    moe_bs_forward(x, ids, tw, w13s, e13, w2s, e2, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_bs_forward(x, ids, tw, w13s, e13, w2s, e2, out=out)
    errs = []
    for seed in (2, 3):
        x2, ids2, tw2 = make_routing(tokens, E, topk, seed=seed)
        x.copy_(x2)
        ids.copy_(ids2)
        tw.copy_(tw2)
        graph.replay()
        rows = torch.linspace(0, tokens - 1, min(tokens, 256)).long()
        ref = run_torch(
            x[rows], dequant(w13, s13), dequant(w2, s2), tw[rows], ids[rows], 0.0
        )
        errs.append(rel_l2(out[rows], ref))
        assert errs[-1] < TOL, f"graph replay: rel L2 {errs[-1]:.4f}"
    return {"gfx": get_gfx_runtime(), "flydsl_pp err": max(errs)}


@benchmark()
def test_moe_blockscale_chunked(tokens, topk, chunks):
    """Token-chunked forward (the large-T path) is bitwise equal to one pass.

    ``_MAX_SLOT_BYTES`` is lowered so ``tokens`` splits into ``chunks`` passes;
    every output row depends only on its own token, so the bytes must match.
    """
    import aiter.ops.flydsl.moe_blockscale_pp as bspp

    E = ROUTED_EXPERTS + int(topk == 9)
    w13, s13, w2, s2 = weights(E, 1024, 4096)
    w13s, w2s = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    e13, e2 = bspp.e8m0_from_pow2(s13), bspp.e8m0_from_pow2(s2)
    x, ids, tw = make_routing(tokens, E, topk, seed=5)
    whole = bspp.moe_bs_forward(x, ids, tw, w13s, e13, w2s, e2)
    saved = bspp._MAX_SLOT_BYTES
    try:
        bspp._MAX_SLOT_BYTES = -(-tokens // chunks) * topk * 4096 * 2
        chunked = bspp.moe_bs_forward(x, ids, tw, w13s, e13, w2s, e2)
    finally:
        bspp._MAX_SLOT_BYTES = saved
    assert torch.equal(whole, chunked), "chunked forward differs from one pass"
    return {"gfx": get_gfx_runtime(), "chunks": chunks, "equal": True}


@benchmark()
def test_moe_blockscale_dispatch(tokens, topk, swiglu_limit):
    """fused_moe reaches the FlyDSL backend through an ``impl__flydsl_bspp__default`` row.

    The tuned table is injected (one row for this shape), and the registered
    entry is wrapped to count its calls; the result must match the reference.
    """
    import os
    import tempfile

    import aiter.fused_moe as fm
    import aiter.ops.flydsl.moe_blockscale_pp as bspp
    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num

    E = ROUTED_EXPERTS + int(topk == 9)
    w13, s13, w2, s2 = weights(E, 1024, 4096)
    w13s, w2s = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    w13s.is_shuffled = w2s.is_shuffled = True
    x, ids, tw = make_routing(tokens, E, topk, seed=7)
    header = (
        "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,"
        "q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,"
        "us2,kernelName2,err2,us,run_1stage,xbf16,flat,tflops,bw,_tag"
    )
    row = (
        f"gfx950,{get_cu_num()},{fm.get_padded_M(tokens)},4096,1024,{E},{topk},"
        "ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn,"
        "QuantType.per_1x128,1,0,256,0,1.0,impl__flydsl_bspp__default,0.0%,0.0,,0.0%,"
        "1.0,0,0,0,0,0,"
    )
    calls = []
    real = bspp.run_moe_bs_pp_impl

    def counting(request, config):
        calls.append(config)
        return real(request, config)

    with tempfile.NamedTemporaryFile("w", suffix="_tuned_fmoe.csv", delete=False) as f:
        f.write(header + "\n" + row + "\n")
    saved_env = os.environ.get("AITER_CONFIG_FMOE")
    try:
        os.environ["AITER_CONFIG_FMOE"] = f.name
        bspp.run_moe_bs_pp_impl = counting
        type(AITER_CONFIGS).get_config_file.cache_clear()
        fm.cfg_2stages = None
        fm.get_2stage_cfgs.cache_clear()
        out = fm.fused_moe(
            x,
            w13s,
            w2s,
            tw,
            ids,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_1x128,
            w1_scale=s13,
            w2_scale=s2,
            dtype=dtypes.bf16,
            swiglu_limit=swiglu_limit or None,
        )
    finally:
        bspp.run_moe_bs_pp_impl = real
        if saved_env is None:
            os.environ.pop("AITER_CONFIG_FMOE", None)
        else:
            os.environ["AITER_CONFIG_FMOE"] = saved_env
        type(AITER_CONFIGS).get_config_file.cache_clear()
        fm.cfg_2stages = None
        fm.get_2stage_cfgs.cache_clear()
        os.unlink(f.name)
    assert calls == ["default"], f"fused_moe did not dispatch to flydsl_bspp: {calls}"
    rows = torch.linspace(0, tokens - 1, min(tokens, 256)).long()
    ref = run_torch(
        x[rows], dequant(w13, s13), dequant(w2, s2), tw[rows], ids[rows], swiglu_limit
    )
    err = rel_l2(out[rows], ref)
    assert err < TOL, f"fused_moe -> flydsl_bspp: rel L2 {err:.4f}"
    return {"gfx": get_gfx_runtime(), "dispatched": calls[0], "err": err}


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "moe_blockscale_pp unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=[1, 32, 128, 1024, 4096, 16384]
    )
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[8, 9])
    parser.add_argument(
        "-l", "--swiglu_limit", type=float, nargs="*", default=[0.0, 1.0, 10.0]
    )
    parser.add_argument(
        "-r", "--routing", nargs="*", default=["random", "invalid", "hot"]
    )
    args = parser.parse_args()

    df = []
    for topk, tokens, limit, routing in itertools.product(
        args.topk, args.tokens, args.swiglu_limit, args.routing
    ):
        if routing != "random" and limit != 0:
            continue  # edge routings once, unclamped
        df.append(test_moe_blockscale(tokens, topk, limit, routing))
    aiter.logger.info(
        "moe_blockscale summary (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )
    df = [
        test_moe_blockscale_dispatch(t, k, limit)
        for k in args.topk
        for t in (4096,)
        for limit in (0.0, 1.0)
    ]
    aiter.logger.info(
        "moe_blockscale fused_moe dispatch (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )
    df = [test_moe_blockscale_graph(t, k) for k in args.topk for t in (128, 4096)]
    aiter.logger.info(
        "moe_blockscale graph replay (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )
    df = [test_moe_blockscale_chunked(4097, k, 3) for k in args.topk]
    aiter.logger.info(
        "moe_blockscale token chunks (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
