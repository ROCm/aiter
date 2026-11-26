# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-FP8 MoE forward (power-of-two 1x128 activation / 128x128 weight scales).

Candidates: the FlyDSL persistent 8-wave ping-pong kernels (``moe_bs_forward``)
and AITER's per_1x128 ``fused_moe``. Two FP32 references on the dequantized
weights (PR 6742's):

* ideal (unquantized) activations: relative L2 < 0.06, the bar of PR 6742;
* quantization-aware, modelling both 1x128 FP8 activation rounds and the bf16
  expert and final outputs: the FlyDSL residual must stay < 0.01, so an
  arithmetic, routing or clamp defect cannot hide in the quantization error.

Shapes: D 4096 / I 1024, 288 routed experts (+1 shared expert, selected by
every token, for top-k 9).
"""

import argparse
import contextlib
import csv
import itertools
import os
import tempfile

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
# Per row too: one wrong row can stay under a global bound (worst seen: 0.062).
ROW_TOL = 0.1
# Against the quantization-aware reference (worst seen: 0.0022, a row 0.0092).
RESIDUAL_TOL = 0.01
RESIDUAL_ROW_TOL = 0.02


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


def fp8_roundtrip(x):
    """``x`` after power-of-two 1x128 E4M3 quantization: scale 2^ceil(log2(max(amax, 1e-10) / 448))."""
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 128, 128)
    scale = torch.exp2(
        torch.ceil(
            torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0)
        )
    )
    return ((blocks / scale).to(dtypes.fp8).float() * scale).reshape(x.shape)


def run_torch(x, w, tw, ids, limit, quantized=False):
    """FP32 oracle: sum_k tw_k * W2[e_k] act(W13[e_k] x); ids outside [0, E) contribute nothing.

    ``w`` = (w13, s13, w2, s2), dequantized one expert at a time. ``quantized``
    rounds x and act through ``fp8_roundtrip`` and the expert and final
    outputs through bf16, as the kernels store them.
    """
    w13, s13, w2, s2 = w
    out = torch.zeros(x.shape[0], w2.shape[1])
    x = fp8_roundtrip(x) if quantized else x.float()
    for e in torch.unique(ids).tolist():
        if e < 0 or e >= w13.shape[0]:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        g, u = (x[tok] @ dequant(w13[e : e + 1], s13[e : e + 1])[0].t()).chunk(2, -1)
        if limit > 0:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        act = torch.nn.functional.silu(g) * u
        if quantized:
            act = fp8_roundtrip(act)
        y = act @ dequant(w2[e : e + 1], s2[e : e + 1])[0].t()
        if quantized:
            y = y.bfloat16().float()
        out.index_add_(0, tok, y * tw[tok, slot, None])
    return out.bfloat16().float() if quantized else out


_WEIGHTS = {}


def weights(E, inter, dim):
    """Seeded fp8 expert weights with power-of-two block scales, made 64 experts at a time."""
    if E not in _WEIGHTS:
        gen = torch.Generator(device="cuda").manual_seed(0)

        def make(n, k):
            parts = [
                requant_pow2(torch.randn(min(64, E - e), n, k, generator=gen) * 0.02)
                for e in range(0, E, 64)
            ]
            return torch.cat([q for q, _ in parts]), torch.cat([s for _, s in parts])

        w13, s13 = make(2 * inter, dim)
        w2, s2 = make(dim, inter)
        _WEIGHTS.clear()
        _WEIGHTS[E] = (w13, s13, w2, s2)
    return _WEIGHTS[E]


def rel_l2(out, ref):
    return ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def row_rel_l2(out, ref):
    """Worst per-row relative L2; a fully dropped token (zero reference row) must be zero."""
    diff = (out.float() - ref).norm(dim=1)
    return (diff / ref.norm(dim=1).clamp_min(1e-6)).max().item()


def bspp_row(tokens, E, topk):
    """A tuned-CSV row sending fused_moe's per_1x128 call for this shape to ``moe_bs_forward``."""
    import aiter.fused_moe as fm
    from aiter.jit.utils.chip_info import get_cu_num

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
    return dict(zip(header.split(","), row.split(",")))


@contextlib.contextmanager
def tuned_rows(rows):
    """Inside the block, fused_moe reads its tuned configs from ``rows`` (dicts) only."""
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS

    def reset():
        type(AITER_CONFIGS).get_config_file.cache_clear()
        fm.cfg_2stages = None
        fm.cfg_2stages_by_file.clear()
        fm.get_2stage_cfgs.cache_clear()

    with tempfile.NamedTemporaryFile("w", suffix="_tuned_fmoe.csv", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved = os.environ.get("AITER_CONFIG_FMOE")
    os.environ["AITER_CONFIG_FMOE"] = f.name
    reset()
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("AITER_CONFIG_FMOE", None)
        else:
            os.environ["AITER_CONFIG_FMOE"] = saved
        reset()
        os.unlink(f.name)


def _setup(tokens, E, topk, seed):
    from aiter.ops.flydsl.moe_blockscale_pp import e8m0_from_pow2

    w = weights(E, 1024, 4096)
    w13s, w2s = shuffle_weight(w[0], (16, 16)), shuffle_weight(w[2], (16, 16))
    packed = (w13s, e8m0_from_pow2(w[1]), w2s, e8m0_from_pow2(w[3]))
    return w, packed, make_routing(tokens, E, topk, seed)


def _route(ids, E, topk, routing):
    """``ids`` with a routing pattern applied in place."""
    routed = E - int(topk == 9)
    if routing == "hot":
        # Every token on expert 0: one expert spans many 256-row tiles. Rotating
        # each row keeps its experts distinct (moe_sorting keeps one route of a
        # token that selects an expert twice; top-k routers never do).
        ids[:, :8] = (ids[:, :8] - ids[:, :1]) % routed
    elif routing == "invalid":
        # Out-of-range and negative ids contribute nothing; token 0 is dropped.
        flat = ids.view(-1)
        flat[::3], flat[1::5] = -1, 2**31 - 1
        flat[2::7] = E
        ids[0] = -1
    elif routing == "empty":
        ids.fill_(-1)
    return ids


def _check(name, out, ref, qref=None):
    """Ideal-reference bounds; with ``qref`` also the quantization-aware residual."""
    assert torch.isfinite(out).all(), f"{name}: non-finite output"
    err, row_err = rel_l2(out, ref), row_rel_l2(out, ref)
    assert (
        err < TOL and row_err < ROW_TOL
    ), f"{name}: rel L2 {err:.4f} / row {row_err:.4f}"
    if qref is None:
        return err, None
    res, row_res = rel_l2(out, qref), row_rel_l2(out, qref)
    assert (
        res < RESIDUAL_TOL and row_res < RESIDUAL_ROW_TOL
    ), f"{name}: residual {res:.4f} / row {row_res:.4f}"
    return err, res


@benchmark()
def test_moe_blockscale(tokens, topk, swiglu_limit, routing):
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_blockscale_pp import moe_bs_forward

    dim, inter = 4096, 1024
    E = ROUTED_EXPERTS + int(topk == 9)
    w, packed, (x, ids, tw) = _setup(tokens, E, topk, seed=tokens * 10 + topk)
    _route(ids, E, topk, routing)
    rows = torch.linspace(0, tokens - 1, min(tokens, 256)).long()
    ref = run_torch(x[rows], w, tw[rows], ids[rows], swiglu_limit)
    qref = run_torch(x[rows], w, tw[rows], ids[rows], swiglu_limit, quantized=True)

    candidates = {}
    out = torch.full_like(x, float("nan"))  # unwritten rows must fail the check
    candidates["flydsl_pp"] = lambda: moe_bs_forward(
        x, ids, tw, *packed, out=out, swiglu_limit=swiglu_limit
    )
    if swiglu_limit == 0 and routing == "random":
        # What fused_moe's per_1x128 dispatch runs for this shape: the
        # flydsl_bspp backend from the T=4096 bucket up (tuned rows), CK / asm
        # below. Timed for the plain routing only.
        w13s, w2s = packed[0], packed[2]
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
            w1_scale=w[1],
            w2_scale=w[3],
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
        # The residual bound models the FlyDSL pipeline's rounding steps only.
        err, residual = _check(
            name, res[rows], ref, qref if name == "flydsl_pp" else None
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        if residual is not None:
            ret[f"{name} residual"] = residual
    if swiglu_limit > 0:
        # The limit must change the result, or the clamp was not exercised.
        unclamped = run_torch(x[rows], w, tw[rows], ids[rows], 0.0, quantized=True)
        assert swiglu_limit > 5 or rel_l2(unclamped, qref) > 0.12, "clamp not exercised"
    # Bitwise determinism (no atomics in the FlyDSL path) for the same inputs.
    a = candidates["flydsl_pp"]().clone()
    b = candidates["flydsl_pp"]()
    assert torch.equal(a, b), "flydsl_pp: nondeterministic output"
    return ret


@benchmark()
def test_moe_blockscale_graph(tokens, topk, experts):
    """One HIP graph (limit 1) replayed on changed x, weights and routing.

    The tile count is read on device, so each replay must follow the new
    routing: random, one hot expert, invalid ids, no valid id at all, then
    random again (no state left from the empty pass). Each replay runs twice and
    must give the same bytes.
    """
    from aiter.ops.flydsl.moe_blockscale_pp import moe_bs_forward

    E = experts + int(topk == 9)
    w, packed, (x, ids, tw) = _setup(tokens, E, topk, seed=1)
    original = ids.clone()
    out = torch.empty_like(x)
    moe_bs_forward(x, ids, tw, *packed, out=out, swiglu_limit=1.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_bs_forward(x, ids, tw, *packed, out=out, swiglu_limit=1.0)
    rows = torch.linspace(0, tokens - 1, min(tokens, 256)).long()
    worst = 0.0
    for routing in ("random", "hot", "invalid", "empty", "random"):
        ids.copy_(_route(original.clone(), E, topk, routing))
        x.mul_(1.03125)
        tw.mul_(0.9375)
        graph.replay()
        first = out.clone()
        graph.replay()
        assert torch.equal(first, out), f"graph replay ({routing}): nondeterministic"
        if routing == "empty":
            assert not out.any(), "graph replay (empty): nonzero output"
            continue
        ref = run_torch(x[rows], w, tw[rows], ids[rows], 1.0)
        qref = run_torch(x[rows], w, tw[rows], ids[rows], 1.0, quantized=True)
        worst = max(worst, _check(f"graph replay ({routing})", out[rows], ref, qref)[1])
    return {"gfx": get_gfx_runtime(), "flydsl_pp residual": worst}


@benchmark()
def test_moe_blockscale_inputs(topk):
    """int64 ids (out of int32 range ones dropped, not wrapped), T = 0, unsupported shapes."""
    from aiter.ops.flydsl.moe_blockscale_pp import moe_bs_forward

    E = ROUTED_EXPERTS + int(topk == 9)
    _, packed, (x, ids, tw) = _setup(64, E, topk, seed=3)
    ids64 = ids.long()
    ids64[:, ::2] = 2**32
    ids64[:, 1::4] = -(2**32)
    masked = ids.clone()
    masked[:, ::2] = -1
    masked[:, 1::4] = -1
    expected = moe_bs_forward(x, masked, tw, *packed)
    assert torch.equal(moe_bs_forward(x, ids64, tw, *packed), expected), "int64 ids"
    empty = moe_bs_forward(x[:0], ids[:0], tw[:0], *packed)
    assert empty.shape == (0, 4096), "T = 0"
    w13, e13, w2, e2 = packed
    try:
        # I = 512: the kernels are built for D 4096 / I 1024 only.
        moe_bs_forward(x, ids, tw, w13[:, :1024], e13[:, :8], w2[:, :, :512], e2)
        raise AssertionError("I = 512 was not rejected")
    except ValueError:
        pass
    return {"gfx": get_gfx_runtime(), "int64": True, "empty": True, "rejects": True}


@benchmark()
def test_moe_blockscale_chunked(tokens, topk, chunks):
    """Token-chunked forward (the large-T path) is bitwise equal to one pass.

    ``_MAX_SLOT_BYTES`` is lowered so ``tokens`` splits into ``chunks`` passes;
    every output row depends only on its own token, so the bytes must match.
    """
    import aiter.ops.flydsl.moe_blockscale_pp as bspp

    E = ROUTED_EXPERTS + int(topk == 9)
    _, packed, (x, ids, tw) = _setup(tokens, E, topk, seed=5)
    whole = bspp.moe_bs_forward(x, ids, tw, *packed)
    saved = bspp._MAX_SLOT_BYTES
    try:
        bspp._MAX_SLOT_BYTES = -(-tokens // chunks) * topk * 4096 * 2
        chunked = bspp.moe_bs_forward(x, ids, tw, *packed)
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
    import aiter.fused_moe as fm
    import aiter.ops.flydsl.moe_blockscale_pp as bspp

    E = ROUTED_EXPERTS + int(topk == 9)
    w, packed, (x, ids, tw) = _setup(tokens, E, topk, seed=7)
    w13s, w2s = packed[0], packed[2]
    w13s.is_shuffled = w2s.is_shuffled = True
    calls = []
    real = bspp.run_moe_bs_pp_impl

    def counting(request, config):
        calls.append(config)
        return real(request, config)

    try:
        bspp.run_moe_bs_pp_impl = counting
        with tuned_rows([bspp_row(tokens, E, topk)]):
            out = fm.fused_moe(
                x,
                w13s,
                w2s,
                tw,
                ids,
                activation=aiter.ActivationType.Silu,
                quant_type=aiter.QuantType.per_1x128,
                w1_scale=w[1],
                w2_scale=w[3],
                dtype=dtypes.bf16,
                swiglu_limit=swiglu_limit or None,
            )
    finally:
        bspp.run_moe_bs_pp_impl = real
    assert calls == ["default"], f"fused_moe did not dispatch to flydsl_bspp: {calls}"
    rows = torch.linspace(0, tokens - 1, min(tokens, 256)).long()
    ref = run_torch(x[rows], w, tw[rows], ids[rows], swiglu_limit)
    qref = run_torch(x[rows], w, tw[rows], ids[rows], swiglu_limit, quantized=True)
    _, residual = _check("fused_moe -> flydsl_bspp", out[rows], ref, qref)
    return {"gfx": get_gfx_runtime(), "dispatched": calls[0], "residual": residual}


def summarize(name, rows):
    aiter.logger.info(
        "%s (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "moe_blockscale_pp unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    # 5185: just past the half-skip bound (T * topk <= 144 * E) at top-k 8;
    # 65536: two token chunks.
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 32, 128, 1024, 4096, 5185, 8192, 16384, 65536],
    )
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[8, 9])
    parser.add_argument(
        "-l", "--swiglu_limit", type=float, nargs="*", default=[0.0, 1.0, 10.0]
    )
    parser.add_argument(
        "-r", "--routing", nargs="*", default=["random", "invalid", "hot"]
    )
    # Routed experts of the graph-replay test (288 is GLM-5.3-Flash).
    parser.add_argument("-e", "--experts", type=int, nargs="*", default=[288, 16, 1024])
    args = parser.parse_args()

    df = []
    for topk, tokens, limit, routing in itertools.product(
        args.topk, args.tokens, args.swiglu_limit, args.routing
    ):
        if routing != "random" and limit != 0:
            continue  # edge routings once, unclamped
        df.append(test_moe_blockscale(tokens, topk, limit, routing))
    summarize("moe_blockscale summary", df)
    summarize(
        "moe_blockscale fused_moe dispatch",
        [
            test_moe_blockscale_dispatch(4096, k, lim)
            for k in args.topk
            for lim in (0.0, 1.0)
        ],
    )
    summarize(
        "moe_blockscale inputs", [test_moe_blockscale_inputs(k) for k in args.topk]
    )
    summarize(
        "moe_blockscale token chunks",
        [test_moe_blockscale_chunked(4097, k, 3) for k in args.topk],
    )
    summarize(
        "moe_blockscale graph replay",
        [
            test_moe_blockscale_graph(t, k, e)
            for e, k, t in itertools.product(args.experts, args.topk, (65, 4096))
        ],
    )


if __name__ == "__main__":
    main()
