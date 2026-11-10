# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Time aiter.fused_moe at the GLM-5.x TP8 decode MoE shape, graph and eager.

Run once per table to compare arms; --rows-expected asserts which arm is loaded.
M = 4 and up are controls: both arms run the same kernel there, so a pass whose
controls move is contaminated and should be discarded rather than averaged.
"""

import argparse
import os

import torch
from einops import rearrange

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.quant import pertoken_quant
from aiter.ops.shuffle import shuffle_weight

BLOCK = 128


def block_quant_weight(w, quant_dtype):
    """Per-(128, 128) block quant of [E, N, K]; scale is flat [E, nbn * nbk]."""
    e, n, k = w.shape
    assert n % BLOCK == 0 and k % BLOCK == 0, f"{n=} {k=} must be multiples of {BLOCK}"
    blocked = rearrange(
        w.view(e, n // BLOCK, BLOCK, k // BLOCK, BLOCK),
        "e nbn bn nbk bk -> e nbn nbk (bn bk)",
    ).contiguous()
    w_q, w_scale = pertoken_quant(blocked, quant_dtype=quant_dtype)
    w_q = rearrange(
        w_q.view(e, n // BLOCK, k // BLOCK, BLOCK, BLOCK),
        "e nbn nbk bn bk -> e (nbn bn) (nbk bk)",
    ).contiguous()
    return w_q, w_scale.view(e, -1)


def torch_moe_fp32_ref(x, w1_q, w1_scale, w2_q, w2_scale, topk_w, topk_i):
    """Dequantize weights to fp32 and run the MoE in torch. Accuracy check only."""
    f32 = torch.float32
    e, n1, k1 = w1_q.shape
    _, n2, k2 = w2_q.shape
    inter_dim = w2_q.shape[2]

    w1 = (
        w1_q.to(f32).view(e, n1 // BLOCK, BLOCK, k1 // BLOCK, BLOCK)
        * w1_scale.to(f32).view(e, n1 // BLOCK, 1, k1 // BLOCK, 1)
    ).view(e, n1, k1)
    w2 = (
        w2_q.to(f32).view(e, n2 // BLOCK, BLOCK, k2 // BLOCK, BLOCK)
        * w2_scale.to(f32).view(e, n2 // BLOCK, 1, k2 // BLOCK, 1)
    ).view(e, n2, k2)
    x = x.to(f32)

    tokens, model_dim = x.shape
    out = torch.zeros((tokens, topk_i.shape[1], model_dim), dtype=f32, device=x.device)
    for eid in range(e):
        mask = topk_i == eid
        if not mask.any():
            continue
        rows = mask.nonzero(as_tuple=True)[0]
        act = x[rows] @ w1[eid].t()
        gate, up = act.split([inter_dim, inter_dim], dim=-1)
        out[mask] = (torch.nn.functional.silu(gate) * up) @ w2[eid].t()
    return (out * topk_w.to(f32).unsqueeze(-1)).sum(dim=1)


def build_rotation(
    tokens,
    model_dim,
    inter_dim,
    experts,
    topk,
    dtype,
    seed,
    shuffle=True,
    keep_ref=False,
):
    """One independent (weights, activations, routing) set."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    quant_dtype = dtypes.fp8

    x = torch.randn((tokens, model_dim), dtype=dtype, device="cuda", generator=gen)
    w1 = (
        torch.randn(
            (experts, inter_dim * 2, model_dim),
            dtype=dtype,
            device="cuda",
            generator=gen,
        )
        / 10
    )
    w2 = (
        torch.randn(
            (experts, model_dim, inter_dim), dtype=dtype, device="cuda", generator=gen
        )
        / 10
    )
    score = torch.randn((tokens, experts), dtype=dtype, device="cuda", generator=gen)
    topk_w, topk_i = fused_topk(x, score, topk, True)

    w1_q, w1_scale = block_quant_weight(w1, quant_dtype)
    w2_q, w2_scale = block_quant_weight(w2, quant_dtype)
    ref = {"w1_ref": w1_q, "w2_ref": w2_q} if keep_ref else {}
    if shuffle:
        # The ASM 1-stage kernels read the 16x16 preshuffled layout and tag the
        # tensor, which is also how fused_moe decides the weights are shuffled.
        w1_q = shuffle_weight(w1_q, (16, 16))
        w2_q = shuffle_weight(w2_q, (16, 16))
    del w1, w2, score

    return {
        "x": x,
        "w1_q": w1_q,
        "w1_scale": w1_scale,
        "w2_q": w2_q,
        "w2_scale": w2_scale,
        "topk_w": topk_w,
        "topk_i": topk_i,
        **ref,
    }


def call_fused_moe(r, out=None):
    return fused_moe(
        r["x"],
        r["w1_q"],
        r["w2_q"],
        r["topk_w"],
        r["topk_i"],
        # Weight scales are per-128x128 blocks; fused_moe remaps this to
        # per_1x128 internally, which is the key the tuned CSV rows carry.
        quant_type=aiter.QuantType.per_128x128,
        activation=aiter.ActivationType.Silu,
        w1_scale=r["w1_scale"],
        w2_scale=r["w2_scale"],
        output=out,
    )


def time_graph(rotations, calls, replays, warmup):
    """Capture `calls` calls into one graph, replay it. Returns us per call."""
    outs = [call_fused_moe(rotations[i % len(rotations)]) for i in range(calls)]
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            call_fused_moe(rotations[i % len(rotations)], out=outs[i])

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (replays * calls), outs[0]


def time_eager(rotations, calls, iters, warmup):
    """Same calls launched one at a time, so host launch cost is included."""
    for _ in range(warmup):
        for i in range(calls):
            call_fused_moe(rotations[i % len(rotations)])
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        for i in range(calls):
            call_fused_moe(rotations[i % len(rotations)])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (iters * calls)


def check_tuned_row(m, args, gfx, cu):
    """Assert the merged table does (or does not) claim this key."""
    import pandas as pd

    from aiter.fused_moe import get_padded_M
    from aiter.jit.core import AITER_CONFIGS

    df = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    hit = df[
        (df.get("gfx") == gfx)
        & (df["cu_num"] == cu)
        & (df["token"] == get_padded_M(m))
        & (df["model_dim"] == args.dim)
        & (df["inter_dim"] == args.idim)
        & (df["expert"] == args.expert)
        & (df["topk"] == args.topk)
    ]
    found = "present" if len(hit) else "absent"
    assert found == args.rows_expected, (
        f"M={m}: expected tuned row {args.rows_expected}, found {found} "
        f"({len(hit)} row(s)). Wrong arm, or the merged config cache is stale."
    )


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    p.add_argument("-m", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("-dim", type=int, default=6144, help="model_dim")
    p.add_argument("-idim", type=int, default=256, help="inter_dim")
    p.add_argument("-e", "--expert", type=int, default=257)
    p.add_argument("-k", "--topk", type=int, default=9)
    p.add_argument(
        "--rotations", type=int, default=12, help="weight sets cycled, for cold reads"
    )
    p.add_argument("--graph-calls", type=int, default=204)
    p.add_argument("--replays", type=int, default=5)
    p.add_argument("--eager-iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--mode", choices=["graph", "eager", "both"], default="both")
    p.add_argument("--no-accuracy", action="store_true")
    p.add_argument(
        "--no-shuffle", action="store_true", help="skip the 16x16 weight preshuffle"
    )
    p.add_argument("--tag", default="", help="label printed with every row")
    p.add_argument("--rows-expected", choices=["present", "absent"])
    p.add_argument(
        "--rows-expected-m",
        type=int,
        nargs="*",
        default=[1, 2],
        help="M values the arms differ at; only these are asserted",
    )
    args = p.parse_args()

    torch.set_default_device("cuda")
    dtype = dtypes.bf16
    gfx, cu = get_gfx(), get_cu_num()
    print(f"# gfx={gfx} cu_num={cu} torch={torch.__version__}")
    print(f"# AITER_CONFIG_FMOE={os.environ.get('AITER_CONFIG_FMOE', '<unset>')}")
    if gfx != "gfx942" or cu != 304:
        print(f"# WARNING: the compared rows are gfx942 / 304, this is {gfx} / {cu}")

    wbytes = (
        args.expert * (args.idim * 2 * args.dim + args.dim * args.idim) * args.rotations
    )
    print(f"# weights across {args.rotations} rotations: {wbytes / 2**30:.2f} GiB")
    print(f"{'tag':>8} {'M':>7} {'mode':>6} {'us/call':>10} {'rel_err':>9}")

    for m in args.m:
        rotations = [
            build_rotation(
                m,
                args.dim,
                args.idim,
                args.expert,
                args.topk,
                dtype,
                seed=1000 + i,
                shuffle=not args.no_shuffle,
                keep_ref=(i == 0 and not args.no_accuracy),
            )
            for i in range(args.rotations)
        ]

        if args.rows_expected is not None and m in args.rows_expected_m:
            check_tuned_row(m, args, gfx, cu)

        rel = float("nan")
        if not args.no_accuracy:
            r = rotations[0]
            got = call_fused_moe(r).to(torch.float32)
            ref = torch_moe_fp32_ref(
                r["x"],
                r["w1_ref"],
                r["w1_scale"],
                r["w2_ref"],
                r["w2_scale"],
                r["topk_w"],
                r["topk_i"],
            )
            rel = ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            assert torch.isfinite(got).all(), f"M={m}: non-finite output"

        if args.mode in ("graph", "both"):
            us, _ = time_graph(rotations, args.graph_calls, args.replays, args.warmup)
            print(f"{args.tag:>8} {m:>7} {'graph':>6} {us:>10.2f} {rel:>9.2e}")
        if args.mode in ("eager", "both"):
            us = time_eager(rotations, args.graph_calls, args.eager_iters, args.warmup)
            print(f"{args.tag:>8} {m:>7} {'eager':>6} {us:>10.2f} {rel:>9.2e}")

        del rotations
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
