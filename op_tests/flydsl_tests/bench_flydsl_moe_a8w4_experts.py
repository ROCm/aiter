# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph-replay benchmark: FlyDSL a8w4 routed experts vs ``aiter.fused_moe`` a8w4.

Not collected by pytest (no test_ prefix). One CUDA graph holds ``--copies``
calls on distinct activations and routes, cycling over ``--layers`` weight sets
(so the weights stream from HBM as in a model's forward pass); rounds alternate
A B B A and the median us per call is reported. Both sides read the same
preshuffled weight bytes and the same MXFP8 (or bf16) activations.

Baselines (``--baseline``):
  default  ``fused_moe`` with its own dispatch (the tuned-config rows of this
           shape, else its untuned default); a count it rejects is reported.
  best     per token count, the fastest of a candidate set of FlyDSL a8w4
           stage-1 / stage-2 kernel pairs (every tuned row of this hidden /
           intermediate size, plus extra pairs), installed as the tuned row.

Examples:

    python op_tests/flydsl_tests/bench_flydsl_moe_a8w4_experts.py --experts 48 --topk 6
    python op_tests/flydsl_tests/bench_flydsl_moe_a8w4_experts.py --experts 16 --topk 3 \
        --layers 4 --tokens 16 40 80 160 320 640 1024 --baseline best
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics

import torch

import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_a8w4_experts import (
    A8W4ExpertWeights,
    flydsl_moe_a8w4_experts,
    flydsl_moe_a8w4_experts_warmup,
)

H, I = 5120, 2304

# Extra (block_m, stage 1, stage 2) FlyDSL a8w4 pairs for --baseline best, beyond
# the tuned rows of this shape.
_EXTRA_PAIRS = (
    (16, "t16x128x256_w4_gui_fp8", "t16x128x128_atomic"),
    (16, "t16x128x256_w3_gui_fp8", "t16x256x128_atomic_bnt2"),
    (16, "t16x128x256_w4_gui_fp8", "t16x128x256_atomic"),
    (32, "t32x128x256_w4_gui_fp8", "t32x256x256_atomic_bnt2"),
    (32, "t32x128x256_w3_gui_fp8", "t32x256x256_atomic"),
    (32, "t32x128x256_w3_gui_fp8", "t32x128x256_atomic_bnt2"),
    (32, "t32x64x256_w4_gui_kw2_fp8", "t32x128x256_atomic"),
    (32, "t32x128x256_w4_gui_fp8", "t32x128x256_atomic"),
    (64, "t64x128x256_w2_gui_fp8", "t64x256x256_atomic_bnt2"),
    (64, "t64x256x256_w4_gui_fp8", "t64x128x256_atomic_bnt2"),
    (64, "t64x128x256_w3_bnt0_gui_fp8", "t64x128x256_atomic"),
    (64, "t64x128x256_w4_bnt0_gui_fp8", "t64x128x256_atomic"),
    (64, "t64x128x256_w4_gui_fp8", "t64x256x256_atomic"),
    (128, "t128x256x256_w4_gui_fp8", "t128x128x256_atomic_bnt2"),
    (128, "t128x256x256_gui_fp8", "t128x128x256_reduce"),
)


def make_weights(e, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(lo, hi, shape):
        return torch.randint(
            lo, hi, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    raw = (
        rnd(0, 256, (e, 2 * I, H // 2)),
        rnd(118, 124, (e, 2 * I, H // 32)),
        rnd(0, 256, (e, H, I // 2)),
        rnd(118, 124, (e, H, I // 32)),
    )
    return A8W4ExpertWeights.from_row_major(*raw)


def quantize_mxfp8(x):
    m, k = x.shape
    blocks = x.float().reshape(m, k // 32, 32)
    u = (blocks.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    e = (u >> 23) & 255
    e = torch.where(((u & 0x7FFFFF) != 0) & (e < 255), e + 1, e).clamp_min(1)
    q = (blocks * torch.exp2(127.0 - e.float()).unsqueeze(-1)).clamp(-448, 448)
    return q.reshape(m, k).to(torch.float8_e4m3fn), e.to(torch.uint8)


def make_routes(n, topk, e, dist, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.zeros(e, device="cuda")
    if dist == "skew":
        order = torch.randperm(e, device="cuda", generator=g)
        ranks = torch.arange(1, e + 1, device="cuda").float()
        logits[order] = -torch.log(ranks)
    u = torch.rand((n, e), device="cuda", generator=g).clamp_min(1e-20)
    ids = (logits - torch.log(-torch.log(u))).topk(topk, dim=1).indices.int()
    wt = torch.softmax(torch.randn((n, topk), device="cuda", generator=g), -1)
    return ids, wt


def fused_moe_call(w: A8W4ExpertWeights, bf16_input: bool):
    """``aiter.fused_moe`` on the same bytes (a8w4: MXFP8 activations, MXFP4 weights)."""
    e = w.num_experts
    w1 = w.w13.view(e, 2 * I, H // 2).view(dtypes.fp4x2)
    w2 = w.w2.view(e, H, I // 2).view(dtypes.fp4x2)
    w1.is_shuffled = w2.is_shuffled = True
    w1.gate_up_interleave, w2.gate_up_interleave = True, False
    s1 = w.w13_scale.view(-1, H // 32).view(dtypes.fp8_e8m0)
    s2 = w.w2_scale.view(-1, I // 32).view(dtypes.fp8_e8m0)

    def call(x, xs, wt, ids):
        return fused_moe(
            x,
            w1,
            w2,
            wt,
            ids,
            w1_scale=s1,
            w2_scale=s2,
            a1_scale=None if bf16_input else xs.view(dtypes.fp8_e8m0),
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            swiglu_limit=10.0,
            quant_dtype_a=dtypes.fp8,
            dtype=torch.bfloat16,
        )

    return call


def _row_key(n_tokens, e, topk):
    return (
        "gfx950",
        256,
        fm.get_padded_M(n_tokens),
        H,
        I,
        e,
        topk,
        str(ActivationType.Silu),
        str(torch.bfloat16),
        str(dtypes.fp8),
        str(dtypes.fp4x2),
        str(QuantType.per_1x32),
        1,
        0,
    )


def candidate_pairs():
    """(block_m, stage 1, stage 2) FlyDSL a8w4 pairs: tuned rows of this H / I plus extras."""
    pairs = {
        (bm, f"flydsl_moe1_afp8_wfp4_bf16_{k1}", f"flydsl_moe2_afp8_wfp4_bf16_{k2}")
        for bm, k1, k2 in _EXTRA_PAIRS
    }
    cfg_dir = os.path.join(os.path.dirname(fm.__file__), "configs")
    for path in glob.glob(os.path.join(cfg_dir, "**", "*fmoe*.csv"), recursive=True):
        with open(path) as f:
            for row in csv.DictReader(f):
                if (
                    row.get("model_dim") == str(H)
                    and row.get("inter_dim") == str(I)
                    and row.get("q_dtype_a") == "torch.float8_e4m3fn"
                    and row.get("q_dtype_w") == "torch.float4_e2m1fn_x2"
                    and row.get("kernelName1", "").startswith("flydsl_moe1_afp8")
                    and row.get("kernelName2", "").startswith("flydsl_moe2")
                ):
                    pairs.add(
                        (int(row["block_m"]), row["kernelName1"], row["kernelName2"])
                    )
    return sorted(pairs)


def install_row(n_tokens, e, topk, pair):
    """Make ``pair`` the tuned row ``fused_moe`` looks up for this count."""
    if fm.cfg_2stages is None:
        try:  # loads the tuned tables; the lookup itself may reject the shape
            fm.get_2stage_cfgs(
                fm.get_padded_M(n_tokens),
                H,
                I,
                e,
                topk,
                torch.bfloat16,
                dtypes.fp8,
                dtypes.fp4x2,
                QuantType.per_1x32,
                True,
                ActivationType.Silu,
                False,
                0,
                0,
                True,
            )
        except Exception as exc:  # noqa: BLE001 (the tables are loaded either way)
            print(f"# tuned-row lookup: {type(exc).__name__}: {exc}", flush=True)
    primary, _ = fm.cfg_2stages
    key = _row_key(n_tokens, e, topk)
    row = dict(primary.get(key) or next(iter(primary.values())))
    bm, k1, k2 = pair
    row.update(block_m=bm, ksplit=0, kernelName1=k1, kernelName2=k2, run_1stage=0)
    primary[key] = row
    fm.get_2stage_cfgs.cache_clear()


def graph_of(fn, inputs):
    fn(*inputs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [fn(*a) for a in inputs]
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, outs


def replay_us(graph, copies, replays):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (replays * copies)


def best_baseline(base_fn, inputs, n, e, topk, copies, replays):
    """The fastest candidate pair for this count: (graph, outs, pair, us)."""
    best = None
    for pair in candidate_pairs():
        try:
            install_row(n, e, topk, pair)
            g, outs = graph_of(base_fn, inputs)
        except Exception as exc:  # noqa: BLE001 (a pair this build cannot run)
            print(f"# skip {pair}: {type(exc).__name__}", flush=True)
            torch.cuda.synchronize()
            continue
        us = min(replay_us(g, copies, replays) for _ in range(2))
        if best is None or us < best[3]:
            best = (g, outs, pair, us)
        else:
            del g, outs
    if best is not None:
        install_row(n, e, topk, best[2])
    return best


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts", type=int, default=48)
    p.add_argument("--topk", type=int, default=6, choices=[3, 6])
    p.add_argument(
        "--tokens", type=int, nargs="+", default=[16, 32, 64, 128, 192, 384, 768]
    )
    p.add_argument("--dist", nargs="+", default=["uniform", "skew"])
    p.add_argument(
        "--baseline", nargs="+", default=["default"], choices=["default", "best"]
    )
    p.add_argument("--layers", type=int, default=1, help="weight sets cycled")
    p.add_argument("--copies", type=int, default=100, help="calls per graph")
    p.add_argument("--rounds", type=int, default=5, help="A B B A rounds")
    p.add_argument("--replays", type=int, default=5)
    p.add_argument("--bf16-input", action="store_true")
    args = p.parse_args()

    e, topk = args.experts, args.topk
    ws = [make_weights(e, seed=i) for i in range(args.layers)]
    flydsl_moe_a8w4_experts_warmup(ws[0], topk, bf16_input=args.bf16_input)
    bases = [fused_moe_call(w, args.bf16_input) for w in ws]
    rows = []
    for dist in args.dist:
        for n in args.tokens:
            inputs = []
            for i in range(args.copies):
                g = torch.Generator(device="cuda").manual_seed(1000 * n + i)
                x = torch.randn((n, H), device="cuda", generator=g).bfloat16()
                xc, xs = (x, None) if args.bf16_input else quantize_mxfp8(x)
                ids, wt = make_routes(n, topk, e, dist, seed=1000 * n + i)
                inputs.append((i % args.layers, xc, xs, wt, ids))

            def ours(li, x, xs, wt, ids):
                return flydsl_moe_a8w4_experts(x, xs, ws[li], wt, ids)

            def base(li, x, xs, wt, ids):
                return bases[li](x, xs, wt, ids)

            g_ours, o_ours = graph_of(ours, inputs)
            graphs = {"flydsl": g_ours}
            row = {"experts": e, "topk": topk, "dist": dist, "tokens": n}
            for mode in args.baseline:
                name = f"fused_moe_{mode}"
                if mode == "default":
                    fm.cfg_2stages = None
                    fm.get_2stage_cfgs.cache_clear()
                    try:
                        graphs[name], o_base = graph_of(base, inputs)
                    except Exception as exc:  # noqa: BLE001 (reported in the row)
                        row[name + "_us"] = f"raises {type(exc).__name__}"
                        torch.cuda.synchronize()
                        continue
                else:
                    best = best_baseline(
                        base, inputs, n, e, topk, args.copies, args.replays
                    )
                    if best is None:
                        row[name + "_us"] = "no candidate runs"
                        continue
                    graphs[name], o_base, pair, _ = best
                    row[name + "_pair"] = pair
                row[name + "_cos"] = round(
                    torch.nn.functional.cosine_similarity(
                        o_ours[0].double().flatten(), o_base[0].double().flatten(), 0
                    ).item(),
                    7,
                )
            names = list(graphs)
            samples = {k: [] for k in names}
            for _ in range(args.rounds):  # A B .. B A
                for k in names + names[::-1]:
                    samples[k].append(replay_us(graphs[k], args.copies, args.replays))
            for k, s in samples.items():
                row[k + "_us"] = round(statistics.median(s), 1)
                row[k + "_spread"] = round(max(s) - min(s), 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del graphs, o_ours, inputs
            torch.cuda.empty_cache()

    cols = ["flydsl"] + [f"fused_moe_{m}" for m in args.baseline]
    print("| dist | tokens | " + " | ".join(f"{c} us" for c in cols) + " |")
    print("|---" * (2 + len(cols)) + "|")
    for r in rows:
        vals = [str(r.get(c + "_us", "-")) for c in cols]
        print(f"| {r['dist']} | {r['tokens']} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
