#!/usr/bin/env python3
"""DSV4-Flash TP1 decode MoE (M=64, HERD on): unit test vs e2e conditions.

Reuses herd_triton_flydsl.py (weights, timing helpers) and adds what the e2e
decode step has and the unit test does not:

  --routing real    Top-7 candidates taken from router logits dumped out of an
                    ATOM decode step (--router-dump: ./router_dump, else
                    /data/cguo/router_dump; L*_S*.pt at bs=64), laid into the
                    unit test's clean-gap logits so FlyDSL and Triton still pick
                    bit-identical experts. "uniform" is the unit test's own.
  --shared on       ATOM's dual-stream MoE: router gate GEMM + routed experts on
                    the main stream, the FP8 shared expert (1x128 e8m0 quant ->
                    blockscale GEMM 4096x4096 -> silu_and_mul(limit=10) -> quant
                    -> blockscale GEMM 4096x2048) on a side stream, joined by an
                    add, all inside one CUDA graph.
  --layers N        N MoE layers with their own weights and their own routing,
                    run back to back per step, so the weights are as cold as in
                    a 43-layer decode step instead of re-read from cache.

Stacks:
  FlyDSL        aiter.topk_gating Top-7 -> fused_moe(herd_topk=6), swiglu_limit=10,
                routed scale 1.5 (what ATOM's patched FlyDSL path calls)
  Triton-ATOM   triton routing (HERD) -> ATOM _fused_experts_silu_gugu over
                zero-copy views of the FlyDSL weights (MXFP8 quant, GEMM1 with
                fused MX requant, GEMM2), as ATOM_USE_TRITON_MOE_DECODE runs it
  Triton-unit   the unit test's static-FP8 Triton path, for reference

All times are per MoE layer (graph or kernel time divided by --layers).
Per-kernel times from a multi-stream graph are only trustworthy with
DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0 DEBUG_HIP_GRAPH_CLASSIC_PATH=1; use
--no-profile for the step time under the default scheduler.
"""

import argparse
import collections
import glob
import os
import re
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herd_triton_flydsl as base  # noqa: E402

import aiter  # noqa: E402
import aiter.ops.topk as flydsl_topk_mod  # noqa: E402
import aiter.ops.triton.moe.moe_routing.routing as triton_routing_mod  # noqa: E402
from aiter import ActivationType, QuantType, dtypes  # noqa: E402
from aiter import gemm_a8w8_blockscale_bpreshuffle, silu_and_mul  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.quant import per_group_quant_hip  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.ops.triton.moe.moe_routing.routing import routing as triton_routing  # noqa: E402
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view  # noqa: E402
from aiter.tuned_gemm import tgemm  # noqa: E402

from atom.model_ops.fused_moe_triton import _fused_experts_silu_gugu  # noqa: E402

CFG = base.MODELS["dsv4flash"]
TOKENS = 64
SWIGLU_LIMIT = 10.0
ROUTED_SCALE = 1.5
HERE = os.path.dirname(os.path.abspath(__file__))
STACKS = ("FlyDSL", "Triton-ATOM", "Triton-unit")
FIRST_STEP = 1
ROUTER_DUMP = next(
    (d for d in (os.path.join(HERE, "router_dump"), "/data/cguo/router_dump") if os.path.isdir(d)),
    os.path.join(HERE, "router_dump"),
)

KERNEL_GROUPS = (
    ("gemm1", r"a8w4_tdm.*K4096|_moe_gemm_a8w4.*SWIGLU_1"),
    ("gemm2", r"a8w4_tdm.*K2048|_moe_gemm_a8w4.*SWIGLU_0"),
    ("route", r"topk_gating|^_topk_|_keepk|moe_route|psum|_combined_routing|"
              r"_sum_bitmatrix|FillFunctor<int>"),
    ("moe_quant", r"multidest_quant|_downcast_to_mxfp|_static_per_tensor_quant"),
    ("reduce", r"moe_gather_reduce|reduce_grouped"),
    ("shared", r"gemm_a8w8_mx128|act_and_mul|dynamic_per_group_scaled_quant"),
)


def group_of(name):
    for g, pat in KERNEL_GROUPS:
        if re.search(pat, name):
            return g
    return "other"


# ---------------------------------------------------------------- routing ---

def real_candidates(path):
    d = torch.load(path)
    scores = F.softplus(d["logits"].float().cuda()).sqrt()
    return (scores + d["bias"].float().cuda()).topk(CFG.topk + 1, dim=1).indices


def router_inputs(candidate_ids, seed):
    """The unit test's clean-gap logits over the given Top-(K+1) candidates,
    levels assigned in the candidates' own rank order."""
    generator = torch.Generator(device="cuda").manual_seed(1000 + seed)
    logits = torch.full((TOKENS, CFG.experts), -10.0, dtype=torch.bfloat16)
    levels = torch.tensor(CFG.levels, dtype=torch.bfloat16).expand(TOKENS, CFG.topk + 1)
    logits.scatter_(1, candidate_ids, levels)
    bias = torch.randn(CFG.experts, generator=generator, dtype=torch.float32) * 0.025
    return logits, bias


def layer_routing(routing, round_idx, layer_idx, n_layers):
    seed = round_idx * 1000 + layer_idx
    if routing == "uniform":
        return base.make_router_inputs(CFG, TOKENS, seed)
    files = sorted(glob.glob(os.path.join(ROUTER_DUMP, "L*_S*.pt")))
    if not files:
        raise SystemExit(f"no L*_S*.pt under {ROUTER_DUMP}; pass --router-dump or run dump_router.sh")
    layers = sorted({os.path.basename(f)[:3] for f in files})
    steps = sorted({os.path.basename(f)[4:-3] for f in files})
    lname = layers[layer_idx * len(layers) // n_layers]
    # S00 is the first decode step, whose routing is far more concentrated than
    # the steady state (about half the unique experts); start past it.
    sname = steps[(FIRST_STEP + round_idx) % len(steps)]
    return router_inputs(real_candidates(os.path.join(ROUTER_DUMP, f"{lname}_{sname}.pt")), seed)


# ----------------------------------------------------------------- stacks ---

def build_flydsl(hidden, logits, bias, weights):
    w1, w2, s1, s2 = weights
    topk_weights = torch.empty(TOKENS, CFG.topk + 1, dtype=torch.float32)
    topk_ids = torch.empty(TOKENS, CFG.topk + 1, dtype=torch.int32)

    def route():
        flydsl_topk_mod._FLYDSL_USE_HERD = True
        aiter.topk_gating(topk_weights, topk_ids, logits, bias, False, 1.0, "sqrtsoftplus")

    def step():
        route()
        return fused_moe(
            hidden, w1, w2, topk_weights, topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=s1, w2_scale=s2,
            gate_mode=GateMode.INTERLEAVE.value,
            dtype=dtypes.bf16,
            quant_dtype_a=dtypes.fp8,
            swiglu_limit=SWIGLU_LIMIT,
            herd_topk=CFG.topk,
            herd_routed_scaling_factor=ROUTED_SCALE,
        )

    route()
    return step, base.herd_kept_ids(CFG, topk_ids, topk_weights)


def build_triton_atom(hidden, logits, bias, flydsl_weights):
    w1, w2, s1, s2 = flydsl_weights
    w1v, w2v = moe_weight_decode_view(w1), moe_weight_decode_view(w2)
    s1v, s2v = s1.transpose(-1, -2), s2.transpose(-1, -2)

    def route():
        triton_routing_mod._USE_HERD = True
        return triton_routing(
            logits, CFG.topk, score_mode="sqrtsoftplus", bias=bias,
            renorm=True, routed_scaling_factor=ROUTED_SCALE,
        )

    def step():
        rd, gather, scatter = route()
        return _fused_experts_silu_gugu(
            hidden, w1v, w2v, rd, gather, scatter,
            w13_scale=s1v, w2_scale=s2v,
            w13_swizzle_layout="GFX1250_SCALE", w2_swizzle_layout="GFX1250_SCALE",
            swiglu_limit=SWIGLU_LIMIT,
        )

    rd, _, _ = route()
    return step, torch.sort(rd.topk_ids.to(torch.int32), dim=1).values


def build_triton_unit(hidden, logits, bias, triton_weights):
    _, step, ids = base.build_triton_dsv4(CFG, hidden, logits, bias, triton_weights, True)
    return step, ids


BUILD = {
    "FlyDSL": lambda h, lg, b, fw, tw: build_flydsl(h, lg, b, fw),
    "Triton-ATOM": lambda h, lg, b, fw, tw: build_triton_atom(h, lg, b, fw),
    "Triton-unit": lambda h, lg, b, fw, tw: build_triton_unit(h, lg, b, tw),
}


# ---------------------------------------------------------- shared expert ---

class SharedExpert:
    """DSV4 shared expert as ATOM runs it on gfx1250 (FP8 1x128 e8m0 blockscale,
    preshuffled weights), dim 4096 -> 2x2048 -> 4096."""

    def __init__(self, seed):
        g = torch.Generator(device="cuda").manual_seed(7 + seed)

        def linear(n, k):
            w = (torch.randn(n, k, generator=g) * 0.05).to(dtypes.fp8)
            s = torch.full((n // 128, k // 128), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
            return shuffle_weight(w, layout=(16, 16)), s

        self.gate_up = linear(2 * CFG.inter_dim, CFG.model_dim)
        self.down = linear(CFG.model_dim, CFG.inter_dim)

    @staticmethod
    def _quant(x):
        return per_group_quant_hip(
            x, quant_dtype=dtypes.fp8, group_size=128,
            transpose_scale=True, scale_type=dtypes.fp8_e8m0,
        )

    def __call__(self, x):
        xq, xs = self._quant(x)
        combined = gemm_a8w8_blockscale_bpreshuffle(xq, self.gate_up[0], xs, self.gate_up[1], dtypes.bf16)
        act = torch.empty(x.shape[0], CFG.inter_dim, dtype=torch.bfloat16)
        silu_and_mul(act, combined, SWIGLU_LIMIT)
        aq, as_ = self._quant(act)
        return gemm_a8w8_blockscale_bpreshuffle(aq, self.down[0], as_, self.down[1], dtypes.bf16)


def moe_layer(routed_step, hidden, gate_w, shared, alt):
    """One DSV4 MoE layer; with `shared`, ATOM's dual_stream_moe_forward."""
    if shared is None:
        def step():
            tgemm.mm(hidden, gate_w, None, otype=dtypes.bf16)
            return routed_step()
        return step

    def step():
        cur = torch.cuda.current_stream()
        alt.wait_stream(cur)
        tgemm.mm(hidden, gate_w, None, otype=dtypes.bf16)
        routed = routed_step()
        with torch.cuda.stream(alt):
            sh = shared(hidden)
        cur.wait_stream(alt)
        return routed + sh

    return step


# ------------------------------------------------------------ measurement ---

def main():
    global FIRST_STEP, ROUTER_DUMP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routing", nargs="+", default=["uniform", "real"], choices=["uniform", "real"])
    ap.add_argument("--shared", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument("--stacks", nargs="+", default=list(STACKS), choices=list(STACKS))
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--first-step", type=int, default=1,
                    help="index of the first router_dump step file to use (0 = first decode step)")
    ap.add_argument("--router-dump", default=ROUTER_DUMP,
                    help="directory of L*_S*.pt router logits from dump_router.sh")
    args = ap.parse_args()
    FIRST_STEP = args.first_step
    ROUTER_DUMP = args.router_dump

    flydsl_topk_mod._FLYDSL_HERD_MIN_M = 16
    flydsl_topk_mod._FLYDSL_HERD_MAX_M = 128
    triton_routing_mod._HERD_MIN_M = 16
    triton_routing_mod._HERD_MAX_M = 128

    def layer_weights():
        flydsl_w, triton_w, _ = base.make_dsv4_weights(CFG)
        # triton_w views the plain weights; dropping it frees them per layer.
        return flydsl_w, (triton_w if "Triton-unit" in args.stacks else None)

    weights = [layer_weights() for _ in range(args.layers)]
    shared = [SharedExpert(i) for i in range(args.layers)]
    gate_w = [torch.randn(CFG.experts, CFG.model_dim, dtype=torch.bfloat16) * 0.02 for _ in range(args.layers)]
    hidden = torch.randn(TOKENS, CFG.model_dim, generator=torch.Generator(device="cuda").manual_seed(TOKENS),
                         dtype=torch.bfloat16)
    alt = torch.cuda.Stream()

    results = collections.defaultdict(lambda: collections.defaultdict(list))
    for routing in args.routing:
        for rnd in range(args.rounds):
            per_stack = {s: [] for s in args.stacks}
            uniques = []
            for li in range(args.layers):
                logits, bias = layer_routing(routing, rnd, li, args.layers)
                built = {s: BUILD[s](hidden, logits, bias, *weights[li]) for s in args.stacks}
                ref = next(iter(built.values()))[1]
                for s, (_, ids) in built.items():
                    if not torch.equal(ids, ref):
                        raise AssertionError(f"{routing} round {rnd} layer {li}: {s} expert ids differ")
                uniques.append(int(ref.unique().numel()))
                for s, (routed_step, _) in built.items():
                    per_stack[s].append(routed_step)
            for shared_mode in args.shared:
                for s in args.stacks:
                    layers = [
                        moe_layer(rs, hidden, gate_w[li], shared[li] if shared_mode == "on" else None, alt)
                        for li, rs in enumerate(per_stack[s])
                    ]

                    def step(layers=layers):
                        out = None
                        for layer in layers:
                            out = layer()
                        return out

                    r = results[(routing, shared_mode, s)]
                    r["unique"].append(statistics.median(uniques))
                    r["e2e"].append(base.graph_us(step) / args.layers)
                    if not args.no_profile:
                        graph = base.capture(step)
                        groups = collections.defaultdict(float)
                        for name, us in base.kernel_us(graph.replay, iters=10).items():
                            groups[group_of(name)] += us / args.layers
                        for g, _ in KERNEL_GROUPS:
                            r[g].append(groups.get(g, 0.0))
                        del graph
            print(f"{routing} round {rnd}: parity PASS, median unique experts {statistics.median(uniques):g}",
                  flush=True)

    cols = ["unique", "e2e"] + ([] if args.no_profile else [g for g, _ in KERNEL_GROUPS])
    print(f"\nper MoE layer, median over {args.rounds} rounds of {args.layers} layers, us"
          " (e2e = CUDA-graph step time / layers; the rest = profiled GPU kernel time)\n")
    print("| routing | shared expert | stack | " + " | ".join(cols) + " |")
    print("|" + "---|" * (3 + len(cols)))
    for key, r in results.items():
        cells = " | ".join(f"{statistics.median(r[c]):.2f}" if c != "unique" else f"{statistics.median(r[c]):g}"
                           for c in cols)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {cells} |")

    print("\nFlyDSL minus Triton-ATOM (positive = FlyDSL slower), us per layer:\n")
    print("| routing | shared expert | " + " | ".join(cols[1:]) + " |")
    print("|" + "---|" * (1 + len(cols)))
    for routing in args.routing:
        for shared_mode in args.shared:
            f = results.get((routing, shared_mode, "FlyDSL"))
            t = results.get((routing, shared_mode, "Triton-ATOM"))
            if not f or not t:
                continue
            cells = " | ".join(f"{statistics.median(f[c]) - statistics.median(t[c]):+.2f}" for c in cols[1:])
            print(f"| {routing} | {shared_mode} | {cells} |")


if __name__ == "__main__":
    main()
