#!/usr/bin/env python3
"""Triton vs FlyDSL MoE on gfx1250, HERD off/on: Selector / Sort / MoE / E2E.

    python herd_triton_flydsl_bench.py                          # DSV4-Flash TP1 a8w4, M=64
    python herd_triton_flydsl_bench.py --model gptoss --M 64    # GPT-OSS TP1 a4w4
    python herd_triton_flydsl_bench.py --model gptoss --M 16 32 64 128 --kernels

The four cases of one seed share the same hidden states, router logits, MXFP4
weights and E8M0 scales. Router logits are constructed with a clear score gap
around top-(K+1), then the script asserts that Triton and FlyDSL select exactly
the same final expert IDs before timing either stack.

DSV4-Flash: aiter.topk_gating (sqrtsoftplus + fp32 bias, K+1 candidates with
HERD) -> fused_moe a8w4 dropping in the route, against Triton
routing(score_mode="sqrtsoftplus") -> static-FP8 moe_gemm_a8w4.

GPT-OSS: aiter.topk_softmax -> fused_moe a4w4 SwiGLU, against Triton flat-topk
routing -> moe_gemm_a4w4 over the same preshuffled GUGU weights, as ATOM's
gfx1250 path hands them to the gluon kernel. The FlyDSL tile must come from
model_configs/gptoss_gfx1250_tuned_grouped_fmoe.csv, or the script exits.

Selector is the kernels that pick expert IDs (FlyDSL router; Triton _topk and
the HERD drop _keepk_sort0). Sort is the rest of the routing (FlyDSL route /
psum / remap; Triton histogram, _combined_routing, memsets). MoE is everything
else, activation quant included. E2E is the CUDA-graph replay of one step.
"""

import argparse
import collections
import csv
import os
import statistics
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

import aiter
import aiter.ops.topk as flydsl_topk_mod
import aiter.ops.triton.moe.moe_routing.routing as triton_routing_mod
from aiter import ActivationType, QuantType
from aiter import benchmark_data_init as bench_init
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import grouped_moe_gfx1250 as flydsl_grouped
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import (
    interleave_gate_up_rows,
    moe_shuffle_scale,
    moe_shuffle_weight,
)
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4, mxfp4_quant
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_routing.routing import routing as triton_routing
from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view, shuffle_scale_moe
from aiter.utility import dtypes

torch.set_default_device("cuda")

MODELS = {
    "dsv4flash": SimpleNamespace(
        name="DSV4-Flash TP1",
        quant="a8w4",
        experts=256,
        topk=6,
        model_dim=4096,
        inter_dim=2048,
        levels=(6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5),
    ),
    "gptoss": SimpleNamespace(
        name="GPT-OSS TP1",
        quant="a4w4",
        experts=128,
        topk=4,
        model_dim=3072,
        inter_dim=3072,
        levels=(5.0, 4.0, 3.0, 2.0, 1.0),
    ),
}
GPTOSS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(aiter.__file__)),
    "configs/model_configs/gptoss_gfx1250_tuned_grouped_fmoe.csv",
)
CSV_INT_COLUMNS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "tile_m2",
    "tile_n2",
    "tile_k2",
    "m_warp",
    "n_warp",
    "m_warp2",
    "n_warp2",
    "num_buffers",
    "num_buffer_stage2",
)
GPTOSS_SWIGLU_ALPHA = 1.702
GPTOSS_SWIGLU_LIMIT = 7.0
ROUTED_SCALE = 1.0
CALLS_PER_GRAPH = 10
GRAPH_REPLAYS = 30
FLYDSL_SORT_KERNELS = (
    "moe_route",
    "moe_contiguous_psum_remap",
    "FillFunctor",
)
TRITON_SELECTOR_KERNELS = ("_topk", "_keepk_sort0")
HERD_KERNEL_MARKERS = ("herd", "_keepk_sort0")
STAGES = ("selector", "sort", "moe")


def make_router_inputs(cfg, tokens, seed):
    generator = torch.Generator(device="cuda").manual_seed(1000 + seed)
    candidate_ids = (
        torch.rand(
            tokens,
            cfg.experts,
            generator=generator,
        )
        .topk(cfg.topk + 1, dim=1)
        .indices
    )
    logits = torch.full((tokens, cfg.experts), -10.0, dtype=torch.bfloat16)
    levels = torch.tensor(cfg.levels, dtype=torch.bfloat16).expand(tokens, cfg.topk + 1)
    logits.scatter_(1, candidate_ids, levels)
    bias = torch.randn(cfg.experts, generator=generator, dtype=torch.float32) * 0.025
    return logits, bias


def make_plain_weights(cfg):
    generator = bench_init.make_generator(0)

    def fp4(rows, logical_k):
        return bench_init.fill_fp4(
            (cfg.experts * rows, logical_k),
            "uniform",
            generator,
        ).view(cfg.experts, rows, logical_k // 2)

    w1_plain = fp4(2 * cfg.inter_dim, cfg.model_dim)
    w2_plain = fp4(cfg.model_dim, cfg.inter_dim)
    s1_plain = bench_init.fill_scale_e8m0(
        (cfg.experts, 2 * cfg.inter_dim, cfg.model_dim // 32),
        "auto",
        generator,
    ).contiguous()
    s2_plain = bench_init.fill_scale_e8m0(
        (cfg.experts, cfg.model_dim, cfg.inter_dim // 32),
        "auto",
        generator,
    ).contiguous()
    return w1_plain, w2_plain, s1_plain, s2_plain


def make_flydsl_scales(cfg, s1_plain, s2_plain):
    return (
        moe_shuffle_scale(
            s1_plain,
            experts_cnt=cfg.experts,
            is_guinterleave=True,
            gate_up=True,
        ),
        moe_shuffle_scale(s2_plain, experts_cnt=cfg.experts),
    )


def make_dsv4_weights(cfg):
    w1_plain, w2_plain, s1_plain, s2_plain = make_plain_weights(cfg)
    flydsl_weights = (
        moe_shuffle_weight(
            w1_plain,
            experts_cnt=cfg.experts,
            is_guinterleave=True,
            gate_up=True,
        ),
        moe_shuffle_weight(w2_plain, experts_cnt=cfg.experts),
        *make_flydsl_scales(cfg, s1_plain, s2_plain),
    )
    triton_weights = (
        w1_plain.transpose(-1, -2),
        w2_plain.transpose(-1, -2),
        moe_shuffle_scale(s1_plain).transpose(-1, -2),
        moe_shuffle_scale(s2_plain).transpose(-1, -2),
    )
    return flydsl_weights, triton_weights, None


def make_gptoss_weights(cfg):
    w1_plain, w2_plain, s1_plain, s2_plain = make_plain_weights(cfg)
    w1 = moe_shuffle_weight(
        w1_plain,
        experts_cnt=cfg.experts,
        is_guinterleave=True,
        gate_up=True,
    )
    w2 = moe_shuffle_weight(w2_plain, experts_cnt=cfg.experts)
    gate_up_bias = torch.randn(cfg.experts, 2 * cfg.inter_dim) * 1e-3
    b1 = interleave_gate_up_rows(gate_up_bias).contiguous()
    b2 = torch.randn(cfg.experts, cfg.model_dim) * 1e-3
    flydsl_weights = (
        w1.view(dtypes.fp4x2),
        w2.view(dtypes.fp4x2),
        *make_flydsl_scales(cfg, s1_plain, s2_plain),
        b1,
        b2,
    )
    s1_u8 = s1_plain.view(torch.uint8)
    s2_u8 = s2_plain.view(torch.uint8)
    triton_weights = (
        moe_weight_decode_view(w1),
        moe_weight_decode_view(w2),
        shuffle_scale_moe(
            interleave_gate_up_rows(s1_u8).transpose(-2, -1),
            scale_kwidth=4,
        ),
        shuffle_scale_moe(s2_u8.transpose(-2, -1), scale_kwidth=4),
        b1,
        b2,
    )
    raw_weights = (w1_plain, s1_u8, gate_up_bias, w2_plain, s2_u8, b2)
    return flydsl_weights, triton_weights, raw_weights


def herd_kept_ids(cfg, candidate_ids, candidate_scores):
    tokens = candidate_ids.shape[0]
    popularity = torch.bincount(
        candidate_ids.reshape(-1).long(),
        minlength=cfg.experts,
    )
    order = torch.argsort(candidate_ids, dim=1)
    ids = candidate_ids.gather(1, order)
    scores = candidate_scores.gather(1, order)
    route_popularity = popularity[ids.long()]
    least = route_popularity == route_popularity.min(1, keepdim=True).values
    least_scores = torch.where(least, scores, torch.inf)
    least &= least_scores == least_scores.min(1, keepdim=True).values
    columns = torch.arange(cfg.topk + 1).expand(tokens, cfg.topk + 1)
    drop = torch.where(least, columns, cfg.topk + 1).min(1).values
    keep = columns != drop[:, None]
    return ids[keep].reshape(tokens, cfg.topk)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def graph_us(fn):
    graph = capture(lambda: [fn() for _ in range(CALLS_PER_GRAPH)])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(GRAPH_REPLAYS):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / CALLS_PER_GRAPH)
    return statistics.median(samples)


def kernel_us(run, iters=20):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
    total = collections.defaultdict(float)
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            total[event.name] += event.device_time / iters
    return total


def kernel_stage(stack, name, route_names):
    if stack == "FlyDSL":
        if name in route_names:
            return "selector"
        if any(pattern in name for pattern in FLYDSL_SORT_KERNELS):
            return "sort"
        return "moe"
    if name in route_names:
        if any(pattern in name for pattern in TRITON_SELECTOR_KERNELS):
            return "selector"
        return "sort"
    return "moe"


def stage_kernel_us(stack, route, step):
    route_names = set(kernel_us(route, iters=3))
    graph = capture(step)
    samples = collections.defaultdict(list)
    per_kernel = collections.defaultdict(list)
    for _ in range(3):
        kernels = kernel_us(graph.replay)
        split = dict.fromkeys(STAGES, 0.0)
        for name, value in kernels.items():
            split[kernel_stage(stack, name, route_names)] += value
            per_kernel[name].append(value)
        for stage in STAGES:
            samples[stage].append(split[stage])
    kernels = {
        name: (kernel_stage(stack, name, route_names), statistics.median(values))
        for name, values in per_kernel.items()
    }
    return {stage: statistics.median(samples[stage]) for stage in STAGES}, kernels


def build_flydsl_dsv4(cfg, hidden, logits, bias, weights, herd):
    tokens = hidden.shape[0]
    width = cfg.topk + 1 if herd else cfg.topk
    topk_weights = torch.empty(tokens, width, dtype=torch.float32)
    topk_ids = torch.empty(tokens, width, dtype=torch.int32)
    w1, w2, s1, s2 = weights

    def route():
        flydsl_topk_mod._FLYDSL_USE_HERD = herd
        aiter.topk_gating(
            topk_weights,
            topk_ids,
            logits,
            bias,
            not herd,
            ROUTED_SCALE if not herd else 1.0,
            "sqrtsoftplus",
        )
        return topk_weights, topk_ids

    def moe():
        return fused_moe(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=s1,
            w2_scale=s2,
            gate_mode=GateMode.INTERLEAVE.value,
            dtype=dtypes.bf16,
            quant_dtype_a=dtypes.fp8,
            herd_topk=cfg.topk if herd else 0,
            herd_routed_scaling_factor=ROUTED_SCALE,
        )

    def step():
        route()
        return moe()

    route()
    final_ids = (
        herd_kept_ids(cfg, topk_ids, topk_weights)
        if herd
        else torch.sort(topk_ids, dim=1).values
    )
    return route, step, final_ids


def build_triton_dsv4(cfg, hidden, logits, bias, weights, herd):
    w1, w2, s1, s2 = weights
    x1_scale = hidden.abs().max().float() / 448.0

    def route():
        triton_routing_mod._USE_HERD = herd
        return triton_routing(
            logits,
            cfg.topk,
            score_mode="sqrtsoftplus",
            bias=bias,
            renorm=True,
            routed_scaling_factor=ROUTED_SCALE,
        )

    routing_data, gather, _ = route()
    x1 = downcast_to_static_fp8(hidden, x1_scale)
    y1 = moe_gemm_a8w4(
        x1,
        w1,
        None,
        s1,
        x1_scale,
        None,
        None,
        routing_data,
        gather_indx=gather,
        swizzle_mx_scale="GFX1250_SCALE",
        out_dtype=torch.bfloat16,
        apply_swiglu=True,
        alpha=1.0,
        limit=None,
        swiglu_add_residual=False,
    )
    x2_scale = y1.abs().max().float() / 448.0
    fp8_dtype = torch.float8_e4m3fn

    def step():
        rd, ga, sc = route()
        x = downcast_to_static_fp8(hidden, x1_scale)
        y = moe_gemm_a8w4(
            x,
            w1,
            None,
            s1,
            x1_scale,
            x2_scale,
            None,
            rd,
            gather_indx=ga,
            swizzle_mx_scale="GFX1250_SCALE",
            out_dtype=fp8_dtype,
            apply_swiglu=True,
            alpha=1.0,
            limit=None,
            swiglu_add_residual=False,
        )
        return moe_gemm_a8w4(
            y,
            w2,
            None,
            s2,
            x2_scale,
            None,
            None,
            rd,
            scatter_indx=sc,
            gammas=rd.gate_scal,
            swizzle_mx_scale="GFX1250_SCALE",
        )

    final_ids = torch.sort(routing_data.topk_ids.to(torch.int32), dim=1).values
    return route, step, final_ids


def build_flydsl_gptoss(cfg, hidden, logits, bias, weights, herd):
    tokens = hidden.shape[0]
    topk_weights = torch.empty(tokens, cfg.topk, dtype=torch.float32)
    topk_ids = torch.empty(tokens, cfg.topk, dtype=torch.int32)
    token_expert_ids = torch.empty(tokens, cfg.topk, dtype=torch.int32)
    w1, w2, s1, s2, b1, b2 = weights

    def route():
        flydsl_topk_mod._FLYDSL_USE_HERD = herd
        aiter.topk_softmax(topk_weights, topk_ids, token_expert_ids, logits, True)
        return topk_weights, topk_ids

    def step():
        route()
        return fused_moe(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=s1,
            w2_scale=s2,
            bias1=b1,
            bias2=b2,
            gate_mode=GateMode.INTERLEAVE.value,
            dtype=dtypes.bf16,
            swiglu_limit=GPTOSS_SWIGLU_LIMIT,
            quant_dtype_a=dtypes.fp4x2,
        )

    route()
    return route, step, torch.sort(topk_ids, dim=1).values


def build_triton_gptoss(cfg, hidden, logits, bias, weights, herd):
    w1, w2, s1, s2, b1, b2 = weights

    def route():
        triton_routing_mod._USE_HERD = herd
        return triton_routing(logits, cfg.topk, sm_first=False)

    def step():
        rd, ga, sc = route()
        x, x_scale = mxfp4_quant(hidden)
        y, y_scale = moe_gemm_a4w4(
            x,
            w1,
            x_scale,
            s1,
            bias=b1,
            routing_data=rd,
            gather_indx=ga,
            swizzle_mx_scale="GFX1250_SCALE",
            preshuffle_weights=True,
            apply_swiglu=True,
            alpha=GPTOSS_SWIGLU_ALPHA,
            limit=GPTOSS_SWIGLU_LIMIT,
            swiglu_add_residual=True,
            out_mx_quant=True,
        )
        return moe_gemm_a4w4(
            y,
            w2,
            y_scale,
            s2,
            bias=b2,
            routing_data=rd,
            scatter_indx=sc,
            gammas=rd.gate_scal,
            swizzle_mx_scale="GFX1250_SCALE",
            preshuffle_weights=True,
        )

    routing_data, _, _ = route()
    final_ids = torch.sort(routing_data.topk_ids.to(torch.int32), dim=1).values
    return route, step, final_ids


BUILDERS = {
    "dsv4flash": (make_dsv4_weights, build_flydsl_dsv4, build_triton_dsv4),
    "gptoss": (make_gptoss_weights, build_flydsl_gptoss, build_triton_gptoss),
}


def gptoss_csv_row(cfg, tokens):
    overrides = sorted(k for k in os.environ if k.startswith("AITER_TDM_"))
    if overrides:
        raise SystemExit(f"AITER_TDM_* overrides beat the tuned CSV: {overrides}")
    padded = flydsl_grouped._get_padded_m(tokens)
    with open(GPTOSS_CSV, newline="") as f:
        expected = [row for row in csv.DictReader(f) if row["token"] == str(padded)]
    if len(expected) != 1:
        raise SystemExit(f"{GPTOSS_CSV} has {len(expected)} rows for token={padded}")
    row = flydsl_grouped._find_grouped_config(
        token_num=padded,
        model_dim=cfg.model_dim,
        inter_dim=cfg.inter_dim,
        experts=cfg.experts,
        topk=cfg.topk,
        activation=ActivationType.Swiglu,
        dtype=dtypes.bf16,
        q_dtype_a=dtypes.fp4x2,
        q_dtype_w=dtypes.fp4x2,
        quant_type=QuantType.per_1x32,
        ep_fused=False,
    )
    expected = expected[0]
    same = row is not None and all(
        flydsl_grouped._as_int(row.get(k), None)
        == flydsl_grouped._as_int(expected.get(k), None)
        for k in CSV_INT_COLUMNS
    )
    same = same and all(
        row.get(k) == expected.get(k) for k in ("kernelName1", "kernelName2")
    )
    if not same:
        raise SystemExit(
            f"FlyDSL grouped config for token={padded} is not the "
            f"{os.path.basename(GPTOSS_CSV)} row: got {row}"
        )
    return (
        f"token={padded} stage1 tile {expected['tile_m']}x{expected['tile_n']}x"
        f"{expected['tile_k']} warps {expected['m_warp']}x{expected['n_warp']}, "
        f"stage2 tile {expected['tile_m2']}x{expected['tile_n2']}x"
        f"{expected['tile_k2']} warps {expected['m_warp2']}x{expected['n_warp2']}, "
        f"{expected['kernelName1']} / {expected['kernelName2']}"
    )


def logits_diff(out, ref):
    out, ref = out.float(), ref.float()
    return float((out - ref).square().sum() / (out.square().sum() + ref.square().sum()))


def mxfp4_dequant(packed, scale):
    from op_tests.triton_tests.utils.mxfp_ref import upcast_from_mxfp

    return upcast_from_mxfp(
        packed.view(torch.uint8),
        scale.view(torch.uint8),
        torch.bfloat16,
        axis=-1,
    ).float()


def flydsl_act_qdq(x):
    packed, scale = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    return mxfp4_dequant(packed, scale.view(torch.uint8)[: x.shape[0]])


def triton_act_qdq(x):
    return mxfp4_dequant(*mxfp4_quant(x))


def gptoss_reference(cfg, hidden, raw_weights, topk_weights, topk_ids, act_qdq):
    """fp32 GPT-OSS MoE over dequantized weights, MXFP4-rounding activations
    with the stack's own quantizer before each GEMM."""
    w1, s1, b1, w2, s2, b2 = raw_weights
    x = act_qdq(hidden)
    out = torch.zeros(hidden.shape[0], cfg.model_dim, dtype=torch.float32)
    for expert in topk_ids.unique().tolist():
        rows, cols = (topk_ids == expert).nonzero(as_tuple=True)
        h = x[rows] @ mxfp4_dequant(w1[expert], s1[expert]).T + b1[expert]
        gate = h[:, : cfg.inter_dim].clamp(max=GPTOSS_SWIGLU_LIMIT)
        up = h[:, cfg.inter_dim :].clamp(-GPTOSS_SWIGLU_LIMIT, GPTOSS_SWIGLU_LIMIT)
        act = gate * torch.sigmoid(GPTOSS_SWIGLU_ALPHA * gate) * (up + 1)
        act = act_qdq(act.to(torch.bfloat16))
        y = act @ mxfp4_dequant(w2[expert], s2[expert]).T + b2[expert]
        out.index_add_(0, rows, y * topk_weights[rows, cols, None])
    return out


def run_tokens(args, cfg, tokens, flydsl_weights, triton_weights, raw_weights):
    csv_hit = gptoss_csv_row(cfg, tokens) if args.model == "gptoss" else None
    hidden_generator = torch.Generator(device="cuda").manual_seed(tokens)
    hidden = torch.randn(
        tokens,
        cfg.model_dim,
        generator=hidden_generator,
        dtype=torch.bfloat16,
    )
    repeat_results = []
    kernel_tables = {}
    output_diffs = {}
    for repeat in range(args.repeats):
        repeat_results.append(
            measure_seeds(
                args,
                cfg,
                tokens,
                hidden,
                flydsl_weights,
                triton_weights,
                raw_weights if repeat == 0 else None,
                kernel_tables if repeat == 0 else None,
                output_diffs,
            )
        )
        print(f"M={tokens} repeat {repeat + 1}/{args.repeats} done", flush=True)
    results = {
        key: {
            metric: min(run[key][metric] for run in repeat_results)
            for metric in repeat_results[0][key]
        }
        for key in repeat_results[0]
    }
    return results, csv_hit, kernel_tables, output_diffs


def measure_seeds(
    args,
    cfg,
    tokens,
    hidden,
    flydsl_weights,
    triton_weights,
    raw_weights,
    kernel_tables,
    output_diffs,
):
    _, build_flydsl, build_triton = BUILDERS[args.model]
    samples = collections.defaultdict(lambda: collections.defaultdict(list))
    for seed in range(args.seeds):
        logits, bias = make_router_inputs(cfg, tokens, seed)
        cases = {}
        for herd in (False, True):
            flydsl = build_flydsl(cfg, hidden, logits, bias, flydsl_weights, herd)
            triton = build_triton(cfg, hidden, logits, bias, triton_weights, herd)
            if not torch.equal(flydsl[2], triton[2]):
                mismatch = int((flydsl[2] != triton[2]).sum())
                raise AssertionError(
                    f"M={tokens} seed={seed} HERD={int(herd)} expert selection "
                    f"mismatch: {mismatch}/{tokens * cfg.topk} IDs differ"
                )
            cases[("FlyDSL", herd)] = flydsl
            cases[("Triton", herd)] = triton
            if seed == 0 and raw_weights is not None:
                topk_weights, topk_ids = (t.clone() for t in flydsl[0]())
                output_diffs[herd] = {
                    stack: logits_diff(
                        case[1](),
                        gptoss_reference(
                            cfg, hidden, raw_weights, topk_weights, topk_ids, qdq
                        ),
                    )
                    for stack, case, qdq in (
                        ("FlyDSL", flydsl, flydsl_act_qdq),
                        ("Triton", triton, triton_act_qdq),
                    )
                }

        for stack in ("FlyDSL", "Triton"):
            for herd in (False, True):
                route, step, ids = cases[(stack, herd)]
                stages, kernels = stage_kernel_us(stack, route, step)
                ran_herd = any(
                    marker in name
                    for name, (stage, _) in kernels.items()
                    if stage != "moe"
                    for marker in HERD_KERNEL_MARKERS
                )
                if ran_herd != herd:
                    raise AssertionError(
                        f"{stack} HERD={int(herd)} M={tokens}: HERD routing "
                        f"kernels present={ran_herd}"
                    )
                sample = samples[(stack, herd)]
                sample["unique"].append(int(ids.unique().numel()))
                for stage in STAGES:
                    sample[stage].append(stages[stage])
                sample["e2e"].append(graph_us(step))
                if seed == 0 and kernel_tables is not None:
                    kernel_tables[(stack, herd)] = kernels
        print(f"M={tokens} seed={seed}: expert selection parity PASS", flush=True)

    return {
        key: {metric: statistics.median(values) for metric, values in sample.items()}
        for key, sample in samples.items()
    }


def pct(before, after):
    return (after - before) / before * 100


def report(args, cfg, tokens, results, csv_hit, kernel_tables, output_diffs):
    print(f"\n### {cfg.name}, M={tokens}, {cfg.quant}, gfx1250\n")
    print(
        "Expert selection parity: PASS "
        f"(FlyDSL == Triton for HERD off/on across {args.seeds} seeds)"
    )
    if csv_hit:
        print(f"FlyDSL CSV hit ({os.path.basename(GPTOSS_CSV)}): {csv_hit}")
    for herd, diffs in output_diffs.items():
        cells = ", ".join(
            f"{stack} {diff:.2e} {'PASS' if diff < 0.01 else 'FAIL'}"
            for stack, diff in diffs.items()
        )
        print(
            f"Output vs fp32 torch reference (own MXFP4 act quant), seed 0, HERD "
            f"{'on' if herd else 'off'} (logits_diff < 0.01): {cells}"
        )
    print()
    print(
        "| Stack | HERD | Unique experts | Selector us | Sort us | MoE us "
        "| Selector+Sort+MoE us | E2E us |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for stack in ("FlyDSL", "Triton"):
        for herd in (False, True):
            row = results[(stack, herd)]
            total = sum(row[stage] for stage in STAGES)
            print(
                f"| {stack} | {'on' if herd else 'off'} | {row['unique']:g} "
                f"| {row['selector']:.2f} | {row['sort']:.2f} | {row['moe']:.2f} "
                f"| {total:.2f} | {row['e2e']:.2f} |"
            )
        off = results[(stack, False)]
        on = results[(stack, True)]
        total_off = sum(off[stage] for stage in STAGES)
        total_on = sum(on[stage] for stage in STAGES)
        print(
            f"| {stack} | delta | {on['unique'] - off['unique']:+g} "
            f"| {on['selector'] - off['selector']:+.2f} "
            f"| {on['sort'] - off['sort']:+.2f} "
            f"| {on['moe'] - off['moe']:+.2f} ({pct(off['moe'], on['moe']):+.1f}%) "
            f"| {total_on - total_off:+.2f} "
            f"| {on['e2e'] - off['e2e']:+.2f} ({pct(off['e2e'], on['e2e']):+.1f}%) |"
        )
    print("\nTriton vs FlyDSL (negative = Triton faster):\n")
    print("| HERD | Selector us | Sort us | MoE us | E2E us |")
    print("|---|---:|---:|---:|---:|")
    for herd in (False, True):
        fly = results[("FlyDSL", herd)]
        tri = results[("Triton", herd)]
        cells = " | ".join(
            f"{tri[metric] - fly[metric]:+.2f}" for metric in (*STAGES, "e2e")
        )
        print(f"| {'on' if herd else 'off'} | {cells} |")
    if args.kernels:
        for (stack, herd), kernels in kernel_tables.items():
            print(f"\nKernels, seed 0, {stack} HERD {'on' if herd else 'off'}:")
            for name, (stage, us) in sorted(
                kernels.items(),
                key=lambda item: (STAGES.index(item[1][0]), -item[1][1]),
            ):
                print(f"  {stage:8s} {us:8.2f}  {name[:150]}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=list(MODELS), default="dsv4flash")
    parser.add_argument("--M", nargs="+", type=int, default=[64])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeat every seed this many times and keep the fastest median",
    )
    parser.add_argument("--kernels", action="store_true")
    args = parser.parse_args()

    if get_gfx() != "gfx1250" or get_arch() != "gfx1250":
        raise SystemExit(
            f"this comparison requires gfx1250, got get_gfx={get_gfx()} "
            f"get_arch={get_arch()}"
        )
    for tokens in args.M:
        if not 16 <= tokens <= 128:
            raise SystemExit(f"HERD serves 16 <= M <= 128, got M={tokens}")

    flydsl_topk_mod._FLYDSL_HERD_MIN_M = 16
    flydsl_topk_mod._FLYDSL_HERD_MAX_M = 128
    triton_routing_mod._HERD_MIN_M = 16
    triton_routing_mod._HERD_MAX_M = 128

    cfg = MODELS[args.model]
    make_weights = BUILDERS[args.model][0]
    weights = make_weights(cfg)
    for tokens in args.M:
        report(args, cfg, tokens, *run_tokens(args, cfg, tokens, *weights))


if __name__ == "__main__":
    main()
