# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Operand builders and the four timed paths of the MiniMax-M3.1 decode MoE layer (E128 top-4, H6144 / I768), shared by
``op_tests/test_fused_moe_m3_decode.py`` and the decode-chain fmoe tuner (``gemm_moe_tune.py --decode-chain``).

Random bytes are valid operands in every kernel layout (they are permutations of the logical tensors), so the
builders need no quantizer. Imports torch and the gfx950 kernels: GPU only.
"""

import statistics

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_route_quant
from aiter.ops.flydsl import fused_moe_m3_decode as m3
from aiter.ops.flydsl.kernels.mega_moe_tp import decode_expert_gemms as gemms
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_chain import route_quant_small
from aiter.ops.flydsl.kernels.moe_slots_shared_reduce import moe_slots_shared_reduce

H, I, E, TOPK, SCALE = 6144, 768, 128, 4, 2.0
ROWS = (8, 16, 32, 64, 128)
DEV = "cuda"
dev = DEV


def _u8(shape, g, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.uint8, device=dev)


def make_layer(seed=0):
    """One layer's weights in the layouts the model holds (random bytes are valid in every layout: they are
    permutations of the logical tensors)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w1 = _u8((E, 2 * I, H // 2), g).view(dtypes.fp4x2)
    w2 = _u8((E, H, I // 2), g).view(dtypes.fp4x2)
    for w in (w1, w2):
        w.is_shuffled = True
    sw1 = (
        (torch.randn(1, 2 * I, H, generator=g, device=dev) * 0.15)
        .to(dtypes.fp8)
        .contiguous()
    )
    sw2 = (
        (torch.randn(1, H, I, generator=g, device=dev) * 0.15)
        .to(dtypes.fp8)
        .contiguous()
    )
    for w in (sw1, sw2):
        w.is_shuffled = True
    return {
        "w1": w1,
        "w2": w2,
        "w1_scale": _u8((E, 2 * I, H // 32), g, 121, 124).view(dtypes.fp8_e8m0),
        "w2_scale": _u8((E, H, I // 32), g, 121, 124).view(dtypes.fp8_e8m0),
        "shared_w1": sw1,
        "shared_w2": sw2,
        "shared_w1_scale": _u8((2 * I, H // 32), g, 121, 124).view(dtypes.fp8_e8m0),
        "shared_w2_scale": _u8((H, I // 32), g, 121, 124).view(dtypes.fp8_e8m0),
    }


def make_router(seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    gate = (torch.randn(E, H, device=dev, generator=g) * 0.05).float().contiguous()
    bias = (torch.randn(E, device=dev, generator=g) * 0.1).float()
    return gate, bias


def make_x(n, seed=2):
    g = torch.Generator(device=dev).manual_seed(seed + n)
    return torch.randn(n, H, device=dev, generator=g).bfloat16()


# ---------------------------------------------------------------- the four paths


def chain_direct(x, L, gate, lines, bias, ecfg=None, rcfg=None):
    """aiter's chain ops, as ATOM #3787 calls them."""
    n = x.shape[0]
    (xq, xs), (q8, s8), w, ids = route_quant_small(x, gate, lines, bias, SCALE, rcfg)
    s13, ss13, ss2 = m3.prepare_decode_weights(
        L["w1_scale"], L["shared_w1_scale"], L["shared_w2_scale"]
    )
    plan = gemms.get_plan(
        n, w13_split=True, tiles=None if ecfg is None else ecfg.to_tiles()
    )
    u = lambda t: t.view(torch.uint8).view(-1)
    slots, shared = gemms.decode_expert_gemms(
        plan,
        u(xq),
        u(xs),
        u(q8),
        u(s8),
        (u(L["w1"]), u(s13), u(L["w2"]), u(L["w2_scale"])),
        (u(L["shared_w1"]), u(ss13), u(L["shared_w2"]), u(ss2)),
        ids,
        w,
    )
    return moe_slots_shared_reduce(slots, shared, topk_ids=ids, num_experts=E), w, ids


def shared_kw(L):
    return {
        "shared_w1": L["shared_w1"],
        "shared_w2": L["shared_w2"],
        "shared_w1_scale": L["shared_w1_scale"],
        "shared_w2_scale": L["shared_w2_scale"],
        "shared_folded": True,
    }


def integrated(x, L, w, ids):
    """fused_moe on bf16 rows and the caller's routes: the quant runs inside the tuned whole-graph row."""
    return fused_moe(
        x,
        L["w1"],
        L["w2"],
        w,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=L["w1_scale"],
        w2_scale=L["w2_scale"],
        swiglu_limit=7.0,
        **shared_kw(L),
    )


def integrated_router(x, L, gate, bias):
    """The router entry (gate GEMM + top-4 + quant) in front of fused_moe on its quantized rows."""
    routed = moe_route_quant(x, gate, bias, SCALE)
    (xq, xs), (q8, s8), w, ids = routed
    return fused_moe(
        xq.view(dtypes.fp4x2),
        L["w1"],
        L["w2"],
        w,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=L["w1_scale"],
        w2_scale=L["w2_scale"],
        a1_scale=xs,
        swiglu_limit=7.0,
        shared_a1=(q8, s8),
        **shared_kw(L),
    )


def _swiglu_oai(gate_up):
    g, u = gate_up[:, :I].float().clamp(max=7.0), gate_up[:, I:].float().clamp(
        -7.0, 7.0
    )
    return (g * torch.sigmoid(1.702 * g) * (u + 1.0)).bfloat16()


def public_path(x, L, gate_bf16, bias):
    """What ATOM runs without ATOM_M3_DECODE_MOE: BF16 gate GEMM, topk_gating, the tuned 2-stage fused_moe, the
    routed scale, and the shared expert as MXFP8 quant + GEMM + SwiGLU + quant + GEMM.
    """
    import aiter
    from aiter import get_hip_quant
    from aiter.ops.topk import topk_gating

    n = x.shape[0]
    logits = torch.nn.functional.linear(x, gate_bf16)
    w = torch.empty(n, TOPK, dtype=torch.float32, device=dev)
    ids = torch.empty(n, TOPK, dtype=torch.int32, device=dev)
    topk_gating(w, ids, logits, bias, True, 1.0, score_func="sigmoid")
    routed = fused_moe(
        x,
        L["w1"],
        L["w2"],
        w,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=L["w1_scale"],
        w2_scale=L["w2_scale"],
        swiglu_limit=7.0,
    )
    routed = routed * SCALE
    quant = get_hip_quant(QuantType.per_1x32)
    xq, xs = quant(x, quant_dtype=dtypes.fp8)
    gu = aiter.gemm_a8w8_mxfp8_bpreshuffle(
        xq, L["shared_w1"].view(2 * I, H), xs, L["shared_w1_scale"], dtype=dtypes.bf16
    )
    h = _swiglu_oai(gu)
    hq, hs = quant(h, quant_dtype=dtypes.fp8)
    sh = aiter.gemm_a8w8_mxfp8_bpreshuffle(
        hq, L["shared_w2"].view(H, I), hs, L["shared_w2_scale"], dtype=dtypes.bf16
    )
    return routed + sh


def integrated_public_router(x, L, gate_bf16, bias):
    from aiter.ops.topk import topk_gating

    n = x.shape[0]
    logits = torch.nn.functional.linear(x, gate_bf16)
    w = torch.empty(n, TOPK, dtype=torch.float32, device=dev)
    ids = torch.empty(n, TOPK, dtype=torch.int32, device=dev)
    topk_gating(w, ids, logits, bias, True, SCALE, score_func="sigmoid")
    return integrated(x, L, w, ids)


def graph_median_us(fn, sets, reps=30):
    """Per-call time of ``fn(set)`` over cold-rotated weight sets, one captured graph replayed ``reps`` times."""
    for s in sets:
        fn(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for s in sets:
            fn(s)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        ev[0].record()
        graph.replay()
        ev[1].record()
        torch.cuda.synchronize()
        ts.append(ev[0].elapsed_time(ev[1]) * 1000 / len(sets))
    return statistics.median(ts)
