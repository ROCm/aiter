#!/usr/bin/env python
"""Tuner for the mxfp4 (a4w4) FlyDSL MoE port on gfx950.

For every (shape, token) it scans the buildable port variants -- BM x B-load x
gemm2-epilog (enumerated by the self-contained `flyg_variants`) -- benchmarks each,
drops numerically-broken ones (cosine gate vs the legacy ``flydsl_moe`` output),
and records the fastest survivor. Output is a ``tuned_fmoe``-schema CSV whose rows
are tagged ``_tag=mxfp4_guinterleave`` (the FlyDSL port; separate from the HIP
mxfp4_moe backend) so ``fused_moe.get_2stage_cfgs`` prefers them when
``w1.shuffle_kind == "mxfp4_guinterleave"``.

The tunable knobs (all encoded in the kernelName the row selects):
  * BM / sort block  : 16 (inline_quant+atomic, decode) / 32 (atomic) / 128 (nonatomic, prefill)
  * B-load           : NT (read-once) vs CACHED (reused) -- gemm1 and gemm2
  * gemm2 epilog     : ATOMIC[_NT] / NONATOMIC / NONATOMIC_MXFP4OUT (fp4 intermediate)

Deploy: merge the emitted rows into the active ``tuned_fmoe.csv`` (or point
``AITER_CONFIG_FMOE`` at the output). De-dup is keyed on the same ``_INDEX_COLS``
``get_2stage_cfgs`` uses, so re-running and re-merging is idempotent per (shape, token).

Examples:
  python tune_mxfp4_moe.py --shapes all -M 1,8,32,128,512,2048,8192,16384
  python tune_mxfp4_moe.py --shapes minimax_b,qwen35_397b -M 256,4096 --out my_tuned.csv
  AITER_MOE_EXPERT_BALANCE=1 python tune_mxfp4_moe.py ...   # stable, even token distribution
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _mxfp4_moe_run,
    _parse_mxfp4_g1_kname,
    fused_moe,
    get_padded_M,
)
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import run_perftest
from aiter.utility.fp4_utils import e8m0_shuffle


# ====================================================================
# Self-contained kernel enumeration + test harness (no bench import).
# The (shape, variant) -> kernelName mapping and the candidate set are
# defined here so the tuner owns "select the optimal kernel" end to end.
# ====================================================================
@dataclass(frozen=True)
class Shape:
    NE: int
    H: int
    INTER: int
    TOPK: int


SHAPES = {
    "kimi_k2.5": Shape(NE=385, H=7168, INTER=512, TOPK=9),
    "dsv3_a": Shape(NE=32, H=7168, INTER=2048, TOPK=8),
    "dsv3_b": Shape(NE=257, H=7168, INTER=512, TOPK=9),
    "dsv3_c": Shape(NE=257, H=7168, INTER=256, TOPK=9),
    "kimik2_a": Shape(NE=384, H=7168, INTER=512, TOPK=8),
    "kimik2_b": Shape(NE=385, H=7168, INTER=256, TOPK=9),
    "minimax_a": Shape(NE=256, H=3072, INTER=1536, TOPK=8),
    "minimax_b": Shape(NE=256, H=3072, INTER=768, TOPK=8),
    "qwen35_397b": Shape(NE=512, H=4096, INTER=256, TOPK=10),
    # DeepSeek V4 geometry (a4w4 port of the fp8fp4 production MoE).
    "dsv4_ep8": Shape(NE=48, H=7168, INTER=3072, TOPK=6),
    "dsv4_tp2": Shape(NE=384, H=7168, INTER=1536, TOPK=6),
    "dsv4_tp4": Shape(NE=384, H=7168, INTER=768, TOPK=6),
    "dsv4_tp6": Shape(NE=384, H=7168, INTER=512, TOPK=6),
    "dsv4_lite": Shape(NE=256, H=4096, INTER=256, TOPK=6),
}


def g1_kernel_name(shape, bm, variant):
    """gemm1 kernelName. variant: 'INLINEQUANT'(BM16) | 'NT'|'CACHED'(BM32/64) | ''(BM128)."""
    base = f"mxfp4_moe_g1_a4w4_NE{shape.NE}_H{shape.H}_E{shape.INTER}_BM{bm}"
    return f"{base}_{variant}" if variant else base


def g2_kernel_name(shape, bm, variant):
    """gemm2 kernelName. variant: ATOMIC[_NT] | NONATOMIC[_CSHUFFLE|_MXFP4OUT]."""
    base = (
        f"mxfp4_moe_g2_a4w4_NE{shape.NE}_H{shape.H}_E{shape.INTER}"
        f"_TOPK{shape.TOPK}_BM{bm}"
    )
    return f"{base}_{variant}"


def _mxfp4out_ok(shape):
    # mxfp4-out (lossy fp4 intermediate) is validated only for the codegen'd Kimi/DSR
    # shapes; it does not transfer to non-Kimi INTER=256.
    return shape.NE in (257, 385) and shape.H == 7168 and shape.INTER == 512


def flyg_variants(shape):
    """All buildable port variants as (label, kernelName1, kernelName2); the sort
    block / gemm1 BM always equals the gemm2 BM. This is the search space the tuner
    scans per (shape, token):
      * BM16 inline_quant x gemm2 ATOMIC[_NT]            (decode)
      * BM32 {NT,CACHED}  x gemm2 ATOMIC[_NT]            (mid M)
      * BM128             x gemm2 NONATOMIC[_CSHUFFLE]   (prefill; +MXFP4OUT if ok)
      * BM64              x gemm2 NONATOMIC_CSHUFFLE     (mid-prefill sweet spot)."""
    out = []
    for g2v in ("ATOMIC_NT", "ATOMIC"):
        out.append(
            (
                f"BM16_IQ+{g2v}",
                g1_kernel_name(shape, 16, "INLINEQUANT"),
                g2_kernel_name(shape, 16, g2v),
            )
        )
    for g1v in ("NT", "CACHED"):
        for g2v in ("ATOMIC_NT", "ATOMIC"):
            out.append(
                (
                    f"BM32_{g1v}+{g2v}",
                    g1_kernel_name(shape, 32, g1v),
                    g2_kernel_name(shape, 32, g2v),
                )
            )
    g2vs = ["NONATOMIC", "NONATOMIC_CSHUFFLE"]
    if _mxfp4out_ok(shape):
        g2vs.append("NONATOMIC_MXFP4OUT")
    for g2v in g2vs:
        out.append(
            (
                f"BM128+{g2v}",
                g1_kernel_name(shape, 128, ""),
                g2_kernel_name(shape, 128, g2v),
            )
        )
    out.append(
        (
            "BM64+NONATOMIC_CSHUFFLE",
            g1_kernel_name(shape, 64, ""),
            g2_kernel_name(shape, 64, "NONATOMIC_CSHUFFLE"),
        )
    )
    return out


def build_weights(shape, device, seed=0):
    """Build fly (legacy preshuffle) + mx (mxfp4_moe a16w4 interleaved) weight sets."""
    torch.manual_seed(seed)
    ne, h, inter = shape.NE, shape.H, shape.INTER
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn((ne, 2 * inter, h), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((ne, h, inter), dtype=dtypes.bf16, device=device) / 10
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    fly = dict(
        w1=shuffle_weight(w1_qt, layout=(16, 16)),
        w2=shuffle_weight(w2_qt, layout=(16, 16)),
        w1_scale=e8m0_shuffle(w1_scale),
        w2_scale=e8m0_shuffle(w2_scale),
    )
    mx_w1 = shuffle_weight_a16w4(w1_qt, 16, True)
    mx_w1.shuffle_kind = "mxfp4_guinterleave"
    mx = dict(
        w1=mx_w1,
        w2=shuffle_weight_a16w4(w2_qt, 16, False),
        w1_scale=shuffle_scale_a16w4(w1_scale, ne, True),
        w2_scale=shuffle_scale_a16w4(w2_scale, ne, False),
    )
    return fly, mx


def build_inputs(shape, M, device, seed=1):
    """Build hidden + topk_ids/topk_weight (expert 0 = shared, rest routed)."""
    torch.manual_seed(seed)
    ne, h, topk = shape.NE, shape.H, shape.TOPK
    hidden = torch.randn((M, h), dtype=dtypes.bf16, device=device) / 10
    n_routed = ne - 1
    shared_id = ne - 1
    n_topk_routed = topk - 1
    g = torch.Generator(device=device).manual_seed(seed)
    bias = torch.randn(n_routed, generator=g, device=device) * 0.5
    scores = torch.randn(M, n_routed, generator=g, device=device) + bias
    routed_w, routed_ids = torch.topk(scores.softmax(-1), n_topk_routed, dim=-1)
    shared_ids = torch.full((M, 1), shared_id, device=device, dtype=routed_ids.dtype)
    shared_w = torch.ones((M, 1), device=device, dtype=routed_w.dtype)
    topk_ids = torch.cat([shared_ids, routed_ids], dim=1).to(torch.int32)
    topk_weight = torch.cat([shared_w, routed_w], dim=1).to(torch.float32)
    return hidden, topk_ids, topk_weight


def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def run_fly(shape, M, fly_w, hidden, topk_ids, topk_weight, iters, warmup):
    """Legacy 2-stage flydsl_moe via fused_moe (no shuffle_kind) -> (us, out)."""

    def fn():
        return fused_moe(
            hidden,
            fly_w["w1"],
            fly_w["w2"],
            topk_weight,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=fly_w["w1_scale"],
            w2_scale=fly_w["w2_scale"],
        )

    out = fn().clone()
    _, us = run_perftest(fn, num_warmup=warmup, num_iters=iters)
    return us, out


def parse_shapes(value):
    if value.strip() == "all":
        return list(SHAPES)
    names = [v.strip() for v in value.split(",") if v.strip()]
    bad = [n for n in names if n not in SHAPES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown shapes: {bad}; choose from {list(SHAPES)}"
        )
    return names


# tuned_fmoe.csv schema (must match get_2stage_cfgs' reader).
CSV_COLS = [
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "tflops",
    "bw",
    "_tag",
]
# Fixed key columns for the a4w4 mxfp4_moe backend. act_type is set to the
# nominal Silu but the lookup matches via the act-disabled fallback, so any
# activation resolves these rows.
FIXED = {
    "act_type": "ActivationType.Silu",
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float4_e2m1fn_x2",
    "q_dtype_w": "torch.float4_e2m1fn_x2",
    "q_type": "QuantType.per_1x32",
    "use_g1u1": 1,
    "doweight_stage1": 0,
    "_tag": "mxfp4_guinterleave",
}
GATE = 0.90  # cosine vs flydsl_moe; rejects broken variants, not the fp4 ceiling


def candidate_variants(shape, M):
    """flyg_variants pruned by M (skip BM128 at tiny M, BM16 at huge M to save time;
    everything in range is scanned exhaustively)."""
    out = []
    for label, k1, k2 in flyg_variants(shape):
        bm = _parse_mxfp4_g1_kname(k1)["BM"]
        if M <= 64 and bm == 128:  # BM128 never wins at tiny M
            continue
        # Always keep BM16: it self-loses to BM128 at large M for codegen'd shapes,
        # but is the only buildable variant for BM16-only shapes (flyg_variants can't
        # tell which is which -- the unbuildable BM32/128 fail fast via try/except).
        out.append((label, k1, k2))
    return out


def time_variant(shape, mx_w, hidden, tids, tw, ref, k1, k2, iters, warmup):
    """Run one variant; return its us, or None if it fails the cosine gate."""

    def fn():
        mx_w["w1"].gemm1_backend = "flydsl"
        mx_w["w2"].gemm2_backend = "flydsl"
        return _mxfp4_moe_run(
            hidden,
            mx_w["w1"],
            mx_w["w2"],
            tids,
            tw,
            shape.TOPK,
            kernelName1=k1,
            kernelName2=k2,
            w1_scale=mx_w["w1_scale"],
            w2_scale=mx_w["w2_scale"],
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
        )

    out = fn().clone()
    if ref is not None and cosine(out, ref) < GATE:
        return None
    _, us = run_perftest(fn, num_warmup=warmup, num_iters=iters)
    return us


def tune_shape(name, shape, M_list, cu_num, writer, iters, warmup):
    dev = torch.device("cuda")
    fly_w, mx_w = build_weights(shape, dev)
    # The legacy a4w4 SEGFAULTS on the dsv4 geometry (uncatchable, kills the tuner),
    # so anchor the cosine gate on the port's own BM16-inline output for dsv4 instead
    # of run_fly. BM16 is the simplest/most-trusted variant and runs for all dsv4 M.
    _dsv4 = name.startswith("dsv4")
    for M in M_list:
        hidden, tids, tw = build_inputs(shape, M, dev)
        if _dsv4:
            try:
                mx_w["w1"].gemm1_backend = "flydsl"
                mx_w["w2"].gemm2_backend = "flydsl"
                ref = _mxfp4_moe_run(
                    hidden,
                    mx_w["w1"],
                    mx_w["w2"],
                    tids,
                    tw,
                    shape.TOPK,
                    kernelName1=g1_kernel_name(shape, 16, "INLINEQUANT"),
                    kernelName2=g2_kernel_name(shape, 16, "ATOMIC"),
                    w1_scale=mx_w["w1_scale"],
                    w2_scale=mx_w["w2_scale"],
                    activation=ActivationType.Silu,
                    quant_type=QuantType.per_1x32,
                ).clone()
            except Exception:
                ref = None
        else:
            try:
                _, ref = run_fly(shape, M, fly_w, hidden, tids, tw, iters, warmup)
            except Exception:
                ref = None  # no anchor -> accept any variant that runs
        best = None
        for label, k1, k2 in candidate_variants(shape, M):
            try:
                us = time_variant(
                    shape, mx_w, hidden, tids, tw, ref, k1, k2, iters, warmup
                )
            except Exception as e:
                if os.environ.get("TUNE_DEBUG"):
                    print(f"    skip {label}: {type(e).__name__}: {e}")
                continue
            if us is None:
                continue
            if best is None or us < best[0]:
                best = (us, label, k1, k2)
        if best is None:
            print(f"{name:>14} M={M:>6}: NO VALID VARIANT")
            continue
        us, label, k1, k2 = best
        p1 = _parse_mxfp4_g1_kname(k1)
        row = {c: "" for c in CSV_COLS}
        row.update(FIXED)
        row.update(
            cu_num=cu_num,
            token=get_padded_M(M),
            model_dim=shape.H,
            inter_dim=shape.INTER,
            expert=shape.NE,
            topk=shape.TOPK,
            block_m=p1["BM"],
            ksplit=p1["kSplitK"],
            kernelName1=k1,
            kernelName2=k2,
            us=round(us, 2),
        )
        writer.writerow(row)
        print(f"{name:>14} M={M:>6}: {label:>26} {us:8.1f}us (token={get_padded_M(M)})")
        del hidden, tids, tw
        torch.cuda.empty_cache()
    del fly_w, mx_w
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--shapes", default="all", help="comma list or 'all'; " + ", ".join(SHAPES)
    )
    ap.add_argument("-M", "--M-list", default="1,8,32,128,512,2048,8192,16384")
    ap.add_argument("--out", default="mxfp4_tuned.csv")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    shapes = parse_shapes(args.shapes)
    M_list = [int(x) for x in args.M_list.split(",")]
    cu_num = get_cu_num()
    print(f"GPU: {torch.cuda.get_device_name(0)} cu_num={cu_num}")
    print(f"shapes={shapes}  M={M_list}  -> {args.out}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for name in shapes:
            try:
                tune_shape(
                    name, SHAPES[name], M_list, cu_num, w, args.iters, args.warmup
                )
                f.flush()
            except Exception as e:
                print(f"{name} FAILED: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
