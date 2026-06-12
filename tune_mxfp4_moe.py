#!/usr/bin/env python
"""Tuner for the mxfp4 (a4w4) FlyDSL MoE port on gfx950.

For every (shape, token) it scans the buildable port variants -- BM x B-load x
gemm2-epilog (enumerated by ``bench_up_moe_v2.flyg_variants``) -- benchmarks each,
drops numerically-broken ones (cosine gate vs the legacy ``flydsl_moe`` output),
and records the fastest survivor. Output is a ``tuned_fmoe``-schema CSV whose rows
are tagged ``_tag=mxfp4_moe`` so ``fused_moe.get_2stage_cfgs`` prefers them when
``w1.shuffle_kind == "mxfp4_moe"``.

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

import torch

import bench_up_moe_v2 as B
from aiter import ActivationType, QuantType
from aiter.fused_moe import _parse_mxfp4_g1_kname, get_padded_M
from aiter.jit.utils.chip_info import get_cu_num

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
    "_tag": "mxfp4_moe",
}
GATE = 0.90  # cosine vs flydsl_moe; rejects broken variants, not the fp4 ceiling


def candidate_variants(shape, M):
    """flyg_variants pruned by M (skip BM128 at tiny M, BM16 at huge M to save time;
    everything in range is scanned exhaustively)."""
    out = []
    for label, k1, k2 in B.flyg_variants(shape):
        bm = _parse_mxfp4_g1_kname(k1)["BM"]
        if M <= 64 and bm == 128:
            continue
        if M >= 8192 and bm == 16:
            continue
        out.append((label, k1, k2))
    return out


def time_variant(shape, mx_w, hidden, tids, tw, ref, k1, k2, iters, warmup):
    """Run one variant; return its us, or None if it fails the cosine gate."""

    def fn():
        mx_w["w1"].gemm1_backend = "flydsl"
        mx_w["w2"].gemm2_backend = "flydsl"
        return B._mxfp4_moe_run(
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
    if ref is not None and B.cosine(out, ref) < GATE:
        return None
    _, us = B.run_perftest(fn, num_warmup=warmup, num_iters=iters)
    return us


def tune_shape(name, shape, M_list, cu_num, writer, iters, warmup):
    dev = torch.device("cuda")
    fly_w, mx_w = B.build_weights(shape, dev)
    for M in M_list:
        hidden, tids, tw = B.build_inputs(shape, M, dev)
        try:
            _, ref = B.run_fly(shape, M, fly_w, hidden, tids, tw, iters, warmup)
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
        "--shapes", default="all", help="comma list or 'all'; " + ", ".join(B.SHAPES)
    )
    ap.add_argument("-M", "--M-list", default="1,8,32,128,512,2048,8192,16384")
    ap.add_argument("--out", default="mxfp4_tuned.csv")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    shapes = B.parse_shapes(args.shapes)
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
                    name, B.SHAPES[name], M_list, cu_num, w, args.iters, args.warmup
                )
                f.flush()
            except Exception as e:
                print(f"{name} FAILED: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
