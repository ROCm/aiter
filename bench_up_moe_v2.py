import argparse
import hashlib
import os
import sys
from dataclasses import dataclass

sys.path.insert(
    0, os.environ.get("AITER_REPO", os.path.dirname(os.path.abspath(__file__)))
)

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import _mxfp4_moe_run, fused_moe, get_padded_M
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import run_perftest
from aiter.utility.fp4_utils import e8m0_shuffle


@dataclass(frozen=True)
class Shape:
    NE: int
    H: int
    INTER: int
    TOPK: int


# mxfmoe(a4w4)-supported configs from aiter/configs/model_configs/*_tuned_fmoe.csv.
# Filter: q_dtype_a==q_dtype_w==fp4x2, q_type==per_1x32, act==Silu, g1u1==1,
# H%256==0, INTER%256==0. Excludes *_fp8fp4 (qa=fp8) and gptoss (Swiglu).
# NOTE: all of these now run flyg via the BM16 inline_quant path:
#   - mxfp4_moe_sort C++ instantiations (mxfp4_moe_aux.cu): NE{257,385}/TOPK9/H7168
#     (Kimi/DSR) + NE256/TOPK8/H3072 (minimax);
#   - the generic moe_sorting fallback in _mxfp4_moe_run for any other NE/TOPK
#     (e.g. dsv3_a NE32, kimik2_a NE384, qwen35 NE512/TOPK10);
#   - the gemm2 K_TILES_TOTAL<=kStages fast path handles D_INTER==256 (INTER256).
# The BM32/BM128 variants still need the H=7168-locked mxfp4_moe_sort_scales, so
# non-KIMI shapes land on the BM16 inline_quant variant (no a-scale kernel).
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
}


def g1_kernel_name(shape, bm, variant):
    """gemm1 kernelName. variant: 'INLINEQUANT'(BM16) | 'NT'|'CACHED'(BM32) | ''(BM128)."""
    base = f"mxfp4_moe_g1_a4w4_NE{shape.NE}_H{shape.H}_E{shape.INTER}_BM{bm}"
    return f"{base}_{variant}" if variant else base


def g2_kernel_name(shape, bm, variant):
    """gemm2 kernelName. variant: 'ATOMIC_NT'|'ATOMIC'(BM16/32) |
    'NONATOMIC'|'NONATOMIC_MXFP4OUT'(BM128)."""
    base = (
        f"mxfp4_moe_g2_a4w4_NE{shape.NE}_H{shape.H}_E{shape.INTER}"
        f"_TOPK{shape.TOPK}_BM{bm}"
    )
    return f"{base}_{variant}"


def _mxfp4out_ok(shape):
    # mxfp4-out is validated only for the codegen'd Kimi/DSR shapes; it does not
    # transfer to non-Kimi INTER=256 (loses to tuned BM32 at mid-M, broken at large M).
    return shape.NE in (257, 385) and shape.H == 7168 and shape.INTER == 512


def flyg_variants(shape):
    """?? (label, kernelName1, kernelName2) flyg ??;gemm1 BM == gemm2 BM?"""
    out = []
    # BM16: g1 INLINEQUANT x g2 {ATOMIC_NT, ATOMIC}
    for g2v in ("ATOMIC_NT", "ATOMIC"):
        out.append(
            (
                f"BM16_IQ+{g2v}",
                g1_kernel_name(shape, 16, "INLINEQUANT"),
                g2_kernel_name(shape, 16, g2v),
            )
        )
    # BM32: g1 {NT, CACHED} x g2 {ATOMIC_NT, ATOMIC}
    for g1v in ("NT", "CACHED"):
        for g2v in ("ATOMIC_NT", "ATOMIC"):
            out.append(
                (
                    f"BM32_{g1v}+{g2v}",
                    g1_kernel_name(shape, 32, g1v),
                    g2_kernel_name(shape, 32, g2v),
                )
            )
    # BM128: g1 '' x g2 {NONATOMIC, NONATOMIC_CSHUFFLE (+MXFP4OUT if shape ok)}.
    # CSHUFFLE = coalesced flat_out write (2-pass cshuffle); fly's mfma_moe2 recipe.
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
    # BM64 cshuffle -- the mid-prefill (M~2048) sweet spot: bigger tile than BM32
    # atomic + coalesced flat write, smaller than BM128 (more tiles at moderate M).
    out.append(
        (
            "BM64+NONATOMIC_CSHUFFLE",
            g1_kernel_name(shape, 64, ""),
            g2_kernel_name(shape, 64, "NONATOMIC_CSHUFFLE"),
        )
    )
    return out


# fly ????:??????? fused_moe ? _INDEX_COLS / get_2stage_cfgs ? keys ???
_FLY_KEY_COLS = [
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
]
_FLY_FIXED = {
    "act_type": "ActivationType.Silu",  # nominal only; unused by the lookup (act ignored)
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float4_e2m1fn_x2",
    "q_dtype_w": "torch.float4_e2m1fn_x2",
    "q_type": "QuantType.per_1x32",
    "use_g1u1": 1,
    "doweight_stage1": 0,
}
# act_type is intentionally excluded: at runtime fused_moe's act-specific (primary)
# lookup is dead (raw ActivationType enum vs CSV string key in the `keys` tuple), so
# it always resolves via the __ignore__ fallback = first untagged row (file order)
# matching the non-act columns. We mirror exactly that. (The tier fallback in
# get_2stage_cfgs only fires for M>=32768 (_PADDED_M_TIERS), outside this bench's range.)
_FLY_MATCH_COLS = [c for c in _FLY_KEY_COLS if c != "act_type"]


def fly_kernel_info(shape, M, csv_path, cu_num):
    """?? fused_moe ??-tag ?????? (tuned: bool, kernelName1, kernelName2)?
    ?? act_type?????????????(????? __ignore__ fallback)?
    tuned=False(??)?? fly ???? heuristics???,?????/???"""
    if not csv_path or not os.path.exists(csv_path):
        return (False, "", "")
    df = pd.read_csv(csv_path)
    if "_tag" not in df.columns:
        return (False, "", "")
    df = df[df["_tag"].fillna("") == ""]
    if len(df) == 0:
        return (False, "", "")
    key = {
        "cu_num": cu_num,
        "token": get_padded_M(M),
        "model_dim": shape.H,
        "inter_dim": shape.INTER,
        "expert": shape.NE,
        "topk": shape.TOPK,
        "dtype": _FLY_FIXED["dtype"],
        "q_dtype_a": _FLY_FIXED["q_dtype_a"],
        "q_dtype_w": _FLY_FIXED["q_dtype_w"],
        "q_type": _FLY_FIXED["q_type"],
        "use_g1u1": _FLY_FIXED["use_g1u1"],
        "doweight_stage1": _FLY_FIXED["doweight_stage1"],
    }
    mask = pd.Series(True, index=df.index)
    for col in _FLY_MATCH_COLS:
        mask &= df[col].astype(str) == str(key[col])
    hit = df[mask]
    if len(hit) > 0:
        row = hit.iloc[0]  # first match in file order (mirrors __ignore__ keep="first")
        return (
            True,
            str(row.get("kernelName1", "")),
            str(row.get("kernelName2", "")),
        )
    return (False, "", "")


def build_weights(shape, device, seed=0):
    torch.manual_seed(seed)
    ne, h, inter = shape.NE, shape.H, shape.INTER
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn((ne, 2 * inter, h), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((ne, h, inter), dtype=dtypes.bf16, device=device) / 10
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)

    # fly: legacy preshuffle (16,16) ??(op_tests/test_moe_2stage.py ??)?
    fly = dict(
        w1=shuffle_weight(w1_qt, layout=(16, 16)),
        w2=shuffle_weight(w2_qt, layout=(16, 16)),
        w1_scale=e8m0_shuffle(w1_scale),
        w2_scale=e8m0_shuffle(w2_scale),
    )

    # mx: mxfp4_moe a16w4 gate/up-interleaved ??,w1 ? shuffle_kind ???
    mx_w1 = shuffle_weight_a16w4(w1_qt, 16, True)
    mx_w1.shuffle_kind = "mxfp4_moe"
    mx = dict(
        w1=mx_w1,
        w2=shuffle_weight_a16w4(w2_qt, 16, False),
        w1_scale=shuffle_scale_a16w4(w1_scale, ne, True),
        w2_scale=shuffle_scale_a16w4(w2_scale, ne, False),
    )
    return fly, mx


def build_inputs(shape, M, device, seed=1):
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


def tensor_hash(*tensors):
    hh = hashlib.sha256()
    for t in tensors:
        b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        hh.update(b)
    return hh.hexdigest()[:16]


def run_fly(shape, M, fly_w, hidden, topk_ids, topk_weight, iters, warmup):
    """legacy 2-stage fly path via fused_moe(? shuffle_kind)??? (us, out)?"""

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


def run_flyg_best(
    shape,
    M,
    mx_w,
    hidden,
    topk_ids,
    topk_weight,
    ref_out,
    iters,
    warmup,
    cos_thresh=0.95,
):
    """?????? flyg ??,????/?????,????? (us, label, out)?
    ?????? None?ref_out=None ??? cosine ???
    cos_thresh ???????????(???????),???? fp4 ??:
    fly/flyg ?? fp4 ???/????,?? cosine ~0.98(mxfp4out ??),??? 0.95?"""
    best = None
    for label, k1, k2 in flyg_variants(shape):

        def fn(_k1=k1, _k2=k2):
            mx_w["w1"].gemm1_backend = "flydsl"
            mx_w["w2"].gemm2_backend = "flydsl"
            return _mxfp4_moe_run(
                hidden,
                mx_w["w1"],
                mx_w["w2"],
                topk_ids,
                topk_weight,
                shape.TOPK,
                kernelName1=_k1,
                kernelName2=_k2,
                w1_scale=mx_w["w1_scale"],
                w2_scale=mx_w["w2_scale"],
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
            )

        try:
            out = fn().clone()
            if ref_out is not None and cosine(out, ref_out) < cos_thresh:
                continue
            _, us = run_perftest(fn, num_warmup=warmup, num_iters=iters)
        except Exception as e:
            if os.environ.get("BENCH_DEBUG"):
                print(f"  skip flyg {label}: {type(e).__name__}: {e}")
            continue
        if best is None or us < best[0]:
            best = (us, label, out)
    return best


def _active_csv():
    return AITER_CONFIGS.AITER_CONFIG_FMOE_FILE


def _short_kernel(name):
    """fly kernelName ??:??????????? '-'?"""
    if not name:
        return "-"
    return name if len(name) <= 40 else "..." + name[-39:]


def _fmt(us):
    return f"{us:.1f}" if us is not None else "FAIL"


def parse_shapes(value):
    if value.strip() == "all":
        return list(SHAPES)
    names = [v.strip() for v in value.split(",") if v.strip()]
    bad = [n for n in names if n not in SHAPES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown shape(s): {bad}; choose from: {', '.join(SHAPES)}, all"
        )
    return names


def run_shape(name, shape, M_list, iters, warmup, want_hash, csv_path, cu_num):
    device = torch.device("cuda")
    fly_w, mx_w = build_weights(shape, device)
    print(
        f"\nshape={name} NE={shape.NE} H={shape.H} "
        f"INTER={shape.INTER} TOPK={shape.TOPK}"
    )
    if want_hash:
        print(
            f"  weights hash: mx_w1={tensor_hash(mx_w['w1'])} "
            f"fly_w1={tensor_hash(fly_w['w1'])}"
        )
    header = (
        f"{'M':>6} | {'flyg us':>10} | {'fly us':>10} | {'fly/flyg':>9} | "
        f"{'cosine':>8} | {'flyg variant':>18} | fly kernel(tuned?)"
    )
    print(header)
    print("-" * len(header))
    for M in M_list:
        hidden, topk_ids, topk_weight = build_inputs(shape, M, device)
        if want_hash:
            print(
                f"  [M={M}] input hash: hidden={tensor_hash(hidden)} "
                f"ids={tensor_hash(topk_ids)}"
            )
        # fly first (its output is the cosine reference for flyg).
        try:
            fly_us, fly_out = run_fly(
                shape, M, fly_w, hidden, topk_ids, topk_weight, iters, warmup
            )
        except Exception as e:
            fly_us, fly_out = None, None
            print(f"{M:>6} | fly FAIL: {type(e).__name__}: {e}")
        tuned, _fk1, fk2 = fly_kernel_info(shape, M, csv_path, cu_num)
        fly_tag = f"{_short_kernel(fk2)} (tuned)" if tuned else "heuristics (untuned)"
        best = run_flyg_best(
            shape, M, mx_w, hidden, topk_ids, topk_weight, fly_out, iters, warmup
        )
        if best is None:
            print(f"{M:>6} | flyg FAIL (all variants) | fly={_fmt(fly_us)} | {fly_tag}")
        else:
            flyg_us, label, flyg_out = best
            cos = cosine(flyg_out, fly_out) if fly_out is not None else float("nan")
            ratio = (fly_us / flyg_us) if fly_us else float("nan")
            print(
                f"{M:>6} | {flyg_us:>10.1f} | {_fmt(fly_us):>10} | "
                f"{ratio:>8.2f}x | {cos:>8.4f} | {label:>18} | {fly_tag}",
                flush=True,
            )
        del hidden, topk_ids, topk_weight
        torch.cuda.empty_cache()
    del fly_w, mx_w
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--M-list", default="4,8,16,32,64,128,256")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default="all",
        help="comma-separated shape names or 'all'; choices: " + ", ".join(SHAPES),
    )
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()
    shapes = args.shapes if isinstance(args.shapes, list) else parse_shapes(args.shapes)
    M_list = [int(x) for x in args.M_list.split(",")]

    cu_num = get_cu_num()
    csv_path = _active_csv()
    print(f"GPU: {torch.cuda.get_device_name(0)}  cu_num={cu_num}")
    print(f"active fly CSV (AITER_CONFIG_FMOE): {csv_path}")
    print(f"shapes: {', '.join(shapes)}")

    for name in shapes:
        try:
            run_shape(
                name,
                SHAPES[name],
                M_list,
                args.iters,
                args.warmup,
                args.hash,
                csv_path,
                cu_num,
            )
        except Exception as e:
            print(f"shape={name} FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
