# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Canonical perf comparison for the fused rotation + per-1x32 MXFP4 quant +
MoE-scale-sort path. THIS is the script to use for all rotation-quant timing.

Every provider produces the SAME end product (the MoE-sorted, MXFP4-shuffled
E8M0 scale + packed fp4x2 output), so latencies are directly comparable:

  * flydsl_quantonly        flydsl_per_1x32_fp4_quant      + mxfp4_moe_sort_fwd
  * flydsl_single_rotation  fused single-(32,32)-R rotation+quant+sort, using
                            its BEST (persist_m, xcd_remap) combo, auto-swept
                            per token count; best_pm shows e.g. "pm16+x"
                            (+x => xcd_remap on, 8-XCD L2 regrouping)
  * hip                     per_1x32_f4_quant_hip          + mxfp4_moe_sort_fwd
  * triton                  per_1x32_f4_quant_triton       + mxfp4_moe_sort_fwd

Timing uses aiter.test_common.run_perftest (back-to-back iters, no per-call
synchronize) so the absolute us are stable and reproducible -- do NOT use a
per-call torch.cuda.synchronize() harness, it adds a fixed ~35us launch/sync
overhead per call and inflates every number.

R is pre-shuffled into MFMA B-fragment order ONCE, outside the timed region
(the dispatch default-preshuffles unmarked R on EVERY call -- a host-side
torch permute that otherwise dominates small-token timings).

Usage::

    python op_tests/bench_single_rotation_quant.py
    python op_tests/bench_single_rotation_quant.py -t 1000 8000 40000 -dim 4096
    python op_tests/bench_single_rotation_quant.py --persist-m 1 8 16 32
"""
import argparse

import torch

from aiter import dtypes, mxfp4_moe_sort_fwd
from aiter.test_common import run_perftest
from aiter.ops.flydsl import (
    flydsl_per_1x32_fp4_quant,
    flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort_inplace as rot_quant_sort,
)
from aiter.ops.flydsl.quant_kernels import mxfp4_singrot_R_preshuffle
from aiter.ops.quant import per_1x32_f4_quant_hip, per_1x32_f4_quant_triton

# Reuse the proven harness helpers from the bit-exact test.
from test_flydsl_rotation_quant_sort_fused import (
    G,
    _make_sorted_ids,
    _orthonormal_R,
)

# Default token sweep (1k/2k/4k/8k/16k/32k/40k/64k -- k == 1024, NOT 1000).
_K1024 = 1024
_DEFAULT_TOKENS = [n * _K1024 for n in (1, 2, 4, 8, 16, 32, 40, 64)]
# Tuning grid for the single-rotation provider. The best (min us) config over
# persist_m x blocks_per_wg x waves_per_wg is reported per remap mode, with the
# winning (pm,K,W) annotated. blocks_per_wg must divide scale_N (=cols/32);
# waves_per_wg in {1,2,4} (BLOCK_DIM = W*64 <= 256).
_DEFAULT_PERSIST_M = [1, 2, 4, 8, 16, 32]
_DEFAULT_BLOCKS_PER_WG = [1, 2, 4]
_DEFAULT_WAVES_PER_WG = [1, 2, 4]
_PROVIDERS = [
    "flydsl_quantonly",
    "flydsl_single_rot",       # best persist_m, NO xcd remap
    "flydsl_single_rot_xcd",   # best persist_m, WITH 8-XCD remap
    "hip",
    "triton",
]


# Peak compute (TFLOP/s) by compute dtype, and HBM bandwidth (bytes/s). The
# rotation is a bf16 MFMA (R32x32 applied to every 32-group), so MFU uses the
# bf16 peak; fp8/fp4 peaks are kept for reference / other dtypes.
_PEAK_TFLOPS = {"bfloat16": 1686.0, "float16": 1686.0,
                "fp8": 3567.0, "fp4": 5663.0}
_HBM_BW = 8.0e12  # 8 TB/s


def _mfu_mbu(us, rows, cols, in_bytes, peak_tflops):
    """(MFU%, MBU%) for the fused single-rotation kernel.

    FLOPs: rotating each 32-group by R(32x32) = 32*32 MACs per group, with
    rows*(cols/32) groups -> 64*rows*cols FLOPs (MAC = 2 flops).
    Bytes : read X (rows*cols*in_bytes) + write packed fp4x2 (rows*cols/2) +
            write e8m0 scale (rows*cols/32).
    """
    if us is None:
        return None, None
    t = us * 1e-6
    flops = 64.0 * rows * cols
    byts = rows * cols * in_bytes + rows * (cols // 2) + rows * (cols // G)
    mfu = flops / t / (peak_tflops * 1e12) * 100.0
    mbu = byts / t / _HBM_BW * 100.0
    return mfu, mbu


def _bench(func, num_iters, num_warmup):
    try:
        _, avg = run_perftest(func, num_iters=num_iters, num_warmup=num_warmup)
        return float(avg)
    except Exception as e:  # noqa: BLE001
        print(f"    ! provider failed: {type(e).__name__}: {e}")
        return None


def _bench_one(token_num, topk, cols, dtype, persist_list, k_list, w_list,
               num_iters, num_warmup, seed=0):
    assert cols % G == 0
    scale_n = cols // G

    sorted_ids, num_valid_ids, rows = _make_sorted_ids(token_num, topk, seed)
    sorted_pad = sorted_ids.shape[0]

    x = (
        torch.randn(
            rows, cols,
            generator=torch.Generator(device="cpu").manual_seed(seed + 1),
            device="cpu",
        )
        .to(dtype)
        .to("cuda")
        .contiguous()
    )

    R_single = _orthonormal_R(1, dtype, seed + 3)[0].contiguous()  # (32,32)
    R_single_in = mxfp4_singrot_R_preshuffle(R_single)             # mark once

    def _sort(scale_plain):
        return mxfp4_moe_sort_fwd(
            scale_plain.view(dtypes.fp8_e8m0),
            sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=token_num, cols=cols,
        )

    results = {}
    best_pm = {}
    metrics = {}
    in_bytes = torch.empty(0, dtype=dtype).element_size()
    peak_tf = _PEAK_TFLOPS.get(str(dtype).split(".")[-1], 1686.0)

    # flydsl_quantonly + sort -------------------------------------------------
    o1 = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    s1 = torch.empty(rows, scale_n, dtype=torch.uint8, device="cuda")

    def _quantonly():
        flydsl_per_1x32_fp4_quant(o1, x, s1)
        return _sort(s1)

    results["flydsl_quantonly"] = _bench(_quantonly, num_iters, num_warmup)

    # flydsl_single_rotation (best persist_m / K / W per remap mode) ----------
    o2 = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    s2 = torch.empty(sorted_pad, scale_n, dtype=dtypes.fp8_e8m0, device="cuda")
    valid_k = [k for k in k_list if scale_n % k == 0]

    def _call(o, s, pm, k, w, rmp):
        rot_quant_sort(
            o, x, R_single_in, s, sorted_ids, num_valid_ids, token_num,
            persist_m=pm, blocks_per_wg=k, waves_per_wg=w, xcd_remap=rmp,
        )

    # Golden = baseline config (pm1/K1/W1, no remap); itself validated against
    # the chained reference in test_flydsl_rotation_quant_sort_fused.py. Every
    # tuned best config must reproduce it byte-for-byte (the persist_m / K / W /
    # xcd_remap knobs are pure scheduling -- they must not change the output).
    o_ref = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    s_ref = torch.zeros(sorted_pad, scale_n, dtype=dtypes.fp8_e8m0,
                        device="cuda")
    _call(o_ref, s_ref, 1, 1, 1, False)
    torch.cuda.synchronize()

    for rmp, key in ((False, "flydsl_single_rot"),
                     (True, "flydsl_single_rot_xcd")):
        best_us, best_cfg = None, None
        for pm in persist_list:
            for k in valid_k:
                for w in w_list:
                    us = _bench(
                        lambda pm=pm, k=k, w=w, rmp=rmp: _call(
                            o2, s2, pm, k, w, rmp),
                        num_iters, num_warmup,
                    )
                    if us is not None and (best_us is None or us < best_us):
                        best_us, best_cfg = us, (pm, k, w)
        results[key] = best_us
        metrics[key] = _mfu_mbu(best_us, rows, cols, in_bytes, peak_tf)
        if best_cfg is None:
            best_pm[key] = ("N/A", None)
            continue
        # Verify the winning config is bit-exact vs the golden baseline.
        pm, k, w = best_cfg
        o_chk = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
        s_chk = torch.zeros(sorted_pad, scale_n, dtype=dtypes.fp8_e8m0,
                            device="cuda")
        _call(o_chk, s_chk, pm, k, w, rmp)
        torch.cuda.synchronize()
        ok = (torch.equal(o_chk, o_ref)
              and torch.equal(s_chk.view(torch.uint8),
                              s_ref.view(torch.uint8)))
        best_pm[key] = (f"pm{pm}/K{k}/W{w}", ok)

    # hip + sort --------------------------------------------------------------
    def _hip():
        _, sc = per_1x32_f4_quant_hip(x)
        return _sort(sc)

    results["hip"] = _bench(_hip, num_iters, num_warmup)

    # triton + sort -----------------------------------------------------------
    st = torch.empty(rows, scale_n, dtype=torch.uint8, device="cuda")

    def _triton():
        per_1x32_f4_quant_triton(x)
        return _sort(st)

    results["triton"] = _bench(_triton, num_iters, num_warmup)

    return results, best_pm, metrics


def _fmt(us):
    return "    N/A   " if us is None else f"{us:9.2f}"


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Canonical MXFP4 rotation+quant+MoE-sort perf comparison.",
)
parser.add_argument("-t", "--token", type=int, nargs="*",
                    default=_DEFAULT_TOKENS,
                    help="token_num values (default 1k..64k incl. 40k; "
                         "k == 1024, e.g. 64k = 65536).")
parser.add_argument("-dim", "--dim", type=int, nargs="*", default=[4096],
                    help="cols (model_dim) values.")
parser.add_argument("-topk", "--topk", type=int, nargs="*", default=[1],
                    help="topk values.")
parser.add_argument("-d", "--dtype", type=dtypes.str2Dtype,
                    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
                    nargs="*", default=[dtypes.d_dtypes["bf16"]],
                    metavar="{bf16, fp16}", help="activation/rotation dtype.")
parser.add_argument("--persist-m", type=int, nargs="*",
                    default=_DEFAULT_PERSIST_M,
                    help="persist_m candidates to sweep for single_rotation.")
parser.add_argument("--blocks-per-wg", type=int, nargs="*",
                    default=_DEFAULT_BLOCKS_PER_WG,
                    help="blocks_per_wg (K) candidates; must divide cols/32.")
parser.add_argument("--waves-per-wg", type=int, nargs="*",
                    default=_DEFAULT_WAVES_PER_WG,
                    help="waves_per_wg (W) candidates; W*64 <= 256.")
parser.add_argument("--num-iters", type=int, default=101,
                    help="timed iters per provider.")
parser.add_argument("--num-warmup", type=int, default=5,
                    help="warmup iters per provider.")


def main():
    args = parser.parse_args()
    header = (
        f"{'token':>6} {'topk':>4} {'cols':>5} {'dtype':>5} | "
        + " ".join(f"{p:>24}" for p in _PROVIDERS)
        + f"  {'best norm (MFU/MBU,chk)':>32} {'best xcd (MFU/MBU,chk)':>32}"
    )
    for dtype in args.dtype:
        for cols in args.dim:
            for topk in args.topk:
                print("\n" + header)
                print("-" * len(header))
                for t in args.token:
                    res, best_pm, metrics = _bench_one(
                        t, topk, cols, dtype, args.persist_m,
                        args.blocks_per_wg, args.waves_per_wg,
                        args.num_iters, args.num_warmup,
                    )
                    dname = str(dtype).split(".")[-1]

                    def _ann(key):
                        mfu, mbu = metrics[key]
                        cfg, ok = best_pm[key]
                        chk = "" if ok is None else (" ok" if ok else " FAIL")
                        if mfu is None:
                            return f"{cfg}{chk}"
                        return f"{cfg} {mfu:4.1f}/{mbu:4.1f}%{chk}"

                    row = (
                        f"{t:>6} {topk:>4} {cols:>5} {dname:>5} | "
                        + " ".join(f"{_fmt(res[p]):>24}" for p in _PROVIDERS)
                        + f"  {_ann('flydsl_single_rot'):>32}"
                        + f" {_ann('flydsl_single_rot_xcd'):>32}"
                    )
                    print(row)
    print("\n(values are mean us/iter; lower is better; all produce the "
          "MoE-sorted scale. flydsl_single_rot = no XCD remap, "
          "flydsl_single_rot_xcd = 8-XCD L2 regroup; each reports its own "
          "best-of-sweep config as pmP/Kk/Ww over persist_m x blocks_per_wg "
          "x waves_per_wg. MFU vs bf16 peak (1686 TF, rotation = 64*rows*cols "
          "flops); MBU vs 8 TB/s (read X bf16 + write fp4x2 + e8m0 scale). "
          "'chk' = best config is bit-exact vs the pm1/K1/W1 golden baseline.)")


if __name__ == "__main__":
    main()
