"""Stage2 FlyDSL a4w4 benchmark: baseline vs swap_ab.

Reads cases from dsv3_fp4_tuned_fmoe.csv, benchmarks FlyDSL stage2
with the current CShuffle epilogue (baseline) and optionally with
swap_ab=True (direct epilog, no CShuffle).

Usage:
    python bench_stage2.py
    python bench_stage2.py --num-iters 50
    python bench_stage2.py --swap-ab          # also benchmark swap_ab variant
    python bench_stage2.py -t 16 64 256       # specific token counts
"""

import argparse
import csv
import os
import sys

import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.test_common import run_perftest, checkAllclose
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
from aiter.ops.moe_op import ck_moe_stage2_fwd

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_A = dtypes.fp4x2
Q_DTYPE_W = dtypes.fp4x2
TORCH_QUANT = aiter.get_torch_quant(Q_TYPE)

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "aiter", "configs", "model_configs", "dsv3_fp4_tuned_fmoe.csv",
)


def load_cases(csv_path, token_filter=None):
    cases = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("_tag", "").strip()
            if tag == "flydsl_fallback":
                continue
            token = int(row["token"])
            if token_filter and token not in token_filter:
                continue
            cases.append(dict(
                token=token,
                model_dim=int(row["model_dim"]),
                inter_dim=int(row["inter_dim"]),
                expert=int(row["expert"]),
                topk=int(row["topk"]),
                block_m=int(row["block_m"]),
            ))
    return cases


def setup_stage2_data(token, model_dim, inter_dim, E, topk, block_m,
                      dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10

    score = torch.zeros((token, E), dtype=dtype)
    start_col = 0
    end_col = topk
    for token_id in range(token):
        score[token_id, start_col:end_col] = 1.0
        start_col = end_col % E
        end_col = start_col + topk
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = TORCH_QUANT(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    a1_qt, a1_scale = TORCH_QUANT(inp, quant_dtype=Q_DTYPE_A)

    from aiter.fused_moe import torch_moe_stage1
    ref1 = torch_moe_stage1(
        a1_qt,
        w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2),
        w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2),
        topk_weights, topk_ids,
        dtype=dtype, activation=ActivationType.Silu,
        quant_type=Q_TYPE, a1_scale=a1_scale, w1_scale=w1_scale,
        doweight=False,
    )

    a2_qt, a2_scale = TORCH_QUANT(ref1, quant_dtype=Q_DTYPE_A)
    a2_qt = a2_qt.view(token, topk, -1)

    ref2 = torch_moe_stage2(
        a2_qt,
        w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2),
        w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2),
        topk_weights, topk_ids,
        dtype=dtype, quant_type=Q_TYPE,
        w2_scale=w2_scale, a2_scale=a2_scale,
        doweight=True,
    )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_m)

    needed = sorted_expert_ids.shape[0] * block_m
    if sorted_ids.shape[0] < needed:
        pad = torch.full((needed - sorted_ids.shape[0],), token,
                         dtype=sorted_ids.dtype, device=sorted_ids.device)
        sorted_ids = torch.cat([sorted_ids, pad])
        sorted_weights = torch.cat([sorted_weights,
            torch.zeros(pad.shape[0], dtype=sorted_weights.dtype,
                        device=sorted_weights.device)])

    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)

    w2_qt_ck = shuffle_weight(w2_qt, (16, 16))
    w2_qt_ck.is_shuffled = True
    w2_scale_ck = e8m0_shuffle(w2_scale)
    w1_dummy = torch.empty((E, inter_dim * 2, model_dim // 2),
                           dtype=w2_qt.dtype, device=w2_qt.device)

    a2_scale_sort = moe_mxfp4_sort(
        a2_scale[:token * topk, :].view(token, topk, -1),
        sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token, block_size=block_m,
    )

    return dict(
        ref_stage2=ref2,
        a2_qt=a2_qt, a2_scale_sort=a2_scale_sort,
        w2_qt_shuf=w2_qt_shuf, w2_scale_shuf=w2_scale_shuf,
        w2_qt_ck=w2_qt_ck, w2_scale_ck=w2_scale_ck, w1_dummy=w1_dummy,
        sorted_ids=sorted_ids, sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids,
        topk_weights=topk_weights, topk_ids=topk_ids,
        token=token, model_dim=model_dim, inter_dim=inter_dim, E=E, topk=topk,
    )


def call_stage2(d, topk, block_m, tile_n=256, mode="atomic", swap_ab=False,
                waves_per_eu=0, use_async_copy=False, total_threads=256):
    return flydsl_moe_stage2(
        inter_states=d["a2_qt"],
        w2=d["w2_qt_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk,
        tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        mode=mode,
        w2_scale=d["w2_scale_shuf"],
        a2_scale=d["a2_scale_sort"],
        sorted_weights=d["sorted_weights"],
        swap_ab=swap_ab,
        waves_per_eu=waves_per_eu,
        use_async_copy=use_async_copy,
        total_threads=total_threads,
    )


def fn_stage2(a2_qt, w2_qt_shuf, sorted_ids, sorted_expert_ids,
              num_valid_ids, w2_scale_shuf, a2_scale_sort,
              sorted_weights, topk, block_m, tile_n=256,
              mode="atomic", swap_ab=False, waves_per_eu=0,
              use_async_copy=False, total_threads=256):
    return flydsl_moe_stage2(
        inter_states=a2_qt,
        w2=w2_qt_shuf,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        mode=mode,
        w2_scale=w2_scale_shuf,
        a2_scale=a2_scale_sort,
        sorted_weights=sorted_weights,
        swap_ab=swap_ab,
        waves_per_eu=waves_per_eu,
        use_async_copy=use_async_copy,
        total_threads=total_threads,
    )


def call_ck_stage2(d, topk, block_m):
    token_num = d["a2_qt"].shape[0]
    model_dim = d["model_dim"]
    out = torch.zeros((token_num, model_dim), dtype=torch.bfloat16, device="cuda")
    ck_moe_stage2_fwd(
        d["a2_qt"], d["w1_dummy"], d["w2_qt_ck"],
        d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
        out, topk, "",
        d["w2_scale_ck"], d["a2_scale_sort"],
        block_m, d["sorted_weights"],
        Q_TYPE, ActivationType.Silu,
    )
    return out


def fn_ck_stage2(a2_qt, w1_dummy, w2_qt_ck, sorted_ids, sorted_expert_ids,
                 num_valid_ids, w2_scale_ck, a2_scale_sort,
                 sorted_weights, topk, block_m):
    token_num = a2_qt.shape[0]
    model_dim = w2_qt_ck.shape[1]
    out = torch.zeros((token_num, model_dim), dtype=torch.bfloat16,
                      device=a2_qt.device)
    ck_moe_stage2_fwd(
        a2_qt, w1_dummy, w2_qt_ck,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk, "",
        w2_scale_ck, a2_scale_sort,
        block_m, sorted_weights,
        Q_TYPE, ActivationType.Silu,
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--swap-ab", action="store_true",
                        help="Also benchmark swap_ab variant")
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=None,
                        help="Filter to specific token counts")
    parser.add_argument("--tile-n", type=int, nargs="+", default=[128, 256],
                        help="tile_n values to benchmark (default: 128 256)")
    parser.add_argument("--waves-per-eu", type=int, default=0,
                        help="Pad LDS to limit occupancy (0=disabled, e.g. 3)")
    parser.add_argument("--use-async-copy", action="store_true",
                        help="Use async DMA for global-to-LDS copy + full A prefetch")
    parser.add_argument("--total-threads", type=int, nargs="+", default=[256],
                        help="Block sizes to benchmark (default: 256). Must be multiples of 64.")
    parser.add_argument("--ck", action="store_true",
                        help="Also benchmark CK baseline for comparison")
    args = parser.parse_args()

    cases = load_cases(args.csv, token_filter=args.tokens)
    ni, nw = args.num_iters, args.num_warmup
    wpe = args.waves_per_eu
    async_copy = args.use_async_copy
    results = []

    print(f"\nBenchmarking stage2 FP4  (iters={ni}, warmup={nw}, waves_per_eu={wpe}, async_copy={async_copy})")
    print("=" * 100)

    for c in cases:
        token = c["token"]
        block_m = c["block_m"]
        topk = c["topk"]
        model_dim = c["model_dim"]
        inter_dim = c["inter_dim"]

        torch.cuda.empty_cache()
        print(f"\n  token={token:>5d}  block_m={block_m}  "
              f"model_dim={model_dim} inter_dim={inter_dim}")

        d = setup_stage2_data(token, model_dim, inter_dim,
                              c["expert"], topk, block_m)

        ref = d["ref_stage2"]

        for tn in args.tile_n:
          for tw in args.total_threads:
            tag = f"tn{tn}_tw{tw}"

            # Baseline (swap_ab=False)
            fly_out = call_stage2(d, topk, block_m, tile_n=tn,
                                  waves_per_eu=wpe,
                                  use_async_copy=async_copy,
                                  total_threads=tw)
            torch.cuda.synchronize()

            err_ratio = checkAllclose(
                ref, fly_out,
                rtol=args.rtol, atol=args.atol,
                msg=f"    [t={token},{tag}] ",
            )
            prec_ok = "PASS" if err_ratio == 0 else (
                "WARN" if err_ratio <= 0.05 else "FAIL")

            common = (
                d["a2_qt"], d["w2_qt_shuf"],
                d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
                d["w2_scale_shuf"], d["a2_scale_sort"],
                d["sorted_weights"],
                topk, block_m, tn, "atomic", False, wpe, async_copy, tw,
            )

            print(f"    perf ({tag})...", end="", flush=True)
            _, us_val = run_perftest(
                fn_stage2, *common,
                num_iters=ni, num_warmup=nw,
            )
            print(f"  {us_val:.2f} us  [{prec_ok}]")

            r = dict(
                token=token, block_m=block_m, tile_n=tn, total_threads=tw,
                model_dim=model_dim, inter_dim=inter_dim,
                us_baseline=us_val,
                err_baseline=err_ratio, prec_baseline=prec_ok,
                us_swap=0.0, err_swap=0.0, prec_swap="N/A",
                us_ck=0.0, err_ck=0.0, prec_ck="N/A",
            )

            # swap_ab variant
            if args.swap_ab:
                fly_out_swap = call_stage2(d, topk, block_m, tile_n=tn,
                                           swap_ab=True,
                                           waves_per_eu=wpe,
                                           use_async_copy=async_copy,
                                           total_threads=tw)
                torch.cuda.synchronize()

                err_swap = checkAllclose(
                    ref, fly_out_swap,
                    rtol=args.rtol, atol=args.atol,
                    msg=f"    [t={token},{tag},swap] ",
                )
                prec_swap = "PASS" if err_swap == 0 else (
                    "WARN" if err_swap <= 0.05 else "FAIL")

                swap_common = (
                    d["a2_qt"], d["w2_qt_shuf"],
                    d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
                    d["w2_scale_shuf"], d["a2_scale_sort"],
                    d["sorted_weights"],
                    topk, block_m, tn, "atomic", True, wpe, async_copy, tw,
                )
                print(f"    perf ({tag},swap)...", end="", flush=True)
                _, us_swap = run_perftest(
                    fn_stage2, *swap_common,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_swap:.2f} us  [{prec_swap}]")
                r.update(us_swap=us_swap, err_swap=err_swap, prec_swap=prec_swap)

            # CK baseline (only once per tile_n, not per total_threads)
            if args.ck and tw == args.total_threads[0]:
                ck_out = call_ck_stage2(d, topk, block_m)
                torch.cuda.synchronize()

                err_ck = checkAllclose(
                    ref, ck_out,
                    rtol=args.rtol, atol=args.atol,
                    msg=f"    [t={token},{tag},ck] ",
                )
                prec_ck = "PASS" if err_ck == 0 else (
                    "WARN" if err_ck <= 0.05 else "FAIL")

                ck_common = (
                    d["a2_qt"], d["w1_dummy"], d["w2_qt_ck"],
                    d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
                    d["w2_scale_ck"], d["a2_scale_sort"],
                    d["sorted_weights"],
                    topk, block_m,
                )
                print(f"    perf ({tag},ck)...", end="", flush=True)
                _, us_ck = run_perftest(
                    fn_ck_stage2, *ck_common,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_ck:.2f} us  [{prec_ck}]")
                r.update(us_ck=us_ck, err_ck=err_ck, prec_ck=prec_ck)

            results.append(r)

    # Summary
    print(f"\n{'='*150}")
    print("SUMMARY: Stage2 FP4 Results")
    print(f"{'='*150}")
    has_swap = args.swap_ab
    has_ck = args.ck

    cols = [f"{'token':>5s}", f"{'bm':>3s}", f"{'tn':>4s}", f"{'tw':>4s}",
            f"{'flydsl':>10s}", f"{'prec':>4s}"]
    if has_swap:
        cols += [f"{'swap_ab':>10s}", f"{'prec':>4s}", f"{'spd_swap':>8s}"]
    if has_ck:
        cols += [f"{'ck':>10s}", f"{'prec':>4s}", f"{'spd_ck':>8s}"]
    print("  " + "  ".join(cols))
    print("  " + "  ".join("-" * len(c.strip()) for c in cols))

    for r in results:
        parts = [f"{r['token']:>5d}", f"{r['block_m']:>3d}", f"{r['tile_n']:>4d}",
                 f"{r['total_threads']:>4d}",
                 f"{r['us_baseline']:>10.2f}", f"{r['prec_baseline']:>4s}"]
        if has_swap:
            spd = r['us_baseline'] / r['us_swap'] if r['us_swap'] > 0 else 0.0
            parts += [f"{r['us_swap']:>10.2f}", f"{r['prec_swap']:>4s}",
                      f"{spd:>7.2f}x"]
        if has_ck:
            spd_ck = r['us_baseline'] / r['us_ck'] if r['us_ck'] > 0 else 0.0
            parts += [f"{r['us_ck']:>10.2f}", f"{r['prec_ck']:>4s}",
                      f"{spd_ck:>7.2f}x"]
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
