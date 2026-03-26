"""Stage1 async-copy vs sync-copy vs CK baseline benchmark.

Reads cases from dsv3_fp4_tuned_moe.csv (first 16 non-fallback rows),
benchmarks FlyDSL stage1 with sync and async copy, and compares to CK times.

Usage:
    python bench_stage1_async.py
    python bench_stage1_async.py --num-iters 50
"""

import argparse
import csv
import os
import sys

import torch
import aiter
from aiter import dtypes, QuantType
from aiter.test_common import run_perftest, checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, cktile_moe_stage1, get_ksplit
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
from aiter import ActivationType
from aiter.ops.moe_op import ck_moe_stage1_fwd

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_A = dtypes.fp4x2
TORCH_QUANT = aiter.get_torch_quant(Q_TYPE)

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "aiter", "configs", "model_configs", "dsv3_fp4_tuned_flydsl.csv",
)


def load_cases(csv_path):
    cases = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("_tag", "").strip()
            if tag == "flydsl_fallback":
                continue
            cases.append(dict(
                token=int(row["token"]),
                model_dim=int(row["model_dim"]),
                inter_dim=int(row["inter_dim"]),
                expert=int(row["expert"]),
                topk=int(row["topk"]),
                block_m=int(row["block_m"]),
                ck_us1=float(row["us1"]),
            ))
    return cases


def setup_data(token, model_dim, inter_dim, E, topk, block_m,
               dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    # score = torch.randn((token, E), dtype=dtype)
    # score = torch.randn((token, E), dtype=dtype)
    # force expert balanced
    score = torch.zeros((token, E), dtype=dtype)
    start_col = 0
    end_col = topk
    for token_id in range(token):
        score[token_id, start_col:end_col] = 1.0
        start_col = end_col % E
        end_col = start_col + topk
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_A)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    a1_qt, a1_scale = TORCH_QUANT(inp, quant_dtype=Q_DTYPE_A)

    sort_block_m = max(32, block_m)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, sort_block_m)

    # Pad sorted_ids so kernel reads don't go OOB
    needed = sorted_expert_ids.shape[0] * sort_block_m
    if sorted_ids.shape[0] < needed:
        pad = torch.full((needed - sorted_ids.shape[0],), token,
                         dtype=sorted_ids.dtype, device=sorted_ids.device)
        sorted_ids = torch.cat([sorted_ids, pad])
        sorted_weights = torch.cat([sorted_weights,
            torch.zeros(pad.shape[0], dtype=sorted_weights.dtype, device=sorted_weights.device)])

    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale)

    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token, block_size=max(32, block_m),
    )

    w2_dummy = torch.empty((E, model_dim // 2, inter_dim), dtype=w1_qt.dtype,
                           device=w1_qt.device)
    w2_ref = torch.empty((E, model_dim, inter_dim // 2), dtype=w1_qt.dtype,
                          device=w1_qt.device)

    gx = inter_dim // 128  # actual gx = _n_in // 2 // tile_n = inter_dim // tile_n
    _sort_bm = max(32, block_m)
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = min(token * topk * _sort_bm,
                      sorted_ids.shape[0]) // _sort_bm
    gy = min(_dense_blks, _all_blks)
    nv = num_valid_ids.cpu().tolist()
    valid_blks = nv[0] // block_m
    total_wg = gx * gy
    print(f"    grid=({gx},{gy},1) wg={total_wg} valid_blks={valid_blks} "
          f"waves={(total_wg+255)//256} (alloc_blks={_all_blks})")

    return dict(
        inp=inp,
        a1_qt=a1_qt, a1_scale=a1_scale,
        w1_qt=w1_qt, w1_scale=w1_scale,
        w1_qt_shuf=w1_qt_shuf,
        w1_scale_shuf=w1_scale_shuf, a1_scale_sort=a1_scale_sort,
        sorted_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids, w2_dummy=w2_dummy, w2_ref=w2_ref,
        topk_weights=topk_weights, topk_ids=topk_ids,
    )


def call_stage1(d, topk, block_m, use_async, k_batch=1, tile_n=128,
                gate_only=False):
    return flydsl_moe_stage1(
        a=d["a1_qt"], w1=d["w1_qt_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=d["w1_scale_shuf"], a1_scale=d["a1_scale_sort"],
        use_async_copy=use_async,
        k_batch=k_batch,
        gate_only=gate_only,
    )


def fn_stage1(a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids,
              num_valid_ids, w1_scale_shuf, a1_scale_sort,
              topk, block_m, use_async, k_batch=1, tile_n=128,
              gate_only=False):
    return flydsl_moe_stage1(
        a=a1_qt, w1=w1_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=w1_scale_shuf, a1_scale=a1_scale_sort,
        use_async_copy=use_async,
        k_batch=k_batch,
        gate_only=gate_only,
    )


def call_ck_stage1(d, topk, block_m):
    token_num = d["a1_qt"].shape[0]
    inter_dim_x2 = d["w1_qt_shuf"].shape[1]
    out = torch.empty((token_num, topk, inter_dim_x2),
                       dtype=torch.bfloat16, device="cuda")
    ck_moe_stage1_fwd(
        d["a1_qt"], d["w1_qt_shuf"], d["w2_dummy"],
        d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
        out, topk, "",
        d["w1_scale_shuf"], d["a1_scale_sort"],
        block_m, None,
        Q_TYPE, ActivationType.Silu,
    )
    return out


def fn_ck_stage1(a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids,
                 num_valid_ids, w1_scale_shuf, a1_scale_sort,
                 topk, block_m):
    token_num = a1_qt.shape[0]
    E = w1_qt_shuf.shape[0]
    inter_dim_x2 = w1_qt_shuf.shape[1]
    model_dim = a1_qt.shape[1] * 2  # fp4x2 packed
    w2_dummy = torch.empty((E, model_dim // 2, inter_dim_x2 // 2),
                           dtype=w1_qt_shuf.dtype, device=a1_qt.device)
    out = torch.empty((token_num, topk, inter_dim_x2),
                       dtype=torch.bfloat16, device=a1_qt.device)
    ck_moe_stage1_fwd(
        a1_qt, w1_qt_shuf, w2_dummy,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk, "",
        w1_scale_shuf, a1_scale_sort,
        block_m, None,
        Q_TYPE, ActivationType.Silu,
    )
    return out


def fn_ck_stage1_a16w4(inp, w1_qt_shuf, sorted_ids, sorted_expert_ids,
                       num_valid_ids, w1_scale_shuf,
                       topk, block_m, split_k=4):
    """CK Tile stage1 via a16w4 path: bf16 activation, fp4 weight, split-K.

    Uses the same code path as fused_moe_2stages line 1185:
      q_dtype_a==fp4x2 + ksplit>1 + is_shuffled -> cktile_moe_stage1
    Weight/scale use CK preshuffle format (shuffle_weight + e8m0_shuffle).
    """
    E = w1_qt_shuf.shape[0]
    inter_dim_x2 = w1_qt_shuf.shape[1]
    model_dim = inp.shape[1]
    w2_dummy = torch.empty((E, model_dim // 2, inter_dim_x2 // 2),
                           dtype=w1_qt_shuf.dtype, device=inp.device)
    out = torch.empty((inp.shape[0], topk, inter_dim_x2),
                       dtype=torch.bfloat16, device=inp.device)
    return cktile_moe_stage1(
        inp, w1_qt_shuf, w2_dummy,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk, block_m,
        a1_scale=None,
        w1_scale=w1_scale_shuf.view(dtypes.fp8_e8m0),
        activation=ActivationType.Silu,
        split_k=split_k,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--launch-test", action="store_true",
                        help="Run launch overhead measurement only")
    args = parser.parse_args()

    if args.launch_test:
        measure_launch_overhead()
        return

    cases = load_cases(args.csv)
    ni, nw = args.num_iters, args.num_warmup
    results = []

    # SPLITK_VALUES = [1, 2, 4]

    print(f"\nBenchmarking split-K  (iters={ni}, warmup={nw})")
    print("=" * 100)

    for c in cases:
        token = c["token"]
        # if token != 1024:
        #     continue
        if token < 16:
            SPLITK_VALUES = [1, 2, 7]
        else:
            SPLITK_VALUES = [1]

        block_m = c["block_m"]
        block_m = 16
        # if token == 8192: 
        #     block_m = 64 
        topk = c["topk"]
        ck_us = c["ck_us1"]
        model_dim = c["model_dim"]
        inter_dim_cfg = c["inter_dim"]
        # inter_dim_cfg = 512
        c["expert"] = 384
        c["topk"] = 8

        for kb in SPLITK_VALUES:
            k_per_batch = model_dim // kb
            if model_dim % kb != 0 or k_per_batch % 256 != 0:
                continue

            torch.cuda.empty_cache()
            tag = f"sk={kb}" if kb > 1 else "baseline"
            print(f"\n  token={token:>3d}  {tag:>10s}  "
                  f"model_dim={model_dim} inter_dim={inter_dim_cfg} k_batch={kb}")

            d = setup_data(token, model_dim, inter_dim_cfg,
                           c["expert"], topk, block_m)

            # Torch reference
            ref_out = torch_moe_stage1(
                d["a1_qt"],
                d["w1_qt"], d["w2_ref"],
                d["topk_weights"], d["topk_ids"],
                dtype=torch.bfloat16,
                activation=ActivationType.Silu,
                quant_type=Q_TYPE,
                a1_scale=d["a1_scale"],
                w1_scale=d["w1_scale"],
            )
            torch.cuda.synchronize()

            # FlyDSL output (tile_n=128)
            fly_out = call_stage1(d, topk, block_m, use_async=True, k_batch=kb)
            torch.cuda.synchronize()

            err_ratio = checkAllclose(
                ref_out, fly_out,
                rtol=args.rtol, atol=args.atol,
                msg=f"    [t={token},sk={kb},tn=128] ",
            )
            prec_ok = "PASS" if err_ratio == 0 else ("WARN" if err_ratio <= 0.05 else "FAIL")

            # FlyDSL output (tile_n=64)
            fly_out_tn64 = call_stage1(d, topk, block_m, use_async=True, k_batch=kb, tile_n=64)
            torch.cuda.synchronize()

            err_ratio_tn64 = checkAllclose(
                ref_out, fly_out_tn64,
                rtol=args.rtol, atol=args.atol,
                msg=f"    [t={token},sk={kb},tn=64] ",
            )
            prec_ok_tn64 = "PASS" if err_ratio_tn64 == 0 else ("WARN" if err_ratio_tn64 <= 0.05 else "FAIL")

            # FlyDSL output (tile_n=32)
            fly_out_tn32 = call_stage1(d, topk, block_m, use_async=True, k_batch=kb, tile_n=32)
            torch.cuda.synchronize()

            err_ratio_tn32 = checkAllclose(
                ref_out, fly_out_tn32,
                rtol=args.rtol, atol=args.atol,
                msg=f"    [t={token},sk={kb},tn=32] ",
            )
            prec_ok_tn32 = "PASS" if err_ratio_tn32 == 0 else ("WARN" if err_ratio_tn32 <= 0.05 else "FAIL")

            # FlyDSL output (gate_only, only when split-K)
            if kb > 1:
                fly_out_go = call_stage1(d, topk, block_m, use_async=True,
                                         k_batch=kb, tile_n=128, gate_only=True)
                torch.cuda.synchronize()
                err_ratio_go = checkAllclose(
                    ref_out, fly_out_go,
                    rtol=args.rtol, atol=args.atol,
                    msg=f"    [t={token},sk={kb},go] ",
                )
                prec_ok_go = "PASS" if err_ratio_go == 0 else ("WARN" if err_ratio_go <= 0.05 else "FAIL")

                fly_out_go64 = call_stage1(d, topk, block_m, use_async=True,
                                           k_batch=kb, tile_n=64, gate_only=True)
                torch.cuda.synchronize()
                err_ratio_go64 = checkAllclose(
                    ref_out, fly_out_go64,
                    rtol=args.rtol, atol=args.atol,
                    msg=f"    [t={token},sk={kb},go64] ",
                )
                prec_ok_go64 = "PASS" if err_ratio_go64 == 0 else ("WARN" if err_ratio_go64 <= 0.05 else "FAIL")

                fly_out_go32 = call_stage1(d, topk, block_m, use_async=True,
                                           k_batch=kb, tile_n=32, gate_only=True)
                torch.cuda.synchronize()
                err_ratio_go32 = checkAllclose(
                    ref_out, fly_out_go32,
                    rtol=args.rtol, atol=args.atol,
                    msg=f"    [t={token},sk={kb},go32] ",
                )
                prec_ok_go32 = "PASS" if err_ratio_go32 == 0 else ("WARN" if err_ratio_go32 <= 0.05 else "FAIL")
            else:
                err_ratio_go = 0.0
                prec_ok_go = "N/A"
                err_ratio_go64 = 0.0
                prec_ok_go64 = "N/A"
                err_ratio_go32 = 0.0
                prec_ok_go32 = "N/A"

            common = (
                d["a1_qt"], d["w1_qt_shuf"],
                d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
                d["w1_scale_shuf"], d["a1_scale_sort"],
                topk, block_m,
            )

            print("    perf (tn=128)...", end="", flush=True)
            _, us_val = run_perftest(
                fn_stage1, *common, True, kb,
                num_iters=ni, num_warmup=nw,
            )
            print(f"  {us_val:.2f} us")

            print("    perf (tn=64)...", end="", flush=True)
            _, us_tn64 = run_perftest(
                fn_stage1, *common, True, kb, 64,
                num_iters=ni, num_warmup=nw,
            )
            print(f"  {us_tn64:.2f} us")

            print("    perf (tn=32)...", end="", flush=True)
            _, us_tn32 = run_perftest(
                fn_stage1, *common, True, kb, 32,
                num_iters=ni, num_warmup=nw,
            )
            print(f"  {us_tn32:.2f} us")

            if kb > 1:
                print("    perf (gate_only)...", end="", flush=True)
                _, us_go = run_perftest(
                    fn_stage1, *common, True, kb, 128, True,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_go:.2f} us")

                print("    perf (go64)...", end="", flush=True)
                _, us_go64 = run_perftest(
                    fn_stage1, *common, True, kb, 64, True,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_go64:.2f} us")

                print("    perf (go32)...", end="", flush=True)
                _, us_go32 = run_perftest(
                    fn_stage1, *common, True, kb, 32, True,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_go32:.2f} us")
            else:
                us_go = 0.0
                us_go64 = 0.0
                us_go32 = 0.0

            # print("    ck perf...", end="", flush=True)
            # _, us_ck= run_perftest(
            #     fn_ck_stage1, *common,
            #     num_iters=ni, num_warmup=nw,
            # )
            # print(f"  {us_ck:.2f} us")

            # cktile_ksplit = get_ksplit(token, topk, c["expert"],
            #                           inter_dim_cfg, model_dim)
            # cktile_block_m = 16 if token < 2048 else 32 if token < 16384 else 64

            # if cktile_ksplit > 1:
            #     if cktile_block_m != block_m:
            #         ct_sorted_ids, _, ct_sorted_expert_ids, ct_num_valid_ids, _ = \
            #             moe_sorting(d["topk_ids"], d["topk_weights"],
            #                         c["expert"], model_dim, torch.bfloat16,
            #                         cktile_block_m)
            #         ct_needed = ct_sorted_expert_ids.shape[0] * cktile_block_m
            #         if ct_sorted_ids.shape[0] < ct_needed:
            #             ct_pad = torch.full(
            #                 (ct_needed - ct_sorted_ids.shape[0],), token,
            #                 dtype=ct_sorted_ids.dtype, device=ct_sorted_ids.device)
            #             ct_sorted_ids = torch.cat([ct_sorted_ids, ct_pad])
            #     else:
            #         ct_sorted_ids = d["sorted_ids"]
            #         ct_sorted_expert_ids = d["sorted_expert_ids"]
            #         ct_num_valid_ids = d["num_valid_ids"]

            #     common_a16w4 = (
            #         d["inp"], d["w1_qt_shuf"],
            #         ct_sorted_ids, ct_sorted_expert_ids, ct_num_valid_ids,
            #         d["w1_scale_shuf"],
            #         topk, cktile_block_m, cktile_ksplit,
            #     )

            #     print(f"    ck a16w4 perf (sk={cktile_ksplit}, bm={cktile_block_m})...",
            #           end="", flush=True)
            #     _, us_ck_a16w4 = run_perftest(
            #         fn_ck_stage1_a16w4, *common_a16w4,
            #         num_iters=ni, num_warmup=nw,
            #     )
            #     print(f"  {us_ck_a16w4:.2f} us")
            # else:
            #     us_ck_a16w4 = 0.0
            #     print(f"    ck a16w4: skipped (ksplit=0, not enough CU pressure)")

            us_ck = 0.0
            us_ck_a16w4 = 0.0

            gx = inter_dim_cfg * 2 // 128
            _sort_bm = max(32, block_m)
            _all_blks = d["sorted_expert_ids"].shape[0]
            _dense_blks = min(token * topk * _sort_bm,
                              d["sorted_ids"].shape[0]) // _sort_bm
            gy = min(_dense_blks, _all_blks)
            k_iters = k_per_batch // 256
            wg = gx * gy * kb
            results.append(dict(
                token=token, block_m=block_m, k_batch=kb,
                inter_dim=inter_dim_cfg, model_dim=model_dim,
                gx=gx, gy=gy, k_iters=k_iters, wg=wg,
                ck_us=us_ck, us=us_val, us_tn64=us_tn64, us_tn32=us_tn32,
                us_go=us_go, us_go64=us_go64, us_go32=us_go32,
                ck_a16w4_us=us_ck_a16w4,
                err_ratio=err_ratio, prec_ok=prec_ok,
                err_ratio_tn64=err_ratio_tn64, prec_ok_tn64=prec_ok_tn64,
                err_ratio_tn32=err_ratio_tn32, prec_ok_tn32=prec_ok_tn32,
                err_ratio_go=err_ratio_go, prec_ok_go=prec_ok_go,
                err_ratio_go64=err_ratio_go64, prec_ok_go64=prec_ok_go64,
                err_ratio_go32=err_ratio_go32, prec_ok_go32=prec_ok_go32,
            ))

    print(f"\n{'=' * 120}")
    print("SUMMARY: Split-K Results")
    print(f"{'=' * 120}")
    hdr = (f"  {'token':>5s}  {'kb':>3s}  {'K_it':>4s}  "
           f"{'wg':>5s}  {'prec':>4s}  "
           f"{'tn128':>9s}  {'tn64':>9s}  {'p64':>4s}  "
           f"{'tn32':>9s}  {'p32':>4s}  "
           f"{'go128':>9s}  {'pgo':>4s}  "
           f"{'go64':>9s}  {'pg64':>4s}  "
           f"{'go32':>9s}  {'pg32':>4s}")
    print(hdr)
    print(f"  {'-'*5}  {'-'*3}  {'-'*4}  "
           f"{'-'*5}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*4}  "
           f"{'-'*9}  {'-'*4}  "
           f"{'-'*9}  {'-'*4}  "
           f"{'-'*9}  {'-'*4}  "
           f"{'-'*9}  {'-'*4}")
    for r in results:
        print(f"  {r['token']:>5d}  {r['k_batch']:>3d}  "
              f"{r['k_iters']:>4d}  "
              f"{r['wg']:>5d}  {r['prec_ok']:>4s}  "
              f"{r['us']:>9.2f}  {r['us_tn64']:>9.2f}  {r['prec_ok_tn64']:>4s}  "
              f"{r['us_tn32']:>9.2f}  {r['prec_ok_tn32']:>4s}  "
              f"{r['us_go']:>9.2f}  {r['prec_ok_go']:>4s}  "
              f"{r['us_go64']:>9.2f}  {r['prec_ok_go64']:>4s}  "
              f"{r['us_go32']:>9.2f}  {r['prec_ok_go32']:>4s}")


if __name__ == "__main__":
    main()
