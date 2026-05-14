"""Stage2 a16w4 (bf16 activation, fp4 weight, bf16 output) benchmark.

Compares FlyDSL a16w4 stage2 against CK Tile a16w4 baseline.
- FlyDSL: flydsl_moe_stage2 with a_dtype="bf16", b_dtype="fp4"
- CK Tile: cktile_moe_stage2 with bf16 activation + fp4 weight

Usage:
    python bench_stage2_a16w4.py
    python bench_stage2_a16w4.py --num-iters 50
    python bench_stage2_a16w4.py --tokens 1,4,16,64,256
"""

import argparse
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.test_common import run_perftest, checkAllclose
from aiter.fused_moe import (
    fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2,
    cktile_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_W = dtypes.fp4x2
TORCH_QUANT = aiter.get_torch_quant(Q_TYPE)

DEFAULT_CASES = [
    dict(token=1, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=4, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=8, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=16, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=32, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=64, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=128, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=256, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=512, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=1024, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=1, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=4, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=8, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=16, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=32, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=64, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=128, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=256, model_dim=7168, inter_dim=1024, expert=384, topk=8),
]


def setup_data(token, model_dim, inter_dim, E, topk, block_m,
               dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10

    score = torch.zeros((token, E), dtype=dtype)
    start_col, end_col = 0, topk
    for tid in range(token):
        score[tid, start_col:end_col] = 1.0
        start_col = end_col % E
        end_col = start_col + topk
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = TORCH_QUANT(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)

    ref_s1 = torch_moe_stage1(
        inp, w1_qt, w2_qt,
        topk_weights, topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=Q_TYPE,
        a1_scale=None,
        w1_scale=w1_scale,
        doweight=False,   # weights applied in stage2
    )

    sort_block_m = block_m
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, sort_block_m)

    needed = sorted_expert_ids.shape[0] * sort_block_m
    if sorted_ids.shape[0] < needed:
        pad = torch.full((needed - sorted_ids.shape[0],), token,
                         dtype=sorted_ids.dtype, device=sorted_ids.device)
        sorted_ids = torch.cat([sorted_ids, pad])
        sorted_weights = torch.cat([sorted_weights,
            torch.zeros(pad.shape[0], dtype=sorted_weights.dtype,
                        device=sorted_weights.device)])

    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    # e8m0_shuffle is what stage2 a16w4 requires (verified by test_flydsl_moe_a16w4)
    w2_scale_shuf = e8m0_shuffle(w2_scale)

    ref_s2 = torch_moe_stage2(
        ref_s1.to(torch.bfloat16), w1_qt, w2_qt,
        topk_weights, topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        a2_scale=None,
        w2_scale=w2_scale,
    )

    num_active_experts = int(topk_ids.unique().numel())

    return dict(
        ref_s1=ref_s1,
        ref_s2=ref_s2,
        w1_qt=w1_qt, w2_qt=w2_qt,
        w2_scale=w2_scale,
        w2_qt_shuf=w2_qt_shuf,
        w2_scale_shuf=w2_scale_shuf,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_active_experts=num_active_experts,
    )


# ---------- FlyDSL a16w4 stage2 ----------

def call_flydsl_a16w4(d, topk, block_m, tile_n=128, mode="atomic", k_batch=1):
    return flydsl_moe_stage2(
        inter_states=d["ref_s1"],
        w2=d["w2_qt_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk,
        tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="bf16", b_dtype="fp4", out_dtype="bf16",
        mode=mode,
        w2_scale=d["w2_scale_shuf"],
        a2_scale=None,
        sorted_weights=d["sorted_weights"],
        sort_block_m=block_m,
        k_batch=k_batch,
    )


def fn_flydsl_a16w4(ref_s1, w2_qt_shuf, sorted_ids, sorted_expert_ids,
                     num_valid_ids, w2_scale_shuf, sorted_weights,
                     topk, block_m, tile_n=128, mode="atomic", k_batch=1):
    return flydsl_moe_stage2(
        inter_states=ref_s1,
        w2=w2_qt_shuf,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="bf16", b_dtype="fp4", out_dtype="bf16",
        mode=mode,
        w2_scale=w2_scale_shuf,
        a2_scale=None,
        sorted_weights=sorted_weights,
        sort_block_m=block_m,
        k_batch=k_batch,
    )


# ---------- CK Tile a16w4 stage2 ----------

def fn_cktile_a16w4(ref_s1, w1_qt, w2_qt_shuf, sorted_ids, sorted_expert_ids,
                    num_valid_ids, w2_scale_shuf, sorted_weights,
                    topk, block_m):
    E = w2_qt_shuf.shape[0]
    model_dim = w2_qt_shuf.shape[1]
    token_num = ref_s1.shape[0]
    out = torch.zeros(
        (token_num, model_dim),
        dtype=torch.bfloat16, device=ref_s1.device,
    )
    inter_dim = ref_s1.shape[-1]
    w1_dummy = torch.empty(
        (E, inter_dim * 2, model_dim // 2),
        dtype=w2_qt_shuf.dtype, device=ref_s1.device,
    )
    cktile_moe_stage2(
        ref_s1, w1_dummy, w2_qt_shuf,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk,
        w2_scale=w2_scale_shuf.view(dtypes.fp8_e8m0),
        a2_scale=None,
        block_m=block_m,
        activation=ActivationType.Swiglu,
        sorted_weights=sorted_weights,
        zeros_out=True,  # zero out output before atomic accumulation
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlyDSL a16w4 vs CK Tile a16w4 stage2")
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--tokens", type=str, default=None,
                        help="Comma-separated token counts (overrides defaults)")
    parser.add_argument("--tile-n", type=int, nargs="+", default=[128, 256],
                        help="tile_n values to test")
    parser.add_argument("--k-batch", type=int, nargs="+", default=[1],
                        help="Inter-block split-K over inter_dim (grid Z); "
                             "must satisfy inter_dim %% (k_batch*256)==0 and "
                             "(inter_dim//k_batch)//256 even >= 2")
    parser.add_argument("--model-dim", type=int, default=3072)
    parser.add_argument("--inter-dim", type=int, default=3072)
    parser.add_argument("--expert", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available
    if not is_flydsl_available():
        print("[SKIP] FlyDSL not available.")
        sys.exit(0)

    if args.tokens:
        cases = [
            dict(token=int(t.strip()), model_dim=args.model_dim,
                 inter_dim=args.inter_dim, expert=args.expert, topk=args.topk)
            for t in args.tokens.split(",")
        ]
    else:
        cases = DEFAULT_CASES

    ni, nw = args.num_iters, args.num_warmup
    k_batch_vals = args.k_batch
    results = []

    print(f"\nBenchmark: FlyDSL a16w4 vs CK Tile a16w4 stage2")
    print(f"  iters={ni}, warmup={nw}")
    print("=" * 130)

    for c in cases:
        token = c["token"]
        model_dim = c["model_dim"]
        inter_dim = c["inter_dim"]
        E = c["expert"]
        topk = c["topk"]

        fly_block_m = 16
        ck_block_m = 16

        torch.cuda.empty_cache()
        print(f"\n  token={token:>5d}  model_dim={model_dim}  inter_dim={inter_dim}"
              f"  E={E}  topk={topk}  fly_bm={fly_block_m}  ck_bm={ck_block_m}")

        d = setup_data(token, model_dim, inter_dim, E, topk, fly_block_m)

        ref_s2 = d["ref_s2"]

        # ---- CK Tile a16w4 — run once per token (no k_batch axis) ----
        ck_common = (
            d["ref_s1"], d["w1_qt"], d["w2_qt_shuf"],
            d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
            d["w2_scale_shuf"], d["sorted_weights"],
            topk, ck_block_m,
        )
        print(f"    --- CK Tile a16w4 (bm={ck_block_m}) ---")
        try:
            ck_out = fn_cktile_a16w4(*ck_common)
            torch.cuda.synchronize()
            err_ck = checkAllclose(
                ref_s2, ck_out,
                rtol=args.rtol, atol=args.atol,
                msg=f"      [ck a16w4 t={token}] ",
            )
            prec_ck = "PASS" if err_ck == 0 else (
                "WARN" if err_ck <= 0.05 else "FAIL")
            print("      perf ...", end="", flush=True)
            _, us_ck = run_perftest(
                fn_cktile_a16w4, *ck_common,
                num_iters=ni, num_warmup=nw,
            )
            print(f"  {us_ck:.2f} us")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}")
            us_ck = -1.0
            prec_ck = "ERR"

        num_active_e = d["num_active_experts"]
        flop = token * topk * model_dim * inter_dim * 2
        data_bytes = (
            token * topk * inter_dim * 2
            + num_active_e * model_dim * inter_dim * 0.5
            + token * model_dim * 2
        )

        for tn in args.tile_n:
            for kb in k_batch_vals:
                tag = f"tn={tn} kb={kb}"

                # ---- FlyDSL a16w4 ----
                print(f"    --- FlyDSL a16w4 ({tag}, bm={fly_block_m}) ---")
                try:
                    fly_out = call_flydsl_a16w4(d, topk, fly_block_m,
                                                tile_n=tn, k_batch=kb)
                    torch.cuda.synchronize()
                    err_fly = checkAllclose(
                        ref_s2, fly_out,
                        rtol=args.rtol, atol=args.atol,
                        msg=f"      [flydsl a16w4 {tag}] ",
                    )
                    prec_fly = "PASS" if err_fly == 0 else (
                        "WARN" if err_fly <= 0.05 else "FAIL")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"      FlyDSL a16w4 FAILED: {e}")
                    err_fly = -1.0
                    prec_fly = "ERR"

                if prec_fly != "ERR":
                    fly_common = (
                        d["ref_s1"], d["w2_qt_shuf"],
                        d["sorted_ids"], d["sorted_expert_ids"],
                        d["num_valid_ids"],
                        d["w2_scale_shuf"], d["sorted_weights"],
                        topk, fly_block_m,
                    )
                    print("      perf ...", end="", flush=True)
                    _, us_fly = run_perftest(
                        fn_flydsl_a16w4, *fly_common, tn, "atomic", kb,
                        num_iters=ni, num_warmup=nw,
                    )
                    print(f"  {us_fly:.2f} us")
                else:
                    us_fly = -1.0

                results.append(dict(
                    token=token, fly_bm=fly_block_m, ck_bm=ck_block_m,
                    tile_n=tn, k_batch=kb,
                    model_dim=model_dim, inter_dim=inter_dim,
                    E=E, topk=topk,
                    num_active_e=num_active_e,
                    fly_us=us_fly, fly_prec=prec_fly,
                    ck_us=us_ck, ck_prec=prec_ck,
                    flop=flop, data_bytes=data_bytes,
                ))

    # ---------- summary ----------
    print(f"\n{'=' * 140}")
    print("SUMMARY: FlyDSL a16w4 vs CK Tile a16w4 stage2")
    print(f"{'=' * 140}")
    hdr = (f"  {'token':>5s}  {'f_bm':>4s}  {'c_bm':>4s}  {'tn':>4s}  {'kb':>3s}  {'actE':>4s}  "
           f"{'fly_a16w4':>10s}  {'fprec':>5s}  "
           f"{'ck_a16w4':>10s}  {'cprec':>5s}  "
           f"{'fly/ck':>8s}  "
           f"{'TFLOPS':>8s}  {'BW(GB/s)':>10s}")
    print(hdr)
    print(f"  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*3}  {'-'*4}  "
          f"{'-'*10}  {'-'*5}  "
          f"{'-'*10}  {'-'*5}  {'-'*8}  "
          f"{'-'*8}  {'-'*10}")

    for r in results:
        fly_us = r["fly_us"]
        ck = r["ck_us"]
        flop = r["flop"]
        data_bytes = r["data_bytes"]

        ratio = f"{fly_us / ck:.2f}x" if ck > 0 and fly_us > 0 else "N/A"
        fly_str = f"{fly_us:.2f}" if fly_us > 0 else "ERR"
        ck_str = f"{ck:.2f}" if ck > 0 else "ERR"

        best_us = fly_us if fly_us > 0 else ck
        tflops = flop / (best_us * 1e6) if best_us > 0 else 0.0
        bw = data_bytes / (best_us * 1e-6) / 1e9 if best_us > 0 else 0.0

        print(f"  {r['token']:>5d}  {r['fly_bm']:>4d}  {r['ck_bm']:>4d}  "
              f"{r['tile_n']:>4d}  {r['k_batch']:>3d}  {r['num_active_e']:>4d}  "
              f"{fly_str:>10s}  {r['fly_prec']:>5s}  "
              f"{ck_str:>10s}  {r['ck_prec']:>5s}  "
              f"{ratio:>8s}  "
              f"{tflops:>8.2f}  {bw:>10.1f}")

    print()


if __name__ == "__main__":
    main()
