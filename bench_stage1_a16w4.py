"""Stage1 a16w4 (bf16 activation, mxfp4 weight, bf16 output) benchmark.

Compares FlyDSL a16w4 stage1 against CK Tile a16w4 baseline.
- FlyDSL separated: silu(gate) * up (gate-up separation, default)
- FlyDSL GUI: gate_up_interleave mode with silu activation
- CK Tile a16w4: Swiglu activation

Usage:
    python bench_stage1_a16w4.py
    python bench_stage1_a16w4.py --num-iters 50
    python bench_stage1_a16w4.py --tokens 1,4,16,64,256
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
    fused_topk, moe_sorting, torch_moe_stage1,
    cktile_moe_stage1,
)
from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

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
    # dict(token=2048, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=4096, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=8192, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=1, model_dim=7168, inter_dim=256, expert=384, topk=9),
    # dict(token=4, model_dim=7168, inter_dim=256, expert=384, topk=9),
    # dict(token=8, model_dim=7168, inter_dim=256, expert=384, topk=9),
    # dict(token=16, model_dim=7168, inter_dim=256, expert=384, topk=9),
    # dict(token=32, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=64, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=128, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=256, model_dim=7168, inter_dim=256, expert=384, topk=8),
]


def setup_data(token, model_dim, inter_dim, E, topk, block_m,
               model_dim_pad=0, inter_dim_pad=0, dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    valid_model_dim = model_dim - model_dim_pad
    valid_inter_dim = inter_dim - inter_dim_pad

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10

    if model_dim_pad > 0:
        inp[:, valid_model_dim:] = 0.0
        w1[:, :, valid_model_dim:] = 0.0
    if inter_dim_pad > 0:
        w1[:, valid_inter_dim:inter_dim, :] = 0.0
        w1[:, inter_dim + valid_inter_dim:, :] = 0.0

    # balanced expert routing (use valid input for topk)
    inp_for_topk = inp[:, :valid_model_dim] if model_dim_pad > 0 else inp
    score = torch.zeros((token, E), dtype=dtype)
    start_col, end_col = 0, topk
    for tid in range(token):
        score[tid, start_col:end_col] = 1.0
        start_col = end_col % E
        end_col = start_col + topk
    topk_weights, topk_ids = fused_topk(inp_for_topk, score, topk, True)

    # quantize weights to fp4x2 with per-1x32 scales
    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)

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

    # preshuffle weights and scales for a16w4
    # GUI mode: gate_up interleaved
    w1_qt_shuf_gui = shuffle_weight_a16w4(w1_qt, 16, True)
    w1_scale_shuf_gui = shuffle_scale_a16w4(w1_scale, E, True)
    # separated mode: gate_up=False
    w1_qt_shuf_sep = shuffle_weight_a16w4(w1_qt, 16, False)
    w1_scale_shuf_sep = shuffle_scale_a16w4(w1_scale, E, False)

    w2_dummy = torch.empty(
        (E, model_dim // 2, inter_dim), dtype=w1_qt.dtype, device=inp.device
    )

    num_active_experts = int(topk_ids.unique().numel())

    return dict(
        inp=inp,
        w1=w1,
        w1_qt=w1_qt,
        w1_scale=w1_scale,
        w1_qt_shuf_gui=w1_qt_shuf_gui,
        w1_scale_shuf_gui=w1_scale_shuf_gui,
        w1_qt_shuf_sep=w1_qt_shuf_sep,
        w1_scale_shuf_sep=w1_scale_shuf_sep,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w2_dummy=w2_dummy,
        num_active_experts=num_active_experts,
    )


# ---------- FlyDSL a16w4 ----------

def call_flydsl_a16w4(d, topk, block_m, tile_n=128, tile_k=256,
                      gate_up_interleave=False, waves_per_eu=0,
                      split_k_intra=1, act="silu",
                      model_dim_pad=0, inter_dim_pad=0):
    if gate_up_interleave:
        w1_shuf = d["w1_qt_shuf_gui"]
        ws_shuf = d["w1_scale_shuf_gui"]
    else:
        w1_shuf = d["w1_qt_shuf_sep"]
        ws_shuf = d["w1_scale_shuf_sep"]
    return flydsl_moe_stage1(
        a=d["inp"], w1=w1_shuf,
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
        act=act,
        w1_scale=ws_shuf,
        gate_up_interleave=gate_up_interleave,
        waves_per_eu=waves_per_eu,
        split_k_intra=split_k_intra,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
    )


def fn_flydsl_a16w4(inp, w1_qt_shuf, sorted_ids, sorted_expert_ids,
                     num_valid_ids, w1_scale_shuf,
                     topk, block_m, tile_n=128, tile_k=256,
                     gate_up_interleave=False, waves_per_eu=0,
                     split_k_intra=1, act="silu",
                     model_dim_pad=0, inter_dim_pad=0):
    return flydsl_moe_stage1(
        a=inp, w1=w1_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
        act=act,
        w1_scale=w1_scale_shuf,
        gate_up_interleave=gate_up_interleave,
        waves_per_eu=waves_per_eu,
        split_k_intra=split_k_intra,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
    )


# ---------- CK Tile a16w4 ----------

def fn_cktile_a16w4(inp, w1_qt_shuf, sorted_ids, sorted_expert_ids,
                    num_valid_ids, w1_scale_shuf,
                    topk, block_m,
                    inter_dim_pad=0, model_dim_pad=0):
    E = w1_qt_shuf.shape[0]
    inter_dim_x2 = w1_qt_shuf.shape[1]
    w2_dummy = torch.empty(
        (E, inp.shape[1] // 2, inter_dim_x2 // 2),
        dtype=w1_qt_shuf.dtype, device=inp.device,
    )
    out = torch.empty(
        (inp.shape[0], topk, inter_dim_x2),
        dtype=torch.bfloat16, device=inp.device,
    )
    ck_n_pad = inter_dim_pad // 64 * 64 * 2
    ck_k_pad = model_dim_pad // 128 * 128
    result = cktile_moe_stage1(
        inp, w1_qt_shuf, w2_dummy,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk, block_m,
        a1_scale=None,
        w1_scale=w1_scale_shuf.view(dtypes.fp8_e8m0),
        n_pad_zeros=ck_n_pad,
        k_pad_zeros=ck_k_pad,
        activation=ActivationType.Swiglu,
        split_k=1,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlyDSL a16w4 vs CK Tile a16w4 stage1")
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--tokens", type=str, default=None,
                        help="Comma-separated token counts (overrides defaults)")
    parser.add_argument("--model-dim", type=int, default=3072)
    parser.add_argument("--inter-dim", type=int, default=3072)
    parser.add_argument("--expert", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=256)
    parser.add_argument("--block-m", type=int, default=0,
                        help="Override fly_block_m (0 = auto)")
    parser.add_argument("--waves-per-eu", type=int, default=0,
                        help="Limit occupancy via LDS padding (0=unlimited)")
    parser.add_argument("--split-k-intra", type=int, default=1,
                        help="Intra-WG split-K factor (1=off, 2=split K in half)")
    parser.add_argument("--act", type=str, default="silu",
                        choices=["silu", "swiglu"],
                        help="Activation function (silu or swiglu)")
    parser.add_argument("--model-dim-pad", type=int, default=0,
                        help="Padding on model_dim (K dimension)")
    parser.add_argument("--inter-dim-pad", type=int, default=0,
                        help="Padding on inter_dim (N dimension)")
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
    act = args.act
    model_dim_pad = args.model_dim_pad
    inter_dim_pad = args.inter_dim_pad
    has_padding = model_dim_pad > 0 or inter_dim_pad > 0
    results = []

    act_label = act.upper()
    pad_label = (f"  model_dim_pad={model_dim_pad}, inter_dim_pad={inter_dim_pad}"
                 if has_padding else "")
    print(f"\nBenchmark: FlyDSL a16w4 (separated / GUI) vs CK Tile a16w4"
          f"  act={act_label}{pad_label}")
    print(f"  iters={ni}, warmup={nw}")
    print("=" * 140)

    torch_act = (ActivationType.Swiglu if act == "swiglu"
                 else ActivationType.Silu)

    for c in cases:
        token = c["token"]
        model_dim = c["model_dim"]
        inter_dim = c["inter_dim"]
        E = c["expert"]
        topk = c["topk"]
        valid_inter_dim = inter_dim - inter_dim_pad
        valid_model_dim = model_dim - model_dim_pad

        fly_block_m = args.block_m if args.block_m > 0 else (
            16 if token <= 16 else 32 if token < 512 else 64 if token < 2049 else 128
        )
        fly_block_m = 16
        ck_block_m = 32 if token < 16384 else 64
        ck_block_m = fly_block_m

        torch.cuda.empty_cache()
        print(f"\n  token={token:>5d}  model_dim={model_dim}  inter_dim={inter_dim}"
              f"  E={E}  topk={topk}  fly_bm={fly_block_m}  ck_bm={ck_block_m}"
              f"  act={act_label}"
              + (f"  mp={model_dim_pad} ip={inter_dim_pad}" if has_padding else ""))

        d = setup_data(token, model_dim, inter_dim, E, topk, fly_block_m,
                       model_dim_pad=model_dim_pad, inter_dim_pad=inter_dim_pad)

        # -- Torch reference --
        w2_ref = torch.empty(
            (E, model_dim, inter_dim // 2),
            dtype=d["w1_qt"].dtype, device="cuda",
        )
        ref_out_full = torch_moe_stage1(
            d["inp"],
            d["w1_qt"], w2_ref,
            d["topk_weights"], d["topk_ids"],
            dtype=torch.bfloat16,
            activation=torch_act,
            quant_type=Q_TYPE,
            a1_scale=None,
            w1_scale=d["w1_scale"],
        )
        torch.cuda.synchronize()
        ref_out = (ref_out_full[:, :, :valid_inter_dim].contiguous()
                   if inter_dim_pad > 0 else ref_out_full)

        # CK's "Swiglu" is standard SiLU gating (silu(gate)*up), so use
        # SiLU reference for CK comparison regardless of --act.
        if act != "silu":
            ref_ck_full = torch_moe_stage1(
                d["inp"],
                d["w1_qt"], w2_ref,
                d["topk_weights"], d["topk_ids"],
                dtype=torch.bfloat16,
                activation=ActivationType.Silu,
                quant_type=Q_TYPE,
                a1_scale=None,
                w1_scale=d["w1_scale"],
            )
            torch.cuda.synchronize()
            ref_ck = (ref_ck_full[:, :, :valid_inter_dim].contiguous()
                      if inter_dim_pad > 0 else ref_ck_full)
        else:
            ref_ck = ref_out

        def _trim_output(out):
            """Trim padding columns from kernel output if needed."""
            if inter_dim_pad > 0:
                return out[:, :, :valid_inter_dim].contiguous()
            return out

        # ---- FlyDSL a16w4 separated ----
        print(f"    --- FlyDSL a16w4 separated {act_label} (bm={fly_block_m}) ---")
        try:
            fly_out = call_flydsl_a16w4(
                d, topk, fly_block_m, tile_n=args.tile_n,
                tile_k=args.tile_k, gate_up_interleave=False,
                waves_per_eu=args.waves_per_eu,
                split_k_intra=args.split_k_intra,
                act=act, model_dim_pad=model_dim_pad,
                inter_dim_pad=inter_dim_pad,
            )
            torch.cuda.synchronize()
            fly_cmp = _trim_output(fly_out)
            err_fly = checkAllclose(
                ref_out, fly_cmp,
                rtol=args.rtol, atol=args.atol,
                msg=f"      [flydsl a16w4 sep {act} t={token}] ",
            )
            prec_fly = "PASS" if err_fly == 0 else (
                "WARN" if err_fly <= 0.05 else "FAIL")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"      FlyDSL a16w4 sep FAILED: {e}")
            err_fly = -1.0
            prec_fly = "ERR"

        if prec_fly != "ERR":
            fly_common = (
                d["inp"], d["w1_qt_shuf_sep"],
                d["sorted_ids"], d["sorted_expert_ids"],
                d["num_valid_ids"],
                d["w1_scale_shuf_sep"],
                topk, fly_block_m,
            )
            print("      perf ...", end="", flush=True)
            _, us_fly_sep = run_perftest(
                fn_flydsl_a16w4, *fly_common, args.tile_n, args.tile_k, False,
                args.waves_per_eu, args.split_k_intra, act,
                model_dim_pad, inter_dim_pad,
                num_iters=ni, num_warmup=nw,
            )
            print(f"  {us_fly_sep:.2f} us")
        else:
            us_fly_sep = -1.0

        # ---- FlyDSL a16w4 GUI (gate_up_interleave) ----
        print(f"    --- FlyDSL a16w4 GUI {act_label} (bm={fly_block_m}) ---")
        try:
            gui_out = call_flydsl_a16w4(
                d, topk, fly_block_m, tile_n=args.tile_n,
                tile_k=args.tile_k, gate_up_interleave=True,
                waves_per_eu=args.waves_per_eu,
                split_k_intra=args.split_k_intra,
                act=act, model_dim_pad=model_dim_pad,
                inter_dim_pad=inter_dim_pad,
            )
            torch.cuda.synchronize()
            gui_cmp = _trim_output(gui_out)
            err_gui = checkAllclose(
                ref_out, gui_cmp,
                rtol=args.rtol, atol=args.atol,
                msg=f"      [flydsl a16w4 gui {act} t={token}] ",
            )
            prec_gui = "PASS" if err_gui == 0 else (
                "WARN" if err_gui <= 0.05 else "FAIL")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"      FlyDSL a16w4 GUI FAILED: {e}")
            err_gui = -1.0
            prec_gui = "ERR"

        if prec_gui != "ERR":
            gui_common = (
                d["inp"], d["w1_qt_shuf_gui"],
                d["sorted_ids"], d["sorted_expert_ids"],
                d["num_valid_ids"],
                d["w1_scale_shuf_gui"],
                topk, fly_block_m,
            )
            print("      perf ...", end="", flush=True)
            _, us_fly_gui = run_perftest(
                fn_flydsl_a16w4, *gui_common, args.tile_n, args.tile_k, True,
                args.waves_per_eu, args.split_k_intra, act,
                model_dim_pad, inter_dim_pad,
                num_iters=ni, num_warmup=nw,
            )
            print(f"  {us_fly_gui:.2f} us")
        else:
            us_fly_gui = -1.0

        # ---- CK Tile a16w4 (Swiglu) ----
        us_ck = -1.0
        if ck_block_m != fly_block_m:
            ck_sorted_ids, _, ck_sorted_expert_ids, ck_num_valid_ids, _ = \
                moe_sorting(d["topk_ids"], d["topk_weights"],
                            E, model_dim, torch.bfloat16, ck_block_m)
            ck_needed = ck_sorted_expert_ids.shape[0] * ck_block_m
            if ck_sorted_ids.shape[0] < ck_needed:
                ck_pad = torch.full(
                    (ck_needed - ck_sorted_ids.shape[0],), token,
                    dtype=ck_sorted_ids.dtype, device=ck_sorted_ids.device)
                ck_sorted_ids = torch.cat([ck_sorted_ids, ck_pad])
        else:
            ck_sorted_ids = d["sorted_ids"]
            ck_sorted_expert_ids = d["sorted_expert_ids"]
            ck_num_valid_ids = d["num_valid_ids"]

        print(f"    --- CK Tile a16w4 (Swiglu, bm={ck_block_m}) ---")
        ck_common = (
            d["inp"], d["w1_qt_shuf_gui"],
            ck_sorted_ids, ck_sorted_expert_ids, ck_num_valid_ids,
            d["w1_scale_shuf_gui"],
            topk, ck_block_m,
            inter_dim_pad, model_dim_pad,
        )
        try:
            ck_out = fn_cktile_a16w4(*ck_common)
            torch.cuda.synchronize()
            ck_cmp = _trim_output(ck_out)
            err_ck = checkAllclose(
                ref_ck, ck_cmp,
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
            print(f"  FAILED: {e}")
            us_ck = -1.0

        num_active_e = d["num_active_experts"]
        flop = token * topk * (2 * valid_inter_dim) * valid_model_dim * 2
        data_bytes = (
            token * valid_model_dim * 2
            + num_active_e * 2 * valid_inter_dim * valid_model_dim * 0.5
            + token * topk * valid_inter_dim * 2
        )

        results.append(dict(
            token=token, fly_bm=fly_block_m, ck_bm=ck_block_m,
            model_dim=model_dim, inter_dim=inter_dim,
            E=E, topk=topk,
            num_active_e=num_active_e,
            sep_us=us_fly_sep, sep_prec=prec_fly,
            gui_us=us_fly_gui, gui_prec=prec_gui,
            ck_us=us_ck,
            flop=flop, data_bytes=data_bytes,
        ))

    # ---------- summary ----------
    print(f"\n{'=' * 160}")
    print("SUMMARY: FlyDSL a16w4 separated vs FlyDSL a16w4 GUI vs CK Tile a16w4 (Swiglu)")
    print(f"{'=' * 160}")
    hdr = (f"  {'token':>5s}  {'f_bm':>4s}  {'c_bm':>4s}  {'actE':>4s}  "
           f"{'sep_silu':>10s}  {'prec':>4s}  "
           f"{'gui_silu':>10s}  {'prec':>4s}  "
           f"{'ck_a16w4':>10s}  "
           f"{'sep/ck':>8s}  {'gui/ck':>8s}  "
           f"{'TFLOPS':>8s}  {'BW(GB/s)':>10s}")
    print(hdr)
    print(f"  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*4}  "
          f"{'-'*10}  {'-'*4}  {'-'*10}  {'-'*4}  "
          f"{'-'*10}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*10}")

    for r in results:
        sep_us = r["sep_us"]
        gui_us = r["gui_us"]
        ck = r["ck_us"]
        flop = r["flop"]
        data_bytes = r["data_bytes"]

        ratio_sep = f"{sep_us / ck:.2f}x" if ck > 0 and sep_us > 0 else "N/A"
        ratio_gui = f"{gui_us / ck:.2f}x" if ck > 0 and gui_us > 0 else "N/A"
        sep_str = f"{sep_us:.2f}" if sep_us > 0 else "ERR"
        gui_str = f"{gui_us:.2f}" if gui_us > 0 else "ERR"
        ck_str = f"{ck:.2f}" if ck > 0 else "ERR"

        best_us = gui_us if gui_us > 0 else sep_us
        tflops = flop / (best_us * 1e6) if best_us > 0 else 0.0
        bw = data_bytes / (best_us * 1e-6) / 1e9 if best_us > 0 else 0.0

        print(f"  {r['token']:>5d}  {r['fly_bm']:>4d}  {r['ck_bm']:>4d}  "
              f"{r['num_active_e']:>4d}  "
              f"{sep_str:>10s}  {r['sep_prec']:>4s}  "
              f"{gui_str:>10s}  {r['gui_prec']:>4s}  "
              f"{ck_str:>10s}  "
              f"{ratio_sep:>8s}  {ratio_gui:>8s}  "
              f"{tflops:>8.2f}  {bw:>10.1f}")

    print()


if __name__ == "__main__":
    main()
