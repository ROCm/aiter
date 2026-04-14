"""Stage1 a8w4 (fp8 activation, fp4 weight, bf16 output) benchmark.

Compares FlyDSL a8w4 stage1 against CK Tile a8w4 and a16w4 baselines.
- FlyDSL default: silu(gate) * up (gate-up separation)
- FlyDSL GUI+swiglu: gate_up_interleave mode with swiglu activation
- CK Tile: Swiglu activation (a8w4 and a16w4)

Usage:
    python bench_stage1_a8w4.py
    python bench_stage1_a8w4.py --num-iters 50
    python bench_stage1_a8w4.py --tokens 1,4,16,64,256
"""

import argparse
import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.test_common import run_perftest, checkAllclose
from aiter.fused_moe import (
    fused_topk, moe_sorting, torch_moe_stage1,
    cktile_moe_stage1, get_ksplit,
)
from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.utility.fp4_utils import moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_W = dtypes.fp4x2
TORCH_QUANT = aiter.get_torch_quant(Q_TYPE)

DEFAULT_CASES = [
    # dict(token=1, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=4, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=8, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=16, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=32, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=64, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=128, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=256, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=512, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=1024, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=2048, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=4096, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=8192, model_dim=7168, inter_dim=256, expert=256, topk=8),
    # dict(token=1, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=4, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=8, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=16, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=32, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=64, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=128, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=256, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=512, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=1024, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=2048, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=4096, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    dict(token=8192, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=1, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=4, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=8, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=16, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=32, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=64, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=128, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=256, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=512, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=1024, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=2048, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=4096, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=8192, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=16384, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=32768, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=131072, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=131072, model_dim=7168, inter_dim=512, expert=384, topk=8),
]


def setup_data(token, model_dim, inter_dim, E, topk, block_m,
               model_dim_pad=0, inter_dim_pad=0, dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10

    # balanced expert routing
    score = torch.zeros((token, E), dtype=dtype)
    start_col = 0
    end_col = topk
    for token_id in range(token):
        score[token_id, start_col:end_col] = 1.0
        start_col = end_col % E
        end_col = start_col + topk
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    valid_model_dim = model_dim - model_dim_pad
    valid_inter_dim = inter_dim - inter_dim_pad
    if model_dim_pad > 0:
        inp[:, valid_model_dim:] = 0.0
        w1[:, :, valid_model_dim:] = 0.0
    if inter_dim_pad > 0:
        w1[:, valid_inter_dim:inter_dim, :] = 0.0
        w1[:, inter_dim + valid_inter_dim:, :] = 0.0

    inp_for_topk = inp[:, :valid_model_dim] if model_dim_pad > 0 else inp

    # quantize weights to fp4x2
    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)

    # fp8 activation (simple cast, scale = ones)
    a1_fp8 = inp.to(dtypes.fp8)
    a1_scale_fp8 = torch.ones(
        (token, model_dim // 32), dtype=dtypes.fp8_e8m0, device=inp.device
    )

    sort_block_m = max(32, block_m)
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

    # a16w4/a8w4 shuffle format (same as test_moe_2stage.py)
    w1_qt_shuf = shuffle_weight_a16w4(w1_qt, 16, True)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, True)

    # Sort activation scales for FlyDSL kernel
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale_fp8[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token, block_size=sort_block_m,
    )


    # CK Tile a8w4: scale for sorted token positions (ones)
    M_sorted = sorted_ids.shape[0]
    a1_scale_ck = torch.ones(
        (M_sorted, model_dim // 32), dtype=dtypes.fp8_e8m0, device=inp.device
    )

    w2_dummy = torch.empty(
        (E, model_dim // 2, inter_dim), dtype=w1_qt.dtype, device=inp.device
    )

    num_active_experts = int(topk_ids.unique().numel())

    return dict(
        inp=inp,
        a1_fp8=a1_fp8,
        a1_scale_fp8=a1_scale_fp8,
        a1_scale_sort=a1_scale_sort,
        a1_scale_ck=a1_scale_ck,
        w1=w1,
        w1_qt=w1_qt,
        w1_scale=w1_scale,
        w1_qt_shuf=w1_qt_shuf,
        w1_scale_shuf=w1_scale_shuf,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w2_dummy=w2_dummy,
        num_active_experts=num_active_experts,
    )


# ---------- FlyDSL a8w4 (silu activation, swiglu not yet in kernel) ----------

def call_flydsl_a8w4(d, topk, block_m, k_batch=1, tile_n=128,
                     act="silu", gate_up_interleave=False,
                     model_dim_pad=0, inter_dim_pad=0, xcd_swizzle=0):
    return flydsl_moe_stage1(
        a=d["a1_fp8"], w1=d["w1_qt_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16",
        act=act,
        w1_scale=d["w1_scale_shuf"], a1_scale=d["a1_scale_sort"],
        use_async_copy=True,
        k_batch=k_batch,
        gate_up_interleave=gate_up_interleave,
        a_scale_one=True,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        xcd_swizzle=xcd_swizzle,
    )


def fn_flydsl_a8w4(a1_fp8, w1_qt_shuf, sorted_ids, sorted_expert_ids,
                    num_valid_ids, w1_scale_shuf, a1_scale_sort,
                    topk, block_m, k_batch=1, tile_n=128,
                    act="silu", gate_up_interleave=False,
                    model_dim_pad=0, inter_dim_pad=0, xcd_swizzle=0):
    return flydsl_moe_stage1(
        a=a1_fp8, w1=w1_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16",
        act=act,
        w1_scale=w1_scale_shuf, a1_scale=a1_scale_sort,
        use_async_copy=True,
        k_batch=k_batch,
        gate_up_interleave=gate_up_interleave,
        a_scale_one=True,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        xcd_swizzle=xcd_swizzle,
    )


# ---------- CK Tile a8w4 (Swiglu, split_k=1) ----------

def fn_cktile_a8w4(a1_fp8, w1_qt_shuf, sorted_ids, sorted_expert_ids,
                   num_valid_ids, w1_scale_shuf, a1_scale_ck,
                   topk, block_m, inter_dim_pad=0, model_dim_pad=0):
    """CK Tile a8w4: fp8 activation + fp4 weight, Swiglu, no split-K."""
    E = w1_qt_shuf.shape[0]
    inter_dim_x2 = w1_qt_shuf.shape[1]
    w2_dummy = torch.empty(
        (E, a1_fp8.shape[1] // 2, inter_dim_x2 // 2),
        dtype=w1_qt_shuf.dtype, device=a1_fp8.device,
    )
    out = torch.empty(
        (a1_fp8.shape[0], topk, inter_dim_x2),
        dtype=torch.bfloat16, device=a1_fp8.device,
    )
    ck_n_pad = inter_dim_pad // 64 * 64 * 2
    ck_k_pad = model_dim_pad // 128 * 128
    cktile_moe_stage1(
        a1_fp8, w1_qt_shuf, w2_dummy,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk, block_m,
        a1_scale=a1_scale_ck,
        w1_scale=w1_scale_shuf.view(dtypes.fp8_e8m0),
        activation=ActivationType.Swiglu,
        split_k=1,
        n_pad_zeros=ck_n_pad,
        k_pad_zeros=ck_k_pad,
    )
    return out

# ---------- CK Tile a16w4 (Swiglu, split_k=1) ----------

def fn_cktile_a16w4(inp, w1_qt_shuf, sorted_ids, sorted_expert_ids,
                    num_valid_ids, w1_scale_shuf,
                    topk, block_m):
    """CK Tile a16w4: bf16 activation + fp4 weight, Swiglu, no split-K."""
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
    cktile_moe_stage1(
        inp, w1_qt_shuf, w2_dummy,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk, block_m,
        a1_scale=None,
        w1_scale=w1_scale_shuf.view(dtypes.fp8_e8m0),
        activation=ActivationType.Swiglu,
        split_k=1,
    )
    return out.to(dtypes.fp8)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlyDSL a8w4 vs CK Tile a8w4/a16w4 stage1 (Swiglu)")
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--tokens", type=str, default=None,
                        help="Comma-separated token counts (overrides defaults)")
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--expert", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--model-dim-pad", type=int, default=0,
                        help="Padding on model_dim (K dimension for stage1)")
    parser.add_argument("--inter-dim-pad", type=int, default=0,
                        help="Padding on inter_dim (N dimension for stage1)")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--xcd-swizzle", type=int, default=0,
                        help="XCD swizzle WGM factor (0=disabled, 4=typical)")
    args = parser.parse_args()

    if args.tokens:
        cases = [
            dict(token=int(t.strip()), model_dim=args.model_dim,
                 inter_dim=args.inter_dim, expert=args.expert, topk=args.topk)
            for t in args.tokens.split(",")
        ]
    else:
        cases = DEFAULT_CASES

    ni, nw = args.num_iters, args.num_warmup
    model_dim_pad = args.model_dim_pad
    inter_dim_pad = args.inter_dim_pad
    has_padding = model_dim_pad > 0 or inter_dim_pad > 0
    results = []

    pad_label = (f"  model_dim_pad={model_dim_pad}, inter_dim_pad={inter_dim_pad}"
                 if has_padding else "")
    print(f"\nBenchmark: FlyDSL a8w4 (silu/gui+swiglu) vs CK Tile a8w4/a16w4 (Swiglu){pad_label}")
    print(f"  iters={ni}, warmup={nw}")
    print("=" * 140)

    for c in cases:
        token = c["token"]
        model_dim = c["model_dim"]
        inter_dim = c["inter_dim"]
        E = c["expert"]
        topk = c["topk"]
        # inter_dim = 512

        # FlyDSL block_m
        fly_block_m = 32 if token < 512 else 64 if token < 2049 else 128

        # CK Tile a8w4: only block_m=32 and 64 available (no 16)
        ck_block_m = 32 if token < 16384 else 64

        # if token < 16:
        #     flydsl_kb_list = [1, 2, 4]
        # else:
        #     flydsl_kb_list = [1]
        flydsl_kb_list = [1]

        torch.cuda.empty_cache()
        valid_inter_dim = inter_dim - inter_dim_pad
        print(f"\n  token={token:>5d}  model_dim={model_dim}  inter_dim={inter_dim}"
              f"  E={E}  topk={topk}  fly_bm={fly_block_m}  ck_bm={ck_block_m}"
              + (f"  mp={model_dim_pad} ip={inter_dim_pad}" if has_padding else ""))

        d = setup_data(token, model_dim, inter_dim, E, topk, fly_block_m,
                       model_dim_pad=model_dim_pad, inter_dim_pad=inter_dim_pad)

        # -- Torch reference (bf16 input, a1_scale=None, like test_moe_2stage) --
        w2_ref = torch.empty(
            (E, model_dim, inter_dim // 2), dtype=d["w1_qt"].dtype, device="cuda"
        )
        ref_silu = torch_moe_stage1(
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
        if inter_dim_pad > 0:
            ref_silu = ref_silu[:, :, :valid_inter_dim]

        ref_swiglu = torch_moe_stage1(
            d["inp"],
            d["w1_qt"], w2_ref,
            d["topk_weights"], d["topk_ids"],
            dtype=torch.bfloat16,
            activation=ActivationType.Swiglu,
            quant_type=Q_TYPE,
            a1_scale=None,
            w1_scale=d["w1_scale"],
        )
        torch.cuda.synchronize()
        if inter_dim_pad > 0:
            ref_swiglu = ref_swiglu[:, :, :valid_inter_dim]

        def _trim_s1_output(out):
            if inter_dim_pad > 0:
                return out[:, :, :valid_inter_dim]
            return out

        for kb in flydsl_kb_list:
            k_per_batch = model_dim // kb if kb > 0 else model_dim
            if model_dim % kb != 0 or k_per_batch % 256 != 0:
                continue

            tag = f"sk={kb}" if kb > 1 else "nosplit"

            # ---- FlyDSL a8w4 (silu) ----
            print(f"    --- FlyDSL a8w4 ({tag}, bm={fly_block_m}) ---")
            try:
                fly_out = call_flydsl_a8w4(d, topk, fly_block_m, k_batch=kb, tile_n=args.tile_n,
                    gate_up_interleave=False,
                    model_dim_pad=model_dim_pad, inter_dim_pad=inter_dim_pad,
                    xcd_swizzle=args.xcd_swizzle)
                torch.cuda.synchronize()
                fly_cmp = _trim_s1_output(fly_out)
                err_fly = checkAllclose(
                    ref_silu, fly_cmp,
                    rtol=args.rtol, atol=args.atol,
                    msg=f"      [flydsl a8w4 t={token},kb={kb}] ",
                )
                prec_fly = "PASS" if err_fly == 0 else (
                    "WARN" if err_fly <= 0.05 else "FAIL")
            except Exception as e:
                print(f"      FlyDSL a8w4 FAILED: {e}")
                err_fly = -1.0
                prec_fly = "ERR"

            if prec_fly != "ERR":
                fly_common = (
                    d["a1_fp8"], d["w1_qt_shuf"],
                    d["sorted_ids"], d["sorted_expert_ids"],
                    d["num_valid_ids"],
                    d["w1_scale_shuf"], d["a1_scale_sort"],
                    topk, fly_block_m,
                )
                print("      perf ...", end="", flush=True)
                _, us_fly = run_perftest(
                    fn_flydsl_a8w4, *fly_common, 1, tile_n=args.tile_n,
                    num_iters=ni, num_warmup=nw, gate_up_interleave=False,
                    model_dim_pad=model_dim_pad, inter_dim_pad=inter_dim_pad,
                    xcd_swizzle=args.xcd_swizzle,
                )
                print(f"  {us_fly:.2f} us")
            else:
                us_fly = -1.0

            # ---- FlyDSL a8w4 GUI (gate_up_interleave + swiglu) ----
            if kb == 1:
                print(f"    --- FlyDSL a8w4 GUI+swiglu (bm={fly_block_m}) ---")
                try:
                    gui_out = call_flydsl_a8w4(
                        d, topk, fly_block_m, k_batch=1, tile_n=args.tile_n,
                        act="swiglu", gate_up_interleave=True,
                        model_dim_pad=model_dim_pad, inter_dim_pad=inter_dim_pad,
                        xcd_swizzle=args.xcd_swizzle,
                    )
                    torch.cuda.synchronize()
                    gui_cmp = _trim_s1_output(gui_out)
                    err_gui = checkAllclose(
                        ref_swiglu, gui_cmp,
                        rtol=args.rtol, atol=args.atol,
                        msg=f"      [flydsl gui+swiglu t={token}] ",
                    )
                    # import pdb; pdb.set_trace()
                    prec_gui = "PASS" if err_gui == 0 else (
                        "WARN" if err_gui <= 0.05 else "FAIL")
                except Exception as e:
                    print(f"      FlyDSL GUI+swiglu FAILED: {e}")
                    err_gui = -1.0
                    prec_gui = "ERR"

                if prec_gui != "ERR":
                    gui_common = (
                        d["a1_fp8"], d["w1_qt_shuf"],
                        d["sorted_ids"], d["sorted_expert_ids"],
                        d["num_valid_ids"],
                        d["w1_scale_shuf"], d["a1_scale_sort"],
                        topk, fly_block_m,
                    )
                    print("      perf ...", end="", flush=True)
                    _, us_gui = run_perftest(
                        fn_flydsl_a8w4, *gui_common, 1, args.tile_n,
                        "swiglu", True,
                        model_dim_pad, inter_dim_pad,
                        args.xcd_swizzle,
                        num_iters=ni, num_warmup=nw,
                    )
                    print(f"  {us_gui:.2f} us")
                else:
                    us_gui = -1.0
            else:
                us_gui = -1.0
                prec_gui = "N/A"

            # ---- CK Tile a8w4 (Swiglu, split_k=1) ----
            # Re-sort with ck_block_m if different from fly_block_m
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
                ck_a1_scale = torch.ones(
                    (ck_sorted_ids.shape[0], model_dim // 32),
                    dtype=dtypes.fp8_e8m0, device="cuda",
                )
            else:
                ck_sorted_ids = d["sorted_ids"]
                ck_sorted_expert_ids = d["sorted_expert_ids"]
                ck_num_valid_ids = d["num_valid_ids"]
                ck_a1_scale = d["a1_scale_ck"]

            print(f"    --- CK Tile a8w4 (Swiglu, bm={ck_block_m}) ---")
            ck_a8w4_common = (
                d["a1_fp8"], d["w1_qt_shuf"],
                ck_sorted_ids, ck_sorted_expert_ids, ck_num_valid_ids,
                d["w1_scale_shuf"], ck_a1_scale,
                topk, ck_block_m,
            )
            try:
                print("      perf ...", end="", flush=True)
                _, us_ck_a8w4 = run_perftest(
                    fn_cktile_a8w4, *ck_a8w4_common,
                    inter_dim_pad=inter_dim_pad, model_dim_pad=model_dim_pad,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_ck_a8w4:.2f} us")
            except Exception as e:
                print(f"  FAILED: {e}")
                us_ck_a8w4 = -1.0

            # ---- CK Tile a16w4 (Swiglu, split_k=1) ----
            print(f"    --- CK Tile a16w4 (Swiglu, bm={ck_block_m}) ---")
            ck_a16w4_common = (
                d["inp"], d["w1_qt_shuf"],
                ck_sorted_ids, ck_sorted_expert_ids, ck_num_valid_ids,
                d["w1_scale_shuf"],
                topk, ck_block_m,
            )
            try:
                print("      perf ...", end="", flush=True)
                _, us_ck_a16w4 = run_perftest(
                    fn_cktile_a16w4, *ck_a16w4_common,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_ck_a16w4:.2f} us")
            except Exception as e:
                print(f"  FAILED: {e}")
                us_ck_a16w4 = -1.0

            # Stage1 FLOP & bandwidth calculation
            # GEMM: X[token*topk, model_dim] @ W[2*inter_dim, model_dim].T
            num_active_e = d["num_active_experts"]
            print(num_active_e)
            flop = token * topk * (2 * inter_dim) * model_dim * 2
            # Data transfer: X(fp8) + W(fp4, active experts only) + Out(bf16)
            data_bytes = (
                token * model_dim * 1                                    # X: fp8
                + num_active_e * 2 * inter_dim * model_dim * 0.5         # W: fp4
                + token * topk * inter_dim * 2                           # Out: bf16
            )

            results.append(dict(
                token=token, fly_bm=fly_block_m, ck_bm=ck_block_m,
                k_batch=kb,
                model_dim=model_dim, inter_dim=inter_dim,
                E=E, topk=topk,
                num_active_e=num_active_e,
                fly_us=us_fly, fly_prec=prec_fly,
                gui_us=us_gui, gui_prec=prec_gui,
                ck_a8w4_us=us_ck_a8w4,
                ck_a16w4_us=us_ck_a16w4,
                flop=flop, data_bytes=data_bytes,
            ))

    # ---------- summary ----------
    print(f"\n{'=' * 180}")
    print("SUMMARY: FlyDSL a8w4 (silu) vs FlyDSL GUI+swiglu vs CK Tile a8w4 (Swiglu) vs CK Tile a16w4 (Swiglu)")
    print(f"{'=' * 180}")
    hdr = (f"  {'token':>5s}  {'f_bm':>4s}  {'c_bm':>4s}  {'kb':>3s}  {'actE':>4s}  "
           f"{'fly_a8w4':>10s}  {'prec':>4s}  "
           f"{'gui_swi':>10s}  {'prec':>4s}  "
           f"{'ck_a8w4':>10s}  {'ck_a16w4':>10s}  "
           f"{'gui/ck8':>8s}  {'gui/ck16':>8s}  "
           f"{'TFLOPS':>8s}  {'BW(GB/s)':>10s}")
    print(hdr)
    print(f"  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*3}  {'-'*4}  "
          f"{'-'*10}  {'-'*4}  {'-'*10}  {'-'*4}  "
          f"{'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*10}")

    for r in results:
        fly_us = r["fly_us"]
        gui_us = r["gui_us"]
        ck8 = r["ck_a8w4_us"]
        ck16 = r["ck_a16w4_us"]
        flop = r["flop"]
        data_bytes = r["data_bytes"]

        ratio_gui_ck8 = f"{gui_us / ck8:.2f}x" if ck8 > 0 and gui_us > 0 else "N/A"
        ratio_gui_ck16 = f"{gui_us / ck16:.2f}x" if ck16 > 0 and gui_us > 0 else "N/A"
        fly_str = f"{fly_us:.2f}" if fly_us > 0 else "ERR"
        gui_str = f"{gui_us:.2f}" if gui_us > 0 else "ERR"
        ck8_str = f"{ck8:.2f}" if ck8 > 0 else "ERR"
        ck16_str = f"{ck16:.2f}" if ck16 > 0 else "ERR"

        # Use gui_swi time for TFLOPS/BW (primary metric); fallback to fly_a8w4
        best_us = gui_us if gui_us > 0 else fly_us
        tflops = flop / (best_us * 1e6) if best_us > 0 else 0.0
        bw = data_bytes / (best_us * 1e-6) / 1e9 if best_us > 0 else 0.0

        print(f"  {r['token']:>5d}  {r['fly_bm']:>4d}  {r['ck_bm']:>4d}  "
              f"{r['k_batch']:>3d}  {r['num_active_e']:>4d}  "
              f"{fly_str:>10s}  {r['fly_prec']:>4s}  "
              f"{gui_str:>10s}  {r.get('gui_prec','N/A'):>4s}  "
              f"{ck8_str:>10s}  {ck16_str:>10s}  "
              f"{ratio_gui_ck8:>8s}  {ratio_gui_ck16:>8s}  "
              f"{tflops:>8.2f}  {bw:>10.1f}")

    print()


if __name__ == "__main__":
    main()
