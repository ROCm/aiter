"""Stage2 a8w4 benchmark: FlyDSL a8w4 vs CK Tile a8w4 performance comparison.

Compares FlyDSL a8w4 stage2 against CK Tile a8w4 baseline.
- FlyDSL: flydsl_moe_stage2 with a_dtype="fp8"
- CK Tile: cktile_moe_stage2 with fp8 activation

Usage:
    python bench_stage2_a8w4.py
    python bench_stage2_a8w4.py --num-iters 50
    python bench_stage2_a8w4.py --tokens 1,4,16,64,256
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
from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.utility.fp4_utils import moe_mxfp4_sort, e8m0_to_f32
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

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
    dict(token=64, model_dim=7168, inter_dim=256, expert=256, topk=8),
    dict(token=128, model_dim=7168, inter_dim=256, expert=256, topk=8),
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
    # dict(token=1024, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=2048, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=4096, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=8192, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=16384, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=32768, model_dim=3072, inter_dim=3072, expert=128, topk=4),
    # dict(token=1, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=4, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=8, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=16, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=32, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=64, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=128, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=256, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=512, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=1024, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=2048, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=4096, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=8192, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=16384, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=32768, model_dim=7168, inter_dim=1024, expert=384, topk=8),
    # dict(token=131072, model_dim=7168, inter_dim=256, expert=384, topk=8),
    # dict(token=131072, model_dim=7168, inter_dim=512, expert=384, topk=8),
]


def torch_dynamic_mxfp8_quant(x: torch.Tensor):
    """MXFP8 quantization reference (e4m3fn with e8m0 block scale, block=32)."""
    BLOCK = 32
    orig_shape = x.shape
    x_f32 = x.reshape(-1, x.shape[-1] // BLOCK, BLOCK).float()

    amax, _ = torch.max(torch.abs(x_f32), dim=-1)
    amax_i32 = amax.view(torch.int32)
    amax_rounded = (amax_i32 + 0x200000) & 0xFF800000
    exp_field = (amax_rounded >> 23) & 0xFF

    e8m0_biased = torch.clamp(exp_field - 8, min=0)
    quant_exp = 254 - e8m0_biased
    quant_scale = (quant_exp << 23).view(torch.float32)

    scaled = x_f32 * quant_scale.unsqueeze(-1)
    fp8_vals = scaled.to(torch.float8_e4m3fn)
    fp8_bytes = fp8_vals.view(torch.uint8)

    e8m0_bytes = e8m0_biased.to(torch.uint8).view(dtypes.fp8_e8m0)
    return fp8_bytes.view(*orig_shape), e8m0_bytes.view(*orig_shape[:-1], orig_shape[-1] // BLOCK)


def dequant_fp8_to_bf16(fp8_bytes: torch.Tensor, scale_e8m0: torch.Tensor,
                         last_dim: int) -> torch.Tensor:
    """Dequantize MXFP8 (uint8 fp8 + e8m0 block scale) back to bf16."""
    BLOCK = 32
    fp8_vals = fp8_bytes.view(dtypes.fp8).to(torch.float32)
    scale_f32 = e8m0_to_f32(scale_e8m0)
    fp8_vals = fp8_vals.view(-1, last_dim // BLOCK, BLOCK) * scale_f32.view(-1, last_dim // BLOCK, 1)
    return fp8_vals.view(-1, last_dim).to(torch.bfloat16)


def setup_data(token, model_dim, inter_dim, E, topk, block_m,
               model_dim_pad=0, inter_dim_pad=0,
               dtype=torch.bfloat16, sort_block_m=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10

    score = torch.randn((token, E), dtype=dtype)
    # score = torch.zeros((token, E), dtype=dtype)
    # start_col, end_col = 0, topk
    # for tid in range(token):
    #     score[tid, start_col:end_col] = 1.0
    #     start_col = end_col % E
    #     end_col = start_col + topk
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    valid_model_dim = model_dim - model_dim_pad
    valid_inter_dim = inter_dim - inter_dim_pad
    if model_dim_pad > 0:
        w2[:, valid_model_dim:, :] = 0.0
    if inter_dim_pad > 0:
        inp[:, valid_inter_dim:inter_dim] = 0.0  # not used by stage2 directly
        w1[:, :, valid_inter_dim:inter_dim] = 0.0  # not used by stage2 directly
        w2[:, :, valid_inter_dim:] = 0.0

    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = TORCH_QUANT(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)

    # Stage1 reference output (bf16, doweight=False for stage2 input)
    ref_s1 = torch_moe_stage1(
        inp, w1_qt, w2_qt,
        topk_weights, topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=Q_TYPE,
        a1_scale=None,
        w1_scale=w1_scale,
        doweight=False,
    )
    # ref_s1 shape: (token, topk, inter_dim)

    # Quantize stage1 output to fp8 (a8w4)
    ref_s1_flat = ref_s1.view(-1, inter_dim)
    a2_fp8_bytes, a2_scale_fp8 = torch_dynamic_mxfp8_quant(ref_s1_flat)
    a2_qt_fp8 = a2_fp8_bytes.view(dtypes.fp8).view(token, topk, inter_dim)
    # a2_scale_fp8 shape: (token*topk, inter_dim // 32)

    sort_block_m = sort_block_m if sort_block_m is not None else max(32, block_m)
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
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)

    # Sort a2 scales
    a2_scale_fp8_sort = moe_mxfp4_sort(
        a2_scale_fp8.view(token, topk, -1),
        sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token, block_size=sort_block_m,
    )

    # Dequantize fp8 back to bf16 to create a fair reference
    # (the FlyDSL kernel operates on fp8 data, so the reference should too)
    ref_s1_dequant = dequant_fp8_to_bf16(
        a2_fp8_bytes, a2_scale_fp8, inter_dim
    ).view(token, topk, inter_dim)

    ref_s2_a8w4 = torch_moe_stage2(
        ref_s1_dequant, w1_qt, w2_qt,
        topk_weights, topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        a2_scale=None,
        w2_scale=w2_scale,
    )

    # CK Tile a8w4: scale for sorted positions (ones)
    M_sorted = sorted_ids.shape[0]
    a2_scale_ck = torch.ones(
        (M_sorted, inter_dim // 32), dtype=dtypes.fp8_e8m0, device=inp.device
    )

    num_active_experts = int(topk_ids.unique().numel())

    return dict(
        ref_s2_a8w4=ref_s2_a8w4,
        a2_qt_fp8=a2_qt_fp8,
        a2_scale_fp8_sort=a2_scale_fp8_sort,
        a2_scale_ck=a2_scale_ck,
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


# ---------- FlyDSL a8w4 stage2 ----------

def call_flydsl_a8w4(d, topk, block_m, tile_n=128, mode="atomic",
                     sort_block_m=None,
                     model_dim_pad=0, inter_dim_pad=0, xcd_swizzle=0):
    _sort_block_m = sort_block_m if sort_block_m is not None else max(32, block_m)
    return flydsl_moe_stage2(
        inter_states=d["a2_qt_fp8"],
        w2=d["w2_qt_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk,
        tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16",
        mode=mode,
        w2_scale=d["w2_scale_shuf"],
        a2_scale=d["a2_scale_fp8_sort"],
        sorted_weights=d["sorted_weights"],
        sort_block_m=_sort_block_m,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        xcd_swizzle=xcd_swizzle,
    )


def fn_flydsl_a8w4(a2_qt_fp8, w2_qt_shuf, sorted_ids, sorted_expert_ids,
                    num_valid_ids, w2_scale_shuf, a2_scale_sort,
                    sorted_weights,
                    topk, block_m, tile_n=128, mode="atomic",
                    sort_block_m=None,
                    model_dim_pad=0, inter_dim_pad=0, xcd_swizzle=0):
    _sort_block_m = sort_block_m if sort_block_m is not None else max(32, block_m)
    return flydsl_moe_stage2(
        inter_states=a2_qt_fp8,
        w2=w2_qt_shuf,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16",
        mode=mode,
        w2_scale=w2_scale_shuf,
        a2_scale=a2_scale_sort,
        sorted_weights=sorted_weights,
        sort_block_m=_sort_block_m,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        xcd_swizzle=xcd_swizzle,
    )


# ---------- CK Tile a8w4 stage2 ----------

def fn_cktile_a8w4(a2_qt_fp8, w1_qt, w2_qt_shuf, sorted_ids, sorted_expert_ids,
                   num_valid_ids, w2_scale_shuf, a2_scale_ck, sorted_weights,
                   topk, block_m,
                   model_dim_pad=0, inter_dim_pad=0):
    E = w2_qt_shuf.shape[0]
    model_dim = w2_qt_shuf.shape[1]
    token_num = a2_qt_fp8.shape[0]
    out = torch.empty(
        (token_num, model_dim),
        dtype=torch.bfloat16, device=a2_qt_fp8.device,
    )
    w1_dummy = torch.empty(
        (E, a2_qt_fp8.shape[-1] * 2, model_dim // 2),
        dtype=w2_qt_shuf.dtype, device=a2_qt_fp8.device,
    )
    ck_n_pad = model_dim_pad // 64 * 64
    ck_k_pad = inter_dim_pad // 128 * 128
    cktile_moe_stage2(
        a2_qt_fp8, w1_dummy, w2_qt_shuf,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk,
        w2_scale=w2_scale_shuf.view(dtypes.fp8_e8m0),
        a2_scale=a2_scale_ck,
        block_m=block_m,
        activation=ActivationType.Swiglu,
        sorted_weights=sorted_weights,
        n_pad_zeros=ck_n_pad,
        k_pad_zeros=ck_k_pad,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlyDSL a8w4 vs CK Tile a8w4 stage2")
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--tokens", type=str, default=None,
                        help="Comma-separated token counts (overrides defaults)")
    parser.add_argument("--tile-n", type=int, nargs="+", default=[128, 256],
                        help="tile_n values to test")
    parser.add_argument("--model-dim-pad", type=int, default=0,
                        help="Padding on model_dim (N dimension for stage2)")
    parser.add_argument("--inter-dim-pad", type=int, default=0,
                        help="Padding on inter_dim (K dimension for stage2)")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--xcd-swizzle", type=int, default=0,
                        help="XCD swizzle WGM factor (0=disabled, 4=typical)")
    args = parser.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available
    if not is_flydsl_available():
        print("[SKIP] FlyDSL not available.")
        sys.exit(0)

    if args.tokens:
        cases = [
            dict(token=int(t.strip()), model_dim=7168, inter_dim=256,
                 expert=384, topk=8)
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
    print(f"\nBenchmark: FlyDSL a8w4 vs CK Tile a8w4 stage2{pad_label}")
    print(f"  iters={ni}, warmup={nw}, xcd_swizzle={args.xcd_swizzle}")
    print("=" * 140)

    for c in cases:
        token = c["token"]
        model_dim = c["model_dim"]
        inter_dim = c["inter_dim"]
        E = c["expert"]
        topk = c["topk"]

        fly_block_m = 32 if token < 512 else 64 if token < 2049 else 128
        # fly_block_m = 128
        # fly_sort_block_m = 64 if fly_block_m == 128 else None
        fly_sort_block_m = None
        # fly_block_m = 128
        ck_block_m = 32 if token < 16384 else 64

        valid_model_dim = model_dim - model_dim_pad

        torch.cuda.empty_cache()
        print(f"\n  token={token:>5d}  model_dim={model_dim}  inter_dim={inter_dim}"
              f"  E={E}  topk={topk}  fly_bm={fly_block_m}"
              f"  fly_sbm={fly_sort_block_m or fly_block_m}  ck_bm={ck_block_m}"
              + (f"  mp={model_dim_pad} ip={inter_dim_pad}" if has_padding else ""))

        d = setup_data(token, model_dim, inter_dim, E, topk, fly_block_m,
                       model_dim_pad=model_dim_pad, inter_dim_pad=inter_dim_pad,
                       sort_block_m=fly_sort_block_m)

        ref_s2 = d["ref_s2_a8w4"]
        if model_dim_pad > 0:
            ref_s2 = ref_s2[:, :valid_model_dim]

        def _trim_output(out):
            if model_dim_pad > 0:
                return out[:, :valid_model_dim]
            return out

        for tn in args.tile_n:
            tag = f"tn={tn}"

            # ---- FlyDSL a8w4 ----
            print(f"    --- FlyDSL a8w4 ({tag}, bm={fly_block_m}) ---")
            try:
                fly_out = call_flydsl_a8w4(d, topk, fly_block_m, tile_n=tn,
                                           sort_block_m=fly_sort_block_m,
                                           model_dim_pad=model_dim_pad,
                                           inter_dim_pad=inter_dim_pad,
                                           xcd_swizzle=args.xcd_swizzle)
                torch.cuda.synchronize()
                fly_cmp = _trim_output(fly_out)
                err_fly = checkAllclose(
                    ref_s2, fly_cmp,
                    rtol=args.rtol, atol=args.atol,
                    msg=f"      [flydsl a8w4 {tag}] ",
                )
                prec_fly = "PASS" if err_fly == 0 else (
                    "WARN" if err_fly <= 0.05 else "FAIL")

                # Diagnostic: when bm=64 fails, also try bm=32 with same sort_block_m=64
                if fly_block_m == 64 and prec_fly == "FAIL":
                    print(f"      [DIAG] re-running with tile_m=32 sort_block_m=64 ...")
                    fly_out_32 = flydsl_moe_stage2(
                        inter_states=d["a2_qt_fp8"],
                        w2=d["w2_qt_shuf"],
                        sorted_token_ids=d["sorted_ids"],
                        sorted_expert_ids=d["sorted_expert_ids"],
                        num_valid_ids=d["num_valid_ids"],
                        topk=topk,
                        tile_m=32, tile_n=tn, tile_k=256,
                        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16",
                        mode="atomic",
                        w2_scale=d["w2_scale_shuf"],
                        a2_scale=d["a2_scale_fp8_sort"],
                        sorted_weights=d["sorted_weights"],
                        sort_block_m=64,
                        model_dim_pad=model_dim_pad,
                        inter_dim_pad=inter_dim_pad,
                    )
                    torch.cuda.synchronize()
                    fly_out_32_cmp = _trim_output(fly_out_32)
                    checkAllclose(
                        ref_s2, fly_out_32_cmp,
                        rtol=args.rtol, atol=args.atol,
                        msg=f"      [DIAG bm32 vs ref] ",
                    )
                    checkAllclose(
                        fly_out_32, fly_out,
                        rtol=args.rtol, atol=args.atol,
                        msg=f"      [DIAG bm32 vs bm64] ",
                    )
                    # Per-row error analysis
                    diff = (fly_out - fly_out_32).abs()
                    row_err = diff.max(dim=1).values
                    bad_rows = (row_err > args.atol).nonzero(as_tuple=True)[0]
                    if len(bad_rows) > 0:
                        print(f"      [DIAG] {len(bad_rows)} rows differ (bm32 vs bm64), "
                              f"first 10: {bad_rows[:10].tolist()}")
                    else:
                        print(f"      [DIAG] bm32 and bm64 outputs MATCH => issue is in ref comparison")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"      FlyDSL a8w4 FAILED: {e}")
                err_fly = -1.0
                prec_fly = "ERR"

            if prec_fly != "ERR":
                fly_common = (
                    d["a2_qt_fp8"], d["w2_qt_shuf"],
                    d["sorted_ids"], d["sorted_expert_ids"],
                    d["num_valid_ids"],
                    d["w2_scale_shuf"], d["a2_scale_fp8_sort"],
                    d["sorted_weights"],
                    topk, fly_block_m,
                )
                print("      perf ...", end="", flush=True)
                _, us_fly = run_perftest(
                    fn_flydsl_a8w4, *fly_common, tile_n=tn,
                    sort_block_m=fly_sort_block_m,
                    model_dim_pad=model_dim_pad, inter_dim_pad=inter_dim_pad,
                    xcd_swizzle=args.xcd_swizzle,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_fly:.2f} us")
            else:
                us_fly = -1.0

            # ---- CK Tile a8w4 ----
            if ck_block_m != fly_block_m:
                ck_sorted_ids, ck_sorted_weights, ck_sorted_expert_ids, ck_num_valid_ids, _ = \
                    moe_sorting(d["topk_ids"], d["topk_weights"],
                                E, model_dim, torch.bfloat16, ck_block_m)
                ck_needed = ck_sorted_expert_ids.shape[0] * ck_block_m
                if ck_sorted_ids.shape[0] < ck_needed:
                    ck_pad = torch.full(
                        (ck_needed - ck_sorted_ids.shape[0],), token,
                        dtype=ck_sorted_ids.dtype, device=ck_sorted_ids.device)
                    ck_sorted_ids = torch.cat([ck_sorted_ids, ck_pad])
                    ck_sorted_weights = torch.cat([ck_sorted_weights,
                        torch.zeros(ck_pad.shape[0], dtype=ck_sorted_weights.dtype,
                                    device=ck_sorted_weights.device)])
                ck_a2_scale = torch.ones(
                    (ck_sorted_ids.shape[0], inter_dim // 32),
                    dtype=dtypes.fp8_e8m0, device="cuda",
                )
            else:
                ck_sorted_ids = d["sorted_ids"]
                ck_sorted_weights = d["sorted_weights"]
                ck_sorted_expert_ids = d["sorted_expert_ids"]
                ck_num_valid_ids = d["num_valid_ids"]
                ck_a2_scale = d["a2_scale_ck"]

            print(f"    --- CK Tile a8w4 (bm={ck_block_m}) ---")
            ck_common = (
                d["a2_qt_fp8"], d["w1_qt"], d["w2_qt_shuf"],
                ck_sorted_ids, ck_sorted_expert_ids, ck_num_valid_ids,
                d["w2_scale_shuf"], ck_a2_scale, ck_sorted_weights,
                topk, ck_block_m,
            )
            try:
                print("      perf ...", end="", flush=True)
                _, us_ck = run_perftest(
                    fn_cktile_a8w4, *ck_common,
                    model_dim_pad=model_dim_pad, inter_dim_pad=inter_dim_pad,
                    num_iters=ni, num_warmup=nw,
                )
                print(f"  {us_ck:.2f} us")
            except Exception as e:
                print(f"  FAILED: {e}")
                us_ck = -1.0

            # FLOP & bandwidth: GEMM X[token*topk, inter_dim] @ W[model_dim, inter_dim].T
            num_active_e = d["num_active_experts"]
            flop = token * topk * model_dim * inter_dim * 2
            data_bytes = (
                token * topk * inter_dim * 1                         # X: fp8
                + num_active_e * model_dim * inter_dim * 0.5         # W: fp4
                + token * model_dim * 2                              # Out: bf16
            )

            results.append(dict(
                token=token, fly_bm=fly_block_m, ck_bm=ck_block_m,
                tile_n=tn,
                model_dim=model_dim, inter_dim=inter_dim,
                E=E, topk=topk,
                num_active_e=num_active_e,
                fly_us=us_fly, fly_prec=prec_fly,
                ck_us=us_ck,
                flop=flop, data_bytes=data_bytes,
            ))

    # ---------- summary ----------
    print(f"\n{'=' * 140}")
    print("SUMMARY: FlyDSL a8w4 vs CK Tile a8w4 stage2")
    print(f"{'=' * 140}")
    hdr = (f"  {'token':>5s}  {'f_bm':>4s}  {'c_bm':>4s}  {'tn':>4s}  {'actE':>4s}  "
           f"{'fly_a8w4':>10s}  {'prec':>4s}  "
           f"{'ck_a8w4':>10s}  "
           f"{'fly/ck':>8s}  "
           f"{'TFLOPS':>8s}  {'BW(GB/s)':>10s}")
    print(hdr)
    print(f"  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  "
          f"{'-'*10}  {'-'*4}  "
          f"{'-'*10}  {'-'*8}  "
          f"{'-'*8}  {'-'*10}")

    for r in results:
        fly_us = r["fly_us"]
        ck = r["ck_us"]
        flop = r["flop"]
        data_bytes = r["data_bytes"]

        ratio_fly_ck = f"{fly_us / ck:.2f}x" if ck > 0 and fly_us > 0 else "N/A"
        fly_str = f"{fly_us:.2f}" if fly_us > 0 else "ERR"
        ck_str = f"{ck:.2f}" if ck > 0 else "ERR"

        best_us = fly_us if fly_us > 0 else ck
        tflops = flop / (best_us * 1e6) if best_us > 0 else 0.0
        bw = data_bytes / (best_us * 1e-6) / 1e9 if best_us > 0 else 0.0

        print(f"  {r['token']:>5d}  {r['fly_bm']:>4d}  {r['ck_bm']:>4d}  "
              f"{r['tile_n']:>4d}  {r['num_active_e']:>4d}  "
              f"{fly_str:>10s}  {r['fly_prec']:>4s}  "
              f"{ck_str:>10s}  "
              f"{ratio_fly_ck:>8s}  "
              f"{tflops:>8.2f}  {bw:>10.1f}")

    print()


if __name__ == "__main__":
    main()
