# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# A4W4 (F4GEMM) test/benchmark for gfx1250, modeled on test_gemm_a4w4.py
# and the aiter-op-test standard (candidates dict + run_perftest loop, a torch
# reference that is only compared (never timed/tabled), TFLOPS + TB/s per
# candidate, one markdown summary table, and a __main__ guard).
#
# One timed candidate per (intype, shape, apre) row:
#   asm : the low-level asm entry with the tile kernel forced by name
#         (the unified gemm_a4w4 API resolves to the same .co, so it is not
#         tabled separately -- a second column would only be confusing).
#
#   MXFP4 (intype=mxfp4): per_1x32 e8m0 scales, gfx1250 weight/scale shuffle
#   NVFP4 (intype=nvfp4): e4m3 per-16 scales + per-tensor global scales

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight_f4, shuffle_scale_f4
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

try:
    import bench_init
except ImportError as e:
    if e.name != "bench_init":
        raise
    from op_tests import bench_init

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)

SUPPORTED_GFX = ["gfx1250"]  # gfx1250-only F4GEMM (preload SGPR) path
MXFP4_SCALE_BLOCK = 32
NVFP4_SCALE_BLOCK = 16

# checkAllclose returns 0 when all-close, else the mismatch fraction. Its own
# verdict thresholds: pass (0) / warning (<= tol_err_ratio) / failed (above).
_TOL_ERR_RATIO = 0.05  # matches checkAllclose default tol_err_ratio


def _verdict(err):
    if err == 0:
        return "pass"
    return "warning" if err <= _TOL_ERR_RATIO else "failed"


def _e4m3_to_f32(s: torch.Tensor) -> torch.Tensor:
    return s.view(torch.float8_e4m3fn).to(torch.float32)


def run_torch_mxfp4(xq, wq, xs, ws, noscale=False):
    # Reference only: fp32 math. Returns fp32; the caller casts to bf16 or
    # quantizes to packed fp4 per outtype. Not timed, not in the table.
    x_f32 = fp4_utils.mxfp4_to_f32(xq)
    w_f32 = fp4_utils.mxfp4_to_f32(wq)
    if noscale:
        # noscale kernel drops all per-block scale loads and uses the HW default
        # scale (1.0), so the reference must ignore the e8m0 scales too.
        return x_f32 @ w_f32.T
    xs = fp4_utils.e8m0_to_f32(xs).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    ws = fp4_utils.e8m0_to_f32(ws).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    return (x_f32 * xs) @ (w_f32 * ws).T


def run_torch_nvfp4(xq, wq, xs, ws, gA, gB, noscale=False):
    # Reference only: fp32 math. Returns fp32 (see run_torch_mxfp4).
    x_f32 = fp4_utils.mxfp4_to_f32(xq)
    w_f32 = fp4_utils.mxfp4_to_f32(wq)
    if noscale:
        # noscale kernel drops the per-block e4m3 scales (HW default 1.0) but
        # STILL folds the per-tensor global scales gA*gB, so the reference must
        # match: skip the per-block scales, keep the global ones.
        return float(gA) * float(gB) * (x_f32 @ w_f32.T)
    xs = _e4m3_to_f32(xs).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    ws = _e4m3_to_f32(ws).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    return float(gA) * float(gB) * (x_f32 * xs) @ (w_f32 * ws).T


def _prep_mxfp4(M, N, K, apre, data_init, scale_init, gen, noscale=False):
    # DATA (fp4 e2m1, packed 2/byte). data & scale are sampled *independently*.
    if data_init == "constant":
        # f4gemm.cpp data_init=0: A=0x22, B=0x33 (fixed representable e2m1).
        xq = torch.full((M, K // 2), 0x22, dtype=torch.uint8)
        wq = torch.full((N, K // 2), 0x33, dtype=torch.uint8)
    else:  # uniform / gaussian / trig / random
        xq = bench_init.fill_fp4((M, K), data_init, gen)
        wq = bench_init.fill_fp4((N, K), data_init, gen)
    # SCALE (e8m0 per-32). auto -> pow2_binomial for E8M0.
    if scale_init == "constant":
        # neutral e8m0 scale 0x7F (exp 0 -> 2^0 = 1.0).
        xs = torch.full((M, K // MXFP4_SCALE_BLOCK), 0x7F, dtype=torch.uint8)
        ws = torch.full((N, K // MXFP4_SCALE_BLOCK), 0x7F, dtype=torch.uint8)
    else:  # auto / pow2_binomial / random
        xs = bench_init.fill_scale_e8m0((M, K // MXFP4_SCALE_BLOCK), scale_init, gen)
        ws = bench_init.fill_scale_e8m0((N, K // MXFP4_SCALE_BLOCK), scale_init, gen)
    ref = run_torch_mxfp4(xq, wq, xs, ws, noscale=noscale)
    inp = dict(
        A=shuffle_weight_f4(xq) if apre else xq,
        B=shuffle_weight_f4(wq),
        sA=shuffle_scale_f4(xs, 7),
        sB=shuffle_scale_f4(ws, 7),
        gA=None,
        gB=None,
    )
    return inp, ref


def _prep_nvfp4(M, N, K, apre, data_init, scale_init, gen, noscale=False):
    # DATA (fp4 e2m1). data & scale sampled independently (bench_init).
    if data_init == "constant":
        # f4gemm.cpp data_init=0: A=0x22, B=0x33 (fixed representable e2m1).
        xq = torch.full((M, K // 2), 0x22, dtype=torch.uint8)
        wq = torch.full((N, K // 2), 0x33, dtype=torch.uint8)
    else:  # uniform / gaussian / trig / random
        xq = bench_init.fill_fp4((M, K), data_init, gen)
        wq = bench_init.fill_fp4((N, K), data_init, gen)
    # SCALE (e4m3 per-16). auto -> gaussian(0.34375,0.08) for E4M3.
    if scale_init == "constant":
        # neutral e4m3 scale 0x38 (exp 7 = bias -> 1.0).
        xs = torch.full((M, K // NVFP4_SCALE_BLOCK), 0x38, dtype=torch.uint8)
        ws = torch.full((N, K // NVFP4_SCALE_BLOCK), 0x38, dtype=torch.uint8)
    else:  # auto / gaussian / random
        xs = bench_init.fill_scale_e4m3((M, K // NVFP4_SCALE_BLOCK), scale_init, gen)
        ws = bench_init.fill_scale_e4m3((N, K // NVFP4_SCALE_BLOCK), scale_init, gen)
    # Per-tensor global scale is NOT part of bench_init: keep neutral.
    gA = gB = 1.0
    ref = run_torch_nvfp4(xq, wq, xs, ws, gA, gB, noscale=noscale)
    inp = dict(
        A=shuffle_weight_f4(xq) if apre else xq,
        B=shuffle_weight_f4(wq),
        sA=shuffle_scale_f4(xs, 8),
        sB=shuffle_scale_f4(ws, 8),
        gA=gA,  # NVFP4 per-tensor global scales (floats)
        gB=gB,
    )
    return inp, ref


@benchmark()  # (intype, M, N, K, apre, scale, outtype, data_init, scale_init, seed) -> cols
def test_gemm(intype, M, N, K, apre, scale, outtype, data_init, scale_init, seed=0,
              mode="perf", dtype=dtypes.bf16):
    block = MXFP4_SCALE_BLOCK if intype == "mxfp4" else NVFP4_SCALE_BLOCK
    assert K % block == 0, f"K must be a multiple of {block}"
    # scale=0 selects the *_noscale kernel: per-block scales are ignored (HW
    # default 1.0). The scale tensors are still built/shuffled and handed to the
    # kernel (API-required); the kernel ignores them and the reference matches.
    noscale = scale == 0
    # outtype=fp4 selects the packed-FP4-output kernel: the fp32 result is
    # quantized to e2m1 (cvt_scale=1) and written 2 vals/byte, so the reference
    # must quantize the same way and the output tensor is fp4x2 [M, N//2].
    out_fp4 = outtype == "fp4"
    out_dtype = dtypes.fp4x2 if out_fp4 else dtype
    gen = bench_init.make_generator(seed)  # fixed seed -> bit-identical buffers
    prep = _prep_mxfp4 if intype == "mxfp4" else _prep_nvfp4
    inp, ref_f32 = prep(M, N, K, apre, data_init, scale_init, gen, noscale=noscale)
    # Reference in the kernel's output form: quantized e2m1 for fp4, else bf16.
    ref = fp4_utils.f32_to_mxfp4(ref_f32) if out_fp4 else ref_f32.to(dtype)
    needTrace = mode == "profile"
    num_iters = 5 if mode == "func" else 101

    # Kernel/.co name for this config. See hsa/gfx1250/f4gemm/f4gemm.csv.
    pre = "ABpreShuffle" if apre else "BpreShuffle"
    ns = "_noscale" if noscale else ""
    base = f"f4gemm_{outtype}_{intype}_{pre}_256x256_4x4_ps{ns}"
    knl = f"_ZN5aiter{len(base)}{base}E"

    def run_asm():
        if intype == "nvfp4":
            return aiter.gemm_nvfp4_asm(
                inp["A"],
                inp["B"],
                inp["sA"],
                inp["sB"],
                inp["gA"],
                inp["gB"],
                dtype=out_dtype,
                a_preshuffle=bool(apre),
                kernelName=knl,
            )
        return aiter.gemm_mxfp4_asm(
            inp["A"],
            inp["B"],
            inp["sA"],
            inp["sB"],
            dtype=out_dtype,
            a_preshuffle=bool(apre),
            kernelName=knl,
        )

    # Only the low-level asm entry is timed/tabled (the unified gemm_a4w4 path
    # resolves to the same .co, so a second column would just be confusing).
    candidates = {"asm": run_asm}

    flops = 2 * M * N * K
    # Output bytes: packed fp4 is M*N/2 bytes; bf16 is M*N*itemsize.
    out_bytes = (M * N) // 2 if out_fp4 else M * N * dtype.itemsize
    nbytes = (
        inp["A"].nbytes
        + inp["B"].nbytes
        + inp["sA"].nbytes
        + inp["sB"].nbytes
        + out_bytes
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        try:
            out, us = run_perftest(fn, num_iters=num_iters, needTrace=needTrace)
        except Exception as e:
            # The .co for this config isn't available (e.g. an fp4-output or
            # noscale variant not yet built/deployed, or nvfp4-fp4 which the
            # shader can't emit). Report it as unsupported and keep the sweep
            # going instead of aborting the whole run.
            aiter.logger.warning(
                "f4gemm not supported: intype=%s outtype=%s scale=%s apre=%s "
                "M=%s N=%s K=%s [%s.co]: %s",
                intype, outtype, scale, apre, M, N, K, base, e,
            )
            ret[f"{name} us"] = float("nan")
            ret[f"{name} TFLOPS"] = float("nan")
            ret[f"{name} TB/s"] = float("nan")
            ret[f"{name} err"] = float("nan")
            ret[f"{name} result"] = "not support"
            continue
        if out_fp4:
            # e2m1 is a coarse, deterministic quantization: compare the dequantized
            # values with zero tolerance == exact fp4-code match (mirrors the host
            # FP4 byte compare). Borderline RNE ties on random data may differ.
            err = checkAllclose(
                fp4_utils.mxfp4_to_f32(ref),
                fp4_utils.mxfp4_to_f32(out),
                rtol=0, atol=0, msg=f"{intype} {name} fp4",
            )
        else:
            err = checkAllclose(ref, out, rtol=1e-1, atol=1.0, msg=f"{intype} {name}")
        ret[f"{name} us"] = round(us, 2)
        ret[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        ret[f"{name} TB/s"] = round(nbytes / us / 1e6, 2)
        ret[f"{name} err"] = err
        ret[f"{name} result"] = _verdict(err)
        if needTrace:
            ret[f"{name} trace"] = f"./aiter_logs/gpu_id_{torch.cuda.current_device()}"
    return ret


def main():
    # Whole-op arch gate goes HERE: @benchmark always returns the call-args dict,
    # so an in-fn return would still emit an args-only NaN row.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gemm_a4w4 (F4GEMM) unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test/benchmark gfx1250 A4W4 (F4GEMM) via the low-level asm entry",
    )
    parser.add_argument(
        "--intype",
        nargs="*",
        choices=["mxfp4", "nvfp4"],
        default=["mxfp4", "nvfp4"],
        help="fp4 input format(s) to sweep, e.g. --intype nvfp4",
    )
    parser.add_argument(
        "--apre",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[1],
        help="A-preshuffle sweep list: 1 preshuffles A, 0 sends it row-major",
    )
    parser.add_argument(
        "--scale",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[1],
        help="scale sweep list: 1 = per-block-scale kernel, 0 = noscale kernel "
        "(per-block scale=1.0; needs the *_noscale.co, see f4gemm.csv)",
    )
    parser.add_argument(
        "--outtype",
        nargs="*",
        choices=["bf16", "fp4"],
        default=["bf16"],
        help="output-format sweep list: bf16 (default) or fp4 (packed e2m1, "
        "cvt_scale=1; needs the f4gemm_fp4_*.co, see f4gemm.csv)",
    )
    parser.add_argument(
        "--data-init",
        dest="data_init",
        nargs="*",
        choices=["constant", "uniform", "gaussian", "trig", "random"],
        default=["constant", "uniform"],
        help="DATA init distribution(s) (mblas-style; sampled independently of scale).\n"
        "Paired position-wise with --scale-init (length-1 broadcasts).\n"
        "  uniform  = FP4 U(-3,3)            [default]\n"
        "  gaussian = N(0,1)                 [norm-dist / LLM-like]\n"
        "  trig     = trig_float in [-2,2]   [optimistic pattern]\n"
        "  random   = pure random e2m1 codes [overly pessimistic]\n"
        "  constant = A=0x22, B=0x33 (deterministic)",
    )
    parser.add_argument(
        "--scale-init",
        dest="scale_init",
        nargs="*",
        choices=["auto", "pow2_binomial", "gaussian", "random", "constant"],
        default=["constant", "auto"],
        help="SCALE init distribution(s) (by scale format)\n"
        "  auto          = format-recommended: mxfp4/E8M0 -> pow2_binomial,\n"
        "                  nvfp4/E4M3 -> gaussian(0.34375,0.08)  [default]\n"
        "  pow2_binomial = 2^(Binomial(21,0.5)-11)   [E8M0 only]\n"
        "  gaussian      = N(0.34375,0.08)           [E4M3 only]\n"
        "  random        = random on-wire byte, modest range\n"
        "  constant      = neutral scale (2^0 = 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed; same seed -> bit-identical data/scale buffers",
    )
    parser.add_argument(
        "--mode",
        choices=["func", "perf", "profile"],
        default="perf",
        help="func=acc only (no table), perf=acc+timing table, profile=perf+trace",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        default=[dtypes.d_dtypes["bf16"]],
        help="output dtype, e.g. -d bf16",
    )
    parser.add_argument(
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        # cluster(4x4)+persistent friendly for the 256x256 tile: M%1024, N%1024.
        default=[(16384, 16384, 16384)],
        help="(M,N,K) tuples, e.g. -mnk 2048,2048,2048 16384,16384,16384",
    )
    args = parser.parse_args()

    # DATA and SCALE init are paired position-wise (NOT crossed), so the default
    # runs exactly two configs: constant+constant and uniform+auto. A length-1
    # list broadcasts against the other axis.
    di_list, si_list = args.data_init, args.scale_init
    if len(di_list) == 1:
        di_list = di_list * len(si_list)
    if len(si_list) == 1:
        si_list = si_list * len(di_list)
    if len(di_list) != len(si_list):
        parser.error(
            "--data-init and --scale-init must have equal length "
            "(or length 1 to broadcast)"
        )
    init_pairs = list(zip(di_list, si_list))

    for dtype in args.dtype:  # one table per output dtype
        # init pair is the OUTERMOST product term -> rows are grouped by
        # (data_init,scale_init) within the single summary table.
        rows = [
            test_gemm(intype, M, N, K, apre, sc, ot, di, si, seed=args.seed,
                      mode=args.mode, dtype=dtype)
            for (di, si), intype, apre, sc, ot, (M, N, K) in itertools.product(
                init_pairs, args.intype, args.apre, args.scale, args.outtype, args.shape
            )
        ]
        if args.mode != "func":
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "gemm_a4w4 (F4GEMM) summary (markdown):\n%s",
                df.to_markdown(index=False),
            )


if __name__ == "__main__":
    main()
