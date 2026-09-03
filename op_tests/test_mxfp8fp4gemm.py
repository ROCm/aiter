# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# ===============================================================================
# gfx1250 F8GEMM ASM Support Matrix
# -------------------------------------------------------------------------------
#  OUTTYPE | A_PRESHUFFLE | B_PRESHUFFLE | B_INTYPE |   M    |   N    |   K
# ---------+--------------+----------+--------+--------+------------------------
#  BF16    |      0       |      1       |  MXFP8   | %1==0  | %16==0 | %128==0
#  BF16    |      0       |      1       |  MXFP4   | %1==0  | %16==0 | %128==0
#  BF16    |      1       |      1       |  MXFP8   | %2==0  | %16==0 | %128==0
#  BF16    |      1       |      1       |  MXFP4   | %2==0  | %16==0 | %128==0
# -------------------------------------------------------------------------------
# Notes:
#  - B_PRESHUFFLE is always 1 (B is always pre-shuffled).
#  - A_PRESHUFFLE=1 tightens the M constraint from %1==0 to %2==0.
#  - K is always a multiple of 128.
#  - OUTTYPE is BF16-only today. fp8 out (e4m3 + per-block E8M0, as f4gemm does)
#    is planned; the sweep axis and dispatch seam below are already in place.
# ===============================================================================

import argparse
import itertools
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.shuffle import (
    shuffle_mxfp8fp4_a,
    shuffle_mxfp8fp4_b,
    shuffle_mxfp8fp4_scale,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

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

SUPPORTED_GFX = ["gfx1250"]

_OUT_DTYPE = {"bf16": dtypes.bf16}

# gfx1250 F8GEMM .co is a persistent shader: it always launches PERSISTENT_TG
# threadgroups regardless of problem size (must match the .co's WG_MAX).
PERSISTENT_TG = 256


def _heuristic_tile(M):
    """Tile (tile_m, tile_n) the cpp dispatch picks for this M (mirrors
    get_heuristic_kernel in asm_mxfp8fp4gemm.cu): M<=64 wastes most of a 256-tall
    tile's rows, so it takes the 64x512 variant; any larger M takes 256x256."""
    return (64, 512) if M <= 64 else (256, 256)


def _report_active_tg(M, N, tile_m, tile_n, label):
    """Warn when the persistent shader's TG slots aren't fully packed.

    The .co always launches PERSISTENT_TG (256) threadgroups. The real work is
    ceil(M/tile_m) * ceil(N/tile_n) tiles; when that isn't a multiple of 256 the
    final wave leaves the leftover TG slots idle (wasted CUs) -> "poor perf".
    (Moved here from the cpp dispatch so the report lives with the test.)
    """
    tg_m = (M + tile_m - 1) // tile_m
    tg_n = (N + tile_n - 1) // tile_n
    active_tg = tg_m * tg_n
    wave_active = (
        PERSISTENT_TG if active_tg % PERSISTENT_TG == 0 else active_tg % PERSISTENT_TG
    )
    info = (
        f"{label}: active {wave_active}/{PERSISTENT_TG} TG "
        f"({tg_m} M-tiles x {tg_n} N-tiles, tile_m={tile_m}, tile_n={tile_n})"
    )
    if active_tg % PERSISTENT_TG == 0:
        aiter.logger.info("dispatch to %s", info)
    else:
        tag = "\033[31mpoor perf\033[0m" if sys.stderr.isatty() else "poor perf"
        aiter.logger.warning("dispatch to %s - %s!", info, tag)


PERF_SHAPES = {
    "a8w8": [
        (32768, 16384, 8192),  # compute-bound
        (2, 1048576, 16384),  # memory-bound (N16K x BS64 folded into M)
    ],
    "a8w4": [
        (16384, 16384, 16384),  # compute-bound
        (2, 1048576, 16384),  # memory-bound
    ],
}
FUNC_SHAPES = [
    # qkv_proj
    (1, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    (16384, 1280, 8192),
    # attn_out
    (1, 8192, 1024),
    (32, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (192, 8192, 1024),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
    (8192, 8192, 1024),
    (16384, 8192, 1024),
    # hipmm gelu_bias
    (32, 3072, 768),
    (4096, 3072, 768),
    (8192, 3072, 768),
    # hipmm preshuffle
    (16, 7424, 8192),
    (32, 7424, 8192),
    (48, 7424, 8192),
    (64, 7424, 8192),
    (4096, 7424, 8192),
    (5120, 7424, 8192),
    (8192, 7424, 8192),
    # partial_tile.
    (128, 384, 8192),
    (66, 384, 8192),
    (65, 384, 8192),
]

MX_SCALE_BLOCK = 32

# checkAllclose returns 0 when all-close, else the mismatch fraction. Its own
# verdict thresholds: pass (0) / warning (<= tol_err_ratio) / failed (above).
_TOL_ERR_RATIO = 0.05  # matches checkAllclose default tol_err_ratio


def _verdict(err):
    if err == 0:
        return "pass"
    return "warning" if err <= _TOL_ERR_RATIO else "failed"


def _support_reason(outtype, apre, M, N, K):
    """Support matrix gate. Returns None if the (outtype,apre,M,N,K) combo
    is supported, else a short reason string (row marked "not support"). Mirrors
    the dispatch heuristic in asm_mxfp8fp4gemm.cu so shapes are skipped before the
    shuffle/prep step rather than crashing on an assert."""
    if outtype not in _OUT_DTYPE:
        return f"outtype {outtype}"  # no kernel for this output format yet
    if K % 128 != 0:
        return "K%128"  # A (m/2,k/128) preshuffle
    if N % 16 != 0:
        return "N%16"  # B 16x16 preshuffle
    if apre and M % 2 != 0:
        return "apre M%2"  # A (m/2,k/128) preshuffle
    return None


def _ref(intype, A, B, sA, sB, M, N):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    A_f32 = A.to(torch.float32)[:M]
    if intype == "a8w4":
        B_f32 = fp4_utils.mxfp4_to_f32(B)[:N]
    else:
        B_f32 = B.to(torch.float32)[:N]
    sA_f = fp4_utils.e8m0_to_f32(sA).repeat_interleave(MX_SCALE_BLOCK, dim=1)
    sB_f = fp4_utils.e8m0_to_f32(sB).repeat_interleave(MX_SCALE_BLOCK, dim=1)
    return (A_f32 * sA_f) @ (B_f32 * sB_f).T


def _const_mxfp8(rows: int, k: int, val: float) -> torch.Tensor:
    # Constant mxfp8 (e4m3): a single representable value, deterministic for perf.
    return torch.full((rows, k), val, dtype=torch.float32).to(torch.float8_e4m3fn)


def _prep(
    intype: str, M: int, N: int, K: int, apre: int, data_init: str, scale_init: str, gen
):
    """Build raw + shuffled device tensors and the f32 golden reference.

    DATA and SCALE are sampled *independently* (bench_init), selected by
    ``data_init`` / ``scale_init``:
      data_init  : uniform (FP8 U(-6,6) / FP4 U(-3,3)) [default] | gaussian |
                   trig | random | constant (A/B = 0.5)
      scale_init : auto (E8M0 -> pow2_binomial) [default] | pow2_binomial |
                   random | constant (neutral 0x7F -> 2^0 = 1.0)
    """
    # DATA: A is mxfp8 (e4m3); B is mxfp4 (e2m1 packed) for a8w4, else mxfp8.
    if data_init == "constant":
        A = _const_mxfp8(M, K, 0.5)
        if intype == "a8w4":
            B = torch.full((N, K // 2), 0x11, dtype=torch.uint8)  # e2m1 nibble 0.5
        else:
            B = _const_mxfp8(N, K, 0.5)
    else:  # uniform / gaussian / trig / random
        A = bench_init.fill_fp8((M, K), data_init, gen)
        if intype == "a8w4":
            B = bench_init.fill_fp4((N, K), data_init, gen)
        else:
            B = bench_init.fill_fp8((N, K), data_init, gen)

    # SCALE: e8m0 per-32 for both operands. auto -> pow2_binomial for E8M0.
    if scale_init == "constant":
        sA = torch.full((M, K // MX_SCALE_BLOCK), 0x7F, dtype=torch.uint8)
        sB = torch.full((N, K // MX_SCALE_BLOCK), 0x7F, dtype=torch.uint8)
    else:  # auto / pow2_binomial / random
        sA = bench_init.fill_scale_e8m0((M, K // MX_SCALE_BLOCK), scale_init, gen)
        sB = bench_init.fill_scale_e8m0((N, K // MX_SCALE_BLOCK), scale_init, gen)

    # fp32 golden; the caller casts/quantizes it to the requested outtype.
    ref_f32 = _ref(intype, A, B, sA, sB, M, N)

    inp = {
        "A": shuffle_mxfp8fp4_a(A) if apre else A,  # B always preshuffled, A per `apre`
        "B": shuffle_mxfp8fp4_b(B),
        "sA": shuffle_mxfp8fp4_scale(sA),
        "sB": shuffle_mxfp8fp4_scale(sB),
    }
    return inp, ref_f32


@benchmark()
def test_gemm(
    intype,
    M,
    N,
    K,
    apre,
    outtype="bf16",
    data_init="uniform",
    scale_init="auto",
    seed=0,
    mode="perf",
    knl_name=None,
):
    # Skip unfittable shapes up front (before prep/shuffle) so they show as
    # "not support" rather than crashing on a shape assert / missing kernel.
    reason = _support_reason(outtype, apre, M, N, K)
    if reason is not None:
        aiter.logger.warning(
            "mxfp8fp4 not supported (%s): intype=%s outtype=%s apre=%s M=%s N=%s K=%s",
            reason,
            intype,
            outtype,
            apre,
            M,
            N,
            K,
        )
        return {
            "gfx": get_gfx(),
            "knl_name": knl_name or "(heuristic)",
            "asm us": float("nan"),
            "asm TFLOPS": float("nan"),
            "asm TB/s": float("nan"),
            "asm err": float("nan"),
            "asm result": f"not support ({reason})",
        }

    assert K % MX_SCALE_BLOCK == 0, f"K must be a multiple of {MX_SCALE_BLOCK}"
    out_dtype = _OUT_DTYPE[outtype]
    gen = bench_init.make_generator(seed)  # fixed seed -> bit-identical buffers
    inp, ref_f32 = _prep(intype, M, N, K, apre, data_init, scale_init, gen)
    ref = ref_f32.to(out_dtype)
    needTrace = mode == "profile"
    num_iters = 5 if mode == "func" else 101

    # Single ASM kernel under test, dispatched by intype. Inputs passed as ARGS so
    # run_perftest can rotate them (defeats the L2 hot-cache). Dispatch is
    # heuristic by default (kernelName=""); an explicit --knl-name forces that .co.
    kern = aiter.gemm_a8w4_mxfp8 if intype == "a8w4" else aiter.gemm_a8w8_mxfp8
    # Dispatch mode. Default (knl_name=None) is heuristic: knl="" lets the op pick
    # the .co by (b_intype, a_preshuffle). Explicit is opt-in via --knl-name:
    # "auto" derives this config's mangled name from the CSV convention (see
    # hsa/gfx1250/mxfp8fp4gemm/mxfp8fp4gemm.csv); any other value is used verbatim.
    if not knl_name:
        knl = ""
    elif knl_name == "auto":
        middle = "mxfp8fp8" if intype == "a8w8" else "mxfp8fp4"
        pre = "ABpreShuffle" if apre else "BpreShuffle"
        base = f"f8gemm_{outtype}_{middle}_{pre}_256x256_4x4_ps"
        knl = f"_ZN5aiter{len(base)}{base}E"
    else:
        knl = knl_name

    def run_asm(A, B, sA, sB):
        return kern(
            A, B, sA, sB, dtype=out_dtype, a_preshuffle=bool(apre), kernelName=knl
        )

    asm_args = (inp["A"], inp["B"], inp["sA"], inp["sB"])
    candidates = {"asm": (run_asm, asm_args)}

    flops = 2 * M * N * K
    # Scale bytes use the LOGICAL (unpadded) size: shuffle_mxfp8fp4_scale pads rows
    # to a multiple of 32, but the shader clamps its scale dim and never reads the
    # padding, so the padded buffer's .nbytes would inflate the reported bandwidth.
    # (A/B shuffles are pure reshapes -- no padding -- so their .nbytes is exact.)
    scale_bytes = (M + N) * (K // MX_SCALE_BLOCK)  # e8m0: 1 byte per 32-K block
    in_bytes = inp["A"].nbytes + inp["B"].nbytes + scale_bytes

    ret = {"gfx": get_gfx(), "knl_name": knl_name or "(heuristic)"}
    # Report TG occupancy for the tile the cpp dispatch picks (M<=64 -> 64x512).
    _middle = "mxfp8fp8" if intype == "a8w8" else "mxfp8fp4"
    _pre = "ABpreShuffle" if apre else "BpreShuffle"
    _tile_m, _tile_n = _heuristic_tile(M)
    _label = f"f8gemm_{outtype}_{_middle}_{_pre}_{_tile_m}x{_tile_n}_4x4_ps"
    _report_active_tg(M, N, _tile_m, _tile_n, _label)
    # Only a missing .co is reported as "not support"; any other failure (OOM,
    # memory fault, shape assert, ...) must propagate, not show as a green cell.
    # An explicit --knl-name that isn't in the cfg is a real error (typo / missing
    # build), so "kernel not in cfg" is benign ONLY on the heuristic path (knl == "").
    _NOT_SUPPORTED_MARKERS = ("cannot get heuristic kernel",)
    if not knl:
        _NOT_SUPPORTED_MARKERS += ("kernel not in cfg_mxfp8fp4gemm",)
    for name, (cand, cand_args) in candidates.items():
        try:
            out, us = run_perftest(
                cand,
                *cand_args,
                num_iters=num_iters,
                needTrace=needTrace,
            )
        except Exception as e:
            if not any(m in str(e) for m in _NOT_SUPPORTED_MARKERS):
                raise
            aiter.logger.warning(
                "mxfp8fp4 no dispatchable kernel: intype=%s outtype=%s apre=%s "
                "M=%s N=%s K=%s: %s",
                intype,
                outtype,
                apre,
                M,
                N,
                K,
                e,
            )
            ret[f"{name} us"] = float("nan")
            ret[f"{name} TFLOPS"] = float("nan")
            ret[f"{name} TB/s"] = float("nan")
            ret[f"{name} err"] = float("nan")
            ret[f"{name} result"] = "not support"
            continue
        # a8w8 (mxfp8xmxfp8) can show a "warning" on ~1 element in 5e5: an
        # ill-conditioned output where sum|terms| (~2.7e5) cancels to a ~0.2
        # residual (ratio ~9e-7). The fp32 accumulation noise floor there is
        # O(1), so any accumulator (kernel or this ref) lands in [-1,+1] noise
        # purely by summation order -- benign, not a kernel defect. a8w4's
        # coarser fp4 B rarely hits it. atol=1.0 keeps such elements a warning.
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-1,
            atol=1.0,
            msg=f"{intype} {name}",
        )
        io_bytes = in_bytes + out.nbytes
        ret[f"{name} us"] = round(us, 2)
        ret[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        ret[f"{name} TB/s"] = round(io_bytes / us / 1e6, 2)
        ret[f"{name} err"] = err
        ret[f"{name} result"] = _verdict(err)
        if needTrace:
            ret[f"{name} trace"] = f"./aiter_logs/gpu_id_{torch.cuda.current_device()}"
    return ret


def main():
    # Whole-op arch gate goes HERE, not inside test_gemm: @benchmark always
    # returns the call-args dict, so an in-fn `return` still emits an args-only row.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mxfp8fp4 gemm (a8w8/a8w4) unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test/benchmark gfx1250 MXFP8x{FP8,FP4} (a8w8 / a8w4) ASM kernels",
    )
    parser.add_argument(
        "--mode",
        choices=["func", "perf", "profile"],
        default="perf",
        help="func=acc only, perf=acc+timing, profile=perf+trace",
    )
    parser.add_argument(
        "--intype",
        nargs="*",
        choices=["a8w8", "a8w4"],
        default=["a8w8", "a8w4"],
        help="input-type sweep list (a8w8 and/or a8w4)",
    )
    parser.add_argument(
        "--apre",
        type=int,
        nargs="*",
        choices=[1, 0],
        default=None,
        help="A-preshuffle sweep list: 1 preshuffles A (M%%2), 0 sends it "
        "row-major (M%%1). Default (unset): perf/profile = [1], func = [1, 0].",
    )
    parser.add_argument(
        "--outtype",
        nargs="*",
        choices=["bf16"],
        default=["bf16"],
        help="output-format sweep list (default: bf16):\n"
        "  bf16 = bf16 [M,N]                     [only format with a kernel]",
    )
    parser.add_argument(
        "--data-init",
        dest="data_init",
        nargs="*",
        choices=["constant", "uniform", "gaussian", "trig", "random"],
        default=None,
        help="DATA init distribution(s) (mblas-style; sampled independently of scale).\n"
        "Paired position-wise with --scale-init (length-1 broadcasts).\n"
        "Default (unset): perf/profile = 'constant uniform', func = 'uniform'\n"
        "  uniform  = FP8 U(-6,6) / FP4 U(-3,3)  [default]\n"
        "  gaussian = N(0,1)                     [norm-dist / LLM-like]\n"
        "  trig     = trig_float in [-2,2]       [optimistic pattern]\n"
        "  random   = pure random on-wire codes  [overly pessimistic]\n"
        "  constant = A/B = 0.5 (deterministic)",
    )
    parser.add_argument(
        "--scale-init",
        dest="scale_init",
        nargs="*",
        choices=["auto", "pow2_binomial", "random", "constant"],
        default=None,
        help="SCALE init distribution(s) (e8m0 for both operands)\n"
        "Default (unset): perf/profile = 'constant auto', func = 'auto'\n"
        "  auto          = E8M0 -> pow2_binomial          [default]\n"
        "  pow2_binomial = 2^(Binomial(21,0.5)-11)\n"
        "  random        = random e8m0 byte, exp in [-2,2]\n"
        "  constant      = neutral scale 0x7F (2^0 = 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed; same seed -> bit-identical data/scale buffers",
    )
    parser.add_argument(
        "--knl-name",
        dest="knl_name",
        default=None,
        help="dispatch mode. Default (unset) = heuristic: the aiter op picks the "
        ".co from mxfp8fp4gemm.csv by (b_intype, a_preshuffle) and shape. Any other "
        "value = force that exact mangled knl_name for all runs (developer debug).",
    )
    # intype x shape is a full product, so each shape is run for both a8w8/a8w4.
    parser.add_argument(
        "-s",
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="(M,N,K) tuples, e.g. -s 16384,16384,8192 128,16384,16384; "
        "unset uses PERF_SHAPES (perf/profile) or FUNC_SHAPES (func)",
    )
    args = parser.parse_args()

    # DATA and SCALE init are paired position-wise (NOT crossed). Mode-aware
    # defaults when unset: perf/profile run constant+constant and uniform+auto;
    # func drops the constant pair (its exact-boundary values trigger e8m0/e4m3
    # edge rounding -> spurious warnings) and runs just uniform+auto. A length-1
    # list broadcasts against the other axis.
    if args.mode == "func":
        default_di, default_si = ["uniform"], ["auto"]
    else:
        default_di, default_si = ["constant", "uniform"], ["constant", "auto"]
    di_list = args.data_init if args.data_init is not None else default_di
    si_list = args.scale_init if args.scale_init is not None else default_si
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

    # A-preshuffle sweep. Mode-aware default when unset: perf/profile exercise only
    # the preshuffled path ([1]); func sweeps both ([1, 0]).
    if args.apre is not None:
        apre_list = args.apre
    elif args.mode in ("perf", "profile"):
        apre_list = [1]
    else:
        apre_list = [1, 0]

    def shapes_for(intype):
        if args.shape is not None:
            return args.shape
        if args.mode == "func":
            return FUNC_SHAPES
        return PERF_SHAPES[intype]

    rows = [
        test_gemm(
            intype,
            M,
            N,
            K,
            apre,
            outtype,
            di,
            si,
            seed=args.seed,
            mode=args.mode,
            knl_name=args.knl_name,
        )
        for apre, (di, si), intype, outtype in itertools.product(
            apre_list, init_pairs, args.intype, args.outtype
        )
        for (M, N, K) in shapes_for(intype)
    ]
    df = pd.DataFrame(rows)
    # Keep knl_name (the actual .co); drop the columns constant within a table.
    df = df.drop(columns=["seed", "gfx", "mode"], errors="ignore")
    aiter.logger.info(
        "mxfp8fp4gemm (F8GEMM) summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    if args.mode == "profile":
        aiter.logger.info("profiler traces written under ./aiter_logs/")


if __name__ == "__main__":
    main()
