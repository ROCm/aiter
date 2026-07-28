# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# A4W4 (F4GEMM) test/benchmark for gfx1250. One timed candidate ("asm") per
# (intype, shape, apre) row; a torch fp32 reference is compared but never timed.
# Default dispatch is heuristic (the aiter op picks the .co from f4gemm.csv by
# (intype, a_preshuffle, outtype)); --knl-name forces an explicit kernel.
#   MXFP4: e8m0 per-32 scales;  NVFP4: e4m3 per-16 scales + per-tensor globals

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a4w4 import MXFP8_OUT_SCALE_BLOCK, unpack_mxfp8_out_scale
from aiter.ops.shuffle import shuffle_scale_f4, shuffle_weight_f4
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

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

SUBK = 256  # asm inner-K step: the ONLY hard shape constraint is K % SUBK == 0

PERF_SHAPES = [(16384, 16384, 16384)]
FUNC_SHAPES = [
    (1024, 1024, 256),
    (1024, 1024, 512),
    (1024, 1024, 768),
    (1024, 1024, 1280),
    (2048, 1024, 256),
    (1024, 2048, 768),
    (2048, 2048, 2048),
    (4096, 4096, 512),
    (1024, 5120, 256),
    (1024, 6144, 256),
    (1024, 7168, 256),
    (5120, 1024, 256),
    (3072, 8192, 256),
    (5120, 5120, 256),
]

MXFP4_SCALE_BLOCK = 32
NVFP4_SCALE_BLOCK = 16
# MXFP8_OUT_SCALE_BLOCK (=128) is imported from gemm_op_a4w4.

# mxfp8 output E8M0 reference: RoundUp (ceil(amax/448)), compared with atol=1
# (kernel rounds within +-1 step). RoundUp keeps ref data <= 448; Even/RNE can
# round the scale down and overflow e4m3 to NaN, so must NOT be used here.
MXFP8_SCALE_MODE = MxScaleRoundModeInt.RoundUp
MXFP8_SCALE_ATOL = 1.0  # +-1 e8m0 step


def _quant_mxfp8_blockN(x_f32, block=MXFP8_OUT_SCALE_BLOCK):
    """Golden mxfp8 output quant: per-128-col block amax -> E8M0 scale via
    fp4_utils.f32_to_mx_e8m0_scale (RoundUp, FP8_E4M3) + e4m3 data. Returns
    (fp8 [M,N], e8m0 row-major [M, N/block])."""
    M, N = x_f32.shape
    assert N % block == 0, f"mxfp8 golden requires N % {block} == 0"
    xb = x_f32.reshape(M, N // block, block)
    amax = xb.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    scale_e8m0 = fp4_utils.f32_to_mx_e8m0_scale(
        amax, mode=MXFP8_SCALE_MODE, dtype=MxDtypeInt.FP8_E4M3
    )
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0).unsqueeze(-1)
    q_fp8 = (xb / scale_f32).reshape(M, N).to(dtypes.fp8)
    return q_fp8, scale_e8m0


def _dequant_mxfp8_blockN(q_fp8, scale_e8m0, block=MXFP8_OUT_SCALE_BLOCK):
    """Inverse of :func:`_quant_mxfp8_blockN`; ``scale_e8m0`` must be row-major."""
    M, N = q_fp8.shape
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0).unsqueeze(-1)
    return (q_fp8.float().reshape(M, N // block, block) * scale_f32).reshape(M, N)


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
    inp = {
        "A": shuffle_weight_f4(xq) if apre else xq,
        "B": shuffle_weight_f4(wq),
        "sA": shuffle_scale_f4(xs, 7),
        "sB": shuffle_scale_f4(ws, 7),
        "gA": None,
        "gB": None,
    }
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
    inp = {
        "A": shuffle_weight_f4(xq) if apre else xq,
        "B": shuffle_weight_f4(wq),
        "sA": shuffle_scale_f4(xs, 8),
        "sB": shuffle_scale_f4(ws, 8),
        "gA": gA,  # NVFP4 per-tensor global scales (floats)
        "gB": gB,
    }
    return inp, ref


@benchmark()  # (intype, M, N, K, apre, outtype, data_init, scale_init, seed) -> cols
def test_gemm(
    intype,
    M,
    N,
    K,
    apre,
    outtype,
    data_init,
    scale_init,
    seed=0,
    mode="perf",
    dtype=dtypes.bf16,
    knl_name=None,
):
    block = MXFP4_SCALE_BLOCK if intype == "mxfp4" else NVFP4_SCALE_BLOCK
    assert K % block == 0, f"K must be a multiple of {block}"
    # Only the packed-fp4 .co is noscale (cvt_scale=1); bf16/fp8 use scales.
    # Scale tensors are still built (API-required) but ignored when noscale.
    noscale = outtype == "fp4"
    out_fp4 = outtype == "fp4"
    out_fp8 = outtype == "fp8"
    out_dtype = dtypes.fp4x2 if out_fp4 else (dtypes.fp8 if out_fp8 else dtype)
    gen = bench_init.make_generator(seed)  # fixed seed -> bit-identical buffers
    prep = _prep_mxfp4 if intype == "mxfp4" else _prep_nvfp4
    inp, ref_f32 = prep(M, N, K, apre, data_init, scale_init, gen, noscale=noscale)
    # Reference in the kernel's output form: packed e2m1 for fp4, block-scaled
    # (fp8 e4m3 data + e8m0 scale) tuple for fp8, else bf16.
    if out_fp4:
        ref = fp4_utils.f32_to_mxfp4(ref_f32)
    elif out_fp8:
        ref = _quant_mxfp8_blockN(ref_f32)  # (ref_fp8, ref_scale_e8m0)
    else:
        ref = ref_f32.to(dtype)
    needTrace = mode == "profile"
    num_iters = 5 if mode == "func" else 101

    # Kernel/.co base name for this config (used for logging, and to derive the
    # mangled knl_name when an explicit dispatch is requested). See
    # hsa/gfx1250/f4gemm/f4gemm.csv.
    pre = "ABpreShuffle" if apre else "BpreShuffle"
    ns = "_noscale" if noscale else ""
    base = f"f4gemm_{outtype}_{intype}_{pre}_256x256_4x4_ps{ns}"

    # Dispatch mode. Default (knl_name=None) is heuristic: kernelName="" lets the
    # aiter op pick the .co from f4gemm.csv by (intype, a_preshuffle, outtype), so
    # the test validates the op's dispatch. Explicit is opt-in via --knl-name:
    # "auto" uses the per-config derived name below; any other value is used verbatim.
    if knl_name is None:
        knl = ""
    elif knl_name == "auto":
        knl = f"_ZN5aiter{len(base)}{base}E"
    else:
        knl = knl_name

    # Pass inputs as ARGS so run_perftest can rotate them (defeats the L2 hot-cache).
    if intype == "nvfp4":

        def run_asm(A, B, sA, sB, gA, gB):
            return aiter.gemm_nvfp4_asm(
                A,
                B,
                sA,
                sB,
                gA,
                gB,
                dtype=out_dtype,
                a_preshuffle=bool(apre),
                kernelName=knl,
            )

        asm_args = (inp["A"], inp["B"], inp["sA"], inp["sB"], inp["gA"], inp["gB"])
    else:

        def run_asm(A, B, sA, sB):
            return aiter.gemm_mxfp4_asm(
                A,
                B,
                sA,
                sB,
                dtype=out_dtype,
                a_preshuffle=bool(apre),
                kernelName=knl,
            )

        asm_args = (inp["A"], inp["B"], inp["sA"], inp["sB"])

    # Only the low-level asm entry is timed/tabled. (fn, args); args are rotated.
    candidates = {"asm": (run_asm, asm_args)}

    flops = 2 * M * N * K
    # Output bytes: packed fp4 = M*N/2; fp8 = M*N (fp8) + M*N/128 (e8m0 scale);
    # bf16 = M*N*itemsize.
    if out_fp4:
        out_bytes = (M * N) // 2
    elif out_fp8:
        out_bytes = M * N + M * (N // MXFP8_OUT_SCALE_BLOCK)
    else:
        out_bytes = M * N * dtype.itemsize
    nbytes = (
        inp["A"].nbytes
        + inp["B"].nbytes
        + inp["sA"].nbytes
        + inp["sB"].nbytes
        + out_bytes
    )

    # Report the actual .co in the table: readable base name for heuristic/"auto",
    # the verbatim knl_name otherwise (kept in the table, see main()).
    actual_knl = knl_name if (knl_name and knl_name != "auto") else base
    ret = {"gfx": get_gfx(), "knl_name": actual_knl}
    # Only a missing .co is reported as "not support"; any other failure (OOM,
    # memory fault, shape assert, ...) must propagate, not show as a green cell.
    _NOT_SUPPORTED_MARKERS = (
        "cannot get heuristic kernel",
        "kernel not in cfg_f4gemm",
    )
    for name, (fn, fn_args) in candidates.items():
        try:
            out, us = run_perftest(
                fn, *fn_args, num_iters=num_iters, needTrace=needTrace
            )
        except Exception as e:
            if not any(m in str(e) for m in _NOT_SUPPORTED_MARKERS):
                raise
            # No .co for this config (e.g. nvfp4-fp4); mark unsupported, keep going.
            aiter.logger.warning(
                "f4gemm not supported: intype=%s outtype=%s noscale=%s apre=%s "
                "M=%s N=%s K=%s [%s.co]: %s",
                intype,
                outtype,
                noscale,
                apre,
                M,
                N,
                K,
                base,
                e,
            )
            ret[f"{name} us"] = float("nan")
            ret[f"{name} TFLOPS"] = float("nan")
            ret[f"{name} TB/s"] = float("nan")
            ret[f"{name} err"] = float("nan")
            ret[f"{name} result"] = "not support"
            continue
        # Func-mode only: check the high-level op contracts by outtype -- bf16 ->
        # gemm_a4w4 (single tensor), fp4 -> gemm_a4w4o4 (single tensor [.,N//2]),
        # fp8 -> gemm_a4w4o8 ((data, scale) tuple). Not timed/tabled.
        if mode == "func":
            a4_kwargs = {"apreshuffle": bool(apre)}
            if intype == "nvfp4":
                # global scales are Optional[Tensor] (schema); a non-None value
                # selects the NVFP4 path -- wrap the float scalars as tensors.
                a4_kwargs.update(
                    global_A_scale=torch.tensor(inp["gA"], device=inp["A"].device),
                    global_B_scale=torch.tensor(inp["gB"], device=inp["A"].device),
                )
            args = (inp["A"], inp["B"], inp["sA"], inp["sB"])
            if out_fp8:
                o, s = aiter.gemm_a4w4o8(*args, **a4_kwargs)
                assert o.shape == out[0].shape and s.shape == out[1].shape, (
                    f"gemm_a4w4o8 shape mismatch: {tuple(o.shape)}/{tuple(s.shape)} "
                    f"vs {tuple(out[0].shape)}/{tuple(out[1].shape)}"
                )
            elif out_fp4:
                res = aiter.gemm_a4w4o4(*args, **a4_kwargs)
                assert not isinstance(res, tuple), "gemm_a4w4o4 must return a tensor"
                assert res.shape == out.shape, (
                    f"gemm_a4w4o4 shape mismatch: {tuple(res.shape)} vs {tuple(out.shape)}"
                )
            else:  # bf16
                res = aiter.gemm_a4w4(*args, dtype=out_dtype, **a4_kwargs)
                assert not isinstance(res, tuple), "gemm_a4w4 must return a tensor"
                assert res.shape == out.shape, (
                    f"gemm_a4w4 shape mismatch: {tuple(res.shape)} vs {tuple(out.shape)}"
                )
        if out_fp4:
            # e2m1 is deterministic: compare dequantized values with zero
            # tolerance (exact fp4-code match). Borderline RNE ties may differ.
            err = checkAllclose(
                fp4_utils.mxfp4_to_f32(ref),
                fp4_utils.mxfp4_to_f32(out),
                rtol=0,
                atol=0,
                msg=f"{intype} {name} fp4",
            )
        elif out_fp8:
            # (fp8 data, packed e8m0). Unpack scale to row-major; judge e8m0 with
            # atol=1 (kernel within +-1 of RNE) and dequant data with tolerance.
            ref_fp8, ref_scale = ref
            o_fp8, o_scale = out  # o_* avoids shadowing the out_fp8 flag
            M_out, N_out = o_fp8.shape
            out_scale_rm = unpack_mxfp8_out_scale(o_scale, M_out, N_out)
            err_s = checkAllclose(
                ref_scale.view(torch.uint8).float(),
                out_scale_rm.view(torch.uint8).float(),
                rtol=0,
                atol=MXFP8_SCALE_ATOL,
                msg=f"{intype} {name} fp8 e8m0 (+-1)",
            )
            err_d = checkAllclose(
                _dequant_mxfp8_blockN(ref_fp8, ref_scale),
                _dequant_mxfp8_blockN(o_fp8, out_scale_rm),
                rtol=1e-1,
                atol=1.0,
                msg=f"{intype} {name} fp8",
            )
            err = max(err_s, err_d)
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
        "--outtype",
        nargs="*",
        choices=["bf16", "fp8", "fp4"],
        default=["bf16", "fp8"],
        help="output-format sweep list (default: bf16 fp8):\n"
        "  bf16 = bf16 [M,N]\n"
        "  fp8  = fp8 e4m3 [M,N] + per-128 E8M0 scale (mxfp8)\n"
        "  fp4  = packed e2m1 [M,N//2] (noscale; mxfp4 only)",
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
        "(func drops constant: its exact-boundary values trigger e8m0/e4m3\n"
        "edge rounding that shows as spurious warnings).\n"
        "  uniform  = FP4 U(-3,3)\n"
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
        default=None,
        help="SCALE init distribution(s) (by scale format)\n"
        "Default (unset): perf/profile = 'constant auto', func = 'auto'\n"
        "  auto          = format-recommended: mxfp4/E8M0 -> pow2_binomial,\n"
        "                  nvfp4/E4M3 -> gaussian(0.34375,0.08)\n"
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
        "--knl-name",
        dest="knl_name",
        default=None,
        help="dispatch mode. Default (unset) = heuristic: the aiter op picks the "
        ".co from f4gemm.csv by (intype, a_preshuffle, outtype), validating dispatch. "
        "'auto' = force the per-config derived knl_name (explicit). Any other value "
        "= use that exact mangled knl_name for all runs (developer experiment/debug).",
    )
    parser.add_argument(
        "--mode",
        choices=["func", "perf", "profile"],
        default="perf",
        help="func=acc+timing table (fewer iters), perf=acc+timing table, profile=perf+trace",
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
        # Unset -> per-mode defaults: perf=PERF_SHAPES (one big square, throughput),
        # func=FUNC_SHAPES (many small/odd shapes, correctness). K must be %SUBK.
        default=None,
        help="(M,N,K) tuples, e.g. -mnk 2048,2048,2048 16384,16384,16384; "
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

    # Shapes: explicit --shape wins; otherwise func sweeps the many small/odd
    # correctness shapes and perf/profile sweep the single throughput square.
    if args.shape is not None:
        shapes = args.shape
    else:
        shapes = FUNC_SHAPES if args.mode == "func" else PERF_SHAPES

    for dtype in args.dtype:  # one table per output dtype
        # init pair is the OUTERMOST product term -> rows are grouped by
        # (data_init,scale_init) within the single summary table.
        rows = [
            test_gemm(
                intype,
                M,
                N,
                K,
                apre,
                ot,
                di,
                si,
                seed=args.seed,
                mode=args.mode,
                dtype=dtype,
                knl_name=args.knl_name,
            )
            for (di, si), intype, apre, ot, (M, N, K) in itertools.product(
                init_pairs, args.intype, args.apre, args.outtype, shapes
            )
        ]
        df = pd.DataFrame(rows)
        # Keep knl_name (the actual .co); drop the columns constant within a table.
        df = df.drop(columns=["seed", "dtype", "gfx", "mode"], errors="ignore")
        aiter.logger.info(
            "gemm_a4w4 (F4GEMM) summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
