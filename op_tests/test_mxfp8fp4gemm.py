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
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_dispatch_loop,
    run_perftest,
)
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


def _heuristic_kernel_base(outtype, intype, apre, M):
    """Mangled-name base the heuristic .co uses for this config, matching the CSV
    convention (hsa/gfx1250/mxfp8fp4gemm/mxfp8fp4gemm.csv). BOTH the tile and the
    cluster depend on M: M<=64 -> 64x512 with cluster 4x1, else 256x256 with
    cluster 4x4 (the cluster is NOT 4x4 for the 64x512 tile). Used for the
    TG-occupancy label so it names the .co the heuristic actually dispatches to."""
    middle = "mxfp8fp8" if intype == "a8w8" else "mxfp8fp4"
    pre = "ABpreShuffle" if apre else "BpreShuffle"
    tile, cluster = ("64x512", "4x1") if M <= 64 else ("256x256", "4x4")
    return f"f8gemm_{outtype}_{middle}_{pre}_{tile}_{cluster}_ps"


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
    intype: str,
    M: int,
    N: int,
    K: int,
    apre: int,
    data_init: str,
    scale_init: str,
    gen,
    need_ref: bool = True,
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

    # fp32 golden; the caller casts/quantizes it to the requested outtype. Skipped
    # when need_ref is False (--skip-ref, or clk/ttrace mode): the golden is a
    # ~34 GB fp32 N-by-K matmul (compute-bound) that would dominate wall time / VRAM
    # and, in clk mode, pull the sampled clocks up to compute freqs.
    ref_f32 = _ref(intype, A, B, sA, sB, M, N) if need_ref else None

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
    clk_seconds=15.0,
    num_iters_arg=None,
    skip_ref=False,
):
    # mode selects the harness path (see main's --mode help):
    #   func/perf/profile -> torch.profiler device-time timing (run_perftest)
    #   clk               -> short profiled burst + unprofiled soak for clk_trace.py
    #   ttrace            -> plain dispatch loop for an external rocprofv3 --att wrap
    clk_mode = mode == "clk"
    under_rocprof = mode == "ttrace"
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
    # Golden skipped when --skip-ref, or implicitly in clk/ttrace mode (clk: the
    # ~34GB fp32 ref would pull sampled clocks up to compute freqs; ttrace:
    # rocprofv3 traces only the filtered f8gemm kernel, so the golden is just extra
    # untraced work). clk/ttrace also don't check accuracy.
    need_ref = not (skip_ref or clk_mode or under_rocprof)
    inp, ref_f32 = _prep(
        intype, M, N, K, apre, data_init, scale_init, gen, need_ref=need_ref
    )
    ref = ref_f32.to(out_dtype) if ref_f32 is not None else None
    needTrace = mode == "profile"
    if num_iters_arg is not None:
        num_iters = num_iters_arg
    else:
        num_iters = 5 if mode == "func" else 101

    # Single ASM kernel under test, dispatched by intype. Inputs passed as ARGS so
    # run_perftest can rotate them (defeats the L2 hot-cache).
    kern = aiter.gemm_a8w4_mxfp8 if intype == "a8w4" else aiter.gemm_a8w8_mxfp8
    # Dispatch: default (knl_name=None) -> knl="" lets the op pick the .co by
    # (b_intype, a_preshuffle) AND M (M<=64 -> 64x512, else 256x256). An explicit
    # --knl-name forces that exact mangled name from the CSV verbatim (dev debug).
    knl = knl_name or ""

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
    _tile_m, _tile_n = _heuristic_tile(M)
    _label = _heuristic_kernel_base(outtype, intype, apre, M)
    _report_active_tg(M, N, _tile_m, _tile_n, _label)

    if under_rocprof:
        # ---- ttrace: rocprofv3 --att feeder (plain dispatch loop) ----
        # run_dispatch_loop uses no torch.profiler / cuda-event / per-iter sync --
        # all of those deadlock with ATT's HSA-queue interception. Just launch the
        # kernel enough times for --kernel-iteration-range to catch it; timing/BW
        # come from a separate clk/perf run, so report nan.
        run_asm, cand_args = candidates["asm"]
        n = num_iters_arg if num_iters_arg is not None else 101
        out, wall_us = run_dispatch_loop(run_asm, *cand_args, num_iters=n, num_warmup=3)
        # WALL-CLOCK throughput (incl. host launch overhead + gaps). Meaningful only
        # when run BARE; under rocprofv3 --att the SQTT trace inflates it. Reported
        # so a bare `--mode ttrace` still yields a rough flops/BW; for the real
        # device-time number use --mode perf/clk. Labeled *wall to avoid confusion.
        io_bytes = in_bytes + out.nbytes
        ret["asm us"] = round(wall_us, 2)
        ret["asm TFLOPS"] = round(flops / wall_us / 1e6, 1)
        ret["asm TB/s"] = round(io_bytes / wall_us / 1e6, 2)
        ret["asm err"] = float("nan")
        ret["asm result"] = "ttrace(wall)"
        aiter.logger.warning(
            "ttrace: %d dispatches; us/TFLOPS/TB-s are WALL-CLOCK (incl. launch "
            "overhead, ATT-inflated under rocprofv3) -- use perf/clk for device time",
            n,
        )
        return ret

    if clk_mode:
        # ---- Clock-decoupling mode (DPM experiment) ----
        # Two phases: (1) a SHORT torch-profiled burst for accurate device-time
        # TFLOPS/BW (same path as normal perf), then (2) an UNPROFILED back-to-back
        # soak for clk_seconds so an external clk_trace.py --wait-pid samples a steady
        # clock window. Golden is skipped (memory-bound decode: the ~34GB fp32 ref
        # would dominate the timeline and pull the clocks up). Reusing the same args
        # every iter is fine: B is far larger than L2, so every iter re-reads from HBM.
        import time as _time

        run_asm, cand_args = candidates["asm"]
        try:
            for _ in range(3):  # warmup: JIT/dispatch, reach steady DPM/thermal
                out = run_asm(*cand_args)
            torch.cuda.synchronize()
        except Exception as e:
            if not any(m in str(e) for m in ("cannot get heuristic kernel",)):
                raise
            aiter.logger.warning("clk-mode: no dispatchable kernel: %s", e)
            ret["asm us"] = float("nan")
            ret["asm TFLOPS"] = float("nan")
            ret["asm TB/s"] = float("nan")
            ret["asm err"] = float("nan")
            ret["asm result"] = "not support"
            return ret

        # (1) Per-iter DEVICE time via a SHORT torch-profiled burst -- the SAME path
        # normal perf uses (run_perftest -> torch profiler -> get_trace_perf), so the
        # TFLOPS/BW match the standard test and exclude host launch overhead. Only
        # ~num_iters events are buffered. Profiling the WHOLE soak instead would buffer
        # one event per launch (1e5-1e6) -> OOM + a giant DataFrame in get_trace_perf,
        # which is exactly why the soak below stays unprofiled.
        out, us = run_perftest(run_asm, *cand_args, num_iters=101, num_warmup=2)
        io_bytes = in_bytes + out.nbytes
        ret["asm us"] = round(us, 2)
        ret["asm TFLOPS"] = round(flops / us / 1e6, 1)
        ret["asm TB/s"] = round(io_bytes / us / 1e6, 2)
        ret["asm err"] = float("nan")
        ret["asm result"] = "clk-mode"

        # (2) Soak: launch back-to-back for clk_seconds so an external clk_trace can
        # sample a steady clock window -- and time it (wall-clock) to also report a
        # SUSTAINED throughput. Unlike the phase-1 device-time number (measured cool,
        # kernel-only), the soak figure is measured in the SAME thermal/clock window
        # the sampler sees (so it's consistent with the reported clock, incl. any
        # throttle) and INCLUDES host launch overhead + inter-kernel gaps -> it is the
        # realistic sustained decode rate, always <= the phase-1 device-time upper
        # bound. Same args every iter is fine: B >> L2, so every iter re-reads HBM.
        CHUNK = 500
        done = 0
        t0 = _time.perf_counter()
        while _time.perf_counter() - t0 < clk_seconds:
            for _ in range(CHUNK):
                out = run_asm(*cand_args)
            done += CHUNK
            torch.cuda.synchronize()
            if num_iters_arg is not None and done >= num_iters_arg:
                break
        wall = _time.perf_counter() - t0
        if wall > 0 and done > 0:
            soak_us = wall / done * 1e6
            ret["asm soak us"] = round(soak_us, 2)
            ret["asm soak TFLOPS"] = round(flops / soak_us / 1e6, 1)
            ret["asm soak TB/s"] = round(io_bytes / soak_us / 1e6, 2)
        aiter.logger.info(
            "clk-mode: device us=%.2f (%.1f TFLOPS, %.2f TB/s, cool/kernel-only via "
            "torch.profiler); soak %d iters/%.1fs -> %.2f us (%.2f TB/s sustained, "
            "same window as clk_trace, golden skipped)",
            us,
            ret["asm TFLOPS"],
            ret["asm TB/s"],
            done,
            wall,
            ret.get("asm soak us", float("nan")),
            ret.get("asm soak TB/s", float("nan")),
        )
        return ret

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
        err = (
            checkAllclose(
                ref.to(dtypes.fp32),
                out.to(dtypes.fp32),
                rtol=1e-1,
                atol=1.0,
                msg=f"{intype} {name}",
            )
            if ref is not None
            else float("nan")  # golden skipped (--skip-ref / clk / ttrace)
        )
        io_bytes = in_bytes + out.nbytes
        ret[f"{name} us"] = round(us, 2)
        ret[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        ret[f"{name} TB/s"] = round(io_bytes / us / 1e6, 2)
        ret[f"{name} err"] = err
        # ref is None under --skip-ref: _verdict(nan) would read as "failed", so
        # report the skip explicitly instead.
        ret[f"{name} result"] = _verdict(err) if ref is not None else "no-ref"
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
        choices=["func", "perf", "profile", "clk", "ttrace"],
        default="perf",
        help="test harness path (one axis; clk/ttrace are no longer separate flags):\n"
        "  func    = accuracy only\n"
        "  perf    = accuracy + device-time timing (torch.profiler)\n"
        "  profile = perf + torch.profiler trace under ./aiter_logs/\n"
        "  clk     = DPM clock soak: short profiled burst for cool device-time\n"
        "            TFLOPS/BW, then a timed unprofiled soak for --clk-seconds so an\n"
        "            external clk_trace.py --wait-pid samples sysfs clocks at 1ms.\n"
        "            Also reports 'soak' wall-clock sustained TFLOPS/BW measured in\n"
        "            that same (throttle-exposed) window. Golden skipped. See\n"
        "            clk_mxfp8fp4gemm.sh.\n"
        "  ttrace  = plain dispatch loop with the internal torch.profiler OFF so an\n"
        "            external rocprofv3 --att wrap does not collide. Golden skipped;\n"
        "            us/TFLOPS/TB-s are wall-clock (ATT-inflated under rocprofv3).\n"
        "            See ttrace_mxfp8fp4gemm.sh.",
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
        "row-major (M%%1). Default (unset): func = [1, 0], all other modes = [1].",
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
        ".co from mxfp8fp4gemm.csv by (b_intype, a_preshuffle) and M (M<=64 -> "
        "64x512, else 256x256). Any value = force that exact mangled knl_name from "
        "the CSV verbatim for all runs (developer debug).",
    )
    # intype x shape is a full product, so each shape is run for both a8w8/a8w4.
    parser.add_argument(
        "-s",
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="(M,N,K) tuples, e.g. -s 16384,16384,8192 128,16384,16384; unset uses "
        "FUNC_SHAPES (func), a single memory-bound representative (clk/ttrace), or "
        "PERF_SHAPES (perf/profile)",
    )
    parser.add_argument(
        "--clk-seconds",
        dest="clk_seconds",
        type=float,
        default=8.0,
        help="clk-mode soak duration in seconds (default 8). The external "
        "clk_trace.py samples sysfs at 1ms and drops the ramp via --settle, so a few "
        "seconds already yield a stable clock median; kept SHORT on purpose so large "
        "shapes / long runs don't heat into thermal throttle. Raise it only if you "
        "specifically want to observe the throttle onset.",
    )
    parser.add_argument(
        "--num-iters",
        dest="num_iters",
        type=int,
        default=None,
        help="override kernel iteration count. func/perf/profile: replaces the "
        "default 101 (perf/profile) / 5 (func). clk: optional hard cap on soak "
        "iters (still stops at --clk-seconds, whichever first). ttrace: number of "
        "plain dispatches (default 101).",
    )
    parser.add_argument(
        "--metrics-json",
        dest="metrics_json",
        default=None,
        help="also write the summary table (one record per row, incl. clk-mode "
        "'soak' throughput) to this JSON path, for clk_merge.py to join against a "
        "clk_trace.py CSV. Works in any mode; most useful with --mode clk.",
    )
    parser.add_argument(
        "--skip-ref",
        dest="skip_ref",
        action="store_true",
        help="skip the fp32 golden reference (accuracy NOT checked; err/result show "
        "no-ref). Auto-on for clk/ttrace. Opt-in for func/perf/profile to cut wall "
        "time and VRAM on large compute-bound shapes.",
    )
    args = parser.parse_args()

    # DATA and SCALE init are paired position-wise (NOT crossed). Mode-aware
    # defaults when unset: perf/profile run constant+constant and uniform+auto;
    # func drops the constant pair (its exact-boundary values trigger e8m0/e4m3
    # edge rounding -> spurious warnings) and runs just uniform+auto. A length-1
    # list broadcasts against the other axis.
    if args.mode == "func":
        default_di, default_si = ["uniform"], ["auto"]
    elif args.mode in ("clk", "ttrace"):
        # clk/ttrace don't check accuracy, and the clock/trace is data-independent
        # for these memory-bound decode shapes -> a single deterministic pair (no
        # point multiplying the soak/trace budget across init distributions).
        default_di, default_si = ["constant"], ["auto"]
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

    # A-preshuffle sweep. Mode-aware default when unset: func sweeps both ([1, 0]);
    # every other mode exercises only the preshuffled path ([1]).
    if args.apre is not None:
        apre_list = args.apre
    elif args.mode == "func":
        apre_list = [1, 0]
    else:
        apre_list = [1]

    def shapes_for(intype):
        if args.shape is not None:
            return args.shape
        if args.mode == "func":
            return FUNC_SHAPES
        if args.mode in ("clk", "ttrace"):
            # DPM decode experiment: sweep only the memory-bound (folded-BS)
            # representative. The compute-bound shape would just burn the soak /
            # trace budget and (in clk) pull clocks up to compute freqs.
            return [PERF_SHAPES[intype][-1]]
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
            clk_seconds=args.clk_seconds,
            num_iters_arg=args.num_iters,
            skip_ref=args.skip_ref,
        )
        for apre, (di, si), intype, outtype in itertools.product(
            apre_list, init_pairs, args.intype, args.outtype
        )
        for (M, N, K) in shapes_for(intype)
    ]
    df = pd.DataFrame(rows)
    if args.metrics_json:
        # Full records (incl. M/N/K/intype and clk-mode soak metrics) for clk_merge.
        import json

        with open(args.metrics_json, "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
        aiter.logger.info("metrics written to %s", args.metrics_json)
    # Keep knl_name (the actual .co); drop the columns constant within a table.
    df = df.drop(columns=["seed", "gfx", "mode"], errors="ignore")
    aiter.logger.info(
        "mxfp8fp4gemm (F8GEMM) summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    if args.mode == "profile":
        aiter.logger.info("profiler traces written under ./aiter_logs/")
    elif args.mode == "ttrace":
        aiter.logger.info(
            "ttrace: rerun under rocprofv3 --att to capture the thread trace "
            "(see ttrace_mxfp8fp4gemm.sh); this bare run only emits the dispatches"
        )
    elif args.mode == "clk":
        aiter.logger.info(
            "clk: pair with clk_trace.py --wait-pid to sample sysfs clocks "
            "(see clk_mxfp8fp4gemm.sh)"
        )


if __name__ == "__main__":
    main()
