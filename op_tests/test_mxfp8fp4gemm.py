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
import csv
import itertools
import sys
import time
from pathlib import Path

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.mxfp8fp4gemm_common import mxfp8fp4_gemm_splitk
from aiter.ops.shuffle import (
    shuffle_mxfp8fp4_a,
    shuffle_mxfp8fp4_b,
    shuffle_mxfp8fp4_scale,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    fill_fp4,
    fill_fp8,
    fill_scale_e8m0,
    make_generator,
    run_perftest,
)
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)

SUPPORTED_GFX = ["gfx1250"]

_OUT_DTYPE = {"bf16": dtypes.bf16}

# gfx1250 F8GEMM .co is a persistent shader: it always launches PERSISTENT_TG
# threadgroups regardless of problem size (must match the .co's WG_MAX).
PERSISTENT_TG = 256


# Per-tile cluster sets (cluster_x, cluster_y) deployed in mxfp8fp4gemm.csv, kept in
# sync with the cover in poc_kl/mi400/mxfp8fp4gemm/run.sh COVER_CONFIGS so the reported
# label names the .co the cpp heuristic (get_heuristic_kernel) actually dispatches to.
_CLUSTERS = {
    (128, 128): [(4, 4)],
    (256, 256): [(4, 4), (2, 4), (4, 2), (2, 2), (1, 1)],
    (64, 512): [(4, 1), (2, 1), (1, 1)],
    (16, 512): [(4, 1), (2, 1), (1, 1)],
}


def _tiles_for(intype, apre):
    """Tiles registered in the csv for this combo. 16x512 is FP4-only (its master
    asserts B_DTYPE_FP4) and is deployed with a_preshuffle=0 only."""
    tiles = [(256, 256), (64, 512)]
    if intype == "a8w8" and apre:
        tiles.append((128, 128))
    if intype == "a8w4" and not apre:
        tiles.append((16, 512))
    return tiles


def _heuristic_tile(M, N, K, intype, apre):
    """Tile (tile_m, tile_n) the cpp dispatch picks (mirrors get_heuristic_kernel in
    asm_mxfp8fp4gemm.cu): a tiny M wastes a taller tile's rows, so M<=16 prefers the
    16x512 decode tile, M<=64 the 64x512 one, larger M 256x256 -- restricted to the
    tiles actually registered for this (intype, apre). The FP8/AP1 indexer shape
    (512,8192,1536) prefers the 128x128 K128/PF8 variant."""
    if (M, N, K) == (512, 8192, 1536) and intype == "a8w8" and apre:
        prefs = [(128, 128), (256, 256), (64, 512)]
    elif M <= 16:
        prefs = [(16, 512), (64, 512), (256, 256)]
    elif M <= 64:
        prefs = [(64, 512), (256, 256)]
    else:
        prefs = [(256, 256), (64, 512)]
    avail = _tiles_for(intype, apre)
    return next(t for t in prefs if t in avail)


def _heuristic_cluster(tile_m, tile_n, M, N):
    """(cluster_x, cluster_y) the cpp dispatch picks within the tile: the largest
    cluster that DIVIDES the tile grid evenly, tie-break by aspect closeness then
    larger cx. 1x1 always fits (n % 1 == 0), so a pick always exists.

    Divisibility, not just cx<=ntiles: a ragged last block leaves its trailing lanes
    with an out-of-range tile id, which the kernel clamps onto the last valid tile
    (redundant work). Must stay in sync with `fits` in asm_mxfp8fp4gemm.cu."""
    mtiles = (M + tile_m - 1) // tile_m
    ntiles = (N + tile_n - 1) // tile_n
    best, best_key = (1, 1), None
    for cx, cy in _CLUSTERS.get((tile_m, tile_n), [(1, 1)]):
        score = cx * cy if (ntiles % cx == 0 and mtiles % cy == 0) else 0
        key = (score, -abs(cx * mtiles - cy * ntiles), cx)
        if best_key is None or key > best_key:
            best, best_key = (cx, cy), key
    return best


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

# Defaults for the six a8w8/AP1 cases used by the native GEMM-only benchmark.
# An explicit --splitk (including 0 for the operator heuristic) takes precedence.
F8GEMM_BENCHMARK_SPLITK = {
    (512, 2048, 7168): 8,  # wqkv_a
    (512, 7168, 16384): 4,  # wo_b
    (512, 6144, 7168): 4,  # gate_up_proj
    (512, 7168, 3072): 4,  # w2
    (512, 65536, 1536): 1,  # wq_b
    (512, 8192, 1536): 1,  # indexer_wq_b
}


def _benchmark_splitk(intype, apre, M, N, K):
    if intype == "a8w8" and apre == 1:
        return F8GEMM_BENCHMARK_SPLITK.get((M, N, K), 0)
    return 0


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
    (512, 8192, 1536),  # K128/PF8 128x128 automatic dispatch
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


def _ref(intype, A, B, sA, sB, M, N, splitk=1):
    # FP32 reference; optionally benchmarked before the formal ASM measurement.
    A_f32 = A.to(torch.float32)[:M]
    if intype == "a8w4":
        B_f32 = fp4_utils.mxfp4_to_f32(B)[:N]
    else:
        B_f32 = B.to(torch.float32)[:N]
    sA_f = fp4_utils.e8m0_to_f32(sA).repeat_interleave(MX_SCALE_BLOCK, dim=1)
    sB_f = fp4_utils.e8m0_to_f32(sB).repeat_interleave(MX_SCALE_BLOCK, dim=1)
    lhs, rhs = A_f32 * sA_f, B_f32 * sB_f
    if splitk > 1:
        assert lhs.shape[1] % splitk == 0
        step = lhs.shape[1] // splitk
        return torch.stack(
            [
                lhs[:, s * step : (s + 1) * step] @ rhs[:, s * step : (s + 1) * step].T
                for s in range(splitk)
            ]
        )
    return lhs @ rhs.T


def _prep(
    intype: str,
    M: int,
    N: int,
    K: int,
    apre: int,
    data_init: str,
    scale_init: str,
    gen,
    reference_splitk=1,
    pre_benchmark=False,
    pre_benchmark_warmup=2,
    pre_benchmark_iters=100,
):
    """Build raw + shuffled device tensors and the f32 golden reference.

    DATA and SCALE are sampled *independently* (test_common), selected by
    ``data_init`` / ``scale_init``:
      data_init  : uniform (FP8 U(-6,6) / FP4 U(-3,3)) [default] | norm |
                   zero | constant (A/B = 0.5)
      scale_init : auto (E8M0 -> pow2_binomial) [default] | pow2_binomial |
                   zero | uniform | norm | constant (neutral 0x7F -> 2^0 = 1.0)
    """
    # DATA: A is mxfp8 (e4m3); B is mxfp4 (e2m1 packed) for a8w4, else mxfp8.
    # constant is built inside fill_fp8/fill_fp4 via constant=: A/B = 0.5, and
    # the a8w4 fp4 B uses the e2m1 nibble byte 0x11 (= 0.5).
    A = fill_fp8((M, K), data_init, gen, constant=0.5)
    if intype == "a8w4":
        B = fill_fp4((N, K), data_init, gen, constant=0x11)
    else:
        B = fill_fp8((N, K), data_init, gen, constant=0.5)

    # SCALE: e8m0 per-32 for both operands. auto -> pow2_binomial for E8M0;
    # constant default is the neutral e8m0 byte 0x7F (2^0 = 1.0).
    sA = fill_scale_e8m0((M, K // MX_SCALE_BLOCK), scale_init, gen)
    sB = fill_scale_e8m0((N, K // MX_SCALE_BLOCK), scale_init, gen)

    # fp32 golden; the caller casts/quantizes it to the requested outtype.
    pre_results = {}
    if pre_benchmark:
        aiter.logger.info("prebenchmark begin: native Torch reference")
        torch_start_ns = time.monotonic_ns()
        ref_f32, ref_us = run_perftest(
            _ref,
            intype,
            A,
            B,
            sA,
            sB,
            M,
            N,
            reference_splitk,
            num_warmup=pre_benchmark_warmup,
            num_iters=pre_benchmark_iters,
            testGraph=False,
            num_rotate_args=0,
        )
        torch_end_ns = time.monotonic_ns()
        pre_results["stage_windows"] = {
            "torch": {"start_mono_ns": torch_start_ns, "end_mono_ns": torch_end_ns}
        }
        pre_results["torch us"] = ref_us
        pre_results["torch profile_gpu_kernels"] = dict(run_perftest.last_gpu_kernels)
    else:
        ref_f32 = _ref(intype, A, B, sA, sB, M, N, reference_splitk)

    inp = {
        "A": shuffle_mxfp8fp4_a(A) if apre else A,  # B always preshuffled, A per `apre`
        "B": shuffle_mxfp8fp4_b(B),
        "sA": shuffle_mxfp8fp4_scale(sA),
        "sB": shuffle_mxfp8fp4_scale(sB),
    }
    if pre_benchmark:
        from aiter.ops.gemm_op_a8w8 import _mxfp8_mxfp8_gemm_asm

        aiter.logger.info("prebenchmark begin: native ASM AP0")
        pre_output = torch.empty(
            (reference_splitk, M, N) if reference_splitk > 1 else (M, N),
            dtype=dtypes.bf16,
            device=A.device,
        )

        def run_pre_ap0(a, b, scale_a, scale_b):
            _mxfp8_mxfp8_gemm_asm(
                a, b, scale_a, scale_b, pre_output, None, 0, reference_splitk
            )
            return pre_output

        ap0_start_ns = time.monotonic_ns()
        pre_out, pre_us = run_perftest(
            run_pre_ap0,
            A,
            inp["B"],
            inp["sA"],
            inp["sB"],
            num_warmup=pre_benchmark_warmup,
            num_iters=pre_benchmark_iters,
            testGraph=False,
            num_rotate_args=0,
        )
        ap0_end_ns = time.monotonic_ns()
        pre_results["stage_windows"]["ap0"] = {
            "start_mono_ns": ap0_start_ns,
            "end_mono_ns": ap0_end_ns,
        }
        pre_results["ap0 us"] = pre_us
        pre_results["ap0 profile_gpu_kernels"] = dict(run_perftest.last_gpu_kernels)
        pre_err = checkAllclose(
            ref_f32.to(dtypes.bf16).to(dtypes.fp32),
            pre_out.to(dtypes.fp32),
            rtol=1e-1,
            atol=1.0,
            msg="prebenchmark AP0",
        )
        pre_results["ap0 err"] = pre_err
        pre_results["ap0 result"] = _verdict(pre_err)
        aiter.logger.info("prebenchmark complete: continuing to formal ASM AP1")
    return inp, ref_f32, pre_results


@benchmark()
def test_gemm(
    intype,
    M,
    N,
    K,
    apre=1,
    outtype="bf16",
    data_init="uniform",
    scale_init="auto",
    seed=0,
    mode="perf",
    knl_name=None,
    splitk=None,
    no_reduce=False,
    num_warmup=2,
    num_iters=None,
    test_graph=False,
    num_rotate=0,
    pre_benchmark=False,
    pre_benchmark_warmup=2,
    pre_benchmark_iters=100,
):
    if splitk is None:
        splitk = _benchmark_splitk(intype, apre, M, N, K)
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
        _tm, _tn = _heuristic_tile(M, N, K, intype, apre)
        _cx, _cy = _heuristic_cluster(_tm, _tn, M, N)
        return {
            "gfx": get_gfx(),
            "knl_name": knl_name or "(heuristic)",
            "tile": f"{_tm}x{_tn}",
            "cluster": f"{_cx}x{_cy}",
            "splitk": splitk or 1,
            "asm us": float("nan"),
            "asm TFLOPS": float("nan"),
            "asm TB/s": float("nan"),
            "asm err": float("nan"),
            "asm result": f"not support ({reason})",
        }

    assert K % MX_SCALE_BLOCK == 0, f"K must be a multiple of {MX_SCALE_BLOCK}"
    out_dtype = _OUT_DTYPE[outtype]
    gen = make_generator(seed)  # fixed seed -> bit-identical buffers
    if no_reduce and (intype != "a8w8" or splitk < 1):
        raise ValueError(
            "--no-reduce requires a8w8 and a positive split-K; pass --splitk for shapes without a benchmark default"
        )
    if pre_benchmark and not (
        intype == "a8w8"
        and apre == 1
        and no_reduce
        and splitk > 0
        and pre_benchmark_warmup >= 0
        and pre_benchmark_iters > 1
    ):
        raise ValueError(
            "--pre-benchmark requires a8w8/AP1/--no-reduce/positive split-K and valid pre-benchmark counts"
        )
    inp, ref_f32, pre_results = _prep(
        intype,
        M,
        N,
        K,
        apre,
        data_init,
        scale_init,
        gen,
        reference_splitk=splitk if no_reduce else 1,
        pre_benchmark=pre_benchmark,
        pre_benchmark_warmup=pre_benchmark_warmup,
        pre_benchmark_iters=pre_benchmark_iters,
    )
    ref = ref_f32.to(out_dtype)
    needTrace = mode == "profile"
    # --iters overrides; unset keeps the mode default (func=5, perf/profile=100).
    num_iters = num_iters if num_iters is not None else (5 if mode == "func" else 100)

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
        _tm, _tn = _heuristic_tile(M, N, K, intype, apre)
        _cx, _cy = _heuristic_cluster(_tm, _tn, M, N)
        base = f"f8gemm_{outtype}_{middle}_{pre}_{_tm}x{_tn}_{_cx}x{_cy}_ps"
        knl = f"_ZN5aiter{len(base)}{base}E"
    else:
        knl = knl_name

    if no_reduce:
        from aiter.ops.gemm_op_a8w8 import _mxfp8_mxfp8_gemm_asm

        # One output buffer, as in the POC host. Allocation and reference are untimed.
        partials = torch.empty(
            (splitk, M, N) if splitk > 1 else (M, N),
            dtype=out_dtype,
            device=inp["A"].device,
        )

    def run_asm(A, B, sA, sB):
        if no_reduce:
            _mxfp8_mxfp8_gemm_asm(
                A, B, sA, sB, partials, knl or None, int(apre), splitk
            )
            return partials
        return kern(
            A,
            B,
            sA,
            sB,
            dtype=out_dtype,
            a_preshuffle=bool(apre),
            kernelName=knl,
            splitk=splitk,
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
    if pre_benchmark:
        ret["pre_benchmark_results"] = pre_results
    ret["reference_splitk"] = splitk if no_reduce else 1
    # Report TG occupancy for the tile+cluster the cpp dispatch picks.
    _middle = "mxfp8fp8" if intype == "a8w8" else "mxfp8fp4"
    _pre = "ABpreShuffle" if apre else "BpreShuffle"
    _tile_m, _tile_n = _heuristic_tile(M, N, K, intype, apre)
    _cx, _cy = _heuristic_cluster(_tile_m, _tile_n, M, N)
    if knl:
        catalog = (
            Path(__file__).resolve().parents[1]
            / "hsa/gfx1250/mxfp8fp4gemm/mxfp8fp4gemm.csv"
        )
        with catalog.open() as source:
            cfg = next(
                (row for row in csv.DictReader(source) if row["knl_name"] == knl), None
            )
        if cfg is not None:
            _tile_m, _tile_n = int(cfg["tile_m"]), int(cfg["tile_n"])
            _cx, _cy = int(cfg["cluster_x"]), int(cfg["cluster_y"])
    _label = f"f8gemm_{outtype}_{_middle}_{_pre}_{_tile_m}x{_tile_n}_{_cx}x{_cy}_ps"
    _report_active_tg(M, N, _tile_m, _tile_n, _label)
    # Structured algo details (mxfp8fp4gemm.csv columns): the cpp-dispatch tile,
    # the aspect-selected cluster and the split-K count the cpp picks (asked for
    # rather than mirrored -- choose_splitk owns the constraints).
    ret["tile"] = f"{_tile_m}x{_tile_n}"
    ret["cluster"] = f"{_cx}x{_cy}"
    ret["splitk"] = splitk or mxfp8fp4_gemm_splitk(
        M, N, K, int(intype == "a8w4"), apre, knl or None
    )
    ret["reduced"] = not no_reduce and ret["splitk"] > 1
    ret["timing_scope"] = "gemm_with_reduce" if ret["reduced"] else "gemm_only"
    # Only a missing .co is reported as "not support"; any other failure (OOM,
    # memory fault, shape assert, ...) must propagate, not show as a green cell.
    # An explicit --knl-name that isn't in the cfg is a real error (typo / missing
    # build), so "kernel not in cfg" is benign ONLY on the heuristic path (knl == "").
    # An explicit --splitk the kernel cannot honour for this shape (odd count, K not
    # divisible, more than WG_MAX TGs, non-256x256 tile) is a skip, not an error --
    # a sweep runs the same count against every shape.
    _NOT_SUPPORTED_MARKERS = ("cannot get heuristic kernel", "is not valid for")
    if not knl:
        _NOT_SUPPORTED_MARKERS += ("kernel not in cfg_mxfp8fp4gemm",)
    for name, (cand, cand_args) in candidates.items():
        try:
            if pre_benchmark:
                ap1_start_ns = time.monotonic_ns()
            out, us = run_perftest(
                cand,
                *cand_args,
                num_iters=num_iters,
                num_warmup=num_warmup,
                testGraph=test_graph,
                num_rotate_args=num_rotate,
                needTrace=needTrace,
            )
            if pre_benchmark:
                ap1_end_ns = time.monotonic_ns()
                pre_results["stage_windows"]["ap1"] = {
                    "start_mono_ns": ap1_start_ns,
                    "end_mono_ns": ap1_end_ns,
                }
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
        ret["profile_gpu_kernels"] = run_perftest.last_gpu_kernels
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
        default=[1],
        help="A-preshuffle sweep list: 1 preshuffles A (M%%2), 0 sends it "
        "row-major (M%%1). Default: [1]. Pass --apre 1 0 to sweep both.",
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
        choices=["zero", "constant", "uniform", "norm", "poc"],
        default=None,
        help="DATA init distribution(s) (sampled independently of scale).\n"
        "Paired position-wise with --scale-init (length-1 broadcasts).\n"
        "Default (unset): perf/profile = 'constant uniform', func = 'uniform'\n"
        "  zero     = all-zero on-wire codes\n"
        "  constant = A/B = 0.5 (deterministic)\n"
        "  uniform  = FP8 U(-6,6) / FP4 U(-3,3)  [default]\n"
        "  norm     = N(0,1)                     [norm-dist / LLM-like]\n"
        "  poc      = {+-0.5,1,1.5,2,3} random    [matches poc perf harness]",
    )
    parser.add_argument(
        "--scale-init",
        dest="scale_init",
        nargs="*",
        choices=["auto", "pow2_binomial", "zero", "constant", "uniform", "norm", "poc"],
        default=None,
        help="SCALE init distribution(s) (e8m0 for both operands)\n"
        "Default (unset): perf/profile = 'constant auto', func = 'auto'\n"
        "  auto          = E8M0 -> pow2_binomial          [default]\n"
        "  pow2_binomial = 2^(Binomial(21,0.5)-11)\n"
        "  zero          = all-zero e8m0 bytes\n"
        "  constant      = neutral scale 0x7F (2^0 = 1.0)\n"
        "  uniform       = U(0.5,2) -> nearest e8m0 byte\n"
        "  norm          = N(1,0.25) -> nearest e8m0 byte\n"
        "  poc           = 2^[-2,2] (exp+127)              [matches poc perf harness]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed; same seed -> bit-identical data/scale buffers",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="warmup iterations before timing (run_perftest num_warmup)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="timed iterations (run_perftest num_iters); unset -> mode default "
        "(func=5, perf/profile=100)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="also time via HIP graph replay (run_perftest testGraph), "
        "minimizing inter-kernel gaps",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        help="rotating input-buffer copies to defeat the L2 hot-cache "
        "(run_perftest num_rotate_args); 0 = auto-size from L2",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="also write the full summary table (all columns) as JSON records "
        "to this path, for CI/regression",
    )
    parser.add_argument(
        "--knl-name",
        dest="knl_name",
        default=None,
        help="dispatch mode. Default (unset) = heuristic: the aiter op picks the "
        ".co from mxfp8fp4gemm.csv by (b_intype, a_preshuffle) and shape. Any other "
        "value = force that exact mangled knl_name for all runs (developer debug).",
    )
    parser.add_argument(
        "--splitk",
        type=int,
        nargs="*",
        default=None,
        help="split-K counts to run. Unset: six a8w8/AP1 benchmark shapes use "
        "8/4/4/4/1/1; other shapes use the operator heuristic. "
        "Explicit 0 always uses the count choose_splitk picks. Several "
        "values sweep them, e.g. --splitk 1 2 4 8. Only the kernel's hard "
        "constraints are checked, so a count deeper than the dispatch would pick "
        "is allowed; 256x256 only.",
    )
    parser.add_argument(
        "--no-reduce",
        action="store_true",
        help="Time ASM GEMM only and validate each split-K plane independently",
    )
    parser.add_argument(
        "--pre-benchmark",
        action="store_true",
        help="Before each AP1 test, benchmark the native Torch reference and native ASM AP0",
    )
    parser.add_argument("--pre-benchmark-warmup", type=int, default=2)
    parser.add_argument("--pre-benchmark-iters", type=int, default=100)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the complete preparation/prebenchmark/formal test per configuration",
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
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.iters is None:
        args.iters = 5 if args.mode == "func" else 100

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

    apre_list = args.apre

    def shapes_for(intype):
        if args.shape is not None:
            return args.shape
        if args.mode == "func":
            return FUNC_SHAPES
        return PERF_SHAPES[intype]

    def splitks_for(intype, apre, M, N, K):
        if args.splitk is not None:
            return args.splitk
        return [_benchmark_splitk(intype, apre, M, N, K)]

    def run_once(repeat, *test_args, **test_kwargs):
        aiter.logger.info("repeat %d/%d begin", repeat, args.repeat)
        row = dict(test_gemm(*test_args, **test_kwargs), repeat=repeat)
        aiter.logger.info(
            "repeat %d/%d result: %s M,N,K=%s,%s,%s %s/%s AP%s splitk=%s "
            "asm=%s us, %s TFLOPS, correctness=%s, err=%s, GPU events=%s/%s",
            repeat,
            args.repeat,
            row["intype"],
            row["M"],
            row["N"],
            row["K"],
            row["data_init"],
            row["scale_init"],
            row["apre"],
            row.get("splitk"),
            row.get("asm us"),
            row.get("asm TFLOPS"),
            row.get("asm result"),
            row.get("asm err"),
            sum(row.get("profile_gpu_kernels", {}).values()),
            row["num_iters"],
        )
        return row

    rows = [
        run_once(
            repeat,
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
            splitk=splitk,
            no_reduce=args.no_reduce,
            num_warmup=args.warmup,
            num_iters=args.iters,
            test_graph=args.graph,
            num_rotate=args.rotate,
            pre_benchmark=args.pre_benchmark,
            pre_benchmark_warmup=args.pre_benchmark_warmup,
            pre_benchmark_iters=args.pre_benchmark_iters,
        )
        for apre, (di, si), intype, outtype in itertools.product(
            apre_list, init_pairs, args.intype, args.outtype
        )
        for (M, N, K) in shapes_for(intype)
        for splitk in splitks_for(intype, apre, M, N, K)
        for repeat in range(1, args.repeat + 1)
    ]
    df_full = pd.DataFrame(rows)
    # JSON keeps every column (config + algo details + results) so each record is
    # self-describing for CI/regression; the markdown table below drops columns
    # that are constant within a run for readability.
    if args.json_out:
        df_full.to_json(args.json_out, orient="records", indent=2)
        aiter.logger.info(
            "wrote JSON summary (%d rows) to %s", len(df_full), args.json_out
        )
    # Keep knl_name (the actual .co) + tile + cluster; drop columns constant within a
    # table (cluster now varies with shape via the aspect heuristic, so it is kept).
    # Nested profiling details stay in JSON; omit them from the terminal table.
    df = df_full.drop(
        columns=[
            "seed",
            "gfx",
            "mode",
            "num_warmup",
            "num_iters",
            "test_graph",
            "num_rotate",
            "pre_benchmark_results",
            "profile_gpu_kernels",
        ],
        errors="ignore",
    )
    aiter.logger.info(
        "mxfp8fp4gemm (F8GEMM) summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    if args.mode == "profile":
        aiter.logger.info("profiler traces written under ./aiter_logs/")


if __name__ == "__main__":
    main()
