# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import math
import os
import re
import sys

# Add parent directory to path to ensure we use local aiter module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat as eirp

import aiter
from aiter import benchmark_data_init as bench_init
from aiter import dtypes
from aiter.benchmark_reporting import print_json_table
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck, gemm_a8w8_blockscale_cktile
from aiter.ops.shuffle import shuffle_mxfp8fp4_a, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest
from aiter.utility import fp4_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomic_splitk_config import atomic_config_path

block_shape = (128, 128)
TEST_NUM_ITERS = 100
WARMUP_ITERS = 5

# Set from __main__. Empty / False keep the plain tuned run.
SPLITK_MODES_TO_TEST = ()
SPLITK_AB = False
FLYDSL_KERNEL_OVERRIDE = None
# Split-K suffix sits between _cn<N> and _apre/_ps<N>.
_SPLITK_SUFFIX_RE = re.compile(
    r"(_cm\d+_cn\d+)(?:_nofsk|_fsk|_atm)?((?:_apre)?(?:_ps\d+)?)$"
)
_SPLITK_SUFFIX = {"atomic": "_atm", "fsk": "_fsk", "none": "_nofsk"}


def kernel_name_with_splitk_mode(name, mode):
    """Rewrite a tuned kernelName to request ``mode``; None if it does not parse."""
    if mode == "tuned":
        return name
    new, hits = _SPLITK_SUFFIX_RE.subn(
        lambda m: m.group(1) + _SPLITK_SUFFIX[mode] + m.group(2), name
    )
    return new if hits else None


def effective_splitk_mode(name, m, n):
    """What the runner will actually do with ``name`` -- the gate can narrow it."""
    from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
        is_compute_wmma_kernel_name,
        parse_wmma_kernel_name,
        resolve_splitk_mode,
    )

    cfg = parse_wmma_kernel_name(name)
    if cfg is None:
        return None
    return resolve_splitk_mode(
        m,
        n,
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["cluster_m"],
        cfg["cluster_n"],
        cfg["split_k"],
        is_compute_wmma_kernel_name(name),
        cfg["splitk_mode"],
    )


def _flydsl_row(m, n, k, path):
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config

    if get_gfx() != "gfx1250" or not os.path.exists(path):
        return None
    try:
        cfg = get_CKGEMM_config(m, n, k, path)
    except (FileNotFoundError, KeyError):
        return None
    if cfg is None or cfg.get("libtype") != "flydsl":
        return None
    return str(cfg.get("kernelName", "")) or None


def tuned_kernel_name(m, n, k, apre=False):
    """The kernelName mainline dispatch picks -- the fused (clustered) row."""
    from aiter.ops.gemm_op_a8w8 import AITER_CONFIGS

    path = (
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_ABPRESHUFFLE_FILE
        if apre
        else AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    )
    return _flydsl_row(m, n, k, path)


def atomic_kernel_name(m, n, k, apre=False):
    """The atomic winner from the separate atomic config; None if untuned."""
    return _flydsl_row(m, n, k, atomic_config_path(apre))


def tuned_flydsl_kernel_name(m, n, k):
    """Base name for --splitk-mode: the override if given, else the tuned row."""
    if FLYDSL_KERNEL_OVERRIDE is not None:
        return FLYDSL_KERNEL_OVERRIDE
    return tuned_kernel_name(m, n, k, apre=False)


def flydsl_kernel_call(
    x, weightshuffle, x_scale, w_scale, kernel_name, m, dtype, a_is_preshuffled=False
):
    """Allocates Out internally like the dispatch does, so perftest rotates the
    same argument set and these numbers stay comparable with the "us" column."""
    from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
        run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250,
    )

    out = torch.empty((m, weightshuffle.shape[0]), dtype=dtype, device=x.device)
    return run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        kernel_name,
        a_is_preshuffled=a_is_preshuffled,
    )


run_gemm_flydsl_kernel = perftest(num_iters=TEST_NUM_ITERS)(flydsl_kernel_call)


# atol as a fraction of the reference RMS. A fixed 1e-2 is a no-op once |out|
# reaches 1e4, leaving near-zero outputs judged on bare rtol -- a bar no
# bf16-output GEMM can clear.
ATOL_REL = 1e-2
# rel-L2 limit, as a multiple of the bf16 output-quantization floor. Unlike the
# mismatch ratio this is insensitive to how many outputs land near zero, so it
# is the gate that actually catches lost precision.
REL_L2_SLACK = 2.0


@perftest(num_iters=TEST_NUM_ITERS)
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    if x_scale.dtype == dtypes.fp8_e8m0:
        x_scale = fp4_utils.e8m0_to_f32(x_scale)
    if w_scale.dtype == dtypes.fp8_e8m0:
        w_scale = fp4_utils.e8m0_to_f32(w_scale)
    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    # fp32, not ``dtype``: the caller needs the un-rounded reference to derive
    # the bf16 quantization floor.
    return F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_bpreshuffle(x, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, weightshuffle, x_scale, w_scale, dtype
    )


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_abpreshuffle(
    x_shuffled, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16
):
    return aiter.gemm_a8w8_blockscale_abpreshuffle(
        x_shuffled, weightshuffle, x_scale, w_scale, dtype
    )


@perftest(num_iters=TEST_NUM_ITERS)
def run_triton(x, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16, backend=None):
    # Direct call into the triton preshuffle kernel, mirroring the dispatch in
    # gemm_a8w8_blockscale_bpreshuffle: reshape the (n, k) preshuffled weight to
    # (n // 16, k * 16) and pass the transposed x_scale.
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale_preshuffle,
    )

    n, k = weightshuffle.shape
    return gemm_a8w8_blockscale_preshuffle(
        x,
        weightshuffle.reshape(n // 16, k * 16),
        x_scale,
        w_scale,
        dtype=dtype,
        backend=backend,
    )


def rel_l2(y, ref_f32):
    """||y - ref|| / ||ref||, evaluated in fp64 so the metric itself is exact."""
    ref = ref_f32.double()
    denom = ref.norm().clamp(min=torch.finfo(torch.float64).tiny)
    return ((y.double() - ref).norm() / denom).item()


def check_rel_l2(y, ref_f32, floor, msg):
    """Gate rel-L2 against the bf16 floor; > REL_L2_SLACK means lost precision."""
    ratio = rel_l2(y, ref_f32) / max(floor, torch.finfo(torch.float32).tiny)
    tag = "\033[31mfailed!\033[0m" if ratio > REL_L2_SLACK else "\033[32mpassed~\033[0m"
    aiter.logger.info(
        f"{msg}[rel-L2 {ratio:.2f}x bf16 floor, limit {REL_L2_SLACK:.1f}x {tag}]"
    )
    return ratio


@benchmark()
def test_gemm(
    dtype,
    m,
    n,
    k,
    bpreshuffle=True,
    use_flydsl=False,
    data_init="uniform",
    scale_init="auto",
    seed=0,
    apre=False,
):
    ret = {}
    block_shape_n, block_shape_k = block_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    generator = bench_init.make_generator(seed)
    if scale_init == "amax":
        # Realistic MXFP8: each block's scale comes from its own amax, and the
        # 1/sqrt(k) weight init puts the output at O(1) like a real layer.
        x, x_scale_raw = bench_init.fill_mx_e8m0(
            (m, k), data_init, generator, block_n=1, block_k=block_shape_k
        )
        weight, w_scale_raw = bench_init.fill_mx_e8m0(
            (n, k),
            data_init,
            generator,
            block_n=block_shape_n,
            block_k=block_shape_k,
            std=1.0 / math.sqrt(k),
        )
    else:
        if data_init == "constant":
            x = torch.full((m, k), 0.5, dtype=dtypes.fp32, device="cuda").to(dtypes.fp8)
            weight = torch.full((n, k), 0.5, dtype=dtypes.fp32, device="cuda").to(
                dtypes.fp8
            )
        else:
            x = bench_init.fill_fp8((m, k), data_init, generator, dtype=dtypes.fp8)
            weight = bench_init.fill_fp8((n, k), data_init, generator, dtype=dtypes.fp8)

        if scale_init == "constant":
            x_scale_raw = torch.full(
                (scale_m, scale_k), 0x7F, dtype=torch.uint8, device="cuda"
            )
            w_scale_raw = torch.full(
                (scale_n, scale_k), 0x7F, dtype=torch.uint8, device="cuda"
            )
        else:
            x_scale_raw = bench_init.fill_scale_e8m0(
                (scale_m, scale_k), scale_init, generator
            )
            w_scale_raw = bench_init.fill_scale_e8m0(
                (scale_n, scale_k), scale_init, generator
            )
    use_flydsl_fp8_scale = use_flydsl and bpreshuffle
    if use_flydsl_fp8_scale:
        x_scale = x_scale_raw.view(dtypes.fp8_e8m0)
        w_scale = w_scale_raw.view(dtypes.fp8_e8m0)
    else:
        x_scale = fp4_utils.e8m0_to_f32(x_scale_raw)
        w_scale = fp4_utils.e8m0_to_f32(w_scale_raw)

    a_f32, _ = run_torch(x, weight, x_scale, w_scale, dtype)
    a = a_f32.to(dtype)
    atol = ATOL_REL * a_f32.pow(2).mean().sqrt().item()
    l2_floor = rel_l2(a, a_f32)

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    gemm_x_scale = x_scale_t if bpreshuffle else x_scale
    gemm_weight = shuffle_weight(weight, layout=(16, 16)) if bpreshuffle else weight
    run_func = run_gemm_bpreshuffle if bpreshuffle else run_gemm
    b, avg_b = run_func(x, gemm_weight, gemm_x_scale, w_scale, dtype)

    err_base = checkAllclose(
        a, b, msg="bpreshuffle", atol=atol, catastrophic_check=True
    )
    l2x_base = check_rel_l2(b, a_f32, l2_floor, "bpreshuffle")
    if bpreshuffle:
        x_scale_strided = x_scale.transpose(0, 1).contiguous().transpose(0, 1)
        b_strided = aiter.gemm_a8w8_blockscale_bpreshuffle(
            x, gemm_weight, x_scale_strided, w_scale, dtype
        )
        checkAllclose(
            a,
            b_strided,
            msg="bpreshuffle strided x_scale",
            atol=atol,
            catastrophic_check=True,
        )
    ret["us"] = avg_b
    ret["TFLOPS"] = m * n * k * 2 / avg_b / 1e6
    ret["TB/s"] = (x.nbytes + weight.nbytes) / avg_b / 1e6
    ret["err"] = err_base
    ret["l2x"] = l2x_base

    if use_flydsl_fp8_scale and SPLITK_MODES_TO_TEST:
        tuned_name = tuned_flydsl_kernel_name(m, n, k)
        if tuned_name is None:
            aiter.logger.warning(
                f"--splitk-mode: no tuned flydsl row for {m}x{n}x{k}, skipped"
            )
        else:
            for mode in SPLITK_MODES_TO_TEST:
                name = kernel_name_with_splitk_mode(tuned_name, mode)
                if name is None:
                    aiter.logger.warning(
                        f"--splitk-mode {mode}: cannot rewrite {tuned_name!r}"
                    )
                    continue
                eff = effective_splitk_mode(name, m, n)
                args = (x, gemm_weight, gemm_x_scale, w_scale, name, m, dtype)
                # Warm-up: without it the first mode measured is taxed 6-16%.
                for _ in range(WARMUP_ITERS):
                    flydsl_kernel_call(*args)
                torch.cuda.synchronize()
                g, avg_g = run_gemm_flydsl_kernel(*args)
                ret[f"sk:{mode} us"] = avg_g
                ret[f"sk:{mode} TFLOPS"] = m * n * k * 2 / avg_g / 1e6
                ret[f"sk:{mode} eff"] = eff
                ret[f"sk:{mode} err"] = checkAllclose(
                    a, g, msg=f"splitk={mode}", atol=atol, catastrophic_check=True
                )
                ret[f"sk:{mode}/bpre"] = avg_g / avg_b

    x_shuffled = None
    if apre and use_flydsl_fp8_scale:
        # A-preshuffle packs adjacent A row pairs, so an odd M needs A -- and only
        # A -- padded to M+1 rows; x_scale and the result keep the true M.
        if m % 2:
            x_apre = torch.zeros((m + 1, k), dtype=x.dtype, device=x.device)
            x_apre[:m] = x
        else:
            x_apre = x
        x_shuffled = shuffle_mxfp8fp4_a(x_apre)
        e, avg_e = run_gemm_abpreshuffle(
            x_shuffled, gemm_weight, gemm_x_scale, w_scale, dtype
        )
        ret["apre us"] = avg_e
        ret["apre TFLOPS"] = m * n * k * 2 / avg_e / 1e6
        ret["apre TB/s"] = (x_apre.nbytes + weight.nbytes) / avg_e / 1e6
        ret["apre err"] = checkAllclose(
            a, e, msg="apre", atol=atol, catastrophic_check=True
        )
        ret["apre l2x"] = check_rel_l2(e, a_f32, l2_floor, "apre")
        ret["apre/bpre"] = avg_e / avg_b

    if use_flydsl_fp8_scale and SPLITK_AB:
        variants = [(False, x, "")]
        if apre and x_shuffled is not None:
            variants.append((True, x_shuffled, "apre "))
        for is_apre, xin, pfx in variants:
            names = {
                "fsk": tuned_kernel_name(m, n, k, is_apre),
                "atm": atomic_kernel_name(m, n, k, is_apre),
            }
            got = {}
            for tag, name in names.items():
                if name is None:
                    aiter.logger.warning(
                        f"--splitk-ab: no {tag} row for {m}x{n}x{k} apre={int(is_apre)}"
                    )
                    continue
                args = (
                    xin,
                    gemm_weight,
                    gemm_x_scale,
                    w_scale,
                    name,
                    m,
                    dtype,
                    is_apre,
                )
                for _ in range(WARMUP_ITERS):
                    flydsl_kernel_call(*args)
                torch.cuda.synchronize()
                y, avg = run_gemm_flydsl_kernel(*args)
                got[tag] = avg
                ret[f"{pfx}{tag} us"] = avg
                ret[f"{pfx}{tag} TFLOPS"] = m * n * k * 2 / avg / 1e6
                eff = effective_splitk_mode(name, m, n)
                want = {"fsk": "fsk", "atm": "atomic"}[tag]
                if eff != want:
                    aiter.logger.warning(
                        f"--splitk-ab: {m}x{n}x{k} apre={int(is_apre)} {tag} asked "
                        f"for {want} but the gate ran {eff} -- not an A/B of the two "
                        f"epilogues; retune that row or pass a config that fits."
                    )
                ret[f"{pfx}{tag} eff"] = eff
                ret[f"{pfx}{tag} cfg"] = name.split("compute_wmma_")[-1]
                ret[f"{pfx}{tag} err"] = checkAllclose(
                    a, y, msg=f"{pfx}{tag}", atol=atol, catastrophic_check=True
                )
            if "fsk" in got and "atm" in got:
                ret[f"{pfx}atm/fsk"] = got["atm"] / got["fsk"]

    if not use_flydsl_fp8_scale:
        tag = "asm"
        weight_asm = shuffle_weight(weight, layout=(16, 16))
        c, avg_c = run_asm(x, weight_asm, x_scale_t, w_scale, dtype)

        err_asm = checkAllclose(a, c, msg=f"{tag}", atol=atol, catastrophic_check=True)
        ret[f"{tag} us"] = avg_c
        ret[f"{tag} TFLOPS"] = m * n * k * 2 / avg_c / 1e6
        ret[f"{tag} TB/s"] = (x.nbytes + weight.nbytes) / avg_c / 1e6
        ret[f"{tag} err"] = err_asm
        ret["asm/base"] = avg_c / avg_b

        # Triton path requires a preshuffled weight. When not preshuffled we simply omit
        # these columns; pd.DataFrame NaN-fills them for those rows in the summary.
        if bpreshuffle:
            d, avg_d = run_triton(x, gemm_weight, x_scale_t, w_scale, dtype)
            err_triton = checkAllclose(
                a, d, msg="triton", atol=atol, catastrophic_check=True
            )
            ret["triton us"] = avg_d
            ret["triton TFLOPS"] = m * n * k * 2 / avg_d / 1e6
            ret["triton TB/s"] = (x.nbytes + weight.nbytes) / avg_d / 1e6
            ret["triton err"] = err_triton
            ret["triton/bpre"] = avg_d / avg_b

    return ret


@perftest(num_iters=TEST_NUM_ITERS)
def run_torch2(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]

    x_scale_ = eirp(x_scale, "m k -> m (k repeat)", repeat=block_shape_k)
    x_scale_ = x_scale_[:m, :k]

    w_scale_ = eirp(w_scale, "n k -> (n repeat) k", repeat=block_shape_n)
    w_scale_ = eirp(w_scale_, "n k -> n (k repeat)", repeat=block_shape_k)
    w_scale_ = w_scale_[:n, :k]

    x_ = x.to(x_scale.dtype) * x_scale_
    weight_ = weight.to(w_scale.dtype) * w_scale_

    out = F.linear(x_.to(dtypes.fp32), weight_.to(dtypes.fp32))
    return out.to(dtype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_asm(x, weight, x_scale, w_scale, dtype=dtypes.bf16, kernel_name=None):
    m, _k = x.shape
    n, _ = weight.shape
    out = torch.empty((m, n), dtype=dtype, device=x.device)
    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(x, weight, out, x_scale, w_scale)


def test_splitk_correctness(m=4, n=2112, k=7168, dtype=dtypes.bf16, splitK=1):
    """Verify that splitK > 0 produces the same output as splitK=0 (within fp tolerance).

    split-K accumulates partial tiles via atomic_add, which changes the floating-point
    reduction order.  We therefore use a relaxed tolerance that matches the cumulative
    rounding error introduced by K-splitting.
    """
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    # CK path (no preshuffle): compare splitK=0 vs splitK>0
    Y_base = torch.empty((m, n), dtype=dtype, device="cuda")
    Y_split = torch.empty((m, n), dtype=dtype, device="cuda")
    gemm_a8w8_blockscale_ck(x, weight, x_scale, w_scale, Y_base, splitK=0)
    gemm_a8w8_blockscale_ck(x, weight, x_scale, w_scale, Y_split, splitK=splitK)
    ck_err = checkAllclose(
        Y_base,
        Y_split,
        msg=f"ck splitK={splitK} vs splitK=0",
        rtol=1e-2,
        atol=1e-2,
        catastrophic_check=True,
    )

    # CKTile path (no preshuffle): compare splitK=0 vs splitK>0
    Y_base_tile = torch.empty((m, n), dtype=dtype, device="cuda")
    Y_split_tile = torch.empty((m, n), dtype=dtype, device="cuda")
    gemm_a8w8_blockscale_cktile(
        x, weight, x_scale, w_scale, Y_base_tile, False, splitK=0
    )
    gemm_a8w8_blockscale_cktile(
        x, weight, x_scale, w_scale, Y_split_tile, False, splitK=splitK
    )
    cktile_err = checkAllclose(
        Y_base_tile,
        Y_split_tile,
        msg=f"cktile splitK={splitK} vs splitK=0",
        rtol=1e-2,
        atol=1e-2,
        catastrophic_check=True,
    )

    print(
        f"test_splitk_correctness(m={m}, n={n}, k={k}, splitK={splitK}): "
        f"ck_err={ck_err:.4g}, cktile_err={cktile_err:.4g}"
    )


# Kept in the JSON record and the results CSV, but dropped from the terminal
# table: constant across a run, already gated in the log, or too wide.
TABLE_DROP_COLS = ("dtype", "seed", "apre", "l2x", "apre l2x", "apre/bpre")
# Folded into the summary header when constant, kept as a column when swept.
TABLE_FOLD_COLS = ("bpreshuffle", "use_flydsl")


def display_view(frame):
    """Trim the results frame down to the terminal summary table.

    Returns ``(view, folded)``: the table, plus the ``{column: value}`` pairs
    that were constant and belong in the header.  ``data_init`` / ``scale_init``
    are swept as a pair, so they collapse to one ``init_mode`` column.
    """
    view = frame.rename(columns={"data_init": "init_mode"})
    drop = [c for c in ("scale_init",) + TABLE_DROP_COLS if c in view.columns]
    # kernelName columns are too wide for the terminal; the CSV/JSON keep them.
    drop += [c for c in view.columns if c.endswith(" cfg") and c not in drop]
    if SPLITK_AB:
        # The baseline columns just repeat what the "fsk" ones already report;
        # a narrowed split-K mode is warned about in the log instead.
        drop += [c for c in ("us", "TFLOPS", "TB/s", "err") if c in view.columns]
        drop += [c for c in view.columns if c.endswith(" eff") and c not in drop]
    folded = {}
    for col in TABLE_FOLD_COLS:
        if col in view.columns and view[col].nunique(dropna=False) == 1:
            folded[col] = view[col].iloc[0]
            drop.append(col)
    return view.drop(columns=drop), folded


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        512,
        1024,
        2048,
        4096,
        6144,
        8192,
        10240,
    ],
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-nk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[
        (24576, 1536),
        # (32768, 512),
        # (7168, 16384),
        # (36864, 7168),
    ],
    help="""N&K of mnk.
    e.g.: -nk 24576,1536""",
)
parser.add_argument(
    "--data-init",
    dest="data_init",
    nargs="+",
    choices=bench_init.DATA_DISTS,
    default=None,
    help="DATA initialization distribution(s), paired position-wise with "
    "--scale-init (length-1 broadcasts). Default: constant uniform",
)
parser.add_argument(
    "--scale-init",
    dest="scale_init",
    nargs="+",
    choices=bench_init.E8M0_SCALE_DISTS,
    default=None,
    help="E8M0 SCALE initialization distribution(s), paired position-wise "
    "with --data-init (length-1 broadcasts). 'amax' derives each block's "
    "scale from its own payload (realistic MXFP8). Default: constant amax",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="RNG seed for input, weight, and scales (default: 0)",
)
parser.add_argument(
    "--bpreshuffle",
    type=dtypes.str2bool,
    nargs="*",
    default=None,
    help="""preshuffle the B (weight) operand or not.
    e.g.: --bpreshuffle True
        or --bpreshuffle False
    """,
)
parser.add_argument(
    "--flydsl",
    action="store_true",
    help="use flydsl fp8 e8m0 scale path (requires --bpreshuffle True)",
)
parser.add_argument(
    "--apre",
    type=dtypes.str2bool,
    nargs="*",
    default=[False],
    help="""also measure the FlyDSL A-preshuffle candidate (requires --flydsl
    --bpreshuffle True). Odd M is padded to M+1 rows for A only.
    Sweeps like --bpreshuffle.
    e.g.: --apre True
        or --apre True False""",
)
parser.add_argument(
    "--splitk-ab",
    action="store_true",
    help="""Compare the two tuned split-K epilogues head to head: the fused
    (clustered) row mainline dispatch uses, vs the atomic winner from the
    separate atomic config (op_tests/tune_a8w8_splitk_atomic.py writes it;
    mainline never reads it). Adds "fsk/atm us|TFLOPS|eff|cfg|err" and
    "atm/fsk"; with --apre 1 the same columns appear again with an "apre "
    prefix. Requires --flydsl.""",
)
parser.add_argument(
    "--flydsl-kernel",
    dest="flydsl_kernel",
    type=str,
    default=None,
    help="""Base kernelName for --splitk-mode instead of the tuned row, so a
    config seen only in an e2e trace can be reproduced verbatim. The split-K
    suffix is still rewritten per --splitk-mode. Applies to every shape in the
    run, so pass one -nk at a time.
    e.g.: --flydsl-kernel flydsl_mxfp8_128_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb4_sk4_cm1_cn2""",
)
parser.add_argument(
    "--splitk-mode",
    dest="splitk_mode",
    nargs="+",
    choices=["tuned", "none", "atomic", "fsk"],
    default=None,
    help="""Also rerun the FlyDSL candidate with these split-K epilogues, using
    the SAME tile as the tuned row (only the epilogue changes), so atomic and
    clustered-fsk can be compared directly. "none" is the separate reduce
    kernel, "tuned" is the row as written. Repeatable; each mode adds
    "sk:<mode> us/TFLOPS/eff/err" columns. The "eff" column is what the dispatch
    gate actually ran -- it can narrow the request. Requires --flydsl.
    e.g.: --splitk-mode atomic fsk none""",
)
parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="""CSV file containing M, N, K columns (one shape per row).
    e.g.: --csv shapes.csv""",
)
parser.add_argument(
    "--json",
    action="store_true",
    help="""Also print the summary as a single-line JSON record, for parent
    benchmark drivers that consume it. The default output is the table.""",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="""Directory to save results CSV.
    e.g.: -o results/""",
)
parser.add_argument(
    "--suffix",
    type=str,
    default="results",
    help="""Suffix for output CSV filename.
    e.g.: --suffix branch""",
)

args = parser.parse_args()

if args.flydsl_kernel:
    if not args.splitk_mode:
        parser.error("--flydsl-kernel requires --splitk-mode")
    FLYDSL_KERNEL_OVERRIDE = args.flydsl_kernel
if args.splitk_ab:
    if not args.flydsl:
        parser.error("--splitk-ab requires --flydsl")
    SPLITK_AB = True
if args.splitk_mode:
    if not args.flydsl:
        parser.error("--splitk-mode requires --flydsl")
    seen = dict.fromkeys(args.splitk_mode)  # de-dup, keep order
    SPLITK_MODES_TO_TEST = tuple(seen)

data_init_list = args.data_init or ["constant", "norm"]
scale_init_list = args.scale_init or ["constant", "amax"]
if len(data_init_list) == 1:
    data_init_list *= len(scale_init_list)
if len(scale_init_list) == 1:
    scale_init_list *= len(data_init_list)
if len(data_init_list) != len(scale_init_list):
    parser.error(
        "--data-init and --scale-init must have equal length (or length 1 to broadcast)"
    )
init_pairs = list(zip(data_init_list, scale_init_list))

if args.bpreshuffle is None:
    args.bpreshuffle = [True] if args.flydsl else [True, False]
l_preshuffle = (
    args.bpreshuffle if isinstance(args.bpreshuffle, list) else [args.bpreshuffle]
)
l_apre = args.apre if isinstance(args.apre, list) else [args.apre]

df = []
if args.csv is not None:
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    shapes_df = pd.read_csv(args.csv)
    print(f"Loaded {len(shapes_df)} shapes from {args.csv}", flush=True)
    for dtype in args.dtype:
        for preshuffle in l_preshuffle:
            for data_init, scale_init in init_pairs:
                for apre in l_apre:
                    for _, row in shapes_df.iterrows():
                        ret = test_gemm(
                            dtype,
                            int(row["M"]),
                            int(row["N"]),
                            int(row["K"]),
                            bpreshuffle=preshuffle,
                            use_flydsl=args.flydsl,
                            data_init=data_init,
                            scale_init=scale_init,
                            seed=args.seed,
                            apre=apre,
                        )
                        df.append(ret)
else:
    for dtype in args.dtype:
        for m in args.m:
            for n, k in args.nk:
                for bpre in l_preshuffle:
                    for apre in l_apre:
                        for data_init, scale_init in init_pairs:
                            ret = test_gemm(
                                dtype,
                                m,
                                n,
                                k,
                                bpreshuffle=bpre,
                                use_flydsl=args.flydsl,
                                data_init=data_init,
                                scale_init=scale_init,
                                seed=args.seed,
                                apre=apre,
                            )
                            df.append(ret)

df = pd.DataFrame([row for row in df if row is not None])
if args.json:
    print_json_table("gemm_a8w8_blockscale summary", df)
if not df.empty:
    table, folded = display_view(df)
    header = "PERFORMANCE SUMMARY"
    if folded:
        header += "   [" + "  ".join(f"{c}={v}" for c, v in folded.items()) + "]"
    print("\n" + "=" * 150)
    print(header)
    print("-" * 150)
    if SPLITK_AB:
        print(
            "  fsk = mainline tuned row   atm = atomic config   "
            "atm/fsk = atm us / fsk us"
        )
        print("  a narrowed split-K mode is warned about in the log above")
    else:
        print("  init_mode                : input distribution (data and block scale)")
        print("  us / TFLOPS / TB/s / err : B-preshuffle GEMM (weight preshuffled)")
        print("  apre us / apre TFLOPS /  : A-preshuffle + B-preshuffle GEMM")
        print("  apre TB/s / apre err       (A and weight both preshuffled)")
        print(
            "  err                      : fraction of elements outside "
            "rtol=1e-2 / atol=1e-2*RMS(ref)"
        )
        print(
            f"  rel-L2 is gated separately at {REL_L2_SLACK:.1f}x the bf16 "
            f"output-quantization floor; see the per-case lines in the log above."
        )
    print("=" * 150)
    print(table.to_string(index=False))
    print("=" * 150)

# Correctness check: verify split-K produces matching results
print("\nRunning split-K correctness checks ...")
for splitK in [1, 2]:
    test_splitk_correctness(m=4, n=512, k=16384, splitK=splitK)

# Save results from benchmarks
if args.output:
    os.makedirs(args.output, exist_ok=True)
    if args.csv:
        csv_filename = os.path.basename(args.csv).replace(".csv", f"_{args.suffix}.csv")
    else:
        csv_filename = f"gemm_a8w8_blockscale_{args.suffix}.csv"
    out_path = os.path.join(args.output, csv_filename)
    df.to_csv(out_path, index=False)
    print(f"Saved results to: {out_path}")
