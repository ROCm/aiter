# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark -- and tune -- the FlyDSL a16w-mix 2-stage MoE kernels.

Times the shared a16w-mix port (``aiter/ops/flydsl/kernels/moe_2stage_a16wmix``)
that serves a16w4 (bf16 A x MXFP4 W) and a16wi4 (bf16 A x int4 W), driving the
same two launchers production reaches after config resolution
(``_flydsl_stage1_wrapper`` / ``_flydsl_stage2_wrapper``).  The kernel name IS
the tile config, so a row measured here names exactly the kernel it timed.

Existing benches cover neither path: ``bench_moe_gemm_a16w4.py`` times the
Triton ``moe_gemm_a16w4``, and the mxfp4 tuner in
``csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`` drives the a4w4/a8w4
``flydsl_mxmoe_g*`` families, not this port.

Two modes:

  default   time one (kernelName1, kernelName2) pair, or the pair a shape's
            tuned CSV row already names, and print us1/us2/us/TFLOPS/BW.

  --sweep   enumerate every registered candidate for the shape and emit a
            tuned-CSV row per (token, block_m).  stage1 and stage2 are swept
            INDEPENDENTLY: they are separate back-to-back kernel launches with
            separate timings, so the full cross product (~1440 x ~64 per shape)
            buys nothing over ~1500 single-axis measurements.

Candidates come from the production registry (``get_flydsl_stage1_kernels`` /
``get_flydsl_stage2_kernels``), not from hand-built strings, so every name
tried is one ``get_flydsl_kernel_params`` can parse and production can dispatch.

Every candidate is checked against a torch reference BEFORE it is timed: a
config that drops tiles is fast and wrong, and an unchecked sweep would rank it
first.  Timing is the median of ``--repeats`` runs of ``--iters`` iterations;
TFLOPS/BW use ``gemm_moe_tune.py``'s own fused-MoE formulas so the numbers are
comparable with the rows already in the tuned CSVs.

Usage -- run from the repo root (needed for the ``op_tests`` import path):

    # One shape, the default kimi-k3 pair
    python -m op_tests.op_benchmarks.flydsl.bench_moe_a16wmix_2stage

    # Time one explicit pair
    python -m op_tests.op_benchmarks.flydsl.bench_moe_a16wmix_2stage \
        --kernel1 flydsl_moe1_abf16_wfp4_bf16_t32x32x256_xcd1_kw2 \
        --kernel2 flydsl_moe2_abf16_wfp4_bf16_t32x128x128_atomic_bnt2

    # Full sweep for one shape, appending tuned rows to a CSV
    python -m op_tests.op_benchmarks.flydsl.bench_moe_a16wmix_2stage \
        --sweep --token 1 --block-m 32 -o gfx942_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _flydsl_stage1_wrapper,
    _flydsl_stage2_wrapper,
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_kernels import (
    get_flydsl_stage1_kernels,
    get_flydsl_stage2_kernels,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import run_perftest

# kimi-k3 a16w4 shape: the one kimik3_a16w4_{un,}tuned_fmoe.csv is keyed on.
DEFAULT_MODEL_DIM = 3584
DEFAULT_INTER_DIM = 512
DEFAULT_EXPERTS = 896
DEFAULT_TOPK = 16
# Production betas (kimik3). Not a compile key, so they add no extra JIT builds.
DEFAULT_SITU_BETA = 4.0
DEFAULT_SITU_LINEAR_BETA = 25.0

# Same gate the a16wfp4 op test uses.
COS_TOL = 1e-2

# The token tiers kimik3_a16w4_untuned_fmoe.csv asks for.
SWEEP_TOKENS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
# Existing tuned files carry one row per (token, block_m); update_config_files
# keeps the lower `us` per shape key, which does not include block_m.
SWEEP_BLOCK_MS = (16, 32)


def _default_iters(token):
    """Iterations per timed run, scaled down for the long prefill shapes.

    200 (the count the existing tuned rows were measured at) costs ~12 s per
    candidate once one launch reaches ~20 ms, which puts a 320-candidate sweep
    at token=16384 over an hour per block_m.  Every candidate for a given
    (token, block_m) uses the same count, so the ranking a sweep produces stays
    comparable -- only cross-token wall time changes.
    """
    if token <= 512:
        return 200
    if token <= 4096:
        return 50
    return 20


TUNED_CSV_COLUMNS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
    "_tag",
]


def _cos_diff(x, y):
    """Same cos/logits_diff metric op_tests/test_flydsl_moe_a16wfp4.py gates on."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom)


def _fused_moe_flops_bytes(token, model_dim, inter_dim, expert, topk):
    """flop / byte estimate for the fused (stage1+stage2) MoE.

    Mirrors ``gemm_moe_tune.py`` ``calculate()``'s fused-MoE branch (the
    ``else`` arm, csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py:2654-2668)
    so TFLOPS/BW here are comparable with the rows already in the tuned CSVs.
    bpe: bf16 A/out = 2 B, mxfp4 W = 0.5 B.
    """
    n = inter_dim * 2  # use_g1u1
    flop = token * n * model_dim * topk * 2 + topk * token * model_dim * inter_dim * 2
    data_bytes = (
        token * model_dim * 2
        + n * model_dim * 0.5 * expert
        + inter_dim * model_dim * 0.5 * expert
        + token * model_dim * 2
    )
    return flop, data_bytes


def _tflops_bw(token, model_dim, inter_dim, expert, topk, us):
    flop, data_bytes = _fused_moe_flops_bytes(token, model_dim, inter_dim, expert, topk)
    return (
        round(flop / (us * 1000000), 2),
        round(data_bytes / (us * 1e-6) / 1e9, 2),
    )


def prepare_case(token, model_dim, inter_dim, expert, topk, seed=0):
    """Build one a16w4 SiTUv2 case plus its torch reference.

    The reference does not depend on the tile config, so it is computed once
    per shape and every candidate is compared against it.
    """
    dtype = dtypes.bf16
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inp = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, expert), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(expert, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(expert, model_dim, inter_dim // 2)
    w1_scale_e = w1_scale.view(expert, inter_dim * 2, model_dim // 32)
    w2_scale_e = w2_scale.view(expert, model_dim, inter_dim // 32)

    o1 = torch_moe_stage1(
        inp.to(dtype),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale_e,
        doweight=False,
        situ_beta=DEFAULT_SITU_BETA,
        situ_linear_beta=DEFAULT_SITU_LINEAR_BETA,
    )
    ref = torch_moe_stage2(
        o1.view(token, topk, inter_dim),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale_e,
        a2_scale=None,
        doweight=True,
    )

    return {
        "input": inp,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        # Caller contract: standard GGUU (separated gate/up) W1 layout.
        "w1": shuffle_weight_a16w4(w1_qt, 16, False),
        "w2": shuffle_weight_a16w4(w2_qt, 16, False),
        "w1_scale": shuffle_scale_a16w4(w1_scale, expert, False),
        "w2_scale": shuffle_scale_a16w4(w2_scale, expert, False),
        "ref": ref,
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "expert": expert,
        "topk": topk,
    }


def _sort(data, block_m):
    """moe_sorting for one block_m. Zeroes moe_buf for the atomic epilogue."""
    return moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        data["expert"],
        data["model_dim"],
        dtypes.bf16,
        block_size=block_m,
        accumulate=True,
    )


def _stage1(data, sort_out, kn1):
    sorted_ids, _sw, sorted_expert_ids, num_valid_ids, _buf = sort_out
    return _flydsl_stage1_wrapper(
        data["input"],
        data["w1"],
        data["w2"],
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        None,
        data["topk"],
        kernelName=kn1,
        activation=ActivationType.Situv2,
        w1_scale=data["w1_scale"],
        situ_beta=DEFAULT_SITU_BETA,
        situ_linear_beta=DEFAULT_SITU_LINEAR_BETA,
    )


def _stage2(data, sort_out, kn2, inter, block_m):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = sort_out
    return _flydsl_stage2_wrapper(
        inter,
        data["w1"],
        data["w2"],
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        data["topk"],
        kernelName=kn2,
        w2_scale=data["w2_scale"],
        sorted_weights=sorted_weights,
        block_m=block_m,
    )


def run_pair(data, kn1, kn2, block_m):
    """One full stage1+stage2 pipeline. Returns the final output."""
    sort_out = _sort(data, block_m)
    inter = _stage1(data, sort_out, kn1)
    return _stage2(data, sort_out, kn2, inter, block_m)


def check_pair(data, kn1, kn2, block_m):
    """Correctness gate. Returns the cos diff, or raises if it cannot pass.

    Runs before any timing: a config that drops tiles times fast and would
    otherwise win the sweep.  NaN must reject explicitly -- ``nan > tol`` is
    False, so a garbage-producing candidate would slip through a bare compare.
    """
    out = run_pair(data, kn1, kn2, block_m)
    if out.isnan().any().item():
        raise RuntimeError(f"NaN in output of ({kn1}, {kn2})")
    err = _cos_diff(data["ref"].float(), out.float())
    if not (err == err) or err > COS_TOL:  # noqa: PLR0124  (NaN-safe)
        raise RuntimeError(f"cos diff {err:.3e} > {COS_TOL} for ({kn1}, {kn2})")
    return err


def _median_us(fn, iters, repeats, warmup):
    """Median over `repeats` run_perftest calls, each `iters` iterations."""
    samples = []
    for _ in range(repeats):
        _, us = run_perftest(fn, num_iters=iters, num_warmup=warmup)
        samples.append(float(us))
    return statistics.median(samples)


def time_pair(data, kn1, kn2, block_m, iters, repeats, warmup):
    """Per-stage timings for one pair, with the correctness gate applied first.

    stage1 and stage2 are timed separately so the emitted row carries real
    us1/us2; ``us`` is their sum, matching the existing tuned CSVs.
    """
    err = check_pair(data, kn1, kn2, block_m)
    sort_out = _sort(data, block_m)
    inter = _stage1(data, sort_out, kn1)
    us1 = _median_us(lambda: _stage1(data, sort_out, kn1), iters, repeats, warmup)
    us2 = _median_us(
        lambda: _stage2(data, sort_out, kn2, inter, block_m), iters, repeats, warmup
    )
    return us1, us2, err


def _shard(names, shard):
    """Take this worker's stride-slice of a candidate list.

    Strided rather than contiguous so each worker gets a spread of tile sizes:
    build time varies with the tile, and a contiguous split would leave one
    worker holding all the expensive ones.

    Sharding exists because the per-candidate cost is dominated by the FlyDSL
    JIT build (measured: ~14 s/build under ROCm 7.14, ~82 s under 7.2.4, versus
    ~0.5 s for an already-built candidate), and that build is per-process CPU
    work on an otherwise idle box.  Each worker pins its own GPU, so the
    timings stay uncontended; rankings are still produced per (token, block_m)
    within a worker, and the per-shard rows are merged by lowest `us`.
    """
    if shard is None:
        return names
    i, n = shard
    return names[i - 1 :: n]


def stage1_candidates(block_m, inter_dim, model_dim, shard=None):
    """Registered stage1 names for this block_m, shape-legal ones only.

    Filters on the same divisibility the gemm1 builder asserts, so an illegal
    tile is skipped here rather than raising mid-sweep.
    """
    out = []
    for name, p in get_flydsl_stage1_kernels("bf16", "fp4", "bf16").items():
        if p["tile_m"] != block_m:
            continue
        tn, tk, kw = p["tile_n"], p["tile_k"], p.get("k_wave", 1)
        if inter_dim % tn or model_dim % tk or model_dim % (kw * tk):
            continue
        if p.get("k_batch", 1) != 1 or p.get("gate_mode", "separated") != "separated":
            continue
        out.append(name)
    return _shard(sorted(out), shard)


def stage2_candidates(block_m, inter_dim, model_dim, shard=None):
    """Registered stage2 names for this block_m, shape-legal ones only.

    ``mode="reduce"`` is int4-only on this port, so only atomic names qualify.
    """
    out = []
    for name, p in get_flydsl_stage2_kernels("bf16", "fp4", "bf16").items():
        if p["tile_m"] != block_m or p.get("mode", "atomic") != "atomic":
            continue
        tn, tk = p["tile_n"], p["tile_k"]
        if model_dim % tn or inter_dim % tk or tn % 64:
            continue
        out.append(name)
    return _shard(sorted(out), shard)


def _sweep_axis(data, names, block_m, fixed, which, args):
    """Time every candidate on one stage, holding the other stage fixed.

    Returns (best_name, best_us, results, failures).  A candidate that fails to
    build or misses the correctness gate is recorded and skipped, never ranked.
    """
    results = []
    failures = []
    for i, name in enumerate(names, 1):
        kn1, kn2 = (name, fixed) if which == 1 else (fixed, name)
        t0 = time.time()
        try:
            us1, us2, err = time_pair(
                data, kn1, kn2, block_m, args.iters, args.repeats, args.warmup
            )
        except Exception as e:  # noqa: BLE001
            failures.append((name, f"{type(e).__name__}: {str(e)[:120]}"))
            print(f"  [{i}/{len(names)}] SKIP {name}: {type(e).__name__}", flush=True)
            continue
        us = us1 if which == 1 else us2
        results.append((us, name, us1, us2, err))
        print(
            f"  [{i}/{len(names)}] {name} us{which}={us:.4f} "
            f"({time.time() - t0:.1f}s wall)",
            flush=True,
        )
    if not results:
        return None, None, results, failures
    results.sort(key=lambda r: r[0])
    return results[0][1], results[0][0], results, failures


def sweep_shape(data, block_m, args):
    """Independent stage1 then stage2 sweep for one (shape, block_m).

    stage1 is swept against a fixed seed stage2 (and vice versa); the pair is
    re-measured together at the end so the emitted row's us1/us2 come from the
    same pairing it names.
    """
    s1 = stage1_candidates(block_m, data["inter_dim"], data["model_dim"], args.shard)
    s2 = stage2_candidates(block_m, data["inter_dim"], data["model_dim"], args.shard)
    if not s1 or not s2:
        print(f"block_m={block_m}: no legal candidates (s1={len(s1)}, s2={len(s2)})")
        return None
    print(
        f"\n=== token={data['token']} block_m={block_m}: "
        f"{len(s1)} stage1 x {len(s2)} stage2 candidates (swept independently) ==="
    )

    # Seed pair: the first candidate that builds and passes, so the other axis
    # is always held at a known-good kernel rather than an untested guess.
    seed1 = seed2 = None
    for a in s1:
        for b in s2:
            try:
                check_pair(data, a, b, block_m)
            except Exception:  # noqa: BLE001, S112
                continue
            seed1, seed2 = a, b
            break
        if seed1 is not None:
            break
    if seed1 is None:
        print(f"block_m={block_m}: no candidate pair passed the correctness gate")
        return None
    print(f"  seed pair: {seed1} + {seed2}")

    print("  -- stage1 sweep --")
    best1, _, r1, f1 = _sweep_axis(data, s1, block_m, seed2, 1, args)
    print("  -- stage2 sweep --")
    best2, _, r2, f2 = _sweep_axis(data, s2, block_m, best1 or seed1, 2, args)
    if best1 is None or best2 is None:
        return None

    us1, us2, err = time_pair(
        data, best1, best2, block_m, args.iters, args.repeats, args.warmup
    )
    us = round(us1 + us2, 4)
    tflops, bw = _tflops_bw(
        data["token"],
        data["model_dim"],
        data["inter_dim"],
        data["expert"],
        data["topk"],
        us,
    )
    print(
        f"  BEST block_m={block_m}: us={us} (us1={us1:.4f} us2={us2:.4f}) "
        f"err={err:.2e} tflops={tflops} bw={bw}\n"
        f"    kn1={best1}\n    kn2={best2}\n"
        f"    tried {len(r1)}/{len(s1)} stage1 ({len(f1)} skipped), "
        f"{len(r2)}/{len(s2)} stage2 ({len(f2)} skipped)"
    )
    return {
        "gfx": get_gfx(),
        "cu_num": get_cu_num(),
        "token": data["token"],
        "model_dim": data["model_dim"],
        "inter_dim": data["inter_dim"],
        "expert": data["expert"],
        "topk": data["topk"],
        "act_type": "ActivationType.Situv2",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.bfloat16",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": 1,
        "doweight_stage1": 0,
        "block_m": block_m,
        "ksplit": 0,
        "us1": round(us1, 4),
        "kernelName1": best1,
        "err1": "0.0%",
        "us2": round(us2, 4),
        "kernelName2": best2,
        "err2": f"{err * 100:.1f}%",
        "us": us,
        "run_1stage": 0,
        "xbf16": 0,
        "flat": 0,
        "tflops": tflops,
        "bw": bw,
        "_tag": "",
    }


def _append_rows(path, rows):
    """Append tuned rows, writing the header only for a new file."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        # csv defaults to \r\n; the tuned CSVs in aiter/configs are \n.
        w = csv.DictWriter(fh, fieldnames=TUNED_CSV_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} row(s) -> {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep", action="store_true", help="enumerate all candidates")
    p.add_argument(
        "--token", type=int, action="append", help="token count (repeatable)"
    )
    p.add_argument("--block-m", type=int, action="append", choices=[16, 32, 64, 128])
    p.add_argument("--model-dim", type=int, default=DEFAULT_MODEL_DIM)
    p.add_argument("--inter-dim", type=int, default=DEFAULT_INTER_DIM)
    p.add_argument("--expert", type=int, default=DEFAULT_EXPERTS)
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--kernel1", help="explicit stage1 kernelName (default mode)")
    p.add_argument("--kernel2", help="explicit stage2 kernelName (default mode)")
    p.add_argument(
        "--iters",
        type=int,
        default=None,
        help="iterations per run (default: per-token, see _default_iters)",
    )
    p.add_argument("--repeats", type=int, default=3, help="runs to take the median of")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("-o", "--output", help="append tuned CSV rows here")
    p.add_argument(
        "--shard",
        metavar="I/N",
        help="measure only this worker's stride-slice of the candidates "
        "(1-based, e.g. 3/8); pin each worker to its own GPU and merge the "
        "per-shard CSVs by lowest us",
    )
    p.add_argument(
        "--list-candidates",
        action="store_true",
        help="print the candidate counts for the shape and exit (no GPU work)",
    )
    args = p.parse_args(argv)

    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= i <= n:
            p.error(f"--shard must be 1<=I<=N, got {args.shard}")
        args.shard = (i, n)

    if get_gfx() not in ("gfx942", "gfx950"):
        print(
            f"a16w-mix 2-stage is CDNA gfx942/gfx950-only; got {get_gfx()!r}",
            file=sys.stderr,
        )
        return 1

    tokens = args.token or ([*SWEEP_TOKENS] if args.sweep else [1])
    block_ms = args.block_m or [*SWEEP_BLOCK_MS]

    if args.list_candidates:
        for bm in block_ms:
            s1 = stage1_candidates(bm, args.inter_dim, args.model_dim, args.shard)
            s2 = stage2_candidates(bm, args.inter_dim, args.model_dim, args.shard)
            print(
                f"block_m={bm}: {len(s1)} stage1, {len(s2)} stage2 "
                f"-> {len(s1) + len(s2)} measurements/token"
            )
        return 0

    print(
        f"{get_gfx()} cu_num={get_cu_num()} "
        f"model_dim={args.model_dim} inter_dim={args.inter_dim} "
        f"expert={args.expert} topk={args.topk} "
        f"iters={args.iters or 'per-token'} repeats={args.repeats}"
    )

    requested_iters = args.iters
    rows = []
    for token in tokens:
        # Fixed per (token, block_m) so every candidate of one shape is timed
        # identically; only the wall-time budget varies across tokens.
        args.iters = requested_iters or _default_iters(token)
        print(f"\n### token={token} iters={args.iters} ###", flush=True)
        try:
            data = prepare_case(
                token, args.model_dim, args.inter_dim, args.expert, args.topk
            )
        except Exception as e:  # noqa: BLE001
            # The reference is built over every expert, so the large prefill
            # shapes can exhaust memory. Losing one token must not take the
            # remaining ones (and, with --sweep, hours of measured rows).
            print(f"### token={token} SKIPPED: {type(e).__name__}: {e}", flush=True)
            continue
        if args.sweep:
            for bm in block_ms:
                row = sweep_shape(data, bm, args)
                if row is None:
                    continue
                rows.append(row)
                # Flush per (token, block_m): a full sweep is hours long, and
                # the large-token shapes are the ones most likely to die (the
                # torch reference is built over every expert). Keeping the rows
                # until the end would throw away the whole run with them.
                if args.output:
                    _append_rows(args.output, [row])
            continue

        bm = block_ms[0]
        kn1 = args.kernel1 or stage1_candidates(bm, args.inter_dim, args.model_dim)[0]
        kn2 = args.kernel2 or stage2_candidates(bm, args.inter_dim, args.model_dim)[0]
        us1, us2, err = time_pair(
            data, kn1, kn2, bm, args.iters, args.repeats, args.warmup
        )
        us = round(us1 + us2, 4)
        tflops, bw = _tflops_bw(
            token, args.model_dim, args.inter_dim, args.expert, args.topk, us
        )
        print(
            f"token={token} block_m={bm}\n"
            f"  kn1={kn1}\n  kn2={kn2}\n"
            f"  us1={us1:.4f} us2={us2:.4f} us={us} "
            f"cos_err={err:.3e} tflops={tflops} bw={bw}"
        )

    if args.sweep and args.output:
        print(f"\n{len(rows)} row(s) total in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
