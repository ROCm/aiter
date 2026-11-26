# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline tile tuner for the gfx950 FlyDSL ``gemm_a16w16`` with an epilogue.

Reads an untuned CSV (``M, N, K, layout, epilogue, dtype, outdtype``), times
every legal config of ``gemm_epilogue_configs.candidate_configs`` through the
same ``gemm_a16w16`` call the KDA wrappers make (epilogue inputs and outputs
included), grades each against an fp32 torch reference, and writes the fastest
correct one per shape to the tuned CSV. Rows of ``-o`` with other keys are
kept and rows with the same key are replaced, so the file never holds two rows
of one shape; run the config-shape guard after editing a family:

    python3 csrc/flydsl_gemm_epilogue/gemm_epilogue_tune.py \\
        -i aiter/configs/model_configs/kimik3_flydsl_gemm_a16w16_epilogue_untuned.csv \\
        -o aiter/configs/model_configs/kimik3_flydsl_gemm_a16w16_epilogue_tuned.csv
    python3 -m unittest op_tests.tuning_tests.test_config_shape_collision

Shape ownership of the Kimi tables: ``kimik3_*`` holds tokens <= 8192 plus
16384 (varlen 8x2048), TP1; ``kimik3_1_*`` holds 32768 and 65536 with
H*K = 12288 / TP for TP 1, 4 and 8.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import logger
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl import gemm_epilogue_configs as cfgs
from aiter.ops.flydsl.kda_layer_bwd import KDA_HEAD_DIM, _validate
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP32,
    GEMM_EPI_AUX_MODES,
    GEMM_EPI_BWD_MODES,
    GEMM_EPI_KDA_MODES,
    gemm_a16w16,
)
from aiter.test_common import checkAllclose, run_perftest

# Kimi-K3 gate_lower_bound for the safe-gate modes
LOWER_BOUND = -5.0


def _inputs(row):
    m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
    layout, mode = row["layout"], int(row["epilogue"])
    dt = cfgs.torch_dtype(row["dtype"])
    a = torch.randn(k, m).to(dt).t() if layout[0] == "t" else torch.randn(m, k).to(dt)
    if layout[1] == "t":
        b = (torch.randn(n, k) * k**-0.5).to(dt).t()
    else:
        b = (torch.randn(k, n) * k**-0.5).to(dt)
    kw = {}
    if mode in GEMM_EPI_KDA_MODES:
        heads = n // KDA_HEAD_DIM
        kw.update(
            epi_group=KDA_HEAD_DIM,
            a_log=torch.linspace(-0.5, 1.2, heads),
            dt_bias=torch.randn(n),
            lower_bound=LOWER_BOUND if mode in (5, 6) else None,
        )
    if mode in GEMM_EPI_AUX_MODES:
        kw["aux"] = torch.randn(m, n)
    return a, b, kw


def _reference(a, b, mode, kw):
    z = a.float() @ b.float()
    if mode == 0:
        return z, None
    if mode in (3, 4):
        s = torch.sigmoid(z)
        return (s if mode == 3 else s * kw["aux"]), None
    n = z.shape[1]
    alpha = kw["a_log"].exp().repeat_interleave(n // kw["a_log"].numel())
    zb = z + kw["dt_bias"]
    if mode == 1:
        return -alpha * F.softplus(zb), None
    if mode == 5:
        return kw["lower_bound"] * torch.sigmoid(alpha * zb), None
    aux = kw["aux"]
    if mode == 2:
        dz = -aux * alpha * torch.sigmoid(zb)
        dterm = -aux * alpha * F.softplus(zb)
    else:
        s = torch.sigmoid(alpha * zb)
        dz = aux * kw["lower_bound"] * s * (1 - s) * alpha
        dterm = dz * zb
    return dz, torch.stack([dterm.sum(0), dz.sum(0)])


def tune_shape(row, iters, max_candidates):
    m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
    layout, mode = row["layout"], int(row["epilogue"])
    out_dtype = cfgs.torch_dtype(row["outdtype"])
    a, b, kw = _inputs(row)
    ref, ref_sums = _reference(a, b, mode, kw)
    extra = {
        "out_dtype_id": (
            GEMM_A16W16_DTYPE_FP32
            if out_dtype == torch.float32
            else GEMM_A16W16_DTYPE_BF16
        ),
        "epilogue": mode,
        "epi_group": KDA_HEAD_DIM if mode in GEMM_EPI_KDA_MODES else 1,
    }
    legal = _validate(m, n, k, layout, extra)
    best = None
    tried = 0
    for cfg in cfgs.candidate_configs(m, n, k, layout, mode):
        user = cfgs.to_user_kwargs(cfg)
        if not legal(user) or tried >= max_candidates:
            continue
        tried += 1
        call_kw = dict(kw)
        if mode in GEMM_EPI_BWD_MODES:
            rows = (m + cfg["block_m"] - 1) // cfg["block_m"]
            call_kw["colsum"] = torch.empty(rows, 2, n)

        def run(user=user, call_kw=call_kw):
            return gemm_a16w16(
                a,
                b,
                user_kwargs=user,
                layout=layout,
                out_dtype=out_dtype,
                epilogue=mode,
                **call_kw,
            )

        try:
            out, us = run_perftest(run, num_iters=iters, num_warmup=3)
        except (AssertionError, ValueError, RuntimeError) as e:
            logger.warning("%s %s: %s", cfgs.config_name(cfg), row.to_dict(), e)
            continue
        err = checkAllclose(
            ref, out.float(), rtol=2e-2, atol=2e-2, msg=cfgs.config_name(cfg)
        )
        if ref_sums is not None:
            sums = call_kw["colsum"].sum(0)
            err = max(err, checkAllclose(ref_sums, sums, rtol=2e-2, atol=1.0))
        if err > 0:
            continue
        if best is None or us < best[1]:
            best = (cfg, us, err)
    if best is None:
        return None
    cfg, us, err = best
    flops = 2 * m * n * k
    out_bytes = torch.empty(0, dtype=out_dtype).element_size()
    nbytes = (m * k + n * k) * a.element_size() + m * n * out_bytes
    if mode in GEMM_EPI_AUX_MODES:
        nbytes += m * n * 4
    return {
        "gfx": get_gfx_runtime(),
        "cu_num": get_cu_num(),
        **{c: row[c] for c in cfgs.KEY_COLUMNS},
        "libtype": "flydsl",
        **{c: int(cfg[c]) for c in cfgs.TILE_COLUMNS},
        "us": round(us, 4),
        "kernelName": cfgs.config_name(cfg),
        "err_ratio": err,
        "tflops": round(flops / us / 1e6, 2),
        "bw": round(nbytes / us / 1e6, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-i", "--untune_file", required=True)
    parser.add_argument("-o", "--tune_file", required=True)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--max_candidates", type=int, default=128, help="legal configs per shape"
    )
    args = parser.parse_args()
    if get_gfx_runtime() != "gfx950":
        raise SystemExit(
            f"gemm_a16w16 epilogues are gfx950-only, got {get_gfx_runtime()}"
        )
    torch.set_default_device("cuda")
    untuned = pd.read_csv(args.untune_file, dtype=str, keep_default_na=False)
    tuned = pd.DataFrame(columns=cfgs.TUNED_COLUMNS)
    if os.path.exists(args.tune_file):
        tuned = pd.read_csv(args.tune_file, dtype=str, keep_default_na=False)
    key = ["gfx", "cu_num", *cfgs.KEY_COLUMNS]
    for _, row in untuned.iterrows():
        result = tune_shape(row, args.iters, args.max_candidates)
        if result is None:
            logger.warning("no correct config for %s", row.to_dict())
            continue
        logger.info(
            "%s -> %s %.2f us", row.to_dict(), result["kernelName"], result["us"]
        )
        new = pd.DataFrame([result], columns=cfgs.TUNED_COLUMNS).astype(str)
        same = (tuned[key].astype(str) == new[key].iloc[0]).all(axis=1)
        tuned = pd.concat([tuned[~same], new], ignore_index=True)
        tuned.to_csv(args.tune_file, index=False)
        torch.cuda.empty_cache()
    cfgs.clear_cache()


if __name__ == "__main__":
    main()
