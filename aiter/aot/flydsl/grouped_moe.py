#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for gfx1250 grouped MoE GEMM kernels."""

from __future__ import annotations

import argparse
import csv
import sys
import time

from aiter.aot.flydsl.common import collect_aot_jobs, compile_only_env, job_identity
from aiter.jit.core import AITER_CONFIGS

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GROUPED_FMOE_FILE]
_WARP_TILE_N = 64
_TILE_K = 256


def _as_bool(value, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes")


def _as_int(value, default: int | None = None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def parse_csv(csv_path: str):
    jobs = []
    seen = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            n_warp = int(row.get("n_warp") or 4)
            job = {
                "kernel_name": row.get("kernelName1", "grouped_gemm1"),
                "model_dim": int(row["model_dim"]),
                "inter_dim": int(row["inter_dim"]),
                "experts": int(row["expert"]),
                "max_m": int(row["token"]),
                "tile_m": int(row.get("tile_m") or 64),
                "tile_n": n_warp * _WARP_TILE_N,
                "tile_k": _TILE_K,
                "m_warp": int(row.get("m_warp") or 1),
                "n_warp": n_warp,
                "num_buffers": int(row.get("num_buffers") or 2),
                "split_k1": int(row.get("split_k1") or 1),
                "split_k2": int(row.get("split_k2") or 1),
                "out_dtype": "bf16" if row.get("dtype") == "torch.bfloat16" else "f16",
                "grouped_persistent_m": _as_bool(row.get("grouped_persistent_m"), True),
                "grouped_contiguous_m": _as_bool(row.get("grouped_contiguous_m"), False),
                "persistent_workers": _as_int(row.get("persistent_workers"), None),
                "stage1_weight_layout": row.get("stage1_weight_layout") or "gguu",
                "act": "swiglu" if "Swiglu" in row.get("act_type", "") else "silu",
                "data_format": (
                    "fp4" if "float4" in row.get("q_dtype_a", "") else "a8w4"
                ),
            }
            key = job_identity(job)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)
    for job in jobs:
        if job.get("grouped_contiguous_m"):
            job["grouped_persistent_m"] = False
    return jobs


def compile_one_config(**job):
    import torch

    from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import (
        compile_moe_grouped_gemm1_a8w4_masked,
        compile_moe_grouped_gemm1_mxfp4_masked,
        compile_moe_grouped_gemm2_a8w4_masked,
        compile_moe_grouped_gemm2_mxfp4_masked,
    )

    t0 = time.time()
    dev = torch.device("cpu")
    dtype = torch.bfloat16 if job["out_dtype"] == "bf16" else torch.float16
    pack = 2 if job["data_format"] == "fp4" else 1
    compiler1 = (
        compile_moe_grouped_gemm1_mxfp4_masked
        if job["data_format"] == "fp4"
        else compile_moe_grouped_gemm1_a8w4_masked
    )
    compiler2 = (
        compile_moe_grouped_gemm2_mxfp4_masked
        if job["data_format"] == "fp4"
        else compile_moe_grouped_gemm2_a8w4_masked
    )
    common = dict(
        model_dim=job["model_dim"],
        inter_dim=job["inter_dim"],
        experts=job["experts"],
        max_m=job["max_m"],
        tile_m=job["tile_m"],
        tile_n=job["tile_n"],
        tile_k=job["tile_k"],
        m_warp=job["m_warp"],
        n_warp=job["n_warp"],
        out_dtype=job["out_dtype"],
        num_buffers=job["num_buffers"],
        grouped_persistent_m=job["grouped_persistent_m"],
        grouped_contiguous_m=job.get("grouped_contiguous_m", False),
        persistent_workers=job["persistent_workers"],
    )
    masked_m = torch.full(
        (job["experts"],), job["max_m"], dtype=torch.int32, device=dev
    )
    y1 = torch.empty((job["experts"], job["max_m"], job["inter_dim"]), dtype=dtype)
    x1 = torch.empty(
        (job["experts"], job["max_m"], job["model_dim"] // pack), dtype=torch.uint8
    )
    w1 = torch.empty(
        (job["experts"], 2 * job["inter_dim"], job["model_dim"] // 2), dtype=torch.uint8
    )
    sx1 = torch.empty(
        (job["experts"], job["max_m"], job["model_dim"] // 32), dtype=torch.uint8
    )
    sw1 = torch.empty(
        (job["experts"], 2 * job["inter_dim"], job["model_dim"] // 32),
        dtype=torch.uint8,
    )
    y2 = torch.empty((job["experts"], job["max_m"], job["model_dim"]), dtype=dtype)
    x2 = torch.empty(
        (job["experts"], job["max_m"], job["inter_dim"] // pack), dtype=torch.uint8
    )
    w2 = torch.empty(
        (job["experts"], job["model_dim"], job["inter_dim"] // 2), dtype=torch.uint8
    )
    sx2 = torch.empty(
        (job["experts"], job["max_m"], job["inter_dim"] // 32), dtype=torch.uint8
    )
    sw2 = torch.empty(
        (job["experts"], job["model_dim"], job["inter_dim"] // 32), dtype=torch.uint8
    )
    with compile_only_env():
        exe1 = compiler1(
            act=job["act"],
            stage1_weight_layout=job["stage1_weight_layout"],
            split_k=job["split_k1"],
            **common,
        )
        exe1(
            y1,
            x1,
            w1,
            sx1,
            sw1,
            masked_m,
            job["max_m"],
            job["inter_dim"],
            job["model_dim"],
            job["experts"],
            stream=0,
        )
        exe2 = compiler2(split_k=job["split_k2"], **common)
        exe2(
            y2,
            x2,
            w2,
            sx2,
            sw2,
            masked_m,
            job["max_m"],
            job["model_dim"],
            job["inter_dim"],
            job["experts"],
            stream=0,
        )
    return {**job, "compile_time": time.time() - t0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    args = parser.parse_args(argv)
    jobs = collect_aot_jobs(args.csv, parse_csv)
    for job in jobs:
        print(compile_one_config(**job))


if __name__ == "__main__":
    main(sys.argv[1:])
