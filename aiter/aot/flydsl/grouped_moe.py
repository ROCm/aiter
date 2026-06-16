#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for gfx1250 grouped MoE GEMM kernels."""

from __future__ import annotations

import argparse
import csv
import sys
import time

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    job_identity,
    override_env,
)
from aiter.jit.core import AITER_CONFIGS

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GROUPED_FMOE_FILE]
_WARP_TILE_N = 64
_TILE_K = 256


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _preshuffled_scale_shape(
    rows: int, k_dim: int, warp_tile: int, tile_k: int = _TILE_K
) -> tuple[int, int]:
    """Mirror moe_grouped_gemm_mxscale_gfx1250._preshuffled_scale_shape.

    The grouped GEMM launchers validate an exact preshuffled E8M0 scale layout
    (see tests.kernels.test_gemm_mxscale_gfx1250.preshuffle_e8m0_scale), so the
    AOT dummy tensors must use the same shape, not the plain (rows, k//32) one.
    """
    k_scale = int(k_dim) // 32
    scale_k_per_tile = int(tile_k) // 32
    if k_scale % scale_k_per_tile != 0:
        raise ValueError(
            f"K scale columns must be divisible by tile_k/32, got {k_scale} and {scale_k_per_tile}"
        )
    wmma_rep = int(warp_tile) // 16
    if wmma_rep < 1:
        raise ValueError(f"warp_tile must be >= 16, got {warp_tile}")
    if int(rows) % wmma_rep != 0:
        raise ValueError(
            f"scale rows must be divisible by wmma_rep={wmma_rep}, got {rows}"
        )
    return int(rows) // wmma_rep, k_scale * wmma_rep


def _as_bool(value, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes")


def _as_int(value, default: int | None = None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _scheduler_variants(row, base_job):
    # Production dispatch (grouped_moe_gfx1250._maybe_grouped_gfx1250_a8w4_moe)
    # hardcodes grouped_persistent_m=False and expert_sched_mode=False; the only
    # runtime axis is dense vs DeepGEMM contiguous-M (auto-enabled for large token
    # counts). Mirror exactly that set so AOT never compiles GEMM variants the
    # runtime cannot launch.
    explicit_contiguous = _as_bool(row.get("grouped_contiguous_m"), False)
    contiguous_modes = [True] if explicit_contiguous else [False, True]
    variants = []
    for contiguous in contiguous_modes:
        variant = dict(base_job)
        variant["grouped_persistent_m"] = False
        variant["grouped_contiguous_m"] = contiguous
        variant["expert_sched_mode"] = False
        variants.append(variant)
    return variants


def parse_csv(csv_path: str):
    jobs = []
    seen = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            # Skip blank/incomplete CSV rows.
            if any(
                row.get(col) is None or str(row.get(col)).strip() == ""
                for col in ("model_dim", "inter_dim", "expert", "token")
            ):
                continue
            n_warp = int(row.get("n_warp") or 4)
            token_num = int(row["token"])
            tile_m = int(row.get("tile_m") or 64)
            m_warp = int(row.get("m_warp") or 1)
            warp_tile_m = tile_m // m_warp
            topk = int(row.get("topk") or 1)
            raw_max_m = _as_int(row.get("max_m"), token_num)
            max_m = max(
                warp_tile_m,
                ((raw_max_m + warp_tile_m - 1) // warp_tile_m) * warp_tile_m,
            )
            base_job = {
                "kernel_name": row.get("kernelName1", "grouped_gemm1"),
                "model_dim": int(row["model_dim"]),
                "inter_dim": int(row["inter_dim"]),
                "experts": int(row["expert"]),
                "max_m": max_m,
                "token_num": token_num,
                "topk": topk,
                "tile_m": tile_m,
                "tile_n": n_warp * _WARP_TILE_N,
                "tile_k": _TILE_K,
                "m_warp": m_warp,
                "n_warp": n_warp,
                "num_buffers": int(row.get("num_buffers") or 2),
                "split_k1": int(row.get("split_k1") or 1),
                "split_k2": int(row.get("split_k2") or 1),
                "out_dtype": "bf16" if row.get("dtype") == "torch.bfloat16" else "f16",
                "persistent_workers": _as_int(row.get("persistent_workers"), None),
                "stage1_weight_layout": row.get("stage1_weight_layout") or "gguu",
                "act": "swiglu" if "Swiglu" in row.get("act_type", "") else "silu",
                "data_format": (
                    "fp4" if "float4" in row.get("q_dtype_a", "") else "a8w4"
                ),
                "gfx": row.get("gfx", ""),
            }
            for job in _scheduler_variants(row, base_job):
                key = job_identity(job)
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(job)
    return jobs


GROUPED_MOE_AOT_ARCH_DEFAULT = "gfx1250"


def compile_one_config(**job):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import (
        compile_moe_grouped_gemm1_a8w4_masked,
        compile_moe_grouped_gemm1_mxfp4_masked,
        compile_moe_grouped_gemm2_a8w4_masked,
        compile_moe_grouped_gemm2_mxfp4_masked,
    )

    aot_arch = job.pop("gfx", "") or GROUPED_MOE_AOT_ARCH_DEFAULT

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
    warp_tile_m = job["tile_m"] // job["m_warp"]
    warp_tile_n = job["tile_n"] // job["n_warp"]
    contiguous = bool(job.get("grouped_contiguous_m", False))
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
        grouped_contiguous_m=contiguous,
        persistent_workers=job["persistent_workers"],
        expert_sched_mode=job["expert_sched_mode"],
    )
    if contiguous:
        act_lead = 1
        ub = job["token_num"] * job["topk"] + job["experts"] * (job["tile_m"] - 1)
        rows = max(job["tile_m"], _align_up(ub, job["tile_m"]))
    else:
        act_lead = job["experts"]
        rows = job["max_m"]
    with compile_only_env(), override_env(
        "FLYDSL_GPU_ARCH", aot_arch
    ), FakeTensorMode():
        masked_m = torch.full(
            (job["experts"],), job["max_m"], dtype=torch.int32, device=dev
        )
        # Contiguous-M layout tensor (mirrors runtime psum_t); None otherwise.
        contiguous_layout = (
            torch.empty((job["experts"],), dtype=torch.int32, device=dev)
            if contiguous
            else None
        )
        y1 = torch.empty((act_lead, rows, job["inter_dim"]), dtype=dtype)
        x1 = torch.empty((act_lead, rows, job["model_dim"] // pack), dtype=torch.uint8)
        w1 = torch.empty(
            (job["experts"], 2 * job["inter_dim"], job["model_dim"] // 2),
            dtype=torch.uint8,
        )
        sx1 = torch.empty(
            (act_lead, *_preshuffled_scale_shape(rows, job["model_dim"], warp_tile_m)),
            dtype=torch.uint8,
        )
        sw1 = torch.empty(
            (
                job["experts"],
                *_preshuffled_scale_shape(
                    2 * job["inter_dim"], job["model_dim"], warp_tile_n
                ),
            ),
            dtype=torch.uint8,
        )
        y2 = torch.empty((act_lead, rows, job["model_dim"]), dtype=dtype)
        x2 = torch.empty((act_lead, rows, job["inter_dim"] // pack), dtype=torch.uint8)
        w2 = torch.empty(
            (job["experts"], job["model_dim"], job["inter_dim"] // 2),
            dtype=torch.uint8,
        )
        sx2 = torch.empty(
            (act_lead, *_preshuffled_scale_shape(rows, job["inter_dim"], warp_tile_m)),
            dtype=torch.uint8,
        )
        sw2 = torch.empty(
            (
                job["experts"],
                *_preshuffled_scale_shape(
                    job["model_dim"], job["inter_dim"], warp_tile_n
                ),
            ),
            dtype=torch.uint8,
        )
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
            _m_tile_map=contiguous_layout,
        )
        # Bias-epilogue variant: runtime calls stage1(..., bias=...) when the model
        # carries per-expert bias (e.g. gpt-oss), which triggers a distinct compiled
        # kernel (gemm1_bias_* / finalize_act_bias). Precompile it alongside the
        # bias-free kernel so neither path JITs at first inference.
        bias1 = torch.empty((job["experts"], 2 * job["inter_dim"]), dtype=dtype)
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
            _m_tile_map=contiguous_layout,
            bias=bias1,
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
            _m_tile_map=contiguous_layout,
        )
        # Bias-epilogue variant for stage2 (gemm2_bias_*); see stage1 note above.
        bias2 = torch.empty((job["experts"], job["model_dim"]), dtype=dtype)
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
            _m_tile_map=contiguous_layout,
            bias=bias2,
        )
    return {**job, "compile_time": time.time() - t0, "compile_arch": aot_arch}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    args = parser.parse_args(argv)
    jobs = collect_aot_jobs(args.csv, parse_csv)
    for job in jobs:
        print(compile_one_config(**job))


if __name__ == "__main__":
    main(sys.argv[1:])
