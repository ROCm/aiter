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
    variants = []
    for contiguous in [False, True]:
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


def _compile_grouped_moe_aux_kernels(job, *, dtype, pack, warp_tile_m, max_m):
    """Precompile the non-GEMM FlyDSL kernels the run-only grouped MoE path
    launches around gemm1/gemm2.

    On the production fast path (``_maybe_grouped_gfx1250_a8w4_moe``) each
    grouped GEMM is bracketed by a set of small FlyDSL kernels:

      * ``moe_route_maps``                 -- atomic-scatter route -> grouped row
      * ``moe_scatter_copy_token``         -- payload route-gather
      * ``moe_scatter_copy_preshuffle_scale`` (gather=True)  -- stage1 scale
      * ``moe_scatter_copy_preshuffle_scale`` (gather=False) -- stage2 scale
      * ``moe_gather_reduce``              -- token-order epilogue (bf16/f16)
      * ``moe_contiguous_psum``            -- DeepGEMM contiguous-M prefix sum

    Under ``FLYDSL_RUNTIME_RUN_ONLY`` any of these missing from the AOT cache
    raises at first inference, so precompile them alongside the GEMMs. The
    launch shapes only need correct dtype/rank -- each module's cache key is the
    build-time geometry (row bytes / wmma_rep / model_dim / topk / out_dtype),
    not the dynamic token/expert scalars passed at launch.
    """
    import torch

    from aiter.ops.flydsl.kernels.moe_contiguous_psum import (
        build_moe_contiguous_psum_module,
    )
    from aiter.ops.flydsl.kernels.moe_gather_reduce import (
        build_moe_gather_reduce_module,
    )
    from aiter.ops.flydsl.kernels.moe_route_maps import (
        build_moe_route_maps_module,
    )
    from aiter.ops.flydsl.kernels.moe_scatter_copy_preshuffle_scale import (
        build_moe_scatter_copy_preshuffle_scale_module,
    )
    from aiter.ops.flydsl.kernels.moe_scatter_copy_token import (
        build_moe_scatter_copy_token_module,
    )

    dev = torch.device("cpu")
    i32 = torch.int32
    u8 = torch.uint8
    E = job["experts"]
    topk = job["topk"]
    model_dim = job["model_dim"]
    inter_dim = job["inter_dim"]
    token_num = max(1, job["token_num"])
    out_dtype = job["out_dtype"]

    numel = token_num * topk
    num_dst = E * max_m
    wmma_rep = warp_tile_m // 16
    scale_k_per_tile = _TILE_K // 32
    tiles_per_expert = max_m // (wmma_rep * 16)
    grid_blocks = (numel + 255) // 256

    # Route -> grouped-row maps (no build-time params).
    route_maps = build_moe_route_maps_module()
    route_maps(
        torch.empty((numel,), dtype=i32, device=dev),
        torch.empty((E,), dtype=i32, device=dev),
        torch.empty((numel,), dtype=i32, device=dev),
        torch.empty((num_dst,), dtype=i32, device=dev),
        numel,
        topk,
        max_m,
        grid_blocks,
        stream=0,
    )

    # Payload route-gather (row width = model_dim // pack).
    row_bytes = model_dim // pack
    scatter_copy = build_moe_scatter_copy_token_module(row_bytes)
    scatter_copy(
        torch.empty((numel, row_bytes), dtype=u8, device=dev),
        torch.empty((num_dst, row_bytes), dtype=u8, device=dev),
        torch.empty((num_dst,), dtype=i32, device=dev),
        num_dst,
        stream=0,
    )

    # Stage1 scale: route-gather + WMMA preshuffle (gather=True).
    ws1 = model_dim // 32
    preshuffle1 = build_moe_scatter_copy_preshuffle_scale_module(
        ws1, wmma_rep, scale_k_per_tile, gather=True
    )
    preshuffle1(
        torch.empty((numel, ws1), dtype=u8, device=dev),
        torch.empty((E * (max_m // wmma_rep), ws1 * wmma_rep), dtype=u8, device=dev),
        torch.empty((num_dst,), dtype=i32, device=dev),
        max_m,
        E,
        tiles_per_expert,
        stream=0,
    )

    # Stage2 scale: already grouped, pure preshuffle (gather=False).
    ws2 = inter_dim // 32
    preshuffle2 = build_moe_scatter_copy_preshuffle_scale_module(
        ws2, wmma_rep, scale_k_per_tile, gather=False
    )
    preshuffle2(
        torch.empty((num_dst, ws2), dtype=u8, device=dev),
        torch.empty((E * (max_m // wmma_rep), ws2 * wmma_rep), dtype=u8, device=dev),
        max_m,
        E,
        tiles_per_expert,
        stream=0,
    )

    # Token-order gather-reduce epilogue (fast path is bf16/f16 only).
    gather_reduce = build_moe_gather_reduce_module(model_dim, topk, out_dtype)
    gather_reduce(
        torch.empty((num_dst, model_dim), dtype=dtype, device=dev),
        torch.empty((token_num, topk), dtype=i32, device=dev),
        torch.empty((token_num, topk), dtype=dtype, device=dev),
        torch.empty((token_num, model_dim), dtype=dtype, device=dev),
        token_num,
        stream=0,
    )

    # DeepGEMM contiguous-M tile-aligned prefix sum (no build-time params).
    contiguous_psum = build_moe_contiguous_psum_module()
    contiguous_psum(
        torch.empty((E,), dtype=i32, device=dev),
        torch.empty((E,), dtype=i32, device=dev),
        torch.empty((E,), dtype=i32, device=dev),
        torch.empty((1,), dtype=i32, device=dev),
        E,
        job["tile_m"],
        stream=0,
    )


def compile_one_config(**job):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import (
        compile_moe_grouped_gemm1_a8w4_masked,
        compile_moe_grouped_gemm1_mxfp4_masked,
        compile_moe_grouped_gemm2_a8w4_masked,
        compile_moe_grouped_gemm2_mxfp4_masked,
        preshuffled_scale_shape,
        preshuffled_b_scale_shape,
    )

    aot_arch = job.pop("gfx", "") or GROUPED_MOE_AOT_ARCH_DEFAULT
    shape_str = (
        f"{job.get('kernel_name', 'grouped_gemm')}  "
        f"model_dim={job['model_dim']} inter_dim={job['inter_dim']} "
        f"E={job['experts']} topk={job['topk']} "
        f"contiguous={bool(job.get('grouped_contiguous_m', False))}"
    )

    t0 = time.time()
    try:
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
        with (
            compile_only_env(),
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            FakeTensorMode(),
        ):
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
            x1 = torch.empty(
                (act_lead, rows, job["model_dim"] // pack), dtype=torch.uint8
            )
            w1 = torch.empty(
                (job["experts"], 2 * job["inter_dim"], job["model_dim"] // 2),
                dtype=torch.uint8,
            )
            sx1 = torch.empty(
                (
                    act_lead,
                    *preshuffled_scale_shape(
                        rows, job["model_dim"], warp_tile_m, _TILE_K
                    ),
                ),
                dtype=torch.uint8,
            )
            sw1 = torch.empty(
                (
                    job["experts"],
                    *preshuffled_b_scale_shape(2 * job["inter_dim"], job["model_dim"]),
                ),
                dtype=torch.uint8,
            )
            y2 = torch.empty((act_lead, rows, job["model_dim"]), dtype=dtype)
            x2 = torch.empty(
                (act_lead, rows, job["inter_dim"] // pack), dtype=torch.uint8
            )
            w2 = torch.empty(
                (job["experts"], job["model_dim"], job["inter_dim"] // 2),
                dtype=torch.uint8,
            )
            sx2 = torch.empty(
                (
                    act_lead,
                    *preshuffled_scale_shape(
                        rows, job["inter_dim"], warp_tile_m, _TILE_K
                    ),
                ),
                dtype=torch.uint8,
            )
            sw2 = torch.empty(
                (
                    job["experts"],
                    *preshuffled_b_scale_shape(job["model_dim"], job["inter_dim"]),
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
            # Non-GEMM auxiliary kernels the run-only fast path launches around
            # the GEMMs (route maps, scatter-copy, scale preshuffle,
            # gather-reduce, contiguous prefix-sum). They are
            # scheduler-variant-independent, so the second variant just hits the
            # on-disk AOT cache.
            _compile_grouped_moe_aux_kernels(
                job,
                dtype=dtype,
                pack=pack,
                warp_tile_m=warp_tile_m,
                max_m=job["max_m"],
            )
        elapsed = time.time() - t0
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  arch={aot_arch}")
        return {**job, "compile_time": elapsed, "compile_arch": aot_arch}
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")
        return {**job, "compile_time": None, "compile_arch": aot_arch}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    args = parser.parse_args(argv)
    jobs = collect_aot_jobs(args.csv, parse_csv)
    print(f"[aiter] FlyDSL GROUPED_MOE AOT: {len(jobs)} kernels")
    results = [compile_one_config(**job) for job in jobs]
    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = len(results) - ok
    print(f"[aiter] FlyDSL GROUPED_MOE AOT: compiled {ok} ok, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
