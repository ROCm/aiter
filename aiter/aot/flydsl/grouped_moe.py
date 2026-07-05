#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for gfx1250 grouped MoE GEMM kernels."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GROUPED_FMOE_FILE]
_WARP_TILE_N = 64
_TILE_K = 256


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _align_max_m(raw_max_m: int, warp_tile_m: int) -> int:
    """Mirror grouped_moe_gfx1250's max_m rounding (>= one warp tile)."""
    return max(int(warp_tile_m), _align_up(raw_max_m, warp_tile_m))


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


def _preshuffled_b_scale_shape(rows: int, k_dim: int) -> tuple[int, int]:
    """Mirror moe_grouped_gemm_mxscale_gfx1250._preshuffled_b_scale_shape.

    Weight (B) scale uses the n32k4 layout (different from the activation/A
    layout above): a 32-row super-block folds into the column dim, so 32 N-rows
    collapse to one row and each k_scale column expands x32. The grouped GEMM
    launchers validate scale_w against THIS shape, so the AOT dummy must match.
    """
    k_scale = int(k_dim) // 32
    if k_scale % 4 != 0:
        raise ValueError(
            f"B-scale k columns (K//32) must be divisible by 4 (K%128==0), got {k_scale}"
        )
    if int(rows) % 32 != 0:
        raise ValueError(f"B-scale rows must be divisible by 32, got {rows}")
    return int(rows) // 32, k_scale * 32


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
    warp_tile_m = base_job["tile_m"] // base_job["m_warp"]
    # Contiguous-M sizes max_m as max(cfg.max_m, token_num*topk) (default 0 when
    # the CSV omits max_m); dense uses cfg.max_m (default token_num). max_m is
    # baked into the GEMM kernel_tag, so AOT must derive it per-mode exactly as
    # grouped_moe_gfx1250 does or the precompiled kernel never gets hit.
    contiguous_max_m = _align_max_m(
        max(_as_int(row.get("max_m"), 0), base_job["token_num"] * base_job["topk"]),
        warp_tile_m,
    )
    variants = []
    for contiguous in contiguous_modes:
        variant = dict(base_job)
        variant["grouped_persistent_m"] = False
        variant["grouped_contiguous_m"] = contiguous
        variant["expert_sched_mode"] = False
        if contiguous:
            variant["max_m"] = contiguous_max_m
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
            max_m = _align_max_m(raw_max_m, warp_tile_m)
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


def _compile_grouped_moe_aux_kernels(job, *, dtype, quant_mode, wmma_rep, contiguous):
    """Precompile the non-GEMM FlyDSL kernels the run-only grouped MoE fast path
    launches around gemm1/gemm2."""
    import torch

    from aiter.ops.flydsl.kernels.moe_contiguous_psum import (
        build_moe_contiguous_psum_remap_module,
    )
    from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
        build_moe_fused_quant_preshuffle_module,
        build_moe_fused_quant_preshuffle_route_ksplit_module,
        build_moe_fused_route_quant_scatter_module,
        build_moe_fused_route_quant_scatter_st_ksplit_module,
    )
    from aiter.ops.flydsl.kernels.moe_gather_reduce import (
        build_moe_gather_reduce_module,
    )
    from aiter.ops.flydsl.kernels.moe_route_maps import (
        build_moe_topids_to_rows_module,
    )

    dev = torch.device("cpu")
    i32 = torch.int32
    u8 = torch.uint8
    bf16 = torch.bfloat16
    E = job["experts"]
    topk = job["topk"]
    model_dim = job["model_dim"]
    inter_dim = job["inter_dim"]
    max_m = job["max_m"]
    tile_m = job["tile_m"]
    token_num = max(1, job["token_num"])
    out_dtype = job["out_dtype"]
    numel = token_num * topk
    grid = max(1, (numel + 255) // 256)

    def _route_ksplit(feat_dim, source_topk, out_e, out_m):
        # build_moe_fused_quant_preshuffle_route_ksplit_module; runtime never
        # sets remap_rows on the grouped MoE fast path (row_starts stays None).
        launch = build_moe_fused_quant_preshuffle_route_ksplit_module(
            feat_dim=feat_dim,
            wmma_rep=wmma_rep,
            quant_mode=quant_mode,
            source_topk=source_topk,
            remap_rows=False,
        )
        launch(
            ptr_arg(torch.empty(0, dtype=bf16, device=dev)),
            ptr_arg(torch.empty(0, dtype=u8, device=dev)),
            ptr_arg(torch.empty(0, dtype=u8, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            1,
            numel,
            grid,
            stream=0,
        )

    def _plain_preshuffle(feat_dim, out_e, out_m, skip_padding):
        launch = build_moe_fused_quant_preshuffle_module(
            feat_dim=feat_dim,
            wmma_rep=wmma_rep,
            quant_mode=quant_mode,
            skip_padding=skip_padding,
        )
        n_rows = out_e * out_m
        launch(
            ptr_arg(torch.empty(0, dtype=bf16, device=dev)),
            ptr_arg(torch.empty(0, dtype=u8, device=dev)),
            ptr_arg(torch.empty(0, dtype=u8, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            n_rows,
            out_m,
            grid,
            stream=0,
        )

    def _topids_to_rows():
        launch = build_moe_topids_to_rows_module()
        launch(
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            numel,
            max_m,
            grid,
            stream=0,
        )

    # --- Stage1 activation prep (a1): fused route + MX-quant + scatter ---
    if contiguous:
        # DeepGEMM contiguous-M: topids_to_rows -> contiguous psum+remap ->
        # route-indexed quant+preshuffle into a single contiguous (E=1) buffer.
        ub = int(token_num) * int(topk) + int(E) * (int(tile_m) - 1)
        contiguous_m = max(int(tile_m), _align_up(ub, int(tile_m)))

        _topids_to_rows()

        psum_remap = build_moe_contiguous_psum_remap_module()
        psum_remap(
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            numel,
            E,
            max_m,
            tile_m,
            stream=0,
        )

        _route_ksplit(
            feat_dim=model_dim,
            source_topk=topk,
            out_e=1,
            out_m=contiguous_m,
        )
        a2_out_e, a2_out_m = 1, contiguous_m
    else:
        # Dense masked layout (one (E, max_m, *) bucket per expert).
        use_routeks_stage1 = token_num > 1 and topk > 1 and quant_mode == "fp4"
        use_st_ksplit = token_num == 1 and topk > 0 and (topk & (topk - 1)) == 0
        if use_routeks_stage1:
            _topids_to_rows()
            _route_ksplit(
                feat_dim=model_dim,
                source_topk=topk,
                out_e=E,
                out_m=max_m,
            )
        else:
            build_route = (
                build_moe_fused_route_quant_scatter_st_ksplit_module
                if use_st_ksplit
                else build_moe_fused_route_quant_scatter_module
            )
            launch = build_route(
                model_dim=model_dim,
                topk=topk,
                wmma_rep=wmma_rep,
                quant_mode=quant_mode,
                use_expert_row_base=False,
                max_m=max_m,
            )
            launch(
                ptr_arg(torch.empty(0, dtype=i32, device=dev)),
                ptr_arg(torch.empty(0, dtype=i32, device=dev)),
                ptr_arg(torch.empty(0, dtype=i32, device=dev)),
                ptr_arg(torch.empty(0, dtype=bf16, device=dev)),
                ptr_arg(torch.empty(0, dtype=u8, device=dev)),
                ptr_arg(torch.empty(0, dtype=u8, device=dev)),
                ptr_arg(torch.empty(0, dtype=i32, device=dev)),
                numel,
                grid,
                stream=0,
            )
        a2_out_e, a2_out_m = E, max_m

    # --- Stage2 activation prep (a2): fused grouped quant + preshuffle ---
    # Runtime passes topids_to_rows whenever route rows fit the output capacity
    # (almost always), taking the route-ksplit path with source_topk=0; the
    # plain skip-padding path is the rare capacity-overflow fallback.
    capacity_rows = a2_out_e * a2_out_m
    if numel < capacity_rows:
        _route_ksplit(
            feat_dim=inter_dim,
            source_topk=0,
            out_e=a2_out_e,
            out_m=a2_out_m,
        )
    else:
        # masked_m is None on the contiguous path -> skip_padding=False;
        # passed on the dense path -> skip_padding=True.
        _plain_preshuffle(
            feat_dim=inter_dim,
            out_e=a2_out_e,
            out_m=a2_out_m,
            skip_padding=not contiguous,
        )

    # --- Epilogue: token-order gather-reduce (bf16/f16 fast path only) ---
    # split_k mirrors stage2's split_k2: when split_k2 > 1 the GEMM emits an
    # unreduced (split_k, E, max_m, model_dim) tensor and gather-reduce folds the
    # split slices itself, so its kernel identity depends on split_k. Hardcoding
    # 1 here makes the token=1 / split_k2>1 CSV rows miss the AOT cache at
    # inference. vec width is token-count dependent at runtime; cover both so
    # run-only coverage holds across inference batch sizes, not just the CSV token.
    split_k = job["split_k2"]
    for vec in (2, 4):
        gather_reduce = build_moe_gather_reduce_module(
            model_dim, topk, out_dtype, split_k, vec
        )
        gather_reduce(
            ptr_arg(torch.empty(0, dtype=dtype, device=dev)),
            ptr_arg(torch.empty(0, dtype=i32, device=dev)),
            ptr_arg(torch.empty(0, dtype=dtype, device=dev)),
            ptr_arg(torch.empty(0, dtype=dtype, device=dev)),
            token_num,
            a2_out_e * a2_out_m * (model_dim // 2),
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
    )

    aot_arch = job.pop("gfx", "") or GROUPED_MOE_AOT_ARCH_DEFAULT
    shape_str = (
        # Use .get() so a missing key can't raise here, outside the try below:
        # an escaping exception would crash the worker (exitcode != 0), which the
        # AOT pool misreads as a transient failure and retries -> potential deadlock.
        f"{job.get('kernel_name', 'grouped_gemm')}  "
        f"model_dim={job.get('model_dim')} inter_dim={job.get('inter_dim')} "
        f"E={job.get('experts')} topk={job.get('topk')} "
        f"contiguous={bool(job.get('grouped_contiguous_m', False))}"
    )

    t0 = time.time()
    try:
        dev = torch.device("cpu")
        dtype = torch.bfloat16 if job["out_dtype"] == "bf16" else torch.float16
        pack = 2 if job["data_format"] == "fp4" else 1
        # Fused prep kernels quantize activations to MXFP8 for a8w4 weights.
        quant_mode = "fp4" if job["data_format"] == "fp4" else "fp8"
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
                    *_preshuffled_scale_shape(rows, job["model_dim"], warp_tile_m),
                ),
                dtype=torch.uint8,
            )
            sw1 = torch.empty(
                (
                    job["experts"],
                    *_preshuffled_b_scale_shape(2 * job["inter_dim"], job["model_dim"]),
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
                    *_preshuffled_scale_shape(rows, job["inter_dim"], warp_tile_m),
                ),
                dtype=torch.uint8,
            )
            sw2 = torch.empty(
                (
                    job["experts"],
                    *_preshuffled_b_scale_shape(job["model_dim"], job["inter_dim"]),
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
            # the GEMMs (fused route+quant+scatter, grouped quant+preshuffle,
            # contiguous prefix-sum(+remap), gather-reduce). These were fused in
            # gfx1250_moe_splitk_fused, so AOT mirrors the new wrappers, not the
            # legacy route_maps/scatter_copy kernels.
            _compile_grouped_moe_aux_kernels(
                job,
                dtype=dtype,
                quant_mode=quant_mode,
                wmma_rep=warp_tile_m // 16,
                contiguous=contiguous,
            )
        elapsed = time.time() - t0
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  arch={aot_arch}")
        return {**job, "compile_time": elapsed, "compile_arch": aot_arch}
    except Exception as e:
        # Catch everything and return cleanly with compile_time=None: the AOT pool
        # keys off exitcode 0 + compile_time=None to mark "produced no kernel" and
        # NOT retry it. An escaping exception crashes the worker (exitcode != 0),
        # which the pool misreads as a transient failure and retries -> deadlock.
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")
        traceback.print_exc()
        return {**job, "compile_time": None, "compile_arch": aot_arch}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    args = parser.parse_args(argv)
    jobs = collect_aot_jobs(args.csv, parse_csv)

    total_t0 = time.time()
    print(f"--- Compiling {len(jobs)} grouped-MoE kernels ---")
    results = run_jobs_parallel(compile_one_config, jobs)
    total_elapsed = time.time() - total_t0

    ok = sum(1 for r in results if r.get("compile_time") is not None)
    fail = len(results) - ok
    print(
        f"\nSummary: {ok} ok, {fail} failed in {total_elapsed:.1f}s",
    )
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
