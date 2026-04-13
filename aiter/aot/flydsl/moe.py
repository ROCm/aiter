#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for MoE / Mixed-MoE FlyDSL kernels from aiter CSV configs.

Reads a tuned CSV config file (e.g. dsv3_fp4_tuned_fmoe.csv), extracts all
unique FlyDSL kernel names, and pre-compiles them into the cache.

Usage:
    # Compile all unique FlyDSL kernels from CSV
    python -m aiter.aot.moe

    # Compile and run kernels on GPU (triggers real compilation + cache write)
    python -m aiter.aot.moe --run_kernel

    # Also test that invalid tile sizes are properly rejected
    python -m aiter.aot.moe --test_bad_tile

    # Custom CSV file
    python -m aiter.aot.moe --csv /path/to/config.csv

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH                      Target GPU architecture (e.g. gfx942, gfx950).
"""

import argparse
import csv
import os
import sys
import time

from aiter.ops.flydsl.moe_kernels import (
    compile_flydsl_moe_stage1,
    compile_flydsl_moe_stage2,
    flydsl_moe_stage1,
    flydsl_moe_stage2,
    get_flydsl_kernel_params,
)

_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../configs/model_configs"
)

DEFAULT_CSVS = [
    os.path.join(_CONFIGS_DIR, "dsv3_fp4_tuned_fmoe.csv"),
    os.path.join(_CONFIGS_DIR, "kimik2_fp4_tuned_fmoe.csv"),
]


def parse_csv(csv_path: str):
    """Parse the CSV and return a list of unique compile jobs.

    Each job is a dict with keys:
        kernel_name, stage, model_dim, inter_dim, experts, topk,
        doweight_stage1 (for stage1), and all params from get_flydsl_kernel_params.

    Deduplicates by (kernel_name, model_dim, inter_dim, experts, topk).
    """
    jobs = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_dim = int(row["model_dim"])
            inter_dim = int(row["inter_dim"])
            experts = int(row["expert"])
            topk = int(row["topk"])
            doweight_stage1 = bool(int(row.get("doweight_stage1", "0")))

            for col in ("kernelName1", "kernelName2"):
                name = row.get(col, "").strip()
                if not name or not name.startswith("flydsl_"):
                    continue

                key = (name, model_dim, inter_dim, experts, topk)
                if key in seen:
                    continue
                seen.add(key)

                params = get_flydsl_kernel_params(name)
                if params is None:
                    print(f"  [WARN] Unknown kernel name: {name}, skipping")
                    continue

                jobs.append(
                    {
                        "kernel_name": name,
                        "model_dim": model_dim,
                        "inter_dim": inter_dim,
                        "experts": experts,
                        "topk": topk,
                        "doweight_stage1": doweight_stage1,
                        **params,
                    }
                )

    return jobs


def _run_kernel(
    stage: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    doweight_stage1: bool = False,
    waves_per_eu: int = 3,
    k_batch: int = 1,
    b_nt: int = 2,
    gate_only: bool = False,
    fuse_fp4_quant: bool = False,
    mode: str = "atomic",
    persist: bool = False,
    sort_block_m: int = 0,
    **kwargs,
):
    """Launch the compiled kernel with random data to verify it runs on GPU."""
    import torch
    from aiter.fused_moe import moe_sorting
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
    from flydsl.runtime.device import get_rocm_arch

    E = experts
    tokens = tile_m  # minimal: one M-tile
    device = torch.device("cuda")
    torch.manual_seed(0)

    ARCH = get_rocm_arch()
    DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

    # --- Routing buffers ---
    _sort_block_m = max(32, tile_m)
    score = torch.randn(tokens, E, device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).float()

    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, _moe_buf = moe_sorting(
        topk_ids.to(torch.int32),
        topk_weights,
        E,
        model_dim,
        torch.float16,
        _sort_block_m,
    )
    if num_valid_ids.numel() > 1:
        num_valid_ids = num_valid_ids[:1].contiguous()

    is_fp4 = b_dtype == "fp4"

    # --- Compute torch reference for sanity check ---
    # Reference uses per-token per-expert matmul on fp32 inputs (before quantization).
    def _stage1_ref(x, w1):
        """Reference: per-token silu(x @ W_gate.T) * (x @ W_up.T)."""
        w_gate = w1[:, :inter_dim, :]  # (E, inter_dim, model_dim)
        w_up = w1[:, inter_dim:, :]  # (E, inter_dim, model_dim)
        out = torch.zeros(tokens, topk, inter_dim, device=device, dtype=torch.float32)
        for t in range(tokens):
            for k in range(topk):
                eid = topk_ids[t, k].item()
                g = x[t] @ w_gate[eid].T  # (inter_dim,)
                u = x[t] @ w_up[eid].T
                out[t, k] = torch.nn.functional.silu(g) * u
        return out

    def _stage2_ref(a2, w2):
        """Reference: per-token a2 @ W2.T, weighted sum over topk."""
        a2_3d = a2.view(tokens, topk, inter_dim)
        out = torch.zeros(tokens, model_dim, device=device, dtype=torch.float32)
        for t in range(tokens):
            for k in range(topk):
                eid = topk_ids[t, k].item()
                out[t] += topk_weights[t, k] * (a2_3d[t, k] @ w2[eid].T)
        return out

    def _print_diff(c_out, c_ref):
        """Print output stats and diff against reference."""
        is_tuple = isinstance(c_out, tuple)
        c_raw = c_out[0] if is_tuple else c_out
        # fuse_fp4_quant output is fp4-packed (half elements) — skip diff
        if c_raw.numel() != c_ref.numel():
            print(
                f"    output shape={tuple(c_raw.shape)} " f"(quantized, skip ref diff)"
            )
            return
        c_check = c_raw.to(torch.float32).view(c_ref.shape)
        abs_diff = (c_check - c_ref).abs()
        print(
            f"    output shape={tuple(c_raw.shape)}, "
            f"max={c_check.abs().max().item():.4f}, "
            f"mean={c_check.abs().mean().item():.4f}"
        )
        print(
            f"    ref check: max_diff={abs_diff.max().item():.4f}, "
            f"mean_diff={abs_diff.mean().item():.4f}"
        )

    if stage == 1:
        x_fp32 = torch.randn(tokens, model_dim, device=device, dtype=torch.float32)
        w1_fp32 = torch.randn(
            E, 2 * inter_dim, model_dim, device=device, dtype=torch.float32
        )
        c_ref = _stage1_ref(x_fp32, w1_fp32)

        if is_fp4:
            x_fp4, x_scale = per_1x32_f4_quant(x_fp32)
            w1_flat = w1_fp32.view(E * (2 * inter_dim), model_dim)
            w1_fp4, w1_scale_raw = per_1x32_f4_quant(w1_flat)
            del w1_fp32, w1_flat

            # Preshuffle weights
            w1_shuffled = shuffle_weight(
                w1_fp4.view(E, 2 * inter_dim, model_dim // 2).view(
                    torch.float4_e2m1fn_x2
                )
            )
            w1_kernel = w1_shuffled.view(torch.uint8).contiguous()

            # Prepare scales (e8m0_shuffle expects 2D)
            w1_scale_1d = e8m0_shuffle(w1_scale_raw).view(torch.uint8).contiguous()
            a1_scale_1d = (
                moe_mxfp4_sort(
                    x_scale[:tokens, :].view(tokens, 1, -1),
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=tokens,
                    block_size=_sort_block_m,
                )
                .view(torch.uint8)
                .contiguous()
            )
            x_q = x_fp4.view(torch.uint8).contiguous().view(tokens, -1)

            _fuse_fq = fuse_fp4_quant
            c_out = flydsl_moe_stage1(
                x_q,
                w1_kernel.view(E, 2 * inter_dim, model_dim // 2),
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                w1_scale=w1_scale_1d,
                a1_scale=a1_scale_1d,
                sorted_weights=sorted_w if doweight_stage1 else None,
                k_batch=k_batch,
                waves_per_eu=waves_per_eu,
                b_nt=b_nt,
                gate_only=gate_only,
                fuse_fp4_quant=_fuse_fq,
                fuse_sort_scale=_fuse_fq,
            )
        else:
            # FP8 path
            from aiter import pertoken_quant

            x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=DTYPE_FP8)
            w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=DTYPE_FP8)
            del w1_fp32
            w1_shuffled = shuffle_weight(w1_q)

            c_out = flydsl_moe_stage1(
                x_q,
                w1_shuffled,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                w1_scale=scale_w1,
                a1_scale=scale_x,
                sorted_weights=sorted_w if doweight_stage1 else None,
                k_batch=k_batch,
                waves_per_eu=waves_per_eu,
                b_nt=b_nt,
                gate_only=gate_only,
            )

        torch.cuda.synchronize()
        _print_diff(c_out, c_ref)
        del c_ref

    elif stage == 2:
        a2_fp32 = torch.randn(
            tokens * topk, inter_dim, device=device, dtype=torch.float32
        )
        w2_fp32 = torch.randn(
            E, model_dim, inter_dim, device=device, dtype=torch.float32
        )
        c_ref = _stage2_ref(a2_fp32, w2_fp32)

        if is_fp4:
            a2_fp4, a2_scale = per_1x32_f4_quant(a2_fp32)
            w2_flat = w2_fp32.view(E * model_dim, inter_dim)
            w2_fp4, w2_scale_raw = per_1x32_f4_quant(w2_flat)
            del w2_fp32, w2_flat

            # Preshuffle weights
            w2_shuffled = shuffle_weight(
                w2_fp4.view(E, model_dim, inter_dim // 2).view(torch.float4_e2m1fn_x2)
            )
            w2_kernel = w2_shuffled.view(torch.uint8).contiguous()

            # Prepare scales (e8m0_shuffle expects 2D)
            w2_scale_1d = e8m0_shuffle(w2_scale_raw).view(torch.uint8).contiguous()
            a2_scale_1d = (
                moe_mxfp4_sort(
                    a2_scale.view(tokens, topk, -1),
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=tokens,
                    block_size=_sort_block_m,
                )
                .view(torch.uint8)
                .contiguous()
            )
            a2_q = (
                a2_fp4.view(torch.uint8).contiguous().view(tokens, topk, inter_dim // 2)
            )

            c_out = flydsl_moe_stage2(
                a2_q,
                w2_kernel.view(E, model_dim, inter_dim // 2),
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                mode=mode,
                w2_scale=w2_scale_1d,
                a2_scale=a2_scale_1d,
                sorted_weights=sorted_w,
                sort_block_m=sort_block_m,
                persist=persist,
            )
        else:
            # FP8 path
            from aiter import pertoken_quant

            a2_q, scale_a2 = pertoken_quant(a2_fp32, quant_dtype=DTYPE_FP8)
            w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=DTYPE_FP8)
            del w2_fp32
            w2_shuffled = shuffle_weight(w2_q)

            c_out = flydsl_moe_stage2(
                a2_q.view(tokens, topk, inter_dim),
                w2_shuffled,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                mode=mode,
                w2_scale=scale_w2,
                a2_scale=scale_a2,
                sorted_weights=sorted_w,
                sort_block_m=sort_block_m,
                persist=persist,
            )

        torch.cuda.synchronize()
        _print_diff(c_out, c_ref)
        del c_ref


def compile_one_config(
    kernel_name: str,
    stage: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp4",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    doweight_stage1: bool = False,
    waves_per_eu: int = 3,
    k_batch: int = 1,
    b_nt: int = 2,
    gate_only: bool = False,
    fuse_fp4_quant: bool = False,
    mode: str = "atomic",
    persist: bool = False,
    sort_block_m: int = 0,
    **kwargs,
) -> dict:
    """Compile one MoE kernel configuration and save to cache.

    Uses COMPILE_ONLY=1 to trigger MLIR compilation and pkl cache write
    without executing the kernel on GPU.

    Returns a dict with timing info.
    """
    shape_str = (
        f"{kernel_name}  "
        f"model_dim={model_dim} inter_dim={inter_dim} "
        f"E={experts} topk={topk}"
    )
    result = {"kernel_name": kernel_name, "shape": shape_str, "compile_time": None}

    t0 = time.time()
    prev_compile_only = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        _run_kernel(
            stage=stage,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            doweight_stage1=doweight_stage1,
            waves_per_eu=waves_per_eu,
            k_batch=k_batch,
            b_nt=b_nt,
            gate_only=gate_only,
            fuse_fp4_quant=fuse_fp4_quant,
            mode=mode,
            persist=persist,
            sort_block_m=sort_block_m,
        )

        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}")
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}: {e}")
    finally:
        if prev_compile_only is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = prev_compile_only

    return result


BAD_TILE_CONFIGS = [
    # (model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k)
    # tile_k=17 makes tile_k_bytes indivisible by 64
    (7168, 256, 257, 9, 32, 64, 17),
]


def test_bad_tile_error():
    """Verify that an unsupported tile size produces a clear compile error."""
    for cfg in BAD_TILE_CONFIGS:
        model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k = cfg
        shape_str = (
            f"model_dim={model_dim} inter_dim={inter_dim} "
            f"E={experts} topk={topk} tile=({tile_m},{tile_n},{tile_k})"
        )
        try:
            compile_flydsl_moe_stage1(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage1=False,
                a_dtype="fp4",
                b_dtype="fp4",
                out_dtype="bf16",
            )
            raise AssertionError(f"No error raised for bad tile: {shape_str}")
        except (ValueError, RuntimeError) as e:
            print(f"  [OK] Correctly rejected bad tile: {shape_str}")
            print(f"       Error: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile MoE / Mixed-MoE FlyDSL kernels from aiter CSV config",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to tuned CSV config file(s)",
    )
    parser.add_argument(
        "--test_bad_tile",
        action="store_true",
        help="Also test that an invalid tile size is properly rejected",
    )
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

    cache_dir = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    arch = os.environ.get("ARCH", "(auto-detect)")

    all_jobs = []
    for csv_path in csv_paths:
        all_jobs.extend(parse_csv(csv_path))

    stage1_jobs = [j for j in all_jobs if j["stage"] == 1]
    stage2_jobs = [j for j in all_jobs if j["stage"] == 2]

    print("=" * 72)
    print("FlyDSL MoE AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Stage1 jobs:  {len(stage1_jobs)}")
    print(f"  Stage2 jobs:  {len(stage2_jobs)}")
    print(f"  Total jobs:   {len(all_jobs)}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Target arch:  {arch}")
    print("=" * 72)

    total_t0 = time.time()
    results = []

    if stage1_jobs:
        print(f"\n--- Stage 1 ({len(stage1_jobs)} kernels) ---")
        for i, job in enumerate(stage1_jobs, 1):
            print(f"\n[{i}/{len(stage1_jobs)}] ", end="")
            r = compile_one_config(**job)
            results.append(r)

    if stage2_jobs:
        print(f"\n--- Stage 2 ({len(stage2_jobs)} kernels) ---")
        for i, job in enumerate(stage2_jobs, 1):
            print(f"\n[{i}/{len(stage2_jobs)}] ", end="")
            r = compile_one_config(**job)
            results.append(r)

    total_elapsed = time.time() - total_t0

    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = sum(1 for r in results if r["compile_time"] is None)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Compiled:     {ok} ok, {fail} failed")
    print(f"  Cache dir:    {cache_dir}")

    if args.test_bad_tile:
        test_bad_tile_error()

    print()

    exit_code = 0
    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        exit_code = 1
    else:
        print("All compilations succeeded. Cache is ready.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
