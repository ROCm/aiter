# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compile for the FlyDSL mxmoe a4w4 MoE port (gemm1 / gemm2).

Parses the flydsl_mxmoe_* port rows from the fp4 tuned CSVs and warms the FlyDSL
disk cache via the same runtime entry points, keyed identically so inference
hits the cache.

Standalone:
    python -m aiter.aot.flydsl.mxfp4_moe [--csv /path/to/foo_fp4_tuned_fmoe.csv]
"""

import argparse
import csv
import glob
import os
import sys
import time

from aiter.aot.flydsl.common import collect_aot_jobs, compile_only_env, override_env
from aiter.jit.core import AITER_ROOT_DIR

_MODEL_CONFIG_DIR = f"{AITER_ROOT_DIR}/aiter/configs/model_configs"
DEFAULT_CSVS = sorted(glob.glob(f"{_MODEL_CONFIG_DIR}/*_fp4_tuned_fmoe.csv"))

# Mirror the runtime gate so the default build skips the opt-in mxfp4-out path.
_MXFP4_INTERMEDIATE = os.environ.get("AITER_MXFP4_INTERMEDIATE", "0") not in ("0", "")


def _job_key(job: dict) -> tuple:
    """Dedup key == the runtime FlyDSL cache key."""
    if job["stage"] == 1:
        return (
            1,
            job["BM"],
            job["use_nt"],
            job["inline_quant"],
            job["D_HIDDEN"],
            job["D_INTER"],
            job["NE"],
            job["topk"],
            job["xcd_swizzle"],
        )
    return (
        2,
        job["BM"],
        job["use_nt"],
        job["NE"],
        job["N_OUT"],
        job["epilog"],
        job["D_INTER"],
        job["D_INTER_REAL"],
        job["xcd_swizzle"],
    )


def parse_csv(csv_path: str):
    """Parse an fp4 tuned CSV into unique mxmoe-port compile jobs (one per stage)."""
    from aiter.ops.flydsl.mxfp4_kname import (
        _is_mxfp4_kname,
        _parse_mxfp4_g1_kname,
        _parse_mxfp4_g2_kname,
    )
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import _epilog_of

    jobs = []
    seen = set()

    def _add(job):
        key = _job_key(job)
        if key in seen:
            return
        seen.add(key)
        jobs.append(job)

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            topk = int(row["topk"])
            # Shape from CSV columns (not the name). Weight is stored at the
            # padded D_INTER; D_INTER_REAL is set only for non-256-aligned shards.
            model_dim = int(row["model_dim"])
            expert = int(row["expert"])
            inter_dim = int(row["inter_dim"])
            d_inter = ((inter_dim + 255) // 256) * 256
            d_inter_real = inter_dim if inter_dim != d_inter else None

            kn1 = (row.get("kernelName1") or "").strip()
            if _is_mxfp4_kname(kn1):
                p1 = _parse_mxfp4_g1_kname(kn1)
                _add(
                    {
                        "stage": 1,
                        "kernel_name": kn1,
                        "BM": p1["BM"],
                        "use_nt": p1["use_nt"],
                        "inline_quant": p1["inline_quant"],
                        "D_HIDDEN": model_dim,
                        "D_INTER": d_inter,
                        "NE": expert,
                        "topk": topk,
                        "xcd_swizzle": p1["xcd_swizzle"],
                    }
                )

            kn2 = (row.get("kernelName2") or "").strip()
            if _is_mxfp4_kname(kn2):
                p2 = _parse_mxfp4_g2_kname(kn2)
                if p2["mxfp4out"] and not _MXFP4_INTERMEDIATE:
                    continue
                _add(
                    {
                        "stage": 2,
                        "kernel_name": kn2,
                        "BM": p2["BM"],
                        "use_nt": p2["use_nt"],
                        "NE": expert,
                        "N_OUT": model_dim,
                        "epilog": _epilog_of(
                            p2["atomic"], p2["mxfp4out"], p2["cshuffle"]
                        ),
                        "D_INTER": d_inter,
                        "D_INTER_REAL": d_inter_real,
                        "topk": topk,  # unused by the kernel; for the entry signature
                        "xcd_swizzle": p2["xcd_swizzle"],
                    }
                )

    return jobs


def _dummy(nbytes=256):
    import torch

    # CPU tensor: AOT precompile is GPU-free; only data_ptr()/.device are read
    # and nothing is dispatched under COMPILE_ONLY.
    return torch.zeros(nbytes, dtype=torch.uint8, device="cpu")


def _compile_stage1(job):
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

    d = _dummy()
    flydsl_mxfp4_gemm1(
        a_quant=d,
        a_scale_sorted_shuffled=d,
        w1_u8=d,
        w1_scale_u8=d,
        sorted_expert_ids=d,
        cumsum_tensor=d,
        m_indices=d,
        inter_sorted_quant=d,
        inter_sorted_shuffled_scale=d,
        hidden_states=d,
        n_tokens=job["BM"],
        BM=job["BM"],
        use_nt=job["use_nt"],
        inline_quant=job["inline_quant"],
        NE=job["NE"],
        D_HIDDEN=job["D_HIDDEN"],
        D_INTER=job["D_INTER"],
        topk=job["topk"],
        xcd_swizzle=job["xcd_swizzle"],
        stream=0,
    )


def _compile_stage2(job):
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2

    epilog = job["epilog"]
    mxfp4out = epilog == "nonatomic_mxfp4"
    d = _dummy()
    flydsl_mxfp4_gemm2(
        inter_sorted_quant=d,
        inter_sorted_shuffled_scale=d,
        w2_u8=d,
        w2_scale_u8=d,
        sorted_expert_ids=d,
        cumsum_tensor=d,
        sorted_token_ids=d,
        sorted_weights=d,
        flat_out=d,
        M_logical=job["BM"],
        max_sorted=job["BM"],
        BM=job["BM"],
        use_nt=job["use_nt"],
        atomic=epilog == "atomic",
        mxfp4out=mxfp4out,
        NE=job["NE"],
        D_HIDDEN=job["N_OUT"],
        D_INTER=job["D_INTER"],
        topk=job["topk"],
        flat_out_scale=_dummy() if mxfp4out else None,
        cshuffle=epilog == "nonatomic_cshuffle",
        D_INTER_REAL=job["D_INTER_REAL"],
        xcd_swizzle=job["xcd_swizzle"],
        stream=0,
    )


def compile_one_config(**job):
    stage = job["stage"]
    shape_str = (
        f"{job['kernel_name']} NE={job['NE']} D_INTER={job['D_INTER']} BM={job['BM']}"
    )
    result = {"kernel_name": job["kernel_name"], "stage": stage, "compile_time": None}

    t0 = time.time()
    try:
        # mxfp4 a4w4 kernels are gfx950-only. In the GPU-free AOT build,
        # get_rocm_arch() detects gfx942 and the gfx950 intrinsics fail to
        # select (LLVM aborts), so pin FLYDSL_GPU_ARCH=gfx950.
        with compile_only_env(), override_env("FLYDSL_GPU_ARCH", "gfx950"):
            if stage == 1:
                _compile_stage1(job)
            else:
                _compile_stage2(job)
        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  stage{stage}  {shape_str}")
    except Exception as e:
        print(f"  [FAIL] compile  stage{stage}  {shape_str}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL mxmoe a4w4 MoE port kernels from fp4 tuned CSVs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to fp4 tuned CSV(s); default: all *_fp4_tuned_fmoe.csv",
    )
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
    if not csv_paths:
        print("Error: no fp4 tuned CSVs found and none given via --csv")
        sys.exit(1)

    all_jobs = collect_aot_jobs(csv_paths, parse_csv)
    stage1_jobs = [j for j in all_jobs if j["stage"] == 1]
    stage2_jobs = [j for j in all_jobs if j["stage"] == 2]

    print("=" * 72)
    print("FlyDSL mxmoe a4w4 MoE-port AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Stage1 jobs:  {len(stage1_jobs)}")
    print(f"  Stage2 jobs:  {len(stage2_jobs)}")
    print(f"  Total jobs:   {len(all_jobs)}")
    print("=" * 72)

    total_t0 = time.time()
    results = []
    for i, job in enumerate(stage1_jobs + stage2_jobs, 1):
        print(f"\n[{i}/{len(all_jobs)}] ", end="")
        results.append(compile_one_config(**job))

    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = sum(1 for r in results if r["compile_time"] is None)
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {time.time() - total_t0:.1f}s")
    print(f"  Compiled:     {ok} ok, {fail} failed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
