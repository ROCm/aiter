#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL implicit-GEMM convolution.

Reads the tuned conv3d CSV, turns every row into a compile job, and fills the
FlyDSL cache so a model run never pays JIT. The cold cost this removes is
large and measured: a first Wan2.1 encode spends ~4.5 minutes in step 1 on
compilation, and a full 679-candidate tuning sweep took 969s cold against 40s
warm.

Two kernels per row, not one:

* ``conv3d_implicit_kernel`` -- the convolution itself.
* ``transpose_ncdhw_ndhwc`` -- the GEMM is channels-last inside, so an NCDHW
  input (the default, and what diffusers hands us) pays a layout conversion
  first. It is a separate ``lru_cache``, so precompiling only the convolution
  would leave this one to JIT.

Unlike the GEMM AOT, no kernel name has to be parsed: the tuned CSV stores the
launch config as five explicit integer columns, so a job is read straight off
the row.

Coverage is exactly the CSV, with no generalisation. ``compile_conv3d_implicit``
takes the whole problem shape as compile-time constants -- the im2col div/mod
folding against ``(kT, kH, kW)`` and ``C/groups`` is where this kernel's
performance comes from -- so a resolution, frame count, bias flag or output
layout outside the table still JITs. ``op_tests/tuning_tests/test_conv3d_aot.py``
pins that down by re-running the op under ``run_only_env()``, where an uncovered
shape raises instead of silently falling back.

Usage::

    # Compile everything in the default (merged) tuned CSV
    python -m aiter.aot.flydsl.conv

    # Custom CSV file(s)
    python -m aiter.aot.flydsl.conv --csv /path/to/conv3d_bf16_tuned.csv

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    GPU_ARCHS / ARCH          Restrict compilation to these architectures.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    cu_num_to_arch,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.kernels.conv3d_implicit import (
    TR_MAX_BIG_S,
    TR_VEC,
    _dispatch,
    _pad_channels,
    compile_conv3d_implicit,
    compile_transpose_ncdhw_ndhwc,
)

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_CONV3D_BF16_FILE]
CONV_AOT_ARCH_DEFAULT = "gfx950"

# Mirrors conv3d_implicit.TUNED_KEY_COLUMNS; the tuned CSV carries these plus
# gfx/cu_num and the five launch-config columns.
_INT_COLS = (
    "N",
    "C",
    "D",
    "H",
    "W",
    "K",
    "kT",
    "kH",
    "kW",
    "stride_d",
    "stride_h",
    "stride_w",
    "pad_d",
    "pad_h",
    "pad_w",
    "dil_d",
    "dil_h",
    "dil_w",
    "groups",
)
_CONFIG_COLS = ("tile_m", "tile_n", "wave_m", "wave_n", "wgm")


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no"}:
        return False
    if normalized in {"1", "true", "yes"}:
        return True
    raise ValueError(f"Expected True/False, got {value!r}")


def _is_matmul_fast_path(row: dict) -> bool:
    """Rows the op answers with torch.matmul, so there is no kernel to compile."""
    return (
        row["groups"] == 1
        and row["kT"] == row["kH"] == row["kW"] == 1
        and row["stride_d"] == row["stride_h"] == row["stride_w"] == 1
        and row["pad_d"] == row["pad_h"] == row["pad_w"] == 0
    )


def parse_csv(csv_path: str):
    """Parse the tuned conv CSV into unique conv and transpose compile jobs."""
    jobs = []
    seen = set()

    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            missing = [c for c in (*_INT_COLS, *_CONFIG_COLS) if c not in row]
            if missing:
                print(f"  [WARN] {csv_path}: missing columns {missing}, skipping row")
                continue
            try:
                shape = {c: int(row[c]) for c in _INT_COLS}
                config = {c: int(row[c]) for c in _CONFIG_COLS}
                has_bias = _parse_bool(row.get("bias"))
                # Recorded by the tuner. Re-deriving it here would need the
                # target's CU count, which a build host may not have.
                splitk = int(row.get("splitK") or 1) or 1
            except ValueError as exc:
                print(f"  [WARN] {csv_path}: unparsable row ({exc}), skipping")
                continue

            if _is_matmul_fast_path(shape):
                continue

            cu_num = int(row.get("cu_num") or 0)
            gfx = row.get("gfx", "")

            groups = shape["groups"]
            cgp = _pad_channels(shape["C"] // groups)
            c_padded = groups * cgp

            conv_job = {
                "kind": "conv3d",
                "kernel_name": "conv3d_implicit_kernel",
                "cu_num": cu_num,
                "gfx": gfx,
                "c_padded": c_padded,
                "has_bias": has_bias,
                "splitk": max(1, splitk),
                **shape,
                **config,
            }
            key = job_identity(conv_job)
            if key not in seen:
                seen.add(key)
                jobs.append(conv_job)

            # The NCDHW->NHWC pre-transpose. Keyed only on (n, padded C, T*H*W),
            # so several convolutions collapse onto one job. Skipped where the
            # op itself falls back to torch.permute.
            s = shape["D"] * shape["H"] * shape["W"]
            big = shape["N"] * c_padded * s > 0x7FFFFFFF
            if c_padded % TR_VEC == 0 and not (big and s > TR_MAX_BIG_S):
                tr_job = {
                    "kind": "transpose",
                    "kernel_name": "transpose_ncdhw_ndhwc",
                    "cu_num": cu_num,
                    "gfx": gfx,
                    "N": shape["N"],
                    "c_padded": c_padded,
                    "s": s,
                }
                key = job_identity(tr_job)
                if key not in seen:
                    seen.add(key)
                    jobs.append(tr_job)

    return jobs


def job_arch(cu_num: int = 0, gfx: str = "") -> str:
    """Target arch a job would compile for -- shared by dispatch and filtering."""
    return gfx or cu_num_to_arch(cu_num, default=CONV_AOT_ARCH_DEFAULT)


def _probe(rank: int, dtype_is_fp32: bool = False):
    """A tiny CPU stand-in for one kernel argument.

    Only the rank and dtype reach the cache key -- the extents are compile-time
    constants baked in by ``compile_*`` -- so eight elements per argument is
    enough, and under ``COMPILE_ONLY`` FlyDSL persists the artifact without
    materialising an execution engine, so the buffer is never dereferenced.
    That is what keeps AOT off the GPU and out of the 442 MiB a real
    ``down_0_1`` activation would cost.

    The rank does matter, though: a rank-1 stand-in compiles fine and then
    misses at runtime, which is silent because the miss just falls back to JIT.
    ``op_tests/tuning_tests/test_conv3d_aot.py`` is what catches that.
    """
    import torch

    shape = (1,) * (rank - 1) + (TR_VEC,) if rank > 1 else (TR_VEC,)
    return torch.empty(
        shape,
        device=torch.device("cpu"),
        dtype=torch.float32 if dtype_is_fp32 else torch.bfloat16,
    )


def _conv_probe_args(splitk: int):
    """Stand-ins for ``(y, x_ndhwc, w_packed, bias)``, matching runtime ranks.

    ``y`` drops to rank 2 on the split-K path, where the epilogue accumulates
    into an ``(npq, k)`` fp32 staging buffer instead of the output tensor.
    """
    y = _probe(2, dtype_is_fp32=True) if splitk > 1 else _probe(5)
    return y, _probe(5), _probe(2), _probe(1, dtype_is_fp32=True)


def _compile_conv3d_to_cache(
    *,
    N: int,
    c_padded: int,
    D: int,
    H: int,
    W: int,
    K: int,
    kT: int,
    kH: int,
    kW: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dil_d: int,
    dil_h: int,
    dil_w: int,
    groups: int,
    has_bias: bool,
    splitk: int,
    tile_m: int,
    tile_n: int,
    wave_m: int,
    wave_n: int,
    wgm: int,
    **kwargs,
):
    del kwargs

    exe = compile_conv3d_implicit(
        N,
        c_padded,
        D,
        H,
        W,
        K,
        kT,
        kH,
        kW,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        # The runtime only reaches the table for zero padding (an asymmetric pad
        # cannot be expressed with one value per axis), so this is the only mode
        # a tuned row can describe.
        "zeros",
        has_bias,
        splitk,
        (tile_m, tile_n, wave_m, wave_n),
        wgm,
        groups,
        # The CSV does not carry a layout, and NCDHW is the default. A
        # channels-last caller compiles a different artifact.
        False,
    )
    with compile_only_env():
        _dispatch(exe, *_conv_probe_args(splitk), stream=None)


def _compile_transpose_to_cache(*, N: int, c_padded: int, s: int, **kwargs):
    del kwargs

    exe = compile_transpose_ncdhw_ndhwc(N, c_padded, s)
    with compile_only_env():
        # (n, t, h, w, c) out, (n, c, t, h, w) in -- both rank 5.
        _dispatch(exe, _probe(5), _probe(5), stream=None)


def compile_one_config(
    kind: str,
    kernel_name: str,
    cu_num: int = 0,
    gfx: str = "",
    **kwargs,
) -> dict:
    """Compile one conv or transpose configuration into the cache."""
    aot_arch = job_arch(cu_num, gfx)
    if kind == "transpose":
        shape_str = (
            f"{kernel_name}  N={kwargs['N']} C={kwargs['c_padded']} S={kwargs['s']}"
        )
    else:
        shape_str = (
            f"{kernel_name}  {kwargs['N']}x{kwargs['C']}x{kwargs['D']}x"
            f"{kwargs['H']}x{kwargs['W']}->{kwargs['K']} "
            f"k{kwargs['kT']}{kwargs['kH']}{kwargs['kW']} "
            f"tile={kwargs['tile_m']}x{kwargs['tile_n']}"
        )
    result = {
        "kernel_name": kernel_name,
        "kind": kind,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    t0 = time.time()
    try:
        with override_env("FLYDSL_GPU_ARCH", aot_arch):
            if kind == "conv3d":
                _compile_conv3d_to_cache(**kwargs)
            elif kind == "transpose":
                _compile_transpose_to_cache(**kwargs)
            else:
                raise ValueError(f"Unknown conv AOT kind: {kind}")
        result["compile_time"] = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile the FlyDSL conv3d kernels from aiter CSV config",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to tuned CSV config file(s); defaults come from AITER_CONFIGS",
    )
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS")

    all_jobs = collect_aot_jobs(csv_paths, parse_csv)
    if arch:
        arch_set = {a.strip() for a in re.split(r"[;,]", arch) if a.strip()}
        n_before = len(all_jobs)
        all_jobs = [
            j for j in all_jobs if job_arch(j["cu_num"], j.get("gfx", "")) in arch_set
        ]
        print(f"[aiter] ARCH={arch}: {len(all_jobs)}/{n_before} jobs match")

    conv_jobs = [j for j in all_jobs if j["kind"] == "conv3d"]
    tr_jobs = [j for j in all_jobs if j["kind"] == "transpose"]

    print("=" * 72)
    print("FlyDSL conv3d AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:              {csv_path}")
    print(f"  conv3d jobs:      {len(conv_jobs)}")
    print(f"  transpose jobs:   {len(tr_jobs)}")
    print(f"  Total jobs:       {len(all_jobs)}")
    print(f"  Cache dir:        {cache_dir}")
    print(f"  Target arch:      {arch or '(all archs found in CSVs)'}")
    print("=" * 72)

    total_t0 = time.time()
    print(f"\n--- Compiling {len(all_jobs)} kernels ---")
    results = run_jobs_parallel(compile_one_config, conv_jobs + tr_jobs)
    total_elapsed = time.time() - total_t0

    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = sum(1 for r in results if r["compile_time"] is None)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Compiled:     {ok} ok, {fail} failed")
    print(f"  Cache dir:    {cache_dir}")
    print()

    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        sys.exit(1)
    print("All compilations succeeded. Cache is ready.")


if __name__ == "__main__":
    main()
