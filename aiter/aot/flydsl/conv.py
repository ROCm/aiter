#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT for the FlyDSL implicit-GEMM convolution.

Each tuned CSV row becomes two conv jobs (``out_ndhwc`` True/False) plus the
NCDHW->NDHWC pre-transpose. Input layout is not in the compile key.

``AITER_CONV3D_DYN_HW`` (default 1) is part of the compile key: one artifact
per layer instead of per resolution. Build and runtime must agree, or the
cache misses silently. Use ``run_only_env()`` to fail on a JIT.

Usage::

    python -m aiter.aot.flydsl.conv
    python -m aiter.aot.flydsl.conv --csv /path/to/bf16_tuned_conv3d.csv
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
from aiter.ops.flydsl.conv_kernels import (
    AITER_CONV3D_DYN_HW,
    LIBTYPE_FLYDSL,
    TUNED_KEY_COLUMNS,
    TUNED_LIBTYPE_COLUMN,
    TUNED_RESULT_COLUMNS,
    _dispatch,
    _dyn_hw_ok,
    _implicit_param_from_problem,
    _is_matmul_fast_path,
    _pad_channels,
    _parse_tuned_bool,
    _resolve_splitk,
)
from aiter.ops.flydsl.kernels.conv3d_gfx950_utils import (
    LDG_VEC,
    make_conv_geometry,
    make_launch_grid,
    make_output_scatter_plan,
    make_tile_config,
    out_extent,
)
from aiter.ops.flydsl.kernels.conv3d_im2col import make_im2col_plan
from aiter.ops.flydsl.kernels.conv3d_implicit_gfx950 import (
    _shape_agnostic_key,
    compile_conv3d_implicit,
)
from aiter.ops.flydsl.kernels.conv3d_transpose import (
    TR_MAX_BIG_S,
    TR_VEC,
    compile_transpose_ncdhw_ndhwc,
)

# _pad_channels rounds C to LDG_VEC, which only implies the transpose's
# c % TR_VEC == 0 while these two widths stay compatible.
assert LDG_VEC % TR_VEC == 0, (
    f"channel padding rounds to a multiple of LDG_VEC={LDG_VEC}, which no longer "
    f"guarantees the transpose's c % {TR_VEC} == 0; parse_csv has to test it again"
)

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_CONV3D_BF16_FILE]
CONV_AOT_ARCH_DEFAULT = "gfx950"

# Innermost probe extent; must be >1 so the unit stride stays on the last axis.
_PROBE_EXTENT = 8

_INT_COLS = tuple(c for c in TUNED_KEY_COLUMNS if c != "bias")
_CONFIG_COLS = TUNED_RESULT_COLUMNS

# Dropped from the dyn_hw compile-dedupe key (one artifact per layer).
_RESOLUTION_COLS = ("N", "D", "H", "W")


def _conv_dedupe_key(job):
    """Compile identity: whole row, or ``_shape_agnostic_key`` plus non-extent fields.

    Falls back to the whole row if plans cannot be derived here.
    """
    if not job["dyn_hw"]:
        return job_identity(job)
    tile = (job["tile_m"], job["tile_n"], job["wave_m"], job["wave_n"])
    try:
        param = _implicit_param_from_problem(
            job["N"],
            job["C"],
            job["D"],
            job["H"],
            job["W"],
            job["K"],
            job["kT"],
            job["kH"],
            job["kW"],
            job["stride_d"],
            job["stride_h"],
            job["stride_w"],
            job["pad_d"],
            job["pad_h"],
            job["pad_w"],
            job["dil_d"],
            job["dil_h"],
            job["dil_w"],
            job["groups"],
            job["has_bias"],
            job["splitk"],
            tile,
            job["wgm"],
            job["out_ndhwc"],
            "zeros",
            True,
        )
        cfg = make_tile_config(param.tile)
        geom = make_conv_geometry(param)
        grid = make_launch_grid(param, geom, cfg)
        agnostic = _shape_agnostic_key(
            grid._replace(grid_x=0, grid_z=0, grid_m=0),
            make_im2col_plan(param, geom, cfg),
            make_output_scatter_plan(param, geom, cfg, grid),
        )
    except (AssertionError, ValueError):
        return job_identity(job)
    return (agnostic,) + tuple(
        sorted((k, v) for k, v in job.items() if k not in _RESOLUTION_COLS)
    )


def _row_npq_per_sample(shape) -> int:
    """``Do * Ho * Wo`` for one CSV row, by the kernel's own extent rule."""
    return (
        out_extent(
            shape["D"], shape["pad_d"], shape["dil_d"], shape["kT"], shape["stride_d"]
        )
        * out_extent(
            shape["H"], shape["pad_h"], shape["dil_h"], shape["kH"], shape["stride_h"]
        )
        * out_extent(
            shape["W"], shape["pad_w"], shape["dil_w"], shape["kW"], shape["stride_w"]
        )
    )


def _requested_archs():
    """ARCH / GPU_ARCHS as a set, or None. Applied in parse_csv so setup.py's run_aot sees it."""
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS")
    if not arch:
        return None
    return {a.strip() for a in re.split(r"[;,]", arch) if a.strip()} or None


def parse_csv(csv_path: str):
    """Parse the tuned conv CSV into unique conv and transpose compile jobs."""
    jobs = []
    seen = set()
    keep_archs = _requested_archs()

    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            # Empty libtype means FlyDSL (pre-libtype tables).
            libtype = row.get(TUNED_LIBTYPE_COLUMN, "")
            if libtype and libtype != LIBTYPE_FLYDSL:
                continue
            missing = [c for c in (*_INT_COLS, *_CONFIG_COLS) if c not in row]
            if missing:
                print(f"  [WARN] {csv_path}: missing columns {missing}, skipping row")
                continue
            try:
                shape = {c: int(row[c]) for c in _INT_COLS}
                config = {c: int(row[c]) for c in _CONFIG_COLS}
                has_bias = _parse_tuned_bool(row.get("bias"))
                # Missing splitK is resolved like the runtime, not defaulted to 1.
                raw_splitk = row.get("splitK", "")
                splitk = (int(raw_splitk) or 1) if raw_splitk else None
            except ValueError as exc:
                print(f"  [WARN] {csv_path}: unparsable row ({exc}), skipping")
                continue

            if _is_matmul_fast_path(shape):
                continue

            cu_num = int(row.get("cu_num") or 0)
            gfx = row.get("gfx", "")
            if keep_archs is not None and job_arch(cu_num, gfx) not in keep_archs:
                continue

            groups = shape["groups"]
            cgp = _pad_channels(shape["C"] // groups)
            c_padded = groups * cgp

            # out_ndhwc is compile-time; input layout is not. Two conv jobs per row.
            tile = (
                config["tile_m"],
                config["tile_n"],
                config["wave_m"],
                config["wave_n"],
            )

            # splitK is in the compile key: match what the runtime will request.
            crs = cgp * shape["kT"] * shape["kH"] * shape["kW"]
            npq = shape["N"] * _row_npq_per_sample(shape)
            splitk = _resolve_splitk(
                splitk,
                npq,
                crs,
                shape["K"],
                None,
                tile,
                groups,
                num_cu=cu_num or None,
            )
            for out_ndhwc in (False, True):
                conv_job = {
                    "kind": "conv3d",
                    "kernel_name": "conv3d_implicit_kernel",
                    "cu_num": cu_num,
                    "gfx": gfx,
                    "has_bias": has_bias,
                    "splitk": splitk,
                    "out_ndhwc": out_ndhwc,
                    # Same derivation as dispatch; dyn_hw is in the compile key.
                    "dyn_hw": _resolve_dyn_hw(
                        shape=shape,
                        splitk=splitk,
                        tile=tile,
                        out_ndhwc=out_ndhwc,
                        has_bias=has_bias,
                    ),
                    **shape,
                    **config,
                }
                key = _conv_dedupe_key(conv_job)
                if key not in seen:
                    seen.add(key)
                    jobs.append(conv_job)

            # Pre-transpose; skipped when the op falls back to torch.permute.
            s = shape["D"] * shape["H"] * shape["W"]
            big = shape["N"] * c_padded * s > 0x7FFFFFFF
            if not (big and s > TR_MAX_BIG_S):
                tr_job = {
                    "kind": "transpose",
                    "kernel_name": "transpose_ncdhw_ndhwc",
                    "cu_num": cu_num,
                    "gfx": gfx,
                    "N": shape["N"],
                    "c_padded": c_padded,
                    "s": s,
                }
                # s is runtime; compile key is (N, C, BIG).
                key = ("transpose", gfx, cu_num, shape["N"], c_padded, big)
                if key not in seen:
                    seen.add(key)
                    jobs.append(tr_job)

    return jobs


def _resolve_dyn_hw(*, shape, splitk, tile, out_ndhwc, has_bias):
    """Whether dispatch would take the variable-resolution path (compile-key field)."""
    probe = _implicit_param_from_problem(
        shape["N"],
        shape["C"],
        shape["D"],
        shape["H"],
        shape["W"],
        shape["K"],
        shape["kT"],
        shape["kH"],
        shape["kW"],
        shape["stride_d"],
        shape["stride_h"],
        shape["stride_w"],
        shape["pad_d"],
        shape["pad_h"],
        shape["pad_w"],
        shape["dil_d"],
        shape["dil_h"],
        shape["dil_w"],
        shape["groups"],
        has_bias,
        splitk,
        tile,
        1,
        out_ndhwc,
        "zeros",
        False,
    )
    geom = make_conv_geometry(probe)
    return _dyn_hw_ok(
        probe.n, probe.c, probe.d, probe.h, probe.w, probe.k, geom.npq, tile
    )


def job_arch(cu_num: int = 0, gfx: str = "") -> str:
    """Target arch a job would compile for -- shared by dispatch and filtering."""
    return gfx or cu_num_to_arch(cu_num, default=CONV_AOT_ARCH_DEFAULT)


def _probe(rank: int, dtype_is_fp32: bool = False):
    """CPU stand-in for one kernel argument. Innermost extent must be >1."""
    import torch

    return torch.empty(
        (1,) * (rank - 1) + (_PROBE_EXTENT,),
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
    C: int,
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
    out_ndhwc: bool = False,
    dyn_hw: bool = False,
    **kwargs,
):
    del kwargs

    exe = compile_conv3d_implicit(
        _implicit_param_from_problem(
            N,
            C,
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
            groups,
            has_bias,
            splitk,
            (tile_m, tile_n, wave_m, wave_n),
            wgm,
            out_ndhwc,
            "zeros",
            dyn_hw,
        )
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
            f"tile={kwargs['tile_m']}x{kwargs['tile_n']} "
            f"out={'NDHWC' if kwargs.get('out_ndhwc') else 'NCDHW'}"
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
        print(f"[aiter] ARCH={arch}: {len(all_jobs)} jobs match")

    conv_jobs = [j for j in all_jobs if j["kind"] == "conv3d"]
    tr_jobs = [j for j in all_jobs if j["kind"] == "transpose"]

    print("=" * 72)
    print("FlyDSL conv3d AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:              {csv_path}")
    n_dyn = sum(1 for j in conv_jobs if j.get("dyn_hw"))
    print(f"  conv3d jobs:      {len(conv_jobs)}  ({n_dyn} variable-resolution)")
    print(f"  transpose jobs:   {len(tr_jobs)}  (all variable-resolution)")
    print(f"  Total jobs:       {len(all_jobs)}")
    print(f"  Cache dir:        {cache_dir}")
    print(f"  Target arch:      {arch or '(all archs found in CSVs)'}")
    print(f"  AITER_CONV3D_DYN_HW={AITER_CONV3D_DYN_HW}  (must match at runtime)")
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
