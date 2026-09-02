#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compile FlyDSL chunk_gdn_h opt (K5) kernels from tuned CSV configs.

Resolves its CSV through ``AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE`` -- the
same merged tuned table the runtime BV lookup reads, so the shapes AOT covers
track whatever has been tuned (including per-model tables under
``model_configs/``). ``BV`` is the exception: all candidates are compiled for
every shape, so sequence lengths the table never measured stay off the JIT
path.

Usage:
    python -m aiter.aot.flydsl.chunk_gdn_h
    python -m aiter.aot.flydsl.chunk_gdn_h --csv /path/to/tuned.csv

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH                      Target GPU architecture (e.g. gfx942, gfx950).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from typing import Any

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    cu_num_to_arch,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.kernels.gdr_prefill import compile_chunk_gated_delta_h
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

CHUNK_GDN_H_AOT_ARCH_DEFAULT = "gfx950"
_KERNEL_NAME = "chunk_gdn_fwd_h_flydsl_opt"

_TORCH_DTYPE = {
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
}

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE]
# Runtime picks BV per batch shape from total_chunks/max_seq_chunks, so a seqlen
# the tuned table never measured can resolve to any of these.
_BV_CANDIDATES = (16, 32, 64)
_G_HEAD_MAJOR = (True, False)
_USE_STATE_INDICES = (False, True)
_FIXED_SWITCHES: dict[str, bool] = {
    "use_g": True,
    "use_gk": False,
    "save_vn": True,
    "wu_contig": True,
    "g_log2_scaled": True,
    "bf16_convert_trunc": True,
}


def _parse_bool(s: str) -> bool:
    """CSV-friendly bool parser. Tolerates ``"True"``/``"False"`` (Python
    ``str(bool)`` style, used by gdr_decode_tuned.csv) plus the more
    permissive ``"1"/"0"``, ``"yes"/"no"`` for handwritten csvs."""
    s = s.strip()
    if s in ("True", "true", "1", "yes"):
        return True
    if s in ("False", "false", "0", "no"):
        return False
    raise ValueError(f"unrecognised bool literal {s!r}")


def _torch_dtype_for_kernel(dtype_str: str):
    import torch

    name = _TORCH_DTYPE.get(dtype_str)
    if name is None:
        raise ValueError(
            f"Unsupported torch dtype name for chunk_gdn_h AOT: {dtype_str!r}"
        )
    return getattr(torch, name)


def parse_csv(csv_path: str) -> list[dict[str, Any]]:
    """Expand opt tuned rows into compile jobs.

    Shapes and layout switches come straight from the row -- including the
    ``cu_num`` it was measured on. ``BV`` does not: the runtime resolves it per
    batch shape (tuned lookup, else the CU heuristic), so a sequence length the
    table never measured can resolve to any candidate. Emitting all of them
    keeps arbitrary ``T``/seqlens off the JIT path, and costs only compile time
    because a row's own tuned ``BV`` is always among them. ``g_head_major`` and
    ``use_state_indices`` are fanned out for a related reason: they are not
    tuned dimensions, but they do fork the compiled artifact.
    """
    jobs: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in rows:
            dtype = (row.get("dtype") or "torch.bfloat16").strip()
            if dtype not in _TORCH_DTYPE:
                print(f"  [WARN] Unsupported dtype {dtype!r}, skipping")
                continue

            try:
                K = int(row["K"])
                V = int(row["V"])
                BT = int(row.get("BT") or 64)
                H = int(row["H"])
                Hg = int(row["Hg"])
                cu_num = int(row.get("cu_num") or 0)
                is_varlen = _parse_bool(row.get("is_varlen") or "True")
                use_h0 = _parse_bool(row.get("use_h0") or "True")
                store_fs = _parse_bool(row.get("store_fs") or "True")
                snapshot_bf16 = _parse_bool(row.get("snapshot_bf16") or "True")
                state_bf16 = _parse_bool(row.get("state_bf16") or "False")
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue

            bvs = tuple(bv for bv in _BV_CANDIDATES if bv <= V and V % bv == 0)
            if not bvs:
                print(
                    f"  [WARN] no BV in {_BV_CANDIDATES} divides V={V}, "
                    f"skipping row in {csv_path}"
                )
                continue

            indices = _USE_STATE_INDICES if (use_h0 and store_fs) else (False,)
            for BV, g_head_major, use_state_indices in itertools.product(
                bvs, _G_HEAD_MAJOR, indices
            ):
                job = {
                    "kernel_name": _KERNEL_NAME,
                    "dtype": dtype,
                    "cu_num": cu_num,
                    "K": K,
                    "V": V,
                    "BT": BT,
                    "BV": BV,
                    "H": H,
                    "Hg": Hg,
                    "is_varlen": is_varlen,
                    "use_h0": use_h0,
                    "store_fs": store_fs,
                    "snapshot_bf16": snapshot_bf16,
                    "state_bf16": state_bf16,
                    "g_head_major": g_head_major,
                    "use_state_indices": use_state_indices,
                    **_FIXED_SWITCHES,
                }
                key = job_identity(job)
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(job)

    return jobs


def _compile_to_cache(
    *,
    dtype: str,
    arch: str,
    K: int,
    V: int,
    BT: int,
    BV: int,
    H: int,
    Hg: int,
    use_g: bool,
    use_gk: bool,
    use_h0: bool,
    store_fs: bool,
    save_vn: bool,
    is_varlen: bool,
    wu_contig: bool,
    state_bf16: bool,
    snapshot_bf16: bool,
    g_log2_scaled: bool,
    g_head_major: bool,
    use_state_indices: bool,
    bf16_convert_trunc: bool,
    **kwargs,
):
    del kwargs

    import torch

    dev = torch.device("cpu")
    torch_dtype = _torch_dtype_for_kernel(dtype)
    state_dtype = torch.bfloat16 if state_bf16 else torch.float32
    snapshot_dtype = torch.bfloat16 if snapshot_bf16 else torch.float32

    N = B = NT = 1
    T = T_flat = BT

    dummy = torch.empty(1, device=dev, dtype=torch.float32)
    int32_dummy = torch.empty(1, device=dev, dtype=torch.int32)

    k = torch.empty((B, T, Hg, K), device=dev, dtype=torch_dtype)
    u = torch.empty((B, H, T_flat, V), device=dev, dtype=torch_dtype)
    w = torch.empty((B, H, T_flat, K), device=dev, dtype=torch_dtype)
    v_new = torch.empty((B, H, T_flat, V), device=dev, dtype=torch_dtype)
    g_shape = (B, H, T_flat) if g_head_major else (B, T_flat, H)
    g = torch.empty(g_shape, device=dev, dtype=torch.float32) if use_g else dummy
    gk = (
        torch.empty((B, T_flat, H, K), device=dev, dtype=torch.float32)
        if use_gk
        else dummy
    )
    h = torch.empty((B, NT, H, V, K), device=dev, dtype=snapshot_dtype)
    h0 = torch.empty((N, H, V, K), device=dev, dtype=state_dtype) if use_h0 else dummy
    ht = torch.empty((N, H, V, K), device=dev, dtype=state_dtype) if store_fs else dummy
    cu_seqlens = (
        torch.zeros((N + 1,), device=dev, dtype=torch.int32)
        if is_varlen
        else int32_dummy
    )
    chunk_offsets = (
        torch.zeros((N + 1,), device=dev, dtype=torch.int32)
        if is_varlen
        else int32_dummy
    )
    state_indices = (
        torch.zeros((N,), device=dev, dtype=torch.int32)
        if use_state_indices
        else int32_dummy
    )

    launch_fn = compile_chunk_gated_delta_h(
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        H=H,
        Hg=Hg,
        USE_G=use_g,
        USE_GK=use_gk,
        USE_INITIAL_STATE=use_h0,
        STORE_FINAL_STATE=store_fs,
        SAVE_NEW_VALUE=save_vn,
        IS_VARLEN=is_varlen,
        WU_CONTIGUOUS=wu_contig,
        STATE_DTYPE_BF16=state_bf16,
        SNAPSHOT_DTYPE_BF16=snapshot_bf16,
        G_IS_LOG2_SCALED=g_log2_scaled,
        USE_STATE_INDICES=use_state_indices,
        SCHED_GFX942=arch.startswith("gfx942"),
        G_HEAD_MAJOR=g_head_major,
        BF16_CONVERT_TRUNC=bf16_convert_trunc,
    )

    with compile_only_env():
        _run_compiled(
            launch_fn,
            k,
            u,
            w,
            v_new,
            g,
            gk,
            h,
            h0,
            ht,
            cu_seqlens,
            chunk_offsets,
            state_indices,
            T,
            T_flat,
            N,
            (V + BV - 1) // BV,  # grid_v
            N * H,  # grid_nh
            fx.Stream(0),
        )


def _format_shape_str(job: dict) -> str:
    return (
        f"chunk_gdn_h_opt  "
        f"K={job.get('K')} V={job.get('V')} BT={job.get('BT')} "
        f"BV={job.get('BV')} H={job.get('H')} Hg={job.get('Hg')} "
        f"dtype={job.get('dtype')} "
        f"snapshot_bf16={job.get('snapshot_bf16')} "
        f"state_bf16={job.get('state_bf16')} "
        f"is_varlen={job.get('is_varlen')} use_h0={job.get('use_h0')} "
        f"store_fs={job.get('store_fs')}"
    )


def compile_one_config(*, cu_num: int = 0, **kwargs) -> dict:
    """Compile one opt configuration and save it to cache."""
    aot_arch = cu_num_to_arch(cu_num, default=CHUNK_GDN_H_AOT_ARCH_DEFAULT)
    kwargs.pop("kernel_name", None)
    shape_str = _format_shape_str(kwargs)
    result = {
        "kernel_name": _KERNEL_NAME,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    from torch._subclasses.fake_tensor import FakeTensorMode

    t0 = time.time()
    try:
        with (
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            FakeTensorMode(),
        ):
            _compile_to_cache(arch=aot_arch, **kwargs)
        result["compile_time"] = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL chunk-gdn-h opt (K5) kernels "
        "from aiter CSV config",
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
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS") or "(auto-detect)"

    jobs = collect_aot_jobs(csv_paths, parse_csv)

    print("=" * 72)
    print("FlyDSL chunk-gated-delta-h opt (K5) AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Total jobs:   {len(jobs)}")
    print("  Compile arch: (from cu_num)")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Target arch:  {arch}")
    print("=" * 72)

    total_t0 = time.time()
    print(f"\n--- Compiling {len(jobs)} kernels ---")
    results = run_jobs_parallel(compile_one_config, jobs)
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

    exit_code = 0
    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        exit_code = 1
    else:
        print("All compilations succeeded. Cache is ready.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
