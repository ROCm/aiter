#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL chunk-gated-delta-h (K5) kernel.

Reads the offline-tuned BV lookup table
``aiter/ops/flydsl/chunk_gdn_h_tuned.jsonl`` (the same file consumed at
runtime by ``_lookup_tuned_bv`` in ``linear_attention_prefill_kernels``),
extracts every unique compile-time configuration, and pre-compiles it
into the FlyDSL disk cache so that the first inference call does not pay
the JIT cost.

Each jsonl entry is compiled twice -- once with ``STATE_DTYPE_BF16=False``
(legacy f32-state runtime path) and once with ``STATE_DTYPE_BF16=True``
(used when the caller passes ``state_dtype=torch.bfloat16``) -- so neither
runtime path pays a JIT cost on first call.

Usage:
    # Compile all unique FlyDSL chunk-gdn-h kernels from the default jsonl
    python -m aiter.aot.flydsl.chunk_gdn_h

    # Custom jsonl file(s)
    python -m aiter.aot.flydsl.chunk_gdn_h --jsonl /path/to/tuned.jsonl

    # Cross-compile every entry for a different GPU arch (host need not
    # be that GPU; FlyDSL emits ISA for the requested target).
    python -m aiter.aot.flydsl.chunk_gdn_h --target-arch gfx942

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH / GPU_ARCHS          Target GPU architecture (logging hint); the
                              actual per-job compile arch comes from the
                              ``arch`` field of each jsonl entry.
    FLYDSL_GPU_ARCH           Per-job arch override applied during compile;
                              ``--target-arch`` takes precedence over both
                              this env var and the ``arch`` field in jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    dedupe_jobs,
    job_identity,
    override_env,
)
from aiter.ops.flydsl.kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h

# Default tuned table lives next to the kernel host wrapper.
_DEFAULT_JSONL = (
    Path(__file__).resolve().parents[2] / "ops" / "flydsl" / "chunk_gdn_h_tuned.jsonl"
)
DEFAULT_JSONLS = [str(_DEFAULT_JSONL)]
# Alias to match the (DEFAULT_CSVS, parse_csv, compile_one_config) contract
# expected by ``aiter.aot.flydsl.common.run_aot_worker``. K5 reads jsonl
# rather than csv, but the worker only cares about the names.
DEFAULT_CSVS = DEFAULT_JSONLS
CHUNK_GDN_H_AOT_ARCH_DEFAULT = "gfx950"
# K5 ships a single kernel today; mirrors the ``@flyc.kernel(name=...)``
# decorator in ``kernels/chunk_gated_delta_h.py`` so the AOT ``result`` dict
# and any failure-mode print share one source of truth.
_KERNEL_NAME = "chunk_gdn_fwd_h_flydsl_vk"

# Map jsonl ``dtype`` string -> torch dtype name used for dummy tensors.
# Only bf16 is exercised by the kernel today (state_t is selected
# separately via ``STATE_DTYPE_BF16``).
_TORCH_DTYPE = {
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
}


def parse_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    """Parse the chunk_gdn_h tuned jsonl and return unique compile jobs.

    Each row in the jsonl already carries every compile-time switch the
    kernel cares about (K/V/BT/H/Hg/use_g/use_gk/use_h0/store_fs/save_vn/
    is_varlen/wu_contig) plus the offline-tuned ``BV``. We only keep the
    fields that actually influence MLIR compilation; ``T_flat``/``N`` and
    ``duration`` are dropped (they affect the host launch grid, not the
    compiled artifact).
    """
    jobs: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] bad jsonl line in {jsonl_path}: {e}")
                continue

            dtype = obj.get("dtype", "torch.bfloat16")
            if dtype not in _TORCH_DTYPE:
                print(f"  [WARN] Unsupported dtype {dtype!r}, skipping")
                continue

            try:
                bv = int(obj["config"]["BV"])
                k = int(obj["K"])
                v = int(obj["V"])
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed entry in {jsonl_path}: {e}")
                continue

            if v % bv != 0 or bv > v:
                print(
                    f"  [WARN] BV={bv} does not divide V={v}, skipping entry "
                    f"in {jsonl_path}"
                )
                continue

            job = {
                "dtype": dtype,
                "arch": obj.get("arch", CHUNK_GDN_H_AOT_ARCH_DEFAULT),
                "K": k,
                "V": v,
                "BT": int(obj.get("BT", 64)),
                "BV": bv,
                "H": int(obj["H"]),
                "Hg": int(obj["Hg"]),
                "use_g": bool(obj.get("use_g", True)),
                "use_gk": bool(obj.get("use_gk", False)),
                "use_h0": bool(obj.get("use_h0", True)),
                "store_fs": bool(obj.get("store_fs", False)),
                "save_vn": bool(obj.get("save_vn", True)),
                "is_varlen": bool(obj.get("is_varlen", False)),
                "wu_contig": bool(obj.get("wu_contig", True)),
                # state dtype is not tracked in the tuned jsonl yet; default
                # f32 here, then main() unconditionally fans out into a bf16
                # twin so both runtime paths are pre-compiled.
                "state_bf16": False,
            }
            key = job_identity(job)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)

    return jobs


# Alias to match the contract expected by
# ``aiter.aot.flydsl.common.run_aot_worker``.
parse_csv = parse_jsonl


def _torch_dtype_for_kernel(dtype_str: str):
    import torch

    name = _TORCH_DTYPE.get(dtype_str)
    if name is None:
        raise ValueError(
            f"Unsupported torch dtype name for chunk_gdn_h AOT: {dtype_str!r}"
        )
    return getattr(torch, name)


def _compile_executable_to_cache(exe, *args) -> None:
    with compile_only_env():
        exe(*args)


def _compile_chunk_gdn_h_to_cache(
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
    **kwargs,
):
    del kwargs

    import torch

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    dev = torch.device("cuda") if has_cuda else torch.device("cpu")
    torch_dtype = _torch_dtype_for_kernel(dtype)
    state_dtype = torch.bfloat16 if state_bf16 else torch.float32

    # Pick a representative T_flat / N for the dummy tensors. These only
    # influence the host launch shape, not the compiled artifact, so any
    # value consistent with BT divisibility works. ``is_varlen`` flips
    # the kernel's cu_seqlens read path at runtime, but the AOT dummy
    # tensor shape is identical in both modes, so we use a single T.
    T_flat = BT
    N = 1
    B = N
    T = T_flat

    k = torch.empty((B, T, Hg, K), device=dev, dtype=torch_dtype)
    v = torch.empty((B, H, T_flat, V), device=dev, dtype=torch_dtype)
    w = torch.empty((B, H, T_flat, K), device=dev, dtype=torch_dtype)
    v_new = torch.empty((B, H, T_flat, V), device=dev, dtype=torch_dtype)
    g = torch.empty((B * T_flat, H), device=dev, dtype=torch.float32)
    gk = torch.empty((B * T_flat, H, K), device=dev, dtype=torch.float32)
    h = torch.empty((B, max(T_flat // BT, 1), H, V, K), device=dev, dtype=torch_dtype)
    h0 = torch.empty((N, H, V, K), device=dev, dtype=state_dtype)
    ht = torch.empty((N, H, V, K), device=dev, dtype=state_dtype)
    # Variable-length book-keeping tensors. FlyDSL JIT does not accept
    # ``None`` for tensor slots, so allocate small int32 buffers even when
    # the kernel branch is disabled.
    cu_seqlens = torch.zeros((N + 1,), device=dev, dtype=torch.int32)
    chunk_offsets = torch.zeros((N + 1,), device=dev, dtype=torch.int32)

    stream = fx.Stream(torch.cuda.current_stream(device=dev) if has_cuda else 0)

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
    )

    grid_v = (V + BV - 1) // BV
    grid_nh = N * H

    _compile_executable_to_cache(
        launch_fn,
        k,
        v,
        w,
        v_new,
        g,
        gk,
        h,
        h0,
        ht,
        cu_seqlens,
        chunk_offsets,
        T,  # T_val
        T_flat,
        N,  # N_val
        grid_v,
        grid_nh,
        stream,
    )


def _format_shape_str(job: dict) -> str:
    """Render a job dict into the one-line summary used by ``[OK]`` /
    ``[FAIL]`` prints."""
    return (
        f"chunk_gdn_h  "
        f"K={job.get('K')} V={job.get('V')} BT={job.get('BT')} "
        f"BV={job.get('BV')} H={job.get('H')} Hg={job.get('Hg')} "
        f"dtype={job.get('dtype')} "
        f"use_g={job.get('use_g')} use_gk={job.get('use_gk')} "
        f"use_h0={job.get('use_h0')} store_fs={job.get('store_fs')} "
        f"save_vn={job.get('save_vn')} is_varlen={job.get('is_varlen')} "
        f"wu_contig={job.get('wu_contig')} state_bf16={job.get('state_bf16')}"
    )


def compile_one_config(
    *,
    dtype: str,
    arch: str,
    K: int,
    V: int,
    BT: int,
    BV: int,
    H: int,
    Hg: int,
    **kwargs,
) -> dict:
    """Compile one chunk-gdn-h configuration and save it to cache."""
    aot_arch = arch or CHUNK_GDN_H_AOT_ARCH_DEFAULT
    shape_str = _format_shape_str(
        {
            "K": K,
            "V": V,
            "BT": BT,
            "BV": BV,
            "H": H,
            "Hg": Hg,
            "dtype": dtype,
            **kwargs,
        }
    )
    result = {
        "kernel_name": _KERNEL_NAME,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    t0 = time.time()
    try:
        with override_env("ARCH", aot_arch), override_env("FLYDSL_GPU_ARCH", aot_arch):
            _compile_chunk_gdn_h_to_cache(
                dtype=dtype,
                arch=aot_arch,
                K=K,
                V=V,
                BT=BT,
                BV=BV,
                H=H,
                Hg=Hg,
                **kwargs,
            )

        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  arch={aot_arch}")
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL chunk-gated-delta-h kernels "
        "from the offline-tuned jsonl table",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        nargs="+",
        default=DEFAULT_JSONLS,
        help="Path(s) to tuned jsonl file(s); defaults to "
        "aiter/ops/flydsl/chunk_gdn_h_tuned.jsonl",
    )
    parser.add_argument(
        "--target-arch",
        type=str,
        default=None,
        help="Override the ``arch`` field of every jsonl entry; useful for "
        "cross-compiling on a host whose GPU differs from the tuned arch "
        "(e.g. ``--target-arch gfx942`` on a gfx950 box).",
    )
    args = parser.parse_args()

    jsonl_paths = [os.path.abspath(p) for p in args.jsonl]
    for jsonl_path in jsonl_paths:
        if not os.path.isfile(jsonl_path):
            print(f"Error: jsonl file not found: {jsonl_path}")
            sys.exit(1)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS") or "(from jsonl)"

    all_jobs = collect_aot_jobs(jsonl_paths, parse_jsonl)

    # ``--target-arch`` rewrites the ``arch`` field of every job, then we
    # dedupe again because two jsonl entries that differed only in arch
    # collapse to the same compile after the override.
    if args.target_arch and all_jobs:
        all_jobs = dedupe_jobs([dict(j, arch=args.target_arch) for j in all_jobs])

    # Fan out into both f32-state and bf16-state variants so neither
    # runtime path pays a JIT cost on first call. ``dedupe_jobs`` drops
    # any pre-existing dup.
    if all_jobs:
        all_jobs = dedupe_jobs(all_jobs + [dict(j, state_bf16=True) for j in all_jobs])

    print("=" * 72)
    print("FlyDSL chunk-gated-delta-h AOT Pre-compilation")
    print("=" * 72)
    for jsonl_path in jsonl_paths:
        print(f"  jsonl:            {jsonl_path}")
    print(f"  Total jobs:       {len(all_jobs)}")
    if args.target_arch:
        print(f"  Compile arch:     {args.target_arch} (overridden by --target-arch)")
    else:
        print("  Compile arch:     (from jsonl 'arch' field)")
    print(f"  Cache dir:        {cache_dir}")
    print(f"  Target arch:      {arch}")
    print("=" * 72)

    total_t0 = time.time()
    results: list[dict] = []

    for i, job in enumerate(all_jobs, 1):
        print(f"\n[{i}/{len(all_jobs)}] ", end="")
        results.append(compile_one_config(**job))

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
