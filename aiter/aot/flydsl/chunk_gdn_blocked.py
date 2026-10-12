#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Precompile blocked GDN prefill kernels from serial K5 configurations."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.kernels.gdr_prefill import compile_chunk_gated_delta_h
from aiter.ops.flydsl.kernels.gdr_prefill.chunk_gdn_carry_gfx950 import (
    compile_chunk_gdn_carry,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

CHUNK_GDN_BLOCKED_AOT_ARCH_DEFAULT = "gfx950"
_KERNEL_NAME = "chunk_gdn_blocked"

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE]


def parse_csv(csv_path: str) -> list[dict[str, Any]]:
    """Expand eligible K5 rows; deduplicate carry across Hg."""
    jobs: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in rows:
            try:
                K = int(row["K"])
                V = int(row["V"])
                H = int(row["H"])
                Hg = int(row["Hg"])
                cu_num = int(row.get("cu_num") or 0)
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue

            dtype = (row.get("dtype") or "torch.bfloat16").strip()
            if dtype != "torch.bfloat16" or K != 128 or V != 128:
                continue

            candidates = (
                {"phase": "build_map", "H": H, "Hg": Hg, "cu_num": cu_num},
                {
                    "phase": "emit",
                    "H": H,
                    "Hg": Hg,
                    "state_bf16": False,
                    "cu_num": cu_num,
                },
                {
                    "phase": "emit",
                    "H": H,
                    "Hg": Hg,
                    "state_bf16": True,
                    "cu_num": cu_num,
                },
                {
                    "phase": "carry",
                    "H": H,
                    "use_initial_state": True,
                    "state_bf16": False,
                    "cu_num": cu_num,
                },
                {
                    "phase": "carry",
                    "H": H,
                    "use_initial_state": True,
                    "state_bf16": True,
                    "cu_num": cu_num,
                },
                {
                    "phase": "carry",
                    "H": H,
                    "use_initial_state": False,
                    "cu_num": cu_num,
                },
            )
            for job in candidates:
                key = job_identity(job)
                if key not in seen:
                    seen.add(key)
                    jobs.append(job)

    return jobs


def _compile_build_map_to_cache(*, arch: str, H: int, Hg: int, **kwargs) -> None:
    del kwargs

    import torch

    dev = torch.device("cpu")
    B = blocks = n_prefill = 1
    T = 64
    K = 128
    packed_v = K + 128
    bv = 64
    dummy = torch.empty(1, device=dev, dtype=torch.float32)
    int32_dummy = torch.empty(1, device=dev, dtype=torch.int32)

    k = torch.empty((B, T, Hg, K), device=dev, dtype=torch.bfloat16)
    packed_u = torch.empty((B, H, T, packed_v), device=dev, dtype=torch.bfloat16)
    w = torch.empty((B, H, T, K), device=dev, dtype=torch.bfloat16)
    g = torch.empty((B, H, T), device=dev, dtype=torch.float32)
    maps = torch.empty((blocks, H, packed_v, K), device=dev, dtype=torch.float32)
    kernel_cu_seqlens = torch.empty((n_prefill + 1,), device=dev, dtype=torch.int32)
    chunk_offsets = torch.empty((n_prefill + 1,), device=dev, dtype=torch.int32)
    block_seq_id = torch.empty((blocks,), device=dev, dtype=torch.int32)
    block_chunk_base = torch.empty((blocks,), device=dev, dtype=torch.int32)
    block_nchunks = torch.empty((blocks,), device=dev, dtype=torch.int32)

    launch = compile_chunk_gated_delta_h(
        K=128,
        V=256,
        BT=64,
        BV=64,
        H=H,
        Hg=Hg,
        USE_G=True,
        USE_GK=False,
        USE_INITIAL_STATE=False,
        STORE_FINAL_STATE=True,
        SAVE_NEW_VALUE=False,
        IS_VARLEN=True,
        WU_CONTIGUOUS=True,
        STATE_DTYPE_BF16=False,
        SNAPSHOT_DTYPE_BF16=True,
        G_IS_LOG2_SCALED=True,
        USE_STATE_INDICES=False,
        SCHED_GFX942=arch.startswith("gfx942"),
        G_HEAD_MAJOR=True,
        BF16_CONVERT_TRUNC=True,
        PHASE="build_map",
        BLOCK_ENTRY_SEED=False,
    )
    with compile_only_env():
        _run_compiled(
            launch,
            k,
            packed_u,
            w,
            dummy,
            g,
            dummy,
            dummy,
            dummy,
            maps,
            kernel_cu_seqlens,
            chunk_offsets,
            int32_dummy,
            block_seq_id,
            block_chunk_base,
            block_nchunks,
            blocks,
            T,
            T,
            n_prefill,
            packed_v // bv,
            blocks * H,
            fx.Stream(0),
        )


def _compile_emit_to_cache(
    *, arch: str, H: int, Hg: int, state_bf16: bool, **kwargs
) -> None:
    del kwargs

    import torch

    dev = torch.device("cpu")
    B = blocks = n_prefill = 1
    T = 64
    K = V = 128
    bv = 64
    dummy = torch.empty(1, device=dev, dtype=torch.float32)
    int32_dummy = torch.empty(1, device=dev, dtype=torch.int32)

    k = torch.empty((B, T, Hg, K), device=dev, dtype=torch.bfloat16)
    u = torch.empty((B, H, T, V), device=dev, dtype=torch.bfloat16)
    w = torch.empty((B, H, T, K), device=dev, dtype=torch.bfloat16)
    v_new = torch.empty((B, H, T, V), device=dev, dtype=torch.bfloat16)
    g = torch.empty((B, H, T), device=dev, dtype=torch.float32)
    h = torch.empty((B, 1, H, V, K), device=dev, dtype=torch.bfloat16)
    entry = torch.empty((blocks, H, K, V), device=dev, dtype=torch.float32)
    final_state = torch.empty(
        (n_prefill, H, V, K),
        device=dev,
        dtype=torch.bfloat16 if state_bf16 else torch.float32,
    )
    kernel_cu_seqlens = torch.empty((n_prefill + 1,), device=dev, dtype=torch.int32)
    chunk_offsets = torch.empty((n_prefill + 1,), device=dev, dtype=torch.int32)
    block_seq_id = torch.empty((blocks,), device=dev, dtype=torch.int32)
    block_chunk_base = torch.empty((blocks,), device=dev, dtype=torch.int32)
    block_nchunks = torch.empty((blocks,), device=dev, dtype=torch.int32)

    launch = compile_chunk_gated_delta_h(
        K=128,
        V=128,
        BT=64,
        BV=64,
        H=H,
        Hg=Hg,
        USE_G=True,
        USE_GK=False,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        SAVE_NEW_VALUE=True,
        IS_VARLEN=True,
        WU_CONTIGUOUS=True,
        STATE_DTYPE_BF16=state_bf16,
        SNAPSHOT_DTYPE_BF16=True,
        G_IS_LOG2_SCALED=True,
        USE_STATE_INDICES=False,
        SCHED_GFX942=arch.startswith("gfx942"),
        G_HEAD_MAJOR=True,
        BF16_CONVERT_TRUNC=True,
        PHASE="emit",
        BLOCK_ENTRY_SEED=True,
    )
    with compile_only_env():
        _run_compiled(
            launch,
            k,
            u,
            w,
            v_new,
            g,
            dummy,
            h,
            entry,
            final_state,
            kernel_cu_seqlens,
            chunk_offsets,
            int32_dummy,
            block_seq_id,
            block_chunk_base,
            block_nchunks,
            blocks,
            T,
            T,
            n_prefill,
            V // bv,
            blocks * H,
            fx.Stream(0),
        )


def _compile_carry_to_cache(
    *, arch: str, H: int, use_initial_state: bool, state_bf16: bool = False, **kwargs
) -> None:
    del arch, kwargs

    import torch

    dev = torch.device("cpu")
    blocks = requests = 1
    K = V = 128
    maps = torch.empty((blocks, H, K + V, K), device=dev, dtype=torch.float32)
    h0_or_maps = torch.empty(
        (requests, H, V, K),
        device=dev,
        dtype=torch.bfloat16 if (use_initial_state and state_bf16) else torch.float32,
    )
    entry = torch.empty((blocks, H, K, V), device=dev, dtype=torch.float32)
    block_prefix = torch.empty((requests + 1,), device=dev, dtype=torch.int32)

    launch = compile_chunk_gdn_carry(
        H=H,
        use_initial_state=use_initial_state,
        STATE_DTYPE_BF16=state_bf16,
    )
    with compile_only_env():
        _run_compiled(
            launch,
            maps,
            h0_or_maps,
            entry,
            block_prefix,
            blocks,
            requests,
            fx.Stream(0),
        )


def _format_shape_str(job: dict[str, Any]) -> str:
    phase = job.get("phase")
    h = job.get("H")
    hg = job.get("Hg")
    use_h0 = job.get("use_initial_state")
    state_bf16 = job.get("state_bf16")
    return (
        f"chunk_gdn_blocked phase={phase} H={h} Hg={hg} use_h0={use_h0} "
        f"state_bf16={state_bf16}"
    )


def compile_one_config(*, cu_num: int = 0, phase: str, **kwargs) -> dict[str, Any]:
    del cu_num
    aot_arch = CHUNK_GDN_BLOCKED_AOT_ARCH_DEFAULT
    shape_str = _format_shape_str({"phase": phase, **kwargs})
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
            if phase == "build_map":
                _compile_build_map_to_cache(arch=aot_arch, **kwargs)
            elif phase == "emit":
                _compile_emit_to_cache(arch=aot_arch, **kwargs)
            elif phase == "carry":
                _compile_carry_to_cache(arch=aot_arch, **kwargs)
            else:
                raise ValueError(f"unknown blocked GDN phase: {phase!r}")
        result["compile_time"] = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile blocked GDN prefill kernels from aiter CSV config",
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
    print("FlyDSL blocked GDN prefill AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Total jobs:   {len(jobs)}")
    print(f"  Compile arch: {CHUNK_GDN_BLOCKED_AOT_ARCH_DEFAULT}")
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

    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        sys.exit(1)
    print("All compilations succeeded. Cache is ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
