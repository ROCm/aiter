# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL decode Top-K-per-row kernel.

The launcher is JIT-specialized on two axes, both baked into the kernel: batch size
(via short_max) and context length (via blocks_per_row, derived from the
logits width). Both are bucketed to powers of two in the runtime config, so the
distinct-kernel count collapses to ~36 per arch across the whole (num_rows, context)
range. This module enumerates those and compiles them into the FlyDSL cache so a
serving worker never pays the first-call JIT compile at runtime for these shapes.

Usage:
    # Compile all distinct Top-K kernels from the default CSV
    python -m aiter.aot.flydsl.topk

    # Custom CSV file(s)
    python -m aiter.aot.flydsl.topk --csv /path/to/config.csv

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache).
    ARCH / GPU_ARCHS          Target GPU architecture(s) (e.g. gfx942, gfx950).
"""

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
    cu_num_to_arch,
    override_env,
)

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
)
DEFAULT_CSVS = [os.path.join(_CONFIG_DIR, "topk_decode_aot.csv")]
TOPK_AOT_ARCH_DEFAULT = "gfx942"


def parse_csv(csv_path: str) -> list[dict[str, Any]]:
    """One job per *distinct* compiled Top-K kernel for each CSV row.

    Each row is (k, max_model_len, max_num_seqs, cu_num). The kernel is specialized
    on two drift axes: num_rows (via short_max + caps) and blocks_per_row (via the
    context length). We sweep both -- num_rows 1..max_num_seqs and every distinct
    blocks_per_row 2..min(32, ceil(max_model_len/items_per_block)) -- and dedupe by
    the runtime _kernel_config (evaluated for the row's target arch), so exactly
    one job per distinct compiled kernel is emitted. A representative context length
    per blocks_per_row suffices because the kernel is context-length-agnostic within
    a blocks_per_row (contexts sharing a bpr reuse one kernel)."""
    from aiter.ops.flydsl.topk_per_row_decode import (
        _TIERED_BLOCK_THREADS,
        _TIERED_LOAD_VEC,
        _kernel_config,
    )

    items_per_block = _TIERED_LOAD_VEC * _TIERED_BLOCK_THREADS

    jobs: list[dict[str, Any]] = []
    seen: set = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            k = int(row["k"])
            mml = int(row["max_model_len"])
            max_seqs = int(row.get("max_num_seqs") or 128)
            cu = int(row.get("cu_num") or 0)
            arch = cu_num_to_arch(cu, default=TOPK_AOT_ARCH_DEFAULT)
            # blocks_per_row = min(32, max(2, ceil(L / items_per_block))); sweep every
            # distinct value up to the row's max context via a representative L.
            max_bpr = min(32, max(2, -(-mml // items_per_block)))
            # Derive configs for the target arch so the dedup + the compile below
            # agree, and match a runtime on that arch.
            with override_env("FLYDSL_GPU_ARCH", arch):
                for bpr in range(2, max_bpr + 1):
                    rep_l = min(bpr * items_per_block, mml)
                    for num_rows in range(1, max_seqs + 1):
                        cfg = _kernel_config(num_rows, rep_l)
                        ident = (k, arch, tuple(sorted(cfg.items())))
                        if ident in seen:
                            continue
                        seen.add(ident)
                        jobs.append(
                            {
                                "kernel_name": (
                                    f"topk_per_row_decode_k{k}_nr{num_rows}"
                                    f"_bpr{cfg['blocks_per_row']}_{arch}"
                                ),
                                "k": k,
                                "num_rows": num_rows,
                                "max_model_len": rep_l,
                                "cu_num": cu,
                            }
                        )
    return jobs


def compile_one_config(
    kernel_name: str,
    k: int,
    num_rows: int,
    max_model_len: int,
    cu_num: int = 0,
    **kwargs,
) -> dict:
    """Compile one Top-K kernel configuration into the FlyDSL cache."""
    del kwargs

    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.ops.flydsl.topk_per_row_decode import (
        _build_launcher,
        _required_seq_rows,
    )

    aot_arch = cu_num_to_arch(cu_num, default=TOPK_AOT_ARCH_DEFAULT)
    result = {
        "kernel_name": kernel_name,
        "shape": kernel_name,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    t0 = time.time()
    try:
        with (
            override_env("ARCH", aot_arch),
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            FakeTensorMode(),
        ):
            launcher, ws_slots, _ = _build_launcher(k, num_rows, max_model_len)

            dev = torch.device("cpu")
            next_n = 1
            logits = torch.empty(
                (num_rows, max_model_len), dtype=torch.float32, device=dev
            )
            seq = torch.empty(
                (max(_required_seq_rows(next_n, num_rows), 1),),
                dtype=torch.int32,
                device=dev,
            )
            idx = torch.empty((num_rows, k), dtype=torch.int32, device=dev)
            ws = torch.empty((max(ws_slots, 1),), dtype=torch.int32, device=dev)
            stream = fx.Stream(0)

            # Call the launcher directly (like the GEMM/MoE AOT), NOT via
            # _run_compiled/flyc.compile. Under COMPILE_ONLY the JitFunction.__call__
            # writes the disk cache under the runtime cache key and returns before
            # engine init / launch; flyc.compile instead builds a runnable
            # CompiledFunction, which loads the code object even under COMPILE_ONLY --
            # loading a cross-arch object faults the GPU. Same cache key either way.
            # Fake tensors give create_buffer_resource its descriptors; COMPILE_ONLY
            # skips the launch, so no tensor data is read.
            with compile_only_env():
                launcher(
                    logits,
                    next_n,
                    seq,
                    idx,
                    ws,
                    num_rows,
                    int(logits.stride(0)),
                    int(logits.stride(1)),
                    stream,
                )

        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  {kernel_name}  arch={aot_arch}")
    except Exception as e:
        print(f"  [FAIL] compile  {kernel_name}  arch={aot_arch}: {e}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL decode Top-K kernels from a config CSV",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--csv", type=str, nargs="+", default=DEFAULT_CSVS)
    args = ap.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for p in csv_paths:
        if not os.path.isfile(p):
            print(f"Error: CSV not found: {p}")
            sys.exit(1)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS") or "(from cu_num)"
    jobs = collect_aot_jobs(csv_paths, parse_csv)

    print("=" * 72)
    print("FlyDSL Top-K AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Total jobs:   {len(jobs)}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Target arch:  {arch}")
    print("=" * 72)

    total_t0 = time.time()
    results = []
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] ", end="")
        results.append(compile_one_config(**job))

    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = sum(1 for r in results if r["compile_time"] is None)

    print("\n" + "=" * 72)
    print(f"  Total time:   {time.time() - total_t0:.1f}s")
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
