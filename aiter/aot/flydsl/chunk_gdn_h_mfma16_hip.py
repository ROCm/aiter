#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL K5 mfma16_hip fork.

``chunk_gdn_h.py`` covers the baseline K5 kernel (``chunk_gdn_fwd_h_flydsl_vk``).
The kernel actually dispatched in production -- ``use_chunk_flydsl=True`` in
``chunk_gated_delta_rule_fwd_opt_vk`` -- is the mfma16 / HIP-aligned fork
``chunk_gdn_fwd_h_flydsl_mfma16_hip``, which is a different compiled product and
gets no coverage from that module. Without this one the first prefill of every
process pays ~2.3s of JIT (~0.1-0.3s even with a warm disk cache).

Shapes come from ``aiter/ops/flydsl/chunk_gdn_h_mfma16_hip_tuned.csv``, the same
table the runtime consults for BV, so adding a model is a csv edit. Each unique
``(arch, dtype, K, V, BT, H, Hg)`` in that file becomes a set of compile jobs.

The csv's ``BV`` column is deliberately *not* used to narrow the fan-out: it
records the best tile for the batch shapes that were benchmarked, while any
other batch shape falls through to the ``_hipeq_select_bv`` rule and can land on
a different tile. Pre-compiling only the tuned values would leave those shapes
JIT-ing mid-serving, so every legal BV is built. Same reasoning for
``snapshot_bf16`` / ``state_bf16``: they are caller-selected dtypes
(``snapshot_dtype`` / ``state_dtype``), so both are built. The remaining
switches are pinned by the production dispatch (see ``_FIXED_SWITCHES``).

Usage:
    # Compile every configuration from the default tuned table
    python -m aiter.aot.flydsl.chunk_gdn_h_mfma16_hip

    # Custom table(s)
    python -m aiter.aot.flydsl.chunk_gdn_h_mfma16_hip --csv /path/to/tuned.csv

    # Cross-compile for another arch (host need not own that GPU)
    python -m aiter.aot.flydsl.chunk_gdn_h_mfma16_hip --target-arch gfx942

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH / GPU_ARCHS          Target arch(es) used for csv rows with an empty
                              ``arch`` column; falls back to the host GPU.
    FLYDSL_GPU_ARCH           Per-job arch override applied during compile;
                              ``--target-arch`` takes precedence.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    dedupe_jobs,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.ops.flydsl.kernels.chunk_gated_delta_h_mfma16x16x16 import (
    compile_chunk_gated_delta_h_mfma16_hip,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

# Tuned table lives next to the kernel host wrapper that reads it at runtime.
_DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "ops"
    / "flydsl"
    / "chunk_gdn_h_mfma16_hip_tuned.csv"
)
DEFAULT_CSVS = [str(_DEFAULT_CSV)]
# Used only when a csv row leaves ``arch`` empty and neither ARCH/GPU_ARCHS nor
# a local GPU can answer.
MFMA16_HIP_AOT_ARCH_DEFAULT = "gfx950"
# Mirrors the ``@flyc.kernel(name=...)`` decorator in
# ``kernels/chunk_gated_delta_h_mfma16x16x16.py``.
_KERNEL_NAME = "chunk_gdn_fwd_h_flydsl_mfma16_hip"

_TORCH_DTYPE = {"torch.bfloat16": "bfloat16"}
# Mirrors ``_HIPEQ_BV_CANDIDATES`` in linear_attention_prefill_kernels.py; the
# runtime picks one of these per batch, so all legal ones are pre-compiled.
_BV_CANDIDATES = (64, 32, 16)

# Caller-selected dtypes (snapshot_dtype / state_dtype): both specializations
# are built regardless of what the tuned rows happened to measure.
_SNAPSHOT_BF16 = (True, False)
_STATE_BF16 = (False, True)

# Switches the production dispatch pins: chunk.py always passes head-major
# g_cumsum with use_exp2 pre-scaling, never gk, always saves v_new, and rejects
# indexed state pools on the FlyDSL path.
_FIXED_SWITCHES: dict[str, bool] = {
    "use_g": True,
    "use_gk": False,
    "use_h0": True,
    "store_fs": True,
    "save_vn": True,
    "wu_contig": True,
    "g_head_major": True,
    "g_log2_scaled": True,
    "use_state_indices": False,
    "bf16_convert_trunc": True,
}

_ARCH_SEP = re.compile(r"[;,\s]+")


def _parse_bool(s: str) -> bool:
    s = s.strip()
    if s in ("True", "true", "1", "yes"):
        return True
    if s in ("False", "false", "0", "no"):
        return False
    raise ValueError(f"unrecognised bool literal {s!r}")


def _detected_arch() -> str | None:
    try:
        from flydsl.runtime.device import get_rocm_arch

        arch = get_rocm_arch()
    except Exception:  # noqa: BLE001 - no GPU / no ROCm on the build host
        return None
    return arch.split(":")[0] if arch else None


def _resolve_archs(row_arch: str | None) -> list[str]:
    """Arch list for one csv row: explicit column, else ARCH/GPU_ARCHS, else
    the host GPU, else ``MFMA16_HIP_AOT_ARCH_DEFAULT``.

    Unlike the tuned tables of the other AOT kinds, nothing in this csv is
    arch-specific measurement data, so the sensible default is "whatever this
    build targets" rather than a literal baked into the file.
    """
    for raw in (row_arch, os.environ.get("ARCH"), os.environ.get("GPU_ARCHS")):
        if not raw or not raw.strip():
            continue
        archs = [
            tok.split(":")[0].lower() for tok in _ARCH_SEP.split(raw.strip()) if tok
        ]
        archs = [a for a in archs if a.startswith("gfx")]
        if archs:
            return list(dict.fromkeys(archs))
    return [_detected_arch() or MFMA16_HIP_AOT_ARCH_DEFAULT]


def parse_csv(csv_path: str) -> list[dict[str, Any]]:
    """Expand the tuned table into unique compile jobs.

    Rows collapse to their distinct ``(arch, dtype, K, V, BT, H, Hg,
    is_varlen)`` shapes -- the tuned table has one row per benchmarked batch
    shape, and ``T_flat``/``N``/``total_chunks`` only steer the host launch
    grid, not the compiled artifact. Each shape then fans out over every legal
    BV and both dtype specializations.

    The fan-out lives here rather than in ``main`` so the wheel build, which
    reaches this module through ``run_aot`` -> ``parse_csv`` only, gets the
    same coverage as a standalone run.
    """
    shapes: dict[tuple, None] = {}

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
                is_varlen = _parse_bool(row.get("is_varlen") or "True")
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue

            for arch in _resolve_archs(row.get("arch")):
                shapes[(arch, dtype, K, V, BT, H, Hg, is_varlen)] = None

    jobs: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for arch, dtype, K, V, BT, H, Hg, is_varlen in shapes:
        bvs = [bv for bv in _BV_CANDIDATES if bv <= V and V % bv == 0]
        if not bvs:
            print(f"  [WARN] no legal BV for V={V}, skipping shape in {csv_path}")
            continue
        for bv, snapshot_bf16, state_bf16 in itertools.product(
            bvs, _SNAPSHOT_BF16, _STATE_BF16
        ):
            job = {
                "kernel_name": _KERNEL_NAME,
                "dtype": dtype,
                "arch": arch,
                "K": K,
                "V": V,
                "BT": BT,
                "BV": bv,
                "H": H,
                "Hg": Hg,
                "is_varlen": is_varlen,
                "snapshot_bf16": snapshot_bf16,
                "state_bf16": state_bf16,
                **_FIXED_SWITCHES,
            }
            key = job_identity(job)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)

    return jobs


def _torch_dtype_for_kernel(dtype_str: str):
    import torch

    name = _TORCH_DTYPE.get(dtype_str)
    if name is None:
        raise ValueError(
            f"Unsupported torch dtype name for chunk_gdn_h mfma16_hip AOT: "
            f"{dtype_str!r}"
        )
    return getattr(torch, name)


def _compile_mfma16_hip_to_cache(
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

    import flydsl.expr as fx
    import torch

    dev = torch.device("cpu")
    torch_dtype = _torch_dtype_for_kernel(dtype)
    state_dtype = torch.bfloat16 if state_bf16 else torch.float32
    snapshot_dtype = torch.bfloat16 if snapshot_bf16 else torch.float32

    # Smallest shape consistent with BT divisibility: only dtype and rank of
    # each slot reach the compiled artifact, T_flat/N shape the host grid.
    N = B = NT = 1
    T = T_flat = BT

    # Disabled slots take the same 1-element placeholders the wrapper passes,
    # so the AOT product keys on the same argument signature as the runtime.
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

    launch_fn = compile_chunk_gated_delta_h_mfma16_hip(
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
        # Must track the compile target, not the build host: the runtime keys
        # this off its own arch (``_IS_GFX942``), so a mismatch is a cache miss.
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
        f"chunk_gdn_h_mfma16_hip  "
        f"K={job.get('K')} V={job.get('V')} BT={job.get('BT')} "
        f"BV={job.get('BV')} H={job.get('H')} Hg={job.get('Hg')} "
        f"dtype={job.get('dtype')} "
        f"snapshot_bf16={job.get('snapshot_bf16')} "
        f"state_bf16={job.get('state_bf16')} "
        f"is_varlen={job.get('is_varlen')} use_h0={job.get('use_h0')} "
        f"store_fs={job.get('store_fs')}"
    )


def compile_one_config(*, arch: str, **kwargs) -> dict:
    """Compile one mfma16_hip configuration and save it to cache."""
    aot_arch = arch or MFMA16_HIP_AOT_ARCH_DEFAULT
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
            _compile_mfma16_hip_to_cache(arch=aot_arch, **kwargs)
        result["compile_time"] = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile the FlyDSL K5 mfma16_hip kernels from the "
        "model table",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to model table csv file(s); defaults to "
        "aiter/ops/flydsl/chunk_gdn_h_mfma16_hip_models.csv",
    )
    parser.add_argument(
        "--target-arch",
        type=str,
        default=None,
        help="Override the arch of every job; useful for cross-compiling on a "
        "host whose GPU differs from the deployment target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded job list without compiling anything.",
    )
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: csv file not found: {csv_path}")
            sys.exit(1)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )

    all_jobs = collect_aot_jobs(csv_paths, parse_csv)
    if args.target_arch and all_jobs:
        all_jobs = dedupe_jobs([dict(j, arch=args.target_arch) for j in all_jobs])

    print("=" * 72)
    print("FlyDSL chunk-gated-delta-h (mfma16_hip) AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  csv:              {csv_path}")
    print(f"  Total jobs:       {len(all_jobs)}")
    archs = sorted({j["arch"] for j in all_jobs})
    print(f"  Compile arch:     {', '.join(archs) if archs else '-'}")
    print(f"  Cache dir:        {cache_dir}")
    print("=" * 72)

    if args.dry_run:
        for job in all_jobs:
            print(f"  {job['arch']}  {_format_shape_str(job)}")
        sys.exit(0)

    total_t0 = time.time()
    print(f"\n--- Compiling {len(all_jobs)} kernels ---")
    results = run_jobs_parallel(compile_one_config, all_jobs)
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
