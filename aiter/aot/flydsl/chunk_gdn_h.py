#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL chunk-gated-delta-h (K5) kernel.

Reads the offline-tuned BV lookup table
``aiter/ops/flydsl/chunk_gdn_h_tuned.csv`` (the same file consumed at
runtime by ``_lookup_csv_bv`` / ``_heuristic_bv`` in
``linear_attention_prefill_kernels``), extracts every unique compile-time
configuration, and pre-compiles it into the FlyDSL disk cache so that the
first inference call does not pay the JIT cost.

Each csv row is compiled twice -- once with ``STATE_DTYPE_BF16=False``
(legacy f32-state runtime path) and once with ``STATE_DTYPE_BF16=True``
(used when the caller passes ``state_dtype=torch.bfloat16``) -- so neither
runtime path pays a JIT cost on first call.

Usage:
    # Compile all unique FlyDSL chunk-gdn-h kernels from the default csv
    python -m aiter.aot.flydsl.chunk_gdn_h

    # Custom csv file(s)
    python -m aiter.aot.flydsl.chunk_gdn_h --csv /path/to/tuned.csv

    # Cross-compile every entry for a different GPU arch (host need not
    # be that GPU; FlyDSL emits ISA for the requested target).
    python -m aiter.aot.flydsl.chunk_gdn_h --target-arch gfx942

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH / GPU_ARCHS          Target GPU architecture (logging hint); the
                              actual per-job compile arch comes from the
                              ``arch`` column of each csv row.
    FLYDSL_GPU_ARCH           Per-job arch override applied during compile;
                              ``--target-arch`` takes precedence over both
                              this env var and the ``arch`` column in csv.
"""

from __future__ import annotations

import argparse
import csv
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

# Mirror of ``linear_attention_prefill_kernels._BV_CANDIDATES`` /
# ``_legal_bv_candidates``. We intentionally do NOT import them from that
# module: importing the runtime host wrapper pulls in ``triton`` and the
# Triton ``gated_delta_rule`` package, whose ``__init__`` runs autotune
# config probing that calls into the Triton driver. In an AOT/CI host with
# no GPU visible (``HIP_VISIBLE_DEVICES=``) that probe raises
# ``RuntimeError: 0 active drivers``. The BV candidate set is a tiny,
# stable constant, so we duplicate the pure logic here to keep the AOT
# entrypoint free of any torch/triton import. Keep in sync with the
# runtime definition.
_BV_CANDIDATES = [16, 32, 64]


def _legal_bv_candidates(V: int) -> list[int]:
    return [c for c in _BV_CANDIDATES if c <= V and V % c == 0]

# Default tuned table lives next to the kernel host wrapper.
_DEFAULT_CSV = (
    Path(__file__).resolve().parents[2] / "ops" / "flydsl" / "chunk_gdn_h_tuned.csv"
)
DEFAULT_CSVS = [str(_DEFAULT_CSV)]
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


def parse_csv(csv_path: str) -> list[dict[str, Any]]:
    """Parse the chunk_gdn_h tuned csv and return unique compile jobs.

    Each row carries a *shape family* + compile-time switches
    (K/V/BT/H/Hg/use_g/use_gk/use_h0/store_fs/save_vn/is_varlen/wu_contig).
    ``T_flat``/``N``/``duration`` are dropped (they affect the host launch
    grid, not the compiled artifact).

    BV handling (runtime-aligned, see ``linear_attention_prefill_kernels``):
        Runtime BV is chosen by the rule-based ``_heuristic_bv``, NOT by the
        csv ``BV`` column -- so a single tuned-csv row can resolve to
        *several* different BVs at runtime depending on ``(T_flat, N,
        min_seqlen)`` (e.g. the H32/Hg16 carve-outs emit BV in {16, 32, 64}).
        Pre-compiling only the csv's ``BV`` value would leave the other
        rule-reachable BVs to pay a first-call JIT. To keep AOT coverage a
        superset of every BV the runtime rule can pick, we ignore the csv
        ``BV`` column for selection and instead fan each shape family out
        over **all legal BVs** for its ``V`` (``_legal_bv_candidates`` --
        the same helper the runtime selector uses). This is the chunk-gdn-h
        analogue of how the MoE/GEMM AOT paths cover their full runtime
        candidate space.

    is_varlen / store_fs handling:
        ``IS_VARLEN`` (cu_seqlens read path) and ``STORE_FINAL_STATE``
        (final-state write-back) are compile-time flags; each True / False
        choice is a distinct compiled artifact. A tuned-csv row only pins
        one combination of them, but at runtime a given shape family can be
        called either batched or varlen, with or without final-state output
        (e.g. the H32/Hg16 carve-out shapes are exercised varlen +
        output_final_state=True, while the csv rows for that family are
        batched + store_fs=False). To make AOT coverage independent of those
        runtime choices (and decoupled from the runtime rule's family list),
        we pre-compile BOTH ``is_varlen`` AND BOTH ``store_fs`` variants for
        EVERY shape family.

    g_log2_scaled handling:
        ``G_IS_LOG2_SCALED`` is also a compile-time flag (see
        ``_compile_chunk_gdn_h_to_cache``). The runtime wrapper defaults to
        ``use_exp2=True`` -> ``g_log2_scaled=True``, which is the path the
        end-to-end prefill and the tests take, so AOT pins it to ``True``.
        The (rarer) ``use_exp2=False`` path is left to a first-call JIT.
    """
    jobs: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dtype = row.get("dtype", "torch.bfloat16")
            if dtype not in _TORCH_DTYPE:
                print(f"  [WARN] Unsupported dtype {dtype!r}, skipping")
                continue

            try:
                k = int(row["K"])
                v = int(row["V"])
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue

            # Fan out over every BV the runtime rule could select for this V,
            # rather than trusting the (now advisory) csv ``BV`` column.
            legal_bvs = _legal_bv_candidates(v)
            if not legal_bvs:
                print(
                    f"  [WARN] no legal BV candidate for V={v}, skipping row "
                    f"in {csv_path}"
                )
                continue

            try:
                base_job = {
                    "dtype": dtype,
                    "arch": row.get("arch") or CHUNK_GDN_H_AOT_ARCH_DEFAULT,
                    "K": k,
                    "V": v,
                    "BT": int(row.get("BT") or 64),
                    "H": int(row["H"]),
                    "Hg": int(row["Hg"]),
                    "use_g": _parse_bool(row.get("use_g") or "True"),
                    "use_gk": _parse_bool(row.get("use_gk") or "False"),
                    "use_h0": _parse_bool(row.get("use_h0") or "True"),
                    "save_vn": _parse_bool(row.get("save_vn") or "True"),
                    # ``is_varlen`` and ``store_fs`` are set per-variant in
                    # the fan-out loop below (both False and True are
                    # pre-compiled for each), so their csv columns are not
                    # read here.
                    "wu_contig": _parse_bool(row.get("wu_contig") or "True"),
                    # state dtype is not tracked in the tuned csv yet; default
                    # f32 here, then main() unconditionally fans out into a
                    # bf16 twin so both runtime paths are pre-compiled.
                    "state_bf16": False,
                }
            except (KeyError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue

            # Cover both compiled IS_VARLEN and STORE_FINAL_STATE paths for
            # every family, so AOT coverage is independent of whether the
            # caller runs batched/varlen and whether it requests the final
            # state -- both are compile-time flags that produce distinct
            # artifacts, and a single tuned-csv row only pins one combination.
            for is_varlen in (False, True):
                for store_fs in (False, True):
                    for bv in legal_bvs:
                        job = dict(
                            base_job,
                            BV=bv,
                            is_varlen=is_varlen,
                            store_fs=store_fs,
                        )
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
            f"Unsupported torch dtype name for chunk_gdn_h AOT: {dtype_str!r}"
        )
    return getattr(torch, name)


def _compile_executable_to_cache(exe, *args) -> None:
    with compile_only_env():
        exe(*args)


def _compile_chunk_gdn_h_to_cache(
    *,
    dtype: str,
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

    dev = torch.device("cpu")
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
    # ``g`` is head-major ``[B, H, T_flat]`` (offset = i_h * T_flat + (bos+row),
    # stride=1) to match the K5 kernel's flat addressing in
    # ``kernels/chunk_gated_delta_h.py`` (and Triton VK / HIP K5). ``gk`` stays
    # token-major ``[B*T_flat, H, K]`` per its own per-K decay addressing
    # (offset = (bos+row)*H*K + i_h*K + k).
    g = torch.empty((B, H, T_flat), device=dev, dtype=torch.float32)
    gk = torch.empty((B * T_flat, H, K), device=dev, dtype=torch.float32)
    h = torch.empty((B, max(T_flat // BT, 1), H, V, K), device=dev, dtype=torch_dtype)
    h0 = torch.empty((N, H, V, K), device=dev, dtype=state_dtype)
    ht = torch.empty((N, H, V, K), device=dev, dtype=state_dtype)
    # Variable-length book-keeping tensors. FlyDSL JIT does not accept
    # ``None`` for tensor slots, so allocate small int32 buffers even when
    # the kernel branch is disabled.
    cu_seqlens = torch.zeros((N + 1,), device=dev, dtype=torch.int32)
    chunk_offsets = torch.zeros((N + 1,), device=dev, dtype=torch.int32)

    stream = fx.Stream(0)

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
        # Match the runtime default path: the K5 wrapper defaults to
        # ``use_exp2=True`` (and the end-to-end prefill passes the Triton K1
        # ``use_exp2`` through, which is True by default), so the runtime
        # compile key carries ``G_IS_LOG2_SCALED=True``. Pre-compiling with
        # the kernel default (False) would never hit that key -- the whole
        # AOT cache would miss on the default path. We pin True here to cover
        # it; the (rarer) ``use_exp2=False`` path still JITs on first call.
        G_IS_LOG2_SCALED=True,
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

    from torch._subclasses.fake_tensor import FakeTensorMode

    t0 = time.time()
    try:
        with override_env("ARCH", aot_arch), override_env(
            "FLYDSL_GPU_ARCH", aot_arch
        ), FakeTensorMode():
            _compile_chunk_gdn_h_to_cache(
                dtype=dtype,
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
        "from the offline-tuned csv table",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to tuned csv file(s); defaults to "
        "aiter/ops/flydsl/chunk_gdn_h_tuned.csv",
    )
    parser.add_argument(
        "--target-arch",
        type=str,
        default=None,
        help="Override the ``arch`` column of every csv row; useful for "
        "cross-compiling on a host whose GPU differs from the tuned arch "
        "(e.g. ``--target-arch gfx942`` on a gfx950 box).",
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
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS") or "(from csv)"

    all_jobs = collect_aot_jobs(csv_paths, parse_csv)

    # ``--target-arch`` rewrites the ``arch`` field of every job, then we
    # dedupe again because two csv rows that differed only in arch
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
    for csv_path in csv_paths:
        print(f"  csv:              {csv_path}")
    print(f"  Total jobs:       {len(all_jobs)}")
    if args.target_arch:
        print(f"  Compile arch:     {args.target_arch} (overridden by --target-arch)")
    else:
        print("  Compile arch:     (from csv 'arch' column)")
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
