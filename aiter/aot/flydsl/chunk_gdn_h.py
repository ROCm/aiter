#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL chunk-gated-delta-h (K5) kernels.

Two compiled products live behind K5 and this module builds both:

``vk``
    The baseline kernel ``chunk_gdn_fwd_h_flydsl_vk``. Its table
    ``aiter/ops/flydsl/chunk_gdn_h_tuned.csv`` is an AOT seed list only:
    runtime BV comes from the ``_heuristic_bv`` rule, which was calibrated
    against these very rows and never reads the file. Every row already
    carries each compile-time switch, so a row *is* a compile job -- built
    twice, ``STATE_DTYPE_BF16`` False (legacy f32-state path) and True
    (``state_dtype=torch.bfloat16`` callers).

    That one-row-one-job shortcut is only sound while the rule keeps
    agreeing with the ``BV`` column; today it does for all 26 rows.

``mfma16_hip``
    The mfma16 / HIP-aligned fork ``chunk_gdn_fwd_h_flydsl_mfma16_hip``,
    which is what ``use_chunk_flydsl=True`` in
    ``chunk_gated_delta_rule_fwd_opt_vk`` actually dispatches. Without it the
    first prefill of every process pays ~2.3s of JIT (~0.1-0.3s even with a
    warm disk cache). AOT reads ``aiter/configs/chunk_gdn_h_mfma16_hip_untuned.csv``
    (untuned-style shape list); runtime BV lookup reads
    ``aiter/configs/chunk_gdn_h_mfma16_hip_tuned.csv`` via ``AITER_CONFIGS``.
    See ``parse_csv_mfma16_hip``.

Usage:
    # Compile both kernels from their default tables
    python -m aiter.aot.flydsl.chunk_gdn_h

    # One kernel only, optionally from custom csv file(s)
    python -m aiter.aot.flydsl.chunk_gdn_h --kernel vk --csv /path/to/tuned.csv
    python -m aiter.aot.flydsl.chunk_gdn_h --kernel mfma16_hip

    # Cross-compile every entry for a different GPU arch (host need not
    # be that GPU; FlyDSL emits ISA for the requested target).
    python -m aiter.aot.flydsl.chunk_gdn_h --target-arch gfx942

    # List the expanded jobs without compiling
    python -m aiter.aot.flydsl.chunk_gdn_h --dry-run

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH / GPU_ARCHS          Target GPU architecture. For ``vk`` this is a
                              logging hint only (its per-job arch comes from
                              the csv ``arch`` column); for ``mfma16_hip`` it
                              answers rows that leave ``arch`` empty.
    FLYDSL_GPU_ARCH           Per-job arch override applied during compile;
                              ``--target-arch`` takes precedence over both
                              this env var and the ``arch`` column in csv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    dedupe_jobs,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_ROOT_DIR
from aiter.ops.flydsl.kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from aiter.ops.flydsl.kernels.chunk_gated_delta_h_mfma16x16x16 import (
    compile_chunk_gated_delta_h_mfma16_hip,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

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

# --------------------------------------------------------------------------
# mfma16_hip fork
# --------------------------------------------------------------------------

# AOT untuned: canonical stub + ``model_configs/*_chunk_gdn_h_mfma16_hip_untuned.csv``.
# Runtime BV: ``chunk_gdn_h_mfma16_hip_tuned.csv`` merged with model_configs via ``AITER_CONFIGS``.
_DEFAULT_CSV_MFMA16_HIP = Path(
    f"{AITER_ROOT_DIR}/aiter/configs/chunk_gdn_h_mfma16_hip_untuned.csv"
)
_MODEL_CONFIG_DIR = Path(f"{AITER_ROOT_DIR}/aiter/configs/model_configs")
DEFAULT_CSVS_MFMA16_HIP = sorted(
    {str(_DEFAULT_CSV_MFMA16_HIP)}
    | {
        str(p)
        for p in _MODEL_CONFIG_DIR.glob("*_chunk_gdn_h_mfma16_hip_untuned.csv")
        if p.is_file()
    }
)
# Used only when a csv row leaves ``arch`` empty and neither ARCH/GPU_ARCHS nor
# a local GPU can answer.
MFMA16_HIP_AOT_ARCH_DEFAULT = "gfx950"
# Mirrors the ``@flyc.kernel(name=...)`` decorator in
# ``kernels/chunk_gated_delta_h_mfma16x16x16.py``.
_MFMA16_HIP_KERNEL_NAME = "chunk_gdn_fwd_h_flydsl_mfma16_hip"

_MFMA16_HIP_TORCH_DTYPE = {"torch.bfloat16": "bfloat16"}
# Mirrors ``_HIPEQ_BV_CANDIDATES`` in linear_attention_prefill_kernels.py; the
# runtime picks one of these per batch, so all legal ones are pre-compiled.
_BV_CANDIDATES = (64, 32, 16)

# Compile both dtype / layout / state-index variants the wrapper may dispatch to.
_SNAPSHOT_BF16 = (True, False)
_STATE_BF16 = (False, True)
_G_HEAD_MAJOR = (True, False)
_USE_STATE_INDICES = (False, True)

# Production dispatch pins from chunk.py (g_cumsum + use_exp2, no gk, save v_new).
_FIXED_SWITCHES: dict[str, bool] = {
    "use_g": True,
    "use_gk": False,
    "save_vn": True,
    "wu_contig": True,
    "g_log2_scaled": True,
    "bf16_convert_trunc": True,
}

_ARCH_SEP = re.compile(r"[;,\s]+")


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


# --------------------------------------------------------------------------
# baseline kernel (chunk_gdn_fwd_h_flydsl_vk)
# --------------------------------------------------------------------------


def parse_csv(csv_path: str) -> list[dict[str, Any]]:
    """Parse the chunk_gdn_h tuned csv and return unique compile jobs.

    Each row already carries every compile-time switch the kernel cares
    about (K/V/BT/H/Hg/use_g/use_gk/use_h0/store_fs/save_vn/is_varlen/
    wu_contig) plus the offline-tuned ``BV``. We only keep the fields
    that actually influence MLIR compilation; ``T_flat``/``N`` and
    ``duration`` are dropped (they affect the host launch grid, not the
    compiled artifact).
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
                bv = int(row["BV"])
                k = int(row["K"])
                v = int(row["V"])
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue

            if v % bv != 0 or bv > v:
                print(
                    f"  [WARN] BV={bv} does not divide V={v}, skipping row "
                    f"in {csv_path}"
                )
                continue

            try:
                job = {
                    "dtype": dtype,
                    "arch": row.get("arch") or CHUNK_GDN_H_AOT_ARCH_DEFAULT,
                    "K": k,
                    "V": v,
                    "BT": int(row.get("BT") or 64),
                    "BV": bv,
                    "H": int(row["H"]),
                    "Hg": int(row["Hg"]),
                    "use_g": _parse_bool(row.get("use_g") or "True"),
                    "use_gk": _parse_bool(row.get("use_gk") or "False"),
                    "use_h0": _parse_bool(row.get("use_h0") or "True"),
                    "store_fs": _parse_bool(row.get("store_fs") or "False"),
                    "save_vn": _parse_bool(row.get("save_vn") or "True"),
                    "is_varlen": _parse_bool(row.get("is_varlen") or "False"),
                    "wu_contig": _parse_bool(row.get("wu_contig") or "True"),
                    # state dtype is not tracked in the tuned csv yet; default
                    # f32 here, then main() unconditionally fans out into a
                    # bf16 twin so both runtime paths are pre-compiled.
                    "state_bf16": False,
                }
            except (KeyError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue

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
        with (
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            FakeTensorMode(),
        ):
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
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


# --------------------------------------------------------------------------
# mfma16_hip fork (chunk_gdn_fwd_h_flydsl_mfma16_hip)
# --------------------------------------------------------------------------


def _detected_arch() -> str | None:
    try:
        from flydsl.runtime.device import get_rocm_arch

        arch = get_rocm_arch()
    except Exception:  # noqa: BLE001 - no GPU / no ROCm on the build host
        return None
    return arch.split(":")[0] if arch else None


def _resolve_archs(row_arch: str | None) -> list[str]:
    """Arch list for one mfma16_hip csv row: explicit column, else
    ARCH/GPU_ARCHS, else the host GPU, else ``MFMA16_HIP_AOT_ARCH_DEFAULT``.

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


def parse_csv_mfma16_hip(csv_path: str) -> list[dict[str, Any]]:
    """Expand the mfma16_hip untuned table into unique compile jobs.

    Rows collapse to their distinct ``(arch, dtype, K, V, BT, H, Hg, is_varlen,
    use_h0, store_fs)`` shapes -- the untuned table has one row per AOT-covered
    compile shape. Each shape then fans out over every legal BV and both dtype
    specializations.
    """
    shapes: dict[tuple, None] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in rows:
            dtype = (row.get("dtype") or "torch.bfloat16").strip()
            if dtype not in _MFMA16_HIP_TORCH_DTYPE:
                print(f"  [WARN] Unsupported dtype {dtype!r}, skipping")
                continue

            try:
                K = int(row["K"])
                V = int(row["V"])
                BT = int(row.get("BT") or 64)
                H = int(row["H"])
                Hg = int(row["Hg"])
                is_varlen = _parse_bool(row.get("is_varlen") or "True")
                use_h0 = _parse_bool(row.get("use_h0") or "True")
                store_fs = _parse_bool(row.get("store_fs") or "True")
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue

            for arch in _resolve_archs(row.get("arch")):
                shapes[(arch, dtype, K, V, BT, H, Hg, is_varlen, use_h0, store_fs)] = (
                    None
                )

    jobs: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for arch, dtype, K, V, BT, H, Hg, is_varlen, use_h0, store_fs in shapes:
        bvs = [bv for bv in _BV_CANDIDATES if bv <= V and V % bv == 0]
        if not bvs:
            print(f"  [WARN] no legal BV for V={V}, skipping shape in {csv_path}")
            continue
        indices = _USE_STATE_INDICES if (use_h0 and store_fs) else (False,)
        for (
            bv,
            snapshot_bf16,
            state_bf16,
            g_head_major,
            use_state_indices,
        ) in itertools.product(
            bvs, _SNAPSHOT_BF16, _STATE_BF16, _G_HEAD_MAJOR, indices
        ):
            job = {
                "kernel_name": _MFMA16_HIP_KERNEL_NAME,
                "dtype": dtype,
                "arch": arch,
                "K": K,
                "V": V,
                "BT": BT,
                "BV": bv,
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


def _torch_dtype_for_mfma16_hip(dtype_str: str):
    import torch

    name = _MFMA16_HIP_TORCH_DTYPE.get(dtype_str)
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

    import torch

    dev = torch.device("cpu")
    torch_dtype = _torch_dtype_for_mfma16_hip(dtype)
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


def _format_shape_str_mfma16_hip(job: dict) -> str:
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


def compile_one_config_mfma16_hip(*, arch: str, **kwargs) -> dict:
    """Compile one mfma16_hip configuration and save it to cache."""
    aot_arch = arch or MFMA16_HIP_AOT_ARCH_DEFAULT
    kwargs.pop("kernel_name", None)
    shape_str = _format_shape_str_mfma16_hip(kwargs)
    result = {
        "kernel_name": _MFMA16_HIP_KERNEL_NAME,
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _fanout_state_dtype(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fan the baseline jobs out into both f32-state and bf16-state variants so
    neither runtime path pays a JIT cost on first call. ``dedupe_jobs`` drops
    any pre-existing dup."""
    return dedupe_jobs(jobs + [dict(j, state_bf16=True) for j in jobs])


@dataclass(frozen=True)
class _KernelSpec:
    title: str
    default_csvs: list[str]
    parse_csv: Callable[[str], list[dict[str, Any]]]
    compile_one_config: Callable[..., dict[str, Any]]
    format_shape: Callable[[dict[str, Any]], str]
    # Post-parse fan-out that only a standalone run gets. ``mfma16_hip`` does
    # its fan-out inside parse_csv instead, so the wheel build sees it too.
    expand_jobs: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None


_KERNEL_SPECS: dict[str, _KernelSpec] = {
    "vk": _KernelSpec(
        title=f"baseline  ({_KERNEL_NAME})",
        default_csvs=DEFAULT_CSVS,
        parse_csv=parse_csv,
        compile_one_config=compile_one_config,
        format_shape=_format_shape_str,
        expand_jobs=_fanout_state_dtype,
    ),
    "mfma16_hip": _KernelSpec(
        title=f"mfma16_hip  ({_MFMA16_HIP_KERNEL_NAME})",
        default_csvs=DEFAULT_CSVS_MFMA16_HIP,
        parse_csv=parse_csv_mfma16_hip,
        compile_one_config=compile_one_config_mfma16_hip,
        format_shape=_format_shape_str_mfma16_hip,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL chunk-gated-delta-h kernels "
        "from the offline-tuned csv tables",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--kernel",
        choices=(*_KERNEL_SPECS, "all"),
        default="all",
        help="Which K5 compiled product to build (default: %(default)s).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to tuned csv file(s); requires a single --kernel. "
        "Defaults to each kernel's own table under aiter/ops/flydsl/.",
    )
    parser.add_argument(
        "--target-arch",
        type=str,
        default=None,
        help="Override the ``arch`` of every job; useful for cross-compiling "
        "on a host whose GPU differs from the tuned arch "
        "(e.g. ``--target-arch gfx942`` on a gfx950 box).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded job list without compiling anything.",
    )
    args = parser.parse_args()

    kernels = tuple(_KERNEL_SPECS) if args.kernel == "all" else (args.kernel,)
    if args.csv and len(kernels) > 1:
        parser.error(
            "--csv applies to one kernel; pass --kernel "
            f"{' or --kernel '.join(_KERNEL_SPECS)}"
        )

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )

    print("=" * 72)
    print("FlyDSL chunk-gated-delta-h AOT Pre-compilation")
    print("=" * 72)
    print(f"  Cache dir:        {cache_dir}")

    plans: list[tuple[_KernelSpec, list[dict[str, Any]]]] = []
    for name in kernels:
        spec = _KERNEL_SPECS[name]
        csv_paths = [os.path.abspath(p) for p in (args.csv or spec.default_csvs)]
        for csv_path in csv_paths:
            if not os.path.isfile(csv_path):
                print(f"Error: csv file not found: {csv_path}")
                sys.exit(1)

        jobs = collect_aot_jobs(csv_paths, spec.parse_csv)
        # ``--target-arch`` rewrites the ``arch`` field of every job, then we
        # dedupe again because two csv rows that differed only in arch
        # collapse to the same compile after the override.
        if args.target_arch and jobs:
            jobs = dedupe_jobs([dict(j, arch=args.target_arch) for j in jobs])
        if spec.expand_jobs and jobs:
            jobs = spec.expand_jobs(jobs)
        plans.append((spec, jobs))

        archs = sorted({j["arch"] for j in jobs})
        arch_note = " (overridden by --target-arch)" if args.target_arch else ""
        print("-" * 72)
        print(f"  {spec.title}")
        for csv_path in csv_paths:
            print(f"    csv:            {csv_path}")
        print(f"    jobs:           {len(jobs)}")
        print(f"    compile arch:   {', '.join(archs) or '-'}{arch_note}")
    print("=" * 72)

    if args.dry_run:
        for spec, jobs in plans:
            for job in jobs:
                print(f"  {job['arch']}  {spec.format_shape(job)}")
        sys.exit(0)

    total_t0 = time.time()
    ok = fail = 0
    for spec, jobs in plans:
        print(f"\n--- Compiling {len(jobs)} kernels: {spec.title} ---")
        results = run_jobs_parallel(spec.compile_one_config, jobs)
        ok += sum(1 for r in results if r["compile_time"] is not None)
        fail += sum(1 for r in results if r["compile_time"] is None)

    total_elapsed = time.time() - total_t0

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
