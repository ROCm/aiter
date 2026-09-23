#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compile the three-phase blocked-scan GDN prefill path.

The blocked path is enabled only for the bf16 K=V=128 specialization. Its
(H, Hg) coverage follows the serial K5 opt CSV so JIT and AOT remain aligned.
"""

from __future__ import annotations

import csv
import time
from typing import Any

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    compile_only_env,
    job_identity,
    override_env,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.kernels.gdr_prefill import compile_chunk_gated_delta_h
from aiter.ops.flydsl.kernels.gdr_prefill.chunk_gdn_carry_gfx950 import (
    compile_chunk_gdn_carry,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

CHUNK_GDN_BLOCKED_AOT_ARCH_DEFAULT = "gfx950"
_KERNEL_NAME = "chunk_gdn_blocked"

# Keep the blocked AOT coverage aligned with serial runtime config resolution.
DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE]


def parse_csv(csv_path: str) -> list[dict[str, Any]]:
    """Expand eligible serial K5 rows into blocked-path phase jobs.

    The blocked launch fixes bf16, K=V=128, BV=64, head-major g, and unindexed
    state. Carry depends only on H, so it is deduplicated across Hg rows.
    """
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
    """Compile one blocked-scan phase configuration and save it to cache."""
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
