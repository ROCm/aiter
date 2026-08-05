# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HSTU Attention Forward API."""

from __future__ import annotations

import csv
import functools
from collections.abc import Callable
from pathlib import Path

import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from aiter import logger
from aiter.ops.flydsl.kernels.hstu_attention_fwd import (
    build_hstu_attention_fwd,
    validate_hstu_attention_fwd,
)
from aiter.ops.triton.utils.common_utils import prev_power_of_2

from .kernels.tensor_shim import _run_compiled, get_dtype_str

__all__ = [
    "flydsl_hstu_attention_fwd",
]


_GPU_ARCH = get_rocm_arch()


def _str2bool(v: bool | str) -> bool:
    # Local copy to avoid importing aiter.utility.dtypes at module load: that
    # pulls in aiter.ops.enum, which triggers a JIT get_module() during the AOT
    # build (setup.py run_aot) before the module exists -> import-time failure.
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Boolean value expected, got {v!r}.")


# Tuned kernel configs
# list of column names in the tuned csv file
_CSV_COLUMNS: list[str] = [
    "arch",
    "dtype",
    "num_heads",
    "head_dim",
    "hidden_dim",
    "batch",
    "max_seq_len",
    "has_window",
    "has_contextual",
    "has_targets",
    "block_m",
    "block_n",
    "num_waves",
    "waves_per_eu",
    "duration",
]


def _problem_key(
    arch: str,
    dtype: str,
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    batch: int,
    max_seq_len: int,
    has_window: bool | str,
    has_contextual: bool | str,
    has_targets: bool | str,
) -> tuple:
    return (
        arch.strip().lower(),
        dtype.strip().lower(),
        int(num_heads),
        int(head_dim),
        int(hidden_dim),
        prev_power_of_2(int(batch)),
        prev_power_of_2(int(max_seq_len)),
        _str2bool(has_window),
        _str2bool(has_contextual),
        _str2bool(has_targets),
    )


@functools.lru_cache
def _tuned_config_map(tuned_file: str | None = None) -> dict[tuple, dict]:
    def _parse_row(row: dict) -> tuple[tuple, float, dict]:
        if set(row.keys()) != set(_CSV_COLUMNS):
            raise KeyError(f"unexpected columns: {set(row.keys()) ^ set(_CSV_COLUMNS)}")

        duration = float(row["duration"])

        problem_key = _problem_key(
            row["arch"],
            row["dtype"],
            row["num_heads"],
            row["head_dim"],
            row["hidden_dim"],
            row["batch"],
            row["max_seq_len"],
            (row["has_window"]),
            row["has_contextual"],
            row["has_targets"],
        )
        kernel_config = {
            "block_m": int(row["block_m"]),
            "block_n": int(row["block_n"]),
            "num_waves": int(row["num_waves"]),
            "waves_per_eu": int(row["waves_per_eu"]),
        }
        return (
            problem_key,
            duration,
            kernel_config,
        )

    default_tuned_file = Path(__file__).resolve().parent / "hstu_attention_tuned.csv"

    tuned_file: Path = Path(tuned_file) if tuned_file else default_tuned_file
    if not tuned_file.is_file():
        return {}

    config_map: dict = {}
    with tuned_file.open(mode="r", encoding="utf-8") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            try:
                problem_key, duration, kernel_config = _parse_row(row)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    f"[FlyDSL HSTU Fwd] skipping invalid tuned row {row_idx} in {tuned_file}: {exc}"
                )
                continue

            if duration <= 0.0:
                continue

            if problem_key not in config_map or duration < config_map[problem_key][0]:
                config_map[problem_key] = (duration, kernel_config)

    return {
        problem_key: kernel_config
        for problem_key, (_, kernel_config) in config_map.items()
    }


def _get_tuned_config(
    *,
    dtype_str: str,
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    batch: int,
    max_seq_len: int,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
) -> dict:
    """
    Returns the tuned kernel config if it exists for the given parameters.
    """

    problem_key = _problem_key(
        _GPU_ARCH,
        dtype_str,
        num_heads,
        head_dim,
        hidden_dim,
        batch,
        max_seq_len,
        max_attn_len > 0,
        contextual_seq_len > 0,
        has_targets,
    )

    return _tuned_config_map().get(problem_key, {})


def _get_default_config(
    *,
    head_dim: int,
    hidden_dim: int,
) -> dict:
    """
    Heuristic config for when tuning is unavailable.
    Derived from a device sweep over shapes on MI300X.
    """

    def as_dict(
        block_m: int,
        block_n: int,
        num_waves: int,
        waves_per_eu: int,
        /,
    ) -> dict:
        return {
            "block_m": block_m,
            "block_n": block_n,
            "num_waves": num_waves,
            "waves_per_eu": waves_per_eu,
        }

    # hidden_dims 96/160/192 don't divide the K/V DMA pass with num_waves=4
    # This map is required so the kernel still runs for these values.
    non_64_divisible_map = {
        96: (96, 48, 3, 0),
        160: (160, 80, 5, 0),
        192: (96, 48, 3, 0),
    }
    if hidden_dim in non_64_divisible_map:
        return as_dict(*non_64_divisible_map[hidden_dim])

    # Key on the K stride the kernel actually processes: head_dim is rounded up to a
    # multiple of 64 (HEAD_DIM_K) for the swizzled K LDS tile. Using the unrounded
    # head_dim here picks too-large a block_n for non-64-aligned dims in the 128-192
    # (and 192-256) band, which measured ~34% slower on gfx942 (144/176 head dims).
    head_dim_k = ((head_dim + 63) // 64) * 64
    dim = max(hidden_dim, head_dim_k)
    if dim >= 256:
        return as_dict(128, 16, 4, 0)
    if dim >= 192:
        return as_dict(128, 32, 4, 2)
    if dim >= 128:
        return as_dict(128, 64, 4, 2)
    return as_dict(128, 32, 4, 2)


@functools.lru_cache(maxsize=16384)
def _compile_launcher(
    *,
    batch: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    has_targets: bool,
    alpha: float,
    max_attn_len: int,
    contextual_seq_len: int,
    dtype_str: str,
    block_m: int | None,
    block_n: int | None,
    num_waves: int | None,
    waves_per_eu: int | None,
) -> Callable:
    #  Config overrides (if provided)
    custom_config: dict = {
        "block_m": block_m,
        "block_n": block_n,
        "num_waves": num_waves,
        "waves_per_eu": waves_per_eu,
    }
    custom_config = {k: v for k, v in custom_config.items() if v is not None}

    # Tuned config entry
    tuned_config = _get_tuned_config(
        dtype_str=dtype_str,
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        batch=batch,
        max_seq_len=max_seq_len,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        has_targets=has_targets,
    )

    # Default hueristic config
    default_config = _get_default_config(
        hidden_dim=hidden_dim,
        head_dim=head_dim,
    )

    kernel_config = {
        **default_config,
        **tuned_config,
        **custom_config,
    }

    kwargs: dict = dict(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        causal=causal,
        max_attn_len=max_attn_len,
        has_targets=has_targets,
        alpha=alpha,
        dtype_str=dtype_str,
        contextual_seq_len=contextual_seq_len,
        **kernel_config,
    )
    validate_hstu_attention_fwd(**kwargs)
    launcher = build_hstu_attention_fwd(**kwargs)
    return launcher


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    max_seq_len: int,
) -> tuple[int, int, int, int, str]:
    tensors: dict[str, torch.Tensor] = {
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": seq_offsets,
    }
    if num_targets is not None:
        tensors["num_targets"] = num_targets

    if not all(t.is_cuda for t in tensors.values()):
        raise ValueError("flydsl_hstu_attention_fwd requires device tensors")
    if not all(t.device == tensors["q"].device for t in tensors.values()):
        raise ValueError("tensors must reside on the same device")

    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            "q/k/v must be rank 3, got "
            f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if q.shape != k.shape:
        raise ValueError(
            "q and k must have the same shape, got "
            f"q={tuple(q.shape)} k={tuple(k.shape)}"
        )
    if v.shape[0] != q.shape[0] or v.shape[1] != q.shape[1]:
        raise ValueError(
            "v must share q's token count and head count, got "
            f"q={tuple(q.shape)} v={tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"q/k/v must share the same dtype, got q={q.dtype} k={k.dtype} v={v.dtype}"
        )

    dtype_str = get_dtype_str(q.dtype)
    num_heads, head_dim = q.shape[-2:]
    hidden_dim = v.shape[2]
    batch = seq_offsets.numel() - 1

    if batch <= 0:
        raise ValueError(
            f"batch (seq_offsets.numel() - 1) must be positive, got {batch}"
        )
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len (N) must be positive, got {max_seq_len}")
    if dtype_str is None:
        raise ValueError(f"Unsupported dtype: get_dtype_str({q.dtype}) is None")
    if num_targets is not None:
        if num_targets.device != q.device:
            raise ValueError(
                f"num_targets must be on q's device ({q.device}), got {num_targets.device}"
            )
        if num_targets.numel() != batch:
            raise ValueError(
                f"num_targets length ({num_targets.numel()}) must equal batch ({batch})"
            )

    return (
        batch,
        num_heads,
        head_dim,
        hidden_dim,
        dtype_str,
    )


def flydsl_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: torch.Tensor | None,
    max_attn_len: int,
    contextual_seq_len: int,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    num_waves: int | None = None,
    waves_per_eu: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    batch, num_heads, head_dim, hidden_dim, dtype_str = _validate_inputs(
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_seq_len=N,
    )

    launcher = _compile_launcher(
        batch=batch,
        max_seq_len=N,
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        causal=causal,
        has_targets=num_targets is not None,
        alpha=alpha,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        dtype_str=dtype_str,
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )

    out = torch.empty_like(v)
    if num_targets is None:
        num_targets = torch.zeros(1, dtype=seq_offsets.dtype, device=out.device)

    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    with torch.cuda.device(q.device.index):
        _run_compiled(
            launcher,
            N,
            batch,
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            seq_offsets.contiguous(),
            num_targets.contiguous(),
            out,
            fx.Stream(launch_stream),
        )
    return out
