# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HSTU Attention Forward API."""

from __future__ import annotations
from aiter.ops.flydsl.kernels.hstu_attention_fwd import (
    validate_hstu_attention_fwd,
    build_hstu_attention_fwd,
)
from aiter.ops.flydsl.kernels.hstu_attention_bwd import (
    validate_hstu_attention_bwd,
    build_hstu_attention_bwd,
)
from aiter.ops.flydsl.kernels.hstu_attention_bwd_dq import (
    build_hstu_attention_bwd_dq,
)
import csv
import functools
import torch
import flydsl.expr as fx

from pathlib import Path
from typing import Optional, Callable

from flydsl.runtime.device import get_rocm_arch
from aiter import logger
from aiter.ops.triton.utils.common_utils import prev_power_of_2
from aiter.utility.dtypes import str2bool

from .kernels.tensor_shim import _run_compiled, get_dtype_str

__all__ = [
    "flydsl_hstu_attention_fwd",
    "flydsl_hstu_attention_bwd",
]


_GPU_ARCH = get_rocm_arch()


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
        str2bool(has_window),
        str2bool(has_contextual),
        str2bool(has_targets),
    )


@functools.lru_cache()
def _tuned_config_map(tuned_file: str | None = None) -> dict[tuple, dict]:
    def _parse_row(row: dict) -> tuple[tuple, float, dict]:
        if set(row.keys()) != set(_CSV_COLUMNS):
            raise KeyError(
                f"unexpected columns: {set(row.keys()) ^ set(_CSV_COLUMNS)}"
            )

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
        kernel_config = dict(
            block_m=int(row["block_m"]),
            block_n=int(row["block_n"]),
            num_waves=int(row["num_waves"]),
            waves_per_eu=int(row["waves_per_eu"]),
        )
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
    batch: int,
    head_dim: int,
    hidden_dim: int,
    num_heads: int,
    max_seq_len: int,
    max_attn_len: int,
) -> dict:
    """
    Heuristic config for when tuning is unavailable.

    Derived from a device sweep over shapes on MI300X.
    - block_m by occupancy
    - block_n by head/hidden dim (bounded by the LDS K+V tile)
    - waves_per_eu lifts residency only for tiny-dim tiles that benefit.
    """

    def as_dict(
        block_m: int,
        block_n: int,
        num_waves: int,
        waves_per_eu: int,
        /,
    ) -> dict:
        return dict(
            block_m=block_m,
            block_n=block_n,
            num_waves=num_waves,
            waves_per_eu=waves_per_eu,
        )

    # hidden_dims 96/160/224 don't divide the K/V DMA pass with num_waves=4
    # This map is so the kernel still runs for these values.
    non_64_divisible_map = {
        96: (96, 48, 3, 0),
        160: (160, 80, 5, 0),
        192: (96, 48, 3, 0),
        224: (112, 112, 7, 0),
    }
    if hidden_dim in non_64_divisible_map:
        return as_dict(*non_64_divisible_map[hidden_dim])

    grid = batch * num_heads * ((max_seq_len + 127) // 128)
    dim = max(head_dim, hidden_dim)

    if dim <= 64:
        if max_attn_len:
            return as_dict(128, 32, 4, 0)

        if grid >= 6144:
            return as_dict(256, 32, 4, 0)
        if grid >= 768:
            return as_dict(128, 32, 4, 2)
        return as_dict(64, 32, 4, 2)

    # dim > 64
    if grid >= 2560 or max_seq_len >= 16384:
        return as_dict(192, 48, 4, 0)
    if grid >= 768:
        return as_dict(128, 64, 4, 2)
    return as_dict(64, 64, 4, 2)


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
    block_m: Optional[int],
    block_n: Optional[int],
    num_waves: Optional[int],
    waves_per_eu: Optional[int],
) -> tuple[str, Callable]:
    #  Config overrides (if provided)
    custom_config: dict = dict(
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )
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
        batch=batch,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        max_attn_len=max_attn_len,
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
        batch=batch,
        causal=causal,
        max_attn_len=max_attn_len,
        has_targets=has_targets,
        alpha=alpha,
        dtype_str=dtype_str,
        max_seq_len=max_seq_len,
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
    num_targets: Optional[torch.Tensor],
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
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    *,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_waves: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    batch, num_heads, head_dim, hidden_dim, dtype_str = _validate_inputs(
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
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
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            seq_offsets.contiguous(),
            num_targets.contiguous(),
            out,
            fx.Stream(launch_stream),
        )
    return out


def _validate_bwd_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
) -> tuple[int, int, int, int, str]:
    """Validate backward inputs, reusing the forward's q/k/v checks.

    dout is the upstream gradient of the forward output O, so it must match v's
    (total_tokens, num_heads, hidden_dim) shape and dtype exactly.
    """
    batch, num_heads, head_dim, hidden_dim, dtype_str = _validate_inputs(
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
    )

    if not dout.is_cuda:
        raise ValueError("flydsl_hstu_attention_bwd requires device tensors")
    if dout.device != q.device:
        raise ValueError("dout must reside on q's device")
    if dout.dim() != 3:
        raise ValueError(f"dout must be rank 3, got {tuple(dout.shape)}")
    if dout.shape != v.shape:
        raise ValueError(
            "dout must share v's shape (it is dO), got "
            f"dout={tuple(dout.shape)} v={tuple(v.shape)}"
        )
    if dout.dtype != v.dtype:
        raise ValueError(
            f"dout must share v's dtype, got dout={dout.dtype} v={v.dtype}"
        )

    return batch, num_heads, head_dim, hidden_dim, dtype_str


# Backward tuned-config plumbing. Same schema as the forward CSV plus a
# sequence_parallel column (the dQ-reduction strategy knob).
_BWD_CSV_COLUMNS: list[str] = [
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
    "sequence_parallel",
    "duration",
]


@functools.lru_cache()
def _bwd_tuned_config_map(tuned_file: str | None = None) -> dict[tuple, dict]:
    def _parse_row(row: dict) -> tuple[tuple, float, dict]:
        if set(row.keys()) != set(_BWD_CSV_COLUMNS):
            raise KeyError(
                f"unexpected columns: {set(row.keys()) ^ set(_BWD_CSV_COLUMNS)}"
            )

        duration = float(row["duration"])

        problem_key = _problem_key(
            row["arch"],
            row["dtype"],
            row["num_heads"],
            row["head_dim"],
            row["hidden_dim"],
            row["batch"],
            row["max_seq_len"],
            row["has_window"],
            row["has_contextual"],
            row["has_targets"],
        )
        kernel_config = dict(
            block_m=int(row["block_m"]),
            block_n=int(row["block_n"]),
            num_waves=int(row["num_waves"]),
            waves_per_eu=int(row["waves_per_eu"]),
            sequence_parallel=str2bool(row["sequence_parallel"]),
        )
        return problem_key, duration, kernel_config

    default_tuned_file = (
        Path(__file__).resolve().parent / "hstu_attention_bwd_tuned.csv"
    )
    tuned_file_path: Path = Path(tuned_file) if tuned_file else default_tuned_file
    if not tuned_file_path.is_file():
        return {}

    config_map: dict = {}
    with tuned_file_path.open(mode="r", encoding="utf-8") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            try:
                problem_key, duration, kernel_config = _parse_row(row)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    f"[FlyDSL HSTU Bwd] skipping invalid tuned row {row_idx} in {tuned_file_path}: {exc}"
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


def _get_bwd_tuned_config(
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
    """Returns the tuned backward config if the CSV has an entry for this problem."""
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
    return _bwd_tuned_config_map().get(problem_key, {})


def _get_bwd_default_config() -> dict:
    """Conservative heuristic default when no tuned entry exists.

    This tile is valid across every supported shape (including asymmetric and
    non-64-divisible dims) and is what the correctness suite runs against.
    Tuned CSV entries override it per problem.
    """
    return dict(block_m=64, block_n=32, num_waves=4, waves_per_eu=0)


@functools.lru_cache(maxsize=16384)
def _compile_bwd_launcher(
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
    block_m: Optional[int],
    block_n: Optional[int],
    num_waves: Optional[int],
    waves_per_eu: Optional[int],
    sequence_parallel: Optional[bool],
) -> tuple[Callable, Callable]:
    """Builds the (dV/dK, dQ) launcher pair, resolving tuned -> default -> custom.

    Returns two launchers: the KV-owned kernel producing dV/dK, and the Q-owned
    kernel producing dQ. Both consume the same tile config.
    """
    custom_config: dict = dict(
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )
    custom_config = {k: v for k, v in custom_config.items() if v is not None}

    tuned_config = _get_bwd_tuned_config(
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

    default_config = _get_bwd_default_config()

    kernel_config = {
        **default_config,
        **tuned_config,
        **custom_config,
    }
    # sequence_parallel is threaded for cache-key stability and tuned-CSV
    # round-tripping; the dQ-reduction strategy it selects is not wired into the
    # build yet, so it is not forwarded to the kernel builders.
    kernel_config.pop("sequence_parallel", None)

    build_kwargs = dict(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        batch=batch,
        causal=causal,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        has_targets=has_targets,
        alpha=alpha,
        dtype_str=dtype_str,
        max_seq_len=max_seq_len,
        **kernel_config,
    )
    dvdk_launcher = build_hstu_attention_bwd(**build_kwargs)
    dq_launcher = build_hstu_attention_bwd_dq(**build_kwargs)
    return dvdk_launcher, dq_launcher


def flydsl_hstu_attention_bwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    *,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_waves: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    sequence_parallel: Optional[bool] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HSTU attention backward: returns (dq, dk, dv).

    Recomputes S = alpha*Q*K^T and sigma from Q,K (nothing is stashed by the
    forward), then dV = A^T dO, dS = M .* (1/N) sigma(1+S(1-sigma)) .* (dO V^T),
    dQ = alpha dS K, dK = alpha dS^T Q. Mirrors the forward's conventions
    (causal-only, {f16,bf16}, alpha-in-score, 1/N on dS, fast-math SiLU).
    """
    batch, num_heads, head_dim, hidden_dim, dtype_str = _validate_bwd_inputs(
        q=q,
        k=k,
        v=v,
        dout=dout,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
    )

    # dV/dK come from the KV-owned kernel; dQ from the Q-owned kernel. Both are
    # single-writer (no atomics): dV/dK reduce over the query index, dQ over the key index.
    # Config precedence: explicit overrides > tuned CSV > default heuristic.
    dvdk_launcher, dq_launcher = _compile_bwd_launcher(
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
        sequence_parallel=sequence_parallel,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    nt = num_targets
    if nt is None:
        nt = torch.zeros(1, dtype=seq_offsets.dtype, device=v.device)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    do_c = dout.contiguous()
    so_c = seq_offsets.contiguous()
    nt_c = nt.contiguous()

    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    with torch.cuda.device(q.device.index):
        _run_compiled(
            dvdk_launcher,
            q_c,
            k_c,
            v_c,
            do_c,
            so_c,
            nt_c,
            dv,
            dk,
            fx.Stream(launch_stream),
        )
        _run_compiled(
            dq_launcher,
            q_c,
            k_c,
            v_c,
            do_c,
            so_c,
            nt_c,
            dq,
            fx.Stream(launch_stream),
        )
    return dq, dk, dv

