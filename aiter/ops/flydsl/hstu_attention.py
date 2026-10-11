# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HSTU attention API."""

from __future__ import annotations

import csv
import functools
from collections.abc import Callable
from pathlib import Path

import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from aiter import logger
from aiter.ops.flydsl.kernels.hstu.hstu_attention_bwd import (
    NUM_GRID_GROUPS,
    build_hstu_attention_bwd_dvdk,
    validate_hstu_attention_bwd,
)
from aiter.ops.flydsl.kernels.hstu.hstu_attention_bwd_dq import (
    build_hstu_attention_bwd_dq,
)
from aiter.ops.flydsl.kernels.hstu.hstu_attention_fwd import (
    build_hstu_attention_fwd,
    validate_hstu_attention_fwd,
)
from aiter.ops.triton.utils.common_utils import prev_power_of_2

from .kernels.tensor_shim import _run_compiled, get_dtype_str

__all__ = [
    "FlydslHstuAttention",
    "flydsl_hstu_attention",
    "flydsl_hstu_attention_bwd",
    "flydsl_hstu_attention_fwd",
]


_GPU_ARCH = get_rocm_arch()


def _str2bool(v: bool | str) -> bool:
    # Avoid utility.dtypes here because its imports require the AOT module to exist.
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Boolean value expected, got {v!r}.")


# Tuned forward-kernel CSV schema.
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

    default_tuned_file = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "model_configs"
        / "hstu_attention_tuned.csv"
    )

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
    """Return the tuned forward config for this problem, if present."""

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
    """Return a forward config for problems without a tuned entry."""

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

    # These dimensions need non-default wave counts to divide the K/V DMA pass.
    non_64_divisible_map = {
        96: (96, 48, 3, 0),
        160: (160, 80, 5, 0),
        192: (96, 48, 3, 0),
    }
    if hidden_dim in non_64_divisible_map:
        return as_dict(*non_64_divisible_map[hidden_dim])

    # Select by the padded K stride used by the swizzled LDS tile.
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
    custom_config: dict = {
        "block_m": block_m,
        "block_n": block_n,
        "num_waves": num_waves,
        "waves_per_eu": waves_per_eu,
    }
    custom_config = {k: v for k, v in custom_config.items() if v is not None}

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
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
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


def _validate_bwd_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    max_seq_len: int,
) -> tuple[int, int, int, int, str]:
    """Validate backward inputs with the shared q/k/v checks.

    dout is the upstream gradient of the forward output O, so it must match v's
    (total_tokens, num_heads, hidden_dim) shape and dtype exactly.
    """
    batch, num_heads, head_dim, hidden_dim, dtype_str = _validate_inputs(
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_seq_len=max_seq_len,
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


# dV/dK owns KV rows; dQ owns query rows. Each single-writer kernel has an
# independent tuned configuration.
# CSV tags identify the fused dV+dK kernel and the dQ kernel.
_BWD_KERNEL_DVDK = "dvdk"
_BWD_KERNEL_DQ = "dq"
_BWD_KERNELS = (_BWD_KERNEL_DVDK, _BWD_KERNEL_DQ)

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
    "kernel",
    "block_m",
    "block_n",
    "num_waves",
    "waves_per_eu",
    "duration_us",
]

# Tile columns are excluded from the backward problem key.
_BWD_TILE_COLUMNS = ("block_m", "block_n", "num_waves", "waves_per_eu")


@functools.lru_cache
def _bwd_tuned_config_map(tuned_file: str | None = None) -> dict[tuple, dict]:
    def _parse_row(row: dict) -> tuple[tuple, float, dict]:
        required = set(_BWD_CSV_COLUMNS)
        if not required.issubset(row.keys()):
            raise KeyError(f"missing columns: {required - set(row.keys())}")

        duration = float(row["duration_us"])

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
        kernel = row["kernel"].strip().lower()
        if kernel not in _BWD_KERNELS:
            raise ValueError(
                f"unknown kernel discriminator {kernel!r} (expected one of {_BWD_KERNELS})"
            )
        kernel_config = {
            "block_m": int(row["block_m"]),
            "block_n": int(row["block_n"]),
            "num_waves": int(row["num_waves"]),
            "waves_per_eu": int(row["waves_per_eu"]),
        }
        return (problem_key, kernel), duration, kernel_config

    default_tuned_file = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "model_configs"
        / "hstu_attention_bwd_tuned.csv"
    )
    tuned_file_path: Path = Path(tuned_file) if tuned_file else default_tuned_file
    if not tuned_file_path.is_file():
        return {}

    config_map: dict = {}
    with tuned_file_path.open(mode="r", encoding="utf-8") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            try:
                parsed = _parse_row(row)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    f"[FlyDSL HSTU Bwd] skipping invalid tuned row {row_idx} in {tuned_file_path}: {exc}"
                )
                continue

            problem_key, duration, kernel_config = parsed

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
    kernel: str,
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
    """Return a backward kernel's tuned config, if present."""
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
    return _bwd_tuned_config_map().get((problem_key, kernel), {})


@functools.lru_cache(maxsize=64)
def _balance_gather_index(B: int, groups: int, device: str) -> torch.Tensor:
    """Map sorted ranks to grid groups with a cached alternating-direction deal.

    Alternating the group direction each round distributes long and short
    sequences across the hardware groups. The mapping depends only on shape.
    """
    rank = torch.arange(B, device=device)
    rnd, pos = rank // groups, rank % groups
    group = torch.where(rnd % 2 == 0, pos, (groups - 1) - pos)
    slot = group * (B // groups) + rnd
    inv = torch.empty_like(rank)
    inv[slot] = rank
    return inv


@functools.lru_cache(maxsize=64)
def _identity_perm(B: int, device: str) -> torch.Tensor:
    """Return a cached identity permutation."""
    return torch.arange(B, dtype=torch.int32, device=device)


def _build_balance_perm(
    seq_offsets: torch.Tensor, groups: int = NUM_GRID_GROUPS
) -> torch.Tensor:
    """Balance quadratic sequence work across contiguous grid groups.

    Sequences are sorted by length, then dealt across groups in alternating
    directions. `perm[slot]` is the sequence processed by that grid slot.
    Unaligned batches use identity order because group boundaries split the deal.
    """
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    B = int(lengths.numel())
    device = seq_offsets.device
    if B % groups != 0:
        return _identity_perm(B, str(device))
    order = torch.argsort(lengths, descending=True)  # sequence indices, longest first
    inv = _balance_gather_index(B, groups, str(device))
    return order[inv].to(torch.int32)


# Fallbacks trade tile size for wave counts that satisfy K/V and dO DMA divisibility.
_BWD_FALLBACK_TILES = (
    (96, 48, 3, 0),
    (160, 80, 5, 0),
    (96, 64, 3, 0),
    (64, 64, 2, 0),
    (112, 112, 7, 0),
    (48, 96, 3, 0),
    (32, 32, 2, 0),
    (16, 16, 1, 0),
)


@functools.lru_cache(maxsize=1024)
def _get_bwd_default_config(
    kernel: str, *, head_dim: int, hidden_dim: int, arch: str | None = None
) -> dict:
    """Return the first fallback tile valid for these dimensions and architecture."""
    base = (128, 32, 4, 0) if kernel == _BWD_KERNEL_DVDK else (64, 32, 4, 0)
    candidates = (base, *_BWD_FALLBACK_TILES)
    last_error = None
    for block_m, block_n, num_waves, waves_per_eu in candidates:
        config = {
            "block_m": block_m,
            "block_n": block_n,
            "num_waves": num_waves,
            "waves_per_eu": waves_per_eu,
        }
        try:
            validate_hstu_attention_bwd(
                1,
                head_dim,
                hidden_dim,
                True,
                0,
                0,
                False,
                1.0,
                "bf16",
                512,
                arch=arch,
                **config,
            )
        except ValueError as e:
            last_error = e
            continue
        return config
    raise ValueError(
        f"no valid backward tile for head_dim={head_dim}, hidden_dim={hidden_dim} on "
        f"{arch or get_rocm_arch()}: none of the {len(candidates)} fallback tiles "
        f"validate (last: {last_error}). Pass an explicit block_m / block_n / "
        f"num_waves, or use a hidden_dim that is a multiple of 64."
    )


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
    block_m: int | None,
    block_n: int | None,
    num_waves: int | None,
    waves_per_eu: int | None,
    has_perm: bool = False,
) -> tuple[Callable, Callable]:
    """Build dV/dK and dQ launchers with independent kernel configurations.

    Explicit overrides take precedence over tuned entries and defaults.
    """
    # Explicit overrides apply to both kernels.
    custom_config: dict = {
        "block_m": block_m,
        "block_n": block_n,
        "num_waves": num_waves,
        "waves_per_eu": waves_per_eu,
    }
    custom_config = {k: v for k, v in custom_config.items() if v is not None}

    def _resolve(kernel: str) -> dict:
        tuned_config = _get_bwd_tuned_config(
            kernel=kernel,
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
        return {
            **_get_bwd_default_config(kernel, head_dim=head_dim, hidden_dim=hidden_dim),
            **tuned_config,
            **custom_config,
        }

    common_kwargs = {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
        "causal": causal,
        "max_attn_len": max_attn_len,
        "contextual_seq_len": contextual_seq_len,
        "has_targets": has_targets,
        "alpha": alpha,
        "dtype_str": dtype_str,
        "max_seq_len": max_seq_len,
        "has_perm": has_perm,
    }
    dvdk_config = _resolve(_BWD_KERNEL_DVDK)
    dq_config = _resolve(_BWD_KERNEL_DQ)

    dvdk_launcher = build_hstu_attention_bwd_dvdk(**common_kwargs, **dvdk_config)
    dq_launcher = build_hstu_attention_bwd_dq(**common_kwargs, **dq_config)
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
    num_targets: torch.Tensor | None,
    max_attn_len: int,
    contextual_seq_len: int,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    num_waves: int | None = None,
    waves_per_eu: int | None = None,
    sort_by_length: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HSTU attention backward: returns (dq, dk, dv).

    Recomputes `Z = alpha * Q * K^T`, then applies the mask, SiLU derivative,
    and `1/N` normalization. dV/dK share one score pass; dQ uses a second pass.
    """
    batch, num_heads, head_dim, hidden_dim, dtype_str = _validate_bwd_inputs(
        q=q,
        k=k,
        v=v,
        dout=dout,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_seq_len=N,
    )

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
        has_perm=sort_by_length,
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

    # Disabled kernels receive a dummy permutation that compile-time guards never read.
    if sort_by_length:
        perm_c = _build_balance_perm(so_c).contiguous()
    else:
        perm_c = torch.zeros(1, dtype=torch.int32, device=v.device)

    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
    with torch.cuda.device(q.device.index):
        _run_compiled(
            dvdk_launcher,
            batch,
            q_c,
            k_c,
            v_c,
            do_c,
            so_c,
            nt_c,
            perm_c,
            dv,
            dk,
            fx.Stream(launch_stream),
        )
        _run_compiled(
            dq_launcher,
            batch,
            q_c,
            k_c,
            v_c,
            do_c,
            so_c,
            nt_c,
            perm_c,
            dq,
            fx.Stream(launch_stream),
        )
    return dq, dk, dv


def _make_bwd_kernel_runners(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
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
    sort_by_length: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> dict:
    """Build independent runners for tuning and validating each backward kernel.

    Returns `{"dvdk": (runner, (dv, dk)), "dq": (runner, (dq,))}`.
    """
    batch, num_heads, head_dim, hidden_dim, dtype_str = _validate_bwd_inputs(
        q=q,
        k=k,
        v=v,
        dout=dout,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_seq_len=N,
    )
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
        has_perm=sort_by_length,
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

    if sort_by_length:
        perm_c = _build_balance_perm(so_c).contiguous()
    else:
        perm_c = torch.zeros(1, dtype=torch.int32, device=v.device)

    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")

    def run_dvdk():
        with torch.cuda.device(q.device.index):
            _run_compiled(
                dvdk_launcher,
                batch,
                q_c,
                k_c,
                v_c,
                do_c,
                so_c,
                nt_c,
                perm_c,
                dv,
                dk,
                fx.Stream(launch_stream),
            )

    def run_dq():
        with torch.cuda.device(q.device.index):
            _run_compiled(
                dq_launcher,
                batch,
                q_c,
                k_c,
                v_c,
                do_c,
                so_c,
                nt_c,
                perm_c,
                dq,
                fx.Stream(launch_stream),
            )

    return {
        _BWD_KERNEL_DVDK: (run_dvdk, (dv, dk)),
        _BWD_KERNEL_DQ: (run_dq, (dq,)),
    }


class FlydslHstuAttention(torch.autograd.Function):
    """Differentiable FlyDSL HSTU attention with gradients for q, k, and v."""

    @staticmethod
    def forward(
        ctx,
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
    ) -> torch.Tensor:
        # Reject unsupported masks before saving autograd state.
        if not causal:
            raise ValueError("flydsl_hstu_attention requires causal=True")

        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        ctx.save_for_backward(*saved_tensors)
        ctx.N = N
        ctx.alpha = alpha
        ctx.causal = causal
        ctx.has_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.contextual_seq_len = contextual_seq_len
        return flydsl_hstu_attention_fwd(
            N,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            num_targets,
            max_attn_len,
            contextual_seq_len,
        )

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            num_targets = ctx.saved_tensors[4] if ctx.has_targets else None
            dq, dk, dv = flydsl_hstu_attention_bwd(
                ctx.N,
                ctx.alpha,
                q,
                k,
                v,
                dout,
                seq_offsets,
                ctx.causal,
                num_targets,
                ctx.max_attn_len,
                ctx.contextual_seq_len,
            )
        # Gradient positions match the forward arguments.
        return None, None, dq, dk, dv, None, None, None, None, None


def flydsl_hstu_attention(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: torch.Tensor | None = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    """Drop-in differentiable HSTU attention (FlyDSL fwd + bwd via autograd)."""
    return FlydslHstuAttention.apply(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        causal,
        num_targets,
        max_attn_len,
        contextual_seq_len,
    )
