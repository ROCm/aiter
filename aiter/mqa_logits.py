# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 MQA logits over a contiguous KV (DSA indexer prefill scores), dispatched per call shape.

``mqa_logits`` takes the same tensors as the Triton/Gluon ``fp8_mqa_logits`` and
picks the backend and its kernel variant from the tuned table
``AITER_CONFIG_MQA_LOGITS`` (``aiter/configs/tuned_mqa_logits.csv``, plus any
``model_configs/*tuned_mqa_logits*.csv``). Tune new shapes with
``csrc/mqa_logits/mqa_logits_tune.py``.

Rows are keyed on ``(gfx, cu_num, heads, head_dim, seq_len, seq_len_kv)``. Both
lengths are tensor sizes, so the lookup never reads device memory. A call uses
the row with the largest tuned ``seq_len_kv`` not above its own, then among
those the largest tuned ``seq_len`` not above its own; a length below every
tuned one uses the smallest. A shape with no row at all uses the Triton/Gluon
``fp8_mqa_logits``, as callers get today.
"""

from __future__ import annotations

import csv
import functools

import torch

from aiter import logger
from aiter.jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

__all__ = ["get_mqa_logits_config", "mqa_logits"]

BACKENDS = ("triton", "flydsl")
DEFAULT_CONFIG = {"backend": "triton", "variant": ""}


@functools.lru_cache(maxsize=1)
def _load_tuned_table() -> dict[tuple, dict]:
    """``{(gfx, cu_num, heads, head_dim): {(seq_len_kv, seq_len): cfg}}``."""
    table: dict[tuple, dict] = {}
    path = AITER_CONFIGS.AITER_CONFIG_MQA_LOGITS_FILE
    try:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(
                line for line in f if not line.lstrip().startswith("#")
            ):
                backend = (row.get("backend") or "").strip()
                if backend not in BACKENDS:
                    continue
                key = (
                    row["gfx"].strip(),
                    int(row["cu_num"]),
                    int(row["heads"]),
                    int(row["head_dim"]),
                )
                shape = (int(row["seq_len_kv"]), int(row["seq_len"]))
                table.setdefault(key, {})[shape] = {
                    "backend": backend,
                    "variant": (row.get("variant") or "").strip(),
                }
    except FileNotFoundError:
        return {}
    except (KeyError, ValueError) as exc:
        logger.warning(f"[mqa_logits] ignoring malformed tuned table {path}: {exc}")
        return {}
    return table


def _nearest_below(values, x: int) -> int:
    below = [v for v in values if v <= x]
    return max(below) if below else min(values)


@functools.lru_cache(maxsize=4096)
def get_mqa_logits_config(
    seq_len: int, seq_len_kv: int, heads: int, head_dim: int
) -> dict:
    """Tuned backend and variant for one call shape on the current GPU."""
    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    rows = _load_tuned_table().get((gfx, cu_num, heads, head_dim), {})
    if rows:
        tuned_kv = _nearest_below({kv for kv, _ in rows}, seq_len_kv)
        tuned_q = _nearest_below({q for kv, q in rows if kv == tuned_kv}, seq_len)
        config = rows[(tuned_kv, tuned_q)]
        source = f"tuned row seq_len={tuned_q} seq_len_kv={tuned_kv}"
    else:
        config = DEFAULT_CONFIG
        source = "default (no tuned row)"
    if AITER_LOG_TUNED_CONFIG:
        logger.info(
            f"[mqa_logits] {gfx} cu_num={cu_num} seq_len={seq_len} "
            f"seq_len_kv={seq_len_kv} heads={heads} head_dim={head_dim}: {config} "
            f"from {source} in {AITER_CONFIGS.AITER_CONFIG_MQA_LOGITS_FILE}"
        )
    return dict(config)


def reload_tuned_table() -> None:
    """Re-read the tuned table after ``AITER_CONFIG_MQA_LOGITS`` changes."""
    AITER_CONFIGS.get_config_file.cache_clear()
    _load_tuned_table.cache_clear()
    get_mqa_logits_config.cache_clear()


@functools.lru_cache(maxsize=1)
def _flydsl_kernels():
    try:
        from aiter.ops.flydsl import fp8_mqa_logits_kernels
    except ImportError:
        return None
    return fp8_mqa_logits_kernels


def flydsl_supports(Q, KV, variant: str) -> bool:
    """Whether the FlyDSL kernel can run ``variant`` for this call."""
    gfx = get_gfx_runtime()
    if gfx not in ("gfx942", "gfx950"):
        return False
    mod = _flydsl_kernels()
    if mod is None or variant not in mod.KERNEL_VARIANTS:
        return False
    if Q.dim() != 3 or KV.dim() != 2 or Q.shape[-1] not in (64, 128):
        return False
    if gfx == "gfx950":
        return Q.dtype == KV.dtype == torch.float8_e4m3fn
    return {Q.dtype, KV.dtype} <= {torch.float8_e4m3fn, torch.float8_e4m3fnuz}


def run_mqa_logits(
    config: dict,
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Launch one explicit config. ``mqa_logits`` and the tuner share it."""
    if config["backend"] == "flydsl":
        return _flydsl_kernels().flydsl_fp8_mqa_logits(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            clean_logits=clean_logits,
            variant=config["variant"],
        )

    from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

    return fp8_mqa_logits(
        Q, KV, kv_scales, weights, cu_starts, cu_ends, clean_logits=clean_logits
    )


def mqa_logits(
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """FP8 MQA logits with the tuned backend for this shape.

    Args:
        Q: ``[seq_len, H, D]`` FP8 E4M3 queries.
        KV: ``[seq_len_kv, D]`` FP8 E4M3 keys.
        kv_scales: ``[seq_len_kv]`` float32 per-key scales.
        weights: ``[seq_len, H]`` float32 head weights.
        cu_starts: ``[seq_len]`` int32, first key of each row's window.
        cu_ends: ``[seq_len]`` int32, one past the last key of each row's window.
        clean_logits: write ``-inf`` outside each row's window. If False, those
            positions are unspecified.

    Returns:
        ``[seq_len, seq_len_kv]`` float32 logits.
    """
    seq_len, heads, head_dim = Q.shape
    config = get_mqa_logits_config(seq_len, KV.shape[0], heads, head_dim)
    if config["backend"] == "flydsl" and not flydsl_supports(Q, KV, config["variant"]):
        config = dict(DEFAULT_CONFIG)
    return run_mqa_logits(
        config, Q, KV, kv_scales, weights, cu_starts, cu_ends, clean_logits
    )
