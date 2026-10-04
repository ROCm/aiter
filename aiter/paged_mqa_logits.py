# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 paged MQA logits (DSA indexer decode scores), dispatched per call shape.

``paged_mqa_logits`` takes the same tensors as the Gluon
``deepgemm_fp8_paged_mqa_logits`` and picks the backend and its launch knobs
from the tuned table ``AITER_CONFIG_PAGED_MQA_LOGITS``
(``aiter/configs/paged_mqa_logits_tuned.csv``, plus any
``model_configs/*paged_mqa_logits_tuned*.csv``). Tune new shapes with
``csrc/paged_mqa_logits/paged_mqa_logits_tune.py``.

Rows are keyed on ``(gfx, cu_num, batch_size, next_n, heads, head_dim,
kv_block_size, preshuffle)``: everything a decode CUDA graph fixes at capture.
Context length is not a key because reading it would need a host sync. A
batch size with no row uses the row of the largest tuned batch size below it;
a shape with no row at all uses Gluon with ``ChunkK=256, WavePerEU=2``.
"""

from __future__ import annotations

import csv
import functools
import math

import torch

from aiter import logger
from aiter.jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from aiter.jit.utils.chip_info import get_cu_num, get_gfx

__all__ = ["get_paged_mqa_logits_config", "paged_mqa_logits"]

BACKENDS = ("gluon", "flydsl")
DEFAULT_CONFIG = {"backend": "gluon", "ChunkK": 256, "WavePerEU": 2, "wg_per_cu": 0}
# FlyDSL kernel: 50 KiB of LDS per CTA, so three CTAs fit on a CU.
FLYDSL_DEFAULT_WG_PER_CU = 3

_SHAPE_KEYS = (
    "batch_size",
    "next_n",
    "heads",
    "head_dim",
    "kv_block_size",
    "preshuffle",
)


def _bool_cell(value) -> bool:
    return str(value).strip() in ("True", "true", "1")


def _int_cell(value, default=0) -> int:
    value = (value or "").strip()
    return int(float(value)) if value else default


@functools.lru_cache(maxsize=1)
def _load_tuned_table() -> dict[tuple, dict]:
    """``{(gfx, cu_num, next_n, heads, head_dim, kvb, preshuffle): {batch: cfg}}``."""
    table: dict[tuple, dict] = {}
    path = AITER_CONFIGS.AITER_CONFIG_PAGED_MQA_LOGITS_FILE
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
                    int(row["next_n"]),
                    int(row["heads"]),
                    int(row["head_dim"]),
                    int(row["kv_block_size"]),
                    _bool_cell(row["preshuffle"]),
                )
                table.setdefault(key, {})[int(row["batch_size"])] = {
                    "backend": backend,
                    "ChunkK": _int_cell(row.get("ChunkK"), DEFAULT_CONFIG["ChunkK"]),
                    "WavePerEU": _int_cell(
                        row.get("WavePerEU"), DEFAULT_CONFIG["WavePerEU"]
                    ),
                    "wg_per_cu": _int_cell(row.get("wg_per_cu")),
                }
    except FileNotFoundError:
        return {}
    except (KeyError, ValueError) as exc:
        logger.warning(
            f"[paged_mqa_logits] ignoring malformed tuned table {path}: {exc}"
        )
        return {}
    return table


@functools.lru_cache(maxsize=1024)
def get_paged_mqa_logits_config(
    batch_size: int,
    next_n: int,
    heads: int,
    head_dim: int,
    kv_block_size: int,
    preshuffle: bool,
) -> dict:
    """Tuned backend and knobs for one call shape on the current GPU."""
    gfx, cu_num = get_gfx(), get_cu_num()
    rows = _load_tuned_table().get(
        (gfx, cu_num, next_n, heads, head_dim, kv_block_size, bool(preshuffle)), {}
    )
    tuned_batch = max((b for b in rows if b <= batch_size), default=None)
    config = rows[tuned_batch] if tuned_batch is not None else DEFAULT_CONFIG
    if AITER_LOG_TUNED_CONFIG:
        shape = dict(
            zip(
                _SHAPE_KEYS,
                (batch_size, next_n, heads, head_dim, kv_block_size, preshuffle),
            )
        )
        source = (
            f"tuned row batch_size={tuned_batch}"
            if tuned_batch is not None
            else "default (no tuned row)"
        )
        logger.info(
            f"[paged_mqa_logits] {gfx} cu_num={cu_num} {shape}: {config} from {source} "
            f"in {AITER_CONFIGS.AITER_CONFIG_PAGED_MQA_LOGITS_FILE}"
        )
    return dict(config)


def reload_tuned_table() -> None:
    """Re-read the tuned table after ``AITER_CONFIG_PAGED_MQA_LOGITS`` changes."""
    AITER_CONFIGS.get_config_file.cache_clear()
    _load_tuned_table.cache_clear()
    get_paged_mqa_logits_config.cache_clear()


def _per_sequence_context_lens(context_lens: torch.Tensor) -> torch.Tensor:
    # Both kernels take one length per sequence: the last column of (B, next_n).
    if context_lens.dim() == 2 and context_lens.shape[1] > 1:
        return context_lens[:, -1].contiguous()
    return context_lens


@functools.lru_cache(maxsize=1)
def _flydsl_kernel():
    try:
        from aiter.ops.flydsl.kernels.mqa_logits import fp8_paged_mqa_logits_gfx950
    except ImportError:
        return None
    return fp8_paged_mqa_logits_gfx950


def flydsl_supports(q_fp8, kv_cache, weights, Preshuffle, KVBlockSize) -> bool:
    """Whether the gfx950 FlyDSL kernel can serve this call."""
    if get_gfx() != "gfx950" or not Preshuffle or q_fp8.dim() != 4:
        return False
    mod = _flydsl_kernel()
    if mod is None:
        return False
    _, next_n, heads, head_dim = q_fp8.shape
    return (
        1 <= next_n <= mod.MAX_NN
        and (heads, head_dim, KVBlockSize)
        == (mod.NUM_HEADS, mod.HEAD_DIM, mod.KV_BLOCK_SIZE)
        and tuple(kv_cache.shape[1:]) == (mod.KV_BLOCK_SIZE, 1, mod.INDEX_DIM)
        and kv_cache.dtype == torch.uint8
        and kv_cache.numel() < 2**31
        and kv_cache.is_contiguous()
        and q_fp8.is_contiguous()
        and weights.is_contiguous()
        and weights.dtype == torch.float32
    )


def flydsl_split_kv(batch_size: int, max_model_len: int, wg_per_cu: int, device) -> int:
    # Bounded by max_model_len rather than the live context lengths, so the
    # launch never reads device memory on the host and stays graph-capturable.
    mod = _flydsl_kernel()
    pages = math.ceil(max_model_len / mod.KV_BLOCK_SIZE)
    total_cu = mod.device_cu_count(device.index)
    wg_per_cu = wg_per_cu or FLYDSL_DEFAULT_WG_PER_CU
    return max(1, min(pages, math.ceil(total_cu * wg_per_cu / batch_size)))


def run_paged_mqa_logits(
    config: dict,
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    out_logits: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    Preshuffle: bool,
    KVBlockSize: int,
) -> torch.Tensor:
    """Launch one explicit config. ``paged_mqa_logits`` and the tuner share it."""
    context_lens = _per_sequence_context_lens(context_lens)
    if config["backend"] == "flydsl":
        _flydsl_kernel().flydsl_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache,
            weights,
            out_logits,
            context_lens,
            kv_indices,
            max_model_len,
            Preshuffle=Preshuffle,
            KVBlockSize=KVBlockSize,
            SplitKV=flydsl_split_kv(
                q_fp8.shape[0], max_model_len, config["wg_per_cu"], q_fp8.device
            ),
        )
        return out_logits

    from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache,
        weights,
        out_logits,
        context_lens,
        kv_indices,
        max_model_len,
        Preshuffle=Preshuffle,
        KVBlockSize=KVBlockSize,
        ChunkK=config["ChunkK"],
        WavePerEU=config["WavePerEU"],
    )
    return out_logits


def paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    out_logits: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    Preshuffle: bool = False,
    KVBlockSize: int = 1,
) -> torch.Tensor:
    """FP8 paged MQA logits with the tuned backend for this shape.

    Args:
        q_fp8: ``[B, next_n, H, D]`` FP8 E4M3 queries.
        kv_cache: ``[num_blocks, KVBlockSize, 1, D + 4]`` uint8, FP8 keys
            co-packed with their FP32 per-token scales (16x16 preshuffled when
            ``Preshuffle``).
        weights: ``[B * next_n, H]`` float32 head weights.
        out_logits: ``[B * next_n, >= max_model_len]`` float32 output. Only
            columns inside each row's causal bound are written.
        context_lens: ``[B]`` or ``[B, next_n]`` int32. With a 2D table the
            last column is the sequence length.
        kv_indices: ``[B, max_blocks]`` int32 block table.
        max_model_len: Logical row width.

    Returns:
        ``out_logits``.
    """
    batch_size, next_n, heads, head_dim = q_fp8.shape
    config = get_paged_mqa_logits_config(
        batch_size, next_n, heads, head_dim, KVBlockSize, bool(Preshuffle)
    )
    if config["backend"] == "flydsl" and not flydsl_supports(
        q_fp8, kv_cache, weights, Preshuffle, KVBlockSize
    ):
        config = dict(DEFAULT_CONFIG)
    return run_paged_mqa_logits(
        config,
        q_fp8,
        kv_cache,
        weights,
        out_logits,
        context_lens,
        kv_indices,
        max_model_len,
        Preshuffle,
        KVBlockSize,
    )
